"""Pointwise ops on 64x128 tiles: unary, binary (same-shape or broadcast b), SwiGLU/GEGLU,
adaLN modulate, RoPE.

Same-shape operands arrive by TMA in pool pages (one page per operand); broadcast
operands (per-group rows, cos/sin tables) are small and L2-resident, so the compute warps read
them directly. Tiles are unswizzled row-major and each thread owns 16-byte chunks (8 elements),
so a warp touches 512 contiguous bytes per access; results leave through the staging buffer by
TMA store.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync

from .. import config as C
from .. import dmath
from ..graph import Binary, Modulate, Rope, SwiGLU, Unary
from ..kernel import COMPUTE_Q, buf_tensor, fill_page, page_tensor, take_page
from .base import Node

TILE = (C.EW_ROWS, C.TILE)
TILE_BYTES = C.EW_ROWS * C.TILE * 2
VEC = 8                                # elements per 16-byte chunk
CPR = C.TILE // VEC                    # chunks per tile row
RPI = C.COMPUTE_THREADS // CPR         # tile rows covered per pass of the compute group


def flat_layout():
    """64x128 row-major unswizzled tile: 16 B chunks are contiguous, so warp access is conflict-free."""
    return cute.make_layout(TILE, stride=(C.TILE, 1))


class EltwiseNode(Node):
    consumer = "compute"
    stage_out = True

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.R, self.Cc = op.out.shape
        assert self.Cc % VEC == 0, "eltwise: last dim must be a multiple of 8"
        self.io = self.view(op.out)
        self.direct = {}
        if isinstance(op, Unary):
            self.kind, tma = "unary", [op.x]
        elif isinstance(op, Binary):
            if op.b.rows == op.a.rows:
                self.kind, tma = "binary", [op.a, op.b]
            else:
                self.kind, tma = "binary_bcast", [op.a]
                self.direct["b"] = self.view(op.b)
                self.group = op.b_group
        elif isinstance(op, SwiGLU):
            self.kind, tma = "swiglu", [op.gate, op.up]
        elif isinstance(op, Modulate):
            self.kind, tma = "modulate", [op.x]
            self.direct["scale"] = self.view(op.scale)
            self.direct["shift"] = self.view(op.shift)
            self.group = op.group
        elif isinstance(op, Rope):
            self.kind, tma = "rope", [op.x]
            self.direct["cos"] = self.view(op.cos)
            self.direct["sin"] = self.view(op.sin)
            self.half = op.head_dim // 2
            self.head_dim = op.head_dim
            self.seq = op.seq
        else:
            raise TypeError(op)
        self.fn = getattr(op, "fn", getattr(op, "act", None))
        self.tma_idx = [self.view(v) for v in tma]
        self.n_tma = len(tma)

    def host(self, views):
        e_l = flat_layout()
        g2s = cpasync.CopyBulkTensorTileG2SOp()
        loads = []
        for i in range(self.n_tma):  # plain helper (not preprocessed): ordinary trace-time loop
            loads.append(cpasync.make_tiled_tma_atom(g2s, views[self.tma_idx[i]], e_l, TILE))
        store = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), views[self.io], e_l, TILE)
        direct = tuple(views[self.direct[k]] for k in sorted(self.direct))
        return (tuple(loads), store, e_l, direct)

    def _direct(self, a, name):
        return a[3][sorted(self.direct).index(name)]

    @cute.jit
    def load(self, ctx, a, slot, pmask, pcur, mq, mqph, cq, cqph):
        loads, e_l = a[0], a[2]
        r = ctx.arg(slot, 0)
        c = ctx.arg(slot, 1)
        for i in cutlass.range_constexpr(self.n_tma):
            atom, tma = loads[i]
            p, pmask, pcur, cq, cqph = fill_page(ctx, slot, COMPUTE_Q, TILE_BYTES, pmask, pcur, cq, cqph)
            sT = page_tensor(ctx.pool, p, 0, e_l, BFloat16)
            g = cute.local_tile(tma, TILE, (None, None))
            tS, tG = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                           cute.group_modes(sT, 0, 2), cute.group_modes(g, 0, 2))
            cute.copy(atom, tG[(None, r + self.layer_off(ctx, slot, self.tma_idx[i], C.EW_ROWS), c)], tS,
                      tma_bar_ptr=ctx.page_full + p)
        return pmask, pcur, mq, mqph, cq, cqph

    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        e_l = a[2]
        r = ctx.arg(slot, 0)
        c = ctx.arg(slot, 1)
        tidx = ctx.tidx
        sOut = buf_tensor(ctx.stage, e_l, BFloat16)
        p0, cq, cqph = take_page(ctx, slot, COMPUTE_Q, cq, cqph)
        sX = page_tensor(ctx.pool, p0, 0, e_l, BFloat16)
        sY = sX
        p1 = p0
        if cutlass.const_expr(self.n_tma == 2):
            p1, cq, cqph = take_page(ctx, slot, COMPUTE_Q, cq, cqph)
            sY = page_tensor(ctx.pool, p1, 0, e_l, BFloat16)

        kind = self.kind
        cc = tidx % CPR
        gc = c * CPR + cc                      # chunk index along the full row
        rx = cute.make_rmem_tensor((VEC,), Float32)
        ry = cute.make_rmem_tensor((VEC,), Float32)
        rz = cute.make_rmem_tensor((VEC,), Float32)
        ro = cute.make_rmem_tensor((VEC,), BFloat16)
        for i in cutlass.range_constexpr(C.EW_ROWS // RPI):
            rl = i * RPI + tidx // CPR
            row = r * C.EW_ROWS + rl
            rx.store(cute.local_tile(sX[(rl, None)], (VEC,), (cc,)).load().to(Float32))
            if cutlass.const_expr(kind == "binary" or kind == "swiglu"):
                ry.store(cute.local_tile(sY[(rl, None)], (VEC,), (cc,)).load().to(Float32))
            elif cutlass.const_expr(kind == "binary_bcast"):
                b = self._direct(a, "b")
                ry.fill(0.0)
                if (row < self.R) & (gc * VEC < self.Cc):
                    brow = row // self.group + self.layer_off(ctx, slot, self.direct["b"])
                    ry.store(cute.local_tile(b[(brow, None)], (VEC,), (gc,)).load().to(Float32))
            elif cutlass.const_expr(kind == "modulate"):
                sc = self._direct(a, "scale")
                sh = self._direct(a, "shift")
                ry.fill(0.0)
                rz.fill(0.0)
                if (row < self.R) & (gc * VEC < self.Cc):
                    srow = row // self.group + self.layer_off(ctx, slot, self.direct["scale"])
                    hrow = row // self.group + self.layer_off(ctx, slot, self.direct["shift"])
                    ry.store(cute.local_tile(sc[(srow, None)], (VEC,), (gc,)).load().to(Float32))
                    rz.store(cute.local_tile(sh[(hrow, None)], (VEC,), (gc,)).load().to(Float32))
            elif cutlass.const_expr(kind == "rope"):
                # rotate-half within each head_dim block: partner chunk is half a head away
                cs = self._direct(a, "cos")
                sn = self._direct(a, "sin")
                hc = self.half // VEC
                dc = cc % (self.head_dim // VEC)
                sgn = Float32(1.0)
                pc = cc - hc
                if dc < hc:
                    sgn = Float32(-1.0)
                    pc = cc + hc
                ry.store(cute.local_tile(sX[(rl, None)], (VEC,), (pc,)).load().to(Float32))
                for e in cutlass.range_constexpr(VEC):
                    ry[e] = ry[e] * sgn
                rz.fill(0.0)
                cv = cute.make_rmem_tensor((VEC,), Float32)
                cv.fill(0.0)
                if row < self.R:
                    pos = row % self.seq
                    cv.store(cute.local_tile(cs[(pos + self.layer_off(ctx, slot, self.direct["cos"]), None)],
                                             (VEC,), (dc,)).load().to(Float32))
                    rz.store(cute.local_tile(sn[(pos + self.layer_off(ctx, slot, self.direct["sin"]), None)],
                                             (VEC,), (dc,)).load().to(Float32))
            for e in cutlass.range_constexpr(VEC):
                x = rx[e]
                if cutlass.const_expr(kind == "unary"):
                    y = dmath.act(self.fn, x)
                elif cutlass.const_expr(kind == "binary" or kind == "binary_bcast"):
                    y = dmath.binary(self.fn, x, ry[e])
                elif cutlass.const_expr(kind == "swiglu"):
                    y = dmath.act(self.fn, x) * ry[e]
                elif cutlass.const_expr(kind == "modulate"):
                    y = x * (Float32(1.0) + ry[e]) + rz[e]
                else:
                    y = x * cv[e] + ry[e] * rz[e]
                ro[e] = y.to(BFloat16)
            cute.local_tile(sOut[(rl, None)], (VEC,), (cc,)).store(ro.load())

        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
        if tidx == 0:
            cute.arch.mbarrier_arrive(ctx.page_empty + p0)
            if cutlass.const_expr(self.n_tma == 2):
                cute.arch.mbarrier_arrive(ctx.page_empty + p1)
        return cq, cqph, accidx, accph, sfph, pvph

    @cute.jit
    def store(self, ctx, a, slot):
        (satom, stma), e_l = a[1], a[2]
        g = cute.local_tile(stma, TILE, (None, None))
        tS, tG = cpasync.tma_partition(satom, 0, cute.make_layout(1),
                                       cute.group_modes(buf_tensor(ctx.stage, e_l, BFloat16), 0, 2),
                                       cute.group_modes(g, 0, 2))
        cute.copy(satom, tS, tG[(None, ctx.arg(slot, 0) + self.layer_off(ctx, slot, self.io, C.EW_ROWS),
                                 ctx.arg(slot, 1))])
