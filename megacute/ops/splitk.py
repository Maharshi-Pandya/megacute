"""SplitFinalize: out = act(ws + bias) + residual for one 128x128 tile, then ws = 0.

The fp32 workspace was filled by the split-K partials' atomic adds (GemmNode, partial mode). It
is small and L2-resident, so the compute warps read it directly (16-byte chunks, as in the
elementwise ops) and zero it behind them, ready for the next launch. The bf16 result leaves
through the staging buffer and the storer warp.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import cpasync

from .. import config as C
from .. import dmath
from ..kernel import buf_tensor
from .base import Node
from .eltwise import CPR, RPI, TILE, VEC, flat_layout


class FinalizeNode(Node):
    consumer = "compute"
    stage_out = True

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.M, self.N = op.out.shape
        assert self.N % VEC == 0, "split-K finalize: N must be a multiple of 8"
        self.act = op.act
        self.iws, self.io = self.view(op.ws), self.view(op.out)
        self.ib = self.view(op.bias) if op.bias is not None else None
        self.ir = self.view(op.residual) if op.residual is not None else None

    def host(self, views):
        ws, out = views[self.iws], views[self.io]
        store = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), out, flat_layout(), TILE)
        bias = views[self.ib] if self.ib is not None else out
        res = views[self.ir] if self.ir is not None else out
        return (ws, store, bias, res)

    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        ws, bias, res = a[0], a[2], a[3]
        m = ctx.arg(slot, 0)
        n = ctx.arg(slot, 1)
        tidx = ctx.tidx
        sOut = buf_tensor(ctx.stage, flat_layout(), BFloat16)
        cc = tidx % CPR
        gc = n * CPR + cc                      # chunk index along the full row
        rv = cute.make_rmem_tensor((VEC,), Float32)
        rb = cute.make_rmem_tensor((VEC,), Float32)
        rr = cute.make_rmem_tensor((VEC,), Float32)
        rz = cute.make_rmem_tensor((VEC,), Float32)
        ro = cute.make_rmem_tensor((VEC,), BFloat16)
        rz.fill(0.0)
        rb.fill(0.0)
        if cutlass.const_expr(self.ib is not None):
            if gc * VEC < self.N:
                rb.store(cute.local_tile(bias[(self.layer_off(ctx, slot, self.ib), None)], (VEC,), (gc,))
                         .load().to(Float32))
        for i in cutlass.range_constexpr(C.TILE // RPI):
            rl = i * RPI + tidx // CPR
            row = m * C.TILE + rl
            rv.fill(0.0)
            rr.fill(0.0)
            if (row < self.M) & (gc * VEC < self.N):
                wchunk = cute.local_tile(ws[(row + self.layer_off(ctx, slot, self.iws), None)], (VEC,), (gc,))
                rv.store(wchunk.load())
                wchunk.store(rz.load())
                if cutlass.const_expr(self.ir is not None):
                    rrow = row + self.layer_off(ctx, slot, self.ir)
                    rr.store(cute.local_tile(res[(rrow, None)], (VEC,), (gc,)).load().to(Float32))
            for e in cutlass.range_constexpr(VEC):
                ro[e] = (dmath.act(self.act, rv[e] + rb[e]) + rr[e]).to(BFloat16)
            cute.local_tile(sOut[(rl, None)], (VEC,), (cc,)).store(ro.load())
        return cq, cqph, accidx, accph, sfph, pvph

    @cute.jit
    def store(self, ctx, a, slot):
        satom, stma = a[1]
        g = cute.local_tile(stma, TILE, (None, None))
        tS, tG = cpasync.tma_partition(satom, 0, cute.make_layout(1),
                                       cute.group_modes(buf_tensor(ctx.stage, flat_layout(), BFloat16), 0, 2),
                                       cute.group_modes(g, 0, 2))
        cute.copy(satom, tS, tG[(None, ctx.arg(slot, 0) + self.layer_off(ctx, slot, self.io, C.TILE),
                                 ctx.arg(slot, 1))])

