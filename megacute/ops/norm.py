"""RMSNorm / LayerNorm over the last dim, a block of rows per instruction.

Rows are independent and need the whole row, so this op skips the page ring: each warp of the
compute group owns rows, each lane owns 8-element (16 B) vectors strided by 256 elements, and
row statistics are reduced with warp shuffles. Rows up to MAX_REG_COLS stay in registers (one
read of x, all loads in flight at once); wider rows take a second pass that re-reads x from L2.
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32

from .. import config as C
from .. import dmath
from ..schedule import norm_rows
from ..kernel import buf_tensor
from .base import Node

VEC = 8
SPAN = 32 * VEC
MAX_REG_COLS = 4096                    # 128 fp32 registers per lane


class NormNode(Node):
    consumer = "compute"

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.R, self.Cc = op.x.shape
        assert self.Cc % VEC == 0, "norm: last dim must be a multiple of 8"
        self.rows = norm_rows(self.R)
        self.stage_scratch = self.rows == 1     # wide mode: cross-warp partial sums in the staging buffer
        self.rms = op.rms
        self.eps = op.eps
        self.ix, self.io = self.view(op.x), self.view(op.out)
        self.iw = self.view(op.weight) if op.weight is not None else None
        self.ib = self.view(op.bias) if op.bias is not None else None

    def host(self, views):
        x, out = views[self.ix], views[self.io]
        w = views[self.iw] if self.iw is not None else x
        b = views[self.ib] if self.ib is not None else x
        return (x, out, w, b)

    @cute.jit
    def compute(self, ctx, a, slot, pslot, pph, accidx, accph, sfph, pvph):
        if cutlass.const_expr(self.rows == 1):
            self._wide(ctx, a, slot)
        else:
            self._rows(ctx, a, slot)
        return pslot, pph, accidx, accph, sfph, pvph

    @cute.jit
    def _wide(self, ctx, a, slot):
        """One row, all compute threads: thread t owns chunks t, t + 128, ...; the per-warp sums are
        combined through the staging buffer."""
        x, out, w, b = a
        tidx = ctx.tidx
        row = ctx.arg(slot, 0)
        n = (self.Cc // VEC + C.COMPUTE_THREADS - 1) // C.COMPUTE_THREADS
        xrow = x[(row + self.layer_off(ctx, slot, self.ix), None)]
        orow = out[(row + self.layer_off(ctx, slot, self.io), None)]
        rx = cute.make_rmem_tensor((n * VEC,), Float32)
        rv = cute.make_rmem_tensor((VEC,), Float32)
        rw = cute.make_rmem_tensor((VEC,), Float32)
        rb = cute.make_rmem_tensor((VEC,), Float32)
        ro = cute.make_rmem_tensor((VEC,), BFloat16)
        rx.fill(0.0)
        for t in cutlass.range_constexpr(n):
            ci = t * C.COMPUTE_THREADS + tidx
            if ci * VEC < self.Cc:
                rv.store(cute.local_tile(xrow, (VEC,), (ci,)).load().to(Float32))
                for e in cutlass.range_constexpr(VEC):
                    rx[t * VEC + e] = rv[e]
        s1 = Float32(0.0)
        s2 = Float32(0.0)
        for i in cutlass.range_constexpr(n * VEC):
            s1 = s1 + rx[i]
            s2 = s2 + rx[i] * rx[i]
        s1 = cute.arch.warp_reduction(s1, operator.add)
        s2 = cute.arch.warp_reduction(s2, operator.add)
        red = buf_tensor(ctx.stage, cute.make_layout((C.COMPUTE_WARPS, 2), stride=(2, 1)), Float32)
        if ctx.lane == 0:
            red[ctx.warp, 0] = s1
            red[ctx.warp, 1] = s2
        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
        s1 = Float32(0.0)
        s2 = Float32(0.0)
        for k in cutlass.range_constexpr(C.COMPUTE_WARPS):
            s1 = s1 + red[k, 0]
            s2 = s2 + red[k, 1]
        inv_n = Float32(1.0 / self.Cc)
        mean = Float32(0.0)
        if cutlass.const_expr(not self.rms):
            mean = s1 * inv_n
        rstd = dmath.rsqrt(s2 * inv_n - mean * mean + Float32(self.eps))
        for t in cutlass.range_constexpr(n):
            ci = t * C.COMPUTE_THREADS + tidx
            if ci * VEC < self.Cc:
                if cutlass.const_expr(self.iw is not None):
                    rw.store(cute.local_tile(w[(self.layer_off(ctx, slot, self.iw), None)], (VEC,), (ci,))
                             .load().to(Float32))
                if cutlass.const_expr(self.ib is not None):
                    rb.store(cute.local_tile(b[(self.layer_off(ctx, slot, self.ib), None)], (VEC,), (ci,))
                             .load().to(Float32))
                for e in cutlass.range_constexpr(VEC):
                    y = (rx[t * VEC + e] - mean) * rstd
                    if cutlass.const_expr(self.iw is not None):
                        y = y * rw[e]
                    if cutlass.const_expr(self.ib is not None):
                        y = y + rb[e]
                    ro[e] = y.to(BFloat16)
                cute.local_tile(orow, (VEC,), (ci,)).store(ro.load())

    @cute.jit
    def _rows(self, ctx, a, slot):
        x, out, w, b = a
        r0 = ctx.arg(slot, 0) * self.rows
        lane = ctx.lane
        nchunks = (self.Cc + SPAN - 1) // SPAN
        full = self.Cc % SPAN == 0               # every lane owns every chunk: no guards
        in_regs = self.Cc <= MAX_REG_COLS
        xs = VEC if in_regs else 0              # rx stride per chunk: whole row, or one chunk reused
        rx = cute.make_rmem_tensor((max(1, nchunks * xs // VEC) * VEC,), Float32)
        rv = cute.make_rmem_tensor((VEC,), Float32)
        rw = cute.make_rmem_tensor((VEC,), Float32)
        rb = cute.make_rmem_tensor((VEC,), Float32)
        ro = cute.make_rmem_tensor((VEC,), BFloat16)
        for rr in cutlass.range_constexpr(self.rows // C.COMPUTE_WARPS):
            row = r0 + rr * C.COMPUTE_WARPS + ctx.warp
            if row < self.R:
                xrow = x[(row + self.layer_off(ctx, slot, self.ix), None)]
                orow = out[(row + self.layer_off(ctx, slot, self.io), None)]
                if cutlass.const_expr(in_regs):
                    rx.fill(0.0)
                    for t in cutlass.range_constexpr(nchunks):
                        ci = t * 32 + lane
                        if cutlass.const_expr(full):
                            rv.store(cute.local_tile(xrow, (VEC,), (ci,)).load().to(Float32))
                        else:
                            rv.fill(0.0)
                            if ci * VEC < self.Cc:
                                rv.store(cute.local_tile(xrow, (VEC,), (ci,)).load().to(Float32))
                        for e in cutlass.range_constexpr(VEC):
                            rx[t * xs + e] = rv[e]
                s1 = Float32(0.0)
                s2 = Float32(0.0)
                for t in cutlass.range_constexpr(nchunks):
                    if cutlass.const_expr(not in_regs):
                        ci = t * 32 + lane
                        rv.fill(0.0)
                        if ci * VEC < self.Cc:
                            rv.store(cute.local_tile(xrow, (VEC,), (ci,)).load().to(Float32))
                        for e in cutlass.range_constexpr(VEC):
                            rx[t * xs + e] = rv[e]
                    for e in cutlass.range_constexpr(VEC):
                        v = rx[t * xs + e]
                        s1 = s1 + v
                        s2 = s2 + v * v
                s1 = cute.arch.warp_reduction(s1, operator.add)
                s2 = cute.arch.warp_reduction(s2, operator.add)
                inv_n = Float32(1.0 / self.Cc)
                mean = Float32(0.0)
                if cutlass.const_expr(not self.rms):
                    mean = s1 * inv_n
                var = s2 * inv_n - mean * mean
                rstd = dmath.rsqrt(var + Float32(self.eps))
                for t in cutlass.range_constexpr(nchunks):
                    ci = t * 32 + lane
                    if ci * VEC < self.Cc:
                        if cutlass.const_expr(not in_regs):
                            rv.store(cute.local_tile(xrow, (VEC,), (ci,)).load().to(Float32))
                            for e in cutlass.range_constexpr(VEC):
                                rx[t * xs + e] = rv[e]
                        if cutlass.const_expr(self.iw is not None):
                            rw.store(cute.local_tile(w[(self.layer_off(ctx, slot, self.iw), None)], (VEC,), (ci,))
                                     .load().to(Float32))
                        if cutlass.const_expr(self.ib is not None):
                            rb.store(cute.local_tile(b[(self.layer_off(ctx, slot, self.ib), None)], (VEC,), (ci,))
                                     .load().to(Float32))
                        for e in cutlass.range_constexpr(VEC):
                            y = (rx[t * xs + e] - mean) * rstd
                            if cutlass.const_expr(self.iw is not None):
                                y = y * rw[e]
                            if cutlass.const_expr(self.ib is not None):
                                y = y + rb[e]
                            ro[e] = y.to(BFloat16)
                        cute.local_tile(orow, (VEC,), (ci,)).store(ro.load())
