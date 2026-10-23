"""Linear: out = act(x @ w^T + bias) + residual, one 128x128 output tile per instruction.

The loader streams (A 128x64 | B 128x64) k-stages through pool pages, the MMA warp
accumulates in one of two TMEM stages (so tile i's epilogue overlaps tile i+1's main loop), and
the compute warpgroup reads the accumulator row-per-thread, applies the fused epilogue and
stages the tile for the storer warp.

A LinearSplit (split-K, see runtime.split_k) runs the same main loop over its K range and
atomically adds the fp32 accumulator into the workspace instead; SplitFinalize does the epilogue.
"""

import operator

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.tensor_utils import LayoutEnum

from .. import config as C
from .. import dmath
from ..graph import LinearSplit
from ..kernel import MMA_Q, TMEM_ACC, buf_tensor, fill_page, page_tensor, ring_next, take_page, wait_bar, wait_deps
from .base import Node, epi_layout

VEC = 8   # bf16 elements per 16-byte load



class GemmNode(Node):
    consumer = "mma"
    early_loads = True      # weight tiles stream in while the activations' producers finish

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.M, self.K = op.x.shape
        self.N = op.w.rows
        assert self.K % C.GEMM_BK == 0
        self.partial = isinstance(op, LinearSplit)
        self.stage_out = True
        self.bn = C.SPLIT_BN if self.partial else C.GEMM_BN
        # swiglu: accumulator columns [0, 64) are gate, [64, 128) up; the output tile is 128x64
        self.glu = not self.partial and op.act == "swiglu"
        # stage/store dtype: fp32 for a split-K workspace, bf16 for outputs and in-place residual partials
        self.cdt = Float32 if self.partial and op.ws.buf.dtype == torch.float32 else BFloat16
        self.rms = op.rms_eps        # folded RMSNorm: rows scaled by rstd of the raw input (fold_norms)
        self.obn = self.bn // 2 if self.glu else self.bn
        self.tiler = (C.GEMM_BM, self.bn, C.GEMM_BK)
        self.a_bytes = C.GEMM_BM * C.GEMM_BK * 2
        self.b_bytes = self.bn * C.GEMM_BK * 2
        self.kc = self.K // C.GEMM_BK // (op.splits if self.partial else 1)   # k-tiles per instruction
        self.ix, self.iw = self.view(op.x), self.view(op.w)
        # thin M (decode, M <= 64): a 128-row A tile per k-stage would be mostly padding. Instead an
        # "A page" holds Mg rows for each of kca k-tiles, stacked as SW128 8-row groups; k-tile j's
        # operand is the ordinary 128x64 descriptor at row j*Mg, whose rows >= Mg read neighbouring
        # data -- garbage confined to accumulator rows >= M, which are never kept. The other pages
        # then carry only weights, kpp k-tiles each.
        self.thin = False
        if self.M <= 64:
            Mg = 8
            while Mg < self.M:
                Mg *= 2
            kpp = C.PAGE_BYTES // self.b_bytes
            kca = max([d for d in range(kpp, self.kc + 1, kpp)
                       if self.kc % d == 0 and d * Mg * C.GEMM_BK * 2 <= C.PAGE_BYTES] or [0])
            if kca:
                self.thin, self.Mg, self.kpp, self.kca = True, Mg, kpp, kca
        if self.partial:
            self.act, self.ib, self.ir = None, None, None
            self.io = self.view(op.ws)                          # fp32 workspace: TMA reduce-add target
        else:
            self.act = op.act
            self.io = self.view(op.out)
            self.ib = self.view(op.bias) if op.bias is not None else None
            self.ir = self.view(op.residual) if op.residual is not None else None

    def host(self, views):
        x, w, out = views[self.ix], views[self.iw], views[self.io]
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, Float32,
            tcgen05.CtaGroup.ONE, self.tiler[:2])
        a_l = cute.select(sm100_utils.make_smem_layout_a(tiled_mma, self.tiler, BFloat16, 1), mode=[0, 1, 2])
        b_l = cute.select(sm100_utils.make_smem_layout_b(tiled_mma, self.tiler, BFloat16, 1), mode=[0, 1, 2])
        e_l = epi_layout(self.obn, self.cdt)
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(g2s, x, a_l, self.tiler, tiled_mma)
        atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(g2s, w, b_l, self.tiler, tiled_mma)
        s2g = cpasync.CopyReduceBulkTensorTileS2GOp() if self.partial else cpasync.CopyBulkTensorTileS2GOp()
        atom_c, tma_c = cpasync.make_tiled_tma_atom(s2g, out, e_l, (C.GEMM_BM, self.obn))
        bias = views[self.ib] if self.ib is not None else out
        res = views[self.ir] if self.ir is not None else out
        x_l, atom_x, tma_x = a_l, atom_a, tma_a
        if self.thin:   # TMA box of one k-tile's Mg rows (same SW128 row layout as the MMA's A operand)
            x_l = cute.select(sm100_utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, (self.Mg, C.GEMM_BK), 1),
                              mode=[0, 1])
            atom_x, tma_x = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), x, x_l, (self.Mg, C.GEMM_BK))
        return (tiled_mma, atom_a, tma_a, atom_b, tma_b, atom_c, tma_c, a_l, b_l, e_l, bias, res, atom_x, tma_x, x_l,
                x)

    @cute.jit
    def load(self, ctx, a, slot, pmask, pcur, mq, mqph, cq, cqph):
        if cutlass.const_expr(self.thin):
            pmask, pcur, mq, mqph = self._load_thin(ctx, a, slot, pmask, pcur, mq, mqph)
        else:
            pmask, pcur, mq, mqph = self._load_tiles(ctx, a, slot, pmask, pcur, mq, mqph)
        return pmask, pcur, mq, mqph, cq, cqph

    @cute.jit
    def _load_thin(self, ctx, a, slot, pmask, pcur, mq, mqph):
        tiled_mma, atom_b, tma_b, b_l = a[0], a[3], a[4], a[8]
        atom_x, tma_x, x_l = a[12], a[13], a[14]
        n = ctx.arg(slot, 1)
        tCgB = tiled_mma.get_slice(0).partition_B(
            cute.local_tile(tma_b, self.tiler, (None, None, None), proj=(None, 1, 1)))
        gX = cute.local_tile(tma_x, (self.Mg, C.GEMM_BK), (None, None))
        k0 = Int32(0)
        if cutlass.const_expr(self.partial):
            k0 = ctx.arg(slot, 2) * self.kc
        mx = self.layer_off(ctx, slot, self.ix, self.Mg)      # M <= 64: the only row block
        nb = n + self.layer_off(ctx, slot, self.iw, self.bn)
        x_bytes = self.kca * self.Mg * C.GEMM_BK * 2
        w_bytes = self.kpp * self.b_bytes
        ppc = self.kca // self.kpp                              # weight pages per A page
        npre = min(ppc, C.POOL_PAGES - 1)
        # chunk 0: the A page is claimed first (consumers pop in claim order), the weight pages
        # stream before the dependency wait, the activations after it
        ap, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, x_bytes, pmask, pcur, mq, mqph)
        for jp in cutlass.range_constexpr(npre):
            p, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, w_bytes, pmask, pcur, mq, mqph, ap)
            for h in cutlass.range_constexpr(self.kpp):
                self._tma(ctx, atom_b, tCgB, b_l, p, h * self.b_bytes, nb, k0 + jp * self.kpp + h)
        wait_deps(ctx, slot)
        self._load_x(ctx, atom_x, gX, x_l, ap, mx, k0)
        for jp in cutlass.range(npre, ppc, unroll=1):
            p, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, w_bytes, pmask, pcur, mq, mqph, ap)
            for h in cutlass.range_constexpr(self.kpp):
                self._tma(ctx, atom_b, tCgB, b_l, p, h * self.b_bytes, nb, k0 + jp * self.kpp + h)
        for c in cutlass.range(1, self.kc // self.kca, unroll=1):
            ap, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, x_bytes, pmask, pcur, mq, mqph, ap)
            self._load_x(ctx, atom_x, gX, x_l, ap, mx, k0 + c * self.kca)
            for jp in cutlass.range(ppc, unroll=1):
                p, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, w_bytes, pmask, pcur, mq, mqph, ap)
                for h in cutlass.range_constexpr(self.kpp):
                    self._tma(ctx, atom_b, tCgB, b_l, p, h * self.b_bytes, nb,
                              k0 + c * self.kca + jp * self.kpp + h)
        return pmask, pcur, mq, mqph

    @cute.jit
    def _load_x(self, ctx, atom, gX, s_l, ap, mx, kb):
        """kca TMA boxes of Mg activation rows, k-tile j at row j*Mg of the A page."""
        for j in cutlass.range(self.kca, unroll=1):
            sT = page_tensor(ctx.pool, ap, j * (self.Mg * C.GEMM_BK * 2), s_l, BFloat16)
            tS, tG = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                           cute.group_modes(sT, 0, 2), cute.group_modes(gX, 0, 2))
            cute.copy(atom, tG[(None, mx, kb + j)], tS, tma_bar_ptr=ctx.page_full + ap)

    @cute.jit
    def _load_tiles(self, ctx, a, slot, pmask, pcur, mq, mqph):
        tiled_mma, atom_a, tma_a, atom_b, tma_b = a[0], a[1], a[2], a[3], a[4]
        a_l, b_l = a[7], a[8]
        m = ctx.arg(slot, 0)
        n = ctx.arg(slot, 1)
        thr = tiled_mma.get_slice(0)
        tCgA = thr.partition_A(cute.local_tile(tma_a, self.tiler, (None, None, None), proj=(1, None, 1)))
        tCgB = thr.partition_B(cute.local_tile(tma_b, self.tiler, (None, None, None), proj=(None, 1, 1)))
        k0 = Int32(0)
        if cutlass.const_expr(self.partial):
            k0 = ctx.arg(slot, 2) * self.kc
        ma = m + self.layer_off(ctx, slot, self.ix, C.GEMM_BM)
        nb = n + self.layer_off(ctx, slot, self.iw, self.bn)
        # weights do not depend on earlier instructions: claim the first stages and start their B
        # loads before the dependency wait; each page completes once its A tile lands after it
        npre = min(self.kc, C.POOL_PAGES)
        pages = []
        for kt in cutlass.range_constexpr(npre):
            p, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, self.a_bytes + self.b_bytes, pmask, pcur, mq, mqph)
            self._tma(ctx, atom_b, tCgB, b_l, p, self.a_bytes, nb, k0 + kt)
            pages.append(p)
        if cutlass.const_expr(self.early_loads):
            wait_deps(ctx, slot)
        for kt in cutlass.range_constexpr(npre):
            self._tma(ctx, atom_a, tCgA, a_l, pages[kt], 0, ma, k0 + kt)
        for kt in cutlass.range(npre, self.kc, unroll=1):
            p, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, self.a_bytes + self.b_bytes, pmask, pcur, mq, mqph)
            self._tma(ctx, atom_a, tCgA, a_l, p, 0, ma, k0 + kt)
            self._tma(ctx, atom_b, tCgB, b_l, p, self.a_bytes, nb, k0 + kt)
        return pmask, pcur, mq, mqph

    @cute.jit
    def _tma(self, ctx, atom, tCg, s_l, p, byte_off, mn, k):
        sT = page_tensor(ctx.pool, p, byte_off, s_l, BFloat16)
        tS, tG = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                       cute.group_modes(sT, 0, 3), cute.group_modes(tCg, 0, 3))
        cute.copy(atom, tG[(None, mn, k)], tS, tma_bar_ptr=ctx.page_full + p)

    @cute.jit
    def mma(self, ctx, a, slot, mq, mqph, accidx, accph, sph, pfph):
        tiled_mma, a_l, b_l = a[0], a[7], a[8]
        acc_shape = tiled_mma.partition_shape_C(self.tiler[:2])
        # stage base fixed at 128 columns whatever this tile's width: stages never overlap
        tAcc = cute.make_tensor(ctx.tmem_ptr + (TMEM_ACC + accidx * C.GEMM_BN),
                                tiled_mma.make_fragment_C(acc_shape).layout)
        wait_bar(ctx, ctx.acc_empty + accidx, accph, slot, "acc_empty")
        if cutlass.const_expr(self.thin):
            mq, mqph = self._mma_thin(ctx, tiled_mma, a_l, b_l, tAcc, slot, mq, mqph)
        else:
            mq, mqph = self._mma_tiles(ctx, tiled_mma, a_l, b_l, tAcc, slot, mq, mqph)
        with cute.arch.elect_one():
            tcgen05.commit(ctx.acc_full + accidx)
        accidx, accph = ring_next(accidx, accph, 2)
        return mq, mqph, accidx, accph, sph, pfph

    @cute.jit
    def _mma_thin(self, ctx, tiled_mma, a_l, b_l, tAcc, slot, mq, mqph):
        for c in cutlass.range(self.kc // self.kca, unroll=1):
            ap, mq, mqph = take_page(ctx, slot, MMA_Q, mq, mqph)
            for jp in cutlass.range(self.kca // self.kpp, unroll=1):
                bp, mq, mqph = take_page(ctx, slot, MMA_Q, mq, mqph)
                for h in cutlass.range_constexpr(self.kpp):
                    xo = (jp * self.kpp + h) * (self.Mg * C.GEMM_BK * 2)
                    tCrA = tiled_mma.make_fragment_A(page_tensor(ctx.pool, ap, xo, a_l, BFloat16))
                    tCrB = tiled_mma.make_fragment_B(page_tensor(ctx.pool, bp, h * self.b_bytes, b_l, BFloat16))
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, (c != 0) | (jp != 0) | (h != 0) | (kb != 0))
                        cute.gemm(tiled_mma, tAcc, tCrA[(None, None, kb)], tCrB[(None, None, kb)], tAcc)
                with cute.arch.elect_one():
                    tcgen05.commit(ctx.page_empty + bp)
            with cute.arch.elect_one():
                tcgen05.commit(ctx.page_empty + ap)
        return mq, mqph

    @cute.jit
    def _mma_tiles(self, ctx, tiled_mma, a_l, b_l, tAcc, slot, mq, mqph):
        for kt in cutlass.range(self.kc, unroll=1):
            p, mq, mqph = take_page(ctx, slot, MMA_Q, mq, mqph)
            tCrA = tiled_mma.make_fragment_A(page_tensor(ctx.pool, p, 0, a_l, BFloat16))
            tCrB = tiled_mma.make_fragment_B(page_tensor(ctx.pool, p, self.a_bytes, b_l, BFloat16))
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (kb != 0))
                cute.gemm(tiled_mma, tAcc, tCrA[(None, None, kb)], tCrB[(None, None, kb)], tAcc)
            with cute.arch.elect_one():
                tcgen05.commit(ctx.page_empty + p)
        return mq, mqph

    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        tiled_mma, e_l, bias, res = a[0], a[9], a[10], a[11]
        m = ctx.arg(slot, 0)
        n = ctx.arg(slot, 1)
        tidx = ctx.tidx
        acc_shape = tiled_mma.partition_shape_C(self.tiler[:2])
        # stage base fixed at 128 columns whatever this tile's width: stages never overlap
        tAcc = cute.make_tensor(ctx.tmem_ptr + (TMEM_ACC + accidx * C.GEMM_BN),
                                tiled_mma.make_fragment_C(acc_shape).layout)
        sStage = buf_tensor(ctx.stage, e_l, self.cdt)

        half = self.bn // 2                    # two column subtiles, one TMEM load each
        epi_tiler = ((cute.size(tAcc, mode=[0, 0]), half),)
        thr_mma = tiled_mma.get_slice(0)
        tAcc_epi = cute.zipped_divide(tAcc, epi_tiler)
        if cutlass.const_expr(self.glu):    # the 128x64 stage is partitioned like one accumulator subtile
            thr_mma = sm100_utils.make_trivial_tiled_mma(
                BFloat16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, Float32,
                tcgen05.CtaGroup.ONE, (C.GEMM_BM, half)).get_slice(0)
        sC_epi = cute.zipped_divide(thr_mma.partition_C(sStage), epi_tiler)
        rep = tcgen05.Repetition.x64 if half == 64 else tcgen05.Repetition.x32
        t2r = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), Float32), tAcc_epi[None, 0])
        thr = t2r.get_slice(tidx)
        tDtC = thr.partition_S(tAcc_epi)
        tDsC = thr.partition_D(sC_epi)
        rAcc = cute.make_rmem_tensor(tDsC[None, None, 0].shape, Float32)
        rC = cute.make_rmem_tensor(tDsC[None, None, 0].shape, BFloat16)

        rstd = Float32(1.0)
        if cutlass.const_expr(self.rms is not None):    # overlaps the MMA main loop
            rstd = self._rstd(ctx, a[15], slot, m)
        wait_bar(ctx, ctx.acc_full + accidx, accph, slot, "acc_full")
        if cutlass.const_expr(self.glu):
            rU = cute.make_rmem_tensor(tDsC[None, None, 0].shape, Float32)
            cute.copy(t2r, tDtC[None, None, 0], rAcc)
            cute.copy(t2r, tDtC[None, None, 1], rU)
            for j in cutlass.range_constexpr(cute.size(rAcc)):
                rC[j] = (dmath.act("silu", rAcc[j] * rstd) * (rU[j] * rstd)).to(BFloat16)
            cute.autovec_copy(rC, tDsC[None, None, 0])
        for i in cutlass.range_constexpr(0 if self.glu else 2):
            cute.copy(t2r, tDtC[None, None, i], rAcc)
            if cutlass.const_expr(self.rms is not None):
                for j in cutlass.range_constexpr(cute.size(rAcc)):
                    rAcc[j] = rAcc[j] * rstd
            if cutlass.const_expr(self.partial and self.cdt is Float32):   # staged as is, TMA reduce-added
                cute.autovec_copy(rAcc, tDsC[None, None, i])
            elif cutlass.const_expr(self.partial):      # bf16 partial, reduce-added into the residual stream
                for j in cutlass.range_constexpr(cute.size(rAcc)):
                    rC[j] = rAcc[j].to(BFloat16)
                cute.autovec_copy(rC, tDsC[None, None, i])
            else:
                for j in cutlass.range_constexpr(cute.size(rAcc)):
                    v = rAcc[j]
                    # 32x32b copies: thread = TMEM lane = row; register j of subtile i = column i*half + j
                    row = m * C.GEMM_BM + tidx
                    col = n * self.bn + (i * half + j)
                    if cutlass.const_expr(self.ib is not None):
                        bv = Float32(0.0)
                        if col < self.N:
                            bv = bias[self.layer_off(ctx, slot, self.ib), col].to(Float32)
                        v = v + bv
                    v = dmath.act(self.act, v)
                    if cutlass.const_expr(self.ir is not None):
                        rv = Float32(0.0)
                        if row < self.M and col < self.N:
                            rv = res[row + self.layer_off(ctx, slot, self.ir), col].to(Float32)
                        v = v + rv
                    rC[j] = v.to(BFloat16)
                cute.autovec_copy(rC, tDsC[None, None, i])
        cute.arch.fence_view_async_tmem_load()
        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
        if tidx == 0:
            cute.arch.mbarrier_arrive(ctx.acc_empty + accidx)
        accidx, accph = ring_next(accidx, accph, 2)
        return cq, cqph, accidx, accph, sfph, pvph

    @cute.jit
    def _rstd(self, ctx, x, slot, m):
        """rsqrt(mean(x_row^2) + eps) of this thread's accumulator row (TMEM lane = row): warp w reduces
        rows 32w..32w+31 one at a time across its lanes, the owning lane keeps the result."""
        lane = ctx.lane
        base = m * C.GEMM_BM + ctx.warp * 32
        nch = (self.K // VEC + 31) // 32
        rv = cute.make_rmem_tensor((VEC,), Float32)
        mine = Float32(0.0)
        for rr in cutlass.range(32, unroll=1):
            row = base + rr
            if row < self.M:
                xrow = x[(row + self.layer_off(ctx, slot, self.ix), None)]
                ss = Float32(0.0)
                for t in cutlass.range_constexpr(nch):
                    ci = t * 32 + lane
                    rv.fill(0.0)
                    if ci * VEC < self.K:
                        rv.store(cute.local_tile(xrow, (VEC,), (ci,)).load().to(Float32))
                    for e in cutlass.range_constexpr(VEC):
                        ss = ss + rv[e] * rv[e]
                ss = cute.arch.warp_reduction(ss, operator.add)
                if lane == rr:
                    mine = dmath.rsqrt(ss * Float32(1.0 / self.K) + Float32(self.rms))
        return mine

    @cute.jit
    def store(self, ctx, a, slot):
        atom_c, tma_c, e_l = a[5], a[6], a[9]
        gC = cute.local_tile(tma_c, (C.GEMM_BM, self.obn), (None, None))
        tS, tG = cpasync.tma_partition(atom_c, 0, cute.make_layout(1),
                                       cute.group_modes(buf_tensor(ctx.stage, e_l, self.cdt),
                                                        0, 2),
                                       cute.group_modes(gC, 0, 2))
        cute.copy(atom_c, tS, tG[(None, ctx.arg(slot, 0) + self.layer_off(ctx, slot, self.io, C.GEMM_BM),
                                  ctx.arg(slot, 1))])
