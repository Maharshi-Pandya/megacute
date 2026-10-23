"""Attention forward: one instruction = one (batch, head, 128-query tile).

Q/K/O are passed as (seq, cols, batch) views and V as (cols, seq, batch), so TMA tiles a head
directly out of a fused QKV buffer and clips at batch boundaries (seq need not be a multiple of
128). Per 128-key block j:

  mma      S = Q K_j^T  -> TMEM [256, 384)          commit s_full, release K page
  compute  thread = query row: read S, signal s_empty; mask; online softmax (running max m,
           sum l); once PV_{j-1} is done (pv_done) rescale O in TMEM if the max moved,
           write P (bf16) to smem, signal p_full
  mma      O += P V_j    -> TMEM [384, 384 + D)       commit pv_done, release V page
  end      compute: O / l -> bf16 -> staging -> TMA store; mma releases the Q page

The MMA warp issues S_{j+1} = Q K_{j+1}^T *before* O += P_j V_j: compute frees S as soon as it
is in registers, so the next QK^T overlaps this block's softmax (the loader streams pages in
the same K0 K1 V0 K2 V1 ... order -- ring positions are consumed strictly in order). As in FA4,
the reference max only moves when a row max grows by more than 2^8, so O is rarely rescaled.

Q, K and V are pool pages handed to the MMA warp; Q's page is held for the whole instruction
and released after the last QK^T. Causal instructions stop at the diagonal block.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

from .. import config as C
from .. import dmath
from ..kernel import MMA_Q, TMEM_O, TMEM_S, buf_tensor, fill_page, page_tensor, take_page, wait_bar
from .base import Node, epi_layout

BQ = C.ATTN_BQ
BK = C.ATTN_BK
NEG_INF = float("-inf")
RESCALE_LOG2 = 8.0      # P <= 2^8 between reference-max updates; fp32 O/l absorb that


class AttentionNode(Node):
    consumer = "mma"
    stage_out = True    # P during the KV loop, then the output tile

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.D = op.head_dim
        assert self.D in (64, 128), "attention: head_dim must be 64 or 128"
        assert BQ == BK == 128
        self.seq_q, self.seq_k = op.seq_q, op.seq_k
        self.causal = op.causal
        self.group = op.heads // op.kv_heads
        self.scale_log2 = float(op.scale) * dmath.LOG2E
        self.iq = self.view(op.q, "sbc", seq=op.seq_q)
        self.ik = self.view(op.k, "sbc", seq=op.seq_k)
        self.iv = self.view(op.v, "csb", seq=op.seq_k)
        self.io = self.view(op.out, "sbc", seq=op.seq_q)
        self.nkv_full = (op.seq_k + BK - 1) // BK
        self.tile_bytes = 128 * self.D * 2

    def nkv(self, ctx, slot):
        if cutlass.const_expr(self.causal):
            return ctx.arg(slot, 2) + 1
        return Int32(self.nkv_full)

    def host(self, views):
        D = self.D
        q, k, v, o = views[self.iq], views[self.ik], views[self.iv], views[self.io]
        K_ = OperandMajorMode.K
        mma_qk = sm100_utils.make_trivial_tiled_mma(BFloat16, K_, K_, Float32, tcgen05.CtaGroup.ONE, (BQ, BK))
        mma_pv = sm100_utils.make_trivial_tiled_mma(BFloat16, K_, OperandMajorMode.MN, Float32,
                                                   tcgen05.CtaGroup.ONE, (BQ, D))
        qk_t = (BQ, BK, D)
        pv_t = (BQ, D, BK)
        q_l = cute.select(sm100_utils.make_smem_layout_a(mma_qk, qk_t, BFloat16, 1), mode=[0, 1, 2])
        k_l = cute.select(sm100_utils.make_smem_layout_b(mma_qk, qk_t, BFloat16, 1), mode=[0, 1, 2])
        v_l = cute.select(sm100_utils.make_smem_layout_b(mma_pv, pv_t, BFloat16, 1), mode=[0, 1, 2])
        p_atom = tcgen05.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(K_, BFloat16, (BQ, BK)), BFloat16)
        p_l = cute.tile_to_shape(p_atom, (BQ, BK), (0, 1))
        o_l = epi_layout(cols=D)
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        atom_q, tma_q = cute.nvgpu.make_tiled_tma_atom_A(g2s, q, q_l, qk_t, mma_qk)
        atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(g2s, k, k_l, qk_t, mma_qk)
        atom_v, tma_v = cute.nvgpu.make_tiled_tma_atom_B(g2s, v, v_l, pv_t, mma_pv)
        atom_o, tma_o = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), o, o_l, (BQ, D))
        return (mma_qk, mma_pv, atom_q, tma_q, atom_k, tma_k, atom_v, tma_v, atom_o, tma_o,
                q_l, k_l, v_l, p_l, o_l)

    # ------------------------------------------------------------------------------ loader
    @cute.jit
    def load(self, ctx, a, slot, pmask, pcur, mq, mqph, cq, cqph):
        mma_qk, mma_pv, atom_q, tma_q, atom_k, tma_k, atom_v, tma_v = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
        q_l, k_l, v_l = a[10], a[11], a[12]
        b = ctx.arg(slot, 0)
        h = ctx.arg(slot, 1)
        qi = ctx.arg(slot, 2)
        nkv = self.nkv(ctx, slot)
        thr_qk = mma_qk.get_slice(0)
        thr_pv = mma_pv.get_slice(0)
        # rank-3 (seq|cols, cols|seq, batch) tensors: rank-2 tiler + rank-3 coordinate
        tCgQ = thr_qk.partition_A(cute.local_tile(tma_q, (BQ, self.D), (None, None, None)))
        tCgK = thr_qk.partition_B(cute.local_tile(tma_k, (BK, self.D), (None, None, None)))
        tCgV = thr_pv.partition_B(cute.local_tile(tma_v, (self.D, BK), (None, None, None)))

        # Q's page stays held for the whole instruction: K/V allocation steps around it
        # merged layers: batch coordinate offset of this layer in each stacked operand
        bq = b + self.layer_off(ctx, slot, self.iq, self.seq_q)
        bk = b + self.layer_off(ctx, slot, self.ik, self.seq_k)
        bv = b + self.layer_off(ctx, slot, self.iv, self.seq_k)
        hk = h // self.group     # GQA: kv head of this query head
        qp, pmask, pcur, mq, mqph = self._load_tile(ctx, slot, atom_q, q_l, tCgQ, (None, qi, h, bq),
                                                    pmask, pcur, mq, mqph, -1)
        p, pmask, pcur, mq, mqph = self._load_tile(ctx, slot, atom_k, k_l, tCgK, (None, 0, hk, bk),
                                                   pmask, pcur, mq, mqph, qp)
        for j in cutlass.range(nkv, unroll=1):
            if j + 1 < nkv:
                p, pmask, pcur, mq, mqph = self._load_tile(ctx, slot, atom_k, k_l, tCgK, (None, j + 1, hk, bk),
                                                           pmask, pcur, mq, mqph, qp)
            p, pmask, pcur, mq, mqph = self._load_tile(ctx, slot, atom_v, v_l, tCgV, (None, hk, j, bv),
                                                       pmask, pcur, mq, mqph, qp)
        return pmask, pcur, mq, mqph, cq, cqph

    @cute.jit
    def _load_tile(self, ctx, slot, atom, s_l, tCg, coord, pmask, pcur, mq, mqph, skip):
        p, pmask, pcur, mq, mqph = fill_page(ctx, slot, MMA_Q, self.tile_bytes, pmask, pcur, mq, mqph, skip)
        sT = page_tensor(ctx.pool, p, 0, s_l, BFloat16)
        tTs, tTg = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                         cute.group_modes(sT, 0, 3), cute.group_modes(tCg, 0, 3))
        cute.copy(atom, tTg[coord], tTs, tma_bar_ptr=ctx.page_full + p)
        return p, pmask, pcur, mq, mqph

    # ------------------------------------------------------------------------------ mma
    @cute.jit
    def _qk(self, ctx, slot, tS, tCrQ, k_l, mq, mqph, sph):
        # own TiledMma: .set() inside this branch must not rebind the shared host object's value,
        # which the compute role reads outside the branch
        mma_qk = sm100_utils.make_trivial_tiled_mma(BFloat16, OperandMajorMode.K, OperandMajorMode.K, Float32,
                                                    tcgen05.CtaGroup.ONE, (BQ, BK))
        p, mq, mqph = take_page(ctx, slot, MMA_Q, mq, mqph)
        wait_bar(ctx, ctx.s_empty, sph, slot, "s_empty")
        tCrK = mma_qk.make_fragment_B(page_tensor(ctx.pool, p, 0, k_l, BFloat16))
        for kb in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
            mma_qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
            cute.gemm(mma_qk, tS, tCrQ[(None, None, kb)], tCrK[(None, None, kb)], tS)
        with cute.arch.elect_one():
            tcgen05.commit(ctx.s_full)
            tcgen05.commit(ctx.page_empty + p)
        return mq, mqph, sph ^ 1

    @cute.jit
    def mma(self, ctx, a, slot, mq, mqph, accidx, accph, sph, pfph):
        mma_qk, mma_pv = a[0], a[1]
        q_l, k_l, v_l, p_l = a[10], a[11], a[12], a[13]
        nkv = self.nkv(ctx, slot)
        tS = cute.make_tensor(ctx.tmem_ptr + TMEM_S,
                              mma_qk.make_fragment_C(mma_qk.partition_shape_C((BQ, BK))).layout)
        tO = cute.make_tensor(ctx.tmem_ptr + TMEM_O,
                              mma_pv.make_fragment_C(mma_pv.partition_shape_C((BQ, self.D))).layout)
        sP = buf_tensor(ctx.pbuf, p_l, BFloat16)
        tCrP = mma_pv.make_fragment_A(mma_pv.get_slice(0).partition_A(sP))

        qp, mq, mqph = take_page(ctx, slot, MMA_Q, mq, mqph)
        tCrQ = mma_qk.make_fragment_A(page_tensor(ctx.pool, qp, 0, q_l, BFloat16))
        mq, mqph, sph = self._qk(ctx, slot, tS, tCrQ, k_l, mq, mqph, sph)
        for j in cutlass.range(nkv, unroll=1):
            if j + 1 < nkv:   # next S = Q K^T first: it overlaps this block's softmax
                mq, mqph, sph = self._qk(ctx, slot, tS, tCrQ, k_l, mq, mqph, sph)
            # O += P V
            vp, mq, mqph = take_page(ctx, slot, MMA_Q, mq, mqph)
            wait_bar(ctx, ctx.p_full, pfph, slot, "p_full")
            pfph = pfph ^ 1
            tCrV = mma_pv.make_fragment_B(page_tensor(ctx.pool, vp, 0, v_l, BFloat16))
            for kb in cutlass.range_constexpr(cute.size(tCrP, mode=[2])):
                mma_pv.set(tcgen05.Field.ACCUMULATE, (j != 0) | (kb != 0))
                cute.gemm(mma_pv, tO, tCrP[(None, None, kb)], tCrV[(None, None, kb)], tO)
            with cute.arch.elect_one():
                tcgen05.commit(ctx.pv_done)
                tcgen05.commit(ctx.page_empty + vp)
        with cute.arch.elect_one():
            tcgen05.commit(ctx.page_empty + qp)
        return mq, mqph, accidx, accph, sph, pfph

    # ------------------------------------------------------------------------------ compute
    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        mma_qk, mma_pv = a[0], a[1]
        p_l, o_l = a[13], a[14]
        D = self.D
        b = ctx.arg(slot, 0)
        h = ctx.arg(slot, 1)
        qi = ctx.arg(slot, 2)
        nkv = self.nkv(ctx, slot)
        tidx = ctx.tidx

        # TMEM views + row-per-thread copies (32x32b: thread t <-> TMEM lane t), 64 columns each
        tS = cute.make_tensor(ctx.tmem_ptr + TMEM_S,
                              mma_qk.make_fragment_C(mma_qk.partition_shape_C((BQ, BK))).layout)
        tO = cute.make_tensor(ctx.tmem_ptr + TMEM_O,
                              mma_pv.make_fragment_C(mma_pv.partition_shape_C((BQ, D))).layout)
        sub = ((BQ, 64),)
        tS_e = cute.zipped_divide(tS, sub)
        tO_e = cute.zipped_divide(tO, sub)
        cS_e = cute.zipped_divide(mma_qk.get_slice(0).partition_C(cute.make_identity_tensor((BQ, BK))), sub)
        cO_e = cute.zipped_divide(mma_pv.get_slice(0).partition_C(cute.make_identity_tensor((BQ, D))), sub)
        ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), Float32)
        st = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x64), Float32)
        ldS = tcgen05.make_tmem_copy(ld, tS_e[None, 0]).get_slice(tidx)
        ldO = tcgen05.make_tmem_copy(ld, tO_e[None, 0]).get_slice(tidx)
        stO = tcgen05.make_tmem_copy(st, tO_e[None, 0]).get_slice(tidx)
        tSt = ldS.partition_S(tS_e)
        tSc = ldS.partition_D(cS_e)
        tOt = ldO.partition_S(tO_e)
        tOt_st = stO.partition_D(tO_e)
        tOc = ldO.partition_D(cO_e)
        NS = BK // 64
        NO = D // 64
        rS = cute.make_rmem_tensor(tSc[None, None, 0].shape, Float32)
        rP = cute.make_rmem_tensor(tSc[None, None, 0].shape, Float32)
        rO = cute.make_rmem_tensor(tOc[None, None, 0].shape, Float32)
        sP = buf_tensor(ctx.pbuf, p_l, BFloat16)
        sOut = buf_tensor(ctx.stage, o_l, BFloat16)
        tPs = ldS.partition_D(cute.zipped_divide(mma_qk.get_slice(0).partition_C(sP), sub))
        tOs = ldO.partition_D(cute.zipped_divide(mma_pv.get_slice(0).partition_C(sOut), sub))
        rB = cute.make_rmem_tensor(tSc[None, None, 0].shape, BFloat16)

        q_row = qi * BQ + tSc[None, None, 0][0][0]    # this thread's query row (TMEM lane)
        m_i = Float32(NEG_INF)
        l_i = Float32(0.0)
        for j in cutlass.range(nkv, unroll=1):
            wait_bar(ctx, ctx.s_full, sfph, slot, "s_full")
            sfph = sfph ^ 1
            # pass 1: row max over this block (scaled to log2 units, masked)
            mx = Float32(NEG_INF)
            for i in cutlass.range_constexpr(NS):
                cute.copy(ldS, tSt[None, None, i], rS)
                crd = tSc[None, None, i]
                for e in cutlass.range_constexpr(cute.size(rS)):
                    kk = j * BK + crd[e][1]
                    s = rS[e] * Float32(self.scale_log2)
                    valid = kk < self.seq_k
                    if cutlass.const_expr(self.causal):
                        valid = valid & (kk <= q_row)
                    if not valid:
                        s = Float32(NEG_INF)
                    rS[e] = s
                    mx = dmath.fmax(mx, s)
                # keep the masked, scaled scores: subtile i reuses rS, so stash them in P space
                for e in cutlass.range_constexpr(cute.size(rS)):
                    if cutlass.const_expr(i == 0):
                        rP[e] = rS[e]
            # second subtile scores stay in rS; first subtile in rP (NS == 2)
            m_new = dmath.fmax(m_i, mx)
            upd = m_new - m_i > Float32(RESCALE_LOG2)     # also true on the first finite max
            m_ref = m_i
            if upd:
                m_ref = m_new
            m_use = m_ref
            if m_ref == Float32(NEG_INF):
                m_use = Float32(0.0)
            corr = Float32(1.0)
            if upd:
                corr = dmath.ex2(m_i - m_use)
            rowsum = Float32(0.0)
            for i in cutlass.range_constexpr(NS):
                for e in cutlass.range_constexpr(cute.size(rS)):
                    s = rP[e] if i == 0 else rS[e]
                    p = dmath.ex2(s - m_use)
                    rowsum = rowsum + p
                    if cutlass.const_expr(i == 0):
                        rP[e] = p
                    else:
                        rS[e] = p
            l_i = l_i * corr + rowsum
            m_i = m_ref
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            if tidx == 0:
                cute.arch.mbarrier_arrive(ctx.s_empty)

            # P buffer and O are free once PV_{j-1} completed; rescale O by corr
            if j > 0:
                wait_bar(ctx, ctx.pv_done, pvph, slot, "pv_done")
                pvph = pvph ^ 1
                if cute.arch.vote_any_sync(upd):          # TMEM ld/st are warp-collective
                    for i in cutlass.range_constexpr(NO):
                        cute.copy(ldO, tOt[None, None, i], rO)
                        for e in cutlass.range_constexpr(cute.size(rO)):
                            rO[e] = rO[e] * corr
                        cute.copy(stO, rO, tOt_st[None, None, i])
                    cute.arch.fence_view_async_tmem_store()
            for i in cutlass.range_constexpr(NS):
                for e in cutlass.range_constexpr(cute.size(rS)):
                    rB[e] = (rP[e] if i == 0 else rS[e]).to(BFloat16)
                cute.autovec_copy(rB, tPs[None, None, i])
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            if tidx == 0:
                cute.arch.mbarrier_arrive(ctx.p_full)

        # epilogue: wait for the last PV, normalize, store
        wait_bar(ctx, ctx.pv_done, pvph, slot, "pv_done")
        pvph = pvph ^ 1
        inv_l = Float32(0.0)
        if l_i > Float32(0.0):
            inv_l = dmath.rcp(l_i)
        for i in cutlass.range_constexpr(NO):
            cute.copy(ldO, tOt[None, None, i], rO)
            for e in cutlass.range_constexpr(cute.size(rO)):
                rB[e] = (rO[e] * inv_l).to(BFloat16)
            cute.autovec_copy(rB, tOs[None, None, i])
        cute.arch.fence_view_async_tmem_load()
        return cq, cqph, accidx, accph, sfph, pvph

    @cute.jit
    def store(self, ctx, a, slot):
        atom_o, tma_o, o_l = a[8], a[9], a[14]
        gO = cute.local_tile(tma_o, (BQ, self.D), (None, None, None))
        tS_o, tG_o = cpasync.tma_partition(atom_o, 0, cute.make_layout(1),
                                           cute.group_modes(buf_tensor(ctx.stage, o_l, BFloat16), 0, 2),
                                           cute.group_modes(gO, 0, 2))
        cute.copy(atom_o, tS_o, tG_o[(None, ctx.arg(slot, 2), ctx.arg(slot, 1),
                                          ctx.arg(slot, 0) + self.layer_off(ctx, slot, self.io, self.seq_q))])
