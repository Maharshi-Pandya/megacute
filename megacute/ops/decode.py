"""Decode-step ops: RoPE + KV-cache append, decode attention, position increment.

One decode step processes one new token per sequence; all sequences share one position pos
(a (1, 1) int32 state tensor), as in MegaKittens' llama decode. pos is read on device, so one
compiled program serves every step, and Increment advances it at the end of the launch.

DecodeAttention, one instruction per (sequence, kv head): the loader streams the sequence's
128-key K and V tiles of that kv head ([0, pos] only) through pool pages to the compute
warpgroup. Each compute warp takes 32 keys of every tile against the head's GQA group (<= 8
query heads, zero-padded to the 16 rows of an mma.sync m16n8k16) and keeps its own online
softmax state; the four partial (max, sum, O) are combined through the staging buffer, which this
op borrows as scratch (stage_scratch). A 16-row group is far below tcgen05's minimum M, and the
op is bandwidth-bound, so warp-level MMAs are the right tool.
"""

import operator

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import warp as wmma
from cutlass.tensor_utils import LayoutEnum

from .. import config as C
from .. import dmath
from ..kernel import COMPUTE_Q, buf_tensor, fill_page, page_tensor, take_page
from .base import Node, epi_layout

VEC = 8         # elements per 16-byte chunk
BKV = 128       # keys per K/V tile (one pool page)
KW = BKV // C.COMPUTE_WARPS    # keys per compute warp per tile
NEG_INF = float("-inf")


def mn_view(t):
    """mma.sync accumulator ((2, 2), M, N) viewed as (rows, cols) of this thread."""
    l = cute.make_layout(t.layout.shape)
    mn = cute.make_layout(((l.shape[0][1], l.shape[1]), (l.shape[0][0], l.shape[2])),
                          stride=((l.stride[0][1], l.stride[1]), (l.stride[0][0], l.stride[2])))
    return cute.make_tensor(t.iterator, cute.composition(t.layout, mn))


def quad(v, op):
    """Reduce over the 4 threads that share an accumulator row."""
    v = op(v, cute.arch.shuffle_sync_bfly(v, offset=2, mask=-1, mask_and_clamp=31))
    return op(v, cute.arch.shuffle_sync_bfly(v, offset=1, mask=-1, mask_and_clamp=31))


class RopeAppendNode(Node):
    consumer = "compute"

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.B, self.H, self.KVH, self.D, self.L = op.qkv.rows, op.heads, op.kv_heads, op.head_dim, op.max_seq
        self.iqkv, self.icos, self.isin, self.ipos = (self.view(v) for v in (op.qkv, op.cos, op.sin, op.pos))
        self.iq, self.ik, self.iv = self.view(op.q_out), self.view(op.k_cache), self.view(op.v_cache)
        self.ws = op.qkv.buf.dtype == torch.float32    # split-K workspace: read, then zeroed

    def host(self, views):
        return tuple(views[i] for i in (self.iqkv, self.icos, self.isin, self.ipos, self.iq, self.ik, self.iv))

    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        qkv, cos, sin, pos_t, qo, kc, vc = a
        tidx = ctx.tidx
        R = C.COMPUTE_WARPS                                    # sequences per instruction
        r0 = ctx.arg(slot, 0) * R
        DV = self.D // VEC
        HC = DV // 2                                           # chunks per half head
        units = (self.H + self.KVH) * HC                       # (head, chunk) pairs to rotate
        nv = self.KVH * DV
        x1 = cute.make_rmem_tensor((VEC,), Float32)
        x2 = cute.make_rmem_tensor((VEC,), Float32)
        c1 = cute.make_rmem_tensor((VEC,), Float32)
        c2 = cute.make_rmem_tensor((VEC,), Float32)
        s1 = cute.make_rmem_tensor((VEC,), Float32)
        s2 = cute.make_rmem_tensor((VEC,), Float32)
        o1 = cute.make_rmem_tensor((VEC,), BFloat16)
        o2 = cute.make_rmem_tensor((VEC,), BFloat16)
        rz = cute.make_rmem_tensor((VEC,), Float32)
        rz.fill(0.0)
        pos = pos_t[0, 0]
        cs = cos[(pos, None)]
        sn = sin[(pos, None)]
        # (sequence, pair) items spread over the whole warpgroup: a decode step has few sequences
        for t in cutlass.range_constexpr((R * units + C.COMPUTE_THREADS - 1) // C.COMPUTE_THREADS):
            i = t * C.COMPUTE_THREADS + tidx
            rr = i // units
            u = i - rr * units
            b = r0 + rr
            if (i < R * units) & (b < self.B):
                xrow = qkv[(b + self.layer_off(ctx, slot, self.iqkv), None)]
                hh = u // HC
                c = u - hh * HC
                x1.store(cute.local_tile(xrow, (VEC,), (hh * DV + c,)).load().to(Float32))
                x2.store(cute.local_tile(xrow, (VEC,), (hh * DV + c + HC,)).load().to(Float32))
                if cutlass.const_expr(self.ws):
                    cute.local_tile(xrow, (VEC,), (hh * DV + c,)).store(rz.load())
                    cute.local_tile(xrow, (VEC,), (hh * DV + c + HC,)).store(rz.load())
                c1.store(cute.local_tile(cs, (VEC,), (c,)).load().to(Float32))
                c2.store(cute.local_tile(cs, (VEC,), (c + HC,)).load().to(Float32))
                s1.store(cute.local_tile(sn, (VEC,), (c,)).load().to(Float32))
                s2.store(cute.local_tile(sn, (VEC,), (c + HC,)).load().to(Float32))
                for e in cutlass.range_constexpr(VEC):     # rotate-half: [x1, x2] -> [-x2, x1]
                    o1[e] = (x1[e] * c1[e] - x2[e] * s1[e]).to(BFloat16)
                    o2[e] = (x2[e] * c2[e] + x1[e] * s2[e]).to(BFloat16)
                if hh < self.H:
                    qrow = qo[(b + self.layer_off(ctx, slot, self.iq), None)]
                    cute.local_tile(qrow, (VEC,), (hh * DV + c,)).store(o1.load())
                    cute.local_tile(qrow, (VEC,), (hh * DV + c + HC,)).store(o2.load())
                else:
                    krow = kc[(b * self.L + pos + self.layer_off(ctx, slot, self.ik), None)]
                    kh = hh - self.H
                    cute.local_tile(krow, (VEC,), (kh * DV + c,)).store(o1.load())
                    cute.local_tile(krow, (VEC,), (kh * DV + c + HC,)).store(o2.load())
        for t in cutlass.range_constexpr((R * nv + C.COMPUTE_THREADS - 1) // C.COMPUTE_THREADS):
            i = t * C.COMPUTE_THREADS + tidx
            rr = i // nv
            v = i - rr * nv
            b = r0 + rr
            if (i < R * nv) & (b < self.B):
                xrow = qkv[(b + self.layer_off(ctx, slot, self.iqkv), None)]
                vrow = vc[(b * self.L + pos + self.layer_off(ctx, slot, self.iv), None)]
                xv = cute.local_tile(xrow, (VEC,), ((self.H + self.KVH) * DV + v,))
                cute.local_tile(vrow, (VEC,), (v,)).store(xv.load().to(BFloat16))
                if cutlass.const_expr(self.ws):
                    xv.store(rz.load())
        return cq, cqph, accidx, accph, sfph, pvph


class DecodeAttentionNode(Node):
    consumer = "compute"
    stage_scratch = True

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.D, self.L = op.head_dim, op.max_seq
        self.G = op.heads // op.kv_heads
        self.scale_log2 = float(op.scale) * dmath.LOG2E
        self.iq, self.io, self.ipos = self.view(op.q), self.view(op.out), self.view(op.pos)
        self.ik, self.iv = self.view(op.k_cache), self.view(op.v_cache)
        self.tile_bytes = BKV * self.D * 2
        # fused RoPE + append (runtime.fuse_rope_attention): the cache holds keys [0, pos) only, the
        # new token joins from shared memory
        self.fused = op.k_new is not None
        if self.fused:
            self.ikn, self.ivn, self.icos, self.isin = (self.view(v) for v in (op.k_new, op.v_new, op.cos, op.sin))
            self.ws = op.q.buf.dtype == torch.float32
        # split-KV (runtime.split_kv): split s streams blocks s, s + S, s + 2S, ... (strided, so any context
        # length spreads over all splits); the last split to arrive combines
        self.S = op.splits
        # head-parallel: a page is one key tile of Kp keys for all 4 heads of the instruction
        self.hp = op.head_parallel
        self.Kp = 32768 // (C.COMPUTE_WARPS * self.D * 2)
        if self.S > 1:
            self.ipart, self.icount = self.view(op.part), self.view(op.count)
            self.KVH = op.kv_heads

    def host(self, views):
        kv_l = epi_layout(cols=self.D)       # 128 x D row-major, TMA-proven 128B swizzle
        rows = BKV
        if self.hp:                          # Kp x D sub-tiles, same row swizzle
            rows = self.Kp
            kv_l = cute.select(sm100_utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, (rows, self.D), 1),
                               mode=[0, 1])
        g2s = cpasync.CopyBulkTensorTileG2SOp()
        atom_k, tma_k = cpasync.make_tiled_tma_atom(g2s, views[self.ik], kv_l, (rows, self.D))
        atom_v, tma_v = cpasync.make_tiled_tma_atom(g2s, views[self.iv], kv_l, (rows, self.D))
        base = (atom_k, tma_k, atom_v, tma_v, kv_l, views[self.iq], views[self.io], views[self.ipos])
        if self.fused:
            base += (views[self.ikn], views[self.ivn], views[self.icos], views[self.isin], views[self.ik], views[self.iv])
        if self.S > 1:
            base += (views[self.ipart], views[self.icount])
        return base

    @cute.jit
    def mine(self, nblk, s):
        """Block count of split s: blocks s, s + S, ... below nblk."""
        n = Int32(0)
        if nblk > s:
            n = (nblk - s + self.S - 1) // self.S
        return n

    def nblk(self, pos):
        """K/V tiles to stream: keys [0, pos] -- or [0, pos) when the new token comes from smem."""
        if cutlass.const_expr(self.fused):
            return (pos + BKV - 1) // BKV
        return (pos + BKV) // BKV

    @cute.jit
    def load(self, ctx, a, slot, pmask, pcur, mq, mqph, cq, cqph):
        if cutlass.const_expr(self.hp):
            pmask, pcur, cq, cqph = self._load_hp(ctx, a, slot, pmask, pcur, cq, cqph)
        else:
            pmask, pcur, cq, cqph = self._load_kp(ctx, a, slot, pmask, pcur, cq, cqph)
        return pmask, pcur, mq, mqph, cq, cqph

    @cute.jit
    def _load_hp(self, ctx, a, slot, pmask, pcur, cq, cqph):
        """Head-parallel: per key tile, one K page and one V page, each holding the Kp x D sub-tiles of
        the instruction's 4 kv heads (one TMA box per head)."""
        atom_k, tma_k, atom_v, tma_v, kv_l, pos_t = a[0], a[1], a[2], a[3], a[4], a[7]
        b = ctx.arg(slot, 0)
        hg = ctx.arg(slot, 1)
        Kp, D = self.Kp, self.D
        pos = pos_t[0, 0]
        nkeys = pos if self.fused else pos + 1
        nt = (nkeys + Kp - 1) // Kp
        gK = cute.local_tile(tma_k, (Kp, D), (None, None))
        gV = cute.local_tile(tma_v, (Kp, D), (None, None))
        rk = b * (self.L // Kp) + self.layer_off(ctx, slot, self.ik, Kp)
        rv = b * (self.L // Kp) + self.layer_off(ctx, slot, self.iv, Kp)
        for j in cutlass.range(nt, unroll=1):
            p, pmask, pcur, cq, cqph = fill_page(ctx, slot, COMPUTE_Q, C.PAGE_BYTES, pmask, pcur, cq, cqph)
            for w in cutlass.range_constexpr(C.COMPUTE_WARPS):
                self._tma(ctx, atom_k, gK, kv_l, p, rk + j, hg * C.COMPUTE_WARPS + w, w * Kp * D * 2)
            p, pmask, pcur, cq, cqph = fill_page(ctx, slot, COMPUTE_Q, C.PAGE_BYTES, pmask, pcur, cq, cqph)
            for w in cutlass.range_constexpr(C.COMPUTE_WARPS):
                self._tma(ctx, atom_v, gV, kv_l, p, rv + j, hg * C.COMPUTE_WARPS + w, w * Kp * D * 2)
        return pmask, pcur, cq, cqph

    @cute.jit
    def _load_kp(self, ctx, a, slot, pmask, pcur, cq, cqph):
        atom_k, tma_k, atom_v, tma_v, kv_l, pos_t = a[0], a[1], a[2], a[3], a[4], a[7]
        b = ctx.arg(slot, 0)
        h = ctx.arg(slot, 1)
        sp = ctx.arg(slot, 2)
        nmine = self.mine(self.nblk(pos_t[0, 0]), sp)
        gK = cute.local_tile(tma_k, (BKV, self.D), (None, None))
        gV = cute.local_tile(tma_v, (BKV, self.D), (None, None))
        rk = b * (self.L // BKV) + sp + self.layer_off(ctx, slot, self.ik, BKV)
        rv = b * (self.L // BKV) + sp + self.layer_off(ctx, slot, self.iv, BKV)
        for jj in cutlass.range(nmine, unroll=1):
            p, pmask, pcur, cq, cqph = fill_page(ctx, slot, COMPUTE_Q, self.tile_bytes, pmask, pcur, cq, cqph)
            self._tma(ctx, atom_k, gK, kv_l, p, rk + jj * self.S, h)
            p, pmask, pcur, cq, cqph = fill_page(ctx, slot, COMPUTE_Q, self.tile_bytes, pmask, pcur, cq, cqph)
            self._tma(ctx, atom_v, gV, kv_l, p, rv + jj * self.S, h)
        return pmask, pcur, cq, cqph

    @cute.jit
    def _tma(self, ctx, atom, g, s_l, p, r, c, byte_off=0):
        sT = page_tensor(ctx.pool, p, byte_off, s_l, BFloat16)
        tS, tG = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                       cute.group_modes(sT, 0, 2), cute.group_modes(g, 0, 2))
        cute.copy(atom, tG[(None, r, c)], tS, tma_bar_ptr=ctx.page_full + p)

    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        if cutlass.const_expr(self.hp):
            cq, cqph = self._compute_hp(ctx, a, slot, cq, cqph)
        else:
            cq, cqph = self._compute_kp(ctx, a, slot, cq, cqph)
        return cq, cqph, accidx, accph, sfph, pvph

    @cute.jit
    def _compute_hp(self, ctx, a, slot, cq, cqph):
        """Head-parallel: compute warp w is kv head 4*group + w over all keys, start to finish -- its own
        query group, online softmax, new token (fused) and output; the warps only meet to free pages."""
        kv_l, q, out, pos_t = a[4], a[5], a[6], a[7]
        D, G, Kp = self.D, self.G, self.Kp
        DV = D // VEC
        b = ctx.arg(slot, 0)
        tidx, warp, lane = ctx.tidx, ctx.warp, ctx.lane
        hw = ctx.arg(slot, 1) * C.COMPUTE_WARPS + warp       # this warp's kv head
        pos = pos_t[0, 0]
        nkeys = pos if self.fused else pos + 1
        nt = (nkeys + Kp - 1) // Kp
        last = nkeys - 1
        # per-warp scratch: Q tile (16 x D bf16; reused for the fp32 O partial, 8 x D) | new k, v | scores
        # (a dynamic warp offset erases the pointer's alignment: assume() restores it for vector access)
        at = lambda off: ctx.stage + cute.assume(off, divby=64)
        sQ = buf_tensor(at(warp * 32 * D), cute.make_layout((16, D), stride=(D, 1)), BFloat16)
        sKV = buf_tensor(at(16384 + warp * 8 * D), cute.make_layout((2, D), stride=(D, 1)), Float32)
        sNew = buf_tensor(at(24576 + warp * 64), cute.make_layout((8,)), Float32)
        sML = buf_tensor(at(24576 + 256 + warp * 64), cute.make_layout((8, 2), stride=(2, 1)), Float32)
        qrow = q[(b + self.layer_off(ctx, slot, self.iq), None)]
        rq = cute.make_rmem_tensor((VEC,), BFloat16)
        if cutlass.const_expr(self.fused):
            self._rope(ctx, a, slot, b, hw, pos, qrow, sQ, sKV, lane, 32)
            cute.arch.sync_warp()
            for r in cutlass.range_constexpr(G):
                dot = Float32(0.0)
                for d in cutlass.range_constexpr(D // 32):
                    dot = dot + sQ[r, d * 32 + lane].to(Float32) * sKV[0, d * 32 + lane]
                dot = cute.arch.warp_reduction(dot, operator.add)
                if lane == 0:
                    sNew[r] = dot * Float32(self.scale_log2)
            self._append_zero(ctx, a, slot, b, hw, pos, qrow, sKV, lane, 32)
        else:
            for t in cutlass.range_constexpr(16 * DV // 32):
                i = t * 32 + lane
                qr = i // DV
                qc = i - qr * DV
                rq.fill(0.0)
                if qr < G:
                    rq.store(cute.local_tile(qrow, (VEC,), ((hw * G + qr) * DV + qc,)).load())
                cute.local_tile(sQ[(qr, None)], (VEC,), (qc,)).store(rq.load())
        cute.arch.sync_warp()

        mma = cute.make_tiled_mma(wmma.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 1, 1),
                                  permutation_mnk=(16, 16, 16))
        thr = mma.get_slice(lane)
        ld = lambda tr: cute.make_copy_atom(wmma.LdMatrix8x8x16bOp(transpose=tr, num_matrices=4), BFloat16)
        ldB = cute.make_tiled_copy_B(ld(False), mma).get_slice(lane)
        ldBt = cute.make_tiled_copy_B(ld(True), mma).get_slice(lane)
        tSrQ = thr.make_fragment_A(thr.partition_A(sQ))
        tSrQ.store(thr.partition_A(sQ).load())
        acc_O = cute.make_rmem_tensor(thr.partition_shape_C((16, D)), Float32)
        acc_O.fill(0.0)
        oM = mn_view(acc_O)
        cS = mn_view(thr.partition_C(cute.make_identity_tensor((16, Kp))))
        row_max = cute.make_rmem_tensor((2,), Float32)
        row_sum = cute.make_rmem_tensor((2,), Float32)
        row_max.fill(NEG_INF)
        row_sum.fill(0.0)
        for j in cutlass.range(nt, unroll=1):
            kp, cq, cqph = take_page(ctx, slot, COMPUTE_Q, cq, cqph)
            sK = page_tensor(ctx.pool, kp, warp * Kp * D * 2, kv_l, BFloat16)
            tSrK = thr.make_fragment_B(thr.partition_B(sK))
            cute.copy(ldB, ldB.partition_S(sK), ldB.retile(tSrK))
            acc_S = cute.make_rmem_tensor(thr.partition_shape_C((16, Kp)), Float32)
            acc_S.fill(0.0)
            for k in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
                cute.gemm(mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)
            self._softmax_tile(acc_S, cS, oM, row_max, row_sum, j * Kp, last)
            rP = cute.make_fragment_like(acc_S, BFloat16)
            rP.store(acc_S.load().to(BFloat16))
            pd = cute.logical_divide(rP.layout, (None, None, 2))
            tOrP = cute.make_tensor(rP.iterator, cute.make_layout(
                ((pd.shape[0], pd.shape[2][0]), pd.shape[1], pd.shape[2][1]),
                stride=((pd.stride[0], pd.stride[2][0]), pd.stride[1], pd.stride[2][1])))
            vp, cq, cqph = take_page(ctx, slot, COMPUTE_Q, cq, cqph)
            sV = page_tensor(ctx.pool, vp, warp * Kp * D * 2, kv_l, BFloat16)
            sVt = cute.composition(sV, cute.make_layout((D, Kp), stride=(Kp, 1)))
            tOrV = thr.make_fragment_B(thr.partition_B(sVt))
            cute.copy(ldBt, ldBt.partition_S(sVt), ldBt.retile(tOrV))
            for k in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
                cute.gemm(mma, acc_O, tOrP[None, None, k], tOrV[None, None, k], acc_O)
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            if tidx == 0:
                cute.arch.mbarrier_arrive(ctx.page_empty + kp)
                cute.arch.mbarrier_arrive(ctx.page_empty + vp)

        # this warp's partial (rows < G) -> its own scratch (the Q tile is dead), then finish per lane
        sOp = buf_tensor(at(warp * 32 * D), cute.make_layout((1, 8, D), stride=(8 * D, D, 1)), Float32)
        sMLw = cute.make_tensor(sML.iterator, cute.make_layout((1, 8, 2), stride=(16, 2, 1)))
        cO = mn_view(thr.partition_C(cute.make_identity_tensor((16, D))))
        lsum = quad(row_sum[0], operator.add)
        row = cO[0, 0][0]
        if row < G:
            if lane % 4 == 0:
                sMLw[0, row, 0] = row_max[0]
                sMLw[0, row, 1] = lsum
            for c in cutlass.range_constexpr(cute.size(oM, mode=[1])):
                sOp[0, row, cO[0, c][1]] = oM[0, c]
        cute.arch.sync_warp()
        orow = out[(b + self.layer_off(ctx, slot, self.io), None)]
        for t in cutlass.range_constexpr((G * DV + 31) // 32):
            i = t * 32 + lane
            r = i // DV
            if i < G * DV:
                self._finish(sMLw, sOp, sNew, sKV, 1, r, i - r * DV, orow, hw)
        return cq, cqph

    @cute.jit
    def _softmax_tile(self, acc_S, cS, oM, row_max, row_sum, k0, last):
        """Online softmax over one key tile (log2 units, scaled before the max), rescaling O."""
        sM = mn_view(acc_S)
        for ri in cutlass.range_constexpr(2):
            mx = Float32(NEG_INF)
            for c in cutlass.range_constexpr(cute.size(sM, mode=[1])):
                s = sM[ri, c] * Float32(self.scale_log2)
                if k0 + cS[ri, c][1] > last:
                    s = Float32(NEG_INF)
                sM[ri, c] = s
                mx = dmath.fmax(mx, s)
            m_new = dmath.fmax(row_max[ri], quad(mx, dmath.fmax))
            m_use = m_new
            if m_new == Float32(NEG_INF):      # no valid key yet in this slice
                m_use = Float32(0.0)
            corr = dmath.ex2(row_max[ri] - m_use)
            ssum = Float32(0.0)
            for c in cutlass.range_constexpr(cute.size(sM, mode=[1])):
                pv = dmath.ex2(sM[ri, c] - m_use)
                sM[ri, c] = pv
                ssum = ssum + pv
            row_sum[ri] = row_sum[ri] * corr + ssum
            row_max[ri] = m_new
            for c in cutlass.range_constexpr(cute.size(oM, mode=[1])):
                oM[ri, c] = oM[ri, c] * corr

    @cute.jit
    def _compute_kp(self, ctx, a, slot, cq, cqph):
        kv_l, q, out, pos_t = a[4], a[5], a[6], a[7]
        D, G = self.D, self.G
        DV = D // VEC
        b = ctx.arg(slot, 0)
        h = ctx.arg(slot, 1)
        tidx, warp, lane = ctx.tidx, ctx.warp, ctx.lane
        pos = pos_t[0, 0]
        sp = ctx.arg(slot, 2)
        nmine = self.mine(self.nblk(pos), sp)
        last = pos - 1 if self.fused else pos        # last key served from the cache

        # staging-buffer scratch: Q (16 x D bf16) | per-warp (max, sum) | per-warp O partials (fp32) |
        # fused: new k, v (fp32) and the new token's score per query row
        sQ = buf_tensor(ctx.stage, cute.make_layout((16, D), stride=(D, 1)), BFloat16)
        sML = buf_tensor(ctx.stage + 4096, cute.make_layout((C.COMPUTE_WARPS, 8, 2), stride=(16, 2, 1)), Float32)
        sOp = buf_tensor(ctx.stage + 8192, cute.make_layout((C.COMPUTE_WARPS, 8, D), stride=(8 * D, D, 1)), Float32)
        sKV = buf_tensor(ctx.stage + 24576, cute.make_layout((2, D), stride=(D, 1)), Float32)
        sNew = buf_tensor(ctx.stage + 24576 + 8 * D, cute.make_layout((8,)), Float32)
        sFlag = buf_tensor(ctx.stage + 24576 + 8 * D + 32, cute.make_layout((1,)), Int32)

        qrow = q[(b + self.layer_off(ctx, slot, self.iq), None)]
        rq = cute.make_rmem_tensor((VEC,), BFloat16)
        if cutlass.const_expr(self.fused):
            self._rope(ctx, a, slot, b, h, pos, qrow, sQ, sKV, tidx, C.COMPUTE_THREADS)
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            for rr in cutlass.range_constexpr((G + C.COMPUTE_WARPS - 1) // C.COMPUTE_WARPS):
                qr = warp + rr * C.COMPUTE_WARPS    # the new token's score for query row qr: one warp per row
                if qr < G:
                    dot = Float32(0.0)
                    for d in cutlass.range_constexpr(D // 32):
                        dot = dot + sQ[qr, d * 32 + lane].to(Float32) * sKV[0, d * 32 + lane]
                    dot = cute.arch.warp_reduction(dot, operator.add)
                    if lane == 0:
                        sNew[qr] = dot * Float32(self.scale_log2)
            if cutlass.const_expr(self.S == 1):     # split-KV: the last split appends instead
                self._append_zero(ctx, a, slot, b, h, pos, qrow, sKV, tidx, C.COMPUTE_THREADS)
        else:
            # this kv head's query group: rows [0, G) of a zero-padded 16-row tile
            for t in cutlass.range_constexpr(16 * DV // C.COMPUTE_THREADS):
                i = t * C.COMPUTE_THREADS + tidx
                qr = i // DV
                qc = i - qr * DV
                rq.fill(0.0)
                if qr < G:
                    rq.store(cute.local_tile(qrow, (VEC,), ((h * G + qr) * DV + qc,)).load())
                cute.local_tile(sQ[(qr, None)], (VEC,), (qc,)).store(rq.load())
        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)

        mma = cute.make_tiled_mma(wmma.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 1, 1),
                                  permutation_mnk=(16, 16, 16))
        thr = mma.get_slice(lane)
        ld = lambda tr: cute.make_copy_atom(wmma.LdMatrix8x8x16bOp(transpose=tr, num_matrices=4), BFloat16)
        ldB = cute.make_tiled_copy_B(ld(False), mma).get_slice(lane)
        ldBt = cute.make_tiled_copy_B(ld(True), mma).get_slice(lane)
        # Q is read once per instruction: plain loads (ldmatrix on this staging-buffer tile faulted with
        # misaligned addresses; K/V ldmatrix from the TMA-swizzled pages is fine)
        tSrQ = thr.make_fragment_A(thr.partition_A(sQ))
        tSrQ.store(thr.partition_A(sQ).load())

        acc_O = cute.make_rmem_tensor(thr.partition_shape_C((16, D)), Float32)
        acc_O.fill(0.0)
        oM = mn_view(acc_O)
        cS = mn_view(thr.partition_C(cute.make_identity_tensor((16, KW))))
        row_max = cute.make_rmem_tensor((2,), Float32)
        row_sum = cute.make_rmem_tensor((2,), Float32)
        row_max.fill(NEG_INF)
        row_sum.fill(0.0)
        for jj in cutlass.range(nmine, unroll=1):
            j = sp + jj * self.S
            kp, cq, cqph = take_page(ctx, slot, COMPUTE_Q, cq, cqph)
            sK = cute.local_tile(page_tensor(ctx.pool, kp, 0, kv_l, BFloat16), (KW, D), (warp, 0))
            tSrK = thr.make_fragment_B(thr.partition_B(sK))
            cute.copy(ldB, ldB.partition_S(sK), ldB.retile(tSrK))
            acc_S = cute.make_rmem_tensor(thr.partition_shape_C((16, KW)), Float32)
            acc_S.fill(0.0)
            for k in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
                cute.gemm(mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)

            # online softmax over this warp's keys (log2 units, scaled before the max)
            sM = mn_view(acc_S)
            k0 = j * BKV + warp * KW
            for ri in cutlass.range_constexpr(2):
                mx = Float32(NEG_INF)
                for c in cutlass.range_constexpr(cute.size(sM, mode=[1])):
                    s = sM[ri, c] * Float32(self.scale_log2)
                    if k0 + cS[ri, c][1] > last:
                        s = Float32(NEG_INF)
                    sM[ri, c] = s
                    mx = dmath.fmax(mx, s)
                m_new = dmath.fmax(row_max[ri], quad(mx, dmath.fmax))
                m_use = m_new
                if m_new == Float32(NEG_INF):      # no valid key yet in this warp's slice
                    m_use = Float32(0.0)
                corr = dmath.ex2(row_max[ri] - m_use)
                ssum = Float32(0.0)
                for c in cutlass.range_constexpr(cute.size(sM, mode=[1])):
                    pv = dmath.ex2(sM[ri, c] - m_use)
                    sM[ri, c] = pv
                    ssum = ssum + pv
                row_sum[ri] = row_sum[ri] * corr + ssum
                row_max[ri] = m_new
                for c in cutlass.range_constexpr(cute.size(oM, mode=[1])):
                    oM[ri, c] = oM[ri, c] * corr

            # O += P V: the S accumulator, as bf16, is the A operand (pairs of n8 tiles -> k16)
            rP = cute.make_fragment_like(acc_S, BFloat16)
            rP.store(acc_S.load().to(BFloat16))
            pd = cute.logical_divide(rP.layout, (None, None, 2))
            tOrP = cute.make_tensor(rP.iterator, cute.make_layout(
                ((pd.shape[0], pd.shape[2][0]), pd.shape[1], pd.shape[2][1]),
                stride=((pd.stride[0], pd.stride[2][0]), pd.stride[1], pd.stride[2][1])))
            vp, cq, cqph = take_page(ctx, slot, COMPUTE_Q, cq, cqph)
            sV = cute.local_tile(page_tensor(ctx.pool, vp, 0, kv_l, BFloat16), (KW, D), (warp, 0))
            sVt = cute.composition(sV, cute.make_layout((D, KW), stride=(KW, 1)))
            tOrV = thr.make_fragment_B(thr.partition_B(sVt))
            cute.copy(ldBt, ldBt.partition_S(sVt), ldBt.retile(tOrV))
            for k in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
                cute.gemm(mma, acc_O, tOrP[None, None, k], tOrV[None, None, k], acc_O)
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            if tidx == 0:
                cute.arch.mbarrier_arrive(ctx.page_empty + kp)
                cute.arch.mbarrier_arrive(ctx.page_empty + vp)

        # per-warp partials -> smem (rows >= G are padding; a thread's second row is >= 8)
        cO = mn_view(thr.partition_C(cute.make_identity_tensor((16, D))))
        lsum = quad(row_sum[0], operator.add)
        row = cO[0, 0][0]
        if row < G:
            if lane % 4 == 0:
                sML[warp, row, 0] = row_max[0]
                sML[warp, row, 1] = lsum
            for c in cutlass.range_constexpr(cute.size(oM, mode=[1])):
                sOp[warp, row, cO[0, c][1]] = oM[0, c]
        cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)

        # combine: thread -> (query row, 8 columns)
        r = tidx // DV
        c = tidx - r * DV
        orow = out[(b + self.layer_off(ctx, slot, self.io), None)]
        if cutlass.const_expr(self.S == 1):
            if r < G:
                self._finish(sML, sOp, sNew, sKV, C.COMPUTE_WARPS, r, c, orow, h)
        else:
            # this split's partial (unnormalised O, max, sum) -> workspace; the last to arrive combines
            part, count = a[-2], a[-1]
            pair = b * self.KVH + h
            prow = part[(pair * self.S + ctx.arg(slot, 2) + self.layer_off(ctx, slot, self.ipart), None)]
            crow = count[(pair + self.layer_off(ctx, slot, self.icount), None)]
            if r < G:
                m = Float32(NEG_INF)
                for w in cutlass.range_constexpr(C.COMPUTE_WARPS):
                    m = dmath.fmax(m, sML[w, r, 0])
                ms = m
                if m == Float32(NEG_INF):           # an empty split (or slice): weights 0, not NaN
                    ms = Float32(0.0)
                ro = cute.make_rmem_tensor((VEC,), Float32)
                ro.fill(0.0)
                lt = Float32(0.0)
                for w in cutlass.range_constexpr(C.COMPUTE_WARPS):
                    sc = dmath.ex2(sML[w, r, 0] - ms)
                    lt = lt + sML[w, r, 1] * sc
                    for e in cutlass.range_constexpr(VEC):
                        ro[e] = ro[e] + sOp[w, r, c * VEC + e] * sc
                cute.local_tile(prow, (VEC,), (r * DV + c,)).store(ro.load())
                if c == 0:
                    prow[G * D + r] = m
                    prow[G * D + G + r] = lt
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            if tidx == 0:
                sFlag[0] = cute.arch.atomic_add(crow.iterator, Int32(1), sem="acq_rel", scope="gpu")
            cute.arch.barrier(barrier_id=1, number_of_threads=C.COMPUTE_THREADS)
            if sFlag[0] == self.S - 1:
                cute.arch.fence_acq_rel_gpu()
                if r < G:
                    base = pair * self.S + self.layer_off(ctx, slot, self.ipart)
                    self._finish_splits(part, base, sNew, sKV, r, c, orow, h)
                if tidx == 0:
                    crow[0] = Int32(0)              # ready for the next launch
                if cutlass.const_expr(self.fused):
                    self._append_zero(ctx, a, slot, b, h, pos, qrow, sKV, tidx, C.COMPUTE_THREADS)
        return cq, cqph


    @cute.jit
    def _finish(self, sML, sOp, sNew, sKV, nw: cutlass.Constexpr, r, c, orow, h):
        """out[row r, chunk c] = combination of the nw per-warp partials (+ the new token)."""
        G, DV = self.G, self.D // VEC
        m = Float32(NEG_INF)
        for w in cutlass.range_constexpr(nw):
            m = dmath.fmax(m, sML[w, r, 0])
        ro = cute.make_rmem_tensor((VEC,), Float32)
        ro.fill(0.0)
        lt = Float32(0.0)
        if cutlass.const_expr(self.fused):     # the new token: one more partial, (score, 1, v_new)
            m = dmath.fmax(m, sNew[r])
            sc = dmath.ex2(sNew[r] - m)
            lt = lt + sc
            for e in cutlass.range_constexpr(VEC):
                ro[e] = sKV[1, c * VEC + e] * sc
        for w in cutlass.range_constexpr(nw):
            sc = dmath.ex2(sML[w, r, 0] - m)
            lt = lt + sML[w, r, 1] * sc
            for e in cutlass.range_constexpr(VEC):
                ro[e] = ro[e] + sOp[w, r, c * VEC + e] * sc
        self._store_out(ro, lt, orow, h, r, c)

    @cute.jit
    def _finish_splits(self, part, base, sNew, sKV, r, c, orow, h):
        """out[row r, chunk c] = combination of the S split partials in the workspace (+ new token)."""
        G, D, DV = self.G, self.D, self.D // VEC
        m = Float32(NEG_INF)
        for s in cutlass.range_constexpr(self.S):
            m = dmath.fmax(m, part[base + s, G * D + r])
        ro = cute.make_rmem_tensor((VEC,), Float32)
        ro.fill(0.0)
        rv = cute.make_rmem_tensor((VEC,), Float32)
        lt = Float32(0.0)
        if cutlass.const_expr(self.fused):
            m = dmath.fmax(m, sNew[r])
            sc = dmath.ex2(sNew[r] - m)
            lt = lt + sc
            for e in cutlass.range_constexpr(VEC):
                ro[e] = sKV[1, c * VEC + e] * sc
        for s in cutlass.range_constexpr(self.S):
            sc = dmath.ex2(part[base + s, G * D + r] - m)
            lt = lt + part[base + s, G * D + G + r] * sc
            rv.store(cute.local_tile(part[(base + s, None)], (VEC,), (r * DV + c,)).load())
            for e in cutlass.range_constexpr(VEC):
                ro[e] = ro[e] + rv[e] * sc
        self._store_out(ro, lt, orow, h, r, c)

    @cute.jit
    def _store_out(self, ro, lt, orow, h, r, c):
        inv = dmath.rcp(lt)
        rb = cute.make_rmem_tensor((VEC,), BFloat16)
        for e in cutlass.range_constexpr(VEC):
            rb[e] = (ro[e] * inv).to(BFloat16)
        cute.local_tile(orow, (VEC,), ((h * self.G + r) * (self.D // VEC) + c,)).store(rb.load())

    @cute.jit
    def _append_zero(self, ctx, a, slot, b, h, pos, qrow, sKV, t0, nt: cutlass.Constexpr):
        """Append the new k, v (bf16-exact in sKV) at cache row b*max_seq + pos; zero the split-K
        workspace entries this (sequence, kv head) read: its q group, k and v columns."""
        kn, vn, kc, vc = a[8], a[9], a[12], a[13]
        D, G = self.D, self.G
        DV = D // VEC
        crow = b * self.L + pos
        kcrow = kc[(crow + self.layer_off(ctx, slot, self.ik), None)]
        vcrow = vc[(crow + self.layer_off(ctx, slot, self.iv), None)]
        krow = kn[(b + self.layer_off(ctx, slot, self.ikn), None)]
        vrow = vn[(b + self.layer_off(ctx, slot, self.ivn), None)]
        ob = cute.make_rmem_tensor((VEC,), BFloat16)
        rz = cute.make_rmem_tensor((VEC,), Float32)
        rz.fill(0.0)
        n = (G + 2) * DV if self.ws else 0
        for t in cutlass.range_constexpr((2 * DV + n + nt - 1) // nt):
            i = t * nt + t0
            if i < DV:
                for e in cutlass.range_constexpr(VEC):
                    ob[e] = sKV[0, i * VEC + e].to(BFloat16)
                cute.local_tile(kcrow, (VEC,), (h * DV + i,)).store(ob.load())
            elif i < 2 * DV:
                for e in cutlass.range_constexpr(VEC):
                    ob[e] = sKV[1, (i - DV) * VEC + e].to(BFloat16)
                cute.local_tile(vcrow, (VEC,), (h * DV + i - DV,)).store(ob.load())
            if cutlass.const_expr(self.ws):
                if (i >= 2 * DV) & (i < 2 * DV + G * DV):
                    cute.local_tile(qrow, (VEC,), (h * G * DV + i - 2 * DV,)).store(rz.load())
                elif (i >= 2 * DV + G * DV) & (i < 3 * DV + G * DV):
                    cute.local_tile(krow, (VEC,), (h * DV + i - 2 * DV - G * DV,)).store(rz.load())
                elif (i >= 3 * DV + G * DV) & (i < 4 * DV + G * DV):
                    cute.local_tile(vrow, (VEC,), (h * DV + i - 3 * DV - G * DV,)).store(rz.load())

    @cute.jit
    def _rope(self, ctx, a, slot, b, h, pos, qrow, sQ, sKV, t0, nt: cutlass.Constexpr):
        """Fused prologue: rotate this group's q into sQ (rows >= G zeroed) and the new k into sKV[0],
        copy the new v into sKV[1] (both bf16-rounded, as the cache will hold them)."""
        kn, vn, cos, sin = a[8], a[9], a[10], a[11]
        D, G = self.D, self.G
        DV = D // VEC
        HC = DV // 2
        cs = cos[(pos, None)]
        sn = sin[(pos, None)]
        krow = kn[(b + self.layer_off(ctx, slot, self.ikn), None)]
        vrow = vn[(b + self.layer_off(ctx, slot, self.ivn), None)]
        x1 = cute.make_rmem_tensor((VEC,), Float32)
        x2 = cute.make_rmem_tensor((VEC,), Float32)
        c1 = cute.make_rmem_tensor((VEC,), Float32)
        c2 = cute.make_rmem_tensor((VEC,), Float32)
        s1 = cute.make_rmem_tensor((VEC,), Float32)
        s2 = cute.make_rmem_tensor((VEC,), Float32)
        o1 = cute.make_rmem_tensor((VEC,), BFloat16)
        o2 = cute.make_rmem_tensor((VEC,), BFloat16)
        ob = cute.make_rmem_tensor((VEC,), BFloat16)
        ob.fill(0.0)
        # rotate-half pairs: G q heads x HC chunk pairs, then the k head's HC pairs
        for t in cutlass.range_constexpr((G * HC + HC + nt - 1) // nt):
            i = t * nt + t0
            r = i // HC
            c = i - r * HC
            if i < (G + 1) * HC:
                c1.store(cute.local_tile(cs, (VEC,), (c,)).load().to(Float32))
                c2.store(cute.local_tile(cs, (VEC,), (c + HC,)).load().to(Float32))
                s1.store(cute.local_tile(sn, (VEC,), (c,)).load().to(Float32))
                s2.store(cute.local_tile(sn, (VEC,), (c + HC,)).load().to(Float32))
            if i < G * HC:
                self._rotate(qrow, (h * G + r) * DV + c, HC, x1, x2, c1, c2, s1, s2, o1, o2)
                cute.local_tile(sQ[(r, None)], (VEC,), (c,)).store(o1.load())
                cute.local_tile(sQ[(r, None)], (VEC,), (c + HC,)).store(o2.load())
            elif i < (G + 1) * HC:
                self._rotate(krow, h * DV + c, HC, x1, x2, c1, c2, s1, s2, o1, o2)
                for e in cutlass.range_constexpr(VEC):
                    sKV[0, c * VEC + e] = o1[e].to(Float32)
                    sKV[0, (c + HC) * VEC + e] = o2[e].to(Float32)
        for t in cutlass.range_constexpr(((16 - G) * DV + DV + nt - 1) // nt):
            i = t * nt + t0
            rp = i // DV                            # padding row (offset from G) / chunk of this item
            if i < (16 - G) * DV:                   # zero padding rows of the 16-row Q tile
                cute.local_tile(sQ[(G + rp, None)], (VEC,), (i - rp * DV,)).store(ob.load())
            elif i < (16 - G) * DV + DV:            # v: keep, bf16-rounded
                j = i - (16 - G) * DV
                o1.store(cute.local_tile(vrow, (VEC,), (h * DV + j,)).load().to(BFloat16))
                for e in cutlass.range_constexpr(VEC):
                    sKV[1, j * VEC + e] = o1[e].to(Float32)


    @cute.jit
    def _rotate(self, row, ci, HC: cutlass.Constexpr, x1, x2, c1, c2, s1, s2, o1, o2):
        """o1, o2 = rotate-half of chunks ci, ci + HC of `row`."""
        x1.store(cute.local_tile(row, (VEC,), (ci,)).load().to(Float32))
        x2.store(cute.local_tile(row, (VEC,), (ci + HC,)).load().to(Float32))
        for e in cutlass.range_constexpr(VEC):     # [x1, x2] -> [-x2, x1]
            o1[e] = (x1[e] * c1[e] - x2[e] * s1[e]).to(BFloat16)
            o2[e] = (x2[e] * c2[e] + x1[e] * s2[e]).to(BFloat16)


class IncrementNode(Node):
    consumer = "compute"

    def __init__(self, prog, op, members=None):
        super().__init__(prog, op, members)
        self.ipos = self.view(op.pos)

    def host(self, views):
        return (views[self.ipos],)

    @cute.jit
    def compute(self, ctx, a, slot, cq, cqph, accidx, accph, sfph, pvph):
        pos_t = a[0]
        if ctx.tidx == 0:
            pos_t[0, 0] = pos_t[0, 0] + Int32(1)
        return cq, cqph, accidx, accph, sfph, pvph
