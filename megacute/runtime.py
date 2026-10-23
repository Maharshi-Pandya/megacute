"""Runtime: compile a Graph into one persistent kernel and run it.

    prog = megacute.compile(g)            # schedule + one cute.compile (TVM-FFI, fake tensors)
    outs = prog(x=x, w=w, ...)            # one kernel launch per call

Every op reads and writes *views*; each distinct (view, mode) becomes one kernel tensor
argument, passed at call time as a strided torch view of its buffer. Dependency counters use
launch epochs: launch e waits for `target * e`, so no memset precedes a launch; counters are
only reset when the next epoch would overflow int32.
"""

from __future__ import annotations

import copy
import os
import time

import torch

from . import config as C
import dataclasses

from .graph import (Attention, Binary, Buffer, DecodeAttention, Graph, Increment, Linear, LinearSplit, Modulate, Norm,
                    Rope, RopeAppend, SplitFinalize, SwiGLU, Unary, View)
from .layers import merge_layers
from .schedule import schedule

MIN_SPLIT_KTILES = 4    # k-tiles per split: below this the atomics outweigh the extra parallelism
INST_US, SM_GBPS = 2.0, 40.0    # split-K cost model, from B200 decode traces: fixed us per instruction, GB/s per SM


def _reads_after(ops, buf, outputs=()) -> bool:
    """Does any of `ops` read `buf`, or is it a graph output?"""
    return any(v.buf is buf for op in ops for v in op.reads) or any(v.buf is buf for v in outputs)


def _same(a: View, b: View) -> bool:
    return (a.buf, a.row0, a.col0, a.shape) == (b.buf, b.row0, b.col0, b.shape)


FOLD_NORM_ROWS = 8     # beyond this, recomputing rstd in every GEMM instruction costs more than the norm hop


def fold_norms(g: Graph) -> Graph:
    """Decode (M <= FOLD_NORM_ROWS): fold each RMSNorm that only feeds GEMMs into them (MegaKittens' rms_qkv /
    rms_upgate). The gain moves into the weights (a derived param W * diag(g), computed at bind
    time) and each GEMM instruction scales its rows by rstd, recomputed from the raw input -- so the
    norm is no longer a separate dependent hop."""
    out = copy.copy(g)
    out.buffers = list(g.buffers)
    derived: dict[int, Buffer] = {}
    ops = list(g.ops)
    keep = []
    for i, op in enumerate(ops):
        users = [o for o in ops[i + 1:] if any(v.buf is getattr(op, "out", None).buf for v in o.reads)] \
            if isinstance(op, Norm) else []
        if not (isinstance(op, Norm) and op.rms and op.bias is None and op.x.rows <= FOLD_NORM_ROWS and users
                and all(isinstance(u, Linear) and _same(u.x, op.out) and u.rms_eps is None and
                        all(v.buf is not op.out.buf for v in (u.residual, u.bias) if v is not None) for u in users)
                and op.out.buf.kind == "temp" and not _reads_after((), op.out.buf, g.outputs)):
            keep.append(op)
            continue
        for u in users:
            w = u.w
            if op.weight is not None:
                d = derived.get(w.buf.id)
                if d is None:
                    d = derived[w.buf.id] = Buffer(f"{w.buf.name}*g", w.buf.rows, w.buf.cols, w.buf.dtype,
                                                   "derived", derive=[])
                    out.buffers.append(d)
                d.derive.append((w.buf, w.row0, w.rows, op.weight))
                w = View(d, w.row0, w.col0, w.rows, w.cols)
            idx = ops.index(u)
            ops[idx] = dataclasses.replace(u, x=op.x, w=w, rms_eps=op.eps)
            ops[idx].layer = getattr(u, "layer", None)
    out.ops = keep
    return out


def fuse_rope_attention(g: Graph) -> Graph:
    """Fold each RopeAppend whose rotated q only feeds the next DecodeAttention (same caches) into
    that attention: one decode op per layer instead of two dependent hops."""
    out = copy.copy(g)
    ops = []
    for i, op in enumerate(g.ops):
        prev = ops[-1] if ops else None
        if (isinstance(op, DecodeAttention) and op.k_new is None and isinstance(prev, RopeAppend)
                and _same(op.q, prev.q_out) and op.k_cache is prev.k_cache and op.v_cache is prev.v_cache
                and op.pos is prev.pos and not _reads_after(g.ops[i + 1:], prev.q_out.buf, g.outputs)):
            H, KVH, D = prev.heads, prev.kv_heads, prev.head_dim
            qkv = prev.qkv
            fused = dataclasses.replace(op, q=qkv.cols_slice(0, H * D), k_new=qkv.cols_slice(H * D, KVH * D),
                                        v_new=qkv.cols_slice((H + KVH) * D, KVH * D), cos=prev.cos, sin=prev.sin)
            fused.layer = getattr(op, "layer", None)
            ops[-1] = fused
            continue
        ops.append(op)
    out.ops = ops
    return out


def split_kv(g: Graph, sm_count: int) -> Graph:
    """Decode attention with few (sequence, kv head) pairs leaves most SMs idle while each streams
    its whole context: split every pair's 128-key blocks over `splits` instructions (at most one
    wave in all), combined in-op by the last split to arrive (flash-decoding without a hop)."""
    out = copy.copy(g)
    out.buffers = list(g.buffers)
    ops = []
    for op in g.ops:
        if (isinstance(op, DecodeAttention) and op.batch * op.kv_heads > 2 * sm_count
                and op.kv_heads % C.COMPUTE_WARPS == 0):
            # many sequences: (sequence, kv head) pairs would take several waves of fixed-cost
            # instructions; group 4 kv heads per instruction, one per compute warp
            new = dataclasses.replace(op, head_parallel=True)
            new.layer = getattr(op, "layer", None)
            ops.append(new)
            continue
        if isinstance(op, DecodeAttention) and op.splits == 1:
            pairs = op.batch * op.kv_heads
            # at most one wave: each split pays a fixed ~7 us (prologue, partial hand-off, fences), so a
            # second wave costs more than the shorter chunks save (B16 Llama-1B: 2 splits were 17% slower)
            s = min(op.max_seq // 128, max(1, sm_count // pairs))
            if s > 1:
                G, D = op.heads // op.kv_heads, op.head_dim
                cols = -(-(G * D + 2 * G) // 4) * 4       # per split: O (G x D), max (G), sum (G)
                part = out.workspace((pairs * s, cols), torch.float32)
                count = out.workspace((pairs, 1), torch.int32)
                new = dataclasses.replace(op, splits=s, part=part, count=count)
                new.layer = getattr(op, "layer", None)
                ops.append(new)
                continue
        ops.append(op)
    out.ops = ops
    return out


def split_k(g: Graph, sm_count: int) -> Graph:
    """Rewrite each Linear with too few output tiles to fill the GPU (decode-like M) into K-split
    partials, on a copy of the graph. Splits divide the k-tiles evenly. The partials go
      - straight into the residual stream (bf16 reduce-add, no finalize) when the Linear adds a
        residual that nothing later reads -- later ops then use the residual buffer as its output;
      - otherwise into a zeroed fp32 workspace + SplitFinalize. A finalize that only feeds a
        RopeAppend is dropped: the RopeAppend reads (and re-zeroes) the workspace itself."""
    ops = []
    out = copy.copy(g)
    out.buffers = list(g.buffers)
    alias: dict[int, View] = {}       # output buffer id -> the residual view now holding it

    def move(v):
        r = alias.get(v.buf.id) if v is not None else None
        return v if r is None else View(r.buf, r.row0 + v.row0, r.col0 + v.col0, v.rows, v.cols)

    src = list(g.ops)
    for i, op0 in enumerate(src):
        op = dataclasses.replace(op0, **{f.name: move(getattr(op0, f.name)) for f in dataclasses.fields(op0)
                                         if isinstance(getattr(op0, f.name), View)})
        op.layer = getattr(op0, "layer", None)
        if isinstance(op, Linear) and op.act != "swiglu":   # swiglu: wide N, fills the GPU anyway
            M, K = op.x.shape
            full = -(-M // C.GEMM_BM) * -(-op.w.rows // C.GEMM_BN)
            tiles = -(-M // C.GEMM_BM) * -(-op.w.rows // C.SPLIT_BN)   # partial tiles are 128x64
            kt = K // C.GEMM_BK
            rows = min(M, C.GEMM_BM)
            # waves x (fixed per-instruction cost + operand bytes at one SM's bandwidth); more, smaller
            # instructions only win until the fixed cost (deps, rstd, epilogue, reduce) dominates
            cost = lambda n, bn, k: -(-n // sm_count) * (INST_US + (bn + rows) * k * 2 / (SM_GBPS * 1e3))
            s, best = 1, cost(full, C.GEMM_BN, K)
            for d in range(2, 9) if full < sm_count else ():   # a full wave of tiles needs no split
                if kt % d == 0 and kt // d >= MIN_SPLIT_KTILES and cost(tiles * d, C.SPLIT_BN, K // d) < best:
                    s, best = d, cost(tiles * d, C.SPLIT_BN, K // d)
            if s > 1:
                r = op.residual
                inplace = (r is not None and op.bias is None and op.act is None
                           and r.buf.kind in ("temp", "state") and op.out.buf.kind == "temp"
                           and (op.out.row0, op.out.col0, op.out.shape) == (0, 0, (op.out.buf.rows, op.out.buf.cols))
                           and not _reads_after(src[i + 1:], r.buf))
                if inplace:
                    alias[op.out.buf.id] = r
                    o = LinearSplit(op.x, op.w, r, s, op.rms_eps)
                    o.layer = op.layer
                    ops.append(o)
                    continue
                ws = out.workspace(op.out.shape, torch.float32)
                pair = [LinearSplit(op.x, op.w, ws, s, op.rms_eps), SplitFinalize(ws, op.out, op.bias, op.act,
                                                                                  op.residual)]
                for o in pair:
                    o.layer = op.layer
                ops += pair
                continue
        if (isinstance(op, RopeAppend) and len(ops) >= 1 and isinstance(ops[-1], SplitFinalize)
                and ops[-1].bias is None and ops[-1].act is None and ops[-1].residual is None
                and _same(op.qkv, ops[-1].out) and not _reads_after(src[i + 1:], op.qkv.buf, g.outputs)):
            ws = ops.pop().ws
            op = dataclasses.replace(op, qkv=ws)
            op.layer = getattr(op0, "layer", None)
        ops.append(op)
    out.ops = ops
    out.outputs = [move(v) for v in g.outputs]
    return out


def _node_for(prog, ops):
    from .ops.eltwise import EltwiseNode
    from .ops.gemm import GemmNode
    from .ops.norm import NormNode
    from .ops.splitk import FinalizeNode

    op = ops[0]
    if isinstance(op, (Linear, LinearSplit)):
        return GemmNode(prog, op, ops)
    if isinstance(op, SplitFinalize):
        return FinalizeNode(prog, op, ops)
    if isinstance(op, (Unary, Binary, SwiGLU, Modulate, Rope)):
        return EltwiseNode(prog, op, ops)
    if isinstance(op, Norm):
        return NormNode(prog, op, ops)
    if isinstance(op, Attention):
        from .ops.attention import AttentionNode
        return AttentionNode(prog, op, ops)
    if isinstance(op, (RopeAppend, DecodeAttention, Increment)):
        from .ops import decode
        return {RopeAppend: decode.RopeAppendNode, DecodeAttention: decode.DecodeAttentionNode,
                Increment: decode.IncrementNode}[type(op)](prog, op, ops)
    raise NotImplementedError(type(op).__name__)


class Program:
    def __init__(self, g: Graph, *, sm_count: int | None = None, verbose: bool = False, debug: bool = False,
                 trace: bool = False):
        # watchdog: report waits/spins that look stuck (printf), else zero cost. MEGACUTE_DEBUG=1
        # turns it on for every Program (e.g. to run a hanging test under it).
        self.debug = debug or os.environ.get("MEGACUTE_DEBUG") == "1"
        self.trace = trace  # per-SM, per-role globaltimer records of every instruction (see trace.py)
        self.sm_count = sm_count or torch.cuda.get_device_properties(0).multi_processor_count
        g = split_kv(fuse_rope_attention(split_k(fold_norms(g), self.sm_count)), self.sm_count)
        if os.environ.get("MEGACUTE_NO_MERGE") == "1":    # A/B switch for debugging layer merging
            self.g, groups = g, [[k] for k in range(len(g.ops))]
        else:
            self.g, groups = merge_layers(g)
        self.sched = schedule(self.g)
        self.views: list[tuple[View, str, dict]] = []
        self._view_key: dict[tuple, int] = {}
        # one kernel node per group of identical per-layer ops; op k runs as (node, layer)
        self.node_of = [None] * len(self.g.ops)
        for n, ks in enumerate(groups):
            for layer, k in enumerate(ks):
                self.node_of[k] = (n, layer)
        self.nodes = [_node_for(self, [self.g.ops[k] for k in ks]) for ks in groups]
        self.verbose = verbose
        self.epoch = 0
        self._compiled = None
        self._bufs: dict[int, torch.Tensor] = {}
        self._view_cache: dict[tuple, tuple] = {}
        self._derived_key: dict[int, tuple] = {}

    # ---- view registry (called by nodes at construction)
    def view_index(self, v: View, mode: str = "2d", **kw) -> int:
        if (v.col0 * torch.empty((), dtype=v.buf.dtype).element_size()) % 16:
            raise ValueError(f"{v}: column offset is not 16-byte aligned")
        key = (v.buf.id, v.row0, v.col0, v.rows, v.cols, mode, tuple(sorted(kw.items())))
        if key not in self._view_key:
            self._view_key[key] = len(self.views)
            self.views.append((v, mode, kw))
        return self._view_key[key]

    @staticmethod
    def _geometry(v: View, mode: str, kw: dict):
        rs = v.buf.cols
        off = v.row0 * rs + v.col0
        if mode == "2d":
            return (v.rows, v.cols), (rs, 1), off
        S = kw["seq"]
        B = v.rows // S
        if mode == "sbc":   # (seq, cols, batch): attention Q/K/O, K-major operands
            return (S, v.cols, B), (rs, 1, S * rs), off
        if mode == "csb":   # (cols, seq, batch): attention V as an MN-major operand
            return (v.cols, S, B), (1, rs, S * rs), off
        raise ValueError(mode)

    # ---- compile
    def compile(self):
        import cutlass
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32, Int32

        from .kernel import Megakernel

        dt = {torch.bfloat16: BFloat16, torch.float32: Float32, torch.int32: Int32}
        fakes = []
        for v, mode, kw in self.views:
            shape, stride, _ = self._geometry(v, mode, kw)
            # A static extent of 1 gets simplified away (stride 0), which yields an invalid TMA
            # descriptor -> "illegal instruction" (e.g. a (1, E) conditioning row). Keep such
            # extents symbolic; the real value is bound from the torch view at call time.
            shape = tuple(cute.sym_int() if d == 1 else d for d in shape)
            fakes.append(cute.runtime.make_fake_tensor(dt[v.buf.dtype], shape, stride, assumed_align=16))
        insts, waits, arrives = self.sched.encode(self.node_of)
        self._insts = insts.cuda()
        self._waits = waits.cuda()
        self._arrives = arrives.cuda()
        self._counters = torch.zeros(max(1, self.sched.num_counters), dtype=torch.int32, device="cuda")
        n = len(self.sched.insts)
        self.trace_cap = min(n, max(64, 8 * -(-n // self.sm_count))) if self.trace else 0
        self._trace = torch.zeros((self.sm_count, 3, max(1, self.trace_cap), 4) if self.trace else (1, 1, 1, 4),
                                  dtype=torch.int64, device="cuda")
        i32 = lambda t: cute.runtime.make_fake_compact_tensor(Int32, tuple(t.shape),
                                                              stride_order=tuple(range(t.dim() - 1, -1, -1)),
                                                              assumed_align=4)
        mk = Megakernel(self.nodes, len(self.sched.insts), self.sm_count, debug=self.debug,
                        trace_cap=self.trace_cap)
        t0 = time.time()
        self._compiled = cute.compile(
            mk, tuple(fakes), i32(self._insts), i32(self._waits), i32(self._arrives), i32(self._counters),
            cute.runtime.make_fake_compact_tensor(cutlass.Int64, tuple(self._trace.shape), stride_order=(3, 2, 1, 0),
                                                  assumed_align=8),
            Int32(1), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi")
        self.compile_seconds = time.time() - t0
        max_target = max((t for i in self.sched.insts for _, t in i.waits), default=1)
        self._max_epoch = (2**31 - 1) // max(1, max_target) - 1
        if self.verbose:
            print(f"[megacute] compiled {len(self.nodes)} nodes, {len(self.sched.insts)} insts, "
                  f"{self.sched.num_counters} counters in {self.compile_seconds:.1f}s")
        return self

    # ---- buffers
    def buffer(self, buf) -> torch.Tensor:
        return self._bufs[buf.id]

    def _bind(self, bindings: dict):
        for b in self.g.inputs + self.g.params + self.g.states:
            if b.name not in bindings:
                raise KeyError(f"missing tensor for {b.kind} '{b.name}'")
            t = bindings[b.name]
            if tuple(t.shape) != (b.rows, b.cols) or t.dtype != b.dtype or not t.is_contiguous() or not t.is_cuda:
                raise ValueError(f"'{b.name}': expected contiguous cuda {b.dtype} {(b.rows, b.cols)}, "
                                 f"got {t.dtype} {tuple(t.shape)}")
            self._bufs[b.id] = t
        for b in self.g.buffers:
            if b.kind in ("temp", "output") and b.id not in self._bufs:
                self._bufs[b.id] = torch.empty(b.rows, b.cols, dtype=b.dtype, device="cuda")
            if b.kind == "workspace" and b.id not in self._bufs:   # split-K: finalize re-zeroes it
                self._bufs[b.id] = torch.zeros(b.rows, b.cols, dtype=b.dtype, device="cuda")
        for b in self.g.buffers:
            if b.kind == "derived":     # norm-folded weights: recomputed only when a source tensor changes
                srcs = [(self._bufs[s.id], self._bufs[gv.buf.id]) for s, _, _, gv in b.derive]
                key = tuple((w.data_ptr(), gn.data_ptr()) for w, gn in srcs)
                if self._derived_key.get(b.id) != key:
                    d = torch.empty(b.rows, b.cols, dtype=b.dtype, device="cuda")
                    for (s, r0, n, gv), (w, gn) in zip(b.derive, srcs):
                        gain = gn[gv.row0:gv.row0 + 1, gv.col0:gv.col0 + gv.cols].float()
                        d[r0:r0 + n] = (w[r0:r0 + n].float() * gain).to(b.dtype)
                    self._bufs[b.id] = d
                    self._derived_key[b.id] = key

    def _view_tensors(self):
        key = tuple(self._bufs[b.id].data_ptr() for b in self.g.inputs + self.g.params + self.g.states)
        hit = self._view_cache.get(key)
        if hit is None:
            out = []
            for v, mode, kw in self.views:
                base = self._bufs[v.buf.id]
                shape, stride, off = self._geometry(v, mode, kw)
                out.append(base.as_strided(shape, stride, off))
            hit = tuple(out)
            if len(self._view_cache) > 64:
                self._view_cache.clear()
            self._view_cache[key] = hit
        return hit

    # ---- run
    def __call__(self, **bindings):
        if self._compiled is None:
            self.compile()
        self._bind(bindings)
        views = self._view_tensors()
        if self.epoch >= self._max_epoch:
            self._counters.zero_()
            self.epoch = 0
        self.epoch += 1
        if self.trace:
            self._trace.zero_()
        self._compiled(views, self._insts, self._waits, self._arrives, self._counters, self._trace,
                       self.epoch)
        outs = [self._bufs[v.buf.id].as_strided(v.shape, (v.buf.cols, 1), v.row0 * v.buf.cols + v.col0)
                for v in self.g.outputs]
        return outs[0] if len(outs) == 1 else tuple(outs)


    def timeline(self):
        """Trace of the most recent launch (requires trace=True). See megacute.trace."""
        from .trace import Timeline
        if not self.trace:
            raise RuntimeError("compile with trace=True to record a timeline")
        torch.cuda.synchronize()
        return Timeline(self, self._trace.cpu())


def compile(g: Graph, **kw) -> Program:
    return Program(g, **kw).compile()
