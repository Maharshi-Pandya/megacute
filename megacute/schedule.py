"""Scheduler: graph -> instruction stream + dependency counters.

1. Tile every op into instructions. Each instruction carries op-specific integer args and
   declares the boxes (rectangles on 2-D buffers) it reads and writes.
2. Find hazards against all earlier instructions -- RAW, WAR and WAW -- by box intersection.
   Earlier means earlier in the global stream, which is also the order the device work queue
   hands instructions out, so every dependency points backwards and the queue cannot deadlock.
3. Group each instruction's dependencies per producer node into a set. Every distinct
   (node, set) becomes one gmem counter whose target is the set's size; all members arrive on
   it, every dependent waits on it. Dependents with identical needs share a counter
   (MegaKittens' scheme), so a GEMM row block waits on one counter, not 16 flags.
4. Encode: instructions as int32 rows, waits and arrivals in side tables (no fixed caps).

Counters are never reset: launch e waits for `target * e` (epoch scheme), so a steady-state
call is a single kernel launch with no memset.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Optional

import torch

from . import config as C
from .graph import (Attention, Binary, DecodeAttention, Graph, Increment, Linear, LinearSplit, Modulate, Norm, Op,
                    Rope, RopeAppend, SplitFinalize, SwiGLU, Unary, View, box)

ROW_BUCKET = 128


@dataclasses.dataclass
class Inst:
    node: int              # index into graph.ops
    args: tuple            # op-specific ints (<= 9)
    reads: list            # boxes
    writes: list           # boxes
    waits: list = dataclasses.field(default_factory=list)    # [(counter, target)]
    arrives: list = dataclasses.field(default_factory=list)  # [counter]


def _ceil(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------------------------

def _gemm_order(tm, tn, group=8):
    """Grouped raster: walk `group` row blocks at a time so consecutive tiles share B in L2."""
    order = []
    for g0 in range(0, tm, group):
        rows = range(g0, min(g0 + group, tm))
        for n in range(tn):
            for m in rows:
                order.append((m, n))
    return order


def tile_op(op: Op) -> list[tuple[tuple, list, list]]:
    T = C.TILE
    if isinstance(op, Linear):
        M, N = op.out.shape
        assert op.x.cols % C.GEMM_BK == 0, f"linear K={op.x.cols} must be a multiple of {C.GEMM_BK}"
        out = []
        on = C.GEMM_BN // 2 if op.act == "swiglu" else C.GEMM_BN    # output columns per tile
        for m, n in _gemm_order(_ceil(M, C.GEMM_BM), _ceil(N, on)):
            r0, c0, o0 = m * C.GEMM_BM, n * C.GEMM_BN, n * on
            reads = [box(op.x, r0, r0 + C.GEMM_BM), box(op.w, c0, c0 + C.GEMM_BN)]
            if op.bias is not None:
                reads.append(box(op.bias, 0, 1, c0, c0 + C.GEMM_BN))
            if op.residual is not None:
                reads.append(box(op.residual, r0, r0 + C.GEMM_BM, c0, c0 + C.GEMM_BN))
            out.append(((m, n), reads, [box(op.out, r0, r0 + C.GEMM_BM, o0, o0 + on)]))
        return out
    if isinstance(op, LinearSplit):
        M, N = op.ws.shape
        kc = op.x.cols // op.splits
        out = []
        for m, n in _gemm_order(_ceil(M, C.GEMM_BM), _ceil(N, C.SPLIT_BN)):
            r0, c0 = m * C.GEMM_BM, n * C.SPLIT_BN
            for ks in range(op.splits):   # adjacent: the splits of one tile run side by side
                reads = [box(op.x, r0, r0 + C.GEMM_BM, ks * kc, (ks + 1) * kc),
                         box(op.w, c0, c0 + C.SPLIT_BN, ks * kc, (ks + 1) * kc)]
                if op.rms_eps is not None:      # rstd needs the whole input rows
                    reads.append(box(op.x, r0, r0 + C.GEMM_BM))
                out.append(((m, n, ks), reads, [box(op.ws, r0, r0 + C.GEMM_BM, c0, c0 + C.SPLIT_BN)]))
        return out
    if isinstance(op, SplitFinalize):
        M, N = op.out.shape
        out = []
        for m in range(_ceil(M, T)):
            for n in range(_ceil(N, T)):
                r0, r1, c0, c1 = m * T, (m + 1) * T, n * T, (n + 1) * T
                reads = [box(op.ws, r0, r1, c0, c1)]
                if op.bias is not None:
                    reads.append(box(op.bias, 0, 1, c0, c1))
                if op.residual is not None:
                    reads.append(box(op.residual, r0, r1, c0, c1))
                out.append(((m, n), reads, [box(op.out, r0, r1, c0, c1), box(op.ws, r0, r1, c0, c1)]))
        return out
    if isinstance(op, (Unary, Binary, SwiGLU, Modulate, Rope)):
        R, Cc = op.out.shape
        TR = C.EW_ROWS
        if isinstance(op, Rope):
            assert T % op.head_dim == 0, "rope: head_dim must divide the 128-column tile"
        out = []
        for r in range(_ceil(R, TR)):
            for c in range(_ceil(Cc, T)):
                r0, r1, c0, c1 = r * TR, (r + 1) * TR, c * T, (c + 1) * T
                tb = lambda v: box(v, r0, r1, c0, c1)
                if isinstance(op, Unary):
                    reads = [tb(op.x)]
                elif isinstance(op, Binary):
                    g = op.b_group
                    reads = [tb(op.a), box(op.b, r0 // g, _ceil(r1, g), c0, c1)]
                elif isinstance(op, SwiGLU):
                    reads = [tb(op.gate), tb(op.up)]
                elif isinstance(op, Modulate):
                    g = op.group
                    reads = [tb(op.x), box(op.scale, r0 // g, _ceil(r1, g), c0, c1),
                             box(op.shift, r0 // g, _ceil(r1, g), c0, c1)]
                else:  # Rope: token positions of these rows, all head_dim columns of cos/sin
                    reads = [tb(op.x)]
                    if TR >= op.seq:
                        rs = [(0, op.seq)]
                    else:
                        s0 = r0 % op.seq
                        rs = [(s0, min(s0 + TR, op.seq))]
                    for a, b_ in rs:
                        reads += [box(op.cos, a, b_), box(op.sin, a, b_)]
                out.append(((r, c), reads, [tb(op.out)]))
        return out
    if isinstance(op, Norm):
        R = op.x.rows
        rows = norm_rows(op.x.rows)
        out = []
        for r in range(_ceil(R, rows)):
            r0, r1 = r * rows, (r + 1) * rows
            reads = [box(op.x, r0, r1)] + [box(p) for p in (op.weight, op.bias) if p is not None]
            out.append(((r,), reads, [box(op.out, r0, r1)]))
        return out
    if isinstance(op, Attention):
        D = op.head_dim
        out = []
        QP = C.ATTN_BQ                     # one instruction = one 128-row query tile
        qt = _ceil(op.seq_q, QP)
        # causal: longest-processing-time first, so the long diagonal tiles do not trail the run
        q_order = list(reversed(range(qt))) if op.causal else list(range(qt))
        for qi in q_order:
            for b in range(op.batch):
                for h in range(op.heads):
                    q0 = b * op.seq_q + qi * QP
                    q1 = min(q0 + QP, (b + 1) * op.seq_q)
                    k0 = b * op.seq_k
                    k1 = k0 + (min(op.seq_k, (qi + 1) * QP) if op.causal else op.seq_k)
                    c0, c1 = h * D, (h + 1) * D
                    hk = h // (op.heads // op.kv_heads)
                    reads = [box(op.q, q0, q1, c0, c1), box(op.k, k0, k1, hk * D, (hk + 1) * D),
                             box(op.v, k0, k1, hk * D, (hk + 1) * D)]
                    out.append(((b, h, qi), reads, [box(op.out, q0, q1, c0, c1)]))
        return out
    if isinstance(op, RopeAppend):
        out = []
        for r in range(_ceil(op.qkv.rows, C.COMPUTE_WARPS)):   # one sequence per compute warp
            r0, r1 = r * C.COMPUTE_WARPS, (r + 1) * C.COMPUTE_WARPS
            L = op.max_seq
            reads = [box(op.qkv, r0, r1), box(op.cos), box(op.sin), box(op.pos)]
            writes = [box(op.q_out, r0, r1), box(op.k_cache, r0 * L, r1 * L), box(op.v_cache, r0 * L, r1 * L)]
            out.append(((r,), reads, writes))
        return out
    if isinstance(op, DecodeAttention):
        D, G, L = op.head_dim, op.heads // op.kv_heads, op.max_seq
        out = []
        hpi = C.COMPUTE_WARPS if op.head_parallel else 1      # kv heads per instruction
        for b in range(op.batch):
            for h in range(op.kv_heads // hpi):
                h0, h1 = h * hpi, (h + 1) * hpi
                cache = [box(op.k_cache, b * L, (b + 1) * L, h0 * D, h1 * D),
                         box(op.v_cache, b * L, (b + 1) * L, h0 * D, h1 * D)]
                q = box(op.q, b, b + 1, h0 * G * D, h1 * G * D)
                reads = [q, box(op.pos)] + cache
                writes = [box(op.out, b, b + 1, h0 * G * D, h1 * G * D)]
                if op.k_new is not None:        # fused RoPE + append: also writes the cache row at pos
                    new = [box(op.k_new, b, b + 1, h0 * D, h1 * D), box(op.v_new, b, b + 1, h0 * D, h1 * D)]
                    reads += new + [box(op.cos), box(op.sin)]
                    writes += cache + ([q] + new if op.q.buf.dtype == torch.float32 else [])
                if op.splits > 1:
                    r = b * op.kv_heads + h
                    ws = [box(op.part, r * op.splits, (r + 1) * op.splits), box(op.count, r, r + 1)]
                    reads, writes = reads + ws, writes + ws
                for s in range(op.splits):
                    out.append(((b, h, s), reads, writes))
        return out
    if isinstance(op, Increment):
        return [((), [], [box(op.pos)])]
    raise NotImplementedError(type(op).__name__)


def norm_rows(rows: int) -> int:
    """Rows per norm instruction. A row is a latency-bound load-reduce-store chain, so rows are
    spread over the grid: one per compute warp, or -- when there are few rows (decode) -- one per
    instruction, split across the whole warpgroup."""
    return 1 if rows <= WIDE_NORM_ROWS else C.COMPUTE_WARPS


WIDE_NORM_ROWS = 256


# --------------------------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------------------------

def _overlap(a, b) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4]


class _BoxIndex:
    """Boxes per buffer, bucketed by row, for fast overlap queries. Instructions with the same box
    share one entry (e.g. all split-K partials of a layer reading the same rows), so a buffer that
    every layer touches (an in-place residual stream) is still cheap to query."""

    def __init__(self):
        self.b = defaultdict(dict)

    def add(self, bx, inst: int):
        for k in range(bx[1] // ROW_BUCKET, (bx[2] - 1) // ROW_BUCKET + 1):
            self.b[(bx[0], k)].setdefault(bx, []).append(inst)

    def query(self, bx) -> set:
        hit = set()
        for k in range(bx[1] // ROW_BUCKET, (bx[2] - 1) // ROW_BUCKET + 1):
            for other, insts in self.b.get((bx[0], k), {}).items():
                if _overlap(bx, other):
                    hit.update(insts)
        return hit


@dataclasses.dataclass
class Schedule:
    graph: Graph
    insts: list[Inst]
    num_counters: int
    node_ranges: list[tuple[int, int]]     # [start, end) of each node's instructions

    # ---- encoding
    def encode(self, node_of=None, device="cpu"):
        """-> (insts [N, IW] int32, waits [2*W] int32, arrives [A] int32). node_of[op] =
        (kernel node, layer) when ops were merged across layers (runtime.merge_layers)."""
        rows, waits, arrives = [], [], []
        for seq, ins in enumerate(self.insts):
            w = [0] * C.IW
            node, layer = node_of[ins.node] if node_of else (ins.node, 0)
            w[C.W_ICODE] = node + 1
            assert len(ins.args) <= C.LAYER_ARG
            for k, a in enumerate(ins.args):
                w[C.W_ARG0 + k] = a
            w[C.W_ARG0 + C.LAYER_ARG] = layer
            w[C.W_NWAIT] = len(ins.waits)
            w[C.W_WAIT_OFF] = len(waits) // 2
            for cnt, tgt in ins.waits:
                waits += [cnt, tgt]
            w[C.W_NARRIVE] = len(ins.arrives)
            w[C.W_ARRIVE_OFF] = len(arrives)
            arrives += ins.arrives
            w[C.W_SEQ] = seq
            rows.append(w)
        t = lambda x: torch.tensor(x or [0], dtype=torch.int32, device=device)
        return t(rows).reshape(-1, C.IW), t(waits), t(arrives)

    def summary(self) -> str:
        lines = [f"{len(self.insts)} instructions, {self.num_counters} counters, "
                 f"{sum(len(i.waits) for i in self.insts)} waits"]
        for k, (op, (a, b)) in enumerate(zip(self.graph.ops, self.node_ranges)):
            lines.append(f"  node {k} {op.kind:<10} insts [{a}, {b})")
        return "\n".join(lines)


def schedule(g: Graph) -> Schedule:
    insts: list[Inst] = []
    node_ranges = []
    for k, op in enumerate(g.ops):
        a = len(insts)
        for args, reads, writes in tile_op(op):
            insts.append(Inst(k, tuple(args), reads, writes))
        node_ranges.append((a, len(insts)))

    # accumulating writes (split-K partials) order against readers and plain writers, but not
    # against each other: atomic adds commute
    readers, writers, accums = _BoxIndex(), _BoxIndex(), _BoxIndex()
    counter_of: dict[tuple, int] = {}
    reach: list[int] = []       # per counter: bitset of the counters it transitively implies
    for i, ins in enumerate(insts):
        acc = g.ops[ins.node].accumulate
        deps = set()
        for bx in ins.reads:                       # RAW
            deps |= writers.query(bx) | accums.query(bx)
        for bx in ins.writes:                      # WAR + WAW
            deps |= readers.query(bx) | writers.query(bx)
            if not acc:
                deps |= accums.query(bx)
        deps.discard(i)
        if g.ops[ins.node].independent:
            deps = {j for j in deps if insts[j].node != ins.node}
        by_node = defaultdict(set)
        for j in deps:
            by_node[insts[j].node].add(j)
        waits = []
        for node, js in sorted(by_node.items()):
            key = (node, frozenset(js))
            if key not in counter_of:
                c = counter_of[key] = len(counter_of)
                # counters this one implies: its members already waited on them before starting
                r = 0
                for j in js:
                    for c2, _ in insts[j].waits:
                        r |= (1 << c2) | reach[c2]
                reach.append(r)
            waits.append((counter_of[key], len(js)))
        # transitive reduction: drop a wait that another wait of this instruction implies (a buffer
        # reused by every layer, like an in-place residual stream, otherwise collects one wait per
        # earlier layer)
        implied = 0
        for c, _ in waits:
            implied |= reach[c]
        ins.waits = [(c, t) for c, t in waits if not (implied >> c) & 1]
        for bx in ins.reads:
            readers.add(bx, i)
        for bx in ins.writes:
            (accums if acc else writers).add(bx, i)
    # members arrive only on counters someone still waits on
    members = {c: js for (_, js), c in counter_of.items()}
    used = sorted({c for ins in insts for c, _ in ins.waits})
    for c in used:
        for j in members[c]:
            insts[j].arrives.append(c)
    return Schedule(g, insts, len(counter_of), node_ranges)
