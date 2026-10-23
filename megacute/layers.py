"""Merge identical ops of repeated layers into one kernel node.

A 28-block DiT is 28 copies of the same ~15 ops. Compiled one node per op, the kernel carries
~400 node bodies (instruction-cache thrash, minutes of compile) and ~800 TMA descriptors (beyond
the kernel parameter limit). Merged, it is ~15 bodies: a node's tensors become one stacked
span per operand, and each instruction carries its layer index (LAYER_ARG); the op adds
layer * delta to its row (or batch) coordinate.

  1. relocate: per-layer temps / workspaces of the same shape that sit in the same operand slot
     of a group move into one stacked buffer, layer i at rows [i*R, (i+1)*R). Moving storage is
     always safe; dependencies are still computed on real boxes.
  2. validate: a group merges only if every operand is either the same view for all members, or
     the same buffer at a constant row stride delta that the op's tiling can address (a TMA-tiled
     operand needs delta % tile == 0, attention needs delta % seq == 0). Otherwise its members
     stay separate nodes.

Parameters merge when the model stacks them (one (n*R, C) param, per-layer row slices), as
the example builders (examples/llama, examples/dit) do. Builders tag ops with g.layer; an op's group key is its signature
plus its rank among same-kind ops of its layer, so a layer's two LayerNorms stay apart.
"""

from __future__ import annotations

import copy
import dataclasses
from collections import OrderedDict

from . import config as C
from .graph import (Attention, Binary, Buffer, DecodeAttention, Graph, Linear, LinearSplit, Modulate, Norm, Rope,
                    SplitFinalize, SwiGLU, Unary, View)


def _views(op):
    return [(f.name, getattr(op, f.name)) for f in dataclasses.fields(op) if isinstance(getattr(op, f.name), View)]


def _signature(op):
    key = [type(op).__name__]
    for f in dataclasses.fields(op):
        v = getattr(op, f.name)
        if isinstance(v, View):
            b = v.buf
            key.append((f.name, b.rows, b.cols, b.dtype, b.kind if b.kind in ("input", "param", "output") else "t",
                        v.col0, v.rows, v.cols))
        elif v is None or isinstance(v, (int, float, str, bool)):
            key.append((f.name, v))
    return tuple(key)


def _unit(op, field):
    """Row granularity at which `field` is addressed: TMA tile height, sequence length (attention
    batch coordinate), or 1 for direct row-indexed access."""
    if isinstance(op, Attention):
        return op.seq_q if field in ("q", "out") else op.seq_k
    if isinstance(op, (Linear, LinearSplit)):
        return C.GEMM_BM if field in ("x", "w", "out", "ws") else 1
    if isinstance(op, SplitFinalize):
        return C.TILE if field == "out" else 1
    if isinstance(op, DecodeAttention):
        return 128 if field in ("k_cache", "v_cache") else 1    # caches are TMA'd in 128-row tiles
    if isinstance(op, Norm):
        return 1
    if isinstance(op, Binary) and field == "b" and op.b.rows != op.a.rows:
        return 1
    if isinstance(op, (Modulate, Rope)) and field not in ("x", "out"):
        return 1
    if isinstance(op, (Unary, Binary, SwiGLU, Modulate, Rope)):
        return C.EW_ROWS
    return 1


def _delta(op_views, field, unit):
    """Constant row stride of a field across members, 0 if identical, None if not mergeable."""
    vs = [dict(v)[field] for v in op_views]
    v0 = vs[0]
    if any(v.buf is not v0.buf or v.col0 != v0.col0 or v.rows != v0.rows for v in vs):
        return None
    d = vs[1].row0 - v0.row0
    if any(v.row0 != v0.row0 + i * d for i, v in enumerate(vs)) or d < 0:
        return None
    if d and (d % unit or d < -(-v0.rows // unit) * unit):   # a tail tile must stay inside its layer's rows
        return None
    return d


def merge_layers(g: Graph):
    """-> (graph with relocated temps, groups: list of op-index lists, one kernel node each)."""
    groups = OrderedDict()
    rank = {}
    for k, op in enumerate(g.ops):
        layer = getattr(op, "layer", None)
        kind = (layer, type(op).__name__, tuple(v for f, v in vars(op).items()
                                                 if isinstance(v, (int, float, str, bool)) and f != "layer"))
        rank[kind] = rank.get(kind, -1) + 1
        groups.setdefault((_signature(op), rank[kind] if layer is not None else None), []).append(k)

    # 1. relocate per-layer temps into stacked buffers
    where: dict[int, tuple[Buffer, int]] = {}      # old buffer id -> (stack, row offset)
    out = copy.copy(g)
    out.buffers = list(g.buffers)
    # A stack has one slot per layer (layer L at rows [L*stride, ...)), so groups that share a buffer
    # family but cover different layers (layer 0 reads the graph input, the last layer feeds the
    # final norm) fill the same stack instead of claiming it with incompatible layouts.
    nlayers = 1 + max((op.layer for op in g.ops if getattr(op, "layer", None) is not None), default=0)
    used: dict[int, set] = {}                      # stack id -> occupied layer slots
    for ks in groups.values():
        if len(ks) < 2:
            continue
        for f, v0 in _views(g.ops[ks[0]]):
            bufs = [getattr(g.ops[k], f).buf for k in ks]
            lays = [g.ops[k].layer for k in ks]
            if (len({b.id for b in bufs}) != len(bufs) or any(b.kind not in ("temp", "workspace") for b in bufs)
                    or any((b.rows, b.cols, b.dtype) != (v0.buf.rows, v0.buf.cols, v0.buf.dtype) for b in bufs)):
                continue
            # layer stride padded to whole 128-row tiles: a tail tile store then lands in padding
            stride = -(-v0.buf.rows // C.TILE) * C.TILE
            stacks = {where[b.id][0].id: where[b.id][0] for b in bufs if b.id in where}
            if len(stacks) > 1 or any(where[b.id][1] != l * stride for b, l in zip(bufs, lays) if b.id in where):
                continue
            if stacks:
                stack = next(iter(stacks.values()))
                if any(l in used[stack.id] for b, l in zip(bufs, lays) if b.id not in where):
                    continue
            else:
                stack = Buffer(f"stack_{bufs[0].name}", nlayers * stride, v0.buf.cols, v0.buf.dtype, bufs[0].kind)
                out.buffers.append(stack)
                used[stack.id] = set()
            for b, l in zip(bufs, lays):
                if b.id not in where:
                    where[b.id] = (stack, l * stride)
                    used[stack.id].add(l)
    out.buffers = [b for b in out.buffers if b.id not in where]

    def move(v):
        if v.buf.id not in where:
            return v
        stack, r = where[v.buf.id]
        return View(stack, r + v.row0, v.col0, v.rows, v.cols)

    out.ops = []
    for op in g.ops:
        new = dataclasses.replace(op, **{f: move(v) for f, v in _views(op)})
        new.layer = getattr(op, "layer", None)
        out.ops.append(new)
    out.outputs = [move(v) for v in g.outputs]

    # 2. validate each group; members that cannot merge become their own nodes
    result = []
    for ks in groups.values():
        ops = [out.ops[k] for k in ks]
        ok = len(ks) > 1 and all(_delta([_views(o) for o in ops], f, _unit(ops[0], f)) is not None
                                 for f, _ in _views(ops[0]))
        result += [ks] if ok else [[k] for k in ks]
    return out, result
