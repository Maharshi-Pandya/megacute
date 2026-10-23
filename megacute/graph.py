"""Graph IR: buffers, 2-D views, and the ops a megakernel can execute.

Every tensor the kernel touches is a row-major 2-D *buffer* (rows, cols). Higher-rank
activations are flattened to (batch * seq, features); attention heads are column ranges of a
buffer. An op reads and writes *views* -- rectangular windows of a buffer -- so a QKV
projection's output can feed attention as three column slices without any copy, and every
dependency the scheduler tracks is an intersection of rectangles on one buffer.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Literal, Optional

import torch

DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32"}

_ids = itertools.count()


@dataclasses.dataclass(eq=False)
class Buffer:
    name: str
    rows: int
    cols: int
    dtype: torch.dtype
    # workspace: zero-initialised; state: an input the kernel updates in place (KV cache, position);
    # derived: a param computed from other params at bind time (runtime.fold_norms: W * norm weight)
    kind: Literal["input", "param", "state", "temp", "output", "workspace", "derived"]
    id: int = dataclasses.field(default_factory=lambda: next(_ids))
    derive: Optional[list] = None   # derived: [(src_buf, row0, rows, gain_view)], rows scaled by gain

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise TypeError(f"{self.name}: unsupported dtype {self.dtype}")
        # TMA needs 16-byte aligned row strides (int32 buffers are never TMA'd)
        if self.dtype != torch.int32 and (self.cols * torch.empty((), dtype=self.dtype).element_size()) % 16:
            raise ValueError(f"{self.name}: row stride of {self.cols} elements is not 16-byte aligned")

    @property
    def full(self) -> "View":
        return View(self, 0, 0, self.rows, self.cols)

    def __repr__(self):
        return f"{self.name}[{self.rows}x{self.cols}]"


@dataclasses.dataclass(frozen=True, eq=False)
class View:
    buf: Buffer
    row0: int
    col0: int
    rows: int
    cols: int

    def __post_init__(self):
        if not (0 <= self.row0 and self.row0 + self.rows <= self.buf.rows
                and 0 <= self.col0 and self.col0 + self.cols <= self.buf.cols):
            raise ValueError(f"view {self} out of bounds of {self.buf}")

    def cols_slice(self, start: int, size: int) -> "View":
        return View(self.buf, self.row0, self.col0 + start, self.rows, size)

    def rows_slice(self, start: int, size: int) -> "View":
        return View(self.buf, self.row0 + start, self.col0, size, self.cols)

    @property
    def shape(self):
        return (self.rows, self.cols)

    def __repr__(self):
        return f"{self.buf.name}[{self.row0}:{self.row0 + self.rows}, {self.col0}:{self.col0 + self.cols}]"


Box = tuple  # (buffer_id, r0, r1, c0, c1), half-open


def box(v: View, r0: int = 0, r1: Optional[int] = None, c0: int = 0, c1: Optional[int] = None) -> Box:
    """A rectangle inside view `v`, in the view's coordinates, mapped to buffer coordinates."""
    r1 = v.rows if r1 is None else r1
    c1 = v.cols if c1 is None else c1
    return (v.buf.id, v.row0 + r0, v.row0 + min(r1, v.rows), v.col0 + c0, v.col0 + min(c1, v.cols))


# --------------------------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------------------------

UNARY = ("relu", "silu", "gelu_tanh", "gelu", "exp", "neg", "sigmoid", "square", "identity")
BINARY = ("add", "sub", "mul", "max", "min")
ACT = (None, "relu", "silu", "gelu_tanh", "gelu", "swiglu")


@dataclasses.dataclass(eq=False)
class Op:
    """Base class. Subclasses define `reads`/`writes` (views) and op-specific params."""

    accumulate = False   # writes are commutative adds: no write-after-write order among them
    independent = False  # its instructions never order against each other (they coordinate themselves)

    @property
    def reads(self) -> list[View]:
        raise NotImplementedError

    @property
    def writes(self) -> list[View]:
        raise NotImplementedError

    @property
    def kind(self) -> str:
        return type(self).__name__


@dataclasses.dataclass(eq=False)
class Linear(Op):
    """out = act(x @ w^T + bias) + residual.   x:(M,K) w:(N,K) bias:(1,N).
    act="swiglu": w rows interleave gate and up in 64-row blocks (gate 0:64, up 0:64, gate 64:128,
    ...), out is (M, N/2) = silu(gate) * up -- fused in the GEMM epilogue (see swiglu_weight).
    rms_eps: x is the raw input of an RMSNorm whose gain is folded into w; each output row is
    scaled by its rstd = rsqrt(mean(x^2) + eps) (runtime.fold_norms)."""
    x: View
    w: View
    out: View
    bias: Optional[View] = None
    act: Optional[str] = None
    residual: Optional[View] = None
    rms_eps: Optional[float] = None

    def __post_init__(self):
        M, K = self.x.shape
        N, K2 = self.w.shape
        assert K == K2, f"linear: K mismatch {K} vs {K2}"
        assert self.act in ACT, self.act
        if self.act == "swiglu":
            assert N % 128 == 0 and self.bias is None and self.residual is None, "swiglu linear: N % 128, no bias/res"
            N //= 2
        assert self.out.shape == (M, N), f"linear: out {self.out.shape} != {(M, N)}"
        if self.bias is not None:
            assert self.bias.shape == (1, N), f"linear: bias must be (1, {N})"
        if self.residual is not None:
            assert self.residual.shape == (M, N)

    @property
    def reads(self):
        return [v for v in (self.x, self.w, self.bias, self.residual) if v is not None]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class LinearSplit(Op):
    """Split-K partials of a Linear: ws += x[:, Kj] @ w[:, Kj]^T for each of `splits` K ranges, as
    TMA reduce-adds (made by runtime.split_k). ws is a zero-initialised fp32 workspace, or -- for a
    residual linear -- the bf16 residual stream itself, updated in place. rms_eps as in Linear."""
    x: View
    w: View
    ws: View
    splits: int
    rms_eps: Optional[float] = None
    accumulate = True

    @property
    def reads(self):
        return [self.x, self.w]

    @property
    def writes(self):
        return [self.ws]


@dataclasses.dataclass(eq=False)
class SplitFinalize(Op):
    """out = act(ws + bias) + residual, then ws = 0 again for the next launch."""
    ws: View
    out: View
    bias: Optional[View] = None
    act: Optional[str] = None
    residual: Optional[View] = None

    @property
    def reads(self):
        return [v for v in (self.ws, self.bias, self.residual) if v is not None]

    @property
    def writes(self):
        return [self.out, self.ws]


@dataclasses.dataclass(eq=False)
class Unary(Op):
    x: View
    out: View
    fn: str

    def __post_init__(self):
        assert self.fn in UNARY, self.fn
        assert self.x.shape == self.out.shape

    @property
    def reads(self):
        return [self.x]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class Binary(Op):
    """out = a <fn> b.  b may be a row vector (1, cols) broadcast over rows, or a per-group row
    (rows/group, cols) broadcast over each run of `group` rows (e.g. per-batch conditioning)."""
    a: View
    b: View
    out: View
    fn: str
    b_group: int = 1  # rows of `a` sharing one row of `b`; a.rows if b is a single row

    def __post_init__(self):
        assert self.fn in BINARY, self.fn
        assert self.a.shape == self.out.shape
        assert self.b.cols == self.a.cols
        assert self.a.rows == self.b.rows * self.b_group, "binary: b rows * b_group must equal a rows"

    @property
    def reads(self):
        return [self.a, self.b]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class SwiGLU(Op):
    """out = act(gate) * up.  (LLaMA MLP: act = silu; GEGLU: gelu_tanh)."""
    gate: View
    up: View
    out: View
    act: str = "silu"

    def __post_init__(self):
        assert self.gate.shape == self.up.shape == self.out.shape

    @property
    def reads(self):
        return [self.gate, self.up]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class Modulate(Op):
    """adaLN: out = x * (1 + scale[g]) + shift[g], g = row // group (DiT conditioning)."""
    x: View
    scale: View
    shift: View
    out: View
    group: int

    def __post_init__(self):
        assert self.x.shape == self.out.shape
        assert self.scale.shape == self.shift.shape == (self.x.rows // self.group, self.x.cols)

    @property
    def reads(self):
        return [self.x, self.scale, self.shift]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class Norm(Op):
    """RMSNorm (bias None, rms=True) or LayerNorm over the last dim, optional affine weight/bias."""
    x: View
    out: View
    weight: Optional[View]
    bias: Optional[View]
    eps: float
    rms: bool

    def __post_init__(self):
        assert self.x.shape == self.out.shape
        for p in (self.weight, self.bias):
            if p is not None:
                assert p.shape == (1, self.x.cols)

    @property
    def reads(self):
        return [v for v in (self.x, self.weight, self.bias) if v is not None]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class Rope(Op):
    """Rotate-half RoPE on `heads` heads of width `head_dim` stored side by side in x.
    cos/sin: (seq, head_dim). Row r of x is token r % seq."""
    x: View
    cos: View
    sin: View
    out: View
    heads: int
    head_dim: int
    seq: int

    def __post_init__(self):
        assert self.x.shape == self.out.shape
        assert self.x.cols == self.heads * self.head_dim
        assert self.cos.shape == self.sin.shape == (self.seq, self.head_dim)
        assert self.x.rows % self.seq == 0

    @property
    def reads(self):
        return [self.x, self.cos, self.sin]

    @property
    def writes(self):
        return [self.out]


@dataclasses.dataclass(eq=False)
class Attention(Op):
    """softmax(q k^T * scale [+ causal mask]) v, per (batch, head).
    q/out: (batch*seq, heads*head_dim) views; head h is columns [h*D, (h+1)*D).
    k/v: (batch*seq_k, kv_heads*head_dim); query head h reads kv head h // (heads // kv_heads) (GQA)."""
    q: View
    k: View
    v: View
    out: View
    batch: int
    seq_q: int
    seq_k: int
    heads: int
    head_dim: int
    causal: bool
    scale: float
    kv_heads: int

    def __post_init__(self):
        D, H = self.head_dim, self.heads
        assert H % self.kv_heads == 0, "attention: heads must be a multiple of kv_heads"
        assert self.q.shape == (self.batch * self.seq_q, H * D)
        assert self.k.shape == self.v.shape == (self.batch * self.seq_k, self.kv_heads * D)
        assert self.out.shape == self.q.shape
        if self.causal:
            assert self.seq_q == self.seq_k, "causal attention requires seq_q == seq_k"

    @property
    def reads(self):
        return [self.q, self.k, self.v]

    @property
    def writes(self):
        return [self.out]


def swiglu_weight(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """(F, K) gate and up projections -> the (2F, K) interleaved weight of linear(act="swiglu")."""
    F_, K = gate.shape
    return torch.stack([gate.view(F_ // 64, 64, K), up.view(F_ // 64, 64, K)], 1).reshape(2 * F_, K)


@dataclasses.dataclass(eq=False)
class RopeAppend(Op):
    """Decode step, one new token per sequence (row b of qkv), all at position pos[0, 0]:
    q_out[b] = rope(q), k_cache[b*max_seq + pos] = rope(k), v_cache[b*max_seq + pos] = v.
    qkv: (B, (heads + 2*kv_heads)*D) as [q | k | v]; cos/sin: (max_seq, D) rotate-half tables;
    caches: (B*max_seq, kv_heads*D). An fp32 qkv is a split-K workspace: it is read, then zeroed
    for the next launch (runtime.split_k drops the QKV finalize)."""
    qkv: View
    cos: View
    sin: View
    pos: View
    q_out: View
    k_cache: View
    v_cache: View
    heads: int
    kv_heads: int
    head_dim: int
    max_seq: int

    def __post_init__(self):
        B, D = self.qkv.rows, self.head_dim
        assert self.qkv.cols == (self.heads + 2 * self.kv_heads) * D
        assert self.q_out.shape == (B, self.heads * D)
        assert self.k_cache.shape == self.v_cache.shape == (B * self.max_seq, self.kv_heads * D)
        assert self.cos.shape == self.sin.shape == (self.max_seq, D) and D % 16 == 0
        assert self.pos.shape == (1, 1) and self.pos.buf.dtype == torch.int32

    @property
    def reads(self):
        return [self.qkv, self.cos, self.sin, self.pos]

    @property
    def writes(self):
        ws = [self.qkv] if self.qkv.buf.dtype == torch.float32 else []
        return [self.q_out, self.k_cache, self.v_cache] + ws


@dataclasses.dataclass(eq=False)
class DecodeAttention(Op):
    """One query token per sequence against its KV cache rows [0, pos], GQA.
    q/out: (B, heads*D); caches: (B*max_seq, kv_heads*D), sequence b at rows [b*max_seq, ...).
    Cache rows past pos must be finite (e.g. zero-initialised): masked P is 0, and 0 * NaN is NaN.
    With k_new/v_new/cos/sin (runtime.fuse_rope_attention folds a RopeAppend in): q, k_new and
    v_new are the un-rotated columns of the QKV row; the op rotates q and k, appends k/v at pos and
    attends over the cache rows [0, pos) plus the new token. An fp32 QKV row is a split-K
    workspace: the op zeroes what it read."""
    q: View
    k_cache: View
    v_cache: View
    pos: View
    out: View
    batch: int
    heads: int
    kv_heads: int
    head_dim: int
    max_seq: int
    scale: float
    k_new: Optional[View] = None
    v_new: Optional[View] = None
    cos: Optional[View] = None
    sin: Optional[View] = None
    # split-KV (runtime.split_kv): `splits` instructions per (sequence, kv head), each over a chunk of
    # 128-key blocks; partials (fp32) and arrival counters (int32) live in zeroed workspaces, and the
    # last split to arrive combines them -- so the splits are independent of each other
    splits: int = 1
    part: Optional[View] = None
    count: Optional[View] = None
    # head-parallel (runtime.split_kv, many sequences): one instruction per (sequence, group of 4 kv
    # heads), compute warp w owning kv head 4*group + w over all keys -- fewer, fatter instructions
    head_parallel: bool = False

    @property
    def independent(self):
        return self.splits > 1

    def __post_init__(self):
        D = self.head_dim
        assert self.heads % self.kv_heads == 0 and self.heads // self.kv_heads <= 8, "decode attention: GQA group <= 8"
        assert D in (64, 128) and self.max_seq % 128 == 0
        assert self.q.shape == self.out.shape == (self.batch, self.heads * D)
        assert self.k_cache.shape == self.v_cache.shape == (self.batch * self.max_seq, self.kv_heads * D)
        assert self.pos.shape == (1, 1) and self.pos.buf.dtype == torch.int32

    @property
    def reads(self):
        return [v for v in (self.q, self.k_cache, self.v_cache, self.pos, self.k_new, self.v_new, self.cos, self.sin,
                            self.part, self.count) if v is not None]

    @property
    def writes(self):
        extra = [v for v in (self.part, self.count) if v is not None]
        if self.k_new is None:
            return [self.out] + extra
        ws = [self.q, self.k_new, self.v_new] if self.q.buf.dtype == torch.float32 else []
        return [self.out, self.k_cache, self.v_cache] + ws + extra


@dataclasses.dataclass(eq=False)
class Increment(Op):
    """pos += 1 on device, after every reader of pos in this launch (ends a decode step)."""
    pos: View

    @property
    def reads(self):
        return []

    @property
    def writes(self):
        return [self.pos]


# --------------------------------------------------------------------------------------------
# graph builder
# --------------------------------------------------------------------------------------------

class Graph:
    """Builds a straight-line program of ops over buffers.

        g = Graph()
        x = g.input("x", (4096, 2048))
        w = g.param("w", (8192, 2048))
        h = g.linear(x, w, act="gelu_tanh")
        g.output(h)
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        self.dtype = dtype
        self.layer = None   # set by builders of repeated blocks; tags ops for megacute.layers
        self.buffers: list[Buffer] = []
        self.ops: list[Op] = []
        self.inputs: list[Buffer] = []
        self.params: list[Buffer] = []
        self.states: list[Buffer] = []
        self.outputs: list[View] = []

    # -- buffers
    def _buf(self, name, shape, kind, dtype=None) -> View:
        rows, cols = shape
        b = Buffer(name, rows, cols, dtype or self.dtype, kind)
        self.buffers.append(b)
        return b.full

    def input(self, name, shape, dtype=None) -> View:
        v = self._buf(name, shape, "input", dtype)
        self.inputs.append(v.buf)
        return v

    def param(self, name, shape, dtype=None) -> View:
        v = self._buf(name, shape, "param", dtype)
        self.params.append(v.buf)
        return v

    def state(self, name, shape, dtype=None) -> View:
        """A caller-owned tensor the kernel reads and updates in place (KV cache, decode position)."""
        v = self._buf(name, shape, "state", dtype)
        self.states.append(v.buf)
        return v

    def temp(self, shape, name=None, dtype=None) -> View:
        return self._buf(name or f"t{len(self.buffers)}", shape, "temp", dtype)

    def workspace(self, shape, dtype) -> View:
        return self._buf(f"ws{len(self.buffers)}", shape, "workspace", dtype)

    def output(self, *views: View):
        for v in views:
            if v.buf.kind == "temp":
                v.buf.kind = "output"
            self.outputs.append(v)

    def _add(self, op: Op):
        for w in op.writes:
            if w.buf.kind in ("input", "param"):
                raise ValueError(f"{op.kind} writes read-only buffer {w.buf}")
        op.layer = self.layer
        self.ops.append(op)

    # -- ops
    def linear(self, x, w, bias=None, act=None, residual=None, out=None) -> View:
        out = out or self.temp((x.rows, w.rows // 2 if act == "swiglu" else w.rows))
        self._add(Linear(x, w, out, bias, act, residual))
        return out

    def unary(self, x, fn, out=None) -> View:
        out = out or self.temp(x.shape)
        self._add(Unary(x, out, fn))
        return out

    def binary(self, a, b, fn, out=None, b_group=None) -> View:
        out = out or self.temp(a.shape)
        self._add(Binary(a, b, out, fn, b_group or a.rows // b.rows))
        return out

    def add(self, a, b, out=None):
        return self.binary(a, b, "add", out)

    def mul(self, a, b, out=None):
        return self.binary(a, b, "mul", out)

    def swiglu(self, gate, up, act="silu", out=None) -> View:
        out = out or self.temp(gate.shape)
        self._add(SwiGLU(gate, up, out, act))
        return out

    def modulate(self, x, scale, shift, group, out=None) -> View:
        out = out or self.temp(x.shape)
        self._add(Modulate(x, scale, shift, out, group))
        return out

    def rms_norm(self, x, weight=None, eps=1e-6, out=None) -> View:
        out = out or self.temp(x.shape)
        self._add(Norm(x, out, weight, None, eps, rms=True))
        return out

    def layer_norm(self, x, weight=None, bias=None, eps=1e-5, out=None) -> View:
        out = out or self.temp(x.shape)
        self._add(Norm(x, out, weight, bias, eps, rms=False))
        return out

    def rope(self, x, cos, sin, heads, head_dim, seq, out=None) -> View:
        out = out or self.temp(x.shape)
        self._add(Rope(x, cos, sin, out, heads, head_dim, seq))
        return out

    def attention(self, q, k, v, *, batch, heads, head_dim, causal=False, scale=None, kv_heads=None,
                  out=None) -> View:
        out = out or self.temp(q.shape)
        seq_q, seq_k = q.rows // batch, k.rows // batch
        self._add(Attention(q, k, v, out, batch, seq_q, seq_k, heads, head_dim, causal,
                            scale if scale is not None else head_dim ** -0.5, kv_heads or heads))
        return out

    def rope_append(self, qkv, cos, sin, pos, k_cache, v_cache, *, heads, kv_heads, head_dim, max_seq,
                    out=None) -> View:
        out = out or self.temp((qkv.rows, heads * head_dim))
        self._add(RopeAppend(qkv, cos, sin, pos, out, k_cache, v_cache, heads, kv_heads, head_dim, max_seq))
        return out

    def decode_attention(self, q, k_cache, v_cache, pos, *, heads, kv_heads, head_dim, max_seq, scale=None,
                         out=None) -> View:
        out = out or self.temp(q.shape)
        self._add(DecodeAttention(q, k_cache, v_cache, pos, out, q.rows, heads, kv_heads, head_dim, max_seq,
                                  scale if scale is not None else head_dim ** -0.5))
        return out

    def increment(self, pos):
        self._add(Increment(pos))

    def __repr__(self):
        lines = [f"Graph({len(self.buffers)} buffers, {len(self.ops)} ops)"]
        for op in self.ops:
            lines.append(f"  {op.kind}: {op.reads} -> {op.writes}")
        return "\n".join(lines)
