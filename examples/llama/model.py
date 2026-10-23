"""Llama graph builders on megacute: prefill over a prompt (llama_stack) and a full decode step
(llama_decode). Weights are stacked over layers so megacute merges each op into one kernel node."""

from __future__ import annotations

import torch

from megacute import Graph


def llama_block(B: int, S: int, E: int, H: int, F: int, eps: float = 1e-6, kv_heads: int | None = None) -> Graph:
    """One pre-norm Llama decoder block (prefill): see llama_stack."""
    return llama_stack(B, S, E, H, kv_heads or H, F, 1, eps)


LLAMA_PARAMS = lambda E, H, KVH, F: [("attn_norm", (1, E)), ("wqkv", ((H + 2 * KVH) * (E // H), E)), ("wo", (E, E)),
                                     ("mlp_norm", (1, E)), ("wgu", (2 * F, E)), ("wd", (E, F))]


def llama_stack(B: int, S: int, E: int, H: int, KVH: int, F: int, n: int, eps: float = 1e-6) -> Graph:
    """n Llama decoder blocks over a prompt, one launch. x: (B*S, E); cos/sin: (S, D).
    Block: RMSNorm -> QKV (GQA: H query, KVH kv heads) -> RoPE(q, k) -> causal attention -> o-proj + res,
    RMSNorm -> gate|up with fused SwiGLU (wgu interleaved, see graph.swiglu_weight) -> down + res.
    Parameters are stacked over layers (wqkv is (n*(H+2KVH)*D, E), ...) so each op merges into one node."""
    D = E // H
    g = Graph()
    x = g.input("x", (B * S, E))
    cos = g.input("cos", (S, D))
    sin = g.input("sin", (S, D))
    ps = {name: g.param(name, (n * r, c)) for name, (r, c) in LLAMA_PARAMS(E, H, KVH, F)}
    for i in range(n):
        g.layer = i
        p = {name: ps[name].rows_slice(i * r, r) for name, (r, _) in LLAMA_PARAMS(E, H, KVH, F)}
        qkv = g.linear(g.rms_norm(x, p["attn_norm"], eps), p["wqkv"])
        q = g.rope(qkv.cols_slice(0, E), cos, sin, heads=H, head_dim=D, seq=S)
        k = g.rope(qkv.cols_slice(E, KVH * D), cos, sin, heads=KVH, head_dim=D, seq=S)
        a = g.attention(q, k, qkv.cols_slice(E + KVH * D, KVH * D), batch=B, heads=H, head_dim=D, causal=True,
                        kv_heads=KVH)
        x = g.linear(a, p["wo"], residual=x)
        m = g.linear(g.rms_norm(x, p["mlp_norm"], eps), p["wgu"], act="swiglu")
        x = g.linear(m, p["wd"], residual=x)
    g.layer = None
    g.output(x)
    return g


def llama_decode(B: int, E: int, H: int, KVH: int, F: int, n: int, max_seq: int, vocab: int | None = None,
                 eps: float = 1e-6) -> Graph:
    """One decode step of an n-layer Llama for B sequences, one launch (MegaKittens llama decode).
    Inputs: cos/sin (max_seq, D). State (updated in place by the launch): x (B, E), the embedded
    tokens on entry and the residual stream after (decode GEMMs reduce-add into it); pos (1, 1)
    int32, the shared position of the new tokens (advanced by one); k_cache/v_cache
    (n*B*max_seq, KVH*D), layer i at rows [i*B*max_seq, ...), sequence b of it at [b*max_seq, ...),
    zero-initialised. Output: logits (B, vocab) after the final norm, or the hidden state (B, E)."""
    D = E // H
    g = Graph()
    x = g.state("x", (B, E))
    cos = g.input("cos", (max_seq, D))
    sin = g.input("sin", (max_seq, D))
    pos = g.state("pos", (1, 1), torch.int32)
    kc = g.state("k_cache", (n * B * max_seq, KVH * D))
    vc = g.state("v_cache", (n * B * max_seq, KVH * D))
    ps = {name: g.param(name, (n * r, c)) for name, (r, c) in LLAMA_PARAMS(E, H, KVH, F)}
    R = B * max_seq
    for i in range(n):
        g.layer = i
        p = {name: ps[name].rows_slice(i * r, r) for name, (r, _) in LLAMA_PARAMS(E, H, KVH, F)}
        kci, vci = kc.rows_slice(i * R, R), vc.rows_slice(i * R, R)
        qkv = g.linear(g.rms_norm(x, p["attn_norm"], eps), p["wqkv"])
        q = g.rope_append(qkv, cos, sin, pos, kci, vci, heads=H, kv_heads=KVH, head_dim=D, max_seq=max_seq)
        a = g.decode_attention(q, kci, vci, pos, heads=H, kv_heads=KVH, head_dim=D, max_seq=max_seq)
        x = g.linear(a, p["wo"], residual=x)
        m = g.linear(g.rms_norm(x, p["mlp_norm"], eps), p["wgu"], act="swiglu")
        x = g.linear(m, p["wd"], residual=x)
    g.layer = None
    if vocab:
        x = g.linear(g.rms_norm(x, g.param("final_norm", (1, E)), eps), g.param("lm_head", (vocab, E)))
    g.increment(pos)
    g.output(x)
    return g
