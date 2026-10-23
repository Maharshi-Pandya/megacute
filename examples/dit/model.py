"""DiT (diffusion transformer) graph builders on megacute: adaLN-Zero conditioning, gated residuals,
GELU MLP. Weights are stacked over layers so megacute merges each op into one kernel node."""

from __future__ import annotations

from megacute import Graph


def dit_block(B: int, S: int, E: int, H: int, F: int, eps: float = 1e-6) -> Graph:
    """DiT block with adaLN-Zero conditioning. x: (B*S, E), c: (B, E).
    mod = Linear(SiLU(c)) -> [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]."""
    return dit_stack(B, S, E, H, F, 1, eps)


DIT_PARAMS = lambda E, F: [("w_ada", (6 * E, E)), ("b_ada", (1, 6 * E)), ("wqkv", (3 * E, E)), ("bqkv", (1, 3 * E)),
                           ("wo", (E, E)), ("bo", (1, E)), ("w1", (F, E)), ("b1", (1, F)), ("w2", (E, F)),
                           ("b2", (1, E))]


def dit_stack(B: int, S: int, E: int, H: int, F: int, n: int, eps: float = 1e-6) -> Graph:
    """n DiT blocks in one graph (one launch), sharing the conditioning c. Each parameter is one
    stacked tensor of n layers, layer i at rows [i*R, (i+1)*R) (e.g. wqkv is (n*3E, E)), which
    lets runtime merge the n copies of every op into one kernel node (see megacute.layers)."""
    g = Graph()
    x = g.input("x", (B * S, E))
    c = g.input("c", (B, E))
    sc = g.unary(c, "silu")
    ps = {name: g.param(name, (n * r, cols)) for name, (r, cols) in DIT_PARAMS(E, F)}
    for i in range(n):
        g.layer = i
        p = {name: ps[name].rows_slice(i * r, r) for name, (r, _) in DIT_PARAMS(E, F)}
        x = _dit_layer(g, x, sc, B, S, E, H, eps, p)
    g.layer = None
    g.output(x)
    return g


def _dit_layer(g, x, sc, B, S, E, H, eps, p):
    D = E // H
    mod = g.linear(sc, p["w_ada"], bias=p["b_ada"])
    shift1, scale1, gate1, shift2, scale2, gate2 = (mod.cols_slice(i * E, E) for i in range(6))
    h = g.modulate(g.layer_norm(x, eps=eps), scale1, shift1, group=S)
    qkv = g.linear(h, p["wqkv"], bias=p["bqkv"])
    a = g.attention(qkv.cols_slice(0, E), qkv.cols_slice(E, E), qkv.cols_slice(2 * E, E),
                    batch=B, heads=H, head_dim=D, causal=False)
    o = g.linear(a, p["wo"], bias=p["bo"])
    x1 = g.add(x, g.binary(o, gate1, "mul", b_group=S))
    h2 = g.modulate(g.layer_norm(x1, eps=eps), scale2, shift2, group=S)
    m = g.linear(g.linear(h2, p["w1"], bias=p["b1"], act="gelu_tanh"), p["w2"], bias=p["b2"])
    return g.add(x1, g.binary(m, gate2, "mul", b_group=S))
