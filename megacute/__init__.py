"""megacute: persistent megakernels for transformer and diffusion graphs, entirely in CuTe DSL.

    import megacute as mc
    g = mc.Graph()
    x = g.input("x", (4096, 2048)); w = g.param("w", (8192, 2048))
    g.output(g.linear(x, w, act="gelu_tanh"))
    prog = mc.compile(g)
    y = prog(x=x_tensor, w=w_tensor)
"""

from .graph import Graph, View, Buffer, swiglu_weight  # noqa: F401
from .schedule import schedule  # noqa: F401


def compile(g, **kw):
    from .runtime import compile as _compile
    return _compile(g, **kw)
