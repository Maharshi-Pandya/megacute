"""Scalar fp32 device math via PTX approximations (ex2/rcp/rsqrt/tanh are single SFU ops)."""

import cutlass.cute as cute
from cutlass import Float32

LOG2E = 1.4426950408889634


def _ptx1(op, x):
    return cute.arch.inline_ptx(f"{op} {{$w0}}, {{$r0}};", write_only_types=[Float32], read_only_args=[Float32(x)])


def ex2(x):
    return _ptx1("ex2.approx.ftz.f32", x)


def rcp(x):
    return _ptx1("rcp.approx.ftz.f32", x)


def rsqrt(x):
    return _ptx1("rsqrt.approx.ftz.f32", x)


def tanh(x):
    return _ptx1("tanh.approx.f32", x)


def sigmoid(x):
    return rcp(Float32(1.0) + ex2(-x * LOG2E))


def fmax(a, b):
    return cute.arch.fmax(Float32(a), Float32(b))


def act(name, x):
    """Activation by name (a Python constant at trace time)."""
    if name is None or name == "identity":
        return x
    if name == "relu":
        return fmax(x, Float32(0.0))
    if name == "silu":
        return x * sigmoid(x)
    if name == "sigmoid":
        return sigmoid(x)
    if name in ("gelu_tanh", "gelu"):
        # tanh approximation for both: torch's exact erf gelu differs by < 1e-3 at bf16 output
        return Float32(0.5) * x * (Float32(1.0) + tanh(Float32(0.7978845608028654) * (x + Float32(0.044715) * x * x * x)))
    if name == "exp":
        return ex2(x * LOG2E)
    if name == "neg":
        return -x
    if name == "square":
        return x * x
    raise ValueError(f"unknown activation {name}")


def binary(name, a, b):
    if name == "add":
        return a + b
    if name == "sub":
        return a - b
    if name == "mul":
        return a * b
    if name == "max":
        return fmax(a, b)
    if name == "min":
        return -fmax(-a, -b)
    raise ValueError(f"unknown binary op {name}")
