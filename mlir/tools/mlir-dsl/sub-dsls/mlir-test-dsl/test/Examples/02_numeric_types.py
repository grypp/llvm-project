# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s
# RUN: %if host-supports-jit %{ %PYTHON %s %}
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numeric types: literal promotion, explicit conversions and host-side results.

A parameter annotated with a dtype (`m.Int32`, `m.Float32`, `m.Float16`, ...)
is a staged value of exactly that type. A Python literal is an `Int32` or a
`Float32` and promotes with the operand it meets, so `a + 1` stays `i32` and
`x * 2.5` stays `f32`; a change of kind or width is spelled out with the dtype
(`m.Float32(a)`) or `m.cast(x, m.Int64)`. Comparisons give `m.Boolean`. A
result returns to the host as a Numeric: `repr(r)` shows dtype and payload
(`Int32(16)`), `r.value`, `int(r)` and `float(r)` unwrap it.
"""

import mlir.mlir_dsl as m


@m.jit
def int_math(a: m.Int32, b: m.Int32) -> m.Int32:
    # The literal `1` becomes an i32 constant: it follows the operand's dtype.
    s = a + 1
    # Integer `//` is arith.floordivsi; no float appears anywhere.
    q = a // b
    # m.max / m.min keep the dtype as well: arith.maxsi / arith.minsi on i32.
    return s + q + m.max(a, b) - m.min(a, b)


@m.jit
def float_math(x: m.Float32, y: m.Float32) -> m.Float32:
    # `2.5` becomes an f32 constant; float `/` is arith.divf.
    return x * 2.5 + x / y


@m.jit
def convert(a: m.Int32, x: m.Float32) -> m.Int64:
    # A dtype used as a function converts: Int32 -> Float32 is arith.sitofp,
    # and from there the f32 rules above apply (`/ 2` is an f32 divf).
    half = m.Float32(a) / 2
    # m.cast(value, dtype) is the general form and picks the op from both
    # dtypes: Float32 -> Int64 is arith.fptosi (truncates toward zero),
    # Int32 -> Int64 is arith.extsi (a signed widening).
    return m.cast(half + x, m.Int64) + m.cast(a, m.Int64)


@m.jit
def half_scale(h: m.Float16, s: m.Float16) -> m.Float16:
    # Two Float16 operands compute in f16. A bare `0.5` would not: it is a
    # Float32 and the wider float wins, so the literal is given the dtype.
    return h * s * m.Float16(0.5)


@m.jit
def is_less(a: m.Int32, x: m.Float32) -> m.Boolean:
    # A comparison is an i1, `m.Boolean`. Mixed operands promote first: the
    # Int32 side is widened to f32 (arith.sitofp), then arith.cmpf olt.
    return a < x


def check(call, r, expected):
    """Assert one host result. Under MLIR_DSL_DRYRUN results are staged (`?`)."""
    if m.is_mlir_op(r):
        return
    # A host result is a Numeric: `.value` is the Python payload, `int()` and
    # `float()` convert it, `repr()` names the dtype.
    assert r.value == expected, (call, r)
    assert int(r) == int(expected) and float(r) == float(expected)
    print(f"{call} -> {r!r}")


def main():
    check("int_math(7, 2)", int_math(7, 2), 8 + 3 + 7 - 2)  # Int32(16)
    check("float_math(2.0, 4.0)", float_math(2.0, 4.0), 5.0 + 0.5)  # Float32(5.5)
    check("convert(7, 1.0)", convert(7, 1.0), 4 + 7)  # Int64(11)
    check("half_scale(1.5, 2.0)", half_scale(1.5, 2.0), 1.5)  # Float16(1.5)
    check("is_less(3, 3.5)", is_less(3, 3.5), True)  # Boolean(True)
    check("is_less(5, 3.5)", is_less(5, 3.5), False)  # Boolean(False)
    print("Numeric types: passed")


if __name__ == "__main__":
    main()
