# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Math functions on ``Numeric`` values over the MLIR math dialect.

Each function takes a ``Numeric`` scalar (or a Python literal, promoted to
``Int32``/``Float32``) and returns a value of the same ``Numeric`` class. A raw
``ir.Value`` passes through and comes back raw. Every function but ``abs``
accepts floats only; an integer operand raises ``ARG_ANNOTATION_MISMATCH``.
"""

from typing import Callable, Optional, Union

from ..... import ir
from .....dialects import math as math_dialect
from ....core.common import DSLUserCodeError
from ....core.diagnostics import DiagId
from ....types.typing import Float32, Int32, Numeric
from .arith import element_type, is_float_type
from ....core.user_op import dsl_user_op

# =============================================================================
# Type alias
# =============================================================================

MathOperand = Union[Numeric, ir.Value, float, int, bool]


# =============================================================================
# Helpers
# =============================================================================


def _coerce_operand(x: MathOperand) -> MathOperand:
    """Promote Python numeric literals to Numeric scalars.

    Python ``bool``/``int`` -> :class:`Int32`, ``float`` -> :class:`Float32`.
    ``bool`` is checked before ``int`` because ``bool`` subclasses ``int``.
    Non-literal operands pass through unchanged.
    """
    if isinstance(x, bool):
        return Int32(int(x))
    if isinstance(x, float):
        return Float32(x)
    if isinstance(x, int):
        return Int32(x)
    return x


def _get_ir_value(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Extract the MLIR value from any supported operand type."""
    x = _coerce_operand(x)
    if isinstance(x, ir.Value):
        return x
    if not isinstance(x, Numeric):
        raise DSLUserCodeError(
            DiagId.ARG_NOT_NUMERIC, arg_name="x", arg_type=type(x).__name__
        )
    return x.ir_value(loc=loc, ip=ip)


def _wrap_result(original: MathOperand, result_ir: ir.Value) -> MathOperand:
    """Wrap an MLIR result back into the original operand's type."""
    if isinstance(original, ir.Value):
        return result_ir
    # Coerce Python literals so we return a Numeric subclass instance instead
    # of attempting to construct e.g. ``float(ir.Value)``.
    original = _coerce_operand(original)
    return type(original)(result_ir)


def _unary_math_op(
    x: MathOperand,
    float_op: Callable[..., ir.Value],
    int_op: Optional[Callable[..., ir.Value]],
    op_name: str,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """Apply a unary math-dialect op builder and re-wrap the result as ``x``.

    ``float_op`` serves float operands; ``int_op`` serves integer operands and
    ``None`` makes an integer operand an ``ARG_ANNOTATION_MISMATCH`` error
    naming ``op_name``.
    """
    x_ir = _get_ir_value(x, loc=loc, ip=ip)

    if is_float_type(element_type(x_ir.type)):
        result = float_op(x_ir, loc=loc, ip=ip)
    else:
        if int_op is None:
            raise DSLUserCodeError(
                DiagId.ARG_ANNOTATION_MISMATCH,
                num=1,
                arg_name="x",
                expected="a floating-point value",
                got=type(_coerce_operand(x)).__name__,
                context={"function": op_name},
            )
        result = int_op(x_ir, loc=loc, ip=ip)

    return _wrap_result(x, result)


# =============================================================================
# Transcendentals
# =============================================================================


@dsl_user_op
def sin(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.sin``: the sine of ``x`` (radians).

    :param x: A float operand
    :return: The sine of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.sin, None, "sin", loc=loc, ip=ip)


@dsl_user_op
def cos(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.cos``: the cosine of ``x`` (radians).

    :param x: A float operand
    :return: The cosine of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.cos, None, "cos", loc=loc, ip=ip)


@dsl_user_op
def exp(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.exp``: ``e ** x``.

    :param x: A float operand
    :return: The exponential of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.exp, None, "exp", loc=loc, ip=ip)


@dsl_user_op
def log(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.log``: the natural logarithm of ``x``.

    :param x: A float operand
    :return: The natural logarithm of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.log, None, "log", loc=loc, ip=ip)


@dsl_user_op
def tanh(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.tanh``: the hyperbolic tangent of ``x``.

    :param x: A float operand
    :return: The hyperbolic tangent of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.tanh, None, "tanh", loc=loc, ip=ip)


@dsl_user_op
def rsqrt(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.rsqrt``: the reciprocal square root ``1 / sqrt(x)``.

    :param x: A float operand
    :return: The reciprocal square root of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.rsqrt, None, "rsqrt", loc=loc, ip=ip)


@dsl_user_op
def sqrt(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.sqrt``: the square root of ``x``.

    :param x: A float operand
    :return: The square root of ``x``, of the same type
    """
    return _unary_math_op(x, math_dialect.sqrt, None, "sqrt", loc=loc, ip=ip)


# =============================================================================
# Rounding and absolute value
# =============================================================================


@dsl_user_op
def abs(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.absf`` / ``math.absi``: the absolute value of ``x``.

    :param x: A float or integer operand
    :return: The absolute value of ``x``, of the same type
    """
    return _unary_math_op(
        x, math_dialect.absf, math_dialect.absi, "abs", loc=loc, ip=ip
    )


@dsl_user_op
def ceil(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.ceil``: ``x`` rounded toward positive infinity.

    :param x: A float operand
    :return: The ceiling of ``x``, of the same (float) type
    """
    return _unary_math_op(x, math_dialect.ceil, None, "ceil", loc=loc, ip=ip)


@dsl_user_op
def floor(
    x: MathOperand,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> MathOperand:
    """``math.floor``: ``x`` rounded toward negative infinity.

    :param x: A float operand
    :return: The floor of ``x``, of the same (float) type
    """
    return _unary_math_op(x, math_dialect.floor, None, "floor", loc=loc, ip=ip)
