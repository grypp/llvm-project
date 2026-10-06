# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR arith dialect helpers: the one scalar emitter behind ``Numeric``.

The operator functions (``add`` ... ``cmp``) take plain ``ir.Value`` operands
of one element type, since type promotion runs in ``types/typing.py`` before
the call, and return the ``ir.Value`` of the op they emit. Signedness is not
carried by the values, so every operator takes it as the ``signed=`` keyword;
``Numeric._binary_op`` calls ``op(lhs_value, rhs_value, signed=...)``.
"""

from typing import Any, Callable, Optional, TYPE_CHECKING, Union

from ..... import ir
from .....dialects import arith, math
from .....extras import types as T
from ....core.common import DSLRuntimeError, DSLUserCodeError
from ....core.diagnostics import DiagId

if TYPE_CHECKING:
    from ....types.typing import Numeric, NumericMeta


# =============================================================================
# Arith Dialect Helper functions
# =============================================================================


def recast_type(src_type: ir.Type, res_elem_type: ir.Type) -> ir.Type:
    """Return ``src_type`` with its element type replaced by ``res_elem_type``.

    Element-wise ops keep the shape: a ``vector<4xf32>`` source yields a
    ``vector<4xf16>`` result type, a scalar source yields the element type.
    """
    if isinstance(src_type, ir.VectorType):
        if src_type.scalable:
            return ir.VectorType.get(
                src_type.shape, res_elem_type, scalable=list(src_type.scalable_dims)
            )
        return ir.VectorType.get(src_type.shape, res_elem_type)
    return res_elem_type


def is_scalar(ty: ir.Type) -> bool:
    """True for a non-shaped type (``i32``, ``f32``; not ``vector<4xf32>``)."""
    return not isinstance(ty, ir.ShapedType)


def element_type(ty: ir.Type) -> ir.Type:
    """Return the element type of a shaped type, or ``ty`` itself for a scalar."""
    if not is_scalar(ty):
        return ty.element_type
    return ty


def is_narrow_precision(ty: ir.Type) -> bool:
    """True for the sub-byte and 8-bit float types (FP8, FP6, FP4 families)."""
    narrow_types = {
        T.f8E3M4(),
        T.f8E8M0FNU(),
        T.f8E4M3FN(),
        T.f8E4M3(),
        T.f8E5M2(),
        T.f8E4M3B11FNUZ(),
        T.f8E5M3FNU(),
        ir.Float8E5M2FNUZType.get(),
        ir.Float8E4M3FNUZType.get(),
        T.f4E2M1FN(),
        T.f6E3M2FN(),
        T.f6E2M3FN(),
    }
    return ty in narrow_types


def is_float_type(ty: ir.Type) -> bool:
    """True for any MLIR float type."""
    return isinstance(ty, ir.FloatType)


def is_integer_like_type(ty: ir.Type) -> bool:
    """True for an integer or ``index`` type."""
    return isinstance(ty, (ir.IntegerType, ir.IndexType))


def _reads_as_signed(signed: Optional[bool]) -> bool:
    """Interpret the ``signed=`` keyword: ``None`` (unknown) reads as signed."""
    return signed is not False


def _python_int_for_integer_attr(value: int, width: int, signed: bool) -> int:
    """Coerce a Python int to a value MLIR's ``IntegerAttr.get`` accepts.

    The bindings only accept the signed range of the target width; an unsigned
    literal may come as ``0xffffffffffffffff`` or as its storage form ``-1``.
    """
    if width <= 0:
        raise DSLRuntimeError(f"Invalid integer width: {width}")

    signed_lo = -(1 << (width - 1))
    signed_hi = (1 << (width - 1)) - 1

    if signed:
        if not (signed_lo <= value <= signed_hi):
            raise DSLRuntimeError(
                f"Signed integer literal {value} does not fit in i{width}"
            )
        return value

    max_unsigned = (1 << width) - 1

    if signed_lo <= value < 0:
        return value

    if not (0 <= value <= max_unsigned):
        raise DSLRuntimeError(
            f"Unsigned integer literal {value} does not fit in unsigned i{width}"
        )

    if value > signed_hi:
        return value - (1 << width)

    return value


def bitcast(
    src: ir.Value,
    res_elem_type: ir.Type,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Reinterpret the bits of ``src`` as the same-width element type ``res_elem_type``."""
    res_type = recast_type(src.type, res_elem_type)
    return arith.bitcast(res_type, src, loc=loc, ip=ip)


def cvtf(
    src: ir.Value,
    res_elem_type: ir.Type,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Convert a float value to another float element type."""
    src_elem_type = element_type(src.type)

    if res_elem_type == src_elem_type:
        return src

    res_type = recast_type(src.type, res_elem_type)

    if res_elem_type.width > src_elem_type.width:
        return arith.extf(res_type, src, loc=loc, ip=ip)

    # bf16 <-> f16: both are 16-bit, arith.truncf requires strict narrowing.
    # Route through an f32 intermediate.
    if (src_elem_type == T.f16() and res_elem_type == T.bf16()) or (
        src_elem_type == T.bf16() and res_elem_type == T.f16()
    ):
        tmp_type = recast_type(src.type, T.f32())
        tmp = arith.extf(tmp_type, src, loc=loc, ip=ip)
        return arith.truncf(res_type, tmp, loc=loc, ip=ip)

    # E8M0 requires upward rounding; all others default to to_nearest_even.
    roundingmode = arith.RoundingMode.upward if res_elem_type == T.f8E8M0FNU() else None
    return arith.truncf(res_type, src, roundingmode=roundingmode, loc=loc, ip=ip)


def fptoi(
    src: ir.Value,
    signed: Optional[bool],
    res_elem_type: ir.Type,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Convert a float value to an integer element type of the given signedness."""
    res_type = recast_type(src.type, res_elem_type)
    if _reads_as_signed(signed):
        return arith.fptosi(res_type, src, loc=loc, ip=ip)
    return arith.fptoui(res_type, src, loc=loc, ip=ip)


def itofp(
    src: ir.Value,
    signed: Optional[bool],
    res_elem_type: ir.Type,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Convert an integer value of the given signedness to a float element type.

    An ``i1`` source is always zero-extended: the arith dialect reads ``1`` in
    ``i1`` as ``-1``, and the DSL treats booleans as unsigned.
    """
    res_type = recast_type(src.type, res_elem_type)
    if _reads_as_signed(signed) and element_type(src.type).width > 1:
        return arith.sitofp(res_type, src, loc=loc, ip=ip)
    return arith.uitofp(res_type, src, loc=loc, ip=ip)


def int_to_int(
    a: ir.Value,
    dst_elem_type: "NumericMeta",
    *,
    src_signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Convert an integer value to the width and signedness of a ``Numeric`` class.

    ``src_signed`` is the signedness of ``a``; ``None`` marks a signless value
    that took no part in DSL arithmetic yet (a raw ``ir.Value`` passed to
    ``Int32(...)``), for which the destination decides the extension.
    """
    dst_signed = dst_elem_type.signed
    src_width = element_type(a.type).width
    dst_width = dst_elem_type.width
    if src_signed is None:
        src_signed = dst_signed

    dst_mlir_type = recast_type(a.type, dst_elem_type.mlir_type)

    if dst_width == src_width:
        return a
    elif _reads_as_signed(src_signed) and not dst_signed:
        # Signed -> Unsigned: widening sign-extends first (C, NumPy and the
        # Python fold agree: Uint64(Int32(-1)) is 2**64 - 1); i1 zero-extends.
        if dst_width > src_width:
            if src_width > 1:
                return arith.extsi(dst_mlir_type, a, loc=loc, ip=ip)
            return arith.extui(dst_mlir_type, a, loc=loc, ip=ip)
        return arith.trunci(dst_mlir_type, a, loc=loc, ip=ip)
    elif src_signed == dst_signed:
        # Same signedness
        if dst_width > src_width:
            if _reads_as_signed(src_signed) and src_width > 1:
                return arith.extsi(dst_mlir_type, a, loc=loc, ip=ip)
            return arith.extui(dst_mlir_type, a, loc=loc, ip=ip)
        return arith.trunci(dst_mlir_type, a, loc=loc, ip=ip)
    else:
        # Unsigned -> Signed: truncation keeps the low bits, which the signed
        # destination then reinterprets.
        if dst_width > src_width:
            return arith.extui(dst_mlir_type, a, loc=loc, ip=ip)
        return arith.trunci(dst_mlir_type, a, loc=loc, ip=ip)


# =============================================================================
# Arith Ops Emitter Helpers
#   - assuming type of lhs and rhs match each other
#   - op name matches python module operator
# =============================================================================


def _cast(
    res_elem_ty: ir.Type,
    src: Union[ir.Value, "Numeric"],
    is_signed: Optional[bool] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Element-wise conversion of ``src`` to the element type ``res_elem_ty``.

    ``is_signed`` is the signedness of the integer side of the conversion (the
    source for an integer source, the destination for float-to-integer);
    ``None`` reads as signed.
    """
    if isinstance(src, ir.Value):
        src_ty = src.type
    else:
        src_ty = type(src).mlir_type
        src = src.ir_value()

    src_elem_ty = element_type(src_ty)

    if src_elem_ty == res_elem_ty:
        return src
    elif is_float_type(src_elem_ty) and is_float_type(res_elem_ty):
        return cvtf(src, res_elem_ty, loc=loc, ip=ip)
    elif is_integer_like_type(src_elem_ty) and is_integer_like_type(res_elem_ty):
        if src_elem_ty.width >= res_elem_ty.width:
            cast_op = arith.trunci
        elif _reads_as_signed(is_signed):
            cast_op = arith.extsi
        else:
            cast_op = arith.extui

        res_ty = recast_type(src_ty, res_elem_ty)
        return cast_op(res_ty, src, loc=loc, ip=ip)
    elif is_float_type(src_elem_ty) and is_integer_like_type(res_elem_ty):
        return fptoi(src, is_signed, res_elem_ty, loc=loc, ip=ip)
    elif is_integer_like_type(src_elem_ty) and is_float_type(res_elem_ty):
        return itofp(src, is_signed, res_elem_ty, loc=loc, ip=ip)
    else:
        raise DSLRuntimeError(
            f"cast from {src_elem_ty} to {res_elem_ty} is not supported"
        )


def const(
    value: Union[int, float, bool, ir.Value, "Numeric"],
    ty: Optional[Union[ir.Type, "NumericMeta"]] = None,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Emit ``arith.constant`` for a Python scalar; a staged value passes through.

    ``ty`` is an ``ir.Type`` or a ``Numeric`` class. Without it a ``Numeric``
    value supplies its own class and a Python scalar picks the type the way
    the literal rule does (``bool`` -> ``i1``, ``int`` -> ``i32``, ``float``
    -> ``f32``). A shaped ``ty`` yields a splat constant. An unsigned literal
    in canonical form (``0xffffffff`` for ``Uint32``) is re-encoded into the
    signed range ``IntegerAttr`` accepts.
    """
    if not isinstance(value, (bool, int, float, ir.Value)):
        # A Numeric: its payload is a Python scalar or an ir.Value, and its
        # class is the type when the caller gives none.
        if not hasattr(value, "value"):
            raise DSLRuntimeError(f"{type(value)} is not supported")
        if ty is None:
            ty = type(value)
        value = value.value

    if isinstance(value, ir.Value):
        return value

    if ty is None:
        if isinstance(value, bool):
            ty = T.bool()
        elif isinstance(value, int):
            ty = T.i32()
        elif isinstance(value, float):
            ty = T.f32()
        else:
            raise DSLRuntimeError(f"{type(value)} is not supported")
    elif not isinstance(ty, ir.Type):
        # A Numeric class: it knows its signedness and its MLIR type.
        if signed is None:
            signed = getattr(ty, "signed", None)
        ty = ty.mlir_type

    if isinstance(ty, ir.ShapedType):
        elem_ty = ty.element_type
        if isinstance(elem_ty, ir.IntegerType):
            if isinstance(value, int) and signed is False:
                value = _python_int_for_integer_attr(value, elem_ty.width, signed=False)
            attr = ir.IntegerAttr.get(elem_ty, value)
        else:
            attr = ir.FloatAttr.get(elem_ty, value)
        return arith.constant(
            ty, ir.DenseElementsAttr.get_splat(ty, attr), loc=loc, ip=ip
        )

    if isinstance(ty, ir.FloatType) and isinstance(value, (bool, int)):
        value = float(value)
    elif is_integer_like_type(ty) and isinstance(value, float):
        value = int(value)

    if isinstance(value, int) and isinstance(ty, ir.IntegerType) and signed is False:
        value = _python_int_for_integer_attr(value, ty.width, signed=False)

    return arith.constant(ty, value, loc=loc, ip=ip)


def _minmax(
    lhs: Union[int, float, ir.Value],
    rhs: Union[int, float, ir.Value],
    *,
    is_min: bool,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Union[int, float, ir.Value]:
    """Build arith min/max, assuming the operands have the same type.

    Two Python scalars fold in Python; one scalar is materialized as a constant
    of the other operand's type.
    """
    if not isinstance(lhs, ir.Value):
        if not isinstance(rhs, ir.Value):
            return min(lhs, rhs) if is_min else max(lhs, rhs)
        lhs = const(lhs, rhs.type, signed=signed, loc=loc, ip=ip)
    elif not isinstance(rhs, ir.Value):
        rhs = const(rhs, lhs.type, signed=signed, loc=loc, ip=ip)

    if is_integer_like_type(element_type(lhs.type)):
        if _reads_as_signed(signed):
            op = arith.minsi if is_min else arith.maxsi
        else:
            op = arith.minui if is_min else arith.maxui
    else:
        op = arith.minimumf if is_min else arith.maximumf
    return op(lhs, rhs, loc=loc, ip=ip)


# =============================================================================
# Operators: free functions called from Numeric._binary_op as
#   op(lhs_value, rhs_value, signed=...)
# =============================================================================


def _is_float_value(v: ir.Value) -> bool:
    """True when the element type of ``v`` is a float type."""
    return is_float_type(element_type(v.type))


def add(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``+``: ``arith.addf`` for floats, ``arith.addi`` for integers."""
    if _is_float_value(lhs):
        return arith.addf(lhs, rhs, loc=loc, ip=ip)
    return arith.addi(lhs, rhs, loc=loc, ip=ip)


def sub(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``-``: ``arith.subf`` for floats, ``arith.subi`` for integers."""
    if _is_float_value(lhs):
        return arith.subf(lhs, rhs, loc=loc, ip=ip)
    return arith.subi(lhs, rhs, loc=loc, ip=ip)


def mul(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``*``: ``arith.mulf`` for floats, ``arith.muli`` for integers."""
    if _is_float_value(lhs):
        return arith.mulf(lhs, rhs, loc=loc, ip=ip)
    return arith.muli(lhs, rhs, loc=loc, ip=ip)


def truediv(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``/``: integer operands are converted to ``f32`` first, as in Python."""
    if _is_float_value(lhs):
        return arith.divf(lhs, rhs, loc=loc, ip=ip)
    lhs = itofp(lhs, signed, T.f32(), loc=loc, ip=ip)
    rhs = itofp(rhs, signed, T.f32(), loc=loc, ip=ip)
    return arith.divf(lhs, rhs, loc=loc, ip=ip)


def floordiv(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``//``: ``arith.divf`` + ``math.floor`` for floats, ``arith.floordivsi``
    (rounding toward negative infinity) for signed and ``arith.divui`` for
    unsigned integers.
    """
    if _is_float_value(lhs):
        q = arith.divf(lhs, rhs, loc=loc, ip=ip)
        return math.floor(q, loc=loc, ip=ip)
    elif _reads_as_signed(signed):
        return arith.floordivsi(lhs, rhs, loc=loc, ip=ip)
    return arith.divui(lhs, rhs, loc=loc, ip=ip)


def mod(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``%``: ``arith.remf`` for floats, ``arith.remsi``/``arith.remui`` for integers."""
    if _is_float_value(lhs):
        return arith.remf(lhs, rhs, loc=loc, ip=ip)
    elif _reads_as_signed(signed):
        return arith.remsi(lhs, rhs, loc=loc, ip=ip)
    return arith.remui(lhs, rhs, loc=loc, ip=ip)


def pow(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``**`` through ``math.powf``/``math.fpowi``; staged int ** int is Meta-only.

    ``math.ipowi`` is lowered only by ``convert-math-to-funcs``, which outlines
    ``func`` functions the core does not use.
    """
    lhs_is_float = _is_float_value(lhs)
    rhs_is_float = _is_float_value(rhs)
    if lhs_is_float and rhs_is_float:
        return math.powf(lhs, rhs, loc=loc, ip=ip)
    elif lhs_is_float:
        return math.fpowi(lhs, rhs, loc=loc, ip=ip)
    elif rhs_is_float:
        lhs = itofp(lhs, signed, T.f32(), loc=loc, ip=ip)
        rhs = cvtf(rhs, T.f32(), loc=loc, ip=ip)
        return math.powf(lhs, rhs, loc=loc, ip=ip)
    raise DSLUserCodeError(DiagId.TYPE_INT_POW_UNSUPPORTED)


def neg(
    x: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Unary ``-``: ``arith.negf`` for floats, ``0 - x`` (``arith.subi``) for integers."""
    if _is_float_value(x):
        return arith.negf(x, loc=loc, ip=ip)
    c0 = const(0, x.type, loc=loc, ip=ip)
    return arith.subi(c0, x, loc=loc, ip=ip)


def abs(
    x: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``abs()``: ``math.absf`` for floats, ``math.absi`` for integers."""
    if _is_float_value(x):
        return math.absf(x, loc=loc, ip=ip)
    return math.absi(x, loc=loc, ip=ip)


def and_(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``&``: ``arith.andi``."""
    return arith.andi(lhs, rhs, loc=loc, ip=ip)


def or_(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``|``: ``arith.ori``."""
    return arith.ori(lhs, rhs, loc=loc, ip=ip)


def xor(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``^``: ``arith.xori``."""
    return arith.xori(lhs, rhs, loc=loc, ip=ip)


def shl(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``<<``: ``arith.shli``."""
    return arith.shli(lhs, rhs, loc=loc, ip=ip)


def shr(
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``>>``: arithmetic ``arith.shrsi`` for signed, logical ``arith.shrui`` for unsigned."""
    if _reads_as_signed(signed):
        return arith.shrsi(lhs, rhs, loc=loc, ip=ip)
    return arith.shrui(lhs, rhs, loc=loc, ip=ip)


# Predicate name (the Python operator's __name__) -> (cmpf, signed cmpi,
# unsigned cmpi). Float comparisons are ordered except `ne`: in Python
# bool(float("nan")) is True, so `!=` is the unordered predicate.
_CMP_PREDICATES = {
    "lt": (arith.CmpFPredicate.OLT, arith.CmpIPredicate.slt, arith.CmpIPredicate.ult),
    "le": (arith.CmpFPredicate.OLE, arith.CmpIPredicate.sle, arith.CmpIPredicate.ule),
    "gt": (arith.CmpFPredicate.OGT, arith.CmpIPredicate.sgt, arith.CmpIPredicate.ugt),
    "ge": (arith.CmpFPredicate.OGE, arith.CmpIPredicate.sge, arith.CmpIPredicate.uge),
    "eq": (arith.CmpFPredicate.OEQ, arith.CmpIPredicate.eq, arith.CmpIPredicate.eq),
    "ne": (arith.CmpFPredicate.UNE, arith.CmpIPredicate.ne, arith.CmpIPredicate.ne),
}


def cmp(
    pred: Union[str, Callable[..., Any]],
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Compare two values into ``i1``; ``pred`` is ``"lt"``/``"le"``/``"gt"``/
    ``"ge"``/``"eq"``/``"ne"`` or the ``operator`` function of that name.
    """
    name = pred if isinstance(pred, str) else getattr(pred, "__name__", "")
    if name not in _CMP_PREDICATES:
        raise DSLRuntimeError(f"cmp: unsupported predicate {pred!r}")
    fpred, spred, upred = _CMP_PREDICATES[name]
    if _is_float_value(lhs):
        return arith.cmpf(fpred, lhs, rhs, loc=loc, ip=ip)
    ipred = spred if _reads_as_signed(signed) else upred
    return arith.cmpi(ipred, lhs, rhs, loc=loc, ip=ip)


def select(
    cond: ir.Value,
    a: ir.Value,
    b: ir.Value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """``a if cond else b`` on values of one type: ``arith.select``."""
    return arith.select(cond, a, b, loc=loc, ip=ip)


def cast(
    src: Union[ir.Value, "Numeric"],
    dst_type: Union[ir.Type, "NumericMeta"],
    *,
    signed: Optional[bool] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Convert ``src`` to ``dst_type``, an ``ir.Type`` or a ``Numeric`` class.

    ``signed`` is the signedness of the integer side of the conversion (see
    ``_cast``); for a float source and a ``Numeric`` destination it defaults
    to the destination's.
    """
    if not isinstance(dst_type, ir.Type):
        if signed is None:
            src_ty = src.type if isinstance(src, ir.Value) else type(src).mlir_type
            if is_float_type(element_type(src_ty)):
                signed = getattr(dst_type, "signed", None)
        dst_type = dst_type.mlir_type
    return _cast(dst_type, src, signed, loc=loc, ip=ip)
