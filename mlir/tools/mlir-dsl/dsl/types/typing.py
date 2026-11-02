# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The DSL type system: scalar ``Numeric`` types, ``Pointer`` and ``@struct``.

A ``Numeric`` holds a Python scalar (a compile-time value that folds) or an
``ir.Value`` (a staged value); its operators promote the operands, fold Python
payloads in Python and emit staged ones through the tracing DSL's
``type_ops`` plugin (an ``OpEmitter``, ``core/mlir_op.py``), which also
answers every SSA type: the module imports no dialect. A ``Pointer`` is one
pointer SSA value of that plugin's choice (``!llvm.ptr`` under the builtin
type ops) with ``(dtype, addrspace)``
metadata; a ``@struct`` is a frozen record of DSL-typed fields and a pytree
that flattens to its fields at every boundary, not an SSA aggregate. Outside a
trace a ``Pointer`` holds a host address and a struct its host field values,
so both can be passed to a ``@jit`` function.
"""

import ctypes
import math
import operator
import struct as _pystruct
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    Type,
    Union,
    cast as tcast,
)

from ... import ir
from ...extras import types as T
from ..core.common import (
    DSLRuntimeError,
    DSLUserCodeError,
    get_current_dsl,
    report_warning,
)
from ..core.diagnostics import DiagId, WarnId
from ..core.mlir_op import current_emitter as _emitter

# =============================================================================
# Metaclasses and the dtype registry
# =============================================================================


class DslType(type):
    """Metaclass for all DSL types in the system.

    :param is_abstract: Whether this type is abstract, defaults to False
    :type is_abstract: bool, optional
    """

    _is_abstract: bool

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        new_cls = super().__new__(cls, name, bases, attrs)
        new_cls._is_abstract = is_abstract
        return new_cls

    @property
    def is_abstract(cls) -> bool:
        """True for the abstract bases (``Numeric``, ``Integer``, ``Float``)."""
        return cls._is_abstract


# Every concrete dtype registers itself here when its class is created
# (``NumericMeta.__new__``), so a sub-DSL's ``class MyFloat(Float, ...)`` is
# found by ``dtype()``, ``from_numpy_dtype()`` and ``Numeric.from_mlir_type()``.
ALL_DTYPES: set = set()
_NAME_TO_DTYPE: dict = {}
_NP_NAME_TO_DTYPE: dict = {}
# MLIR type spelling (``"i32"``, ``"ui8"``, ``"f8E4M3FN"``) -> dtype. Type
# spellings do not depend on the context, so the index is built once and
# cleared whenever a dtype registers.
_MLIR_NAME_TO_DTYPE: dict = {}


def _register_dtype(cls: "NumericMeta") -> None:
    """Index a concrete dtype by class name and NumPy name; drop the MLIR index."""
    ALL_DTYPES.add(cls)
    _NAME_TO_DTYPE[cls.__name__] = cls
    if cls._np_dtype_name is not None:
        _NP_NAME_TO_DTYPE[cls._np_dtype_name] = cls
    _MLIR_NAME_TO_DTYPE.clear()


class NumericMeta(DslType):
    """Metaclass for numeric types providing width and numpy dtype information.

    :param width: Bit width of the numeric type, defaults to 8
    :type width: int
    :param np_dtype_name: Name of the corresponding NumPy scalar type, or None
        when NumPy has no matching type
    :type np_dtype_name: str, optional
    :param mlir_type: Callable returning the corresponding MLIR type
    :type mlir_type: Callable[[], ir.Type], optional
    :param is_abstract: Whether the type is abstract, defaults to False
    :type is_abstract: bool, optional
    :param ctype: The ctypes representative used to pass a value of this type
        to a compiled function, or None when the type is staged-only
    :type ctype: type, optional
    """

    width: int
    bytes: int
    ctype: Optional[type]
    _mlir_type: Optional[Callable[[], ir.Type]]
    _np_dtype_name: Optional[str]

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 8,
        np_dtype_name: Optional[str] = None,
        mlir_type: Optional[Callable[[], ir.Type]] = None,
        is_abstract: bool = False,
        ctype: Optional[type] = None,
        **kwargs: Any,
    ) -> Any:
        # Instances carry their payload in the one slot ``Numeric`` declares.
        attrs.setdefault("__slots__", ())
        new_cls = super().__new__(cls, name, bases, attrs, is_abstract=is_abstract)

        new_cls._mlir_type = staticmethod(mlir_type) if mlir_type is not None else None
        new_cls.width = width
        new_cls.bytes = max(1, (width + 7) // 8)
        new_cls.ctype = ctype
        new_cls._np_dtype_name = np_dtype_name
        if not is_abstract:
            _register_dtype(new_cls)
        return new_cls

    @property
    def scalar_mlir_type(cls) -> ir.Type:
        """The scalar MLIR type of this dtype (``i32``, ``f16``, ...); needs an
        active ``ir.Context``."""
        if cls._mlir_type is None:
            raise DSLRuntimeError(f"{cls.__name__} has no MLIR type")
        return cls._mlir_type()

    @property
    def mlir_type(cls) -> ir.Type:
        """The SSA type of this dtype under the tracing DSL's ``type_ops``
        plugin: the builtin scalar type (``i32``) under the builtin type ops,
        whatever another plugin answers (a rank-0 tile, ...)."""
        return _emitter().mlir_type(cls)


def cast(
    obj: Union[bool, int, float, ir.Value, "Numeric"], type_: Type["Numeric"]
) -> "Numeric":
    """Cast an object to the specified numeric type.

    :param obj: Object to be cast
    :type obj: Union[bool, int, float, ir.Value, Numeric]
    :param type_: Target numeric type
    :type type_: Type[Numeric]
    :return: Object cast to the target numeric type
    :rtype: Numeric

    An abstract target (``Integer``, ``Float``, ``Numeric``) accepts a value
    that already is an instance of it and returns it unchanged. A Python
    ``bool``, ``int`` or ``float`` takes the concrete dtype of the family that
    the tracing DSL's ``type_ops`` plugin assigns it (``OpEmitter.infer_dtype``);
    the core has no concrete type of its own to build, so it is a user error
    when no DSL is tracing or the plugin assigns none.

    Example::

        x = cast(5, Int32)  # Int32(5)
        y = cast(3.14, Float32)  # Float32(3.14)
        z = cast(5, Numeric)  # Int32(5) under the shipped type ops
    """
    if type_.is_abstract:
        if isinstance(obj, type_):
            return obj
        return _infer_dtype(obj, type_)(obj)  # type: ignore[arg-type]
    # The annotation's constructor performs the conversion.
    return type_(obj)  # type: ignore[arg-type]


def _infer_dtype(value: Any, family: Type["Numeric"]) -> Type["Numeric"]:
    """The concrete dtype of the abstract ``family`` that the tracing DSL's
    ``type_ops`` plugin assigns to the Python number ``value``."""

    def rejected(reason: str) -> DSLUserCodeError:
        return DSLUserCodeError(
            f"Cannot cast a `{type(value).__name__}` to the abstract type "
            f"`{family.__name__}`: {reason}.",
            suggestion="Pass a DSL value of that family (`Int32(x)`, ...) or "
            "use a concrete type instead, e.g. `Int32` or `Float32`.",
        )

    if not isinstance(value, (bool, int, float)):
        raise rejected("it is not a Python number")
    if get_current_dsl() is None:
        raise rejected("no DSL is tracing, so there is no `type_ops` plugin to ask")
    dtype = _emitter().infer_dtype(value, family)
    if dtype is None:
        raise rejected("the DSL's `type_ops` plugin assigns it no dtype")
    if not (
        isinstance(dtype, NumericMeta)
        and not dtype.is_abstract
        and issubclass(dtype, family)
    ):
        raise DSLRuntimeError(
            f"the `type_ops` plugin assigned `{dtype!r}` to a "
            f"`{type(value).__name__}` under `{family.__name__}`, which is not "
            "a concrete dtype of that family"
        )
    return dtype


_INTEGER_DTYPE_NAMES: dict = {
    (8, True): "int8",
    (16, True): "int16",
    (32, True): "int32",
    (64, True): "int64",
    (8, False): "uint8",
    (16, False): "uint16",
    (32, False): "uint32",
    (64, False): "uint64",
}


class IntegerMeta(NumericMeta):
    """Metaclass for integer types providing signedness information.

    :param width: Bit width of the integer type, defaults to 32
    :type width: int
    :param signed: Whether the integer type is signed, defaults to True
    :type signed: bool
    :param mlir_type: Callable returning the corresponding MLIR type
    :type mlir_type: Callable[[], ir.Type], optional

    :ivar signed: Whether the integer type is signed
    :vartype signed: bool
    """

    signed: bool
    # Value range this type stores exactly; see ``Integer.__init__``.
    _exact_range: tuple

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 32,
        signed: bool = True,
        mlir_type: Optional[Callable[[], ir.Type]] = None,
        is_abstract: bool = False,
    ) -> Any:
        np_dtype_name = (
            "bool_" if width == 1 else _INTEGER_DTYPE_NAMES.get((width, signed))
        )
        if width == 1:
            ctype: Optional[type] = ctypes.c_bool
        elif np_dtype_name is not None:
            ctype = getattr(ctypes, f"c_int{width}" if signed else f"c_uint{width}")
        else:
            ctype = None

        new_cls = super().__new__(
            cls,
            name,
            bases,
            attrs,
            width,
            np_dtype_name,
            mlir_type,
            is_abstract,
            ctype=ctype,
        )
        new_cls.signed = signed
        # Precomputed once per type so ``Integer.__init__`` can range-check
        # without rebuilding the bounds on every construction. bool folds
        # everything nonzero to True, so only 0 and 1 survive its cast.
        if width == 1:
            new_cls._exact_range = (0, 1)
        elif signed:
            new_cls._exact_range = (-(2 ** (width - 1)), 2 ** (width - 1) - 1)
        else:
            new_cls._exact_range = (0, 2**width - 1)
        return new_cls

    def __str__(cls) -> str:
        return f"{cls.__name__}"

    @property
    def is_integer(cls) -> bool:
        return True

    @property
    def is_float(cls) -> bool:
        return False

    @property
    def zero(cls) -> int:
        return 0

    @property
    def min(cls) -> int:
        if cls.signed:
            return -(2 ** (cls.width - 1))
        else:
            return 0

    @property
    def max(cls) -> int:
        if cls.signed:
            return 2 ** (cls.width - 1) - 1
        else:
            return 2**cls.width - 1


class FloatMeta(NumericMeta):
    """Metaclass for floating-point types.

    :param width: Bit width of the float type, defaults to 32
    :type width: int
    :param mlir_type: Callable returning the corresponding MLIR type
    :type mlir_type: Callable[[], ir.Type], optional
    :param is_abstract: Whether this is an abstract base class, defaults to False
    :type is_abstract: bool, optional
    :param exponent_width: Exponent bits of the format
    :type exponent_width: int, optional
    :param mantissa_width: Mantissa bits of the format
    :type mantissa_width: int, optional
    :param np_dtype_name: Name of the corresponding NumPy scalar type
    :type np_dtype_name: str, optional
    :param ctype: The ctypes representative, or None when staged-only
    :type ctype: type, optional
    """

    _exponent_width: int
    _mantissa_width: int

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 32,
        mlir_type: Optional[Callable[[], ir.Type]] = None,
        is_abstract: bool = False,
        *,
        exponent_width: Optional[int] = None,
        mantissa_width: Optional[int] = None,
        np_dtype_name: Optional[str] = None,
        ctype: Optional[type] = None,
    ) -> Any:
        new_cls = super().__new__(
            cls,
            name,
            bases,
            attrs,
            width,
            np_dtype_name,
            mlir_type,
            is_abstract,
            ctype=ctype,
        )
        if exponent_width is not None:
            new_cls._exponent_width = exponent_width
        if mantissa_width is not None:
            new_cls._mantissa_width = mantissa_width
        return new_cls

    def __str__(cls) -> str:
        return f"{cls.__name__}"

    @property
    def is_integer(cls) -> bool:
        return False

    @property
    def is_float(cls) -> bool:
        return True

    @property
    def zero(cls) -> float:
        return 0.0

    def recast_width(cls, width: int) -> Type["Float"]:
        """Return the IEEE float dtype of ``width`` bits (16, 32 or 64).

        Used by ``_promote_float`` to move an integer operand into the float
        family at the wider of the two operand widths.
        """
        type_map = {
            32: Float32,
        }
        if width not in type_map:
            raise DSLRuntimeError(f"Unsupported float width: {width}")
        return type_map[width]


# =============================================================================
# Type promotion
# =============================================================================


def _binary_op_type_promote(
    a: "Numeric",
    b: "Numeric",
    promote_bool: bool = False,
    *,
    op_name: str = "this operation",
) -> tuple:
    """Promote two numeric operands following type promotion rules.

    :param a: First numeric operand
    :type a: Numeric
    :param b: Second numeric operand
    :type b: Numeric
    :param promote_bool: Whether to promote boolean types to Int32 for arithmetic operations, defaults to False
    :type promote_bool: bool, optional
    :param op_name: Name of the operation, for the diagnostic
    :type op_name: str, optional
    :return: Tuple containing promoted operands and their resulting type
    :rtype: tuple[Numeric, Numeric, Type[Numeric]]

    Type promotion rules:

    1. Same dtype (and not two ``Boolean`` with ``promote_bool``): unchanged.
    2. Either operand is a float (``_promote_float``): an integer operand is
       recast to the IEEE float of ``max(width_a, width_b)`` bits; then the
       wider float wins, and at equal width ``Float64 > Float32 > Float16``
       (so ``Float32`` over ``TFloat32``, ``Float16`` over ``BFloat16``).
       Two narrow floats (fp8, fp6, fp4) of different formats, or a width the
       IEEE family lacks, raise ``TYPE_IMPLICIT_PROMOTION_UNSUPPORTED``: the
       user converts explicitly.
    3. Both operands are integers (``_promote_integer``): with
       ``promote_bool`` two ``Boolean`` become ``Int32`` first; mixed
       signedness picks the unsigned dtype when its width is at least the
       signed one's, else the signed dtype; same signedness picks the wider.
    """
    a_type = a.dtype
    b_type = b.dtype

    # Early return for same types (except when they're bools that need promotion)
    if a_type == b_type and not (promote_bool and a_type is Boolean):
        return a, b, a_type

    # Handle floating point promotions
    if a_type.is_float or b_type.is_float:
        return _promote_float(a, b, a_type, b_type, op_name=op_name)

    # Handle bool promotion for arithmetic operations
    if promote_bool:
        if a_type is Boolean and b_type is Boolean:
            # Only promote to Int32 when both are bool
            a = a.to(Int32)
            b = b.to(Int32)
            a_type = b_type = a.dtype

    # Same type, no promotion needed (also covers both-bool -> Int32 above)
    if a_type == b_type:
        return a, b, a_type

    # At this point both must be Integer subclasses (float branch above already returned).
    return _promote_integer(
        a, b, tcast("Type[Integer]", a_type), tcast("Type[Integer]", b_type)
    )


def _apply_promotion(a: "Numeric", b: "Numeric", res_type: Type["Numeric"]) -> tuple:
    """Cast each operand to ``res_type`` (only when its dtype differs)."""
    new_a = a.to(res_type) if a.dtype != res_type else a
    new_b = b.to(res_type) if b.dtype != res_type else b
    return new_a, new_b, res_type


def _promote_float(
    a: "Numeric",
    b: "Numeric",
    a_type: Type["Numeric"],
    b_type: Type["Numeric"],
    *,
    op_name: str = "this operation",
) -> tuple:
    """Promotion policy when at least one operand is a float type."""
    orig_a_type, orig_b_type = a_type, b_type
    a_width = a_type.width
    b_width = b_type.width

    def unsupported() -> DSLUserCodeError:
        return DSLUserCodeError(
            DiagId.TYPE_IMPLICIT_PROMOTION_UNSUPPORTED,
            lhs_type=orig_a_type.__name__,
            rhs_type=orig_b_type.__name__,
            op=op_name,
        )

    # If one type is integer, convert it to the float type
    if a_type.is_float != b_type.is_float:
        if max(a_width, b_width) != 32:
            raise unsupported()
        if a_type.is_float:
            b_type = a_type.recast_width(max(a_width, b_width))  # type: ignore[attr-defined]
        else:
            a_type = b_type.recast_width(max(a_width, b_width))  # type: ignore[attr-defined]

    # Both are float types - handle precision promotion
    if a_width > b_width and a_width >= 16:
        res_type = a_type
    elif b_width > a_width and b_width >= 16:
        res_type = b_type
    elif a_width == b_width:
        # Same bitwidth - handle special cases like TFloat32 -> Float32 and BFloat16 -> Float16
        if a_type is Float32 or b_type is Float32:
            res_type = Float32
        else:
            raise unsupported()
    else:
        raise unsupported()

    return _apply_promotion(a, b, res_type)


def _promote_integer(
    a: "Numeric", b: "Numeric", a_type: Type["Integer"], b_type: Type["Integer"]
) -> tuple:
    """Promotion policy when both operands are integers (same dtype already handled).

    Mixed signedness picks the unsigned type when its width is at least the
    signed width, else the signed one; same signedness picks the wider type.
    """
    a_signed = a_type.signed
    b_signed = b_type.signed
    a_width = a_type.width
    b_width = b_type.width

    # Mixed signedness case
    if a_signed != b_signed:
        unsigned_type = a_type if not a_signed else b_type
        signed_type = a_type if a_signed else b_type
        unsigned_width = a_width if not a_signed else b_width

        if unsigned_width >= signed_type.width:
            # Promote both to unsigned of larger width
            res_type = unsigned_type
        else:
            # Promote both to signed of larger width
            res_type = signed_type

        return _apply_promotion(a, b, res_type)

    # Same signedness, different width - promote to larger width
    if a_width >= b_width:
        return a, b.to(a.dtype), a.dtype
    else:
        return a.to(b.dtype), b, b.dtype


# =============================================================================
# Binary operator wrapper
# =============================================================================

# Python operator -> the emitter method of the same name, resolved per call.
_OPERATOR_EMITTERS: dict = {
    operator.add: lambda *a, **k: _emitter().add(*a, **k),
    operator.sub: lambda *a, **k: _emitter().sub(*a, **k),
    operator.mul: lambda *a, **k: _emitter().mul(*a, **k),
    operator.truediv: lambda *a, **k: _emitter().truediv(*a, **k),
    operator.floordiv: lambda *a, **k: _emitter().floordiv(*a, **k),
    operator.mod: lambda *a, **k: _emitter().mod(*a, **k),
    operator.pow: lambda *a, **k: _emitter().pow(*a, **k),
    operator.and_: lambda *a, **k: _emitter().and_(*a, **k),
    operator.or_: lambda *a, **k: _emitter().or_(*a, **k),
    operator.xor: lambda *a, **k: _emitter().xor(*a, **k),
    operator.lshift: lambda *a, **k: _emitter().shl(*a, **k),
    operator.rshift: lambda *a, **k: _emitter().shr(*a, **k),
    operator.lt: lambda *a, **k: _emitter().cmp("lt", *a, **k),
    operator.le: lambda *a, **k: _emitter().cmp("le", *a, **k),
    operator.gt: lambda *a, **k: _emitter().cmp("gt", *a, **k),
    operator.ge: lambda *a, **k: _emitter().cmp("ge", *a, **k),
    operator.eq: lambda *a, **k: _emitter().cmp("eq", *a, **k),
    operator.ne: lambda *a, **k: _emitter().cmp("ne", *a, **k),
}

_COMPARISON_OPS = (
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.eq,
    operator.ne,
)


def _binary_op(
    op: Callable[..., Any],
    promote_operand: bool = True,
    promote_bool: bool = False,
    flip: bool = False,
) -> Callable[..., Any]:
    """Wrapper for binary operations on Numeric types.

    This wrapper handles type promotion, operation execution, and result type
    determination for binary operations between Numeric types. Two Python
    payloads fold in Python through ``op``; otherwise both operands are
    materialized as ``ir.Value`` of the promoted dtype and the type-ops emitter
    method of the same name is called as ``emit(lhs, rhs, signed=...)``.

    :param op: The binary operation to perform (e.g., operator.add, operator.sub)
    :type op: callable
    :param promote_operand: Whether to promote operands to the same type, defaults to True
    :type promote_operand: bool, optional
    :param promote_bool: Whether to promote boolean operands to Int32, defaults to False
    :type promote_bool: bool, optional
    :param flip: Whether to flip the operands when calling the operation, defaults to False
    :type flip: bool, optional
    """

    def wrapper(
        lhs: "Numeric",
        rhs: Union[int, float, bool, ir.Value, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Any:
        orig_lhs_type = type(lhs)
        orig_rhs_type = type(rhs)

        ty = type(lhs)
        # Canonicalize the right operand to a Numeric for promotion.
        if not isinstance(rhs, Numeric):
            if isinstance(rhs, ir.Value):
                # A raw SSA value of a scalar dtype takes part; anything else
                # (a vector, a pointer) gets to implement the reflected op.
                if _lookup_mlir_type(rhs.type) is None:
                    return NotImplemented
            elif not isinstance(rhs, (int, float, bool)):
                # This allows rhs class to implement __rmul__
                return NotImplemented

            rhs = as_numeric(rhs)

        # The result type defaults to the left-hand side's.
        res_type = ty

        if promote_operand:
            lhs, rhs, res_type = _binary_op_type_promote(
                lhs, rhs, promote_bool, op_name=getattr(op, "__name__", "operator")
            )
        else:
            rhs = ty(rhs)  # type: ignore[arg-type]

        if op in _COMPARISON_OPS:
            res_type = Boolean
        elif op is operator.truediv and isinstance(lhs, Integer):
            res_type = Float32
        elif promote_bool and orig_lhs_type is Boolean and orig_rhs_type is Boolean:
            res_type = Boolean

        lhs_val = lhs.value
        rhs_val = rhs.value
        if flip:
            lhs_val, rhs_val = rhs_val, lhs_val

        if isinstance(lhs_val, ir.Value) or isinstance(rhs_val, ir.Value):
            # Both operands share one dtype after promotion; a Python payload
            # becomes a constant of that dtype.
            operand_type = lhs.dtype
            signed = getattr(operand_type, "signed", None)
            lhs_val = _emitter().const(lhs_val, operand_type, loc=loc, ip=ip)
            rhs_val = _emitter().const(rhs_val, operand_type, loc=loc, ip=ip)
            emit = _OPERATOR_EMITTERS.get(op, op)
            res_val = emit(lhs_val, rhs_val, signed=signed, loc=loc, ip=ip)
        else:
            res_val = op(lhs_val, rhs_val)
        return res_type(res_val, loc=loc, ip=ip)

    return wrapper


# =============================================================================
# Numeric
# =============================================================================


class Numeric(metaclass=NumericMeta, is_abstract=True):
    """Base class for all numeric types in the DSL.

    This class provides the foundation for both Integer and Float types,
    implementing basic arithmetic operations. Instances are immutable value
    objects: the one slot ``value`` holds a Python scalar (a compile-time
    value) or an ``ir.Value`` (a staged value).

    :param value: The value to store in the numeric type
    :type value: Union[bool, int, float, ir.Value, Numeric]

    :ivar value: The stored numeric value
    :vartype value: Union[bool, int, float, ir.Value]
    """

    __slots__ = ("value",)

    # Injected by NumericMeta.__new__ on every concrete subclass.
    width: ClassVar[int]
    bytes: ClassVar[int]
    ctype: ClassVar[Optional[type]]
    _np_dtype_name: ClassVar[Optional[str]]

    value: Any

    def __init__(
        self,
        value: Union[bool, int, float, ir.Value, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Store the payload; ``loc``/``ip`` keep the constructor signature
        uniform across dtypes (the subclasses emit the conversion ops)."""
        self.value = value

    def __str__(self) -> str:
        """The Python payload, or ``?`` for a staged value."""
        if isinstance(self.value, ir.Value):
            return "?"
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.value)})"

    def __hash__(self) -> int:
        return hash(type(self)) ^ hash(self.value)

    @property
    def dtype(self) -> Type["Numeric"]:
        return type(self)

    def to(
        self,
        dtype: Type,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Any:
        """Convert this numeric value to another numeric type.

        If the target type is the same as the current type, returns self.
        Otherwise, creates a new instance of the target type with the same
        value. ``ir.Value`` as the target materializes a Python payload through
        the type ops' ``const``; ``int``/``float``/``bool`` return the Python
        payload and require a compile-time value.

        :param dtype: The target numeric type to convert to
        :type dtype: Union[Type["Numeric"], Type[int], Type[float], Type[bool], Type[ir.Value]]
        :return: A new instance of the target type, or self if types match
        :rtype: Numeric

        Example:

        .. code-block:: python

            x = Int32(5)
            y = x.to(Float32)  # Converts to Float32(5.0)
            z = x.to(int)      # Returns Python int 5.
        """
        if dtype is type(self):
            return self
        elif isinstance(dtype, NumericMeta):
            return dtype(self, loc=loc, ip=ip)
        elif dtype is ir.Value:
            if isinstance(self.value, ir.Value):
                return self.value
            return _emitter().const(self.value, type(self), loc=loc, ip=ip)
        elif dtype in (int, float, bool):
            if isinstance(self.value, ir.Value):
                raise DSLUserCodeError(
                    DiagId.PHASE_REQUIRES_CONSTANT,
                    what=f"`{dtype.__name__}({type(self).__name__})`",
                )
            return dtype(self.value)
        else:
            raise DSLRuntimeError(f"unable to convert {type(self)} to {dtype}")

    def ir_value(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> ir.Value:
        return self.to(ir.Value, loc=loc, ip=ip)

    def bitcast(
        self,
        dtype: "Type[Numeric]",
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        """Reinterpret the bits of this value as a different numeric type.

        The source and target types must have the same bit width.

        :param dtype: Target DSL type (e.g., ``Float32`` when self is ``Int32``).
        :return: A new instance of ``dtype`` with the same bit pattern.
        """
        if not isinstance(dtype, NumericMeta):
            raise DSLRuntimeError(f"dtype must be a Numeric type, but got {dtype}")
        if dtype is type(self):
            return self
        ir_val = self.ir_value(loc=loc, ip=ip)
        result = _emitter().bitcast(ir_val, dtype.mlir_type, loc=loc, ip=ip)
        return dtype(result)

    def __bool__(self) -> bool:
        if isinstance(self.value, (int, float, bool)):
            return bool(self.value)
        raise DSLUserCodeError(DiagId.PHASE_DYNAMIC_TO_STATIC_BOOL)

    def __index__(self) -> int:
        if isinstance(self.value, (int, float, bool)):
            return operator.index(self.value)
        raise DSLUserCodeError(DiagId.PHASE_DYNAMIC_INDEX)

    def __int__(self) -> int:
        if isinstance(self.value, (int, float, bool)):
            return int(self.value)
        raise DSLUserCodeError(DiagId.PHASE_REQUIRES_CONSTANT, what="`int()`")

    def __float__(self) -> float:
        if isinstance(self.value, (int, float, bool)):
            return float(self.value)
        raise DSLUserCodeError(DiagId.PHASE_REQUIRES_CONSTANT, what="`float()`")

    # -- Unary operators -----------------------------------------------------

    def __neg__(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        ty = type(self)
        if isinstance(self.value, ir.Value):
            signed = getattr(ty, "signed", None)
            res = _emitter().neg(self.value, signed=signed, loc=loc, ip=ip)
            return ty(res, loc=loc, ip=ip)
        return ty(-self.value)

    def __abs__(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        ty = type(self)
        if isinstance(self.value, ir.Value):
            signed = getattr(ty, "signed", None)
            res = _emitter().abs(self.value, signed=signed, loc=loc, ip=ip)
            return ty(res, loc=loc, ip=ip)
        return ty(abs(self.value))

    @staticmethod
    def _from_python_value(
        value: Union[bool, int, float, ir.Value, "Numeric"],
    ) -> "Numeric":
        """Wrap a Python literal in the dtype the literal rule assigns.

        ``bool`` -> ``Boolean``; ``int`` -> ``Int32``, or ``Int64`` when it
        does not fit in 32 bits; ``float`` -> ``Float32``; an ``ir.Value``
        takes the dtype that claims its MLIR type; a ``Numeric`` is returned
        as is.
        """
        if isinstance(value, Numeric):
            return value

        if isinstance(value, bool):
            res_type: Type["Numeric"] = Boolean
        elif isinstance(value, int):
            res_type = (
                Int32 if (value <= 2147483647) and (value >= -2147483648) else Int64
            )
        elif isinstance(value, float):
            res_type = Float32
        elif isinstance(value, ir.Value):
            res_type = Numeric.from_mlir_type(value.type)
        else:
            raise DSLUserCodeError(
                DiagId.ARG_NOT_NUMERIC,
                arg_name="value",
                arg_type=type(value).__name__,
            )
        return res_type(value)

    # -- Binary operators ----------------------------------------------------

    def __add__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.add, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __sub__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.sub, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __mul__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.mul, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __floordiv__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.floordiv, promote_bool=True)(
            self, other, loc=loc, ip=ip
        )

    def __truediv__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.truediv, promote_bool=True)(
            self, other, loc=loc, ip=ip
        )

    def __mod__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.mod, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __radd__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return self.__add__(other, loc=loc, ip=ip)

    def __rsub__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.sub, promote_bool=True, flip=True)(
            self, other, loc=loc, ip=ip
        )

    def __rmul__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return self.__mul__(other, loc=loc, ip=ip)

    def __rfloordiv__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.floordiv, promote_bool=True, flip=True)(
            self, other, loc=loc, ip=ip
        )

    def __rtruediv__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.truediv, promote_bool=True, flip=True)(
            self, other, loc=loc, ip=ip
        )

    def __rmod__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.mod, promote_bool=True, flip=True)(
            self, other, loc=loc, ip=ip
        )

    def __eq__(  # type: ignore[override]
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Boolean":
        return _binary_op(operator.eq)(self, other, loc=loc, ip=ip)

    def __ne__(  # type: ignore[override]
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Boolean":
        return _binary_op(operator.ne)(self, other, loc=loc, ip=ip)

    def __lt__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Boolean":
        return _binary_op(operator.lt)(self, other, loc=loc, ip=ip)

    def __le__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Boolean":
        return _binary_op(operator.le)(self, other, loc=loc, ip=ip)

    def __gt__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Boolean":
        return _binary_op(operator.gt)(self, other, loc=loc, ip=ip)

    def __ge__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Boolean":
        return _binary_op(operator.ge)(self, other, loc=loc, ip=ip)

    def __pow__(
        self,
        other: Union[int, float, bool, "Numeric"],
        mod: Union[int, float, bool, "Numeric", None] = None,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        if mod is not None:
            # No staged modular-exponentiation form.
            return NotImplemented
        return _binary_op(operator.pow)(self, other, loc=loc, ip=ip)

    def __rpow__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.pow, flip=True)(self, other, loc=loc, ip=ip)

    @staticmethod
    def from_mlir_type(mlir_type: ir.Type) -> Type["Numeric"]:
        """Return the dtype whose MLIR type is ``mlir_type``.

        Signless and signed integer types map to the signed dtype of that
        width, unsigned ones to the unsigned dtype.
        """
        dt = _lookup_mlir_type(mlir_type)
        if dt is None:
            raise DSLUserCodeError(
                DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(mlir_type)
            )
        return dt


def as_numeric(obj: Union[bool, int, float, ir.Value, Numeric]) -> Numeric:
    """Convert a Python primitive value to a Numeric type.

    :param obj: Python primitive value to convert
    :type obj: Union[bool, int, float, ir.Value, Numeric]
    :return: The converted Numeric object
    :rtype: Numeric

    Example::

        .. code-block:: python

            x = as_numeric(5)  # Converts to Int32
            y = as_numeric(3.14)  # Converts to Float32
            z = as_numeric(True)  # Converts to Boolean
    """
    if isinstance(obj, Numeric):
        return obj
    return Numeric._from_python_value(obj)


def _wrap_to_exact_range(value: int, exact_range: tuple) -> int:
    """Wrap ``value`` into ``exact_range`` the way a C integer cast would.

    Keeps the low ``width`` bits and reinterprets them with the range's
    signedness, e.g. ``Int32(1 << 34) -> 0``. ``(0, 1)`` is the boolean range,
    where every nonzero value folds to 1 instead of wrapping.
    """
    if exact_range == (0, 1):
        return int(bool(value))
    lo, hi = exact_range
    return (value - lo) % (hi - lo + 1) + lo


def _float_to_int(x: float, exact_range: tuple) -> int:
    """Narrow ``x`` to the integer range with a deterministic two's-complement wrap."""
    return _wrap_to_exact_range(int(x), exact_range)


# =============================================================================
# Integer / Float / Boolean
# =============================================================================


class Integer(Numeric, metaclass=IntegerMeta, mlir_type=T.i32, is_abstract=True):
    """A class representing integer values with specific width and signedness.

    :param x: The input value to convert to this integer type
    :type x: Union[bool, int, float, ir.Value, Integer, Float]

    Type conversion behavior:

    * Python scalars (bool, int, float): converted through the target dtype's
      C cast. A value whose magnitude exceeds the target width is narrowed
      and emits ``TYPE_INT_LITERAL_OUT_OF_RANGE`` for an integer literal
      (``Int8(256) -> 0``) or ``TYPE_FLOAT_TO_INT_OUT_OF_RANGE`` for a float
      one. To materialize a specific bit pattern intentionally, mask first
      (``Int8(256 & 0xFF)``).
    * MLIR Value with IntegerType: width differences handled by extension or
      truncation (``i8 -> i32``).
    * MLIR Value with FloatType: MLIR float-to-int conversion.
    * Integer: MLIR int-to-int conversion or the target dtype's C cast.
    * Float: MLIR float-to-int conversion (``Int32(Float32(5.7)) -> 5``).

    Example usage:

    .. code-block:: python

        x = Int32(5)  # From integer
        y = Int32(True)  # From boolean
        z = Int32(3.7)  # From float (truncates)
        w = Int32(x)  # From same Integer type
    """

    # Injected by IntegerMeta.__new__ on every concrete subclass.
    signed: ClassVar[bool]
    _exact_range: ClassVar[tuple]

    def __init__(
        self,
        x: Union[bool, int, float, ir.Value, "Integer", "Float"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        ty = type(self)

        if isinstance(x, (bool, int, float)):
            if isinstance(x, float) and not math.isfinite(x):
                raise DSLUserCodeError(
                    f"Cannot convert the float `{x!r}` to `{ty.__name__}`: it is "
                    "not a finite number.",
                    suggestion="Pass a finite value, or handle NaN/infinity before "
                    "converting to an integer type.",
                )

            exact_range = ty._exact_range
            if exact_range[0] <= x <= exact_range[1]:
                # Already representable: int() truncates floats toward zero and
                # folds bools to 0/1, which is what the cast does too.
                x_val = int(x)
            elif isinstance(x, float):
                x_val = _float_to_int(x, exact_range)
            else:
                x_val = _wrap_to_exact_range(int(x), exact_range)
            # A value whose truncation lands outside the target type's width is
            # silently narrowed by the cast above, losing magnitude. Surface
            # that loss as a warning, under a dedicated code per literal kind.
            # Integer literals are tested against the union of the signed and
            # unsigned ranges of that width, since a mask or flag word
            # naturally reaches a signed type as ``Int32(0xFFFFFFFF)``. Float
            # literals are tested against the type's own ``[min, max]``.
            # ``Boolean`` (width 1) has no magnitude range and is excluded.
            if ty.width > 1:
                int_val = int(x)
                if isinstance(x, float):
                    lossless_lo, lossless_hi = ty.min, ty.max
                else:
                    lossless_lo = -(1 << (ty.width - 1))
                    lossless_hi = (1 << ty.width) - 1
                if int_val < lossless_lo or int_val > lossless_hi:
                    if isinstance(x, float):
                        report_warning(
                            WarnId.TYPE_FLOAT_TO_INT_OUT_OF_RANGE,
                            stacklevel=3,
                            value=x,
                            type=ty.__name__,
                            min=ty.min,
                            max=ty.max,
                            result=x_val,
                        )
                    else:
                        report_warning(
                            WarnId.TYPE_INT_LITERAL_OUT_OF_RANGE,
                            stacklevel=3,
                            value=int_val,
                            type=ty.__name__,
                            min=ty.min,
                            max=ty.max,
                            wrapped=x_val,
                            mask=(1 << ty.width) - 1,
                        )
            else:
                x_val = bool(x_val)
        elif type(x) == ty:
            x_val = x.value  # type: ignore[assignment]
        elif isinstance(x, ir.Value):
            scalar = _emitter().scalar_type(x.type)
            if isinstance(scalar, ir.IntegerType):
                x_val = x
                if scalar.width != ty.width:
                    # signless -> (u)int
                    x_val = _emitter().int_to_int(x, ty, loc=loc, ip=ip)
            elif isinstance(scalar, ir.FloatType):
                # float -> (u)int
                x_val = _emitter().fptoi(x, ty.signed, ty.mlir_type, loc=loc, ip=ip)
            else:
                raise DSLUserCodeError(
                    DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(x.type)
                )
        elif isinstance(x, Integer):
            if isinstance(x.value, ir.Value):
                x_val = _emitter().int_to_int(
                    x.value, ty, src_signed=type(x).signed, loc=loc, ip=ip
                )
            else:
                # For non-MLIR values, wrap the way the target's C cast would.
                x_val = _wrap_to_exact_range(int(x.value), ty._exact_range)
                if ty.width == 1:
                    x_val = bool(x_val)
        elif isinstance(x, Float):
            # float -> int is handled by Integer.__init__ recursively
            Integer.__init__(self, x.value, loc=loc, ip=ip)
            return
        else:
            raise DSLUserCodeError(
                DiagId.ARG_NOT_NUMERIC,
                arg_name=f"{ty.__name__}(...)",
                arg_type=type(x).__name__,
            )

        super().__init__(x_val)

    def __invert__(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Integer":
        res_type = type(self)
        if isinstance(self.value, ir.Value):
            all_ones = _emitter().const(-1, res_type, loc=loc, ip=ip)
            res = _emitter().xor(self.value, all_ones, loc=loc, ip=ip)
            return res_type(res, loc=loc, ip=ip)
        return res_type(~int(self.value))

    def __lshift__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.lshift)(self, other, loc=loc, ip=ip)

    def __rlshift__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            return NotImplemented
        return other_.__lshift__(self, loc=loc, ip=ip)  # type: ignore[call-arg]

    def __rshift__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.rshift)(self, other, loc=loc, ip=ip)

    def __rrshift__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            return NotImplemented
        return other_.__rshift__(self, loc=loc, ip=ip)  # type: ignore[call-arg]

    def __and__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.and_)(self, other, loc=loc, ip=ip)

    def __rand__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return self.__and__(other, loc=loc, ip=ip)  # type: ignore[call-arg]

    def __or__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.or_)(self, other, loc=loc, ip=ip)

    def __ror__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return self.__or__(other, loc=loc, ip=ip)  # type: ignore[call-arg]

    def __xor__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return _binary_op(operator.xor)(self, other, loc=loc, ip=ip)

    def __rxor__(
        self,
        other: Union[int, float, bool, "Numeric"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        return self.__xor__(other, loc=loc, ip=ip)  # type: ignore[call-arg]


# Standard IEEE-754 binary formats keyed by DSL type name, mapping to
# (``struct`` format code, max finite magnitude, smallest positive subnormal).
# Used by ``Float.__init__`` to detect a Python-float literal that overflows to
# +/-inf or underflows to 0 in the target type -- numpy-free, via exact
# ``struct`` round-trips. Only these standard formats are probed; non-IEEE /
# narrow types (bf16, tf32, fp8, fp6, fp4) are absent and narrow later in IR.
_IEEE_FLOAT_PROBE: dict = {
    "Float32": ("f", 3.4028234663852886e38, 2.0**-149),
}


class Float(Numeric, metaclass=FloatMeta, mlir_type=T.f32, is_abstract=True):
    """A class representing floating-point values.

    :param x: The input value to convert to this float type.
    :type x: Union[bool, int, float, ir.Value, Integer, Float]

    Type conversion behavior:

    1. Python scalars (bool, int, float): kept at full Python-float precision
       and narrowed during IR emission. A value whose magnitude cannot be
       represented in the target type collapses to +/-inf (overflow) or 0
       (underflow) and emits a ``TYPE_FLOAT_LITERAL_OVERFLOW`` /
       ``TYPE_FLOAT_LITERAL_UNDERFLOW`` warning.
    2. MLIR Value with FloatType: converted between float types when the
       width differs.
    3. MLIR Value with IntegerType: MLIR int-to-float conversion.
    4. Integer: MLIR int-to-float conversion (``Float32(Int32(5)) -> 5.0``).
    5. Float: direct conversion between float types.
    """

    def __init__(
        self,
        x: Union[bool, int, float, ir.Value, "Integer", "Float"],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        ty = type(self)

        if isinstance(x, (bool, int, float)):
            fx = float(x)
            # A finite, nonzero Python float whose magnitude cannot be
            # represented in the target type collapses to +/-inf (overflow) or
            # 0 (underflow) once narrowed, losing the value entirely. Surface
            # that catastrophic loss, but NOT ordinary rounding loss, which is
            # inherent to every float literal. The full-precision Python
            # double is kept; the probe only detects the collapse.
            struct_code, max_finite, min_subnormal = _IEEE_FLOAT_PROBE.get(
                ty.__name__, (None, None, None)
            )
            if struct_code is not None and fx != 0.0 and math.isfinite(fx):
                try:
                    narrowed = _pystruct.unpack(
                        struct_code, _pystruct.pack(struct_code, fx)
                    )[0]
                except OverflowError:
                    # binary16 pack raises rather than saturating to inf.
                    narrowed = math.copysign(math.inf, fx)
                if math.isinf(narrowed):
                    report_warning(
                        WarnId.TYPE_FLOAT_LITERAL_OVERFLOW,
                        stacklevel=3,
                        value=fx,
                        type=ty.__name__,
                        max=max_finite,
                        wrapped=narrowed,
                    )
                elif narrowed == 0.0:
                    report_warning(
                        WarnId.TYPE_FLOAT_LITERAL_UNDERFLOW,
                        stacklevel=3,
                        value=fx,
                        type=ty.__name__,
                        tiny=min_subnormal,
                        wrapped=narrowed,
                    )
            super().__init__(fx)
        elif isinstance(x, ir.Value):
            scalar = _emitter().scalar_type(x.type)
            if isinstance(scalar, ir.IntegerType):
                x = _emitter().itofp(x, None, ty.mlir_type, loc=loc, ip=ip)
            elif isinstance(scalar, ir.FloatType):
                if scalar != ty.scalar_mlir_type:
                    x = _emitter().cvtf(x, ty.mlir_type, loc=loc, ip=ip)
            else:
                raise DSLUserCodeError(
                    DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(x.type)
                )
            super().__init__(x)
        elif isinstance(x, Integer):
            if isinstance(x.value, ir.Value):
                x = _emitter().itofp(
                    x.value, type(x).signed, ty.mlir_type, loc=loc, ip=ip
                )
            else:
                x = float(x.value)
            super().__init__(x)
        elif isinstance(x, Float):
            Float.__init__(self, x.value, loc=loc, ip=ip)
        else:
            raise DSLUserCodeError(
                DiagId.ARG_NOT_NUMERIC,
                arg_name=f"{ty.__name__}(...)",
                arg_type=type(x).__name__,
            )


class Boolean(Integer, metaclass=IntegerMeta, width=1, signed=True, mlir_type=T.bool):
    """Boolean type representation in the DSL.

    This class represents boolean values in the DSL, with a width of 1 bit.

    :param a: Value to convert to Boolean
    :type a: Union[bool, int, float, ir.Value, Numeric]

    Conversion rules:

    1. Python bool/int/float: converted using Python's bool() function.
    2. Numeric: uses the Numeric.value to construct Boolean recursively.
    3. MLIR Value with IntegerType: direct when its width is 1, otherwise
       compared with 0 through the type ops' ``cmp("ne")``.
    4. MLIR Value with FloatType: compared with 0.0 through the type ops'
       ``cmp("ne")`` (unordered under the builtin arith type ops, so NaN is
       true there).
    """

    def __init__(
        self,
        a: Union[bool, int, float, ir.Value, Numeric],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        if isinstance(a, (bool, int, float)):
            value: Union[bool, ir.Value] = bool(a)
        elif isinstance(a, Numeric):
            Boolean.__init__(self, a.value, loc=loc, ip=ip)
            return
        elif isinstance(a, ir.Value):
            if _emitter().scalar_type(a.type) == T.bool():
                value = a
            else:
                dtype = _lookup_mlir_type(a.type)
                if dtype is None:
                    raise DSLUserCodeError(
                        DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(a.type)
                    )
                zero = _emitter().const(0, dtype, loc=loc, ip=ip)
                value = _emitter().cmp("ne", a, zero, loc=loc, ip=ip)
        else:
            raise DSLUserCodeError(
                DiagId.ARG_NOT_NUMERIC,
                arg_name="Boolean(...)",
                arg_type=type(a).__name__,
            )
        super().__init__(value, loc=loc, ip=ip)

    def __neg__(  # type: ignore[override]
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Numeric":
        """Negation is not supported for the boolean type."""
        raise DSLUserCodeError(
            "The operator `-` is not supported for `Boolean`.",
            suggestion="Use `not x` for the logical negation, or convert first, "
            "e.g. `-Int32(x)`.",
        )


# =============================================================================
# Scalar dtypes
# =============================================================================


class Int32(Integer, metaclass=IntegerMeta, width=32, signed=True, mlir_type=T.i32):
    ...


class Int64(Integer, metaclass=IntegerMeta, width=64, signed=True, mlir_type=T.i64):
    ...


class Float32(
    Float,
    metaclass=FloatMeta,
    width=32,
    mlir_type=T.f32,
    exponent_width=8,
    mantissa_width=23,
    np_dtype_name="float32",
    ctype=ctypes.c_float,
):
    ...


# =============================================================================
# dtype lookup
# =============================================================================


def _mlir_type_index() -> dict:
    """The MLIR type spelling -> dtype index, built on first use in a context."""
    if not _MLIR_NAME_TO_DTYPE:
        for dt in ALL_DTYPES:
            if dt.is_integer:
                width = dt.width
                if dt.signed:
                    _MLIR_NAME_TO_DTYPE[f"i{width}"] = dt
                    _MLIR_NAME_TO_DTYPE[f"si{width}"] = dt
                else:
                    _MLIR_NAME_TO_DTYPE[f"ui{width}"] = dt
            else:
                _MLIR_NAME_TO_DTYPE[str(dt.mlir_type)] = dt
    return _MLIR_NAME_TO_DTYPE


def _lookup_scalar_type(scalar: ir.Type) -> Optional[Type[Numeric]]:
    """Return the dtype of a scalar MLIR type, or None when no dtype claims it."""
    with scalar.context:
        index = _mlir_type_index()
    return index.get(str(scalar))


def _lookup_mlir_type(mlir_type: ir.Type) -> Optional[Type[Numeric]]:
    """Return the dtype whose SSA type ``mlir_type`` is under the active
    type_ops plugin, or None when no dtype claims it."""
    scalar = _emitter().scalar_type(mlir_type)
    if scalar is None:
        return None
    return _lookup_scalar_type(scalar)


__all__ = [
    "DslType",
    "Numeric",
    "NumericMeta",
    "IntegerMeta",
    "FloatMeta",
    "Boolean",
    "Integer",
    "Int32",
    "Int64",
    "Float",
    "Float32",
    "ALL_DTYPES",
    "as_numeric",
    "cast",
]
