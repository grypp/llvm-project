# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fixed one-dimensional SSA vectors: the ``Vector`` leaf over ``vector<N x T>``.

A ``Vector`` wraps one ``vector<N x T>`` value whose element type is a
``Numeric`` dtype from the type registry. Vectors live in registers, are
immutable and have no Python invocation ABI: extract a lane or reduce to
return a scalar from a JIT entry point. The SSA type and every op come from
the active dialect's emitter (``vector_type``, ``from_elements``,
``broadcast``, ``extract``, ``reduce``, and the element-wise arithmetic on the
vector-typed operands): ``vector<N x T>`` and the ``vector`` dialect under the
LLVM world, a rank-1 tile under a tile dialect. ``util/tree_utils.py``
registers ``Vector`` as a leaf at import, so a raw vector result
(``Pointer.load(count=N)``, a dialect call) is wrapped by ``wrap_ir_value``.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Type, Union

from ... import ir
from ..core.common import DSLRuntimeError, DSLUserCodeError
from ..core.diagnostics import DiagId
from ..core.mlir_op import current_emitter as _emitter
from ..core.user_op import dsl_user_op
from .typing import (
    Numeric,
    NumericMeta,
    _lookup_mlir_type,
    _lookup_scalar_type,
    as_numeric,
)

__all__ = ["Vector"]


# =============================================================================
# Scalar coercion helpers
# =============================================================================


def _scalar(value: Any, arg_name: str = "elements") -> Numeric:
    """Return ``value`` as a ``Numeric``.

    A raw scalar ``ir.Value`` takes the registry dtype of its type, a Python
    literal follows the literal rule (``bool`` -> ``Boolean``, ``int`` ->
    ``Int32``/``Int64``, ``float`` -> ``Float32``).

    :param value: A ``Numeric``, a Python scalar or a scalar ``ir.Value``.
    :param arg_name: The argument name reported by ``ARG_NOT_NUMERIC``.
    :return: The ``Numeric`` view of ``value``.
    """
    if isinstance(value, ir.Value):
        dtype = _lookup_mlir_type(value.type)  # the dialect's notion of a scalar
        if dtype is not None:
            return dtype(value)
    if isinstance(value, (bool, int, float, Numeric)):
        return as_numeric(value)
    arg_type = str(value.type) if isinstance(value, ir.Value) else type(value).__name__
    raise DSLUserCodeError(DiagId.ARG_NOT_NUMERIC, arg_name=arg_name, arg_type=arg_type)


def _coerce_scalar(value: Any, dtype: Type[Numeric], op: str) -> Numeric:
    """Return ``value`` as an instance of exactly ``dtype``.

    A literal is built at ``dtype`` (a ``float`` into an integer dtype is
    rejected), a typed scalar must already have it: the DSL picks no common
    type for vector lanes, so a mismatch is
    ``TYPE_IMPLICIT_PROMOTION_UNSUPPORTED``.

    :param value: A Python literal, a ``Numeric`` or a scalar ``ir.Value``.
    :param dtype: The element dtype of the vector.
    :param op: The operation named in the diagnostic (``"+"``, ``"Vector([...])"``).
    :return: A ``Numeric`` of type ``dtype``.
    """
    if isinstance(value, (bool, int, float)):
        rhs_type = "float" if isinstance(value, float) and not dtype.is_float else None
    else:
        value = _scalar(value)
        rhs_type = None if type(value) is dtype else type(value).__name__
    if rhs_type is not None:
        raise DSLUserCodeError(
            DiagId.TYPE_IMPLICIT_PROMOTION_UNSUPPORTED,
            lhs_type=dtype.__name__,
            rhs_type=rhs_type,
            op=op,
        )
    return value if isinstance(value, Numeric) else dtype(value)


def _static_int(
    value: Any, arg_name: str, num: int, expected: str, ok: Callable[[int], bool]
) -> int:
    """Return ``value`` as a compile-time Python ``int`` satisfying ``ok``.

    A ``Numeric`` with a Python payload is unwrapped; a staged one is
    ``PHASE_DYNAMIC_INDEX``; anything else that is not an ``int`` passing
    ``ok`` is ``ARG_ANNOTATION_MISMATCH``.

    :param value: The lane count or lane index given by the user.
    :param arg_name: The argument name reported in the diagnostic.
    :param num: The 1-based argument position reported in the diagnostic.
    :param expected: The description of the accepted values.
    :param ok: The predicate a valid ``int`` must satisfy.
    :return: The validated ``int``.
    """
    if isinstance(value, Numeric):
        if isinstance(value.value, ir.Value):
            raise DSLUserCodeError(DiagId.PHASE_DYNAMIC_INDEX)
        value = value.value
    if type(value) is not int or not ok(value):
        got = repr(value) if type(value) is int else type(value).__name__
        raise DSLUserCodeError(
            DiagId.ARG_ANNOTATION_MISMATCH,
            num=num,
            arg_name=arg_name,
            expected=expected,
            got=got,
        )
    return value


def _vector_dtype(value: ir.Value, dtype: Optional[Type[Numeric]]) -> Type[Numeric]:
    """Check that ``value`` is a fixed 1-D vector and return its element dtype.

    :param value: The SSA value to wrap.
    :param dtype: The expected element dtype, or None to read it from the type.
    :return: The element dtype class.
    """
    ty = value.type
    shape = _emitter().vector_shape(ty)
    if shape is None:
        raise DSLUserCodeError(DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(ty))
    elem, _ = shape
    if dtype is None:
        found = _lookup_scalar_type(elem)
        if found is None:
            raise DSLUserCodeError(DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(ty))
        return found
    if not isinstance(dtype, NumericMeta) or dtype.scalar_mlir_type != elem:
        raise DSLRuntimeError(f"Vector dtype {dtype} does not match {ty}")
    return dtype


# =============================================================================
# The Vector
# =============================================================================


class Vector:
    """An immutable, nonempty vector with a fixed lane count and element dtype.

    ``Vector([x, y])`` builds lanes with the emitter's ``from_elements`` (a typed lane
    fixes the dtype of untyped literals, so ``[x, 0]`` works for integer and
    float ``x``); ``Vector.splat(x, 4)`` broadcasts one scalar; ``Vector(v)``
    wraps an existing ``vector<N x T>`` SSA value. Arithmetic requires equal
    vector types: Python literals broadcast at the element dtype, typed scalars
    must have that dtype. ``v[i]`` needs a compile-time lane and ``v.sum()``
    returns a scalar of the element dtype.

    :param v: Lane values (literals, ``Numeric``s, scalar ``ir.Value``s) or one
        ``vector<N x T>`` SSA value to wrap.
    :param dtype: Element dtype; inferred from the lanes or the MLIR type when omitted.
    """

    __slots__ = ("_value", "_dtype")

    @dsl_user_op
    def __init__(
        self,
        v: Union[Sequence[Any], ir.Value],
        *,
        dtype: Optional[Type[Numeric]] = None,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        if not isinstance(v, ir.Value):
            if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
                raise DSLUserCodeError(
                    DiagId.ARG_NOT_NUMERIC,
                    arg_name="elements",
                    arg_type=type(v).__name__,
                )
            if len(v) == 0:
                raise DSLUserCodeError(
                    DiagId.CALL_MISSING_ARG,
                    function_name="Vector",
                    missing="at least one element",
                )
            if dtype is None:
                typed = [x for x in v if isinstance(x, (Numeric, ir.Value))]
                dtype = type(_scalar(typed[0] if typed else v[0]))
            lanes = [_coerce_scalar(x, dtype, "Vector([...])") for x in v]
            vec_type = _emitter().vector_type(dtype, len(lanes))
            elements = [x.ir_value(loc=loc, ip=ip) for x in lanes]
            v = _emitter().from_elements(vec_type, elements, loc=loc, ip=ip)
        self._dtype = _vector_dtype(v, dtype)
        self._value = v

    @classmethod
    def from_ir(
        cls, value: ir.Value, *, dtype: Optional[Type[Numeric]] = None
    ) -> "Vector":
        """Wrap an existing fixed 1-D vector SSA value without emitting an op.

        :param value: A ``vector<N x T>`` SSA value.
        :param dtype: Its element dtype; read from the type when omitted.
        :return: The ``Vector`` view of ``value``.
        """
        result = object.__new__(cls)
        result._dtype = _vector_dtype(value, dtype)
        result._value = value
        return result

    @classmethod
    @dsl_user_op
    def splat(
        cls,
        value: Any,
        lanes: Any,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Vector":
        """Broadcast a scalar to a positive, compile-time-known number of lanes.

        :param value: The scalar to broadcast (a literal, ``Numeric`` or scalar ``ir.Value``).
        :param lanes: The lane count, a positive Python ``int``.
        :return: A ``Vector`` of ``lanes`` copies of ``value`` (the emitter's ``broadcast``).
        """
        if isinstance(lanes, Numeric) and isinstance(lanes.value, ir.Value):
            raise DSLUserCodeError(
                DiagId.PHASE_REQUIRES_CONSTANT, what="The lane count of `Vector.splat`"
            )
        lanes = _static_int(
            lanes, "lanes", 2, "a positive Python `int`", lambda n: n > 0
        )
        value = _scalar(value, "value")
        vec_type = _emitter().vector_type(type(value), lanes)
        result = _emitter().broadcast(
            vec_type, value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
        return cls.from_ir(result, dtype=type(value))

    @property
    def value(self) -> ir.Value:
        """The underlying ``vector<N x T>`` SSA value."""
        return self._value

    def ir_value(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> ir.Value:
        """The underlying SSA value (the leaf protocol's accessor; emits nothing)."""
        return self._value

    @property
    def dtype(self) -> Type[Numeric]:
        """The element dtype (e.g. ``Float32``, ``Int32``)."""
        return self._dtype

    @property
    def mlir_type(self) -> ir.Type:
        """The SSA type of the value (``vector<N x T>`` under the LLVM world)."""
        return self._value.type

    @property
    def lanes(self) -> int:
        """The number of lanes ``N``."""
        return _emitter().vector_shape(self._value.type)[1]

    def __len__(self) -> int:
        return self.lanes

    def __iter__(self):
        """Iterate over the lanes, extracting each as a scalar ``Numeric``."""
        return (self[i] for i in range(self.lanes))

    def __bool__(self) -> bool:
        """A vector has no static truth value: ``PHASE_DYNAMIC_TO_STATIC_BOOL``."""
        raise DSLUserCodeError(DiagId.PHASE_DYNAMIC_TO_STATIC_BOOL)

    @dsl_user_op
    def __getitem__(
        self,
        lane: Any,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Numeric:
        """Extract one lane at a compile-time index (the emitter's ``extract``).

        :param lane: A Python ``int`` in ``range(-N, N)``; negative indices count from the end.
        :return: The lane as a scalar of the element dtype.
        """
        n = self.lanes
        lane = _static_int(
            lane, "lane", 1, f"a lane index in range({n})", lambda i: -n <= i < n
        )
        lane = lane + n if lane < 0 else lane
        return self._dtype(_emitter().extract(self._value, lane, loc=loc, ip=ip))

    @dsl_user_op
    def sum(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Numeric:
        """Reduce all lanes to one scalar (the emitter's ``reduce("add")``).

        :return: The sum of the lanes as a scalar of the element dtype.
        """
        result = _emitter().reduce("add", self._value, loc=loc, ip=ip)
        return self._dtype(result)

    def _wrap_like(self, result_ir: ir.Value) -> "Vector":
        """Wrap an element-wise result; a changed element type (integer ``/``
        yields ``f32``) re-resolves the dtype from the registry."""
        keep = (
            _emitter().vector_shape(result_ir.type)[0] == self._dtype.scalar_mlir_type
        )
        return Vector.from_ir(result_ir, dtype=self._dtype if keep else None)

    def _binary(
        self,
        op: Callable[..., ir.Value],
        symbol: str,
        other: Any,
        *,
        reverse: bool = False,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "Vector":
        """Emit ``op`` on equal vector types.

        A scalar ``other`` is splat first (a literal through ``const`` then
        ``broadcast``, a typed scalar of the element dtype through ``broadcast``). A ``Vector`` of another type (lane count or
        dtype) is ``TYPE_IMPLICIT_PROMOTION_UNSUPPORTED``.

        :param op: The emitter op to emit (an ``OpEmitter`` method such as ``add``).
        :param symbol: The operator spelling reported in diagnostics.
        :param other: The right operand (or the left one when ``reverse``).
        :param reverse: Whether ``other`` is the left operand (``__rsub__``).
        :return: The element-wise result.
        """
        signed = getattr(self._dtype, "signed", None)
        if isinstance(other, (bool, int, float)):
            _coerce_scalar(other, self._dtype, symbol)
            scalar = _emitter().const(other, self._dtype, signed=signed, loc=loc, ip=ip)
            splat = _emitter().broadcast(self.mlir_type, scalar, loc=loc, ip=ip)
            other = Vector.from_ir(splat, dtype=self._dtype)
        elif not isinstance(other, Vector):
            scalar = _coerce_scalar(_scalar(other, "other"), self._dtype, symbol)
            scalar_ir = scalar.ir_value(loc=loc, ip=ip)
            splat = _emitter().broadcast(self.mlir_type, scalar_ir, loc=loc, ip=ip)
            other = Vector.from_ir(splat, dtype=self._dtype)
        if other.mlir_type != self.mlir_type:
            raise DSLUserCodeError(
                DiagId.TYPE_IMPLICIT_PROMOTION_UNSUPPORTED,
                lhs_type=str(self.mlir_type),
                rhs_type=str(other.mlir_type),
                op=symbol,
            )
        lhs, rhs = (other, self) if reverse else (self, other)
        return self._wrap_like(
            op(lhs._value, rhs._value, signed=signed, loc=loc, ip=ip)
        )

    @dsl_user_op
    def __add__(self, other: Any, *, loc: Any = None, ip: Any = None) -> "Vector":
        return self._binary(_emitter().add, "+", other, loc=loc, ip=ip)

    __radd__ = __add__

    @dsl_user_op
    def __sub__(self, other: Any, *, loc: Any = None, ip: Any = None) -> "Vector":
        return self._binary(_emitter().sub, "-", other, loc=loc, ip=ip)

    @dsl_user_op
    def __rsub__(self, other: Any, *, loc: Any = None, ip: Any = None) -> "Vector":
        return self._binary(_emitter().sub, "-", other, reverse=True, loc=loc, ip=ip)

    @dsl_user_op
    def __mul__(self, other: Any, *, loc: Any = None, ip: Any = None) -> "Vector":
        return self._binary(_emitter().mul, "*", other, loc=loc, ip=ip)

    __rmul__ = __mul__

    @dsl_user_op
    def __truediv__(self, other: Any, *, loc: Any = None, ip: Any = None) -> "Vector":
        return self._binary(_emitter().truediv, "/", other, loc=loc, ip=ip)

    @dsl_user_op
    def __rtruediv__(self, other: Any, *, loc: Any = None, ip: Any = None) -> "Vector":
        return self._binary(
            _emitter().truediv, "/", other, reverse=True, loc=loc, ip=ip
        )
