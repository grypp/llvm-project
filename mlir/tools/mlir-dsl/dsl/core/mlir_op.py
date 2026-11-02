# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The hook protocol behind the type system, implemented by the ``type_ops`` plugin.

``Int32(6) + a``, ``Vector([a, b]) * v`` and ``p[i]`` are the same program in
every DSL built on ``mlir.dsl``; which IR they become is the business of the
tracing DSL's ``type_ops`` plugin (``core.plugin.TypeOpsPlugin``, an
:class:`OpEmitter`). An :class:`OpEmitter` answers, for the core types, what
SSA type a value has and which op implements each operation:

* scalars: ``mlir_type``/``scalar_type``, ``infer_dtype`` (the dtype a Python
  number takes under an abstract annotation) and the arithmetic, comparison
  and conversion ops of ``Numeric``;
* vectors: ``vector_type``/``vector_shape`` and ``from_elements``,
  ``broadcast``, ``extract``, ``reduce``;
* pointers: ``pointer_type``/``pointer_space`` and ``load``, ``store``,
  ``ptr_add``, ``inttoptr``, ``ptrtoint``, ``addrspacecast``.

The types call :func:`current_emitter`: the ``type_ops`` plugin of the
tracing DSL (the ``plugins/type_ops`` composer in the test DSL); a DSL that names
none cannot use the types in a trace. The default type hooks describe MLIR's
builtin types (``i32``, ``vector<N x T>``); a type-ops plugin over a tile
dialect answers rank-0 and rank-1 tiles instead. The op signatures are those
of ``plugins/type_ops/arith.py`` (operands as ``ir.Value``, ``signed=``,
``loc=``, ``ip=``); an op a plugin lacks raises ``DSLRuntimeError``.
"""

from __future__ import annotations

from typing import Any

from ... import ir
from .common import DSLRuntimeError, DSLUserCodeError, get_current_dsl
from .diagnostics import DiagId

__all__ = ["OpEmitter", "current_emitter"]


class OpEmitter:
    """The hook protocol a ``type_ops`` plugin implements for the core types."""

    def _unsupported(self, name: str) -> Any:
        raise DSLRuntimeError(f"{type(self).__name__} does not emit `{name}`")

    # -- types ---------------------------------------------------------------

    def mlir_type(self, dtype: Any) -> ir.Type:
        """The SSA type of a scalar dtype; the dtype's scalar type by default."""
        return dtype.scalar_mlir_type

    def scalar_type(self, mlir_type: ir.Type) -> ir.Type | None:
        """The scalar an SSA type of this dialect carries (``i32`` for a rank-0
        ``i32`` tile), or None when it is not a scalar of the dialect. By
        default a scalar is its own scalar type and shaped types are not."""
        if isinstance(mlir_type, ir.ShapedType):
            return None
        return mlir_type

    def infer_dtype(self, value: Any, family: Any) -> Any:
        """The concrete dtype a Python ``bool``, ``int`` or ``float`` takes
        where the program asks for an abstract ``family`` (``Numeric``,
        ``Integer``, ``Float``): a ``@jit`` or kernel parameter, or a frozen
        record's field, annotated with the family and given a Python number.
        The core asks for those three Python types only and checks that the
        answer is a concrete dtype of the family. None assigns no dtype and
        makes the value an ``ARG_ANNOTATION_MISMATCH`` at the boundary: the
        default, so a plugin that does not decide rejects Python numbers
        there; a DSL value of the family is taken as it is either way."""
        return None

    # -- scalar ops (the ``Numeric`` operators) ------------------------------

    def const(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("const")

    def add(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("add")

    def sub(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("sub")

    def mul(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("mul")

    def truediv(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("truediv")

    def floordiv(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("floordiv")

    def mod(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("mod")

    def pow(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("pow")

    def neg(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("neg")

    def abs(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("abs")

    def and_(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("and_")

    def or_(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("or_")

    def xor(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("xor")

    def shl(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("shl")

    def shr(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("shr")

    def cmp(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("cmp")

    def select(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("select")

    def cast(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("cast")

    def bitcast(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("bitcast")

    def cvtf(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("cvtf")

    def fptoi(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("fptoi")

    def itofp(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("itofp")

    def int_to_int(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("int_to_int")

    def minmax(self, *args: Any, **kwargs: Any) -> Any:
        return self._unsupported("minmax")


def current_emitter() -> OpEmitter:
    """The ``type_ops`` plugin of the tracing DSL."""
    dsl = get_current_dsl()
    if dsl is None:
        # Plain Python: no DSL is tracing, so there is no dialect to ask.
        raise DSLUserCodeError(
            DiagId.CALL_OUTSIDE_JIT,
            api="a DSL type's MLIR type or op",
            decorator="@jit",
        )
    emitter = getattr(dsl.plugins, "type_ops", None)
    if emitter is None:
        raise DSLRuntimeError(
            "the core types need a `type_ops` plugin, and the tracing DSL names "
            "none (`plugins = Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith, ...))`)",
            context={"dsl": dsl.name, **dsl._unavailable_context()},
        )
    return emitter
