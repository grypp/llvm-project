# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``type_ops`` plugin over MLIR's upstream dialects.

:class:`UpstreamDialectTypeOps` is the shipped ``TypeOpsPlugin``. It has no
dialect of its own: it routes each hook group to an op module emitting ops of
an in-tree MLIR dialect, ``UpstreamDialectTypeOps(scalars=arith,
vectors=vector, memory=llvm)`` in the test DSL. The scalar operators of
``Numeric`` go to the ``scalars`` module (``arith.py`` beside this file, with
``math`` behind pow, abs and floor), the ``Vector`` ops to ``vectors``
(``vector.py``) and the ``Pointer`` types, memory ops and pointer casts to
``memory`` (``llvm.py``). Every part is optional: a part left None means the
DSL has no such values, and the first use of one raises
``CALL_PLUGIN_REQUIRED`` naming the missing part (``Int32`` stays a type,
``a + b`` is the error; a ``Vector`` or a ``Pointer`` is the error). Any module
offering the same functions can take a part's place; a module over ``memref``
would be the natural ``memory`` alternative, none ships here. A DSL over
another representation (tiles, tensors) writes its own ``TypeOpsPlugin``
instead, or subclasses this one as ``examples/14_type_ops_plugin.py`` does.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from ...core.common import DSLUserCodeError
from ...core.diagnostics import DiagId
from ...core.plugin import TypeOpsPlugin
from ...types.typing import Float, Float32, Integer, as_numeric

__all__ = ["UpstreamDialectTypeOps"]

# The shipped module of each part, named in the diagnostic of a missing one.
_SHIPPED = {"scalars": "arith", "vectors": "vector", "memory": "llvm"}


class UpstreamDialectTypeOps(TypeOpsPlugin):
    """Scalars, vectors and pointers over three dialect modules.

    The scalar type hooks (``mlir_type``, ``scalar_type``) are the
    ``OpEmitter`` defaults, the plain MLIR scalar types (``i32``, ``f32``):
    they hold with or without a ``scalars`` module, so a DSL without scalar
    operators still passes an ``Int32`` argument; so does ``infer_dtype``, the
    builtin literal rule for a Python number under an abstract annotation. The
    vector and pointer type hooks and every op hook belong to their part;
    without it they raise.

    :param scalars: The module behind the scalar operators (``arith.py`` beside
        this file),
        or None for a DSL without them
    :param vectors: The module behind the ``Vector`` ops (``vector.py``),
        or None for a DSL without vectors
    :param memory: The module behind the ``Pointer`` types, memory ops and
        pointer casts (``llvm.py``), or None for a DSL without pointers
    """

    name = ""

    def __init__(
        self,
        *,
        scalars: ModuleType | None = None,
        vectors: ModuleType | None = None,
        memory: ModuleType | None = None,
    ) -> None:
        self._scalars = scalars
        self._vectors = vectors
        self._memory = memory
        parts = [
            getattr(module, "__name__", "?").rsplit(".", 1)[-1]
            for module in (scalars, vectors, memory)
            if module is not None
        ]
        self.name = type(self).name or ("+".join(parts) if parts else "type_ops")

    def _part(self, group: str, what: str) -> ModuleType:
        """The module of ``group``; ``CALL_PLUGIN_REQUIRED`` when the DSL has none."""
        module = getattr(self, f"_{group}")
        if module is None:
            raise DSLUserCodeError(
                DiagId.CALL_PLUGIN_REQUIRED,
                name=what,
                plugin=f"a `{group}` module in its type ops",
                fix=f"UpstreamDialectTypeOps(..., {group}=mlir.dsl.plugins.type_ops.{_SHIPPED[group]})",
            )
        return module

    # -- types ---------------------------------------------------------------

    def infer_dtype(self, value: Any, family: Any) -> Any:
        """The builtin literal rule, narrowed to the family. Under ``Numeric``
        a Python number takes the dtype a literal takes in a body: a ``bool``
        is ``Boolean``, an ``int`` ``Int32`` (``Int64`` when it does not fit),
        a ``float`` ``Float32``. Under ``Float`` every number is ``Float32``;
        under ``Integer`` a ``float`` gets no dtype (None), since no integer
        holds its fraction. A type hook: it holds without a ``scalars`` module."""
        if issubclass(family, Float):
            return Float32
        dtype = as_numeric(value).dtype
        if issubclass(family, Integer) and not dtype.is_integer:
            return None
        return dtype

    # -- scalar ops ----------------------------------------------------------

    def const(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "const").const(*args, **kwargs)

    def add(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "add").add(*args, **kwargs)

    def sub(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "sub").sub(*args, **kwargs)

    def mul(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "mul").mul(*args, **kwargs)

    def truediv(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "truediv").truediv(*args, **kwargs)

    def floordiv(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "floordiv").floordiv(*args, **kwargs)

    def mod(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "mod").mod(*args, **kwargs)

    def pow(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "pow").pow(*args, **kwargs)

    def neg(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "neg").neg(*args, **kwargs)

    def abs(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "abs").abs(*args, **kwargs)

    def and_(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "and_").and_(*args, **kwargs)

    def or_(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "or_").or_(*args, **kwargs)

    def xor(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "xor").xor(*args, **kwargs)

    def shl(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "shl").shl(*args, **kwargs)

    def shr(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "shr").shr(*args, **kwargs)

    def cmp(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "cmp").cmp(*args, **kwargs)

    def cast(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "cast").cast(*args, **kwargs)

    def bitcast(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "bitcast").bitcast(*args, **kwargs)

    def cvtf(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "cvtf").cvtf(*args, **kwargs)

    def fptoi(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "fptoi").fptoi(*args, **kwargs)

    def itofp(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "itofp").itofp(*args, **kwargs)

    def int_to_int(self, *args: Any, **kwargs: Any) -> Any:
        return self._part("scalars", "int_to_int").int_to_int(*args, **kwargs)
