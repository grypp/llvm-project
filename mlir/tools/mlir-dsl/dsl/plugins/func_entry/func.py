# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``func`` entry plugin: a ``func.func`` with the C interface whose results
travel back to the host in one ``!llvm.struct`` read through ``ctypes``."""

from __future__ import annotations

import ctypes
from typing import Any

from .... import ir
from ....dialects import func, llvm
from ...core.common import DSLUserCodeError
from ...core.diagnostics import DiagId
from ...core.plugin import FuncEntryPlugin
from ...types import typing as _t

__all__ = ["Entry"]


def _float_from_bits(bits: int, exponent_width: int, mantissa_width: int) -> float:
    """Decode an IEEE-style binary float held as an integer bit pattern."""
    sign = -1.0 if bits >> (exponent_width + mantissa_width) & 1 else 1.0
    exponent = (bits >> mantissa_width) & ((1 << exponent_width) - 1)
    mantissa = bits & ((1 << mantissa_width) - 1)
    bias = (1 << (exponent_width - 1)) - 1
    if exponent == (1 << exponent_width) - 1:
        return sign * (float("nan") if mantissa else float("inf"))
    if exponent == 0:
        return sign * mantissa * 2.0 ** (1 - bias - mantissa_width)
    return sign * (1.0 + mantissa / (1 << mantissa_width)) * 2.0 ** (exponent - bias)


def _result_ctype(prototypes: list[Any]) -> type:
    """The ``ctypes`` type of the result slot for the given result dtypes."""
    ctypes_types: list[type] = []
    for prototype in prototypes:
        if prototype.ctype is None:
            raise DSLUserCodeError(
                DiagId.TYPE_RETURN_MISMATCH,
                got=f"a `{prototype.__name__}`",
                detail=" (that type has no host representation)",
            )
        ctypes_types.append(prototype.ctype)
    if len(ctypes_types) == 1:
        return ctypes_types[0]
    return type(
        "_Result",
        (ctypes.Structure,),
        {"_fields_": [(f"f{i}", ct) for i, ct in enumerate(ctypes_types)]},
    )


def _scalar_from_ctypes(dtype: Any, raw: Any) -> Any:
    """Convert the ctypes field of a result into ``dtype``."""
    value = getattr(raw, "value", raw)
    if (
        isinstance(dtype, _t.FloatMeta)
        and dtype.ctype is not None
        and not issubclass(dtype.ctype, (ctypes.c_float, ctypes.c_double))
    ):
        value = _float_from_bits(
            int(value), dtype._exponent_width, dtype._mantissa_width
        )
    return dtype(value)


class Entry(FuncEntryPlugin):
    """``func.func`` with ``llvm.emit_c_interface``; several
    results are packed into one ``!llvm.struct`` and read back through a
    ``ctypes.Structure``. Stateless: ``generate_func_op`` returns the op and
    ``generate_return`` takes it back. A DSL over another dialect may reuse it
    as its ``func_entry`` when its types are legal ``func`` operands."""

    name = "func"

    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> tuple[Any, ir.Block]:
        fop = func.FuncOp(name, (list(arg_types), []), loc=loc)
        fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        if arg_attrs:
            fop.arg_attrs = ir.ArrayAttr.get(list(arg_attrs))
        # Per-argument source locations.
        return fop, fop.add_entry_block(arg_locs=[loc for _ in arg_types])

    def generate_return(self, func_op: Any, values: list[Any], loc: Any = None) -> None:
        if values:
            func_op.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get(list(func_op.type.inputs), [v.type for v in values])
            )
        func.ReturnOp(list(values), loc=loc)

    def pack_results(
        self, values: list[Any], prototypes: list[Any], loc: Any = None
    ) -> tuple[list[Any], Any]:
        if not values:
            return [], None
        slot = _result_ctype(prototypes)
        if len(values) == 1:
            return [values[0]], slot
        struct_type = llvm.StructType.get_literal([v.type for v in values])
        packed = llvm.mlir_undef(struct_type, loc=loc)
        for i, value in enumerate(values):
            packed = llvm.insertvalue(packed, value, [i], loc=loc)
        return [packed], slot

    def unpack_result(self, slot: Any, raw: Any, prototypes: list[Any]) -> list[Any]:
        raws = (
            [raw]
            if len(prototypes) == 1
            else [getattr(raw, f"f{i}") for i in range(len(prototypes))]
        )
        return [_scalar_from_ctypes(p, r) for p, r in zip(prototypes, raws)]
