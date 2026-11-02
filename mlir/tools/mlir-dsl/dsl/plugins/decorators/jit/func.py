# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``func`` jit plugin: ``@jit`` over a ``func.func`` with the C interface
whose results travel back to the host in one ``!llvm.struct`` read through
``ctypes``.

The decorator, the call pipeline and the inline launch are
``DecoratorPlugin``'s defaults; this plugin adds the entry protocol only."""

from __future__ import annotations

from typing import Any

from ..... import ir
from .....dialects import func, llvm
from ....core.plugin import DecoratorPlugin

__all__ = ["Jit"]


class Jit(DecoratorPlugin):
    """``@jit`` over ``func.func`` with ``llvm.emit_c_interface``; several
    results are packed into one ``!llvm.struct`` and read back through a
    ``ctypes.Structure``. Stateless: ``generate_func_op`` returns the op and
    ``generate_return`` takes it back. A DSL over another dialect may reuse it
    as its ``@jit`` when its types are legal ``func`` operands."""

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

    def generate_return(self, op: Any, values: list[Any], loc: Any = None) -> None:
        if values:
            op.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get(list(op.type.inputs), [v.type for v in values])
            )
        func.ReturnOp(list(values), loc=loc)

    def pack_results(
        self, values: list[Any], prototypes: list[Any], loc: Any = None
    ) -> tuple[list[Any], Any]:
        if not values:
            return [], None
        slot = None
        if len(values) == 1:
            return [values[0]], slot
        struct_type = llvm.StructType.get_literal([v.type for v in values])
        packed = llvm.mlir_undef(struct_type, loc=loc)
        for i, value in enumerate(values):
            packed = llvm.insertvalue(packed, value, [i], loc=loc)
        return [packed], slot
