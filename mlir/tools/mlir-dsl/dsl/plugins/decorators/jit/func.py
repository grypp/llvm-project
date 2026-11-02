# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The @jit entry plugin: trace a Python function into func.func."""

from typing import Any

from ..... import ir
from .....dialects import func
from ....core.plugin import DecoratorPlugin


class Jit(DecoratorPlugin):
    name = "func"

    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> tuple[Any, ir.Block]:
        fop = func.FuncOp(name, (list(arg_types), []), loc=loc)
        fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        if arg_attrs:
            fop.arg_attrs = ir.ArrayAttr.get(list(arg_attrs))
        return fop, fop.add_entry_block(arg_locs=[loc for _ in arg_types])

    def generate_return(self, op: Any, values: list[Any], loc: Any = None) -> None:
        if values:
            op.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get(list(op.type.inputs), [v.type for v in values])
            )
        func.ReturnOp(list(values), loc=loc)
