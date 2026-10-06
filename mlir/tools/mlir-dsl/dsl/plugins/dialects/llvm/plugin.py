# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The LLVM world as a dialect plugin: the core's default for the types."""

from __future__ import annotations

from typing import Any

from ..... import execution_engine, passmanager
from ....compiler.compiler import Compiler
from ....core.plugin import DialectPlugin
from .emitter import LlvmEmitter
from .entry import LlvmHostGenHelper

__all__ = ["LlvmDialectPlugin"]


class LlvmDialectPlugin(DialectPlugin):
    """The types on MLIR's builtin types and ``arith``/``math``/``vector``/
    ``llvm`` ops, the ``func.func`` host entry with the C interface, the
    ``mlir`` pass manager and execution engine as the compiler, and the
    lowering of that whole world, host entry included, to the LLVM dialect. ``BaseDSL`` installs it (after ``ScfDialectPlugin``) unless a
    listed dialect plugin brings an emitter of its own."""

    name = "llvm"
    emitter = LlvmEmitter()
    host_gen_helper = LlvmHostGenHelper

    def install(self, dsl: Any) -> None:
        # One compiler per DSL instance: the pass manager and the execution
        # engine of the ``mlir`` Python bindings.
        self.compiler_provider = Compiler(passmanager, execution_engine)

    def pipeline_passes(self) -> list[str]:
        return [
            "convert-vector-to-llvm",
            "convert-arith-to-llvm",
            "convert-math-to-llvm",
            "convert-func-to-llvm",
        ]
