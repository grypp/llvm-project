# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""mlir.emitc_dsl: the EmitC test sub-DSL and the namespace to write programs against.

``import mlir.emitc_dsl as e``: the core names (``e.Int32``, ``e.Pointer``,
``e.BaseDSL``, ``e.Plugins``, ...), the ``scf`` syntax helpers (``e.range``,
``e.for_``/``e.if_``/``e.while_``/``e.yield_``, ``e.and_``/``e.or_``/...) and the
DSL itself: ``@e.jit``, ``e.EmitCTestDSL``, ``e.EmitCScalarOps``. The programs
are the ones written against ``mlir.mlir_dsl``; the ``+`` and ``*`` in them
become EmitC ops, and ``mlir-translate --mlir-to-cpp`` finishes the job.
"""

from mlir.dsl import *  # noqa: F401,F403  the core names, re-exported
from mlir.dsl import __all__ as _core_names
from mlir.dsl.plugins.ast_preprocessor import DSLPreprocessor, range
from mlir.dsl.plugins.ast_preprocessor.scf import (
    LoopUnroll,
    WhileLoopContext,
    all_,
    and_,
    any_,
    for_,
    if_,
    in_,
    not_,
    or_,
    while_,
    yield_,
)

from .emitc_dsl import EmitCScalarOps, EmitCTestDSL, jit

__all__ = [
    *_core_names,
    "DSLPreprocessor",
    "range",
    "LoopUnroll",
    "WhileLoopContext",
    "all_",
    "and_",
    "any_",
    "for_",
    "if_",
    "in_",
    "not_",
    "or_",
    "while_",
    "yield_",
    "EmitCScalarOps",
    "EmitCTestDSL",
    "jit",
]
