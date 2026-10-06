# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""mlir.mlir_dsl: the reference sub-DSL on ``mlir.dsl`` and the namespace to write programs against.

``import mlir.mlir_dsl as m``: the core names (``m.Int32``, ``m.Pointer``,
``m.struct``, ``m.Vector``, ``m.BaseDSL``, ...), the plugins ``MlirDSL`` is
assembled from (``m.for_``/``m.if_``/``m.while_``/``m.yield_`` and the
``scf`` plugins, the LLVM world, the gpu index helpers when the gpu bindings
are present) and the DSL itself: ``@m.jit``, ``@m.kernel``, ``m.compile``,
``m.MlirDSL``. A sub-DSL of your own exports its own namespace the same way.
"""

from mlir.dsl import *  # noqa: F401,F403  the core names, re-exported
from mlir.dsl import __all__ as _core_names
from mlir.dsl.plugins.ast_preprocessor import (
    DSLPreprocessor,
    ScfASTPreprocessorPlugin,
    range,
)
from mlir.dsl.plugins.dialects.llvm import (
    LlvmDialectPlugin,
    LlvmEmitter,
    LlvmHostGenHelper,
)
from mlir.dsl.plugins.dialects.scf import (
    LoopUnroll,
    ScfDialectPlugin,
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

from .mlir_dsl import MlirDSL, compile, jit, kernel

__all__ = [
    *_core_names,
    "DSLPreprocessor",
    "ScfASTPreprocessorPlugin",
    "range",
    "LlvmDialectPlugin",
    "LlvmEmitter",
    "LlvmHostGenHelper",
    "LoopUnroll",
    "ScfDialectPlugin",
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
    "MlirDSL",
    "compile",
    "jit",
    "kernel",
]

# Kernel-body index helpers, present when the gpu plugin imports.
try:
    from mlir.dsl.plugins.dialects.gpu import (
        GridConstant,
        block_dim,
        block_idx,
        grid_constant,
        grid_dim,
        thread_idx,
    )
except ImportError:
    pass
else:
    __all__ += [
        "GridConstant",
        "block_dim",
        "block_idx",
        "grid_constant",
        "grid_dim",
        "thread_idx",
    ]
