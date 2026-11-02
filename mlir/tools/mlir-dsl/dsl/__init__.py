# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""mlir.dsl: a Python DSL base layer for MLIR (see mlir/tools/mlir-dsl/README.md).

The core: the types (``Int32``, ``Pointer``, ``@struct``, ``Vector``), the
staging decision and the host boundary (``BaseDSL``), the plugin roles
(``TypeOpsPlugin``, ``ASTPreprocessorPlugin``, ``CompilerPlugin``) and the
families (``DecoratorPlugin``, ``AdapterPlugin``), and
the ``Plugins`` record a DSL names them in, and the extension points
(``register_leaf``, ``register_jit_arg_adapter``). It emits no dialect op of
its own: each plugin folder ships the op modules it emits through, and
``mlir.dsl.plugins`` the role implementations and the plugin families. A
sub-DSL is a ``BaseDSL`` subclass with its own ``plugins`` record;
``mlir.mlir_dsl`` is the test DSL and the namespace to write programs against.
"""

# The types
from .types.typing import (
    DslType,
    Numeric,
    NumericMeta,
    IntegerMeta,
    FloatMeta,
    Boolean,
    Integer,
    Int32,
    Int64,
    Float,
    Float32,
    ALL_DTYPES,
    as_numeric,
    cast,
)

# The sub-DSL authoring surface
from .core.dsl import BaseDSL
from .core.plugin import (
    ASTPreprocessorPlugin,
    AdapterPlugin,
    CompilerPlugin,
    DecoratorPlugin,
    Plugin,
    Plugins,
    TypeOpsPlugin,
)
from .core.mlir_op import OpEmitter, current_emitter

# Diagnostics
from .core.common import DSLRuntimeError, DSLUserCodeError
from .core.diagnostics import DiagId

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
    "BaseDSL",
    "ASTPreprocessorPlugin",
    "Plugin",
    "Plugins",
    "TypeOpsPlugin",
    "CompilerPlugin",
    "DecoratorPlugin",
    "AdapterPlugin",
    "OpEmitter",
    "current_emitter",
    "DSLRuntimeError",
    "DSLUserCodeError",
    "DiagId",
]
