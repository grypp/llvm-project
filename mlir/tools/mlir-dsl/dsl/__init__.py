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
    Int2,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,
    Float,
    Float16,
    BFloat16,
    TFloat32,
    Float32,
    Float64,
    Float8E5M2,
    Float8E4M3,
    Float8E4M3FN,
    Float8E4M3B11FNUZ,
    Float8E3M4,
    Float8E8M0FNU,
    Float8E5M3FNU,
    Float8E5M2FNUZ,
    Float8E4M3FNUZ,
    Float4E2M1FN,
    Float6E2M3FN,
    Float6E3M2FN,
    ALL_DTYPES,
    dtype,
    from_numpy_dtype,
    as_numeric,
    as_ir_value,
    cast,
    align,
    Pointer,
    TypedPointer,
    inttoptr,
    Struct,
    struct,
    make_struct,
)
from .types.vector import Vector
from .types.typing import max_ as max, min_ as min

# The sub-DSL authoring surface
from .core.dsl import BaseDSL
from .core.plugin import (
    ASTPreprocessorPlugin,
    AdapterPlugin,
    CompilerPlugin,
    DecoratorPlugin,
    DeferredDecoratorCall,
    Plugin,
    Plugins,
    TypeOpsPlugin,
)
from .core.mlir_op import OpEmitter, current_emitter
from .core.staging import Executor, is_mlir_op
from .util.tree_utils import register_leaf
from .core.arguments import JitArgAdapterRegistry

register_jit_arg_adapter = JitArgAdapterRegistry.register_jit_arg_adapter

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
    "Int2",
    "Int4",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Uint128",
    "Float",
    "Float16",
    "BFloat16",
    "TFloat32",
    "Float32",
    "Float64",
    "Float8E5M2",
    "Float8E4M3",
    "Float8E4M3FN",
    "Float8E4M3B11FNUZ",
    "Float8E3M4",
    "Float8E8M0FNU",
    "Float8E5M3FNU",
    "Float8E5M2FNUZ",
    "Float8E4M3FNUZ",
    "Float4E2M1FN",
    "Float6E2M3FN",
    "Float6E3M2FN",
    "ALL_DTYPES",
    "dtype",
    "from_numpy_dtype",
    "as_numeric",
    "as_ir_value",
    "cast",
    "align",
    "Pointer",
    "TypedPointer",
    "inttoptr",
    "Struct",
    "struct",
    "make_struct",
    "Vector",
    "max",
    "min",
    "BaseDSL",
    "ASTPreprocessorPlugin",
    "Plugin",
    "Plugins",
    "TypeOpsPlugin",
    "CompilerPlugin",
    "DecoratorPlugin",
    "DeferredDecoratorCall",
    "AdapterPlugin",
    "OpEmitter",
    "current_emitter",
    "Executor",
    "is_mlir_op",
    "register_leaf",
    "register_jit_arg_adapter",
    "DSLRuntimeError",
    "DSLUserCodeError",
    "DiagId",
]
