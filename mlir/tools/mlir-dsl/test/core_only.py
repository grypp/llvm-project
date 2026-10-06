# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# Import graph: the core (`core/`, `types/`, `util/`, `runtime/`, `compiler/`)
# imports without the `mlir.mlir_dsl` sub-DSL and without any plugin, hence
# without the `gpu`/`nvvm` bindings; the default dialects (`scf`, the LLVM
# world) are imported only when a DSL is built. A `BaseDSL` subclass with
# `plugins = []` traces through the core and reports their pass list.
import sys

import numpy as np

from mlir import execution_engine, passmanager
from mlir.dsl.core import executor, mlir_op, user_op
from mlir.dsl.compiler import jit_executor
from mlir.dsl.compiler.compiler import Compiler
from mlir.dsl.core import common, diagnostics, env_manager, plugin
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.runtime import jit_arg_adapters
from mlir.dsl.types import vector
from mlir.dsl.types.typing import Float32, Int32, Pointer
from mlir.dsl.util import logger, phase_profiler, tree_utils

for name in (
    "mlir.mlir_dsl",
    "mlir.dialects.arith",
    "mlir.dialects.scf",
    "mlir.dialects.llvm",
    "mlir.dialects.func",
    "mlir.dsl.plugins",
    "mlir.dsl.plugins.ast_preprocessor",
    "mlir.dsl.plugins.dialects.gpu",
    "mlir.dialects.gpu",
    "mlir.dialects.nvvm",
):
    print(f"loaded {name}: {name in sys.modules}")
assert "mlir.dialects.gpu" not in sys.modules
# CHECK: loaded mlir.mlir_dsl: False
# CHECK: loaded mlir.dialects.arith: False
# CHECK: loaded mlir.dialects.scf: False
# CHECK: loaded mlir.dialects.llvm: False
# CHECK: loaded mlir.dialects.func: False
# CHECK: loaded mlir.dsl.plugins: False
# CHECK: loaded mlir.dsl.plugins.ast_preprocessor: False
# CHECK: loaded mlir.dsl.plugins.dialects.gpu: False
# CHECK: loaded mlir.dialects.gpu: False
# CHECK: loaded mlir.dialects.nvvm: False


class CpuOnlyDSL(BaseDSL):
    plugins = []
    _jit_arg_adapter_scope = "mlir"

    def __init__(self):
        super().__init__(
            name="MLIR_DSL",
            dsl_package_name=["mlir", "dsl"],
            compiler_provider=Compiler(passmanager, execution_engine),
            pass_sm_arch_name="cubin-chip",
            preprocess=False,
        )


@CpuOnlyDSL.jit
def scale_store(a: Int32, out: Pointer[Float32]) -> Int32:
    out[0] = Float32(a)
    return a * 2 + 1


# CHECK-LABEL: func.func @scale_store(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[OUT:[^:]+]]: !llvm.ptr) -> i32
# CHECK-NOT:     gpu.
# CHECK:         %[[F:.+]] = arith.sitofp %[[A]] : i32 to f32
# CHECK:         %[[P:.+]] = llvm.getelementptr %[[OUT]][0] : (!llvm.ptr) -> !llvm.ptr, f32
# CHECK:         llvm.store %[[F]], %[[P]] <alignment = 4> : f32, !llvm.ptr
# CHECK:         %[[M:.+]] = arith.muli %[[A]], %{{.+}} : i32
# CHECK:         %[[R:.+]] = arith.addi %[[M]], %{{.+}} : i32
# CHECK:         return %[[R]] : i32
# CHECK-NOT:     gpu.
# CHECK-NOT:     nvvm.
# CHECK:         RESULT: ?
# CHECK:         pipeline: builtin.module(convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)
# CHECK:         gpu bindings loaded: False
print("RESULT:", scale_store(3, np.zeros(4, np.float32)))
print("pipeline:", CpuOnlyDSL()._get_pipeline(None))
print("gpu bindings loaded:", "mlir.dialects.gpu" in sys.modules)
