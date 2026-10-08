# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# Import graph: the core (`core/`, `types/`, `util/`) imports without the
# `mlir.mlir_dsl` sub-DSL, without any plugin or dialect module, hence without
# the `gpu`/`nvvm` bindings, the execution engine or the pass manager; a
# dialect is imported only when a DSL names a plugin that emits it. A `BaseDSL` subclass naming the builtin type ops, the
# `func.func` entry and the execution-engine compiler traces through the core
# and reports its own pass list.
import sys

import numpy as np

from mlir.dsl.core import mlir_op, remarks, staging, user_op
from mlir.dsl.core import arguments, common, diagnostics, env_manager
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.types import vector
from mlir.dsl.types.typing import Float32, Int32, Pointer
from mlir.dsl.util import logger, profiler, tree_utils

for name in (
    "mlir.mlir_dsl",
    "mlir.dialects.arith",
    "mlir.dialects.scf",
    "mlir.dialects.llvm",
    "mlir.dialects.func",
    "mlir.dsl.plugins.type_ops",
    "mlir.dsl.plugins",
    "mlir.dsl.plugins.ast_preprocessor",
    "mlir.dsl.plugins.decorators",
    "mlir.dsl.plugins.compiler",
    "mlir.dialects.gpu",
    "mlir.dialects.nvvm",
    "mlir.execution_engine",
    "mlir.passmanager",
):
    print(f"loaded {name}: {name in sys.modules}")
assert "mlir.dialects.gpu" not in sys.modules
# CHECK: loaded mlir.mlir_dsl: False
# CHECK: loaded mlir.dialects.arith: False
# CHECK: loaded mlir.dialects.scf: False
# CHECK: loaded mlir.dialects.llvm: False
# CHECK: loaded mlir.dialects.func: False
# CHECK: loaded mlir.dsl.plugins.type_ops: False
# CHECK: loaded mlir.dsl.plugins: False
# CHECK: loaded mlir.dsl.plugins.ast_preprocessor: False
# CHECK: loaded mlir.dsl.plugins.decorators: False
# CHECK: loaded mlir.dsl.plugins.compiler: False
# CHECK: loaded mlir.dialects.gpu: False
# CHECK: loaded mlir.dialects.nvvm: False
# CHECK: loaded mlir.execution_engine: False
# CHECK: loaded mlir.passmanager: False

# The plugins a CPU-only DSL needs, imported only now (numpy arrays are an
# adapter plugin's, like torch tensors; the core adapts no host buffer).
from mlir.dsl.plugins.adapters.numpy import NumpyPlugin
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.func_entry import func
from mlir.dsl.plugins.type_ops import arith, llvm, vector
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps


class CpuOnlyDSL(BaseDSL):
    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
        func_entry=func.Entry(),
        compiler=execution_engine.Compiler(),
        adapters=[NumpyPlugin()],
    )

    def pipeline(self):
        # The sub-DSL owns its pipeline: the passes lowering arith, scf and the
        # func entry to the LLVM dialect, spelled here (no plugin publishes any).
        return [
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
        ]

    def __init__(self):
        super().__init__(
            name="MLIR_DSL",
            dsl_package_name=["mlir", "dsl"],
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
# CHECK:         plugins: ['arith+vector+llvm', 'func', 'execution_engine', 'numpy'] {}
# CHECK:         pipeline: builtin.module(convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)
# CHECK:         gpu bindings loaded: False
print("RESULT:", scale_store(3, np.zeros(4, np.float32)))
print(
    "plugins:", [p.name for p in CpuOnlyDSL().plugins], CpuOnlyDSL().unavailable_plugins
)
print("pipeline:", CpuOnlyDSL()._get_pipeline(None))
print("gpu bindings loaded:", "mlir.dialects.gpu" in sys.modules)
