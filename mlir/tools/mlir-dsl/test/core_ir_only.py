# RUN: env MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# A DSL that names no compiler plugin is an IR generator. A `@jit` call
# traces, prints the module under PRINT_IR and returns the trace result with
# no DRYRUN set: there is nothing to compile or run. `compile_only=True` on
# such a DSL is the error naming the missing plugin. The DSL names only the
# `func.Jit` decorator plugin and the arith type ops, so neither the execution
# engine nor the pass manager module is imported for it.
import sys

from mlir.dsl.core.common import DSLRuntimeError
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith
from mlir.dsl.types.typing import Int32


class IrOnlyDSL(BaseDSL):
    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith),
        decorators=[func.Jit()],
    )

    def __init__(self):
        super().__init__(name="MLIR_DSL")


@IrOnlyDSL.jit
def twice(a: Int32) -> Int32:
    return a + a


# CHECK-LABEL: func.func @twice(
# CHECK-SAME:    %[[A:[^:]+]]: i32) -> i32
# CHECK:         %[[R:.+]] = arith.addi %[[A]], %[[A]] : i32
# CHECK:         return %[[R]] : i32
result = twice(4)
print("RESULT:", type(result).__name__)
dsl = IrOnlyDSL()
print("PLUGINS:", [p.name for p in dsl.plugins], dsl.plugins.compiler)
print(
    "ENGINE LOADED:",
    "mlir.execution_engine" in sys.modules,
    "mlir.passmanager" in sys.modules,
)
try:
    twice(4, compile_only=True)
    print("COMPILE: no error")
except DSLRuntimeError as e:
    print("COMPILE:", "this DSL has no compiler" in str(e))
# CHECK: RESULT: Int32
# CHECK: PLUGINS: ['arith', 'func'] None
# CHECK: ENGINE LOADED: False False
# CHECK: COMPILE: True
