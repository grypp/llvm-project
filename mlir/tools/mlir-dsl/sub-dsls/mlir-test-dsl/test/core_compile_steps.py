# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: host-supports-jit
# A type-only argument compiles for that dtype without a concrete host value.
import mlir.mlir_dsl as m

@m.jit
def plus_one(value: m.Int32) -> m.Int32:
    return value + 1

compiled = m.compile(plus_one, m.Int32)
assert compiled(9) == 10
assert compiled(20) == 21
print("PLACEHOLDER: compiled once, invoked with two values")
# CHECK: PLACEHOLDER: compiled once, invoked with two values
