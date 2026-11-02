# RUN: env MLIR_DSL_ENABLE_TVM_FFI=1 %PYTHON %s
# REQUIRES: host-supports-jit, tvm_ffi
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The scalar export is executable, including compile-time-value checks."""
import mlir.mlir_dsl as m
from mlir.dsl.plugins.adapters.tvm_ffi import TvmFfiJitCompiledFunction


@m.jit
def scale(x: m.Int32, factor) -> m.Int32:
    return x * factor


compiled = m.compile(scale, 7, 3)
assert isinstance(compiled, TvmFfiJitCompiledFunction)
assert int(compiled(7, 3)) == 21
assert compiled.tvm_ffi_function(8, 3) == 24
try:
    compiled.tvm_ffi_function(8, 4)
except Exception as error:
    assert "3" in str(error), str(error)
else:
    raise AssertionError("the specialized constant must be checked")
print("scalar TVM-FFI execution and constant check: passed")
