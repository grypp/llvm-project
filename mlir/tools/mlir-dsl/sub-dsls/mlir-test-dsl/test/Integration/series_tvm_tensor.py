# RUN: env MLIR_DSL_ENABLE_TVM_FFI=1 %PYTHON %s
# REQUIRES: host-supports-jit, tvm_ffi
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execute tensor metadata guards and the decorated-function pointer path."""
import numpy as np
import tvm_ffi
from tvm_ffi import libinfo
from mlir import execution_engine, ir
import mlir.mlir_dsl as m
from mlir.dsl.plugins.adapters.tvm_ffi import (
    CallProvider,
    DirectCallProvider,
    TvmFfiJitCompiledFunction,
    attach_ffi_func,
    spec,
)


class CheckedOnly(CallProvider):
    def __call__(self, block, context):
        with ir.InsertionPoint(block):
            DirectCallProvider("unused").store_result(context.raw_result, None)
        return block


with ir.Context(), ir.Location.unknown():
    n = spec.Var("n", "int64")
    module = ir.Module.create()
    attach_ffi_func(
        module,
        "check_shapes",
        [
            spec.Tensor("x", [n, 2], "float32", device_type="cpu"),
            spec.Tensor("y", [n, 2], "float32", device_type="cpu"),
        ],
        CheckedOnly(),
    )
    assert module.operation.verify()
    engine = execution_engine.ExecutionEngine(
        module, shared_libs=[libinfo.find_libtvm_ffi()]
    )
    check = tvm_ffi.Function.__from_mlir_packed_safe_call__(
        engine.raw_lookup("__tvm_ffi_check_shapes"), keep_alive_object=engine
    )
    x = tvm_ffi.from_dlpack(np.zeros((3, 2), dtype=np.float32))
    y = tvm_ffi.from_dlpack(np.ones((3, 2), dtype=np.float32))
    assert check(x, y) is None
    for invalid in (
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.int32),
        np.zeros((3, 4), dtype=np.float32)[:, ::2],
    ):
        try:
            check(x, tvm_ffi.from_dlpack(invalid))
        except Exception:
            pass
        else:
            raise AssertionError("the tensor ABI must reject incompatible metadata")


@m.jit
def first(x: m.Pointer[m.Int32]) -> m.Int32:
    return x[0]


values = np.array([7, 9], dtype=np.int32)
compiled = m.compile(first, values)
assert isinstance(compiled, TvmFfiJitCompiledFunction)
assert int(compiled(values)) == 7
print("tensor guards, symbolic dimensions and pointer calls: passed")
