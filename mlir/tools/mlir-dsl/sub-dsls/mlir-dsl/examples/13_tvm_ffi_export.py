# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TVM-FFI export: call a compiled function through the TVM-FFI ABI, outside the DSL.

With `MLIR_DSL_ENABLE_TVM_FFI=1` set before the import, every compiled module
also carries `llvm.func @__tvm_ffi_<name>`: a wrapper with the TVM-FFI calling
convention that checks the arguments and calls the entry. `m.compile(...)` then
returns a function whose `tvm_ffi_function` is a plain `tvm_ffi.Function`, usable
by any TVM-FFI host. Look at how staged and Meta arguments show up in the exported
signature. `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` prints the wrapper instead of
running; without the `tvm_ffi` package (`pip install apache-tvm-ffi`) this skips.
"""

import os

# The plugin reads the variable when the DSL instance is created: set it first.
os.environ.setdefault("MLIR_DSL_ENABLE_TVM_FFI", "1")

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.plugins.thirdparty.tvm_ffi import available


@m.jit
def scaled_dot(
    n: m.Int32, alpha, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]
) -> m.Float32:
    # `n`, `x`, `y` are staged: the wrapper decodes them from the TVM-FFI call
    # (`int32`, two `DataPointer`s). `alpha` has no DSL annotation, so it is
    # Meta: folded into the trace and the symbol (`scaled_dot_20`); the wrapper
    # only asserts that a caller repeats the value it was compiled with.
    acc = m.Float32(0.0)
    for i in range(n):
        acc += alpha * x[i] * y[i]
    return acc


def main():
    if not available():
        print("TVM-FFI export: skipped (pip install apache-tvm-ffi)")
        return
    x = np.arange(8, dtype=np.float32)
    y = np.full(8, 0.5, dtype=np.float32)
    expected = float(np.dot(2.0 * x, y))

    # Compile without running: the module now holds `func.func @scaled_dot_20`
    # and `llvm.func @__tvm_ffi_scaled_dot_20(handle, args, num_args, result)`.
    compiled = m.compile(scaled_dot, 8, 2.0, x, y)
    if os.environ.get("MLIR_DSL_DRYRUN"):  # a dry run traces but does not JIT
        print("TVM-FFI export: passed (dry run, wrapper traced only)")
        return

    # The export is a `tvm_ffi.Function` over that symbol: it speaks TVM-FFI,
    # not DSL. Scalars are Python values and a DataPointer is an address, which
    # is what a DLPack host hands over as well.
    exported = compiled.tvm_ffi_function
    print("exported:", type(exported).__module__, type(exported).__name__)
    result = exported(8, 2.0, x.ctypes.data, y.ctypes.data)
    assert result == expected, result
    print(f"exported(8, 2.0, &x, &y) = {result}")

    # The Meta argument is baked in: another value is not a new specialization
    # (that happens only through the DSL's own call) but a rejected call.
    try:
        exported(8, 3.0, x.ctypes.data, y.ctypes.data)
    except ValueError as e:
        print("exported(8, 3.0, ...):", e)
    else:
        raise AssertionError("the wrapper must reject another Meta value")

    # Calling the compiled object goes through the same wrapper; the DSL adapts
    # NumPy buffers to addresses and renders the rejection as a diagnostic.
    assert float(compiled(8, 2.0, x, y)) == expected
    try:
        compiled(8, 3.0, x, y)
    except m.DSLUserCodeError as e:
        assert e.diag_id is m.DiagId.CALL_TVM_FFI_ARGS
        print("compiled(8, 3.0, x, y):", e.diag_id.name)
    print("TVM-FFI export: passed")


if __name__ == "__main__":
    main()
