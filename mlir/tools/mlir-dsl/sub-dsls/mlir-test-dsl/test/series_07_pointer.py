# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# Scalar pointer IR and raw-address marshalling do not depend on DLPack.
import ctypes
import mlir.mlir_dsl as m


@m.jit
def update(p: m.Pointer[m.Int32], i: m.Int32) -> m.Int32:
    old = p[i]
    p[i] = old + 5
    return p[i]


# CHECK-LABEL: func.func @update(
# CHECK-SAME: !llvm.ptr
# CHECK: llvm.getelementptr
# CHECK: llvm.load
# CHECK: arith.addi
# CHECK: llvm.store
storage = (ctypes.c_int32 * 3)(10, 20, 30)
result = update(ctypes.addressof(storage), 1)
if not m.is_mlir_op(result):
    assert int(result) == 25 and list(storage) == [10, 25, 30]
    print("UPDATE: 25 [10, 25, 30]")
# EXEC: UPDATE: 25 [10, 25, 30]


@m.jit
def bad_index(p: m.Pointer[m.Int32]):
    return p[1.5]


try:
    bad_index(0)
except m.DSLUserCodeError as exc:
    assert exc.code == "POINTER_INDEX_UNSUPPORTED"
    print("INDEX: POINTER_INDEX_UNSUPPORTED")
else:
    raise AssertionError("floating-point pointer index was accepted")
# CHECK: INDEX: POINTER_INDEX_UNSUPPORTED
