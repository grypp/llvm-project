# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
import ctypes
import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.type_ops import llvm as dsl_llvm
from mlir.extras import types as T
P, F32, I64 = m.Pointer, m.Float32, m.Int64
name = lambda t: t.__name__
buf = (ctypes.c_float * 8)()
# tf32 has no storage type in the `llvm` dialect: a scalar load or store goes
# through its 32-bit container (`i32`) and a `builtin.unrealized_conversion_cast`
# (`arith.bitcast` rejects the pair), and `getelementptr` strides by the
# container; `storage_type` answers that rule for any element type. The shipped
# lowering cannot reconcile a live tf32 cast, so only the IR is checked.
@m.jit
def tf32_storage(t: P[m.TFloat32]):
    (t + 1).store(t.load())
    t[2] = t[0]


# CHECK-LABEL: func.func @tf32_storage(
# CHECK-SAME:    %[[T:[^:]+]]: !llvm.ptr)
# CHECK:         %[[T1:.+]] = llvm.getelementptr %[[T]][1] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         %[[W:.+]] = llvm.load %[[T]] <alignment = 4> : !llvm.ptr -> i32
# CHECK:         %[[V:.+]] = builtin.unrealized_conversion_cast %[[W]] : i32 to tf32
# CHECK:         %[[S:.+]] = builtin.unrealized_conversion_cast %[[V]] : tf32 to i32
# CHECK:         llvm.store %[[S]], %[[T1]] <alignment = 4> : i32, !llvm.ptr
# CHECK:         %[[T0:.+]] = llvm.getelementptr %[[T]][0] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         llvm.load %[[T0]] <alignment = 4> : !llvm.ptr -> i32
# CHECK:         %[[T2:.+]] = llvm.getelementptr %[[T]][2] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         llvm.store %{{.+}}, %[[T2]] <alignment = 4> : i32, !llvm.ptr
# CHECK:         STORAGE: i32 i8 None None
dsl = m.MlirTestDSL()
if dsl.envar.dryrun:
    tf32_storage(ctypes.addressof(buf))
    with ir.Context():
        print(
            "STORAGE:",
            dsl_llvm.storage_type(T.tf32()),
            dsl_llvm.storage_type(T.f8E5M2()),
            dsl_llvm.storage_type(T.f32()),
            dsl_llvm.storage_type(T.i32()),
        )


# `Pointer` is subclassable: address arithmetic, `tospace` and `& mask` build
# instances of the pointer's own class (`type(self)`, `from_address`), a
# subclass computes addresses its own way by overriding `_gep`, and
# `_vector_result` picks the leaf a vector load returns.
class MyVec:
    def __init__(self, value):
        self.value = value


class MyPtr(m.Pointer):
    def _gep(self, offset, *, loc=None, ip=None):
        if self.space != 3:
            return super()._gep(offset, loc=loc, ip=ip)
        # No getelementptr in space 3 here: arithmetic on the integer address.
        address = self.toint(loc=loc, ip=ip) + offset * self.dtype.bytes
        return type(self).from_address(address, self.space, self.dtype, loc=loc, ip=ip)

    def _vector_result(self, value):
        return MyVec(value)


@m.jit
def subclassed(p: P[F32]) -> I64:
    mp = MyPtr(p)
    q = mp + 1  # space 0: the inherited getelementptr, still a MyPtr
    s = mp.tospace(3)  # a MyPtr in space 3
    t = s + 2  # the override: 8 bytes on, through inttoptr
    al = mp & 15
    print("SUBCLASS:", name(type(q)), name(type(s)), name(type(t)), name(type(al)))
    return t.toint() - s.toint()


# CHECK:       SUBCLASS: MyPtr MyPtr MyPtr MyPtr
# CHECK-LABEL: func.func @subclassed(
# CHECK-SAME:    %[[P:[^:]+]]: !llvm.ptr) -> i64
# CHECK:         llvm.getelementptr %[[P]][1] : (!llvm.ptr) -> !llvm.ptr, f32
# CHECK:         %[[S:.+]] = llvm.addrspacecast %[[P]] : !llvm.ptr to !llvm.ptr<3>
# CHECK:         %[[SI:.+]] = llvm.ptrtoint %[[S]] : !llvm.ptr<3> to i64
# CHECK:         %[[C8:.+]] = arith.constant 8 : i64
# CHECK:         %[[TI:.+]] = arith.addi %[[SI]], %[[C8]] : i64
# CHECK:         llvm.inttoptr %[[TI]] : i64 to !llvm.ptr<3>
# EXEC:          RESULT: 8
print("RESULT:", subclassed(ctypes.addressof(buf)))
