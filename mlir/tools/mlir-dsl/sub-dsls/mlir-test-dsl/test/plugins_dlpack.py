# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The dlpack plugin: `DlpackTensor` reads a DLPack tensor's metadata through
# the `_mlirDslDlpack` extension and the plugin's protocol adapter turns any
# argument with `__dlpack__` into a contiguous `Pointer` of the matching dtype.
# The objects below speak DLPack only; numpy arrays and torch tensors take the
# same path, it is the one inbound adapter of the test DSL.
import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.plugins.adapters import dlpack


class Wrapped:
    """A tensor known only through the DLPack protocol."""

    def __init__(self, array):
        self.array = array

    def __dlpack__(self, **kwargs):
        return self.array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()


def err(label, fn):
    try:
        fn()
        print(label, "-> no error")
    except m.DSLUserCodeError as e:
        print(f"{label}: {e.diag_id.name}")


dsl = m.MlirTestDSL()
# CHECK: PLUGIN: True ['dlpack']
print(
    "PLUGIN:", dlpack.available(), [p.name for p in dsl.plugins if p.name == "dlpack"]
)

grid = np.arange(6, dtype=np.int32).reshape(2, 3)
t = dlpack.DlpackTensor(Wrapped(grid))
# CHECK: VIEW: (2, 3) (3, 1) 2 Int32 host 0 True 24 True tensor<2x3xInt32>_host
print(
    "VIEW:",
    t.shape,
    t.strides,
    t.rank,
    t.dtype.__name__,
    t.device,
    t.device_id,
    t.is_contiguous,
    t.size_in_bytes,
    t.data_ptr == grid.ctypes.data,
    t,
)
# A transposed view is not one row-major block.
# CHECK: TRANSPOSED: (3, 2) (1, 3) False
view = dlpack.DlpackTensor(Wrapped(grid.T))
print("TRANSPOSED:", view.shape, view.strides, view.is_contiguous)
# The element types the DSL maps, and one it does not.
# CHECK: DTYPES: ['Int8', 'Int16', 'Int64', 'Uint8', 'Uint16', 'Uint32', 'Uint64', 'Float16', 'Float32', 'Float64']
names = []
for (
    np_dtype
) in "int8 int16 int64 uint8 uint16 uint32 uint64 float16 float32 float64".split():
    names.append(dlpack.DlpackTensor(Wrapped(np.zeros(2, np_dtype))).dtype.__name__)
print("DTYPES:", names)
# CHECK: complex: TYPE_UNKNOWN_DTYPE_NAME
err("complex", lambda: dlpack.DlpackTensor(Wrapped(np.zeros(2, np.complex64))))
# A bool tensor is one byte per element; a read-only array is accepted (the
# descriptor never writes); an array DLPack cannot describe is a diagnostic,
# not the binding's TypeError.
flags = np.ones(5, dtype=np.bool_)
print(
    "BOOL:",
    dlpack.DlpackTensor(flags).dtype.__name__,
    dlpack.DlpackTensor(flags).size_in_bytes,
)
# CHECK: BOOL: Boolean 5
frozen = np.arange(3, dtype=np.float32)
frozen.setflags(write=False)
print(
    "READ-ONLY:",
    dlpack.DlpackTensor(frozen).shape,
    dlpack.DlpackTensor(frozen).dtype.__name__,
)
# CHECK: READ-ONLY: (3,) Float32
# CHECK: datetime: ARG_BUFFER_INVALID
err("datetime", lambda: dlpack.DlpackTensor(np.zeros(2, "datetime64[s]")))
# The pointer keeps the descriptor (and so the tensor) alive.
p = t.pointer()
# CHECK: POINTER: Int32 host True
print("POINTER:", p.dtype.__name__, p._kind, p._keepalive is t)


@m.jit
def axpy(
    n: m.Int32, alpha: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]
):
    for i in range(n):
        y[i] = alpha * x[i] + y[i]


@m.jit
def first(x) -> m.Int32:
    return x[0]


xs = np.arange(4, dtype=np.float32)
ys = np.ones(4, dtype=np.float32)
axpy(4, 2.0, Wrapped(xs), Wrapped(ys))
# CHECK-LABEL: func.func @axpy(
# CHECK-SAME:    %{{.+}}: i32, %{{.+}}: f32, %{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr)
# EXEC:          AXPY: [1. 3. 5. 7.]
print("AXPY:", ys)
# An unannotated DLPack argument is a staged pointer, not a Meta value.
# CHECK-LABEL: func.func @first(
# CHECK-SAME:    %{{.+}}: !llvm.ptr) -> i32
# EXEC:          FIRST: 7
print("FIRST:", first(Wrapped(np.array([7, 8], dtype=np.int32))))
# CHECK: strided: ARG_BUFFER_INVALID
err("strided", lambda: axpy(2, 1.0, Wrapped(xs[::2]), Wrapped(ys)))
# CHECK: dtype: ARG_ANNOTATION_MISMATCH
err("dtype", lambda: axpy(2, 1.0, Wrapped(grid), Wrapped(ys)))
