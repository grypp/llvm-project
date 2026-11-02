# RUN: env MLIR_DSL_ENABLE_TVM_FFI=1 MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: env MLIR_DSL_ENABLE_TVM_FFI=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=OFF
# REQUIRES: host-supports-jit, tvm_ffi
# The TVM-FFI export plugin (`plugins/adapters/tvm_ffi/`) and the
# builder it drives (`tvm_ffi_builder`, DSL-owned MLIR emission). The plugin is
# installed when the `tvm_ffi` package is importable (the record drops it
# otherwise), never imports it at import time and is inert unless
# MLIR_DSL_ENABLE_TVM_FFI is set. Enabled, it maps the
# Python signature of each host entry to `spec` parameters (scalar annotations
# through NumericToTVMFFIDtype to `Var`, pointers to `DataPointer`, Meta values
# to `Const*`), adds `llvm.func @__tvm_ffi_<name>` with the TVM-FFI ABI that
# checks the arguments and calls the entry (`DirectCallProvider`), and after
# the JIT calls the compiled function through `tvm_ffi.Function`; a rejected
# argument is `tvm_ffi:CALL_REJECTED`. The `spec` kinds, their signature text and
# the `tvm_ffi:UNSUP_PARAM` path are the builder's own.
import os
import sys
from dataclasses import replace

import numpy as np

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.adapters import tvm_ffi as plugin_module
from mlir.dsl.plugins.adapters.tvm_ffi import NumericToTVMFFIDtype, TvmFfiPlugin
from mlir.dsl.plugins.adapters.tvm_ffi import (
    DirectCallProvider,
    NopCallProvider,
    spec,
)
from mlir.dsl.plugins.adapters.tvm_ffi import attach_ffi_func, rename_tvm_ffi_function

available = plugin_module.available
DRYRUN = bool(os.environ.get("MLIR_DSL_DRYRUN"))


def expect(label, fn):
    try:
        fn()
        print(label, "no error")
    except m.DSLUserCodeError as e:
        print(label, e.diag_id.name)
        print(str(e))
    except m.DSLRuntimeError as e:
        print(label, "INTERNAL", e.message)


# =============================================================================
# Gating: lazy import, listing, enabling, the missing package
# =============================================================================
print("LAZY:", "tvm_ffi" in sys.modules, available(), "tvm_ffi" in sys.modules)
# CHECK: LAZY: False True False
# OFF:   LAZY: False True False
print("SYMBOL:", plugin_module.tvm_ffi_symbol("axpy"))
print("TABLE:", ", ".join(f"{k.__name__}={v}" for k, v in NumericToTVMFFIDtype.items()))
# CHECK: SYMBOL: __tvm_ffi_axpy
# CHECK: TABLE: Boolean=bool, Int8=int8, Int16=int16, Int32=int32, Int64=int64, Uint8=uint8, Uint16=uint16, Uint32=uint32, Uint64=uint64, Float16=float16, BFloat16=bfloat16, Float32=float32, Float64=float64

# With the package hidden the plugin is not available: the record drops it at
# construction and remembers it, with or without the variable set (the
# CONFIG_INVALID check of `install` is reached only with the package present).
saved_module, saved_var = sys.modules.get("tvm_ffi"), os.environ.get(
    "MLIR_DSL_ENABLE_TVM_FFI"
)
sys.modules["tvm_ffi"] = None
os.environ["MLIR_DSL_ENABLE_TVM_FFI"] = "1"


class MissingDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, adapters=[TvmFfiPlugin()])


expect("MISSING", MissingDSL)
print("DROPPED:", list(MissingDSL().plugins.adapters), MissingDSL().unavailable_plugins)
# CHECK: MISSING no error
# CHECK: DROPPED: [] {'adapters[tvm_ffi]': 'TvmFfiPlugin'}
# OFF:   MISSING no error
# OFF:   DROPPED: [] {'adapters[tvm_ffi]': 'TvmFfiPlugin'}
del os.environ["MLIR_DSL_ENABLE_TVM_FFI"]


class InertDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, adapters=[TvmFfiPlugin()])


inert = InertDSL()
print("HIDDEN:", available(), list(inert.plugins.adapters), inert.unavailable_plugins)
# CHECK: HIDDEN: False [] {'adapters[tvm_ffi]': 'TvmFfiPlugin'}
# OFF:   HIDDEN: False [] {'adapters[tvm_ffi]': 'TvmFfiPlugin'}
if saved_module is None:
    del sys.modules["tvm_ffi"]
else:
    sys.modules["tvm_ffi"] = saved_module
if saved_var is not None:
    os.environ["MLIR_DSL_ENABLE_TVM_FFI"] = saved_var

listed = [p for p in m.MlirTestDSL.plugins.adapters if p.name == "tvm_ffi"]
dsl = m.MlirTestDSL()
installed = [p for p in dsl.plugins.adapters if p.name == "tvm_ffi"][0]
print(
    "PLUGIN:",
    len(listed),
    installed is not listed[0],
    installed.dsl is dsl,
    installed.enabled,
)
libs = installed.shared_libs()
print(
    "LIBS:", len(libs), all(os.path.basename(p).startswith("libtvm_ffi") for p in libs)
)
# The plugin is installed as a copy bound to the instance; enabled it hands
# `libtvm_ffi` to the ExecutionEngine (the wrapper calls its error entry points).
# CHECK: PLUGIN: 1 True True True
# CHECK: LIBS: 1 True
# OFF:   PLUGIN: 1 True True False
# OFF:   LIBS: 0 True


# =============================================================================
# The export plan and the wrapper
# =============================================================================
@m.struct
class Params:
    n: m.Int32


@m.jit
def axpy(n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
    for i in range(n):
        y[i] = a * x[i] + y[i]


@m.jit
def consts(n, flag, scale, opt) -> m.Uint32:
    acc = m.Uint32(0)
    for i in range(n):
        acc += i
    return acc


@m.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2


@m.jit
def with_struct(p: Params) -> m.Int32:
    return p.n


@m.jit
def shared(q: m.Pointer[m.Int32, 1]):
    pass


@m.jit
def cumsum(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


x = np.arange(8, dtype=np.float32)
y = np.ones(8, dtype=np.float32)
axpy(8, 2.0, x, y)
print("AXPY:", np.allclose(y, 2.0 * x + 1.0))
# The wrapper has the TVM-FFI ABI (handle, args, num_args, result -> i32),
# checks the argument count and each argument's type, then calls the entry.
# CHECK-LABEL: func.func @axpy(
# CHECK:       llvm.func @__tvm_ffi_axpy(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %[[N:.*]]: i32, %{{.*}}: !llvm.ptr) -> i32
# CHECK:         llvm.icmp "eq" %[[N]], %{{.*}} : i32
# CHECK:         func.call @axpy(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, f32, !llvm.ptr, !llvm.ptr) -> ()
# CHECK:         llvm.return %{{.*}} : i32
# EXEC:         AXPY: True
# OFF:          AXPY: True

print("CONSTS:", consts(4, True, 1.5, None))
# A Meta argument is no operand but a Const* the wrapper asserts; the result
# is widened to the i64 result slot by its annotation's signedness.
# CHECK-LABEL: func.func @consts_4_True_15_None() -> i32
# CHECK:       llvm.func @__tvm_ffi_consts_4_True_15_None(
# CHECK:         %[[R:.*]] = func.call @consts_4_True_15_None() : () -> i32
# CHECK:         llvm.zext %[[R]] : i32 to i64
# EXEC:         CONSTS: 6
# OFF:          CONSTS: 6

print("CUMSUM:", cumsum(10), type(cumsum(10)).__name__)
# A `func.func` host entry is called with `func.call`.
# CHECK-LABEL: func.func @cumsum(
# CHECK:       llvm.func @__tvm_ffi_cumsum(
# CHECK:         %[[R:.*]] = func.call @cumsum(%{{.*}}) : (i32) -> i32
# CHECK:         llvm.sext %[[R]] : i32 to i64
# EXEC:         CUMSUM: 45 Int32
# OFF:          CUMSUM: 45 Int32

with_struct(Params(n=1))
shared(m.Pointer(0, dtype=m.Int32, space=1))
twice(21)
plans = installed._plans
print("PLANS:", [spec.signature(name, plan.params) for name, plan in plans.items()])
# A struct argument and a pointer outside address space 0 are not exportable:
# those functions keep the packed entry (a warning in the log), no plan.
# CHECK: PLANS: ['axpy(n: int32, a: float32, x: DataPointer, y: DataPointer)', 'consts_4_True_15_None(n: Int(4), flag: Bool(True), scale: Float(1.5), opt: None)', 'cumsum(n: int32)', 'twice(n: int32)']
# OFF:   PLANS: []
if installed.enabled:
    plan = plans["consts_4_True_15_None"]
    print(
        "PLAN:",
        sorted(plans["axpy"].pointer_positions),
        sorted(plan.const_positions),
        plan.result_dtype.__name__,
    )
    # CHECK: PLAN: [2, 3] [0, 1, 2, 3] Uint32


# =============================================================================
# The call path: through the wrapper, and its errors
# =============================================================================
if not DRYRUN:
    compiled = m.compile(twice, 21)
    print(
        "CLASS:",
        type(compiled).__name__,
        type(getattr(compiled, "tvm_ffi_function", None)).__name__,
    )
    # EXEC: CLASS: TvmFfiJitCompiledFunction Function
    # OFF:  CLASS: JitCompiledFunction NoneType
if not DRYRUN and installed.enabled:
    exported = m.compile(consts, 4, True, 1.5, None).tvm_ffi_function
    print("EXPORTED:", exported(4, True, 1.5, None))
    # EXEC: EXPORTED: 6
    # Called from outside the DSL, the wrapper rejects another compile-time
    # constant and a wrong type; through the DSL the error is a diagnostic.
    try:
        exported(5, True, 1.5, None)
    except Exception as e:
        print("CONST_MISMATCH:", type(e).__name__, "Mismatched Meta value" in str(e))
    # EXEC: CONST_MISMATCH: ValueError True
    try:
        compiled.tvm_ffi_function("ten")
    except Exception as e:
        print("TYPE_MISMATCH:", type(e).__name__, "expected int" in str(e))
    # EXEC: TYPE_MISMATCH: TypeError True
    try:
        compiled("ten")
    except m.DSLUserCodeError as e:
        print("DSL_ERROR:", e.diag_id.name)
    # EXEC: DSL_ERROR: CALL_REJECTED


# =============================================================================
# The builder: DirectCallProvider, attach_ffi_func, rename, rejections
# =============================================================================
with ir.Context(), ir.Location.unknown():
    i32 = ir.IntegerType.get_signless(32)
    direct = DirectCallProvider(
        "twice", result_type=i32, result_signed=False, callee_kind="func"
    )
    print(
        "DIRECT:",
        direct.target_func,
        direct.result_type,
        direct.result_signed,
        direct.callee_kind,
    )
    expect("CALLEE_KIND", lambda: DirectCallProvider("f", callee_kind="bogus"))
    # CHECK: DIRECT: twice i32 False func
    # CHECK: CALLEE_KIND INTERNAL DirectCallProvider: unknown callee kind `bogus`

    def attach(name, params, provider=None):
        module = ir.Module.create()
        attach_ffi_func(module, name, params, provider or NopCallProvider())
        return module

    module = attach("old", [spec.Var("n", "int32")])
    rename_tvm_ffi_function(module, "old", "new")
    print(
        "RENAMED:",
        "llvm.func @__tvm_ffi_new(" in str(module),
        "@__tvm_ffi_old" in str(module),
    )
    expect("RENAME_MISSING", lambda: rename_tvm_ffi_function(module, "nope", "x"))
    # CHECK: RENAMED: True False
    # CHECK: RENAME_MISSING INTERNAL Function '@__tvm_ffi_nope' not found in the module.

    expect("LANES", lambda: attach("lanes", [spec.Var("v", "int32x2")]))
    expect(
        "ENV_STREAM",
        lambda: attach("env", [spec.EnvStream("env"), spec.Var("n", "int32")]),
    )
    vec = ir.Module.parse(
        'module { llvm.func @vec() -> vector<2xi32> attributes {llvm.linkage = "external"} }'
    )
    vec_result = DirectCallProvider("vec", result_type=ir.Type.parse("vector<2xi32>"))
    expect("VECTOR_RESULT", lambda: attach_ffi_func(vec, "vec", [], vec_result))
    # CHECK: LANES UNSUP_PARAM
    # CHECK: error[tvm_ffi:UNSUP_PARAM]:{{.*}} The TVM-FFI export does not support this parameter: Unsupported Var dtype: int32x2.
    # CHECK: ENV_STREAM UNSUP_PARAM
    # CHECK: {{.*}}EnvStream cannot be detected in `env(n: int32)` we need parameters to contain GPU Tensors.
    # CHECK: VECTOR_RESULT UNSUP_PARAM
    # CHECK: {{.*}}unsupported result type vector<2xi32>.


# =============================================================================
# spec: the parameter kinds, their signature text, the f4x2 conversion
# =============================================================================
n = spec.Var("n", "int32", divisibility=16)
t = spec.Tensor("x", [n, 128], "float32")
dp = spec.DataPointer("p")
pair = spec.TupleParam("pair", [spec.Var("p0", "int32"), dp, spec.ConstInt("k", 4)])
consts_ = [spec.ConstBool("b", 1), spec.ConstFloat("f", 2), spec.ConstNone("opt")]
handles = [spec.Stream("s"), spec.EnvStream("env")]  # an EnvStream has no slot
print(
    "SIG:",
    spec.signature(
        "f", [n, t, spec.Shape("dims", [n, 4]), dp, *consts_, pair, *handles]
    ),
)
# Const* values are normalised to int/bool/float.
# CHECK: SIG: f(n: int32, x: Tensor([n, 128], float32), dims: Shape([n, 4]), p: DataPointer, b: Bool(True), f: Float(2.0), opt: None, pair: Tuple[int32, DataPointer, Int(4)], s: Stream)
print(
    "TENSOR:",
    t.data.name,
    str(t.data.dtype),
    t.device_id.name,
    t.dlpack_device_type,
    t.device_type_name,
)
# CHECK: TENSOR: x.data handle x.device.index 2 cuda
with spec.DefaultConfig(device_type="cpu"):
    cpu = spec.Tensor("c", [n], "int8")
print(
    "DEVICE:",
    cpu.dlpack_device_type,
    cpu.device_type_name,
    spec.DefaultConfig.current().device_type,
)
# CHECK: DEVICE: 1 cpu cuda
f4 = spec.Tensor("f4", [n, 8], "float4_e2m1fn", device_type="cpu", data_alignment=16)
packed = spec.create_map_tensor_dtype_f4x2_to_f4_spec(f4)
print(
    "F4X2:",
    spec.format_param_type(packed),
    packed.map_tensor_dtype_f4x2_to_f4,
    packed.data_alignment,
)
# CHECK: F4X2: Tensor([n, 4], float4_e2m1fnx2) True 16
odd = spec.Tensor("o", [n, 7], "float4_e2m1fn")
expect("F4X2_ODD", lambda: spec.create_map_tensor_dtype_f4x2_to_f4_spec(odd))
# CHECK: F4X2_ODD UNSUP_PARAM
# CHECK: {{.*}}Dimension 1 with stride=1 must be even.


class Foreign(spec.Param):
    name = "foreign"


expect("FOREIGN", lambda: spec.signature("f", [n, Foreign()]))
# CHECK: FOREIGN UNSUP_PARAM
# CHECK: {{.*}}Unsupported parameter type: <class '__main__.Foreign'>.
