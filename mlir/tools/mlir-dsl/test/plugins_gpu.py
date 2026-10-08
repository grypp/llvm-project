# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 MLIR_DSL_ARCH=sm_90 %PYTHON %s 2>&1 | FileCheck %s
# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=NOARCH
# The gpu kernels plugin: `gpu.Kernels` is the decorator
# plugin in MlirTestDSL's `Plugins` record that adds `@kernel`, installed
# whenever the gpu bindings are present (kernel function, `gpu.module`
# container, launch); it validates the arch at install and merges its chip
# option into a `<PREFIX>_PIPELINE` override through `pipeline_options()`;
# `MlirTestDSL.pipeline` puts `gpu-lower-to-nvvm-pipeline{cubin-chip=<arch>}`
# ahead of the LLVM lowering when an arch is set. Calling a `@kernel` prepares a deferred launch; its
# `.launch` emits the `gpu.func` (gpu.kernel, `known_block_size` for a static
# block) in `gpu.module @kernels` and a synchronous `gpu.launch_func` with i64
# dimensions and the host values flattened to operands; the index helpers read
# the NVVM special registers as Int32. The launch validates its LaunchConfig and
# the buffer kinds and raises the namespaced `gpu:` diagnostics. Traced only (no
# GPU, no CUDA toolkit); the lowering is MLIR's. Without an arch the plugin is
# still installed and the first launch diagnoses the missing arch.
import os
from dataclasses import replace
from typing import Annotated

import numpy as np

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.decorators.kernels import gpu as g


def report(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except m.DSLUserCodeError as e:
        print(str(e))
    else:
        print("OK", fn.__name__)


# =============================================================================
# Availability, the target and the pipeline
# =============================================================================
print(
    "AVAILABLE:",
    g.Kernels.available(),
    isinstance(m.MlirTestDSL.plugins.named("gpu"), g.Kernels),
)
# The kernels plugin is named in MlirTestDSL's record; `available()` says whether
# the gpu bindings are built, with or without an arch.
# CHECK:  AVAILABLE: True True
# NOARCH: AVAILABLE: True True
rejected = []
for arch in (None, "", "sm90", "gfx90a", "SM_90", "sm_90b", 90):
    try:
        g.check_arch(arch, var="MLIR_DSL_ARCH")
    except m.DSLUserCodeError as e:
        rejected.append(e.diag_id.name)
print(
    "ARCH:",
    [g.check_arch(a, var="MLIR_DSL_ARCH") for a in ("sm_80", "sm_90a", "sm_100f")],
    set(rejected),
    len(rejected),
)
# CHECK:  ARCH: ['sm_80', 'sm_90a', 'sm_100f'] {'CONFIG_UNSUPPORTED_ARCH'} 7
# NOARCH: ARCH: ['sm_80', 'sm_90a', 'sm_100f'] {'CONFIG_UNSUPPORTED_ARCH'} 7
# CHECK:  error[gpu:CONFIG_UNSUPPORTED_ARCH]:{{.*}} The GPU architecture `<unset>` is not a CUDA target this DSL can compile for; `MY_ARCH` must name one such as `sm_80` or `sm_90a`.
# CHECK:  suggestion:{{.*}}Set the environment variable `MY_ARCH=<arch>`, e.g. `MY_ARCH=sm_90a`.
report(g.check_arch, "", var="MY_ARCH")
print(
    "DIAGS:",
    g.GpuDiagId.namespace,
    [d.name for d in g.GpuDiagId],
    g.Kernels.diag_ids is g.GpuDiagId,
)
# CHECK:  DIAGS: gpu ['LAUNCH_INVALID_DIMENSION', 'LAUNCH_INVALID_GRID', 'LAUNCH_OUTSIDE_JIT', 'LAUNCH_NEVER_ISSUED', 'LAUNCH_ALREADY_ISSUED', 'LAUNCH_HOST_BUFFER', 'LAUNCH_STREAM_UNSUPPORTED', 'CONFIG_UNSUPPORTED_ARCH'] True


class GpuDSL(m.MlirTestDSL):
    # The plugin named explicitly with its chip option; the record installs a
    # copy bound to the instance (NOARCH still builds it).
    plugins = replace(
        m.MlirTestDSL.plugins,
        decorators=[func.Jit(), g.Kernels(chip_option="cubin-chip")],
    )


dsl = GpuDSL()
print(
    "INSTALLED:",
    repr(dsl.envar.arch),
    isinstance(dsl.plugins.named("gpu"), g.Kernels),
)
print(
    "SEAMS:",
    dsl.plugins.named("gpu") is not GpuDSL.plugins.named("gpu"),
    dsl.plugins.named("gpu").dsl is dsl,
    dsl.plugins.named("gpu").pipeline_options(),
)
print("PASSES:", [p for p in dsl.pipeline() if p.startswith("gpu-")])
print("PIPELINE:", dsl._get_pipeline(None))
# Without an arch the DSL's pipeline has no gpu pass (the launch diagnoses
# the missing arch).
# CHECK:  INSTALLED: 'sm_90' True
# CHECK:  SEAMS: True True {'cubin-chip': 'sm_90'}
# CHECK:  PASSES: ['gpu-lower-to-nvvm-pipeline{cubin-chip=sm_90}']
# CHECK:  PIPELINE: builtin.module(gpu-lower-to-nvvm-pipeline{cubin-chip=sm_90},convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)
# NOARCH: INSTALLED: {{None|''}} True
# NOARCH: SEAMS: True True {}
# NOARCH: PASSES: []
# NOARCH: PIPELINE: builtin.module(convert-scf-to-cf,convert-cf-to-llvm,
dsl.envar.arch = "sm_80"  # the arch is a property of the instance
print("OVERRIDE:", [p for p in dsl.pipeline() if p.startswith("gpu-")])
del dsl.envar.arch
# CHECK:  OVERRIDE: ['gpu-lower-to-nvvm-pipeline{cubin-chip=sm_80}']
# NOARCH: OVERRIDE: ['gpu-lower-to-nvvm-pipeline{cubin-chip=sm_80}']

# An invalid MLIR_DSL_ARCH fails at DSL construction.
saved_arch = os.environ.get("MLIR_DSL_ARCH")
os.environ["MLIR_DSL_ARCH"] = "gfx90a"


class BadArchDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit(), g.Kernels()])


report(BadArchDSL)
# CHECK:  error[gpu:CONFIG_UNSUPPORTED_ARCH]:{{.*}} The GPU architecture `gfx90a` is not a CUDA target this DSL can compile for; `MLIR_DSL_ARCH` must name one
# NOARCH: error[gpu:CONFIG_UNSUPPORTED_ARCH]:{{.*}} `gfx90a`
if saved_arch is None:
    del os.environ["MLIR_DSL_ARCH"]
else:
    os.environ["MLIR_DSL_ARCH"] = saved_arch


# =============================================================================
# A kernel launched from a host: the module shape
# =============================================================================
@m.kernel
def axpy(n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
    tx, _, _ = m.thread_idx()
    bx, _, _ = m.block_idx()
    bdx, _, _ = m.block_dim()
    i = bx * bdx + tx
    if i < n:
        y[i] = a * x[i] + y[i]


@m.jit
def axpy_host(
    n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]
):
    axpy(n, a, x, y).launch(grid=[4], block=[128])


# CHECK-LABEL: module attributes {gpu.container_module} {
# CHECK-NEXT:    gpu.module @kernels {
# CHECK-NEXT:      gpu.func @kernel_axpy_0(%[[N:[^:]+]]: i32, %[[A:[^:]+]]: f32, %[[X:[^:]+]]: !llvm.ptr, %[[Y:[^:]+]]: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>, sym_visibility = "public"} {
# CHECK-NEXT:        %[[TX:.+]] = nvvm.read.ptx.sreg.tid.x : i32
# CHECK:             %[[BX:.+]] = nvvm.read.ptx.sreg.ctaid.x : i32
# CHECK:             %[[BDX:.+]] = nvvm.read.ptx.sreg.ntid.x : i32
# CHECK:             %[[MUL:.+]] = arith.muli %[[BX]], %[[BDX]] : i32
# CHECK-NEXT:        %[[I:.+]] = arith.addi %[[MUL]], %[[TX]] : i32
# CHECK-NEXT:        %[[COND:.+]] = arith.cmpi slt, %[[I]], %[[N]] : i32
# CHECK-NEXT:        scf.if %[[COND]] {
# CHECK:               llvm.getelementptr %[[X]][%[[I]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
# CHECK:             gpu.return
# CHECK:           func.func @axpy_host(%[[HN:[^:]+]]: i32, %[[HA:[^:]+]]: f32, %[[HX:[^:]+]]: !llvm.ptr, %[[HY:[^:]+]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
# CHECK-DAG:         %[[G0:.+]] = arith.constant 4 : i64
# CHECK-DAG:         %[[B0:.+]] = arith.constant 128 : i64
# CHECK:             gpu.launch_func @kernels::@kernel_axpy_0 blocks in (%[[G0]], %{{.+}}, %{{.+}}) threads in (%[[B0]], %{{.+}}, %{{.+}}) : i64 args(%[[HN]] : i32, %[[HA]] : f32, %[[HX]] : !llvm.ptr, %[[HY]] : !llvm.ptr)
# CHECK-NEXT:        return
# CHECK-NOT:         index
# CHECK-NOT:         gpu.thread_id
# CHECK:           OK axpy_host
# NOARCH:          error[gpu:CONFIG_UNSUPPORTED_ARCH]:{{.*}} The GPU architecture `<unset>` is not a CUDA target this DSL can compile for
report(axpy_host, 512, 2.0, 0, 0)


# =============================================================================
# Kernel arguments
# =============================================================================
@m.struct
class Params:
    n: m.Int32
    scale: m.Float32


Pointers = tuple[m.Pointer[m.Float32], m.Pointer[m.Float32]]


@m.kernel
def scaled(p: Params, xs: Pointers, k, v: m.Float32):
    tx, _, _ = m.thread_idx()
    if tx < p.n:
        for i in range(k):
            xs[0][tx * k + i] = xs[1][tx] * p.scale + v


@m.jit
def args_host(p: Params, xs: Pointers):
    scaled(p, xs, 2, 1.5).launch(grid=1, block=32)
    scaled(p, xs, 3, 2.5).launch(grid=1, block=32)


# A struct flattens to one operand per field, a tuple to one operand per
# pointer, a Python literal is a constant operand and an unannotated argument
# is a compile-time value baked into the kernel and its name
# (`kernel_<name>_<meta values>_<count>`); the loop unrolls.
# CHECK-LABEL: gpu.func @kernel_scaled_2_0(
# CHECK-SAME:    %[[PN:[^:]+]]: i32, %[[PF:[^:]+]]: f32, %[[DST:[^:]+]]: !llvm.ptr, %[[SRC:[^:]+]]: !llvm.ptr, %[[V:[^:]+]]: f32) kernel
# CHECK:         arith.cmpi slt, %{{.+}}, %[[PN]] : i32
# CHECK-COUNT-2: llvm.store
# CHECK-NOT:     llvm.store
# CHECK:         gpu.return
# CHECK-LABEL: gpu.func @kernel_scaled_3_1(
# CHECK-COUNT-3: llvm.store
# CHECK:       func.func @args_host(
# CHECK-SAME:    %[[HN:[^:]+]]: i32, %[[HF:[^:]+]]: f32, %[[HA:[^:]+]]: !llvm.ptr, %[[HB:[^:]+]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
# CHECK-DAG:     %[[C15:.+]] = arith.constant 1.500000e+00 : f32
# CHECK:         gpu.launch_func @kernels::@kernel_scaled_2_0 blocks in ({{.*}}) threads in ({{.*}}) : i64 args(%[[HN]] : i32, %[[HF]] : f32, %[[HA]] : !llvm.ptr, %[[HB]] : !llvm.ptr, %[[C15]] : f32)
# CHECK:         gpu.launch_func @kernels::@kernel_scaled_3_1
# CHECK:       OK args_host
ptr = m.Pointer(0, dtype=m.Float32)
report(args_host, Params(n=4, scale=2.0), (ptr, ptr))


@m.kernel
def constant_ptr(p: Annotated[m.Pointer, m.grid_constant], n: m.Int32):
    pass


@m.kernel
def returns(x: m.Pointer[m.Float32]) -> m.Int32:
    return m.Int32(1)


@m.jit
def grid_constant_host(p: m.Pointer[m.Float32], n: m.Int32):
    constant_ptr(p, n).launch()


@m.jit
def return_host(x: m.Pointer[m.Float32]):
    returns(x).launch()


@m.jit
def too_few(x: m.Pointer[m.Float32]):
    constant_ptr(x).launch()


# CHECK-LABEL: gpu.func @kernel_constant_ptr_0(
# CHECK-SAME:    %{{[^:]+}}: !llvm.ptr {cuda.grid_constant}, %{{[^:]+}}: i32) kernel
# CHECK:       OK grid_constant_host
# CHECK:       error[TYPE_RETURN_MISMATCH]:{{.*}} This function returns a `Int32`, which a compiled function cannot return from a kernel.
# CHECK:       suggestion:{{.*}}A kernel cannot return a value: write its results through a `Pointer` argument
# CHECK:       error[CALL_ARGUMENTS]:{{.*}} The call to `{{.*}}` does not match its parameters: 1 positional and 0 keyword argument(s) do not bind
report(grid_constant_host, 0, 1)
report(return_host, 0)
report(too_few, 0)


# =============================================================================
# Deferred launches
# =============================================================================
@m.kernel
def fill(x: m.Pointer[m.Float32], v: m.Float32):
    tx, _, _ = m.thread_idx()
    x[tx] = v


@m.jit
def deferred(x: m.Pointer[m.Float32], n: m.Int32):
    launcher = fill(x, 3.0)  # prepares the launch only
    x[0] = m.Float32(4.0)
    launcher.launch(grid=[1], block=[4])  # emitted here
    if n > 0:
        fill(x, 1.0).launch(grid=[1], block=[1])
    for i in range(n):
        fill(x, 2.0).launch(grid=[1], block=[2])


# Every launch traces its own `gpu.func`, numbered within the trace.
# CHECK-LABEL: func.func @deferred(
# CHECK-SAME:    %[[X:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32) attributes {llvm.emit_c_interface} {
# CHECK:         llvm.store %{{.+}}, %{{.+}} <alignment = 4> : f32, !llvm.ptr
# CHECK:         gpu.launch_func @kernels::@kernel_fill_0 blocks in ({{.*}}) threads in ({{.*}}) : i64 args(%[[X]] : !llvm.ptr, %{{.+}} : f32)
# CHECK:         scf.if %{{.+}} {
# CHECK:           gpu.launch_func @kernels::@kernel_fill_1
# CHECK:         scf.for %{{.+}} = %{{.+}} to %[[N]] step %{{.+}} : i32 {
# CHECK:           gpu.launch_func @kernels::@kernel_fill_2
# CHECK:       OK deferred
report(deferred, 0, 2)
# The kernel records live on the plugin and describe the last trace (reset
# when the next trace starts).
kernels = m.MlirTestDSL().plugins.named("gpu")
print(
    "LAST TRACE:", kernels.num_kernels, kernels.launch_count, list(kernels.kernel_info)
)
# Without an arch every launch stops at the arch check after its kernel was
# built, so the last trace counts one kernel and no launch.
# CHECK:  LAST TRACE: 3 3 ['kernel_fill_0', 'kernel_fill_1', 'kernel_fill_2']
# NOARCH: LAST TRACE: 1 0 []


@m.jit
def never(x: m.Pointer[m.Float32]):
    fill(x, 1.0)


# CHECK:      error[gpu:LAUNCH_NEVER_ISSUED]:{{.*}} Kernel `fill` was called but never launched. Calling a `@kernel` function only prepares a launch; the kernel does not run until `.launch(...)` is called on the result.
# CHECK:      -->{{.*}}plugins_gpu.py:[[#@LINE-4]]:5
# CHECK:      suggestion:{{.*}}Launch the kernel, e.g. `fill(...).launch(grid=[...], block=[...])`.
report(never, 0)


@m.jit
def twice(x: m.Pointer[m.Float32]):
    launcher = fill(x, 1.0)
    launcher.launch()
    launcher.launch()


# CHECK:      error[gpu:LAUNCH_ALREADY_ISSUED]:{{.*}} Kernel `fill` is launched twice from one prepared call; `fill(...)` runs once.
# CHECK:      suggestion:{{.*}}Call `fill(...)` again for a second launch, one `.launch(...)` each.
# CHECK:      error[gpu:LAUNCH_OUTSIDE_JIT]:{{.*}} Kernel `fill` is being launched from plain Python, but a kernel can only be launched from inside a function decorated with `@jit`.
# CHECK:      suggestion:{{.*}}Wrap the launch in a host function decorated with `@jit` and call that.
report(twice, 0)
report(lambda: fill(0, 1.0).launch())


# =============================================================================
# LaunchConfig: dimensions, shared memory and their validation
# =============================================================================
@m.jit
def config(x: m.Pointer[m.Float32], n: m.Int32, nb: m.Uint32, bytes_: m.Int64):
    fill(x, 1.0).launch(m.LaunchConfig(grid=[2], block=64, smem=m.Int32(256)))
    fill(x, 1.0)(grid=n, block=[nb], cluster=[2], smem=bytes_)
    fill(x, 1.0).launch(grid=[0, 1, 1], block=[32])


# `launch` takes a LaunchConfig or its fields and the launcher is callable;
# scalar dimensions pad to three; a Python-valued Integer folds to a static
# `known_block_size`; staged dimensions are promoted to i64 by signedness and
# guarded by `scf.if (all > 0)`; `smem` is an i32 operand; a static zero grid
# emits the kernel but no launch.
# CHECK-LABEL: gpu.func @kernel_fill_0(%{{.+}}: !llvm.ptr, %{{.+}}: f32) kernel attributes {known_block_size = array<i32: 64, 1, 1>,
# CHECK-LABEL: gpu.func @kernel_fill_1(%{{.+}}: !llvm.ptr, %{{.+}}: f32) kernel attributes {sym_visibility = "public"} {
# CHECK-LABEL: gpu.func @kernel_fill_2(%{{.+}}: !llvm.ptr, %{{.+}}: f32) kernel attributes {known_block_size = array<i32: 32, 1, 1>,
# CHECK-LABEL: func.func @config(
# CHECK-SAME:    %[[X:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32, %[[NB:[^:]+]]: i32, %[[BYTES:[^:]+]]: i64) attributes {llvm.emit_c_interface} {
# CHECK-DAG:     %[[G0:.+]] = arith.constant 2 : i64
# CHECK-DAG:     %[[B0:.+]] = arith.constant 64 : i64
# CHECK-DAG:     %[[SMEM:.+]] = arith.constant 256 : i32
# CHECK:         gpu.launch_func @kernels::@kernel_fill_0 blocks in (%[[G0]], %{{.+}}, %{{.+}}) threads in (%[[B0]], %{{.+}}, %{{.+}}) : i64 dynamic_shared_memory_size %[[SMEM]] args(
# CHECK-DAG:     %[[GN:.+]] = arith.extsi %[[N]] : i32 to i64
# CHECK-DAG:     %[[BN:.+]] = arith.extui %[[NB]] : i32 to i64
# CHECK-DAG:     %[[SM:.+]] = arith.trunci %[[BYTES]] : i64 to i32
# CHECK:         %[[ZERO:.+]] = arith.constant 0 : i64
# CHECK-NEXT:    %[[GP:.+]] = arith.cmpi sgt, %[[GN]], %[[ZERO]] : i64
# CHECK-NEXT:    %[[BP:.+]] = arith.cmpi sgt, %[[BN]], %[[ZERO]] : i64
# CHECK-NEXT:    %[[ALL:.+]] = arith.andi %[[GP]], %[[BP]] : i1
# CHECK-NEXT:    scf.if %[[ALL]] {
# CHECK-NEXT:      gpu.launch_func @kernels::@kernel_fill_1 clusters in (%{{.+}}, %{{.+}}, %{{.+}}) blocks in (%[[GN]], %{{.+}}, %{{.+}}) threads in (%[[BN]], %{{.+}}, %{{.+}}) : i64 dynamic_shared_memory_size %[[SM]] args(
# CHECK-NOT:     @kernel_fill_2
# CHECK:         return
# CHECK:       OK config
report(config, 0, 4, 128, 512)


def launch_with(*launch_args, **launch_kwargs):
    @m.jit
    def host(x: m.Pointer[m.Float32]):
        fill(x, 1.0).launch(*launch_args, **launch_kwargs)

    return host


# CHECK:      error[gpu:LAUNCH_INVALID_GRID]:{{.*}} The `grid` launch argument can have at most 3 entries, but it has 4.
# CHECK:      suggestion:{{.*}}Pass at most three values for `grid`, e.g. `grid=[128, 1, 1]`.
# CHECK:      error[gpu:LAUNCH_INVALID_DIMENSION]:{{.*}} Element 1 of the `grid` launch argument is a `int` with the value -2, but every entry must be a non-negative integer (`int` or a staged `Integer`).
# CHECK:      suggestion:{{.*}}Pass integers for every entry of `grid`, e.g. `grid=[2, 1, 1]`.
# CHECK:      Element 0 of the `block` launch argument is a `int` with the value 0
# CHECK:      Element 0 of the `grid` launch argument is a `float`, but every entry
# CHECK:      Element 2 of the `block` launch argument is a `Boolean`, but every entry
# CHECK:      error[gpu:LAUNCH_STREAM_UNSUPPORTED]:{{.*}} Kernel `fill` is launched with `async_deps`, but launches are synchronous: no stream can be passed.
# CHECK:      suggestion:{{.*}}Leave `async_deps` empty; the launch completes before the call returns.
# CHECK:      error[ARG_NOT_NUMERIC]:{{.*}} Argument `smem` expects a numeric value, but this call passes a value of type `str`.
report(launch_with(grid=[1, 2, 3, 4]), 0)
report(launch_with(grid=[1, -2]), 0)
report(
    launch_with(block=0), 0
)  # a zero grid skips the launch, a zero block is an error
report(launch_with(grid=2.5), 0)
report(launch_with(block=[1, 1, m.Boolean(True)]), 0)
report(launch_with(async_deps=[0]), 0)
report(launch_with(smem="4k"), 0)


# =============================================================================
# Buffer kinds at the call boundary
# =============================================================================
@m.jit
def launches(x: m.Pointer[m.Float32]):
    fill(x, 1.0).launch()


@m.jit
def host_only(x: m.Pointer[m.Float32]):
    x[0] = m.Float32(2.0)


host_array = np.zeros(4, dtype=np.float32)
device_ptr = m.Pointer(0x1000, dtype=m.Float32, kind="device")
print("KINDS:", m.Pointer(0x1000).kind, device_ptr.kind, m.Pointer(device_ptr).kind)
# The trace sees one `!llvm.ptr` per buffer, so the side is checked on the
# argument's kind after tracing; a bare address is "unknown" and never checked.
# CHECK:      KINDS: unknown device device
# CHECK:      error[gpu:LAUNCH_HOST_BUFFER]:{{.*}} Argument `x` is a host buffer (`ndarray`), but this function launches a device kernel: device kernels take device addresses.
# CHECK:      suggestion:{{.*}}Allocate `x` on the device with your framework and pass that tensor.
# CHECK:      OK launches
# CHECK:      OK launches
# CHECK:      error[ARG_BUFFER_INVALID]:{{.*}} Argument `x` cannot be used as a `Pointer` argument: it is a device buffer, but `host_only` runs on the host (its trace launched no kernel).
# CHECK:      suggestion:{{.*}}Pass one contiguous block of memory on the device the function runs on
# CHECK:      OK host_only
report(launches, host_array)
report(launches, device_ptr)
report(launches, 0x1000)
report(host_only, device_ptr)
report(host_only, host_array)


# =============================================================================
# The index helpers outside a kernel body
# =============================================================================
@m.jit
def host_tid(n: m.Int32) -> m.Int32:
    tx, _, _ = m.thread_idx()
    return tx + n


# CHECK:      error[CALL_OUTSIDE_JIT]:{{.*}} `thread_idx()` was called from plain Python, but it can only be used inside a function decorated with `@kernel`.
# CHECK:      suggestion:{{.*}}Move this call into a function decorated with `@kernel`, then call that function.
# CHECK:      error[CALL_OUTSIDE_JIT]:{{.*}} `grid_dim()` was called from plain Python
report(host_tid, 1)  # a `@jit` host body is not a kernel body
report(m.grid_dim)


# =============================================================================
# A DSL without the plugin, one with it but no arch, and a sub-DSL's kernel entry
# =============================================================================
class CpuDSL(m.MlirTestDSL):
    plugins = replace(
        m.MlirTestDSL.plugins, decorators=[func.Jit()]
    )  # no kernels plugin


@CpuDSL.kernel
def cpu_kernel(x: m.Pointer[m.Float32]):
    pass


@CpuDSL.jit
def cpu_host(x: m.Pointer[m.Float32]):
    cpu_kernel(x).launch()


# CHECK:  error[CALL_PLUGIN_REQUIRED]:{{.*}} `@kernel` needs the `Kernels` plugin, which this DSL does not name.
# CHECK:  suggestion:{{.*}}Name it: `plugins = Plugins(..., decorators=[Kernels()])`.
# NOARCH: error[CALL_PLUGIN_REQUIRED]:{{.*}} `@kernel` needs the `Kernels` plugin
report(cpu_host, 0)


@GpuDSL.kernel
def noop(x: m.Pointer[m.Float32]):
    pass


@GpuDSL.jit
def explicit_host(x: m.Pointer[m.Float32]):
    noop(x).launch()


# CHECK:      OK explicit_host
# NOARCH:     error[gpu:CONFIG_UNSUPPORTED_ARCH]:{{.*}} The GPU architecture `<unset>` is not a CUDA target this DSL can compile for; `MLIR_DSL_ARCH` must name one
# NOARCH-NOT: OK explicit_host
report(explicit_host, 0)


class TaggedKernels(g.Kernels):
    """A sub-DSL's kernels plugin: tags every kernel function it builds."""

    def generate_func_op(self, name, arg_types, arg_attrs, loc=None):
        fop, block = super().generate_func_op(name, arg_types, arg_attrs, loc)
        fop.attributes["tagged.by"] = ir.StringAttr.get("TaggedDSL")
        return fop, block


class TaggedDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit(), TaggedKernels()])


@TaggedDSL.kernel
def tagged(x: m.Pointer[m.Float32]):
    pass


@TaggedDSL.jit
def tagged_host(x: m.Pointer[m.Float32]):
    tagged(x).launch(grid=[2], block=[8])


# CHECK-LABEL: gpu.func @kernel_tagged_0(
# CHECK-SAME:    kernel attributes {tagged.by = "TaggedDSL", known_block_size = array<i32: 8, 1, 1>, sym_visibility = "public"} {
# CHECK:       OK tagged_host
report(tagged_host, 0)
