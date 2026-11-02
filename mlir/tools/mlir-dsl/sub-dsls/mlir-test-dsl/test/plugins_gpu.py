# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
from dataclasses import replace
from typing import Annotated
import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.decorators.kernels import gpu_plugin as g
dsl = m.MlirTestDSL()
kernels = dsl.plugins.decorator('kernel')
def report(fn, *args, **kwargs):
    fn(*args, **kwargs)
    print('OK', fn.__name__)
class _Tag:
    """A test-only `Annotated` marker; the core turns it into an argument attribute."""

    def __extract_mlir_attributes__(self):
        return [ir.DictAttr.get({"test.tag": ir.UnitAttr.get()})]

containers = []
launched = []


class NamedKernels(g.Kernels):
    """A target with its own symbol scheme, `k_<name>_<count>`, nothing mangled
    in; it reads the container and the host module of the trace in progress."""

    def kernel_name(self, dsl, func, base, count, *, cargs, signature):
        containers.append((self.container.operation.name, self.module.operation.name))
        return f"k_{base}_{count}"


class ByvalKernels(g.Kernels):
    """A target that marks every pointer operand of a kernel `{test.byval}`."""

    def kernel_arg_attrs(self, jit_args, signature):
        attrs = []
        for ty, attr in zip(
            jit_args.types, super().kernel_arg_attrs(jit_args, signature)
        ):
            entries = {na.name: na.attr for na in attr}
            if str(ty) == "!llvm.ptr":
                entries["test.byval"] = ir.UnitAttr.get()
            attrs.append(ir.DictAttr.get(entries))
        return attrs


class ResultKernels(g.Kernels):
    """A target whose launch hands the kernel body's Python result to the host."""

    def check_kernel_result(self, result):
        return result

    def launch_result(self, launch_ret, kernel_result):
        return kernel_result


class NamedDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit(), NamedKernels()])


class ByvalDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit(), ByvalKernels()])


class ResultDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit(), ResultKernels()])


def named_launch():
    @NamedDSL.kernel
    def axpy(a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
        y[0] = a * x[0] + y[0]

    @NamedDSL.jit
    def named_host(a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
        axpy(a, x, y).launch(grid=[1], block=[1])
        axpy(a, x, y).launch(grid=[1], block=[1])

    return named_host


# A target names the kernel symbols (`kernel_name`: the Python name and the
# index within the trace here, no mangled compile-time arguments) and reads the
# container and the host module of the trace in progress, which are public and
# cleared when the trace ends.
# CHECK-LABEL: gpu.func @k_axpy_0(
# CHECK-LABEL: gpu.func @k_axpy_1(
# CHECK:       gpu.launch_func @kernels::@k_axpy_0
# CHECK:       gpu.launch_func @kernels::@k_axpy_1
# CHECK:       OK named_host
# CHECK:       CONTAINER: [('gpu.module', 'builtin.module'), ('gpu.module', 'builtin.module')]
# CHECK:       PUBLIC: None None False
report(named_launch(), 2.0, 0, 0)
print("CONTAINER:", containers)
print("PUBLIC:", kernels.container, kernels.module, hasattr(kernels, "_container"))


@ByvalDSL.kernel
def byval(
    x: Annotated[m.Pointer[m.Float32], _Tag()], n: m.Int32, y: m.Pointer[m.Int32]
):
    y[0] = n


@ByvalDSL.jit
def byval_host(x: m.Pointer[m.Float32], n: m.Int32, y: m.Pointer[m.Int32]):
    byval(x, n, y).launch()


# A target adds argument attributes to the kernel's parameters
# (`kernel_arg_attrs`, over what the argument boundary produced, here the
# `Annotated` marker of `x`): the pointer operands, the scalar is left alone.
# CHECK-LABEL: gpu.func @kernel_byval_0(
# CHECK-SAME:    %{{[^:]+}}: !llvm.ptr {test.byval, test.tag}, %{{[^:]+}}: i32, %{{[^:]+}}: !llvm.ptr {test.byval}) kernel
# CHECK:       OK byval_host
report(byval_host, 0, 1, 0)


@ResultDSL.kernel
def counted(x: m.Pointer[m.Float32], n):
    x[0] = m.Float32(n)
    return n * 2  # a Python value: `n` is a compile-time argument


@ResultDSL.jit
def result_host(x: m.Pointer[m.Float32]):
    launched.append(counted(x, 4).launch(grid=[1], block=[1]))
    launched.append(counted(x, 5).launch(grid=[1], block=[1]))


# A target gives the kernel body's Python result a meaning: `check_kernel_result`
# accepts it (the shipped plugin rejects any result, see `returns` above) and
# `launch_result` makes `.launch()` yield it to the host body.
# CHECK-LABEL: gpu.func @kernel_counted_4_0(
# CHECK-LABEL: gpu.func @kernel_counted_5_1(
# CHECK:       OK result_host
# CHECK:       LAUNCHED: [8, 10]
report(result_host, 0)
print("LAUNCHED:", launched)
