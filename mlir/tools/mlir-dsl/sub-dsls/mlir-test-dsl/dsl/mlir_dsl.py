# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The test sub-DSL ``MlirTestDSL``: the ``mlir.dsl`` core assembled from the shipped plugins.

``MlirTestDSL`` is one ``BaseDSL`` subclass naming one ``Plugins`` record: the
builtin types over ``arith``/``math``/``vector``/``llvm`` ops, a ``func.func``
host entry, gpu kernels, Python control flow as ``scf``, the ``mlir`` pass
manager and execution engine as its compiler, and the ``dlpack`` and
``tvm_ffi`` adapters (numpy arrays and torch tensors arrive through DLPack). A plugin whose dependency is absent is dropped from the
record at the first construction. ``MLIR_DSL_*`` is its environment prefix. A
sub-DSL of your own is the same shape with another record
(``examples/11_custom_dsl.py``).
"""

from collections.abc import Callable
from typing import Any

from mlir.dsl.core.common import DSLUserCodeError
from mlir.dsl.core.diagnostics import DiagId, register_dsl_package
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.plugins.ast_preprocessor import scf
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.adapters import dlpack, tvm_ffi
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.decorators.kernels import gpu_plugin
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith, llvm, vector

__all__ = ["LOWER_TO_LLVM", "MlirTestDSL", "compile", "jit", "kernel"]

#: The passes lowering what the test DSL emits (arith, math, vector and llvm ops
#: on builtin types, the ``scf`` control flow, the ``func.func`` entry) to the
#: LLVM dialect, in order. The sub-DSL owns its pipeline; no plugin publishes
#: passes. A DSL built from the same parts lists these in its ``pipeline()``.
LOWER_TO_LLVM: tuple[str, ...] = (
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-vector-to-llvm",
    "convert-arith-to-llvm",
    "convert-math-to-llvm",
    "convert-func-to-llvm",
    "reconcile-unrealized-casts",
)

# Frames of this package are DSL-internal when a diagnostic locates user code.
register_dsl_package("mlir.mlir_dsl")


class MlirTestDSL(BaseDSL):
    """The test sub-DSL: the shipped plugins in one record.

    The gpu kernels plugin, the compiler and the adapters are dropped when
    their dependency is absent (``available()``), so the same class traces on
    a machine without the gpu bindings or the execution engine.
    """

    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
        ast_preprocessor=scf.ASTPreprocessor(),
        compiler=execution_engine.Compiler(),
        # @jit over a func.func host entry, then @kernel over gpu.func.
        decorators=[func.Jit(), gpu_plugin.Kernels(chip_option="cubin-chip")],
        # Inbound: anything speaking DLPack (numpy arrays, torch tensors on the
        # host or a device) through the nanobind extension. Outbound: the
        # compiled entry as a tvm_ffi.Function when enabled.
        adapters=[dlpack.DlpackPlugin(), tvm_ffi.TvmFfiPlugin()],
    )

    def pipeline(self) -> list[str]:
        """The gpu lowering first, when an architecture is set (a launch without
        one that is compiled is a ``gpu:CONFIG_MISSING_ARCH`` diagnostic; a dry run
        needs none), then the lowering
        of the upstream dialects the builtin types and the ``scf`` control flow
        emit."""
        passes: list[str] = []
        kernels = self.plugins.named("gpu")
        if kernels is not None and self.envar.arch:
            # MLIR's upstream gpu lowering (outlining, NVVM, gpu-module-to-binary)
            # for the architecture; the plugin only names the chip option.
            passes.append(
                f"gpu-lower-to-nvvm-pipeline{{{kernels.chip_option}={self.envar.arch}}}"
            )
        passes.extend(LOWER_TO_LLVM)
        return passes

    def __init__(self) -> None:
        super().__init__(name="MLIR_DSL")


jit = MlirTestDSL.jit
kernel = MlirTestDSL.kernel


def compile(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Compile a ``@jit`` function for ``args`` without running it.

    :param func: A function decorated with ``jit``
    :param args: Representative arguments; runtime values fix the argument
        types, Python values specialise the compiled function
    :return: The compiled function, callable with matching runtime arguments
    """
    if not hasattr(func, "_dsl_cls"):
        raise DSLUserCodeError(DiagId.CALL_MISSING_JIT_DECORATOR)
    return func(*args, compile_only=True, **kwargs)
