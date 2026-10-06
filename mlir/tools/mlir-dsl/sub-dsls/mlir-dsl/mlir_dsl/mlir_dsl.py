# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The reference sub-DSL ``MlirDSL``: the ``mlir.dsl`` core assembled with the shipped plugins.

``MlirDSL`` is one ``BaseDSL`` subclass: Python control flow through the
``scf`` AST preprocessor plugin, the core's default dialects (``scf`` and the
LLVM world), the ``gpu``, ``tvm_ffi``, ``pytorch`` and ``dlpack`` plugins when
each is available, the ``mlir`` pass manager and execution engine as its
compiler, and ``MLIR_DSL_*`` as its environment prefix. A sub-DSL of your own
is the same assembly with another plugin list (``examples/11_custom_dsl.py``).
"""

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from mlir.dsl.core.common import DSLUserCodeError
from mlir.dsl.core.diagnostics import DiagId, register_dsl_package
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.core.plugin import ASTPreprocessorPlugin, Plugin
from mlir.dsl.plugins.ast_preprocessor import ScfASTPreprocessorPlugin

try:
    from mlir.dsl.plugins.dialects import gpu as _gpu
except ImportError:  # the gpu plugin module or the gpu/nvvm bindings are absent
    _gpu = None  # type: ignore[assignment]

try:
    from mlir.dsl.plugins.thirdparty import tvm_ffi as _tvm_ffi
except ImportError:  # the export plugin module is absent
    _tvm_ffi = None  # type: ignore[assignment]

try:
    from mlir.dsl.plugins.thirdparty import pytorch as _pytorch
except ImportError:  # the pytorch plugin module is absent
    _pytorch = None  # type: ignore[assignment]

try:
    from mlir.dsl.plugins.thirdparty import dlpack as _dlpack
except ImportError:  # the dlpack plugin module is absent
    _dlpack = None  # type: ignore[assignment]

__all__ = ["MlirDSL", "compile", "jit", "kernel"]

# Frames of this package are DSL-internal when a diagnostic locates user code.
register_dsl_package("mlir.mlir_dsl")


class _DefaultPlugins:
    """Descriptor resolving ``MlirDSL.plugins`` once, on first access."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.owner, self.attr = owner, name

    def __get__(self, obj: Any, cls: type | None = None) -> list[Plugin]:
        candidates = [
            plugin()
            for plugin in (
                _gpu.GpuPlugin if _gpu is not None else None,
                _tvm_ffi.TvmFfiPlugin if _tvm_ffi is not None else None,
                _pytorch.PyTorchPlugin if _pytorch is not None else None,
                _dlpack.DlpackPlugin if _dlpack is not None else None,
            )
            if plugin is not None and plugin.available()
        ]
        setattr(self.owner, self.attr, candidates)
        return candidates


class MlirDSL(BaseDSL):
    """The default sub-DSL: ``func.func`` host entries, scalar types on
    ``arith``, Python control flow as ``scf`` and, when available, the ``gpu``,
    ``tvm_ffi``, ``pytorch`` and ``dlpack`` plugins."""

    # The gpu, tvm_ffi, pytorch and dlpack plugins, each when ``available()``;
    # resolved on first access (the first instance, or a subclass reading
    # ``MlirDSL.plugins``), so a dependency probe such as ``MLIR_DSL_ARCH``
    # counts at that time rather than at import.
    plugins: ClassVar[Sequence[Plugin]] = _DefaultPlugins()  # type: ignore[assignment]
    # The AST preprocessor of MlirDSL; a sub-DSL listing another
    # ``ASTPreprocessorPlugin`` in ``plugins`` replaces it.
    default_ast_preprocessor: ClassVar[
        ASTPreprocessorPlugin | None
    ] = ScfASTPreprocessorPlugin()
    _jit_arg_adapter_scope = "mlir"

    def __init__(self) -> None:
        pass_sm_arch_name = "cubin-chip"
        super().__init__(
            name="MLIR_DSL",
            dsl_package_name=["mlir", "mlir_dsl"],
            pass_sm_arch_name=pass_sm_arch_name,
            preprocess=True,
        )


jit = MlirDSL.jit
kernel = MlirDSL.kernel


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
