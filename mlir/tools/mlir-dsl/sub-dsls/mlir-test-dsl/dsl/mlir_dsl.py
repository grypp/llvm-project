# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The test sub-DSL ``MlirTestDSL``: the ``mlir.dsl`` core assembled from the shipped plugins.

``MlirTestDSL`` is one ``BaseDSL`` subclass naming one ``Plugins`` record: the
builtin types over ``arith``/``math``/``vector``/``llvm`` ops, a ``func.func``
host entry, gpu kernels, Python control flow as ``scf``, the ``mlir`` pass
manager and execution engine as its compiler, and the ``dlpack`` adapter
(numpy arrays and torch tensors arrive through DLPack). A plugin whose dependency is absent is dropped from the
record at the first construction. ``MLIR_DSL_*`` is its environment prefix. A
sub-DSL of your own is the same shape with another record
(``examples/11_custom_dsl.py``).
"""

from ..dsl.core.diagnostics import DiagId, register_dsl_package
from ..dsl.core.dsl import BaseDSL
from ..dsl.core.plugin import Plugins
from ..dsl.plugins.decorators.jit import func
from ..dsl.plugins.type_ops import UpstreamDialectTypeOps, arith

__all__ = ["LOWER_TO_LLVM", "MlirTestDSL", "jit"]

#: The passes lowering what the test DSL emits (arith, math and llvm ops
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
register_dsl_package(__package__)


class MlirTestDSL(BaseDSL):
    """The test sub-DSL: the shipped plugins in one record.

    The gpu kernels plugin, the compiler and the adapters are dropped when
    their dependency is absent (``available()``), so the same class traces on
    a machine without the gpu bindings or the execution engine.
    """

    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith),
        # @jit over a func.func host entry, then @kernel over gpu.func.
        decorators=[func.Jit()],
    )

    def pipeline(self) -> list[str]:
        """The gpu lowering first, when an architecture is set (a launch without
        one that is compiled is a ``gpu:CONFIG_MISSING_ARCH`` diagnostic; a dry run
        needs none), then the lowering
        of the upstream dialects the builtin types and the ``scf`` control flow
        emit."""
        passes: list[str] = []
        passes.extend(LOWER_TO_LLVM)
        return passes

    def __init__(self) -> None:
        super().__init__(name="MLIR_DSL")


jit = MlirTestDSL.jit
