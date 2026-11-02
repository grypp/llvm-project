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
(``test/Examples/11_custom_dsl.py``).
"""

from ..dsl.core.diagnostics import DiagId, register_dsl_package
from ..dsl.core.dsl import BaseDSL
from ..dsl.core.plugin import Plugins
from ..dsl.plugins.decorators.jit import func
from ..dsl.plugins.type_ops import UpstreamDialectTypeOps, arith

__all__ = ["MlirTestDSL", "jit"]

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

    def __init__(self) -> None:
        super().__init__(name="MLIR_DSL")


jit = MlirTestDSL.jit
