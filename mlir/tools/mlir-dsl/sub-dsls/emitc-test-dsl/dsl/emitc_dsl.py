# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The EmitC test sub-DSL: the same programs, the ops of another dialect.

``EmitCTestDSL`` names a ``type_ops`` plugin that answers the scalar ``add`` and
``mul`` hooks with the EmitC dialect and inherits everything else from
``UpstreamDialectTypeOps`` (types, constants, casts, the memory ops through
``llvm``); its control flow is the ``scf`` preprocessor plugin's and its entry
the ``func.Jit`` plugin's. It names no compiler: the IR is the product, and
``mlir-translate --mlir-to-cpp`` turns a traced module into C.
"""

from __future__ import annotations

from typing import Any

from mlir.dialects import emitc
from mlir.dsl.core.diagnostics import register_dsl_package
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.plugins.ast_preprocessor import scf
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith, llvm, vector

__all__ = ["EmitCScalarOps", "EmitCTestDSL", "jit"]

# Frames of this package are DSL-internal when a diagnostic locates user code.
register_dsl_package("mlir.emitc_dsl")


class EmitCScalarOps(UpstreamDialectTypeOps):
    """``UpstreamDialectTypeOps`` with ``add`` and ``mul`` answered by EmitC.

    The core has promoted the operands already, so the result type is theirs:
    the ``emitc.add``/``emitc.mul`` carry the same ``f32`` or ``i32`` as the
    ``arith`` ops they replace. Every other hook is inherited.
    """

    name = "emitc-scalars"

    def add(
        self,
        lhs: Any,
        rhs: Any,
        *,
        signed: bool | None = None,
        loc: Any = None,
        ip: Any = None,
    ) -> Any:
        return emitc.add(lhs.type, lhs, rhs, loc=loc, ip=ip)

    def mul(
        self,
        lhs: Any,
        rhs: Any,
        *,
        signed: bool | None = None,
        loc: Any = None,
        ip: Any = None,
    ) -> Any:
        return emitc.mul(lhs.type, lhs, rhs, loc=loc, ip=ip)


class EmitCTestDSL(BaseDSL):
    """A trace-only DSL over EmitC arithmetic.

    ``@jit`` builds a ``func.func``, the body's ``+`` and ``*`` become
    ``emitc.add`` and ``emitc.mul``, ``for``/``if``/``while`` become ``scf``
    regions, and there is no compiler plugin: a call returns the trace result
    and ``EMITC_DSL_PRINT_IR=1`` prints the module.
    """

    plugins = Plugins(
        type_ops=EmitCScalarOps(scalars=arith, vectors=vector, memory=llvm),
        ast_preprocessor=scf.ASTPreprocessor(),
        decorators=[func.Jit()],
    )

    def pipeline(self) -> list[str]:
        """No lowering: the traced module is handed on as it is."""
        return []

    def __init__(self) -> None:
        super().__init__(name="EMITC_DSL")


jit = EmitCTestDSL.jit
