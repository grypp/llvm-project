# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Trace-time helpers referenced by the code the AST preprocessor generates.

The helpers forward to the :class:`Executor` of
the DSL that is tracing, which the DSL fills through
:meth:`Executor.set_functions`.
"""

import builtins
from types import BuiltinFunctionType
from typing import Any

from ...core.common import DSLUserCodeError
from ...core.diagnostics import DiagId
from ...core.staging import (
    Executor,
    executor,
    is_mlir_op,
)

__all__ = [
    "Executor",
    "executor",
    "is_mlir_op",
    "redirect_builtin_function",
]


# =============================================================================
# Builtin redirection
# =============================================================================
def redirect_builtin_function(fcn: Any) -> Any:
    """Map a builtin the rewritten code calls to the DSL's replacement.

    ``exec``/``eval`` are rejected
    (``UNSUP_SYNTAX``); any other builtin function goes through the active
    DSL's ``builtin_redirector``; anything else is returned unchanged.
    """
    if isinstance(fcn, BuiltinFunctionType):
        if fcn in (builtins.exec, builtins.eval):
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                what=f"The built-in function `{fcn.__name__}`",
                detail=": do that work in plain Python before the call and pass the result in",
            )
        redirector = executor._builtin_redirector
        if redirector is not None:
            return redirector(fcn)
    return fcn
