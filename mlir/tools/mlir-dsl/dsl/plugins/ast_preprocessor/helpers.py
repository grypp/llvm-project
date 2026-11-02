# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Trace-time helpers referenced by the code the AST preprocessor generates.

``DSLPreprocessor`` rewrites native ``for``/``if``/``while`` statements into
region functions decorated with the selectors below (``loop_selector``,
``if_selector``, ``while_selector``) and routes expressions through the
executors (``if_executor``, ``ifexp_executor``, ``compare_executor``,
``assert_executor``, ...). The selectors forward to the :class:`Executor` of
the DSL that is tracing, which the DSL fills through
:meth:`Executor.set_functions`. The module also holds the loop
iterables (``range``) and the trace-time dispatch helpers
(``materialize_for_iter``, ``is_dynamic_range``) that choose between a staged
loop and a native Python loop.
"""

import builtins
import inspect
import itertools
from collections.abc import Callable, Iterator, Sequence
from types import BuiltinFunctionType
from typing import Any, Optional

from ...core.common import DSLUserCodeError, _user_error
from ...core.diagnostics import DiagId
from ...core.staging import (
    Executor,
    _active_executor,
    executor,
    is_mlir_op,
)
from ...util.logger import log

__all__ = [
    "Executor",
    "executor",
    "is_mlir_op",
    "bool_cast",
    "redirect_builtin_function",
]


def bool_cast(value: Any) -> bool:
    """The rewrite of ``bool(value)``: ``PHASE_REQUIRES_CONSTANT`` on a staged value."""
    if is_mlir_op(value):
        raise _user_error(
            DiagId.PHASE_REQUIRES_CONSTANT,
            what="Explicit boolean conversion",
        )
    return bool(value)


# =============================================================================
# Builtin redirection
# =============================================================================
def redirect_builtin_function(fcn: Any) -> Any:
    """Map a builtin the rewritten code calls to the DSL's replacement.

    ``bool`` becomes :func:`bool_cast`; ``exec``/``eval`` are rejected
    (``UNSUP_SYNTAX``); any other builtin function goes through the active
    DSL's ``builtin_redirector``; anything else is returned unchanged.
    """
    if fcn is builtins.bool:
        return bool_cast

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
