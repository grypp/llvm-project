# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``scf`` AST preprocessor plugin: Python control flow staged as ``scf``.

:class:`ASTPreprocessor` pairs the ``DSLPreprocessor`` rewrite with the
executors of ``executors.py``: a ``for``/``if``/``while`` on a staged value
becomes ``scf.for``/``scf.if``/``scf.while``, comparisons and ternaries go
through the DSL types, and ``max``/``min``/``any``/``all`` are redirected to
the DSL versions when an argument is staged. Any DSL built on ``BaseDSL``
names it to get Python control flow (``Plugins(ast_preprocessor=
scf.ASTPreprocessor())``); the test DSL does.
"""

import builtins
import functools
from collections.abc import Callable
from typing import Any

from ....core.common import DSLUserCodeError
from ....core.diagnostics import DiagId
from ....core.dsl import BaseDSL
from ....core.plugin import ASTPreprocessorPlugin
from ....types.typing import as_ir_value, max_ as max, min_ as min
from ....core.staging import is_mlir_op
from ..preprocessor import DSLPreprocessor
from .builders import WhileLoopContext, for_, if_, while_, yield_
from .executors import (
    LoopUnroll,
    _compare_executor,
    _if_execute_dynamic,
    _ifexp_execute_dynamic,
    _loop_execute_range_dynamic,
    _while_execute_dynamic,
    all_,
    and_,
    any_,
    in_,
    not_,
    or_,
)

__all__ = [
    "ASTPreprocessor",
    "LoopUnroll",
    "WhileLoopContext",
    "all_",
    "and_",
    "any_",
    "as_ir_value",
    "for_",
    "if_",
    "in_",
    "not_",
    "or_",
    "while_",
    "yield_",
]


def _builtin_redirector(fcn: Callable[..., Any]) -> Callable[..., Any]:
    """Route ``max``/``min``/``any``/``all`` to the DSL's versions when an argument is staged."""

    def builtin_wrapper(fcn: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_mlir_op(args):
            return fcn(*args, **kwargs)
        if kwargs:
            # Redirected built-ins do not support keyword arguments
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                what=f"`{getattr(fcn, '__name__', repr(fcn))}` with keyword arguments",
                detail=" when one of its arguments is a runtime value: pass positional arguments only",
            )
        if fcn is builtins.max:
            return max(*args)
        elif fcn is builtins.min:
            return min(*args)
        elif fcn is builtins.any:
            return any_(*args)
        elif fcn is builtins.all:
            return all_(*args)
        raise DSLUserCodeError(
            DiagId.UNSUP_SYNTAX,
            what=f"The built-in function `{getattr(fcn, '__name__', str(fcn))}`",
            detail=" when one of its arguments is a runtime value: do that work in plain Python before the call, or compute it with runtime operations",
        )

    return functools.partial(builtin_wrapper, fcn)


class ASTPreprocessor(ASTPreprocessorPlugin):
    """The ``scf`` AST preprocessor plugin: ``DSLPreprocessor`` plus executors
    that stage native control flow as ``scf.for``/``scf.if``/``scf.while`` and
    route comparisons, ternaries and ``max``/``min``/``any``/``all`` through
    the DSL.

    :param closure_check: Reject nested functions capturing variables inside
        staged regions (the default; see ``ASTPreprocessorPlugin.closure_check``)
    """

    name = "scf"
    preprocessor_class = DSLPreprocessor

    def __init__(self, *, closure_check: bool = True) -> None:
        self.closure_check = closure_check

    def executors(self, dsl: BaseDSL) -> dict[str, Any]:
        return dict(
            loop_execute_range_dynamic=_loop_execute_range_dynamic,
            if_dynamic=_if_execute_dynamic,
            while_dynamic=_while_execute_dynamic,
            compare_executor=_compare_executor,
            builtin_redirector=_builtin_redirector,
            ifexp_dynamic=_ifexp_execute_dynamic,
        )
