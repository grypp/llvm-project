# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``scf`` AST preprocessor plugin: Python control flow staged as ``scf``.

:class:`ScfASTPreprocessorPlugin` pairs the ``DSLPreprocessor`` rewrite with
the executors of the ``scf`` dialect plugin: a ``for``/``if``/``while`` on a
staged value becomes ``scf.for``/``scf.if``/``scf.while``, comparisons and
ternaries go through the DSL types, and ``max``/``min``/``any``/``all`` are
redirected to the DSL versions when an argument is staged. Any DSL built on
``BaseDSL`` lists it to get Python control flow; ``MlirDSL`` has it as its
``default_ast_preprocessor``.
"""

import builtins
import functools
from collections.abc import Callable
from typing import Any

from ...core.common import DSLUserCodeError
from ...core.diagnostics import DiagId
from ...core.dsl import BaseDSL
from ...core.plugin import ASTPreprocessorPlugin
from ...types.typing import max_ as max, min_ as min
from ...core.executor import is_dynamic_expression
from ..dialects.scf.executors import (
    _compare_executor,
    _if_execute_dynamic,
    _ifexp_execute_dynamic,
    _loop_execute_range_dynamic,
    _while_execute_dynamic,
    all_,
    any_,
)
from .preprocessor import DSLPreprocessor

__all__ = ["ScfASTPreprocessorPlugin"]


def _builtin_redirector(fcn: Callable[..., Any]) -> Callable[..., Any]:
    """Route ``max``/``min``/``any``/``all`` to the DSL's versions when an argument is staged."""

    def builtin_wrapper(fcn: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_dynamic_expression(args):
            return fcn(*args, **kwargs)
        if kwargs:
            # Redirected built-ins do not support keyword arguments
            raise DSLUserCodeError(
                DiagId.CALL_BUILTIN_KWARGS_UNSUPPORTED,
                fcn=getattr(fcn, "__name__", repr(fcn)),
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
            DiagId.UNSUP_BUILTIN,
            name=getattr(fcn, "__name__", str(fcn)),
            detail=" when one of its arguments is a runtime value",
        )

    return functools.partial(builtin_wrapper, fcn)


class ScfASTPreprocessorPlugin(ASTPreprocessorPlugin):
    """The default AST preprocessor plugin: ``DSLPreprocessor`` plus executors that stage
    native control flow as ``scf.for``/``scf.if``/``scf.while`` and route
    comparisons, ternaries and ``max``/``min``/``any``/``all`` through the DSL.

    :param closure_check: Reject nested functions capturing variables inside
        staged regions (the default; see ``ASTPreprocessorPlugin.closure_check``)
    """

    name = "scf_preprocessor"
    preprocessor_class = DSLPreprocessor

    def __init__(self, *, closure_check: bool = True) -> None:
        self.closure_check = closure_check

    def executors(self, dsl: BaseDSL) -> dict[str, Any]:
        return dict(
            is_dynamic_expression=is_dynamic_expression,
            loop_execute_range_dynamic=_loop_execute_range_dynamic,
            if_dynamic=_if_execute_dynamic,
            while_dynamic=_while_execute_dynamic,
            compare_executor=_compare_executor,
            builtin_redirector=_builtin_redirector,
            ifexp_dynamic=_ifexp_execute_dynamic,
        )
