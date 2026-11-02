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

from ....types.typing import as_ir_value, max_ as max, min_ as min
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
