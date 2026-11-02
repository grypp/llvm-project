# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Staging: Python values versus MLIR ops.

A traced function handles two kinds of values. A Python value is known while
tracing and folds into the program; an MLIR op (the result of one) is only
known when the compiled program runs. :func:`is_mlir_op` tells them apart:
inside a trace the active DSL answers (``BaseDSL.is_mlir_op``, which a sub-DSL
with its own value model overrides), the default judging a leaf by whether it holds an SSA value. At the host
boundary :func:`is_argument_meta` applies the same split to the arguments of
a ``@jit`` call: the annotation decides, a DSL type is an MLIR op and anything
else is a Python value the trace specialises on.

Every ``BaseDSL`` also owns one :class:`Executor`: the DSL's meaning of
Python's own keywords (``for``, ``if``, ``while``, ``x if c else y``, the
comparison operators, the built-in functions) when their operands are MLIR
ops. Python cannot run ``if n > 0:`` on a value that does not exist yet, so the
AST preprocessor rewrites each such statement into a call on the executor, and
the DSL decides what the keyword means: build an ``scf.if``, unroll a loop,
emit an ``arith.cmpi``. This is language virtualization of the keywords; the
rest of Python keeps its ordinary meaning. The helpers the rewritten code
calls (``plugins/ast_preprocessor/helpers.py``) reach the active DSL's
executor through the module-level ``executor`` proxy.
"""

import dataclasses
from inspect import Parameter
from typing import Annotated, get_args, get_origin
from collections.abc import Callable
from typing import Any, Optional

from ... import ir
from .common import DSLRuntimeError, DSLUserCodeError, get_current_dsl
from ..types.typing import NumericMeta
from ..util.tree_utils import (
    contains_leaf,
    is_frozen_dataclass,
    is_leaf,
    is_staged_leaf,
)

__all__ = [
    "is_argument_meta",
    "is_mlir_op",
    "is_reserved_python_func_arg",
]

# =============================================================================
# The boundary: which arguments of a @jit call are Python values
# =============================================================================


def is_reserved_python_func_arg(
    arg_index: int, arg_name: str, func: Optional[Callable[..., Any]]
) -> bool:
    """True for the receiver of a method, never a runtime argument: ``self`` in
    the first position, or ``cls`` in the first position of a ``classmethod``."""

    if arg_index != 0:
        return False

    if arg_name == "self":
        return True

    if func:
        is_classmethod = isinstance(func, classmethod) or (
            hasattr(func, "__func__") and isinstance(func.__func__, classmethod)
        )
        return arg_name == "cls" and is_classmethod
    return False


def _is_type_argument(arg: Any, arg_annotation: Any) -> bool:
    """True if ``arg`` is a class passed where the annotation is absent or
    ``type[X]``: a type is always a Python value."""

    return isinstance(arg, type) and (
        arg_annotation is Parameter.empty or get_origin(arg_annotation) is type
    )


def _is_dsl_type_annotation(annotation: Any) -> bool:
    """True if the annotation declares a DSL type: a ``Numeric`` class, a
    ``Pointer[T]`` or a registered leaf class, possibly inside
    ``Annotated[...]``. ``int``, ``str``, a user class or none are Python
    annotations (their arguments are Python values)."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if isinstance(annotation, (NumericMeta)):
        return True
    return isinstance(annotation, type) and is_leaf(annotation)


def is_argument_meta(
    arg: Any,
    arg_annotation: Any,
    arg_name: str,
    arg_index: int,
    owning_func: Callable[..., Any],
) -> bool:
    """Whether a ``@jit`` argument is a Python value the trace specialises on
    (the alternative being an MLIR op the compiled function takes at run time).

    Decided by the annotation and the value alone: a reserved receiver, a type
    argument and ``None`` are Python values; so is any argument whose
    annotation is not a DSL type and whose (adapted) value neither is nor
    contains a leaf instance. Everything else becomes an MLIR op.

    :param arg: The (adapted) argument value
    :param arg_annotation: The parameter's annotation
    """
    if (
        is_reserved_python_func_arg(arg_index, arg_name, owning_func)
        or _is_type_argument(arg, arg_annotation)
        or arg is None
    ):
        return True
    return not _is_dsl_type_annotation(arg_annotation) and not contains_leaf(arg)


def _is_mlir_op_leaf(value: Any) -> bool:
    """One value, no container walk: a raw ``ir.Value`` or ``ir.BlockArgumentList``,
    or a registered leaf whose payload is an SSA value (a staged ``Int32``,
    never a host one or a host ``Pointer``). ``BaseDSL.is_mlir_op``'s default."""
    return isinstance(value, ir.BlockArgumentList) or is_staged_leaf(value)


def _any_leaf(value: Any, judge: Callable[[Any], bool]) -> bool:
    """Walk tuples, lists and frozen records; ``judge`` decides each leaf."""
    if isinstance(value, (tuple, list)):
        return any(_any_leaf(x, judge) for x in value)
    if is_frozen_dataclass(value) and not isinstance(value, type):
        return any(
            _any_leaf(getattr(value, f.name, None), judge)
            for f in dataclasses.fields(value)
        )
    return bool(judge(value))


def is_mlir_op(value: Any) -> bool:
    """Whether ``value`` is an MLIR op (its result) rather than a Python value.

    A tuple, list or frozen record (a ``@struct``, a frozen dataclass) is an
    MLIR op when any of its leaves is; dicts and other objects are Python
    values. Each leaf is judged by the DSL that is tracing
    (``BaseDSL.is_mlir_op``, which a sub-DSL with its own value model
    overrides); outside a trace a leaf is an MLIR op when it holds an SSA
    value. User code branches on it at trace time where Python values and
    MLIR ops need different handling::

        if is_mlir_op(n):
            ...  # an MLIR op: build IR
        else:
            ...  # a Python value: plain Python
    """
    dsl = get_current_dsl()
    judge = dsl.is_mlir_op if dsl is not None else _is_mlir_op_leaf
    return _any_leaf(value, judge)
