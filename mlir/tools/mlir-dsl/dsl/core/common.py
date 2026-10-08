# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The exception classes of the DSL (:class:`DSLUserCodeError` for the author's
mistakes, :class:`DSLRuntimeError` for internal invariants), the
:class:`DSLWarning` channel and the context variables that name the active DSL
instance and its environment manager.
"""

import contextlib
import contextvars
import warnings
from typing import Any, Dict, Generator, Optional, Union

from .diagnostics import (
    DiagCatalog,
    WarnId,
    find_user_source_location as _find_user_source_location,
    render_user_diagnostic as _render_user_diagnostic,
)

__all__ = [
    "DSLBaseError",
    "DSLRuntimeError",
    "DSLUserCodeError",
    "DSLUserCodeRuntimeError",
    "DSLUserCodeTypeError",
    "DSLWarning",
    "report_warning",
    "get_current_env_manager",
    "active_dsl",
    "get_current_dsl",
    "in_trace",
]


# =============================================================================
# Active DSL / environment manager
# =============================================================================

# The DSL whose decorated function is running on this thread, or None. One
# variable answers every question the core asks about "now": whose types emit
# (``get_current_dsl``), whose settings the helpers read
# (``get_current_env_manager``) and whether a call of another decorated
# function lands inside an open trace (``in_trace``). ``BaseDSL.run`` sets it
# for a call from Python; nothing is set outside one.
_active_dsl: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "active_dsl", default=None
)


def get_current_dsl() -> Any:
    """The DSL currently running a decorated function, or ``None`` outside
    one; outside, every consumer falls back to its plain-Python default
    (``builtins.range``, the literal promotion rules, ...)."""
    return _active_dsl.get()


def get_current_env_manager() -> Any:
    """The settings (``envar``) of the current DSL, or ``None`` outside one."""
    return getattr(_active_dsl.get(), "envar", None)


def in_trace() -> bool:
    """Whether a decorated function is being traced on this thread, so that a
    call of another one is a ``launch`` into that trace, not a ``call`` from
    Python (``DecoratorPlugin``)."""
    return _active_dsl.get() is not None


@contextlib.contextmanager
def active_dsl(dsl: Any) -> Generator[None, None, None]:
    """Make ``dsl`` the current DSL for the block. An error leaving the block
    carries the DSL's settings, so its rendering names the right environment
    prefix (``<PREFIX>_SHOW_STACKTRACE=1``)."""
    token = _active_dsl.set(dsl)
    try:
        yield
    except Exception as e:
        if not hasattr(e, "_dsl_env_manager"):
            try:
                setattr(e, "_dsl_env_manager", getattr(dsl, "envar", None))
            except (AttributeError, TypeError):
                pass
        raise
    finally:
        _active_dsl.reset(token)


# =============================================================================
# DSL Exceptions
# =============================================================================


def _format_cause(cause: Any) -> str:
    """Render an error's underlying cause, or empty string when there is none."""
    return f"Caused exception: {cause}" if cause else ""


class DSLBaseError(Exception):
    """
    Base exception for DSL-related errors.

    :param message: The diagnostic text
    :param line: Source line the error points at
    :param snippet: Source excerpt shown instead of reading ``filename``
    :param filename: Source file the error points at
    :param context: Extra key/value details (or free text) for the rendering
    :param suggestion: One or several fixes to show
    :param cause: The underlying exception, if any
    """

    # Subclasses set this to True to render the "compiler bug, please report"
    # envelope instead of the "here is your mistake" block. See
    # ``render_user_diagnostic``.
    _is_internal: bool = False

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        snippet: Optional[str] = None,
        filename: Optional[str] = None,
        context: Optional[Union[Dict[str, Any], str]] = None,
        suggestion: Union[str, list[str], tuple[str, ...], None] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.message = message
        self.line = line
        self.filename = filename
        self.snippet = snippet
        self.context = context
        self.suggestion = suggestion
        self.cause = cause

        super().__init__(message)

    def _generate_cause(self) -> str:
        """
        Generates a string representation of the cause of the error, if available.
        """
        return _format_cause(self.cause)

    def _format_message(self) -> str:
        """Format via the single shared user-diagnostic renderer.

        Every DSL error renders through ``render_user_diagnostic`` so the
        output looks identical no matter which layer (AST pre-processing,
        tracing, or runtime) raised it -- there is exactly one rendering
        mechanism to change.  Rendering happens lazily in ``__str__`` so that
        constructing an error never reads source files or walks the stack.
        """
        return _render_user_diagnostic(self)

    def __str__(self) -> str:
        return self._format_message()


class DSLRuntimeError(DSLBaseError):
    """An internal / compiler error -- NOT the author's fault.

    Rendered as the "compiler bug, please report" envelope (see
    ``render_user_diagnostic``): a ``DSLRuntimeError`` means the DSL hit a
    "should never happen" / failed-to-build-IR / wrapped-backend situation that
    the kernel author cannot fix.  For mistakes in the author's kernel raise
    ``DSLUserCodeError`` with a ``DiagId`` instead, so the author gets a
    "here is your mistake + how to fix it" message.
    """

    _is_internal = True


class DSLUserCodeError(DSLBaseError):
    """Raised when an error is detected in the author's kernel code.

    Covers mutation/phase violations, scope errors, type mismatches,
    unsupported constructs, configuration mistakes, and similar author-facing
    diagnostics.  ``filename``/``lineno`` may be passed explicitly; when they
    are not, the nearest author frame on the call stack is used so the error
    still points at the user's code (see ``find_user_source_location``).

    The first positional argument is normally a :class:`DiagId` from the error
    catalog (or a member of a namespaced :class:`DiagCatalog` such as the gpu
    plugin's); the catalog message and fixes are filled from the keyword
    ``**fields`` and the stable code is appended automatically::

        raise DSLUserCodeError(
            DiagId.TYPE_UNSTABLE_JOIN,
            filename="/path/to/user.py",
            lineno=42,
            var="accum", old_type="Int32", new_type="Float32",
        )

    A free-form string message is still accepted for one-off diagnostics that
    do not yet have a catalog entry::

        raise DSLUserCodeError(
            "Scope Error: variable `a` escapes its scope",
            filename="/path/to/user.py",
            lineno=42,
            suggestion="Define the variable before the loop.",
        )
    """

    def __init__(
        self,
        diag_or_message: Any,
        filename: Optional[str] = None,
        lineno: Optional[int] = None,
        col_offset: Optional[int] = None,
        end_col_offset: Optional[int] = None,
        cause: Optional[BaseException] = None,
        suggestion: Optional[Union[str, list]] = None,
        context: Optional[Union[Dict[str, Any], str]] = None,
        snippet: Optional[str] = None,
        **fields: Any,
    ) -> None:
        self.diag_id: Optional[DiagCatalog] = None
        self.code: Optional[str] = None
        if isinstance(diag_or_message, DiagCatalog):
            self.diag_id = diag_or_message
            self.code = diag_or_message.code
            message, catalog_fixes = diag_or_message.fill(**fields)
            if suggestion is None:
                suggestion = list(catalog_fixes)
        else:
            if fields:
                raise DSLRuntimeError(
                    "DSLUserCodeError received template fields "
                    f"{sorted(fields)} but the first argument is a plain "
                    "string, not a DiagId."
                )
            message = diag_or_message

        self.col = col_offset
        self.end_col = end_col_offset
        if filename is None and lineno is None:
            filename, lineno, self.col, self.end_col = _find_user_source_location()

        super().__init__(
            message,
            line=lineno,
            filename=filename,
            snippet=snippet,
            cause=cause,
            suggestion=suggestion,
            context=context,
        )


class DSLUserCodeRuntimeError(DSLUserCodeError, RuntimeError):
    """``DSLUserCodeError`` that is also catchable as ``RuntimeError``.

    Raise this (instead of plain ``DSLUserCodeError``) when replacing a raw
    ``RuntimeError`` that user code may already catch -- e.g. the MLIR
    bindings' "requires a Context" error, which trace-time helpers probe
    with ``except RuntimeError:`` to detect that no context is active.
    """


class DSLUserCodeTypeError(DSLUserCodeError, TypeError):
    """``DSLUserCodeError`` that is also catchable as ``TypeError``.

    Raise this when replacing a raw ``TypeError`` that user code may already
    catch -- e.g. nanobind overload-resolution failures caused by a missing
    default MLIR context.
    """


class DSLWarning(UserWarning):
    """A non-fatal author-facing warning, rendered like a ``DSLUserCodeError``.

    The author's code is not wrong, only at risk (e.g. an implicit promotion
    that can silently bite later), so this renders the shared diagnostic block
    with a yellow ``warning`` header instead of ``error``.  It subclasses
    ``UserWarning`` so it flows through the standard ``warnings`` module
    (filterable, deduplicated); prefer raising it via
    ``report_warning(WarnId.X, ...)``.

    The first positional argument is normally a :class:`WarnId` (the warnings
    catalog -- separate from the :class:`DiagId` error catalog); a free-form
    string is also accepted.
    """

    _severity = "warning"
    _is_internal = False

    def __init__(
        self,
        warn_or_message: Any,
        filename: Optional[str] = None,
        lineno: Optional[int] = None,
        snippet: Optional[str] = None,
        suggestion: Optional[Union[str, list]] = None,
        context: Optional[Union[Dict[str, Any], str]] = None,
        cause: Optional[BaseException] = None,
        **fields: Any,
    ) -> None:
        self.warn_id: Optional[DiagCatalog] = None
        self.code: Optional[str] = None
        if isinstance(warn_or_message, DiagCatalog):
            self.warn_id = warn_or_message
            self.code = warn_or_message.code
            message, catalog_fixes = warn_or_message.fill(**fields)
            if suggestion is None:
                suggestion = list(catalog_fixes)
        else:
            if fields:
                raise DSLRuntimeError(
                    "DSLWarning received template fields "
                    f"{sorted(fields)} but the first argument is a plain "
                    "string, not a WarnId."
                )
            message = warn_or_message

        self.col: Optional[int] = None
        self.end_col: Optional[int] = None
        if filename is None and lineno is None:
            filename, lineno, self.col, self.end_col = _find_user_source_location()
        self.message = message
        self.filename = filename
        self.line = lineno
        self.suggestion = suggestion
        self.context = context
        self.cause = cause
        self.snippet = snippet
        super().__init__(message)

    def _generate_cause(self) -> str:
        return _format_cause(self.cause)

    def __str__(self) -> str:
        return _render_user_diagnostic(self)


def report_warning(
    warn_id: WarnId,
    *,
    filename: str | None = None,
    lineno: int | None = None,
    stacklevel: int = 2,
    **fields: Any,
) -> None:
    """Emit *warn_id* as a non-fatal :class:`DSLWarning` (does not raise).

    The warning analogue of ``raise DSLUserCodeError(DiagId.X, ...)``: it
    renders through the one shared diagnostic block (yellow ``warning``
    header) and flows through the standard ``warnings`` module so it is
    filterable and deduplicated.
    """
    warnings.warn(
        DSLWarning(warn_id, filename=filename, lineno=lineno, **fields),
        stacklevel=stacklevel,
    )
