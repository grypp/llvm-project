# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The active trace and errors reported by the DSL authoring layer."""

import contextlib
import contextvars
from collections.abc import Generator
from typing import Any


# ===----------------------------------------------------------------------===
# Active Trace
# ===----------------------------------------------------------------------===

_active_dsl: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "active_dsl", default=None
)


def get_current_dsl() -> Any:
    """Return the DSL whose function is being traced, or None."""
    return _active_dsl.get()


def in_trace() -> bool:
    """Whether a decorated call should inline into an existing trace."""
    return get_current_dsl() is not None


@contextlib.contextmanager
def active_dsl(dsl: Any) -> Generator[None, None, None]:
    token = _active_dsl.set(dsl)
    try:
        yield
    finally:
        _active_dsl.reset(token)


# ===----------------------------------------------------------------------===
# DSL Errors
# ===----------------------------------------------------------------------===


class DSLBaseError(Exception):
    """Base exception for DSL configuration and user program errors."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        self.message = message
        self.cause = cause
        super().__init__(message)


class DSLRuntimeError(DSLBaseError):
    """A malformed DSL configuration or unsupported authoring request."""


class DSLUserCodeError(DSLBaseError):
    """An invalid decorated function or call, with a stable diagnostic ID."""

    def __init__(self, diag_id: Any, *, cause: Exception | None = None, **fields: Any):
        self.diag_id = diag_id
        self.fields = fields
        super().__init__(diag_id.value.format(**fields), cause=cause)
