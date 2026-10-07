# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR op helper functions: source locations and trace-time verification.

Source locations come from MLIR itself. ``enter_traceback_locations`` turns
on the bindings' traceback locations (``ir.loc_tracebacks``): every op built
while they are on carries the Python line and column range that built it,
with the frames of the DSL packages, the ``mlir`` bindings and the standard
library skipped, so the location is the user's line. ``BaseDSL`` enters it
around a trace when ``debuginfo`` is set (depth 1) or ``LOC_TRACEBACKS=N``
asks for a call chain; under ``debug`` the DSL's own frames are kept, so DSL
developers see where inside the DSL an op was built. An explicit ``loc=``
always wins.

``dsl_user_op`` wraps every user-facing op builder of the DSL: it passes the
caller's ``loc=`` through, verifies the ops the builder emits as soon as they
are built when ``verify_trace`` (or ``debug``) is set, and turns a builder
called outside any trace into ``CALL_OUTSIDE_JIT``.
"""

import importlib
import sysconfig
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from ... import ir
from .common import (
    DSLBaseError,
    DSLRuntimeError,
    DSLUserCodeRuntimeError,
    DSLUserCodeTypeError,
    get_current_env_manager,
)
from .diagnostics import DiagId, _DSL_PACKAGES, _is_dsl_module
from ..util.logger import log

__all__ = ["dsl_user_op", "enter_traceback_locations"]


def _verify_trace_enabled() -> bool:
    """Whether the active environment asks for trace-time verification
    (``verify_trace``, or ``debug`` which implies it); off outside a DSL call."""
    mgr = get_current_env_manager()
    if mgr is None:
        return False
    return bool(mgr.verify_trace) or bool(mgr.debug)


# =============================================================================
# Source locations: MLIR's traceback locations with the DSL frames skipped
# =============================================================================


def _frame_filter_paths(include_dsl_frames: bool) -> tuple[list[str], list[str]]:
    """The file prefixes whose frames a location must skip, and the DSL
    package prefixes kept when ``include_dsl_frames`` is set.

    Skipped: the standard library (``contextlib`` mediates the ``with``
    builders), every top-level member of the ``mlir`` package (the bindings,
    the dialect wrappers) and the registered DSL packages wherever they are
    installed. Under ``include_dsl_frames`` the DSL packages are kept instead.
    """
    skipped: list[str] = []
    dsl_dirs: list[str] = []
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib"):
        if paths.get(key):
            skipped.append(paths[key])
    for prefix in _DSL_PACKAGES:
        try:
            module = importlib.import_module(prefix)
        except Exception:  # noqa: BLE001 - a registered but uninstalled sub-DSL
            continue
        dsl_dirs.extend(str(Path(p)) for p in getattr(module, "__path__", []))
    mlir_root = importlib.import_module("mlir")
    for root in getattr(mlir_root, "__path__", []):
        for child in Path(root).iterdir():
            if child.name == "__pycache__":
                continue
            if _is_dsl_module(f"mlir.{child.stem}"):
                dsl_dirs.append(str(child))
            else:
                skipped.append(str(child))
    return skipped, dsl_dirs


def enter_traceback_locations(depth: int, *, include_dsl_frames: bool = False) -> Any:
    """Turn on MLIR's traceback locations for the ops built until the returned
    context manager exits; ``None`` when ``depth`` is 0 or the bindings lack
    the feature (a limited-API build).

    :param depth: How many Python frames a location records, innermost first
        (``1`` is the user's line; more adds the callers as a call site chain)
    :param include_dsl_frames: Keep the DSL packages' frames (``debug``), so
        an op is attributed to the DSL line that built it
    """
    if depth <= 0:
        return None
    try:
        globals_ = ir._globals  # type: ignore[attr-defined]
        skipped, dsl_dirs = _frame_filter_paths(include_dsl_frames)
        for path in skipped:
            globals_.register_traceback_file_exclusion(path)
        for path in dsl_dirs:
            if include_dsl_frames:
                globals_.register_traceback_file_inclusion(path)
            else:
                globals_.register_traceback_file_exclusion(path)
        context = ir.loc_tracebacks(max_depth=depth)
    except (ValueError, TypeError, AttributeError):
        return None
    context.__enter__()
    return context


def _verify_new_block_ops(snap_block: Any, snap_n_ops: int, snap_tail: Any) -> Any:
    """Verify the ops the wrapped builder appended to ``snap_block``.

    ``snap_tail`` is the block's last operation before the call, ``None`` for an
    empty block; the ops after it are the ones the wrapper appended. Anything
    inserted further up the block, and the ops that were already there, are
    left to module-verify time. The anchor is found tail to head, but the
    verification runs head to tail, so a wrapper that builds several malformed
    ops still reports the first one.

    :return: The tail operation when it was verified here, so the caller can
        skip verifying that one a second time.
    """
    ops = snap_block.operations
    new_count = len(ops) - snap_n_ops
    if new_count <= 0:
        return None

    # OperationList materialization is O(block size). Most DSL wrappers append
    # one op, so walk the new tail by negative index instead of list-slicing
    # the whole block on every wrapper call.
    window = 0
    verified_tail = None
    for offset in range(1, new_count + 1):
        op = ops[-offset]
        operation = getattr(op, "operation", op)
        if snap_tail is not None and operation == snap_tail:
            break
        window = offset
        if offset == 1:
            verified_tail = operation
    for offset in range(window, 0, -1):
        ops[-offset].verify()
    return verified_tail


def _is_missing_context_error(e: BaseException) -> bool:
    """Return True for the raw errors the MLIR bindings raise when an
    IR-building call runs with no active ``ir.Context``: the ``RuntimeError``
    of the default context lookup, the ``ValueError`` of ``Location.current``
    / ``InsertionPoint.current``, or a nanobind overload whose trailing
    ``context`` parameter cannot be filled.
    """
    if isinstance(e, DSLBaseError):
        return False
    if isinstance(e, RuntimeError):
        return "requires a Context" in str(e)
    if isinstance(e, ValueError):
        return "Value not set" in str(e) or "requires a Context" in str(e)
    if isinstance(e, TypeError):
        return "incompatible function arguments" in str(e)
    return False


def dsl_user_op(op_func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for user-facing DSL op wrappers.

    1. Passes the caller's ``loc=`` through (source locations otherwise come
       from MLIR's traceback locations, see :func:`enter_traceback_locations`).
    2. Runs trace-time MLIR verification on each newly-built op when
       ``verify_trace`` (or ``debug``) is on, so verifier errors surface at
       the call site rather than at module-verify time.

    Verification snapshots ``InsertionPoint.current.block`` and its last op
    before invoking ``op_func``, then calls ``verify()`` on the ops left after
    that anchor; ``verify()`` recurses through regions. The anchor, not the op
    count, bounds the walk: a wrapper may insert *before* it, and the ops
    already in the block can include one still under construction (the
    enclosing ``scf.if`` whose body is not filled in yet). A wrapper that
    returns an ``OpView`` directly is always verified.

    :param op_func: The user-facing API function; must accept ``loc=None``.
    :type op_func: Callable
    :return: The wrapped user-facing API function.
    :rtype: Callable
    """

    @wraps(op_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log().debug("[dsl_user_op] %s called with %d args", op_func.__name__, len(args))
        # The caller's loc= is passed through; None lets MLIR's traceback
        # locations fill it when they are on.
        loc: Any = kwargs.pop("loc", None)
        verifier_error = False
        verify_trace = _verify_trace_enabled()

        # __init__ wrappers either wrap an existing ir.Value or build a trivial
        # constant that always verifies, and run on hot paths: skip the
        # block-diff for them, materializing `block.operations` per call would
        # turn a kernel build into O(N^2).
        is_init = getattr(op_func, "__name__", "") == "__init__"

        # Snapshot the insertion block so newly-built ops can be verified after
        # op_func returns; the bindings strip `.result` for value-producing ops,
        # so the return value alone does not see every op.
        snap_block: Any = None
        snap_n_ops: int = 0
        snap_tail: Any = None
        if verify_trace and not is_init:
            try:
                snap_block = ir.InsertionPoint.current.block
                snap_ops = snap_block.operations
                snap_n_ops = len(snap_ops)
                # Anchor the diff on the tail op rather than on the count: the
                # ops op_func appends land after it, whatever it inserts higher
                # up the block does not (see `_verify_new_block_ops`).
                if snap_n_ops:
                    tail = snap_ops[-1]
                    snap_tail = getattr(tail, "operation", tail)
            except ValueError:
                # No active InsertionPoint: skip trace-time verification.
                snap_block = None

        try:
            res_or_list = op_func(*args, **kwargs, loc=loc)
            verifier_error = True
            # A context manager (if_, for_) has its body filled by the
            # surrounding `with` block after op_func returns; verifying here
            # would see an empty region. Module-verify time covers it.
            is_cm = hasattr(res_or_list, "__enter__") and hasattr(
                res_or_list, "__exit__"
            )
            # Fast path: nothing appended, nothing to materialize.
            tail_diff_ran = (
                snap_block is not None
                and not is_cm
                and len(snap_block.operations) > snap_n_ops
            )
            verified_tail = None
            if tail_diff_ran:
                verified_tail = _verify_new_block_ops(snap_block, snap_n_ops, snap_tail)
            if hasattr(res_or_list, "verify"):
                # A wrapper that returns an OpView directly: cross-block
                # builders and calls with no active insertion point.
                res_or_list.verify()
            elif tail_diff_ran and verified_tail is None:
                # The block grew, yet nothing landed after the anchor: the
                # wrapper's op went in higher up, so reach it through the value
                # it returned. Never verify the anchor itself, it may still be
                # under construction.
                owner = getattr(res_or_list, "owner", None)
                if owner is not None and hasattr(owner, "verify"):
                    operation = getattr(owner, "operation", owner)
                    if operation != snap_tail:
                        owner.verify()

        except DSLBaseError:
            raise
        except Exception as e:
            func_name = getattr(op_func, "__name__", str(op_func))
            if "unexpected keyword argument 'loc'" in str(e):
                raise DSLRuntimeError(
                    f"Function '{func_name}' decorated with @dsl_user_op does not "
                    "accept the required 'loc' parameter.",
                    suggestion=[
                        f"1. Add 'loc=None' as a keyword-only parameter to {func_name}:",
                        f"  def {func_name}(..., *, loc=None):",
                        "",
                        "2. Remove the @dsl_user_op decorator if location tracking is not needed",
                    ],
                    cause=e,
                ) from e
            if verifier_error:
                raise DSLRuntimeError(
                    f"Operation verification failed in '{func_name}'", cause=e
                ) from e

            # A missing-context failure means the op was invoked outside any
            # @jit/@kernel compilation (at Python module level): an author
            # mistake, not a DSL bug. The raised diagnostic stays catchable as
            # the raw type it replaces.
            if ir.Context.current is None and _is_missing_context_error(e):
                err_cls = (
                    DSLUserCodeTypeError
                    if isinstance(e, TypeError)
                    else DSLUserCodeRuntimeError
                )
                raise err_cls(
                    DiagId.CALL_OUTSIDE_JIT, api=func_name, decorator="@jit", cause=e
                ) from e

            raise

        return res_or_list

    return wrapper
