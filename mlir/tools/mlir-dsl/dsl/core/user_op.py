# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR op helper functions: user-site locations and trace-time verification.

``dsl_user_op`` wraps every user-facing op builder of the DSL. It attributes
the ops the builder emits to the first frame outside the DSL package, as one
``NameLoc(FileLineColLoc)`` per op, and optionally verifies them as soon as
they are built. Both behaviours are driven by the active environment manager
(``debuginfo``, ``debug``, ``verify_trace``); outside a DSL call neither runs.
"""

import dis
import inspect
import itertools
import linecache
import types
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, Optional

from ... import ir
from .common import (
    DSLBaseError,
    DSLRuntimeError,
    DSLUserCodeRuntimeError,
    DSLUserCodeTypeError,
    get_current_env_manager,
)
from .diagnostics import DiagId, _is_dsl_module, register_dsl_package
from ..util.logger import log

__all__ = ["dsl_user_op", "loc_transform", "register_dsl_package"]


# Scoped stack of loc transforms used by dialect-specific compilation contexts.
# This module stays unaware of any particular dialect or debug-info schema;
# callers decide what a transformed loc means.
_LOC_TRANSFORMS: ContextVar[tuple[Callable[[Any], Any], ...]] = ContextVar(
    "_LOC_TRANSFORMS", default=()
)


def _active_env_flags() -> tuple[bool, bool, bool]:
    """Return ``(debuginfo, include_lib_frame, verify_trace)`` for this call.

    ``debuginfo`` turns on source locations; ``debug`` attributes ops to the
    closest (library) frame, so DSL developers see where inside the DSL an op
    was built, and also turns on trace-time verification; ``verify_trace``
    turns on verification alone. Outside a DSL call there is no manager and
    every flag is off.
    """
    mgr = get_current_env_manager()
    if mgr is None:
        return False, False, False
    debug = bool(mgr.debug)
    return bool(mgr.debuginfo), debug, bool(mgr.verify_trace) or debug


def _verify_new_block_ops(snap_block: Any, snap_n_ops: int, snap_tail: Any) -> Any:
    """Verify the ops ``opFunc`` appended to ``snap_block``.

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


@contextmanager
def loc_transform(transform: Callable[[Any], Any]) -> Iterator[None]:
    """Temporarily rewrite locs produced by ``@dsl_user_op``.

    The decorator still owns the generic work: find the Python user frame and
    build the usual MLIR source loc. This hook lets a frontend or dialect wrap
    that loc while it is building a scoped construct::

        with loc_transform(lambda loc: ir.Location.name("scope", childLoc=loc)):
            dsl_add(a, b)  # receives loc=scope("file.py":line:col)

    Transforms are scoped and stackable; the innermost runs first. ``transform``
    receives each generated or caller-provided location, which may be ``None``,
    and returns its replacement; the stack is restored on every exit.

    :param transform: Callable used to rewrite locations within this scope.
    :type transform: Callable[[Any], Any]
    """
    if not callable(transform):
        raise DSLRuntimeError("loc_transform(transform): transform must be callable")

    token = _LOC_TRANSFORMS.set(_LOC_TRANSFORMS.get() + (transform,))
    try:
        yield
    finally:
        _LOC_TRANSFORMS.reset(token)


def _apply_loc_transforms(loc: Any) -> Any:
    """Apply the active scoped loc transforms to ``loc``."""
    for transform in reversed(_LOC_TRANSFORMS.get()):
        loc = transform(loc)
    return loc


def _is_framework_stack_frame(frame: types.FrameType) -> bool:
    """Return True when ``frame`` runs DSL code rather than the user's.

    Frames are classified by module name against the ``register_dsl_package``
    registry, which is robust across source, build-tree and installed layouts.
    ``contextlib`` frames mediate the explicit ``with`` builders and are
    infrastructure rather than the user's call site.
    """
    module_name = frame.f_globals.get("__name__", "") or ""
    return module_name == "contextlib" or _is_dsl_module(module_name)


def _find_user_frame(
    start_frame: Optional[types.FrameType], *, include_lib_frame: bool = False
) -> Optional[types.FrameType]:
    """Walk up from ``start_frame`` to the first user (non-library) frame.

    Falls back to ``start_frame`` when every frame is framework code, and
    returns it directly when ``include_lib_frame`` is set.
    """
    if include_lib_frame:
        return start_frame

    frame = start_frame
    while frame is not None:
        if not _is_framework_stack_frame(frame):
            return frame
        frame = frame.f_back
    return start_frame


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


# Cache of ``inspect.getsourcefile(frame) or inspect.getfile(frame)`` keyed on
# ``co_filename``: one lookup per distinct source file for the per-op hot path.
_SOURCE_FILE_CACHE: dict[str, str] = {}

# Python >= 3.11: code objects carry a position table and ``inspect.Traceback``
# accepts ``positions=``. Older interpreters fall back to ``getframeinfo``; the
# 3.11+ names are resolved dynamically so the module imports there too.
_HAS_CO_POSITIONS: bool = hasattr(types.CodeType, "co_positions")
_CO_POSITIONS: Any = getattr(types.CodeType, "co_positions", None)
_DIS_POSITIONS: Any = getattr(dis, "Positions", None)


def _fast_frameinfo(frame: types.FrameType) -> inspect.Traceback:
    """Build the same ``inspect.Traceback`` as ``inspect.getframeinfo(frame)``
    without its per-call cost.

    ``getframeinfo`` re-resolves the source file and runs ``inspect.findsource``
    on every call, the single largest trace-time cost of a compile (it runs
    once per built op). The same fields come straight from the frame:
    positions from the code object's position table, the context line from
    ``linecache``, the filename from a per-file cache.
    """
    if not _HAS_CO_POSITIONS:
        return inspect.getframeinfo(frame)
    code = frame.f_code
    lasti = frame.f_lasti
    if lasti >= 0:
        positions = next(itertools.islice(_CO_POSITIONS(code), lasti // 2, None))
    else:
        positions = (None, None, None, None)
    if positions[0] is None:
        positions = (frame.f_lineno,) + tuple(positions[1:])
    lineno = positions[0]

    co_filename = code.co_filename
    filename = _SOURCE_FILE_CACHE.get(co_filename)
    if filename is None:
        filename = inspect.getsourcefile(frame) or inspect.getfile(frame)
        _SOURCE_FILE_CACHE[co_filename] = filename

    line = linecache.getline(filename, lineno, frame.f_globals)
    code_context = [line] if line else None
    return inspect.Traceback(
        filename,
        lineno,
        code.co_name,
        code_context,
        0 if code_context else None,
        positions=_DIS_POSITIONS(*positions),
    )


def _get_caller_frame_info(
    *, include_lib_frame: bool = False
) -> Optional[inspect.Traceback]:
    """Return lightweight frame info for the DSL user call site.

    Skips the wrapper frame, applies framework-frame filtering, and avoids
    ``inspect.getframeinfo()``'s source-context lookup on the hot path.
    """
    cur_frame = inspect.currentframe()
    if cur_frame is None:
        return None
    wrapper_frame = cur_frame.f_back
    start_frame = wrapper_frame.f_back if wrapper_frame is not None else None
    frame = _find_user_frame(start_frame, include_lib_frame=include_lib_frame)
    del cur_frame
    if frame is None:
        return None
    return _fast_frameinfo(frame)


def _get_location_from_frame_info(frameInfo: inspect.Traceback) -> ir.Location:
    """Build an MLIR location from captured Python frame information.

    The file/line/column portion becomes the child ``FileLineColLoc``, while
    the name location carries either the source snippet, when available, or
    the Python function name.
    """
    # In Python < 3.11, getframeinfo returns a NamedTuple without positions.
    if not hasattr(frameInfo, "positions"):
        file_loc = ir.Location.file(frameInfo.filename, frameInfo.lineno, 0)
    else:
        file_loc = ir.Location.file(
            frameInfo.filename,
            frameInfo.positions.lineno,  # type: ignore[attr-defined]
            frameInfo.positions.col_offset or 0,  # type: ignore[attr-defined]
        )
    return ir.Location.name(
        (
            "".join([c.strip() for c in frameInfo.code_context])
            if frameInfo.code_context
            else frameInfo.function
        ),
        childLoc=file_loc,
    )


def dsl_user_op(opFunc: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for user-facing DSL op wrappers.

    1. Attaches source locations when ``debuginfo`` is on, so diagnostics and
       IR dumps point back at the user's Python call site.
    2. Runs trace-time MLIR verification on each newly-built op when
       ``verify_trace`` (or ``debug``) is on, so verifier errors surface at
       the call site rather than at module-verify time.

    Verification snapshots ``InsertionPoint.current.block`` and its last op
    before invoking ``opFunc``, then calls ``verify()`` on the ops left after
    that anchor; ``verify()`` recurses through regions. The anchor, not the op
    count, bounds the walk: a wrapper may insert *before* it, and the ops
    already in the block can include one still under construction (the
    enclosing ``scf.if`` whose body is not filled in yet). A wrapper that
    returns an ``OpView`` directly is always verified.

    :param opFunc: The user-facing API function; must accept ``loc=None``.
    :type opFunc: Callable
    :return: The wrapped user-facing API function.
    :rtype: Callable
    """

    @wraps(opFunc)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log().debug("[dsl_user_op] %s called with %d args", opFunc.__name__, len(args))
        # Pop loc= from kwargs so callers that still pass it don't break. The
        # wrapper replaces it only when source-location tracking is enabled.
        loc: Any = kwargs.pop("loc", None)
        frameInfo = None
        verifier_error = False
        debuginfo, include_lib_frame, verify_trace = _active_env_flags()

        if loc is None and debuginfo and ir.Context.current is not None:
            frameInfo = _get_caller_frame_info(include_lib_frame=include_lib_frame)
            try:
                if frameInfo is not None:
                    loc = _get_location_from_frame_info(frameInfo)
            except RuntimeError:
                # The bindings could not build the location (the context went
                # away under a validation-only call): proceed with loc=None so
                # the wrapped function's own validation can still fire.
                pass
        loc = _apply_loc_transforms(loc)

        # __init__ wrappers either wrap an existing ir.Value or build a trivial
        # constant that always verifies, and run on hot paths: skip the
        # block-diff for them, materializing `block.operations` per call would
        # turn a kernel build into O(N^2).
        is_init = getattr(opFunc, "__name__", "") == "__init__"

        # Snapshot the insertion block so newly-built ops can be verified after
        # opFunc returns; the bindings strip `.result` for value-producing ops,
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
                # ops opFunc appends land after it, whatever it inserts higher
                # up the block does not (see `_verify_new_block_ops`).
                if snap_n_ops:
                    tail = snap_ops[-1]
                    snap_tail = getattr(tail, "operation", tail)
            except ValueError:
                # No active InsertionPoint: skip trace-time verification.
                snap_block = None

        try:
            res_or_list = opFunc(*args, **kwargs, loc=loc)
            verifier_error = True
            # A context manager (if_, for_) has its body filled by the
            # surrounding `with` block after opFunc returns; verifying here
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
            func_name = getattr(opFunc, "__name__", str(opFunc))
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
                    f"Operation verification failed in '{func_name}'",
                    filename=frameInfo.filename if frameInfo is not None else None,
                    line=frameInfo.lineno if frameInfo is not None else None,
                    cause=e,
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
