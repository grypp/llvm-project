# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This module provides jit executor related classes.

A compiled host entry is invoked through the packed ``_mlir_<name>`` wrapper
that ``ExecutionEngine::packFunctionArguments`` defines for every public
function of the LLVM module: it takes one ``void**`` whose slot ``i`` points at
argument ``i`` and, when the function returns a value, whose slot ``nargs``
points at the result storage. The ``_mlir_ciface_`` wrapper that the entry's
``llvm.emit_c_interface`` attribute adds is for external callers; the executor
does not use it.

``ExecutionArgs`` binds a Python call to the compiled signature and marshals
each runtime argument into ``c_void_p`` slots through its leaf registry entry
after adapting it (a Python scalar under a concrete ``Numeric`` annotation
takes the slot's ``scalar_fast_packer`` when its value needs no conversion);
``JitExecutor`` builds the ``void**`` array (the user's slots, then the
``trailing_args`` a DSL appends to the entry, then the result slot) and calls
the wrapper; ``JitCompiledFunction`` holds the engine, the module and the entry
point and is what the in-memory cache (``util/cache.py``) stores under the
module hash. A DSL whose entries carry more than the Python signature says
(a status result, a context parameter) subclasses the two: the
``JitCompiledFunction`` factories ``_make_jit_module``/``_make_executor`` and
the ``JitExecutor`` hooks ``result_storage``/``check_result``/``_before_call``.
"""

import ctypes
import dataclasses
import inspect
import os
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .... import ir
from ...core.common import (
    DSLBaseError,
    DSLRuntimeError,
    DSLUserCodeError,
    _user_error,
)
from ...core.diagnostics import DiagId
from ...core.arguments import (
    JitArgAdapterRegistry,
    adapt_pointer_address,
    check_pointer_annotation,
)
from ...core.staging import is_reserved_python_func_arg
from ...types import typing as t
from ...util.logger import log
from ...util.tree_utils import (
    Leaf,
    trees_equal,
    _meta_equal,
    contains_leaf,
    describe_tree_difference,
    is_staged_leaf,
    leaf_entry,
    tree_flatten,
)

__all__ = [
    "ExecutionArgs",
    "JitCompiledFunction",
    "JitExecutor",
    "JitModule",
    "ScalarPacker",
    "lookup_packed_function",
    "scalar_fast_packer",
]

PACKED_FUNCTION_PREFIX = "_mlir_"


# =============================================================================
# Packed invocation
# =============================================================================


# The capture of ``lookup_packed_function`` redirects the process-wide fd 2,
# so two threads resolving entries at once would read each other's text.
_STDERR_CAPTURE_LOCK = threading.Lock()


def _flush_stderr() -> None:
    """Flush Python's ``sys.stderr`` so its buffer and fd 2 agree; a stream
    without a usable ``flush`` (a replaced or closed one) is left alone."""
    try:
        sys.stderr.flush()
    except (AttributeError, OSError, ValueError):
        pass


class _CapturedStderr:
    """Redirect fd 2 into a temporary file for the duration of a ``with`` block.

    The JIT materializes a module lazily, at the first lookup into it, and
    reports a symbol it cannot resolve on the process's stderr (``JIT session
    error: Symbols not found: [ ... ]``) rather than through the lookup's
    result; the capture turns that report into ``text``. Python's own
    ``sys.stderr`` buffer is flushed on both sides, so text written through it
    inside the block is captured and nothing written before leaks in. When
    fd 2 cannot be duplicated (a closed descriptor, an embedded interpreter
    without one) the block runs uncaptured and ``text`` stays empty.
    """

    def __init__(self) -> None:
        self.text = ""
        self._saved_fd: int | None = None
        self._file: Any = None

    def __enter__(self) -> "_CapturedStderr":
        _flush_stderr()
        try:
            self._saved_fd = os.dup(2)
            self._file = tempfile.TemporaryFile(mode="w+b")
            os.dup2(self._file.fileno(), 2)
        except OSError:
            # No capture: the lookup still runs, its report stays on stderr.
            self._release()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._saved_fd is None:
            return
        try:
            _flush_stderr()
            os.dup2(self._saved_fd, 2)
            self._file.seek(0)
            self.text = self._file.read().decode("utf-8", errors="replace")
        finally:
            self._release()

    def _release(self) -> None:
        if self._saved_fd is not None:
            os.close(self._saved_fd)
            self._saved_fd = None
        if self._file is not None:
            self._file.close()
            self._file = None


def lookup_packed_function(
    engine: Any,
    function_name: str,
    *,
    libs_var: str = "",
    capture_stderr: bool = True,
) -> Any:
    """Resolve the packed ``_mlir_<name>`` wrapper of an ``llvm.func``.

    ``ExecutionEngine.raw_lookup`` is ``mlirExecutionEngineLookupPacked``,
    which prepends the ``_mlir_`` prefix itself, so the plain symbol name is
    passed (upstream's ``lookup`` only stacks ``_mlir_ciface_`` in front).

    The lookup is where the JIT materializes the module, so a symbol it cannot
    resolve (a runtime library that was not loaded) surfaces here: as a report
    on the process's stderr and a null address. With ``capture_stderr`` the
    report is captured around the lookup, under a lock since fd 2 is shared.
    On success whatever was captured was not the lookup's and is written back
    to stderr, in order; on failure it is the error's detail. ``libs_var``
    names the setting that lists the runtime libraries (``<PREFIX>_LIBS``):
    when given, a failed lookup is the user's configuration,
    ``CONFIG_INVALID`` on that variable; without it, an internal error that
    quotes the report in its context.

    :param engine: An ``ExecutionEngine`` built from the lowered module
    :param function_name: The symbol name of the ``llvm.func``
    :param libs_var: The runtime-library setting a failed lookup is blamed on
    :param capture_stderr: Capture the JIT's report around the lookup
    :return: A ``ctypes`` callable taking one ``void**`` array
    :raises DSLUserCodeError: ``CONFIG_INVALID`` on ``libs_var``, when given
    :raises DSLRuntimeError: The lookup failed and no ``libs_var`` was given
    """
    packed_name = PACKED_FUNCTION_PREFIX + function_name
    address = 0
    failure: BaseException | None = None
    captured = ""
    if capture_stderr:
        with _STDERR_CAPTURE_LOCK, _CapturedStderr() as capture:
            try:
                address = engine.raw_lookup(function_name)
            except RuntimeError as exc:
                failure = exc
        captured = capture.text
    else:
        try:
            address = engine.raw_lookup(function_name)
        except RuntimeError as exc:
            failure = exc

    if address and failure is None:
        if captured:
            sys.stderr.write(captured)
            _flush_stderr()
        return ctypes.CFUNCTYPE(None, ctypes.c_void_p)(address)

    report = captured.strip()
    if libs_var:
        detail = f"the compiled entry `{function_name}` could not be resolved"
        if report:
            detail += f"; the JIT engine reported: {report}"
        raise _user_error(
            DiagId.CONFIG_INVALID,
            var=libs_var,
            detail=detail,
            context={"symbol": packed_name},
            suggestion=(
                f"Add the shared library that defines the missing symbols to "
                f"`{libs_var}`, or pass it to the call as an extra link library."
            ),
            cause=failure,
        )
    context: dict[str, Any] = {"symbol": packed_name}
    if report:
        context["stderr"] = report
    if failure is not None:
        raise DSLRuntimeError(
            f"lookup of `{packed_name}` failed in the JIT engine",
            context=context,
            cause=failure,
        ) from failure
    raise DSLRuntimeError(
        f"function `{function_name}` was not found in the JIT engine",
        context=context,
        suggestion=(
            "The packed wrapper exists for every public `llvm.func` of the "
            "lowered module; check that the pipeline left the entry point in "
            "place and that every runtime library it calls into was loaded."
        ),
    )


# =============================================================================
# Argument binding and marshalling
# =============================================================================


# The direct packer of one ``Numeric``-annotated slot: the Python type it
# accepts exactly (``int``, ``float`` or ``bool``) and the function that packs
# a value of that type into its slot, or returns None when the value needs the
# constructor path (a conversion, a range warning).
ScalarPacker = tuple[type, Callable[[Any], list[ctypes.c_void_p] | None]]


def scalar_fast_packer(dtype: t.NumericMeta) -> ScalarPacker | None:
    """The direct packer of a concrete ``Numeric`` annotation, or None.

    A Python scalar passed for a ``Numeric``-annotated parameter goes through
    the dtype's constructor (``cast``), which converts, range-checks and
    warns; the packer is the shortcut for the values on which the constructor
    would only box: it writes them straight into the dtype's C form, the
    bytes the leaf marshal of the cast value produces. It accepts exactly one
    Python type and answers None for a value outside its window, which then
    takes the constructor path with its conversions and warnings:

    * an integer dtype: an ``int`` inside the dtype's exact range (a ``bool``
      is not accepted, the constructor folds it);
    * a width-1 integer (``Boolean``): a ``bool``;
    * a float dtype of ``IEEE_FLOAT_PROBE``: a ``float`` that is zero or whose
      magnitude lies between the format's smallest subnormal and its largest
      finite value, what the constructor neither overflows nor underflows;
      NaN and the infinities are outside.

    An abstract dtype (the trace picks the concrete one), a dtype without a
    ``ctype`` and a float format outside the probe table have no packer.
    """
    if dtype.is_abstract or getattr(dtype, "ctype", None) is None:
        return None
    to_ctype = dtype._to_ctype
    pointer = t.make_owning_c_pointer
    if isinstance(dtype, t.IntegerMeta):
        if dtype.width == 1:

            def pack_bool(x: Any) -> list[ctypes.c_void_p] | None:
                return [pointer(to_ctype(x))]

            return (bool, pack_bool)
        low, high = dtype._exact_range

        def pack_int(x: Any) -> list[ctypes.c_void_p] | None:
            if low <= x <= high:
                return [pointer(to_ctype(x))]
            return None

        return (int, pack_int)
    if isinstance(dtype, t.FloatMeta):
        probe = t.IEEE_FLOAT_PROBE.get(dtype.__name__)
        if probe is None:
            return None
        _, max_finite, min_subnormal = probe

        def pack_float(x: Any) -> list[ctypes.c_void_p] | None:
            if x == 0.0 or min_subnormal <= abs(x) <= max_finite:
                return [pointer(to_ctype(x))]
            return None

        return (float, pack_float)
    return None


@dataclass
class ArgMeta:
    """The runtime signature of a compiled function, precomputed once so the
    call path binds arguments by index: positional and keyword-only names in
    signature order, each parameter's annotation with its ``Numeric`` /
    ``Pointer[T]`` flags, the defaults, and the name -> index map."""

    pos_names: list[str]
    kwonly_names: list[str]
    all_names: list[str]
    annotated_types: list[object]
    numeric_flags: list[bool]
    pointer_flags: list[bool]
    name_to_index: dict[str, int]
    pos_defaults: list[object]
    kwonly_defaults: list[object]
    arg_count: int
    # Index -> the Meta value a ``compile()``d function was specialized for.
    meta_values: dict[int, Any] = field(default_factory=dict)
    # Index -> the host shape (treedef) a ``compile()``d function was built for.
    shapes: dict[int, Any] = field(default_factory=dict)
    # Per slot, the ``scalar_fast_packer`` of a concrete ``Numeric``
    # annotation, None for every other slot.
    scalar_packers: list[ScalarPacker | None] = field(default_factory=list)


_UNSET = object()


def _marshal(
    arg: Any, arg_name: str, index: int, function_name: str
) -> list[ctypes.c_void_p]:
    """Marshal one adapted runtime argument into its ``c_void_p`` slots.

    A registered leaf (``Numeric``, host ``Pointer``, a sub-DSL leaf)
    contributes the owning pointers of its registry ``marshal``; tuples, lists
    and frozen records (a ``@struct``, a frozen dataclass) contribute their
    elements in order; a Meta value contributes
    nothing (the trace specialised on it). Staged values cannot cross.
    """
    entry = leaf_entry(type(arg))
    if entry is not None:
        if is_staged_leaf(arg):
            raise DSLUserCodeError(
                DiagId.ARG_UNSUPPORTED_TYPE,
                num=index + 1,
                arg_name=arg_name,
                function_name=function_name,
                arg_type=type(arg).__name__,
            )
        if entry.marshal is None:
            raise DSLUserCodeError(
                DiagId.ARG_UNSUPPORTED_TYPE,
                num=index + 1,
                arg_name=arg_name,
                function_name=function_name,
                arg_type=type(arg).__name__,
                detail=": the DSL knows the type, but it has no host representation",
            )
        with JitArgAdapterRegistry.using_argument(arg_name, index):
            return list(entry.marshal(arg))
    if isinstance(arg, ir.Value):
        raise DSLUserCodeError(
            DiagId.ARG_UNSUPPORTED_TYPE,
            num=index + 1,
            arg_name=arg_name,
            function_name=function_name,
            arg_type=type(arg).__name__,
        )
    if isinstance(arg, (tuple, list)):
        return [
            p for item in arg for p in _marshal(item, arg_name, index, function_name)
        ]
    if isinstance(arg, (set, frozenset, dict)) and contains_leaf(arg):
        raise DSLUserCodeError(
            DiagId.CONTAINER_UNSUPPORTED, var=arg_name, type=type(arg).__name__
        )
    if dataclasses.is_dataclass(arg) and not isinstance(arg, type):
        # A frozen record (a ``@struct``, a frozen dataclass): its fields in order.
        slots: list[ctypes.c_void_p] = []
        for f in dataclasses.fields(arg):
            value = getattr(arg, f.name, _UNSET)
            if value is _UNSET:
                raise DSLUserCodeError(
                    DiagId.CONTAINER_INVALID_RECORD,
                    var=arg_name,
                    type=type(arg).__name__,
                    detail=f"its field `{f.name}` has no value",
                )
            slots.extend(_marshal(value, arg_name, index, function_name))
        return slots
    return []


class ExecutionArgs:
    """Runtime argument binder for compiled JIT functions.

    Binds a call's ``args``/``kwargs`` to the filtered runtime signature (the
    reserved receiver parameters removed), applies the annotation cast for
    ``Numeric`` annotations and the address adaptation for ``Pointer[T]``
    annotations, looks up the registered adapters in ``adapter_scope`` and
    marshals every runtime value into ``c_void_p`` slots.
    """

    def __init__(
        self,
        signature: inspect.Signature,
        function_name: str,
        adapter_scope: str | None = None,
    ) -> None:
        self.function_name = function_name
        self.signature = self.filter_runtime_signature(signature)
        self.original_signature = signature
        self._missing = object()
        self._meta = self._build_meta()
        self._jit_arg_adapter_scope = adapter_scope
        self._tls = threading.local()

    def set_adapter_scope(self, adapter_scope: str | None) -> None:
        """Select adapters for the DSL that compiled this function.

        Reset the per-thread adapter cache so changing the scope can never reuse
        a callable selected for a different dialect.
        """
        if adapter_scope != self._jit_arg_adapter_scope:
            self._jit_arg_adapter_scope = adapter_scope
            self._tls = threading.local()

    def set_meta_values(self, values: dict[str, Any]) -> None:
        """Record the Meta values (by parameter name) this function was
        specialized for; a later call with another value is rejected
        (``CALL_META_VALUE_MISMATCH``) instead of silently running the
        specialization. Names outside the runtime signature are ignored."""
        self._meta.meta_values = {
            self._meta.name_to_index[name]: value
            for name, value in values.items()
            if name in self._meta.name_to_index
        }

    def set_shapes(self, shapes: dict[str, Any]) -> None:
        """Record the host shape (the flattened tree of the adapted argument,
        Meta values and leaf prototypes included) this function was compiled
        for; a call with another shape is ``ARG_ANNOTATION_MISMATCH`` instead
        of a mis-marshalled call. Names outside the runtime signature are ignored."""
        self._meta.shapes = {
            self._meta.name_to_index[name]: treedef
            for name, treedef in shapes.items()
            if name in self._meta.name_to_index
        }

    def _check_shape(self, arg: Any, index: int) -> None:
        expected = self._meta.shapes.get(index)
        if expected is None:
            return
        _, _, actual = tree_flatten(arg, return_ir_values=False)
        if trees_equal(expected, actual):
            return
        name = self._meta.all_names[index]
        raise DSLUserCodeError(
            DiagId.ARG_ANNOTATION_MISMATCH,
            num=index + 1,
            arg_name=name,
            expected="a value shaped like the one this function was compiled for",
            got=f"a different one ({describe_tree_difference(expected, actual, name)})",
        )

    def _build_meta(self) -> ArgMeta:
        """Precompute metadata for the fast-path execution; static per signature."""
        sig = self.signature

        pos_names: list[str] = []
        kwonly_names: list[str] = []
        annotated_types: list[object] = []
        numeric_flags: list[bool] = []
        pointer_flags: list[bool] = []
        name_to_index: dict[str, int] = {}
        pos_defaults: list[object] = []
        kwonly_defaults: list[object] = []
        scalar_packers: list[ScalarPacker | None] = []

        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwonly_names.append(name)
                if param.default is not inspect.Parameter.empty:
                    kwonly_defaults.append(param.default)
                else:
                    kwonly_defaults.append(self._missing)
            else:
                pos_names.append(name)
                if param.default is not inspect.Parameter.empty:
                    pos_defaults.append(param.default)
                else:
                    pos_defaults.append(self._missing)
            annotation = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            )
            annotated_types.append(annotation)
            numeric = isinstance(annotation, t.NumericMeta)
            numeric_flags.append(numeric)
            pointer_flags.append(isinstance(annotation, t.TypedPointer))
            scalar_packers.append(scalar_fast_packer(annotation) if numeric else None)
            name_to_index[name] = len(pos_names) + len(kwonly_names) - 1

        return ArgMeta(
            pos_names=pos_names,
            kwonly_names=kwonly_names,
            all_names=pos_names + kwonly_names,
            annotated_types=annotated_types,
            numeric_flags=numeric_flags,
            pointer_flags=pointer_flags,
            name_to_index=name_to_index,
            pos_defaults=pos_defaults,
            kwonly_defaults=kwonly_defaults,
            arg_count=len(pos_names) + len(kwonly_names),
            scalar_packers=scalar_packers,
        )

    def get_rectified_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[Any]:
        """Rectify ``args`` and ``kwargs`` into the runtime argument list."""
        pos_count = len(self._meta.pos_names)
        if len(args) > pos_count:
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=self.function_name,
                detail=f"{len(args)} positional argument(s) passed, at most {len(self._meta.pos_names)} accepted",
            )

        # Start with every slot marked missing, we overwrite as values/defaults bind
        rectified = [self._missing] * self._meta.arg_count
        pos_len = len(args)

        # Fill positional slots with the values from the caller
        rectified[:pos_len] = args

        # Fill positional slots the caller skipped with the defaults
        for i in range(pos_len, pos_count):
            default = self._meta.pos_defaults[i]
            if default is not self._missing:
                rectified[i] = default

        # Fill keyword-only slots with the defaults before user kwargs
        for j, default in enumerate(self._meta.kwonly_defaults):
            idx = pos_count + j
            if default is not self._missing:
                rectified[idx] = default

        # Fill keyword slots with the values from the caller
        for name, value in kwargs.items():
            idx = self._meta.name_to_index.get(name)
            if idx is None:
                raise DSLUserCodeError(
                    DiagId.CALL_ARGUMENTS,
                    function_name=self.function_name,
                    detail=f"no parameter is named `{name}`",
                )
            if idx < pos_len:
                raise DSLUserCodeError(
                    DiagId.CALL_ARGUMENTS,
                    function_name=self.function_name,
                    detail=f"`{name}` is passed twice, by position and by keyword",
                )
            rectified[idx] = value

        # Identity, not ``in``: a numpy array argument would turn ``==`` into
        # an elementwise comparison.
        if any(value is self._missing for value in rectified):
            missing_args = [
                name
                for i, name in enumerate(self._meta.all_names)
                if rectified[i] is self._missing
            ]
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=self.function_name,
                detail="no value for "
                + ", ".join(f"`{name}`" for name in missing_args),
            )

        return rectified

    def generate_execution_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[ctypes.c_void_p], list[Any]]:
        """Bind and marshal a call without an MLIR context.

        :return: The ``c_void_p`` slots in signature order and the adapted
            arguments, which must stay alive for the duration of the call
        """
        return self._generate_execution_args(args, kwargs)

    def _adapt_and_marshal(
        self,
        arg: Any,
        index: int,
        cache: dict[type, Any],
        adapted_args: list[Any],
    ) -> list[ctypes.c_void_p]:
        """Resolve, apply and marshal an adapter in the active scope."""
        meta = self._meta
        arg_name = meta.all_names[index]
        adapted: Any = None
        if meta.pointer_flags[index]:
            adapted = adapt_pointer_address(
                arg, meta.annotated_types[index], arg_name=arg_name, arg_index=index
            )
        if adapted is None:
            arg_type = type(arg)
            adapter = cache.get(arg_type)
            if adapter is None:
                adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
                if adapter is not None:
                    cache[arg_type] = adapter
            if adapter is not None:
                with JitArgAdapterRegistry.using_argument(arg_name, index):
                    adapted = adapter(arg)
            if adapted is None and meta.pointer_flags[index]:
                # Neither an address nor an adaptable buffer: a Meta value
                # would contribute no slot and the packed call would read past
                # the argument array, so reject it here.
                check_pointer_annotation(
                    arg,
                    meta.annotated_types[index],  # type: ignore[arg-type]
                    arg_name=arg_name,
                    arg_index=index,
                )
        if adapted is not None:
            arg = adapted
            adapted_args.append(arg)
        self._check_shape(arg, index)
        return _marshal(arg, arg_name, index, self.function_name)

    def _generate_execution_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[ctypes.c_void_p], list[Any]]:
        """Bind, cast (``Numeric`` annotations), adapt and marshal a call.

        A call with exactly the positional arguments skips the rectification;
        the adapter resolved for each argument's type is cached per thread.
        """
        meta = self._meta
        n = meta.arg_count

        tls = self._tls
        adapter_caches = getattr(tls, "adapter_caches", None)
        if adapter_caches is None:
            adapter_caches = [dict() for _ in range(n)]
            tls.adapter_caches = adapter_caches

        input_args: Sequence[Any]
        if not kwargs and len(args) == n:
            input_args = args
        else:
            input_args = self.get_rectified_args(args, kwargs)

        for index, expected in meta.meta_values.items():
            if not _meta_equal(input_args[index], expected):
                raise DSLUserCodeError(
                    DiagId.CALL_META_VALUE_MISMATCH,
                    num=index + 1,
                    arg_name=meta.all_names[index],
                    function_name=self.function_name,
                    expected=repr(expected),
                    got=repr(input_args[index]),
                )

        adapted_args: list[Any] = []
        exe_args: list[ctypes.c_void_p] = []
        try:
            self._marshal_all(input_args, meta, adapter_caches, adapted_args, exe_args)
        except RecursionError:
            raise DSLUserCodeError(
                DiagId.CONTAINER_TOO_DEEP, var=self.function_name, type="arguments"
            ) from None
        return exe_args, adapted_args

    def _pinned_dtype(self, index: int) -> Any:
        """The dtype the function was compiled for at parameter ``index``,
        whose annotation is abstract: the prototype of the recorded shape
        (``set_shapes``) when that is one ``Numeric`` leaf, else None."""
        shape = self._meta.shapes.get(index)
        if isinstance(shape, Leaf) and isinstance(shape.prototype, t.NumericMeta):
            return shape.prototype
        return None

    def _marshal_all(
        self,
        input_args: Sequence[Any],
        meta: ArgMeta,
        adapter_caches: list[dict],
        adapted_args: list[Any],
        exe_args: list[ctypes.c_void_p],
    ) -> None:
        packers = meta.scalar_packers
        for index, arg in enumerate(input_args):
            if meta.numeric_flags[index] and not isinstance(arg, ir.Value):
                packer = packers[index] if index < len(packers) else None
                if packer is not None and type(arg) is packer[0]:
                    # A Python scalar the dtype's constructor would only box:
                    # straight into the slot, same bytes as the cast path.
                    slots = packer[1](arg)
                    if slots is not None:
                        exe_args.extend(slots)
                        continue
                annotation = meta.annotated_types[index]
                abstract = annotation.is_abstract  # type: ignore[attr-defined]
                if abstract and isinstance(arg, (bool, int, float)):
                    # The trace chose the dtype under the abstract annotation
                    # and the function is compiled for it: a Python number
                    # takes it here, no plugin is asked.
                    annotation = self._pinned_dtype(index) or annotation
                try:
                    arg = t.cast(arg, annotation)  # type: ignore[arg-type]
                except DSLBaseError as exc:
                    raise DSLUserCodeError(
                        DiagId.ARG_ANNOTATION_MISMATCH,
                        num=index + 1,
                        arg_name=meta.all_names[index],
                        expected=f"a `{annotation.__name__}`",  # type: ignore[attr-defined]
                        got=f"a `{type(arg).__name__}`",
                        cause=exc,
                    ) from exc
                if abstract:
                    # A DSL value of another dtype of the family would be
                    # mis-marshalled: it has to match the compiled shape.
                    self._check_shape(arg, index)
                exe_args.extend(
                    _marshal(arg, meta.all_names[index], index, self.function_name)
                )
            else:
                exe_args.extend(
                    JitArgAdapterRegistry.call_with_scope(
                        self._jit_arg_adapter_scope,
                        self._adapt_and_marshal,
                        arg,
                        index,
                        adapter_caches[index],
                        adapted_args,
                    )
                )

    def filter_runtime_signature(self, sig: inspect.Signature) -> inspect.Signature:
        """Drop the reserved ``self``/``cls`` parameters; ``*args``/``**kwargs`` stay."""
        filtered_params = []
        for i, (name, param) in enumerate(sig.parameters.items()):
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                filtered_params.append(param)
                continue
            if is_reserved_python_func_arg(i, name, None):
                continue
            filtered_params.append(param)

        return sig.replace(parameters=filtered_params)


# =============================================================================
# Compiled function handles
# =============================================================================


class JitModule:
    """Holds the execution engine and the packed entry of a compiled module.

    :param engine: The ``ExecutionEngine``
    :param capi_func: The packed entry from :func:`lookup_packed_function`
    :param execution_args: The argument binder of the entry's signature
    """

    def __init__(
        self,
        engine: Any,
        capi_func: Any,
        execution_args: ExecutionArgs,
    ) -> None:
        self.engine = engine
        self.capi_func = capi_func
        self.execution_args = execution_args


def _slot_address(pointer: Any) -> int:
    """The address a ``trailing_args`` entry names: an ``int``, a ``c_void_p``
    or any other ctypes pointer."""
    if isinstance(pointer, int):
        return pointer
    if isinstance(pointer, ctypes.c_void_p):
        return pointer.value or 0
    return ctypes.cast(pointer, ctypes.c_void_p).value or 0


class JitExecutor:
    """An executable function that calls the packed entry of a :class:`JitModule`.

    The ``void**`` array of a call holds the marshalled user arguments, then
    ``trailing_args``, then the result slot: the packed wrapper reads parameter
    ``i`` through slot ``i`` and stores the result through the last slot.

    :param jit_module: The engine, entry and argument binder to call through
    :param jit_time_profiling: Log the marshalling and call times
    :param result_ctype: The ``ctypes`` type of the function's one result
        (``None`` for a ``void`` function); the packed wrapper stores the
        result through the trailing slot of the argument array
    :param trailing_args: Extra trailing parameters of the entry that the
        Python signature does not name (a context, a status cell): one
        ``c_void_p`` (or address) per parameter, pointing at that parameter's
        storage, which the caller keeps alive; an owning pointer from
        ``make_owning_c_pointer`` keeps its own
    """

    def __init__(
        self,
        jit_module: JitModule,
        jit_time_profiling: bool = False,
        *,
        result_ctype: type | None = None,
        trailing_args: Sequence[Any] = (),
    ) -> None:
        # JitExecutor keeps the JitModule alive so that the underlying
        # ExecutionEngine is not discarded until runtime callables are
        # garbage collected.
        self.jit_module = jit_module
        self._result_ctype = result_ctype
        # The pointer objects are kept so what they own stays alive.
        self.trailing_args = tuple(trailing_args)
        self._trailing_addresses = [_slot_address(p) for p in self.trailing_args]
        self._num_extra_args = len(self._trailing_addresses) + (
            1 if result_ctype is not None else 0
        )

        self._get_invoke_packed_args_func = self._get_invoke_packed_args
        self.capi_func = self.jit_module.capi_func
        self._tls = threading.local()

    def _get_invoke_packed_args(
        self, exe_args: Sequence[Any], result: Any
    ) -> ctypes.Array:
        """The ``void**`` array of the call: one slot per marshalled argument,
        the trailing parameters, then the result slot. Each ``exe_arg`` is
        expected to be a ``c_void_p`` (no ``ctypes.cast``); the per-thread
        buffer is reused when it fits."""
        num_base_args = len(exe_args)
        total_args = num_base_args + self._num_extra_args

        # Re-use the packed args buffer if possible
        tls = self._tls
        packed_args = getattr(tls, "packed_args", None)
        capacity = getattr(tls, "capacity", 0)
        if packed_args is None or capacity < total_args:
            packed_args = (ctypes.c_void_p * total_args)()
            tls.packed_args = packed_args
            tls.capacity = total_args

        for i in range(num_base_args):
            arg = exe_args[i]
            packed_args[i] = (
                arg if type(arg) is ctypes.c_void_p else ctypes.c_void_p(arg).value
            )
        # Written on every call: a container argument may change the slot
        # count between calls, moving the trailing slots in the reused buffer.
        next_slot = num_base_args
        for address in self._trailing_addresses:
            packed_args[next_slot] = address
            next_slot += 1
        if result is not None:
            packed_args[next_slot] = ctypes.addressof(result)
        return packed_args

    def generate_execution_args(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[ctypes.c_void_p], list[Any]]:
        """Bind and marshal a call; see :meth:`ExecutionArgs.generate_execution_args`."""
        return self.jit_module.execution_args.generate_execution_args(args, kwargs)

    def result_storage(self) -> Any:
        """The ``result_ctype`` object a call stores its result through: a
        fresh one per call. A subclass may hand out one per thread, whose
        address then stays the same across that thread's calls."""
        return self._result_ctype()  # type: ignore[misc]

    def check_result(self, raw: Any) -> Any:
        """The call's result from what the entry stored: the Python value of
        a scalar ``result_ctype``, the ``ctypes.Structure`` of an aggregate
        one. A subclass reads a status word here and raises its own error."""
        return getattr(raw, "value", raw)

    def _before_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """First thing in :meth:`__call__`, before the arguments are bound:
        a subclass validates or records the call here. No-op by default."""

    def run_compiled_program(self, exe_args: Sequence[Any]) -> Any:
        """Call the packed entry with marshalled ``exe_args``.

        :return: :meth:`check_result` of the stored result (the Python value
            for a scalar ``result_ctype``, the ``ctypes.Structure`` for an
            aggregate one), ``None`` for ``void``
        """
        result = self.result_storage() if self._result_ctype is not None else None
        try:
            packed_args = self._get_invoke_packed_args_func(exe_args, result)
            self.capi_func(packed_args)
        except DSLBaseError:
            raise
        except Exception as exc:
            raise DSLRuntimeError("the compiled function crashed", cause=exc) from exc
        if result is None:
            return None
        return self.check_result(result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Marshal the Python arguments and call the compiled function."""
        self._before_call(args, kwargs)
        # ``adapted_args`` owns the buffers the slots point at; it must live
        # until the call returns.
        exe_args, adapted_args = self.generate_execution_args(*args, **kwargs)
        return self.run_compiled_program(exe_args)


class JitCompiledFunction:
    """Holds a compiled function.

    :param ir_module: The lowered module the engine was built from
    :param engine: The ``ExecutionEngine``
    :param capi_func: The packed entry from :func:`lookup_packed_function`
    :param signature: The Python signature the call arguments bind to
    :param function_name: The entry's symbol name
    :param jit_time_profiling: Log the marshalling and call times
    :param result_ctype: See :class:`JitExecutor`

    ``kernel_info`` and ``has_kernels`` describe the kernels of the trace; a
    kernels plugin fills them through ``finish_compiled_function`` (a
    ``DecoratorPlugin`` hook).

    A DSL whose entries need more than the default call (a result slot to
    check, trailing parameters, a resource to bind) subclasses this and
    overrides the factories: :meth:`_make_jit_module` builds the shared
    :class:`JitModule` once, :meth:`_make_executor` builds every
    :class:`JitExecutor` (its own subclass, with the options
    :meth:`executor` was given) and :meth:`_before_call` runs first on each
    direct call. The compiler plugin's ``load`` picks the class
    (``func_type``).
    """

    # The adapter scope of the binder until the DSL that compiled the function
    # selects its own through ``execution_args.set_adapter_scope``.
    _jit_arg_adapter_scope: ClassVar[str | None] = None

    def __init__(
        self,
        ir_module: ir.Module,
        engine: Any,
        capi_func: Any,
        signature: inspect.Signature,
        function_name: str,
        jit_time_profiling: bool = False,
        *,
        result_ctype: type | None = None,
    ) -> None:
        self.ir_module = ir_module
        self.engine = engine
        self.capi_func = capi_func
        self.function_name = function_name
        self.kernel_info: dict[str, Any] = {}
        self.execution_args = ExecutionArgs(
            signature, self.function_name, adapter_scope=self._jit_arg_adapter_scope
        )
        self.jit_time_profiling = jit_time_profiling
        self.result_ctype = result_ctype
        self.has_kernels = False

        # This runtime state is stored here so that we can preserve the module
        # in the compiler cache. Callers can extend the lifetime of the module
        # by creating and retaining the executor.
        self.jit_module: JitModule | None = None
        self._executor_lock = threading.RLock()
        self._default_executor: JitExecutor | None = None

    def _validate_engine(self) -> None:
        """Raise unless the function has an engine and an entry to call."""
        if self.engine is None or self.capi_func is None:
            raise DSLRuntimeError(
                "The compiled function does not have a valid execution engine.",
                context={"function": self.function_name},
            )

    def executor(self, **options: Any) -> JitExecutor:
        """An executor bound to this function's engine; the ``JitModule`` is
        built once (:meth:`_make_jit_module`) and shared by every executor of
        this function, each built by :meth:`_make_executor` with ``options``.

        :param options: Passed on to :meth:`_make_executor`: the
            :class:`JitExecutor` keywords (``trailing_args``) by default, a
            subclass's own (a device, a stream) when it overrides the factory
        :return: A callable :class:`JitExecutor`
        :raises DSLRuntimeError: No engine or entry (a cache record only)
        """
        self._validate_engine()
        with self._executor_lock:
            if self.jit_module is None:
                self.jit_module = self._make_jit_module()
            return self._make_executor(self.jit_module, **options)

    def _make_jit_module(self) -> JitModule:
        """The :class:`JitModule` every executor of this function shares;
        built once, under the executor lock. A subclass returns its own
        (one that also holds what its executors need loaded)."""
        return JitModule(self.engine, self.capi_func, self.execution_args)

    def _make_executor(self, jit_module: JitModule, **options: Any) -> JitExecutor:
        """A :class:`JitExecutor` over ``jit_module`` with the function's
        profiling flag and result type; ``options`` are the executor's
        keywords. A subclass builds its own executor class and consumes its
        own options here."""
        return JitExecutor(
            jit_module,
            self.jit_time_profiling,
            result_ctype=self.result_ctype,
            **options,
        )

    def _before_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """First thing in :meth:`__call__`, before the arguments are bound:
        a subclass validates or records the call here. No-op by default."""

    def generate_execution_args(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[ctypes.c_void_p], list[Any]]:
        """Bind and marshal a call; see :meth:`ExecutionArgs.generate_execution_args`."""
        return self.execution_args.generate_execution_args(args, kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Marshal the Python arguments and call the compiled function.

        :return: The raw result: a Python scalar for a numeric result, the
            ``ctypes.Structure`` for an aggregate one, None for ``void``
        """
        self._before_call(args, kwargs)
        exe_args, adapted_args = self.execution_args.generate_execution_args(
            args, kwargs
        )
        executor = self._default_executor
        if executor is not None:  # Only lock on first call
            return executor.run_compiled_program(exe_args)
        return self.run_compiled_program(exe_args)

    def run_compiled_program(self, exe_args: Sequence[Any]) -> Any:
        """Call the compiled function with marshalled ``exe_args`` through the
        default executor, created on first use."""
        with self._executor_lock:
            if self._default_executor is None:
                log().debug("Creating default executor for [%s]", self.function_name)
                self._default_executor = self.executor()
        return self._default_executor.run_compiled_program(exe_args)
