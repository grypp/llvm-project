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
after adapting it; ``JitExecutor`` builds the ``void**`` array and calls the
wrapper; ``JitCompiledFunction`` holds the engine, the module and the entry
point and is what the in-memory ``JitCacheDict`` stores under the module hash.
"""

import ctypes
import dataclasses
import inspect
import threading
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ... import ir
from ..core.common import DSLBaseError, DSLRuntimeError, DSLUserCodeError
from ..core.diagnostics import DiagId
from ..runtime.jit_arg_adapters import (
    JitArgAdapterRegistry,
    adapt_pointer_address,
    check_pointer_annotation,
    is_reserved_python_func_arg,
)
from ..types import typing as t
from ..util.logger import log
from ..util.tree_utils import (
    _check_tree_equal,
    _meta_equal,
    contains_leaf,
    describe_tree_difference,
    is_staged_leaf,
    leaf_entry,
    tree_flatten,
)

__all__ = [
    "ExecutionArgs",
    "JitCacheDict",
    "JitCompiledFunction",
    "JitExecutor",
    "JitModule",
    "lookup_packed_function",
]

PACKED_FUNCTION_PREFIX = "_mlir_"


# =============================================================================
# Packed invocation
# =============================================================================


def lookup_packed_function(engine: Any, function_name: str) -> Any:
    """Resolve the packed ``_mlir_<name>`` wrapper of an ``llvm.func``.

    ``ExecutionEngine.raw_lookup`` is ``mlirExecutionEngineLookupPacked``,
    which prepends the ``_mlir_`` prefix itself, so the plain symbol name is
    passed (upstream's ``lookup`` only stacks ``_mlir_ciface_`` in front).

    :param engine: An ``ExecutionEngine`` built from the lowered module
    :param function_name: The symbol name of the ``llvm.func``
    :return: A ``ctypes`` callable taking one ``void**`` array
    """
    packed_name = PACKED_FUNCTION_PREFIX + function_name
    try:
        address = engine.raw_lookup(function_name)
    except RuntimeError as exc:
        raise DSLRuntimeError(
            f"lookup of `{packed_name}` failed in the JIT engine", cause=exc
        ) from exc
    if not address:
        raise DSLRuntimeError(
            f"function `{function_name}` was not found in the JIT engine",
            context={"symbol": packed_name},
            suggestion=(
                "The packed wrapper exists for every public `llvm.func` of the "
                "lowered module; check that the pipeline left the entry point in "
                "place and that every runtime library it calls into was loaded."
            ),
        )
    return ctypes.CFUNCTYPE(None, ctypes.c_void_p)(address)


# =============================================================================
# Argument binding and marshalling
# =============================================================================


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
                DiagId.ARG_NOT_MARSHALABLE,
                arg_name=arg_name,
                arg_type=type(arg).__name__,
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
                    DiagId.CONTAINER_FIELD_UNSET,
                    var=arg_name,
                    type=type(arg).__name__,
                    field=f.name,
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
        if _check_tree_equal(expected, actual):
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
            numeric_flags.append(isinstance(annotation, t.NumericMeta))
            pointer_flags.append(isinstance(annotation, t.TypedPointer))
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
        )

    def get_rectified_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[Any]:
        """Rectify ``args`` and ``kwargs`` into the runtime argument list."""
        pos_count = len(self._meta.pos_names)
        if len(args) > pos_count:
            raise DSLUserCodeError(
                DiagId.CALL_TOO_MANY_ARGS,
                expected=len(self._meta.pos_names),
                provided=len(args),
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
                raise DSLUserCodeError(DiagId.CALL_UNEXPECTED_KWARG, argument_name=name)
            if idx < pos_len:
                raise DSLUserCodeError(
                    DiagId.CALL_DUPLICATE_ARGUMENT, argument_name=name
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
                DiagId.CALL_MISSING_ARG,
                function_name=self.function_name,
                missing=", ".join(f"`{name}`" for name in missing_args),
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

    def _marshal_all(
        self,
        input_args: Sequence[Any],
        meta: ArgMeta,
        adapter_caches: list[dict],
        adapted_args: list[Any],
        exe_args: list[ctypes.c_void_p],
    ) -> None:
        for index, arg in enumerate(input_args):
            if meta.numeric_flags[index] and not isinstance(arg, ir.Value):
                annotation = meta.annotated_types[index]
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
# In-memory compile cache
# =============================================================================


class JitCacheDict:
    """A dictionary of :class:`JitCompiledFunction` objects keyed by module hash.

    An entry registered with its ``funcBody`` is dropped when that Python
    function is garbage collected, so compiled functions do not leak. With
    ``max_elems`` set the dictionary evicts least-recently-used entries.

    :param max_elems: Capacity; ``None`` is unlimited, ``0`` disables the cache
    """

    def __init__(self, max_elems: int | None = None) -> None:
        self._dict: OrderedDict[
            Any, tuple[Any, weakref.finalize | None]
        ] = OrderedDict()
        self.max_elems = max_elems

    def get(self, key: Any) -> Any | None:
        """The cached value for ``key`` (moved to most-recently-used), or None."""
        if self.max_elems == 0:
            return None
        value = self._dict.get(key)
        if value is None:
            return None
        obj, _ = value
        if self.max_elems is not None:
            self._dict.move_to_end(key, last=True)
        return obj

    def set(self, key: Any, value: Any, funcBody: Any = None) -> None:
        """Store ``value`` under ``key``, tied to ``funcBody``'s lifetime."""
        if self.max_elems == 0:
            return
        if value is funcBody:
            raise DSLRuntimeError(
                "value and funcBody cannot be the same object to avoid circular references"
            )

        # Detach any existing finalizer for this key so that collection of the
        # old value cannot accidentally remove or interfere with the new entry.
        old = self._dict.get(key)
        if old is not None:
            _, old_finalize = old
            if old_finalize is not None:
                old_finalize.detach()

        def _remove_entry(
            k: Any, self_ref: weakref.ref[JitCacheDict] = weakref.ref(self)
        ) -> None:
            # Called from GC/finalizer; be defensive and avoid raising.
            self_obj = self_ref()
            if self_obj is not None:
                self_obj.delete(k)

        self._dict[key] = (
            value,
            (
                None
                if funcBody is None
                else weakref.finalize(funcBody, _remove_entry, key)
            ),
        )
        if self.max_elems is not None:
            self._dict.move_to_end(key, last=True)
            while len(self._dict) > self.max_elems:
                _, (_, finalize) = self._dict.popitem(last=False)
                if finalize is not None:
                    finalize.detach()

    def __contains__(self, key: Any) -> bool:
        return key in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def delete(self, key: Any) -> None:
        """Drop ``key`` if present, detaching its finalizer."""
        entry = self._dict.pop(key, None)
        if entry is not None:
            _, finalize = entry
            if finalize is not None:
                finalize.detach()

    def clear(self) -> None:
        """Drop every entry, detaching the finalizers."""
        for _, finalize in self._dict.values():
            if finalize is not None:
                finalize.detach()
        self._dict.clear()


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


class JitExecutor:
    """An executable function that calls the packed entry of a :class:`JitModule`.

    :param jit_module: The engine, entry and argument binder to call through
    :param jit_time_profiling: Log the marshalling and call times
    :param result_ctype: The ``ctypes`` type of the function's one result
        (``None`` for a ``void`` function); the packed wrapper stores the
        result through the trailing slot of the argument array
    """

    def __init__(
        self,
        jit_module: JitModule,
        jit_time_profiling: bool = False,
        *,
        result_ctype: type | None = None,
    ) -> None:
        # JitExecutor keeps the JitModule alive so that the underlying
        # ExecutionEngine is not discarded until runtime callables are
        # garbage collected.
        self.jit_module = jit_module
        self.profiler = timer(enable=jit_time_profiling) if jit_time_profiling else None
        self._result_ctype = result_ctype
        self._num_extra_args = 1 if result_ctype is not None else 0

        if self.profiler is not None:
            self._get_invoke_packed_args_func = self.profiler(
                self._get_invoke_packed_args
            )
            self.capi_func = self.profiler(self.jit_module.capi_func)
        else:
            self._get_invoke_packed_args_func = self._get_invoke_packed_args
            self.capi_func = self.jit_module.capi_func
        self._tls = threading.local()

    def _get_invoke_packed_args(
        self, exe_args: Sequence[Any], result: Any
    ) -> ctypes.Array:
        """The ``void**`` array of the call: one slot per marshalled argument
        plus the result slot. Each ``exe_arg`` is expected to be a ``c_void_p``
        (no ``ctypes.cast``); the per-thread buffer is reused when it fits."""
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
        if result is not None:
            packed_args[num_base_args] = ctypes.addressof(result)
        return packed_args

    def generate_execution_args(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[ctypes.c_void_p], list[Any]]:
        """Bind and marshal a call; see :meth:`ExecutionArgs.generate_execution_args`."""
        return self.jit_module.execution_args.generate_execution_args(args, kwargs)

    def run_compiled_program(self, exe_args: Sequence[Any]) -> Any:
        """Call the packed entry with marshalled ``exe_args``.

        :return: The result as a Python value for a scalar ``result_ctype``,
            the ``ctypes.Structure`` for an aggregate one, ``None`` for ``void``
        """
        result = self._result_ctype() if self._result_ctype is not None else None
        try:
            packed_args = self._get_invoke_packed_args_func(exe_args, result)
            self.capi_func(packed_args)
        except DSLBaseError:
            raise
        except Exception as exc:
            raise DSLRuntimeError("the compiled function crashed", cause=exc) from exc
        if result is None:
            return None
        return getattr(result, "value", result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Marshal the Python arguments and call the compiled function."""
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
    :param kernel_info: Per-kernel attributes recorded by the trace
    :param jit_time_profiling: Log the marshalling and call times
    :param result_ctype: See :class:`JitExecutor`
    :param has_kernels: Whether the trace built kernels
    """

    # The adapter scope of the binder until the DSL that compiled the function
    # selects its own through ``execution_args.set_adapter_scope``.
    _jit_arg_adapter_scope: ClassVar[
        str | None
    ] = JitArgAdapterRegistry.GPU_DIALECT_SCOPE

    def __init__(
        self,
        ir_module: ir.Module,
        engine: Any,
        capi_func: Any,
        signature: inspect.Signature,
        function_name: str,
        kernel_info: dict[str, Any] | None = None,
        jit_time_profiling: bool = False,
        *,
        result_ctype: type | None = None,
        has_kernels: bool = False,
    ) -> None:
        self.ir_module = ir_module
        self.engine = engine
        self.capi_func = capi_func
        self.function_name = function_name
        self.kernel_info = kernel_info if kernel_info is not None else {}
        self.execution_args = ExecutionArgs(
            signature, self.function_name, adapter_scope=self._jit_arg_adapter_scope
        )
        self.jit_time_profiling = jit_time_profiling
        self.result_ctype = result_ctype
        self.has_kernels = has_kernels

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

    def to(self) -> JitExecutor:
        """An executor bound to this function's engine; the ``JitModule`` is
        shared by every executor of this function.

        :return: A callable :class:`JitExecutor`
        :raises DSLRuntimeError: No engine or entry (a cache record only)
        """
        self._validate_engine()
        with self._executor_lock:
            if self.jit_module is None:
                self.jit_module = JitModule(
                    self.engine, self.capi_func, self.execution_args
                )
            return JitExecutor(
                self.jit_module,
                self.jit_time_profiling,
                result_ctype=self.result_ctype,
            )

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
                # We use a weak reference here so that this instance does not keep
                # this object alive as it holds a reference to self.
                proxy_self = weakref.proxy(self)
                self._default_executor = proxy_self.to()
        return self._default_executor.run_compiled_program(exe_args)
