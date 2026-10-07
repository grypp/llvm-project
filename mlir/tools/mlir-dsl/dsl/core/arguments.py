# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The host boundary of a ``@jit`` call: how a Python argument becomes a DSL value.

``BaseDSL`` adapts every argument when it builds the trace and the JIT
executor does it again on every call, both through ``JitArgAdapterRegistry``:
tuples and lists element-wise, frozen records through
``DefaultDataclassAdapter``, ``numpy.ndarray`` buffers into host ``Pointer``
values (the ``pytorch`` and ``dlpack`` plugins add theirs). Adapters are keyed
by type, per scope, lazily by qualified type name, or by protocol;
``register_jit_arg_adapter`` is the user-facing extension point.
``adapt_pointer_address`` is the annotation-driven step for ``Pointer[T]``
parameters, never a registry entry, so an unannotated ``int`` stays a Python
value. Whether an adapted argument is a Python value or an MLIR op is decided
in ``staging.is_argument_meta``. No ``int``/``float``/``bool`` adapter lives
here; a sub-DSL that stages Python scalars registers them in its own scope.
"""

import ctypes
import functools
import typing
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from typing import (
    Any,
    Callable,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
)

import numpy as np

from .common import DSLRuntimeError, DSLUserCodeError
from .diagnostics import DiagId
from ..types.typing import (
    NumericMeta,
    Pointer,
    TypedPointer,
    cast,
    from_numpy_dtype,
)
from ..util.tree_utils import contains_leaf, is_frozen_dataclass, is_leaf
from .staging import _is_dsl_type_annotation

__all__ = [
    "JitArgAdapterRegistry",
    "DefaultDataclassAdapter",
    "adapt_pointer_address",
    "check_pointer_annotation",
]


_ScopeArgs = ParamSpec("_ScopeArgs")
_ScopeResult = TypeVar("_ScopeResult")


class JitArgAdapterRegistry:
    """
    A registry to keep track of the JIT argument adapters.

    An adapter is a callable that converts a Python object into a value the
    DSL can pass into compiled code: a leaf instance (``Numeric``, ``Pointer``,
    a ``@struct`` record, ...) or a container of them that ``tree_flatten``
    walks. The converted value can then be further processed by DSL to generate
    arguments for JIT functions.
    """

    # Common adapters shared by every DSL, keyed by Python type. Scoped
    # adapters are intentionally not mirrored here: choosing one as the global
    # value would make behavior depend on module import order.
    jit_arg_adapter_registry: dict[type, Any] = {}

    # DSL-specific (scoped) adapters, keyed by ``(scope, type)``. A Python type may
    # have a different adapter in each scope (for example, a stream handle maps
    # to a different DSL value depending on the DSL compiling the function).
    scoped_jit_arg_adapter_registries: dict[str, dict[type, Any]] = {}

    _active_scope: ContextVar[str | None] = ContextVar(
        "jit_arg_adapter_scope", default=None
    )

    # The (name, index) of the argument being adapted, so an adapter can name
    # it in a diagnostic; set by the boundary around each adapter call.
    _active_argument: ContextVar[tuple[str, int] | None] = ContextVar(
        "jit_arg_adapter_argument", default=None
    )

    # Adapters keyed by fully-qualified type name ("module.QualName") for
    # types whose defining module is too expensive to import at registration
    # time (e.g. torch). Cached in jit_arg_adapter_registry for each concrete
    # type on first lookup of an instance, which can only exist once the module
    # is loaded.
    lazy_jit_arg_adapter_registry: dict[str, Any] = {}

    # Top-level module names of the lazy registrations, so lookups of
    # unrelated types skip the qualified-name construction entirely.
    _lazy_adapter_module_roots: set[str] = set()

    # Fallback for frozen dataclasses, which have no type to key on; set by
    # ``set_default_dataclass_adapter`` (``DefaultDataclassAdapter`` below).
    default_dataclass_adapter: Callable[[object], Any] | None = None

    # ``(predicate, adapter)`` pairs for arguments recognised by a protocol
    # rather than a type (``__dlpack__``); tried after the type-keyed lookups.
    protocol_jit_arg_adapters: list[tuple[Callable[[object], bool], Any]] = []

    @classmethod
    def register_jit_arg_adapter(
        cls,
        python_type: "type | str | None" = None,
        *,
        scope: str | None = None,
        lazy: bool = False,
    ) -> Callable[[Any], Any]:
        """Register a JIT argument adapter callable.

        Used as a decorator on any callable::

            @JitArgAdapterRegistry.register_jit_arg_adapter(MyType)
            def adapt_my_type(arg):
                ...

            @JitArgAdapterRegistry.register_jit_arg_adapter(MyType)
            class MyTypeAdapter:
                ...

        Common adapters are registered per type. DSL-specific (scoped) adapters can
        pass ``scope=...`` and are registered per ``(scope, type)`` pair.
        Registering the same type twice in the same scope raises an error.

        With ``lazy=True`` the type is named by its fully-qualified
        "module.QualName" string instead, so registration never imports the
        defining module (e.g. torch). An instance of the type can only reach
        a JIT function after the application has imported its module, so the
        adapter is cached for the concrete type on first lookup. Named base
        classes also match their subclasses, allowing registration against a
        stable public type when implementations use private concrete types.

        :param python_type: The Python type, or its ``"module.QualName"`` when lazy
        :param scope: The adapter scope; None registers a common adapter
        :param lazy: Key the adapter by qualified type name
        :return: The decorator, which returns the adapter unchanged
        :raises DSLRuntimeError: On a malformed registration or a duplicate
        """

        if python_type is None:
            raise DSLRuntimeError(
                "a Python type must be provided for registering JIT argument adapter"
            )
        lazy_module_root: str | None = None
        if lazy:
            if not isinstance(python_type, str):
                raise DSLRuntimeError(
                    "a fully-qualified 'module.QualName' string must be provided "
                    "for registering a lazy JIT argument adapter"
                )
            name_parts = python_type.split(".")
            if len(name_parts) < 2 or not all(name_parts):
                raise DSLRuntimeError(
                    "lazy JIT argument adapter type name must be fully-qualified "
                    "as 'module.QualName'"
                )
            if scope is not None:
                raise DSLRuntimeError(
                    "lazy JIT argument adapters do not support scoped registration"
                )
            lazy_module_root = name_parts[0]
        elif isinstance(python_type, str):
            raise DSLRuntimeError(
                "non-lazy JIT argument adapters must be registered with a Python type"
            )

        def decorator(adapter: Any) -> Any:
            if not callable(adapter):
                raise DSLRuntimeError(
                    "a callable must be provided for registering JIT argument adapter"
                )

            registry: Any
            if lazy:
                registry = cls.lazy_jit_arg_adapter_registry
            elif scope is None:
                registry = cls.jit_arg_adapter_registry
            else:
                registry = cls.scoped_jit_arg_adapter_registries.setdefault(scope, {})
            if python_type in registry:
                raise DSLRuntimeError(
                    f"JIT argument adapter for {python_type} is already registered!",
                    context={
                        "Scope": scope,
                        "Registered adapter": registry[python_type],
                        "Adapter to be registered": adapter,
                    },
                )
            registry[python_type] = adapter
            if lazy_module_root is not None:
                cls._lazy_adapter_module_roots.add(lazy_module_root)
            return adapter

        return decorator

    @classmethod
    def _promote_lazy_adapter(cls, python_type: type) -> Any:
        """Move the lazy entry matching ``python_type`` (or a base of it) into
        the type-keyed registry and return it; None when no name matches."""
        for candidate_type in python_type.__mro__:
            type_qualname = f"{candidate_type.__module__}.{candidate_type.__qualname__}"
            adapter = cls.lazy_jit_arg_adapter_registry.get(type_qualname)
            if adapter is not None:
                cls.jit_arg_adapter_registry[python_type] = adapter
                return adapter
        return None

    @classmethod
    @contextmanager
    def using_scope(cls, scope: str | None) -> Iterator[None]:
        """Use ``scope`` for adapter lookup, including nested adaptations."""
        token = cls._active_scope.set(scope)
        try:
            yield
        finally:
            cls._active_scope.reset(token)

    @classmethod
    def call_with_scope(
        cls,
        scope: str | None,
        callback: Callable[_ScopeArgs, _ScopeResult],
        *args: _ScopeArgs.args,
        **kwargs: _ScopeArgs.kwargs,
    ) -> _ScopeResult:
        """Call ``callback`` with an adapter scope and restore the prior scope.

        This avoids the generator-based context-manager overhead on the compiled
        launch path while preserving the scope for nested adapter lookups.
        """
        token = cls._active_scope.set(scope)
        try:
            return callback(*args, **kwargs)
        finally:
            cls._active_scope.reset(token)

    @classmethod
    @contextmanager
    def using_argument(cls, arg_name: str, arg_index: int) -> Iterator[None]:
        """Name the argument being adapted, for the adapters' diagnostics."""
        token = cls._active_argument.set((arg_name, arg_index))
        try:
            yield
        finally:
            cls._active_argument.reset(token)

    @classmethod
    def active_argument(cls) -> tuple[str, int]:
        """The ``(name, index)`` of the argument being adapted, if one is named."""
        return cls._active_argument.get() or ("<argument>", 0)

    @classmethod
    def get_registered_adapter(cls, arg: object) -> Any:
        """The adapter for ``arg``'s type, or None.

        Lookup order: the active scope (``using_scope``), the common registry,
        the lazy registry (promoted on first match), then, with no scope
        active, the single scope that registers the type; two such scopes
        raise an ambiguity error instead of choosing by import order. The
        default dataclass adapter is the last resort for frozen records.

        :param arg: The argument value (its ``type`` is the key)
        :raises DSLRuntimeError: Ambiguous unscoped lookup
        :raises DSLUserCodeError: ``CONTAINER_INVALID_RECORD``
        """
        python_type = type(arg)
        resolved_scope = cls._active_scope.get()
        adapter = None
        if resolved_scope is not None:
            adapter = cls.scoped_jit_arg_adapter_registries.get(resolved_scope, {}).get(
                python_type
            )

        if adapter is None:
            adapter = cls.jit_arg_adapter_registry.get(python_type)

        if (
            adapter is None
            and cls.lazy_jit_arg_adapter_registry
            and not cls._lazy_adapter_module_roots.isdisjoint(
                candidate_type.__module__.partition(".")[0]
                for candidate_type in python_type.__mro__
            )
        ):
            adapter = cls._promote_lazy_adapter(python_type)

        if adapter is None and resolved_scope is None:
            scoped_matches = [
                (registered_scope, registry[python_type])
                for registered_scope, registry in (
                    cls.scoped_jit_arg_adapter_registries.items()
                )
                if python_type in registry
            ]
            if len(scoped_matches) == 1:
                adapter = scoped_matches[0][1]
            elif len(scoped_matches) > 1:
                raise DSLRuntimeError(
                    f"JIT argument adapter for {python_type} is ambiguous; "
                    "perform the lookup inside "
                    "JitArgAdapterRegistry.using_scope(...) instead",
                    context={
                        "Registered scopes": [scope for scope, _ in scoped_matches]
                    },
                )

        if adapter is None:
            for predicate, candidate in cls.protocol_jit_arg_adapters:
                if predicate(arg):
                    adapter = candidate
                    break

        if adapter is None and cls.default_dataclass_adapter is not None:
            adapter = cls._default_dataclass_adapter_for(arg)
        return adapter

    @classmethod
    def register_protocol_adapter(
        cls, predicate: Callable[[object], bool], adapter: Any
    ) -> None:
        """Register ``adapter`` for every argument ``predicate`` accepts.

        Protocol adapters are consulted after the type-keyed registries, in
        registration order; registering the same pair twice is a no-op.
        """
        if (predicate, adapter) not in cls.protocol_jit_arg_adapters:
            cls.protocol_jit_arg_adapters.append((predicate, adapter))

    @classmethod
    def _default_dataclass_adapter_for(cls, arg: object) -> Any:
        """The default dataclass adapter if ``arg`` is a frozen record (not a
        registered leaf), else None; a
        non-frozen dataclass holding a DSL-typed field or a leaf raises
        ``CONTAINER_INVALID_RECORD``, one holding neither is a Python value."""
        if not is_dataclass(arg) or isinstance(arg, type) or is_leaf(arg):
            return None
        if not is_frozen_dataclass(arg):
            hints = _field_hints(type(arg))
            dsl_typed = any(
                _is_dsl_type_annotation(hints.get(f.name, f.type)) for f in fields(arg)
            )
            if dsl_typed or contains_leaf(arg):
                raise DSLUserCodeError(
                    DiagId.CONTAINER_INVALID_RECORD,
                    var=JitArgAdapterRegistry.active_argument()[0],
                    type=type(arg).__name__,
                    detail="it is a dataclass that is not frozen, so an update made on one path would be lost",
                )
            return None
        return cls.default_dataclass_adapter

    @classmethod
    def set_default_dataclass_adapter(cls, adapter: Callable[[object], Any]) -> None:
        """Install the fallback adapter for frozen dataclasses. A dataclass
        that is a registered leaf is adapted by its leaf entry instead."""
        cls.default_dataclass_adapter = adapter


_UNSET = object()


@functools.lru_cache(maxsize=None)
def _field_hints(cls: type) -> dict[str, Any]:
    """The resolved field annotations of a dataclass (a string annotation
    under ``from __future__ import annotations`` becomes the type)."""
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {f.name: f.type for f in fields(cls)}


class DefaultDataclassAdapter:
    """Adapter for frozen dataclass typed JIT arguments.

    ``DefaultDataclassAdapter(arg)`` returns a new ``type(arg)`` record for
    ``tree_flatten``: ``Numeric``-annotated fields cast, ``Pointer[T]``
    fields adapted and checked against the annotation, other fields adapted
    through the registry when an adapter exists. The record is rebuilt field
    by field (no ``__init__``), so ``init=False`` fields and custom
    constructors are fine; instance attributes that are not fields are kept.
    """

    def __new__(cls, arg: object) -> Any:
        if not is_frozen_dataclass(arg):
            raise DSLUserCodeError(
                DiagId.CONTAINER_INVALID_RECORD,
                var=JitArgAdapterRegistry.active_argument()[0],
                type=type(arg).__name__,
                detail="it is a dataclass that is not frozen, so an update made on one path would be lost",
            )
        arg_name, arg_index = JitArgAdapterRegistry.active_argument()
        hints = _field_hints(type(arg))
        new = object.__new__(type(arg))
        for f in fields(arg):  # type: ignore[arg-type]
            arg_field = getattr(arg, f.name, _UNSET)
            if arg_field is _UNSET:
                raise DSLUserCodeError(
                    DiagId.CONTAINER_INVALID_RECORD,
                    var=arg_name,
                    type=type(arg).__name__,
                    detail=f"its field `{f.name}` has no value",
                )
            annotation = hints.get(f.name, f.type)
            field_name = f"{arg_name}.{f.name}"
            if isinstance(annotation, NumericMeta) and not isinstance(
                arg_field, annotation
            ):
                value = cast(arg_field, annotation)  # type: ignore[arg-type]
            elif isinstance(annotation, TypedPointer):
                value = adapt_pointer_address(
                    arg_field, annotation, arg_name=field_name, arg_index=arg_index
                )
                if value is None:
                    check_pointer_annotation(
                        arg_field, annotation, arg_name=field_name, arg_index=arg_index
                    )
                    value = arg_field
            else:
                arg_adapter = JitArgAdapterRegistry.get_registered_adapter(arg_field)
                if arg_adapter is None:
                    value = arg_field
                else:
                    with JitArgAdapterRegistry.using_argument(field_name, arg_index):
                        value = arg_adapter(arg_field)
            object.__setattr__(new, f.name, value)
        for name, value in getattr(arg, "__dict__", {}).items():
            if not hasattr(new, name):
                object.__setattr__(new, name, value)
        return new


JitArgAdapterRegistry.set_default_dataclass_adapter(DefaultDataclassAdapter)


# =============================================================================
# JIT Argument Adapters
# =============================================================================


@JitArgAdapterRegistry.register_jit_arg_adapter(tuple)
@JitArgAdapterRegistry.register_jit_arg_adapter(list)
def _convert_python_sequence(arg: Union[tuple, list]) -> Union[tuple, list]:
    """Adapt each element of a tuple or list in turn; an element without an
    adapter is kept. The container type is preserved."""
    adapted_arg = []
    for elem in arg:
        adapter = JitArgAdapterRegistry.get_registered_adapter(elem)
        adapted_arg.append(elem if adapter is None else adapter(elem))
    return type(arg)(adapted_arg)


def _check_contiguous(arg: Any, contiguous: bool) -> None:
    """Raise ``ARG_BUFFER_INVALID`` unless the buffer is one contiguous block."""
    if not contiguous:
        arg_name, _ = JitArgAdapterRegistry.active_argument()
        raise DSLUserCodeError(
            DiagId.ARG_BUFFER_INVALID,
            arg_name=arg_name,
            arg_type=f"{type(arg).__module__}.{type(arg).__qualname__}",
            detail=f"it is a `{type(arg).__module__}.{type(arg).__qualname__}` that is not contiguous in memory",
        )


@JitArgAdapterRegistry.register_jit_arg_adapter(np.ndarray)
def _convert_numpy_array(arg: np.ndarray) -> Pointer:
    """
    Adapt a C-contiguous numpy array to a host ``Pointer`` over its data: the
    dtype from the array (else ``TYPE_UNKNOWN_DTYPE_NAME``), the array kept
    alive for the call; no shape or stride crosses the boundary.
    """
    _check_contiguous(arg, arg.flags.c_contiguous)
    return Pointer(
        arg.ctypes.data,
        dtype=from_numpy_dtype(arg.dtype),
        kind="host",
        keepalive=arg,
    )


# =============================================================================
# The ``Pointer[T]`` annotation path
# =============================================================================


def _python_pointer_address(arg: object) -> Optional[int]:
    """The address held by an ``int``, ``c_void_p`` or ctypes pointer, else None."""
    if isinstance(arg, bool):
        return None
    if isinstance(arg, int):
        return arg
    if isinstance(arg, ctypes.c_void_p):
        return 0 if arg.value is None else int(arg.value)
    if isinstance(arg, ctypes._Pointer):
        value = ctypes.cast(arg, ctypes.c_void_p).value
        return 0 if value is None else int(value)
    return None


def _pointer_annotation_text(dtype: Any, space: int) -> str:
    """The annotation spelling of a pointer type, for diagnostics."""
    name = getattr(dtype, "__name__", str(dtype))
    return f"`Pointer[{name}]`" if space == 0 else f"`Pointer[{name}, {space}]`"


def check_pointer_annotation(
    value: Any, typed_pointer: TypedPointer, *, arg_name: str, arg_index: int
) -> None:
    """Raise ``ARG_ANNOTATION_MISMATCH`` unless ``value`` is a ``Pointer`` of the
    annotated dtype and address space.

    :param value: The adapted argument
    :param typed_pointer: The ``Pointer[T]``/``Pointer[T, space]`` annotation
    :param arg_name: The parameter name, for the diagnostic
    :param arg_index: The parameter's position, for the diagnostic
    """
    if isinstance(value, Pointer):
        if value.dtype is typed_pointer.dtype and value.space == typed_pointer.space:
            return
        got = f"a {_pointer_annotation_text(value.dtype, value.space)}"
    else:
        got = f"a `{type(value).__name__}`"
    raise DSLUserCodeError(
        DiagId.ARG_ANNOTATION_MISMATCH,
        num=arg_index + 1,
        arg_name=arg_name,
        expected=f"a {_pointer_annotation_text(typed_pointer.dtype, typed_pointer.space)}",
        got=got,
    )


def adapt_pointer_address(
    arg: Any, typed_pointer: TypedPointer, *, arg_name: str = "", arg_index: int = 0
) -> Optional[Pointer]:
    """Adapt the value passed for a ``Pointer[T]``-annotated parameter.

    A bare address (``int``, ``ctypes.c_void_p``, a ctypes pointer; e.g.
    ``tensor.data_ptr()`` or ``0`` for null) becomes a host ``Pointer`` of the
    annotated dtype and space with kind ``"unknown"`` (negative ->
    ``ARG_ANNOTATION_MISMATCH``). A ``Pointer``, or a buffer the registry adapts
    to one, is checked against the annotation (``ARG_ANNOTATION_MISMATCH``).
    None when nothing applies, so the caller continues with its own checks.
    Never a registry entry: an unannotated ``int`` stays a Python value.

    :param arg: The argument value
    :param typed_pointer: The ``Pointer[T]``/``Pointer[T, space]`` annotation
    :param arg_name: The parameter name, for diagnostics
    :param arg_index: The parameter's position, for diagnostics
    :return: The host ``Pointer``, or None
    """
    if not isinstance(arg, Pointer):
        address = _python_pointer_address(arg)
        if address is not None:
            return Pointer(
                address, dtype=typed_pointer.dtype, space=typed_pointer.space
            )
        adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
        if adapter is None:
            return None
        with JitArgAdapterRegistry.using_argument(arg_name, arg_index):
            arg = adapter(arg)
    check_pointer_annotation(arg, typed_pointer, arg_name=arg_name, arg_index=arg_index)
    return arg
