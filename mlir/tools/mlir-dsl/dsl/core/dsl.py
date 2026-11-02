# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace Python functions containing MLIR builders into verified modules.

Python arguments specialize IR construction: the body receives them unchanged
and returns None. A decorator plugin supplies the function op and terminator;
builders in the body use the normal MLIR bindings and active insertion point.
"""

import inspect
import os
import re
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar

from ... import ir
from .common import DSLRuntimeError, DSLUserCodeError, active_dsl, in_trace
from .diagnostics import DiagId
from .plugin import DecoratorPlugin, Plugins


# ===----------------------------------------------------------------------===
# DSL Class Construction
# ===----------------------------------------------------------------------===


class DSLSingletonMeta(type):
    """Keep one instance per concrete DSL class.

    Requesting an abstract base returns its first existing concrete instance.
    """

    _instances: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls.__dict__.get("_abstract_dsl_base"):
                for instance in cls._instances.values():
                    if isinstance(instance, cls):
                        return instance
                raise DSLRuntimeError(
                    f"`{cls.__name__}` is an abstract DSL base: initialize a "
                    "concrete subclass of it first"
                )
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]

    def clear_instances(cls) -> None:
        """Forget the instance of ``cls`` so the next ``cls()`` builds a new one."""
        if cls in cls._instances:
            del cls._instances[cls]

    def __getattr__(cls, name: str) -> Any:
        """Explain missing decorators while preserving AttributeError semantics."""
        if name.startswith("_"):
            raise AttributeError(name)
        record = cls.__dict__.get("plugins")
        for klass in cls.__mro__[1:]:
            if record is not None:
                break
            record = klass.__dict__.get("plugins")
        names = sorted(p.decorator_name for p in getattr(record, "decorators", ()))
        have = (
            f"its decorators are {names}"
            if names
            else "its record names no decorator plugin"
        )
        raise AttributeError(
            f"`{cls.__name__}` has no attribute `{name}`; {have}. A decorator "
            "comes from a `DecoratorPlugin` in the record, e.g. "
            "`plugins = Plugins(decorators=[func.Jit()])` for `@jit`."
        )


@dataclass(frozen=True)
class DSLLocation:
    """Python source location of DSL code, used to annotate the generated IR."""

    filename: str
    lineno: int
    col_offset: int
    function_name: str


class _PluginDecorator:
    """Bind a decorator to the DSL class it is accessed through.

    Cache one decorator per class, so repeated attribute access preserves
    identity. Closures retain the class and decorator kind, not bound methods;
    the installed plugin is resolved when the decorated function is called.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._bound: dict[type, Callable[..., Any]] = {}

    def __get__(self, obj: Any, owner: type | None = None) -> Callable[..., Any]:
        cls = owner if owner is not None else type(obj)
        bound = self._bound.get(cls)
        if bound is None:
            kind = self.kind

            def decorator(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
                return cls.jit_runner(
                    cls,
                    kind,
                    cls.get_location_from_frame(
                        inspect.currentframe().f_back  # type: ignore[union-attr]
                    ),
                    *decorator_args,
                    **decorator_kwargs,
                )

            decorator.__name__ = kind
            decorator.__qualname__ = f"{cls.__name__}.{kind}"
            decorator.__doc__ = (
                f"The ``@{kind}`` decorator of ``{cls.__name__}``, added by a plugin."
            )
            bound = self._bound[cls] = decorator
        return bound


# ===----------------------------------------------------------------------===
# DSL Implementation
# ===----------------------------------------------------------------------===


class BaseDSL(metaclass=DSLSingletonMeta):
    """A DSL's decorator lifecycle and dialect-independent tracing services."""

    _abstract_dsl_base: ClassVar[bool] = True

    plugins: ClassVar[Plugins] = Plugins()
    _name_mangling_prefix: ClassVar[str] = ""

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        """Install decorators from a record defined on this class.

        Inherited records keep the parent's decorators. ``abstract=True``
        marks a DSL family base that cannot be instantiated directly.
        """
        super().__init_subclass__(**kwargs)
        cls._abstract_dsl_base = abstract
        record = cls.__dict__.get("plugins")
        if record is not None and not isinstance(record, Plugins):
            raise DSLRuntimeError(
                f"`{cls.__name__}.plugins` must be a `Plugins` record, not a "
                f"`{type(record).__name__}`: write "
                "`plugins = Plugins(decorators=[func.Jit()])`"
            )
        if isinstance(record, Plugins):
            for plugin in record.decorators:
                for name, decorator in plugin.decorators(cls).items():
                    if name in cls.__dict__:
                        continue  # the class defines its own on purpose
                    inherited = next(
                        (
                            k.__dict__[name]
                            for k in cls.__mro__[1:]
                            if name in k.__dict__
                        ),
                        None,
                    )
                    if inherited is not None and not isinstance(
                        inherited, _PluginDecorator
                    ):
                        raise DSLRuntimeError(
                            f"`{type(plugin).__name__}` adds a decorator `@{name}`, but "
                            f"`{name}` is already a `BaseDSL` attribute; pick another "
                            "`decorator_name`"
                        )
                    setattr(cls, name, decorator)

    def __init__(self, *, name: str) -> None:
        if not name:
            raise DSLRuntimeError("a DSL needs a name: its environment prefix")
        self.name = name
        self.decorator_location: DSLLocation | None = None
        self.plugins, self.unavailable_plugins = type(self).plugins.resolve()
        for plugin in self.plugins:
            plugin.install(self)

    @classmethod
    def _get_dsl(cls) -> Any:
        return cls()

    def register_dialects(self, context: ir.Context) -> None:
        """Register any out-of-tree dialects used by the DSL's builders."""

    # ===------------------------------------------------------------------===
    # Decorator Lifecycle
    # ===------------------------------------------------------------------===

    @staticmethod
    def _lazy_initialize_dsl(func: Any) -> None:
        """Resolve a decorated function's DSL instance on its first call."""
        if hasattr(func, "_dsl_cls"):
            func._dsl_object = func._dsl_cls._get_dsl()
            delattr(func, "_dsl_cls")

    @staticmethod
    def jit_runner(
        cls: type["BaseDSL"],
        kind: str,
        location: DSLLocation,
        *decorator_args: Any,
        **decorator_kwargs: Any,
    ) -> Any:
        """Validate and wrap a function with its plugin's decorator.

        ``decorate`` processes options before wrapping; ``wrap`` sees the
        completed wrapper. Calls resolve the DSL lazily and dispatch to its
        installed plugin, entering ``call_scope`` for top-level calls.

        Capture the decorator's source location, not its frame: the wrapper
        outlives the decorating call.
        """
        # The record's plugin, found on the class: the instance is still lazy.
        plugin = cls._decorator_plugin(kind)

        def jit_runner_decorator(func: Any) -> Any:
            if not inspect.isfunction(func):
                raise DSLUserCodeError(
                    DiagId.CALL_NOT_CALLABLE,
                    decorator=f"@{kind}",
                    got=f"a `{type(func).__name__}`",
                )
            func._dsl_cls = cls
            func._decorator_kind = kind
            func._decorator_location = location
            if plugin is not None:
                func = plugin.decorate(func, **decorator_kwargs)
            elif decorator_kwargs:
                unknown = sorted(decorator_kwargs)
                raise DSLUserCodeError(
                    DiagId.CALL_ARGUMENTS,
                    function_name=f"@{kind}",
                    detail=f"no option is named `{unknown[0]}`",
                )

            @wraps(func)
            def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Resolve the instance plugin at call time; retain no bound method.
                if in_trace():
                    cls._lazy_initialize_dsl(func)
                    func._dsl_object.decorator_location = func._decorator_location
                    return cls._dispatch_decorated(func, *args, **kwargs)
                with cls.call_scope(func):
                    cls._lazy_initialize_dsl(func)
                    func._dsl_object.decorator_location = func._decorator_location
                    return cls._dispatch_decorated(func, *args, **kwargs)

            if plugin is not None:
                return plugin.wrap(func, jit_wrapper)
            return jit_wrapper

        if len(decorator_args) == 1 and callable(decorator_args[0]):
            return jit_runner_decorator(decorator_args[0])  # bare `@jit`
        if decorator_args:
            # Decorator options must be keyword arguments.
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=f"@{kind}",
                detail=f"it takes no option, but {len(decorator_args)} positional "
                "value(s) were given",
            )
        return jit_runner_decorator  # `@jit()`

    @classmethod
    def make_decorator(cls, kind: str) -> Any:
        """Create a class-bound decorator for a plugin's ``decorators`` mapping.

        Both ``@DSL.jit`` and ``@DSL.jit(...)`` bind the class they are accessed
        through, including a subclass inheriting the decorator.
        """
        return _PluginDecorator(kind)

    @classmethod
    def _decorator_plugin(cls, kind: str) -> DecoratorPlugin | None:
        """Find a decorator in the class records before an instance exists."""
        for klass in cls.__mro__:
            record = klass.__dict__.get("plugins")
            if isinstance(record, Plugins):
                plugin = record.decorator(kind)
                if plugin is not None:
                    return plugin
        return None

    @classmethod
    def _dispatch_decorated(cls, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Dispatch through the installed plugin, diagnosing missing plugins."""
        dsl = func._dsl_object
        kind = func._decorator_kind
        plugin = dsl.plugins.decorator(kind)
        if plugin is None:
            installed = cls._decorator_plugin(kind)
            plugin_name = (
                "DecoratorPlugin" if installed is None else type(installed).__name__
            )
            raise DSLUserCodeError(
                DiagId.CALL_PLUGIN_REQUIRED,
                name=f"@{kind}",
                plugin=f"the `{plugin_name}` plugin",
                fix=f"plugins = Plugins(decorators=[{plugin_name}()])",
                context=(
                    {"unavailable plugins": dict(dsl.unavailable_plugins)}
                    if dsl.unavailable_plugins
                    else None
                ),
            )
        return plugin._dispatch(dsl, func, *args, **kwargs)

    @classmethod
    def call_scope(cls, func: Any) -> Any:
        """Provide a scope for top-level calls, including lazy initialization.

        A DSL can use this for locking or per-call settings. Nested calls
        share the outer call's scope.
        """
        return nullcontext()

    # ===------------------------------------------------------------------===
    # Source Locations
    # ===------------------------------------------------------------------===

    @staticmethod
    def get_location_from_frame(frame: Any) -> DSLLocation:
        """The :class:`DSLLocation` of the Python ``frame`` (a decorator's call site)."""
        return DSLLocation(
            filename=inspect.getsourcefile(frame) or "<unknown>",
            lineno=frame.f_lineno,
            col_offset=0,
            function_name=frame.f_code.co_name,
        )

    def get_ir_location(self, location: DSLLocation | None = None) -> Any:
        """Convert a Python source location to a named MLIR location."""
        if location is None:
            location = self.decorator_location
        if location is None:
            return ir.Location.unknown()

        file_loc = ir.Location.file(
            location.filename, location.lineno, location.col_offset
        )
        return ir.Location.name(location.function_name, childLoc=file_loc)

    # ===------------------------------------------------------------------===
    # Argument Binding
    # ===------------------------------------------------------------------===

    def _get_function_bound_args(
        self, sig: inspect.Signature, function_name: str, *args: Any, **kwargs: Any
    ) -> inspect.BoundArguments:
        """Bind supplied arguments and apply defaults; missing values are checked later."""
        try:
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
        except TypeError as e:
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=function_name,
                detail=f"{len(args)} positional and {len(kwargs)} keyword argument(s) do not bind to its runtime parameters ({e})",
                cause=e,
            ) from e
        return bound_args

    def _canonicalize_args(
        self, bound_args: inspect.BoundArguments
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Separate bound arguments into positional and keyword arguments."""
        return bound_args.args, bound_args.kwargs

    def _check_arg_count(
        self,
        sig: inspect.Signature,
        bound_args: inspect.BoundArguments,
        function_name: str,
    ) -> bool:
        """Raise ``CALL_ARGUMENTS`` for a parameter without default that
        ``bound_args`` leaves unbound.

        :return: Whether the signature has ``*args`` or ``**kwargs``
        """
        has_varargs = False
        for param in sig.parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                has_varargs = True
                continue
            if (
                param.default is inspect.Parameter.empty
                and param.name not in bound_args.arguments
            ):
                raise DSLUserCodeError(
                    DiagId.CALL_ARGUMENTS,
                    function_name=function_name,
                    detail=f"no value for `{param.name}`",
                )
        return has_varargs

    def _get_signature(self, func: Callable[..., Any]) -> inspect.Signature:
        """Read the function signature and resolve string annotations."""
        try:
            return inspect.signature(func, eval_str=True)
        except NameError as e:
            raise DSLUserCodeError(
                DiagId.SCOPE_UNBOUND_NAME_IN_TRACE,
                var=getattr(e, "name", None) or "<annotation>",
                function_name=getattr(func, "__name__", "<function>"),
                cause=e,
            ) from e

    def mangle_name(
        self, function_name: str, args: tuple[Any, ...], sig: inspect.Signature
    ) -> str:
        """Include compile-time positional arguments in the function symbol."""
        for _, arg in zip(sig.parameters.values(), args):
            if isinstance(arg, type):
                function_name = f"{function_name}_{arg.__name__}"
            elif isinstance(arg, (list, tuple)):
                function_name = f"{function_name}_{'_'.join(map(str, arg))}"
            else:
                function_name = f"{function_name}_{arg}"
        function_name = re.sub(r"0x[a-fA-F0-9]{8,16}", "", function_name)
        function_name = re.sub(r"[^a-zA-Z0-9_$.-]", "_", function_name)[:180]
        return self._name_mangling_prefix + function_name

    # ===------------------------------------------------------------------===
    # IR Tracing
    # ===------------------------------------------------------------------===

    def trace_body(
        self,
        entry: DecoratorPlugin,
        name: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        arg_types: list[Any],
        arg_attrs: list[Any],
        *,
        loc: Any = None,
    ) -> tuple[Any, ir.Block, Any]:
        """Build the entry op and run the Python body at its insertion point."""
        func_op, entry_block = entry.generate_func_op(
            name, list(arg_types), list(arg_attrs), loc=loc
        )
        with ir.InsertionPoint(entry_block):
            body_scope = getattr(entry, "body_scope", None)
            scope = (
                nullcontext()
                if body_scope is None
                else body_scope(self, func, signature=sig)
            )
            with scope:
                result = func(*args, **kwonlyargs)
        return func_op, entry_block, result

    def generate_original_ir(
        self,
        entry: DecoratorPlugin,
        func: Callable[..., Any],
        function_name: str,
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        location: DSLLocation | None = None,
    ) -> ir.Module:
        """Trace one function into a fresh module and verify the completed IR."""
        loc = self.get_ir_location(location)
        module = ir.Module.create(loc=loc)
        with ir.InsertionPoint(module.body):
            func_op, entry_block, result = self.trace_body(
                entry, function_name, func, args, kwonlyargs, sig, [], [], loc=loc
            )
            if result is not None or sig.return_annotation not in (
                inspect.Signature.empty,
                None,
            ):
                raise DSLUserCodeError(
                    DiagId.TYPE_RETURN_MISMATCH,
                    got=f"a `{type(result).__name__}`",
                    detail=" (only functions returning None are supported)",
                )
            with ir.InsertionPoint(entry_block):
                entry.generate_return(func_op, [], loc=loc)
        module.operation.verify()
        return module

    def run(
        self,
        entry: DecoratorPlugin,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Bind Python arguments, trace the function, and optionally print its IR."""
        # Reject reserved framework keywords before binding, including None
        # and False values. They are never Python specialization arguments.
        reserved = sorted(
            {
                "pipeline",
                "container_attrs",
                "no_cache",
                "extra_link_libs",
                "compile_only",
            }.intersection(kwargs)
        )
        if reserved:
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=func.__name__,
                detail=f"`{reserved[0]}` is a reserved framework keyword; "
                "compilation options are not supported by this tracing interface",
            )
        sig = self._get_signature(func)
        bound = self._get_function_bound_args(sig, func.__name__, *args, **kwargs)
        self._check_arg_count(sig, bound, func.__name__)
        args, kwargs = self._canonicalize_args(bound)
        # An SSA value from another context is not a Python specialization.
        # SSA operands passed between helpers inside a trace bypass this entry.
        if any(isinstance(arg, ir.Value) for arg in (*args, *kwargs.values())):
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=func.__name__,
                detail="the host entry accepts only compile-time Python arguments",
            )
        function_name = self.mangle_name(func.__name__, args, sig)
        with active_dsl(self), ir.Context() as ctx:
            ctx.enable_multithreading(False)
            self.register_dialects(ctx)
            for plugin in self.plugins:
                plugin.register_dialects(ctx)
            with self.get_ir_location(func._decorator_location):
                module = self.generate_original_ir(
                    entry,
                    func,
                    function_name,
                    args,
                    kwargs,
                    sig,
                    location=func._decorator_location,
                )
                if os.environ.get(f"{self.name}_PRINT_IR", "0") == "1":
                    print(module)
