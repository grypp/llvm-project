# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The DSL base class :class:`BaseDSL`.

A sub-DSL inherits :class:`BaseDSL` and names its plugins in one ``Plugins``
record on the class. The base owns what is the same for every DSL: the
decorator machinery every decorator plugin builds on
(:meth:`BaseDSL.make_decorator`, :meth:`BaseDSL.jit_runner`; ``@jit`` itself
is the ``func.Jit`` plugin's), the AST
preprocessor run, the host argument boundary, tracing the body into the entry
the decorator plugin builds, the pass pipeline, the in-memory and on-disk
compile caches and the invocation through the ``compiler`` plugin, and the
services a decorator plugin traces with (:meth:`BaseDSL.bind_arguments`,
:meth:`BaseDSL.trace_body`).
"""

import hashlib
import inspect
import io
import logging
import os
import re
import threading
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from functools import partial, wraps
from types import UnionType
from typing import Annotated, Any, ClassVar, NoReturn, Union, get_args, get_origin

from ... import ir
from ..types import typing as t
from ..util import profiler, tree_utils
from ..util.cache import (
    JitCacheDict,
    ModuleLocks,
    dump_cache_to_path,
    framework_identity,
    get_default_generated_ir_path,
    load_cache_from_path,
    read_bytecode_and_check_crc32,
    save_ir,
    toolchain_identity,
    write_bytecode_with_crc32,
)
from ..util.logger import log
from ..util.profiler import timer
from .arguments import JitArgAdapterRegistry, adapt_pointer_address
from .common import (
    DSLBaseError,
    DSLRuntimeError,
    DSLUserCodeError,
    DSLUserCodeValueError,
    DSLWarning,
    _user_error,
    active_dsl,
    in_trace,
)
from .diagnostics import DiagId, filter_traceback, find_user_source_location
from .env_manager import EnvironmentVarManager
from .plugin import DecoratorPlugin, Plugins, _NoRemarkSession
from .remarks import REMARK_POLICIES
from .staging import Executor, _is_mlir_op_leaf, is_argument_meta
from .user_op import enter_traceback_locations

__all__ = [
    "BaseDSL",
    "CompileStep",
    "DSLLocation",
    "DSLSingletonMeta",
    "JitFuncArgs",
    "PendingCompile",
    "normalize_shared_library_paths",
]

# ===----------------------------------------------------------------------===
# Global Variables
# ===----------------------------------------------------------------------===

# Characters stripped from a mangled function name, and the precomputed
# translation table (mangle_name runs per compile, plus once per kernel trace).
_MANGLE_UNWANTED_CHARS = r"'-![]#,.<>()\":{}=%?@;"
_MANGLE_TRANSLATION_TABLE = str.maketrans("", "", _MANGLE_UNWANTED_CHARS)


def _normalize_shared_library_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return existing, canonical shared-library paths without duplicates."""
    normalized_paths: list[str] = []
    for path in paths:
        if not path:
            raise DSLRuntimeError("an empty shared library path was given")
        normalized_path = os.path.realpath(os.path.abspath(path))
        if not os.path.exists(normalized_path):
            raise DSLRuntimeError(
                f"shared library not found: {normalized_path}",
                suggestion="Check the paths in `<PREFIX>_LIBS` and `extra_link_libs`.",
            )
        if normalized_path not in normalized_paths:
            normalized_paths.append(normalized_path)
    return tuple(normalized_paths)


# The public name: a DSL normalizes its own library lists the same way (an
# empty entry and a missing file are ``DSLRuntimeError``).
normalize_shared_library_paths = _normalize_shared_library_paths


# ===----------------------------------------------------------------------===
# DSL Class Construction
# ===----------------------------------------------------------------------===


class DSLSingletonMeta(type):
    """
    Metaclass implementing the Singleton pattern for DSL classes.

    One instance exists per concrete subclass, kept in ``_instances``.
    Requesting an abstract base (``BaseDSL`` itself, or a class defined with
    ``abstract=True``) returns the first instance of one of its concrete
    subclasses instead of building one of its own; ``clear_instances`` serves
    tests.
    """

    _instances: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            log().info("DSLSingletonMeta __call__ for %s", cls)
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
            log().info("Active DSL singleton instances: %s", cls._instances)
            return cls._instances[cls]

    def clear_instances(cls) -> None:
        """Forget the instance of ``cls`` so the next ``cls()`` builds a new one."""
        log().info("Clearing DSL singleton instances for %s", cls)
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


@dataclass(frozen=True)
class JitFuncArgs:
    """Arguments related to a compiled function.

    ``values`` are the marshalled runtime values, ``types`` and ``attributes``
    the MLIR argument types and attribute dicts; ``adapted_python_args`` holds,
    per Python argument in signature order, ``None`` when the argument was
    used as is or the value the annotation cast, the ``Pointer[T]`` address
    adaptation or a JitArgAdapter produced.
    """

    values: list[Any]
    types: list[Any]
    attributes: list[Any]
    adapted_python_args: list[Any]


@dataclass(frozen=True)
class _ResultSpec:
    """How the result of a compiled function maps back to Python: the flattened
    shape of the traced return, the host-side slot descriptor the decorator
    plugin built for it (opaque to the core; the shipped ``func.Jit`` uses a
    ``ctypes`` type) and that plugin, which reads the slot back."""

    treedef: Any
    slot: Any
    entry: Any


@dataclass
class _TraceResult:
    """What one trace of a call produced, handed to the compile half of the
    call (:meth:`BaseDSL.after_trace_result`, :meth:`BaseDSL.emit_artifact`,
    :meth:`BaseDSL.invoke_compiled`): the verified module and its cache key,
    the Python value the body returned and the slot its compiled result comes
    back through, and the call's arguments in every form the rest of the call
    reads (the canonical Python values, the values the trace saw, the
    marshalled runtime arguments of the packed entry)."""

    module: ir.Module
    module_hash: str | None
    result: Any
    result_spec: _ResultSpec | None
    function_name: str
    pipeline: str
    sig: inspect.Signature
    args: tuple[Any, ...]
    kwonlyargs: dict[str, Any]
    trace_args: tuple[Any, ...]
    exe_args: list[Any]
    extra_link_libs: tuple[str, ...]


@dataclass
class PendingCompile:
    """The rest of a call, deferred: :meth:`BaseDSL.emit_artifact` returns one
    instead of a result, and the decorated function's wrapper runs it
    (:meth:`finish`) once the DSL's ``call_scope`` has been left, or inside
    the scope when ``needs_call_scope`` is set. ``run`` is the DSL's compile
    step and whatever follows it (an invocation, a handle); ``scope`` is the
    trace's context (the ``ir.Context``, location and remark session the core
    held for the trace), filled in by the core and closed after ``run``. The
    step runs on the calling thread, with no DSL active."""

    run: Callable[[], Any]
    scope: Any = None
    needs_call_scope: bool = False

    def finish(self) -> Any:
        """Run the step inside its scope, then close the scope: the rest of
        the call."""
        with self.scope if self.scope is not None else nullcontext():
            return self.run()


@dataclass
class CompileStep:
    """One compile of a traced module, built by :meth:`BaseDSL.prepare_compile`
    and run by :meth:`run`: the stages of ``compile_and_cache`` as an object a
    DSL can hold, hand to another thread or run later (the step of a
    :class:`PendingCompile`). ``compile_options`` are the DSL's
    ``_compile_options()`` as evaluated when the step was prepared, so the
    step never reads the DSL's settings later; ``post_processors`` run on the
    compiled function once per compile, never on a cached function;
    ``build_engine=False`` lowers the module without an engine, for a call
    whose artifact is the lowered IR (the returned handle refuses calls);
    ``func_type`` is the compiled-function class the compiler plugin's
    ``load`` builds. :meth:`build` is the backend stage: a DSL compiling
    through something other than the compiler plugin's engine overrides it in
    a subclass named by ``BaseDSL._compile_step_class``."""

    dsl: "BaseDSL"
    module: ir.Module
    module_hash: str | None
    function_name: str
    pipeline: str
    sig: inspect.Signature
    no_cache: bool
    result_spec: _ResultSpec | None = None
    extra_link_libs: tuple[str, ...] = ()
    func: Callable[..., Any] | None = None
    func_type: Any = None
    build_engine: bool = True
    compile_options: dict[str, Any] = field(default_factory=dict)
    post_processors: list[Callable[[Any], None]] = field(default_factory=list)
    #: Set by :meth:`run` when the lowered module came from the file cache;
    #: :meth:`build` then skips the pass pipeline.
    from_file_cache: bool = field(default=False, init=False)

    def run(self) -> Any:
        """Compile the module and publish the compiled function.

        Under the module hash's lock (``BaseDSL.module_locks``): a thread
        that had to wait for the hash checks the in-memory cache again and
        takes the function the first thread published (its post-processors
        skipped); otherwise the file cache is consulted (a hit replaces the
        module with its lowered form), then :meth:`build` runs, the decorator
        plugins' ``finish_compiled_function``, the adapters'
        ``wrap_compiled_function`` and the post-processors, and the function
        is published to the memory cache and, after a file-cache miss, to the
        file cache. Nothing is locked or cached under ``no_cache``.

        :return: The compiled function
        """
        dsl = self.dsl
        keys = () if self.no_cache or self.module_hash is None else (self.module_hash,)
        with dsl.module_locks.hold(keys) as waited:
            if waited:
                cached = dsl.jit_cache.get(self.module_hash)
                if cached is not None and cached.capi_func is not None:
                    dsl.cache_hits += 1
                    log().info(
                        "JIT cache hit IN-MEMORY function=[%s] module_hash=[%s]",
                        self.function_name,
                        self.module_hash,
                    )
                    return cached
            file_cache_enabled = bool(keys) and not dsl.envar.disable_file_caching
            if file_cache_enabled:
                cached_module = load_cache_from_path(
                    dsl.name,
                    self.module_hash,
                    path=get_default_generated_ir_path(dsl.name),
                    bytecode_reader=read_bytecode_and_check_crc32,
                )
                if cached_module is not None:
                    dsl.file_cache_hits += 1
                    log().info(
                        "JIT cache hit IN-FILE function=[%s] module_hash=[%s]",
                        self.function_name,
                        self.module_hash,
                    )
                    self.module = cached_module
                    self.from_file_cache = True
            if not self.from_file_cache:
                dsl.cache_misses += 1
                log().info(
                    "JIT cache miss function=[%s] module_hash=[%s]",
                    self.function_name,
                    self.module_hash,
                )
            jit_function = self.build()
            jit_function.result_spec = self.result_spec  # type: ignore[attr-defined]
            execution_args = getattr(jit_function, "execution_args", None)
            if execution_args is not None:
                execution_args.set_adapter_scope(dsl._jit_arg_adapter_scope)
            # Decorator plugins record what the trace did; adapters then wrap
            # the function for another ABI (their symbols are in the engine).
            for plugin in dsl.plugins.decorators:
                plugin.finish_compiled_function(dsl, jit_function)
            for plugin in dsl.plugins.adapters:
                jit_function = plugin.wrap_compiled_function(dsl, jit_function)
            for process in self.post_processors:
                process(jit_function)

            if not self.no_cache:
                # The entry lives as long as the decorated function does.
                dsl.jit_cache.set(self.module_hash, jit_function, self.func)
            if file_cache_enabled and not self.from_file_cache:
                dump_cache_to_path(
                    dsl.name,
                    jit_function,
                    self.module_hash,
                    path=get_default_generated_ir_path(dsl.name),
                    bytecode_writer=lambda f: write_bytecode_with_crc32(
                        f, jit_function.ir_module
                    ),
                )
            return jit_function

    def build(self) -> Any:
        """The backend stage: lower the module (unless it came lowered from
        the file cache), build its engine (unless ``build_engine`` is off)
        and bind the entry through the compiler plugin's ``load``, with the
        DSL's ``_load_options``.

        :return: The compiled function the plugin's ``load`` built
        """
        dsl = self.dsl
        if not self.build_engine:
            if not self.from_file_cache:
                dsl._maybe_profile(dsl.compile_module)(
                    self.module,
                    self.pipeline,
                    self.function_name,
                    **self.compile_options,
                )
            engine = None
        else:
            shared_libs = dsl.get_shared_libs(self.extra_link_libs)
            if self.from_file_cache:
                engine = dsl._maybe_profile(dsl.jit_lowered_module)(
                    self.module, shared_libs, self.function_name
                )
            else:
                engine = dsl._maybe_profile(dsl.compile_and_jit)(
                    self.module,
                    self.pipeline,
                    shared_libs,
                    self.function_name,
                    **self.compile_options,
                )
        # Binding the entry materializes the JIT'd code (ORC compiles lazily),
        # so it belongs to the jit phase together with engine construction.
        load = profiler.timed("jit")(dsl._maybe_profile(dsl._compiler.load))
        options = {
            "jit_time_profiling": dsl.envar.jit_time_profiling,
            "result_ctype": (
                self.result_spec.slot if self.result_spec is not None else None
            ),
            "func_type": self.func_type,
            "libs_var": f"{dsl.envar.prefix}_LIBS",
            **dsl._load_options(self),
        }
        return load(self.module, engine, self.function_name, self.sig, **options)


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
    """The base of every DSL: one singleton instance per concrete subclass.

    The class attributes below are the sub-DSL knobs; the instance attributes
    set in ``__init__`` are the per-DSL state (environment manager, caches,
    executors, installed plugins). Every decorator, ``@MyDSL.jit`` included,
    comes from a decorator plugin of the record and is installed on the class
    by ``__init_subclass__``; all dispatch through :meth:`jit_runner` to the
    plugin's ``call`` (from Python) or ``launch`` (inside a trace).
    """

    _env_class: type[EnvironmentVarManager] = EnvironmentVarManager
    #: The class of the in-memory JIT cache ``__init__`` builds as
    #: ``jit_cache`` (``max_elems=`` sized from the environment); a DSL with a
    #: cache policy of its own names a ``JitCacheDict`` subclass.
    _jit_cache_class: ClassVar[type[JitCacheDict]] = JitCacheDict
    #: The class of the compile steps :meth:`prepare_compile` builds; a DSL
    #: whose compile runs through another backend names a ``CompileStep``
    #: subclass with its own ``build``.
    _compile_step_class: ClassVar[type[CompileStep]] = CompileStep
    #: True on an abstract DSL base: ``BaseDSL`` itself and every class
    #: defined with ``class Family(BaseDSL, abstract=True)``. Calling one
    #: returns the instance of a concrete subclass, never builds its own.
    _abstract_dsl_base: ClassVar[bool] = True
    _jit_arg_adapter_scope: ClassVar[str | None] = None
    # Optional compiler-recognized component inserted by a DSL's name mangler.
    _name_mangling_prefix: ClassVar[str] = ""
    #: The class of the errors the core raises for the author's mistakes: a
    #: ``DSLUserCodeError`` subclass taking its constructor arguments. Every
    #: core raise site builds its error through :meth:`user_error`, so a DSL
    #: with an exception hierarchy of its own sees the core's errors arrive as
    #: its members.
    user_error_class: ClassVar[type[DSLUserCodeError]] = DSLUserCodeError
    #: Catalog ids substituted at the core raise sites, ``{core id: own id}``:
    #: a DSL with a diagnostic catalog of its own reports the core's mistakes
    #: under its codes and wording (the fields of the raise site fill the
    #: alias's templates).
    diag_id_aliases: ClassVar[dict[Any, Any]] = {}
    #: The warnings ``<PREFIX>_WARNINGS_IGNORE`` silences: the DSL's own
    #: (``DSLWarning`` and its subclasses), never every warning of the process.
    _warning_category: ClassVar[type[Warning]] = DSLWarning
    #: What the DSL is made of: one plugin per role (``type_ops``,
    #: ``ast_preprocessor``, ``compiler``) and
    #: the families (``decorators``, ``adapters``). ``__init__``
    #: resolves the record once per
    #: instance: a plugin whose ``available()`` is False is dropped (and listed
    #: in ``unavailable_plugins``), the rest are copied and installed in record
    #: order. The base names nothing; a sub-DSL assembles itself here.
    plugins: ClassVar[Plugins] = Plugins()

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

    @classmethod
    def user_error(cls, diag_id: Any, **fields: Any) -> DSLUserCodeError:
        """The error this DSL raises for ``diag_id``: an instance of
        :attr:`user_error_class` carrying the id :attr:`diag_id_aliases` maps
        ``diag_id`` to, with ``fields`` as ``DSLUserCodeError`` takes them.
        The core's raise sites call it on the DSL (or DSL class) in hand;
        ``common._user_error`` is the same for the active DSL."""
        return _user_error(diag_id, cls, **fields)

    def on_trace_name_error(
        self,
        func: Callable[..., Any],
        error: NameError,
        filename: str | None,
        lineno: int | None,
    ) -> NoReturn:
        """What a ``NameError`` met while tracing ``func`` becomes.

        ``trace_body`` calls it with the error and the user's line that raised
        it (the innermost frame of its traceback). The default reports
        ``SCOPE_UNBOUND_NAME_IN_TRACE`` there through :meth:`user_error`: a
        name read on a path that never set it, the usual cause once the
        rewrite has outlined a branch. A DSL whose rewrite introduces names of
        its own overrides this to tell its unbound reads apart; a hook that
        returns lets the ``NameError`` propagate as it is."""
        raise self.user_error(
            DiagId.SCOPE_UNBOUND_NAME_IN_TRACE,
            filename=filename,
            lineno=lineno,
            cause=error,
            var=getattr(error, "name", None) or f"<in {func.__name__}>",
            function_name=func.__name__,
        ) from error

    def _remark_session(self, context: ir.Context) -> Any:
        """The compiler plugin's remark session for ``context`` under the
        ``REMARKS``/``REMARKS_POLICY``/``REMARKS_OUTPUT`` settings; a session
        collecting nothing for a DSL without a compiler (a trace-only DSL)."""
        compiler = self.plugins.compiler
        if compiler is None:
            return _NoRemarkSession()
        return compiler.remark_session(context, **self._remark_options())

    def _remark_options(self) -> dict[str, Any]:
        """The session options of a compile, handed to the compiler plugin's
        ``remark_session`` (directly, and through ``compile``): the
        ``REMARKS``, ``REMARKS_POLICY`` and ``REMARKS_OUTPUT`` settings. A DSL
        whose compiler plugin takes other options overrides this."""
        return {
            "remark_filter": self.envar.remarks,
            "remark_policy": self._remarks_policy,
            "remark_output": self.envar.remarks_output,
        }

    @property
    def _remarks_policy(self) -> str:
        """The ``REMARKS_POLICY`` setting, checked once remarks are requested:
        one of ``REMARK_POLICIES`` (a user's typo is ``CONFIG_INVALID``)."""
        policy = self.envar.remarks_policy
        if self.envar.remarks and policy not in REMARK_POLICIES:
            raise DSLUserCodeValueError(
                DiagId.CONFIG_INVALID,
                var=f"{self.envar.prefix}_REMARKS_POLICY",
                detail=f"`{policy}` is not a remark policy; expected one of "
                + ", ".join(f"`{name}`" for name in REMARK_POLICIES),
            )
        return policy

    def __init__(
        self,
        *,
        name: str,
        dsl_package_name: list[str] | None = None,
    ) -> None:
        """
        Initialize the DSL with its providers and environment settings.

        :param name: Name of the DSL; the environment variable prefix
            (``<name>_DRYRUN``, ...) and the logging label
        :param dsl_package_name: The package the AST rewrite imports
            ``and_``/``or_``/``not_``/``as_ir_value`` from, as path parts; None
            means the ``ast_preprocessor`` plugin's own package. Only a DSL
            that re-exports those helpers under its own namespace names it.

        Reads the environment through ``EnvironmentVarManager``, configures
        warnings and logging, and installs the class's ``plugins`` record last,
        so the subclass's own ``__init__`` still runs after them and wins. The
        DSL rewrites native control flow exactly when its record names an
        ``ast_preprocessor`` plugin with a preprocessor (and
        ``<PREFIX>_AST_PREPROCESSOR`` is not 0); there is no other switch.
        """
        # Enforcing initialization of instance variables
        if not name:
            raise DSLRuntimeError("a DSL needs a name: its environment prefix")

        self.name: str = name
        self.decorator_location: DSLLocation | None = None
        # Read environment variables
        self.envar: EnvironmentVarManager = self._create_environment_manager()
        # Set once the plugins are installed: True exactly when the record
        # names an ``ast_preprocessor`` plugin with a preprocessor and the
        # environment does not turn the rewrite off.
        self.enable_preprocessor: bool = False
        self.preprocessor: Any = None
        # This cache uses hash of original ir and env as key. Enabled by default
        self.jit_cache: JitCacheDict = self._jit_cache_class(
            max_elems=0 if self.envar.no_cache else self.envar.jit_cache_max_elems
        )
        # One compile per module hash at a time: a second thread with the same
        # hash waits for the first and takes its compiled function.
        self.module_locks: ModuleLocks = ModuleLocks()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.file_cache_hits: int = 0
        # The prepared calls (``DeferredDecoratorCall``) of the open top-level trace,
        # checked when the trace ends; None outside a trace.
        self._deferred_calls: list[Any] | None = None
        # The structured remarks of the last compile (``RemarkSession.remarks``).
        self.collected_remarks: list[dict[str, Any]] = []

        # The DSL's own warnings only: the process's other warnings are not
        # this DSL's to silence.
        if self.envar.warnings_ignore:
            warnings.filterwarnings("ignore", category=self._warning_category)

        # Path of the dumped MLIR file; set by build_module when KEEP_IR is active.
        self.dump_mlir_path: Any = None
        # The same raw dump under the stage's name, and every dump of the last
        # build_module by stage (``raw``, ``clean``, ``after-<pass>``, ...).
        self.dump_mlir_raw_path: Any = None
        self.dump_mlir_paths: dict[str, str] = {}
        # The function being traced; set on every path that reaches the
        # argument boundary (``_prepare_compilation``, ``bind_arguments``).
        self.traced_function: Callable[..., Any] | None = None
        # The control-flow executors of this instance, filled from the AST
        # preprocessor plugin's ``executors`` once the plugins are installed.
        self.executor: Executor = Executor()
        log().info("Initializing %s DSL", name)

        if self.envar.jit_time_profiling:
            self.profiler: Any = timer(enable=True)

        # Resolve the class's plugin record for this instance: a copy of every
        # available plugin, installed in record order (roles, then families), so
        # the subclass's own ``__init__`` resumes after them and wins.
        self.plugins, self.unavailable_plugins = type(self).plugins.resolve()
        for plugin in self.plugins:
            plugin.install(self)
        for adapter in self.plugins.adapters:
            adapter.register(self)
        # The AST preprocessor supplies the executors and the preprocessor;
        # naming it is what turns the rewrite on.
        ast_preprocessor = self.plugins.ast_preprocessor
        if ast_preprocessor is not None:
            self.executor.set_functions(**ast_preprocessor.executors(self))
            # The plugin builds the preprocessor, which recognises the
            # decorators the class installs; a plugin that builds none
            # supplies executors only.
            self.preprocessor = ast_preprocessor.make_preprocessor(
                self,
                dsl_package_name or ast_preprocessor.helpers_package(),
                decorator_names=[
                    p.decorator_name for p in type(self).plugins.decorators
                ],
            )
            if self.preprocessor is not None:
                self.enable_preprocessor = bool(self.envar.ast_preprocessor)

    @property
    def _compiler(self) -> Any:
        """The compiler plugin of this DSL; an error when it names none."""
        compiler = self.plugins.compiler
        if compiler is None:
            raise DSLRuntimeError(
                "this DSL has no compiler: name one in its plugins, e.g. "
                "`plugins = Plugins(compiler=execution_engine.Compiler())`",
                context={"dsl": self.name, **self._unavailable_context()},
            )
        return compiler

    def _unavailable_context(self) -> dict[str, Any]:
        """The plugins the class named but ``available()`` rejected, for a
        diagnostic's context; empty when none."""
        if not self.unavailable_plugins:
            return {}
        return {"unavailable plugins": dict(self.unavailable_plugins)}

    def _create_environment_manager(self) -> EnvironmentVarManager:
        """Create the environment manager for this DSL's prefix."""
        return self._env_class(self.name)

    @classmethod
    def _get_dsl(cls) -> Any:
        """The instance of ``cls``; the singleton metaclass builds it once."""
        return cls()  # type: ignore[call-arg]

    def register_dialects(self, context: ir.Context) -> None:
        """Register the DSL's dialects on the trace ``context``, before any op
        is built. Upstream dialects are on every context already, so the
        default does nothing; a sub-DSL on an out-of-tree dialect registers it
        here (``mynewdialect.register_dialect(context)``)."""

    def shared_libs(self) -> list[str]:
        """Library paths the DSL hands to the engine besides ``<PREFIX>_LIBS``
        and the plugins' (``MlirTestDSL``: the CUDA runtime library when gpu
        kernels can be built)."""
        return []

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
    def _filter_call_traceback(func: Any, exc: BaseException) -> None:
        """Drop the DSL's frames from the traceback of ``exc``, raised by a
        call of the decorated ``func`` (:func:`filter_traceback`), when the
        DSL filters stack traces (``<PREFIX>_FILTER_STACKTRACE``, on unless
        ``DEBUG`` or ``SHOW_STACKTRACE`` asks for everything). The region
        functions the AST preprocessor plugin generates
        (``generated_frame_names``) are collapsed into the user's frame. A
        function whose DSL instance does not exist yet (its construction is
        what failed) is left alone."""
        dsl = getattr(func, "_dsl_object", None)
        if dsl is None or not dsl.envar.filter_stacktrace:
            return
        preprocessor = dsl.plugins.ast_preprocessor
        filter_traceback(
            exc, generated_names=getattr(preprocessor, "generated_frame_names", ())
        )

    def preprocess_key(self, func: Any) -> Hashable:
        """What the rewrite of ``func`` depends on: the preprocessor's class
        and the helpers module it imports. A decorated function is rewritten
        once per key (:meth:`_preprocess_and_replace_code`), so a DSL whose
        rewrite has modes (another preprocessor swapped in for a block of
        code) returns a key that changes with them, and the function is
        rewritten again, from its original code, the next time it is called."""
        ast_preprocessor = self.plugins.ast_preprocessor
        helpers_module = (
            None if ast_preprocessor is None else ast_preprocessor.helpers_module
        )
        return (type(self.preprocessor), helpers_module)

    @classmethod
    def _preprocess_and_replace_code(cls, func: Any) -> None:
        """
        Run ast transformation and replace the function's code object, once
        per :meth:`preprocess_key`.
        """
        # Ensure the DSL instance is materialized before touching _dsl_object
        cls._lazy_initialize_dsl(func)
        dsl = func._dsl_object
        # Update the decorator location to the new function
        dsl.decorator_location = func._decorator_location

        key = dsl.preprocess_key(func)
        if (
            getattr(func, "_preprocessed", False) is True
            and getattr(func, "_preprocess_key", None) == key
        ):
            return
        # The code object the function was defined with: every rewrite starts
        # from it, and a rewrite turned off restores it.
        if not hasattr(func, "_original_code"):
            func._original_code = func.__code__
        func.__code__ = func._original_code
        if not dsl.enable_preprocessor:
            func._preprocessed = True
            func._preprocess_key = key
            return

        fcn_ptr = dsl.run_preprocessor(func)
        if fcn_ptr:
            func.__code__ = (
                fcn_ptr.__code__
                if not isinstance(fcn_ptr, staticmethod)
                else fcn_ptr.__func__.__code__
            )
        func._preprocess_key = key

    @staticmethod
    def jit_runner(
        cls: type["BaseDSL"],
        kind: str,
        location: DSLLocation,
        *decorator_args: Any,
        **decorator_kwargs: Any,
    ) -> Any:
        """
        The decorator machinery shared by every decorator a plugin adds
        (``@jit`` included): validate the target, stamp it with the DSL class,
        the decorator's kind and the call site, run the plugin's ``decorate``
        with the decorator's options and wrap the function so that a call
        materialises the DSL instance, rewrites the function (when the DSL
        preprocesses; again when its :meth:`preprocess_key` changed) and hands
        the call to the plugin through :meth:`_dispatch_decorated`; the
        plugin's ``wrap`` sees the wrapper last. A call made outside any trace
        runs inside :meth:`call_scope`. The options (``@jit(label=...)``) are
        the plugin's to take; the default ``decorate`` takes none.

        ``location`` is the user's call site, already resolved to a value by
        the caller via :meth:`get_location_from_frame`: the returned decorator
        outlives this call, so it takes a location rather than a frame.
        """
        log().info("jit_runner")
        # The record's plugin, found on the class: the instance is still lazy.
        plugin = cls._decorator_plugin(kind)

        def jit_runner_decorator(func: Any) -> Any:
            if not inspect.isfunction(func):
                raise cls.user_error(
                    DiagId.CALL_NOT_CALLABLE,
                    decorator=f"@{kind}",
                    got=f"a `{type(func).__name__}`",
                )
            func._dsl_cls = cls
            # The decorator the function carries ("jit", "kernel", ...).
            func._decorator_kind = kind
            func._decorator_location = location
            if plugin is not None:
                func = plugin.decorate(func, **decorator_kwargs)
            elif decorator_kwargs:
                unknown = sorted(decorator_kwargs)
                raise cls.user_error(
                    DiagId.CALL_ARGUMENTS,
                    function_name=f"@{kind}",
                    detail=f"no option is named `{unknown[0]}`",
                )

            @wraps(func)
            def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                # The wrapper holds the DSL class and the function only; the
                # plugin is looked up at call time. No DSL is made active
                # here: the dispatch reads the current one to tell a call from
                # Python (none active; ``run`` then activates this DSL) from a
                # call inside a trace, which runs in the scope of the
                # top-level call it belongs to. A ``PendingCompile`` the
                # dispatch hands back (``emit_artifact``) is the rest of the
                # call, run once the call scope is left unless it asks to run
                # inside it.
                try:
                    if in_trace():
                        cls._preprocess_and_replace_code(func)
                        result = cls._dispatch_decorated(func, *args, **kwargs)
                        if isinstance(result, PendingCompile):
                            return result.finish()
                        return result
                    with cls.call_scope(func):
                        cls._preprocess_and_replace_code(func)
                        result = cls._dispatch_decorated(func, *args, **kwargs)
                        if (
                            isinstance(result, PendingCompile)
                            and result.needs_call_scope
                        ):
                            return result.finish()
                    if isinstance(result, PendingCompile):
                        return result.finish()
                    return result
                except Exception as exc:
                    # The outermost frame of the DSL on a call from Python: the
                    # DSL's frames leave the traceback here when the DSL
                    # filters stack traces. A bare raise adds no entry.
                    cls._filter_call_traceback(func, exc)
                    raise

            if plugin is not None:
                return plugin.wrap(func, jit_wrapper)
            return jit_wrapper

        if len(decorator_args) == 1 and callable(decorator_args[0]):
            return jit_runner_decorator(decorator_args[0])  # bare `@jit`
        if decorator_args:
            # `@jit(3)`: the options are keywords, so a value here is a
            # mistake, not a configuration.
            raise cls.user_error(
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
            raise dsl.user_error(
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
    # Pipeline
    # ===------------------------------------------------------------------===

    def pipeline(self) -> list[str]:
        """The pass list of this DSL, in order: what lowers the ops its plugins
        emit. The base lists nothing (the core emits no dialect); a sub-DSL
        returns its own, spelled against the dialects its plugins emit. Wrapped
        as ``builtin.module(...)`` by :meth:`_get_pipeline`; a ``pipeline=``
        call keyword or ``<PREFIX>_PIPELINE`` replaces it."""
        return []

    def _get_pipeline(self, pipeline: str | None) -> str:
        """The pipeline string of a compile: an explicit ``pipeline`` (a call
        keyword) as given, else the ``<PREFIX>_PIPELINE`` environment variable
        with the plugins' ``pipeline_options()`` appended, else
        :meth:`pipeline` wrapped as ``builtin.module(...)``."""
        if pipeline is not None:
            return pipeline
        if self.envar.pipeline is not None:
            options: dict[str, str] = {}
            for plugin in self.plugins:
                options.update(plugin.pipeline_options())
            return self.preprocess_pipeline(self.envar.pipeline, options)
        return "builtin.module(" + ",".join(self.pipeline()) + ")"

    def preprocess_pipeline(self, pipeline: str, options: dict[str, str]) -> str:
        """Append ``options`` (``name=value`` pairs, the plugins'
        ``pipeline_options()``) to the ``<PREFIX>_PIPELINE`` string, merging
        into an existing ``{...}`` option block when the pipeline has one; no
        options leave it as given."""
        opt_str = ""
        for k, v in options.items():
            if v:
                opt_str += f"{k}={v} "

        if opt_str:
            # Automatically append the pipeline options if any is specified through env var
            match = re.compile(r"{(.+)}").search(pipeline)
            if match:
                opt_str = f"{{{match[1]} {opt_str}}}"
                # A callable replacement substitutes opt_str verbatim: a string
                # replacement is a template, so backslashes in option values
                # (e.g. a Windows path C:\...) would be parsed as escapes.
                pipeline = re.sub(r"{.+}", lambda _: opt_str, pipeline)
            else:
                pipeline = pipeline.rstrip(")") + f"{{{opt_str}}})"
        return pipeline

    # ===------------------------------------------------------------------===
    # Staging
    # ===------------------------------------------------------------------===

    def is_mlir_op(self, value: Any) -> bool:
        """Whether one value is an MLIR op rather than a Python value, for this
        DSL. ``mlir.dsl.is_mlir_op`` calls it for each leaf of a tuple, list or
        frozen record (the containers are walked for you). The default: a raw
        SSA value or a registered leaf whose payload is one. A sub-DSL with its
        own value model overrides it."""
        return _is_mlir_op_leaf(value)

    # ===------------------------------------------------------------------===
    # Name mangling
    # ===------------------------------------------------------------------===

    def _is_meta_argument(self, arg: Any, spec_ty: Any) -> bool:
        """True if the boundary will treat ``arg`` as a compile-time value: no
        DSL annotation, no leaf inside it and no registered adapter for it;
        what remains is judged by :meth:`_is_meta_at_boundary`, the same
        policy the boundary applies, seen before adaptation (the name
        mangling and ``compile()``'s Meta values read it)."""
        if isinstance(arg, (ir.Type, ir.Value)) or isinstance(
            spec_ty, (ir.Type, ir.Value)
        ):
            return False
        annotation = spec_ty
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        if isinstance(annotation, (t.NumericMeta, t.TypedPointer)):
            return False
        if (
            isinstance(annotation, type)
            and tree_utils.leaf_entry(annotation) is not None
        ):
            return False
        if tree_utils.contains_leaf(arg):
            return False
        if isinstance(arg, (list, tuple)):
            # The sequence adapter adapts per element, so a sequence is a Meta
            # value iff every element is one (and is then folded element-wise).
            return all(
                self._is_meta_argument(elem, inspect.Parameter.empty) for elem in arg
            )
        with JitArgAdapterRegistry.using_scope(self._jit_arg_adapter_scope):
            if JitArgAdapterRegistry.get_registered_adapter(arg) is not None:
                return False
        # The position is unknown here, so the receiver rule does not apply.
        return self._is_meta_at_boundary(arg, spec_ty, "", -1, self.traced_function)

    def _mangle_folds(self, arg: Any, annotation: Any) -> bool:
        """Whether :meth:`mangle_name` folds ``arg`` (passed to a parameter
        annotated ``annotation``) into the symbol: by default the Meta
        arguments, the values the trace specialises on. A DSL that wants one
        symbol per value of an annotated scalar too folds those as well."""
        return self._is_meta_argument(arg, annotation)

    def _mangle_render(self, arg: Any, annotation: Any) -> str:
        """The spelling of a folded ``arg`` in the symbol, before the unwanted
        characters are stripped: a type by its name, a sequence element by
        element, anything else through ``str``. A DSL spells its own kinds of
        values here (an instance by its class name, say)."""
        if isinstance(arg, type):
            return arg.__name__
        if isinstance(arg, (list, tuple)):
            return "_".join(map(str, arg))
        return str(arg)

    def mangle_name(
        self, function_name: str, args: tuple[Any, ...], sig: inspect.Signature
    ) -> str:
        """Does simple name mangling: the arguments :meth:`_mangle_folds`
        selects (the Meta arguments by default) are folded into the symbol as
        :meth:`_mangle_render` spells them, so every specialisation of a
        function gets its own name."""

        # sig.parameters maybe longer than args, but since canonicalized_args
        # only contains positional arguments, we can rely on zip to truncate
        for param, arg in zip(sig.parameters.values(), args):
            if not self._mangle_folds(arg, param.annotation):
                continue
            rendered = self._mangle_render(arg, param.annotation)
            function_name = f"{function_name}_{rendered}"
        function_name = function_name.translate(_MANGLE_TRANSLATION_TABLE)
        # Identify addresses and drop them. Match upper-case hex too: Windows
        # (MSVC %p) id() reprs are upper-case zero-padded.
        function_name = re.sub(r"0x[a-fA-F0-9]{8,16}", "", function_name)
        function_name = re.sub(r"\s+", "_", function_name).replace("/", "_")
        # max fname is 256 character, leave space
        function_name = function_name[:180]
        if self._name_mangling_prefix:
            function_name = f"{self._name_mangling_prefix}_{function_name}"
        log().info("Final mangled function name: %s", function_name)
        return function_name

    # ===------------------------------------------------------------------===
    # Block arguments -> traced Python arguments
    # ===------------------------------------------------------------------===

    @staticmethod
    def _restore_tree(treedef: Any, restore_leaf: Callable[[Any], Any]) -> Any:
        """Rebuild the container described by ``treedef``, restoring every
        leaf through ``restore_leaf(leaf)`` and writing META slots back verbatim."""
        if isinstance(treedef, tree_utils.Leaf):
            if treedef.is_none:
                return None
            if treedef.is_meta:
                return treedef.meta
            return restore_leaf(treedef)
        children = [
            BaseDSL._restore_tree(child, restore_leaf)
            for child in treedef.child_treedefs
        ]
        return treedef.node_type.from_iterable(treedef.node_metadata, children)

    @staticmethod
    def _restore_from_block_args(treedef: Any, block_args: Sequence[Any]) -> Any:
        """Rebuild a traced argument from the block arguments of its leaves."""
        values = iter(block_args)

        def restore_leaf(leaf: Any) -> Any:
            entry = tree_utils.leaf_entry(leaf.node_metadata.cls)
            leaf_values = [next(values) for _ in tree_utils.leaf_ir_types(leaf)]
            return entry.from_ir_values(leaf.prototype, leaf_values)

        return BaseDSL._restore_tree(treedef, restore_leaf)

    def generate_execution_arguments(
        self,
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        fop: Any,
        sig: inspect.Signature,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Create list of arguments that will be passed to the traced function body"""

        def gen_exec_arg(
            idx: int,
            arg: Any,
            parameter: inspect.Parameter,
            fop_args: list[Any],
            iv_block_args: int,
        ) -> tuple[Any, int]:
            arg_name = parameter.name
            arg_spec = parameter.annotation
            log().debug("Processing [%d] Argument [%s : %s]", idx, arg_name, arg_spec)

            # A Python value is handed to the body as is; ``traced_function``
            # supplies the owning-function context the meta test uses to
            # detect reserved ``self``/``cls`` parameters.
            if self._is_meta_at_boundary(
                arg, arg_spec, arg_name, idx, self.traced_function
            ):
                return arg, iv_block_args
            with JitArgAdapterRegistry.using_scope(self._jit_arg_adapter_scope):
                ir_arg, consumed = self._restore_argument(
                    arg, arg_spec, fop_args, iv_block_args
                )
            return ir_arg, iv_block_args + consumed

        block = fop if isinstance(fop, ir.Block) else fop.regions[0].blocks[0]
        fop_args = list(block.arguments)
        ir_args = []
        ir_kwargs = {}
        iv_block_args = 0
        for i, (arg, param) in enumerate(zip(args, sig.parameters.values())):
            ir_arg, iv_block_args = gen_exec_arg(i, arg, param, fop_args, iv_block_args)
            ir_args.append(ir_arg)

        for i, (name, arg) in enumerate(kwonlyargs.items()):
            ir_arg, iv_block_args = gen_exec_arg(
                i, arg, sig.parameters[name], fop_args, iv_block_args
            )
            ir_kwargs[name] = ir_arg

        return ir_args, ir_kwargs

    # ===------------------------------------------------------------------===
    # Host argument boundary
    # ===------------------------------------------------------------------===

    def _is_meta_at_boundary(
        self, arg: Any, spec: Any, name: str, idx: int, func: Any
    ) -> bool:
        """Whether the argument ``arg`` of the parameter ``name`` (annotated
        ``spec``, at position ``idx`` of ``func``) is a Python value the trace
        specialises on, the alternative being a runtime value the compiled
        function takes. The default is :func:`is_argument_meta`: decided by
        the annotation and the (adapted) value alone, a value the DSL has no
        leaf for being a Python value. A DSL with another policy (every
        unannotated argument a runtime value, a class of its own speaking a
        protocol the core does not know) decides here, and pairs the verdict
        with :meth:`_flatten_jit_arg` and :meth:`_restore_argument` for the
        values the core cannot flatten or rebuild itself."""
        return is_argument_meta(arg, spec, name, idx, func)

    def _restore_argument(
        self, arg: Any, spec: Any, block_args: Sequence[Any], start: int
    ) -> tuple[Any, int]:
        """Rebuild the value the traced body receives for the runtime argument
        ``arg`` (annotated ``spec``) from the entry's ``block_args``, starting
        at index ``start``; returns the value and the number of block arguments
        it consumed. The default wraps one block argument in the annotated
        dtype when the value is not of it yet (a Python number cast at the
        boundary, ``compile()``'s type-only placeholder; an abstract annotation
        takes the dtype of the block argument), and rebuilds any other value
        leaf by leaf in its Python shape. A DSL restores the values its
        :meth:`_flatten_jit_arg` flattened itself here."""
        base_spec = get_args(spec)[0] if get_origin(spec) is Annotated else spec
        if isinstance(base_spec, t.NumericMeta) and not isinstance(arg, base_spec):
            # Numeric type coercion for a staged argument: the block arg
            # already has the target MLIR type. Wrap it directly instead of
            # casting the caller's value, which would emit casts referencing
            # SSA values from an outer region. An abstract annotation
            # (``Numeric``) is no type to wrap in: the block argument's type,
            # fixed at the boundary, names the dtype (a kernel is traced with
            # the raw call arguments, so the Python number is still here).
            block_arg = block_args[start]
            dtype = (
                t.Numeric.from_mlir_type(block_arg.type)
                if base_spec.is_abstract
                else base_spec
            )
            return dtype(block_arg), 1
        _, _, treedef = tree_utils.tree_flatten(arg, return_ir_values=False)
        n_args = sum(
            len(tree_utils.leaf_ir_types(leaf))
            for _, leaf in tree_utils.tree_leaves(treedef)
        )
        restored = self._restore_from_block_args(
            treedef, block_args[start : start + n_args]
        )
        return restored, n_args

    @staticmethod
    def _is_type_placeholder(arg: Any, spec: Any) -> bool:
        """Whether ``arg`` is a concrete dtype class passed for a parameter
        whose annotation is a numeric DSL type it belongs to: ``compile()``'s
        type-only placeholder, a type with no execution value (the function is
        compiled for the type and called with values later)."""
        if get_origin(spec) is Annotated:
            spec = get_args(spec)[0]
        return (
            isinstance(arg, t.NumericMeta)
            and not arg.is_abstract
            and isinstance(spec, t.NumericMeta)
            and issubclass(arg, spec)
        )

    def _compiled_argument_shape(self, arg: Any, spec: Any) -> Any:
        """The host shape a compiled function remembers for the argument
        ``arg`` (``ExecutionArgs.set_shapes``): its flattened tree; a type-only
        placeholder stands for a value of its dtype."""
        if self._is_type_placeholder(arg, spec):
            arg = arg(0)
        return tree_utils.tree_flatten(arg, return_ir_values=False)[2]

    def _validate_arg(
        self, arg: Any, arg_index: int, arg_name: str, arg_spec: Any
    ) -> Any:
        """Check the (adapted) ``arg`` against a leaf-class or ``Pointer[T]`` annotation.

        A ``Pointer[T]`` annotation requires a ``Pointer``; a ``@struct`` class
        or a registered leaf class other than a ``Numeric`` (``Vector``, a
        sub-DSL leaf) requires an instance of that class. Returns an
        ``ARG_ANNOTATION_MISMATCH`` error, or None when the annotation is of
        another kind or satisfied. A sub-DSL may extend this.
        """
        spec = arg_spec
        if get_origin(spec) is Annotated:
            spec = get_args(spec)[0]
        if isinstance(arg, ir.Value):
            return None
        if isinstance(spec, t.TypedPointer):
            if isinstance(arg, t.Pointer):
                return None
            space = f", {spec.space}" if spec.space else ""
            expected = f"a `Pointer[{spec.dtype.__name__}{space}]`"
        elif (
            isinstance(spec, type)
            and not isinstance(spec, t.NumericMeta)
            and (issubclass(spec, t.Struct) or tree_utils.leaf_entry(spec) is not None)
        ):
            if isinstance(arg, spec):
                return None
            expected = f"a `{spec.__name__}`"
        else:
            return None
        return self.user_error(
            DiagId.ARG_ANNOTATION_MISMATCH,
            num=arg_index + 1,
            arg_name=arg_name,
            expected=expected,
            got=f"a `{type(arg).__name__}`",
        )

    @staticmethod
    def _extract_annotation_markers(spec_ty: Any, arg: Any) -> list[Any]:
        """Extract ``Annotated[...]`` markers from an annotation matching ``arg``.

        Returns the marker list from the first annotation shape that matches
        the runtime value; an empty list when none applies. Handled shapes:
        ``Annotated[T, marker]``, each member of a ``Union``/``T | ...``, and
        ``list[Annotated[T, marker]]``/``tuple[Annotated[T, marker], ...]``
        (one container layer is peeled so per-element markers survive).
        """
        candidate_sub_types = (
            get_args(spec_ty)
            if get_origin(spec_ty) is Union or isinstance(spec_ty, UnionType)
            else (spec_ty,)
        )
        for sub_ty in candidate_sub_types:
            ty, *markers = (
                get_args(sub_ty) if get_origin(sub_ty) is Annotated else (sub_ty,)
            )
            if markers and isinstance(ty, type) and isinstance(arg, ty):
                return markers
            # A ``Pointer[T]`` annotation: the value is adapted (and checked)
            # after the markers are read, so any value matches here.
            if markers and isinstance(ty, t.TypedPointer):
                return markers

            container_origin = get_origin(sub_ty)
            if (
                container_origin in (list, tuple)
                and isinstance(arg, (list, tuple))
                and arg
            ):
                container_args = get_args(sub_ty)
                if container_args and get_origin(container_args[0]) is Annotated:
                    inner_ty, *inner_markers = get_args(container_args[0])
                    if (
                        inner_markers
                        and isinstance(inner_ty, type)
                        and all(isinstance(e, inner_ty) for e in arg)
                    ):
                        return inner_markers
        return []

    @staticmethod
    def _marker_attributes(marker: Any) -> list[ir.DictAttr]:
        """The argument attribute dicts an ``Annotated`` marker contributes."""
        extract = getattr(marker, "__extract_mlir_attributes__", None)
        return list(extract()) if extract is not None else []

    def _check_unsupported_jit_arg(
        self,
        *,
        arg: Any,
        arg_name: str,
        arg_index: int,
        function_name: str,
        is_host: bool,
        jit_arg_type: list[Any],
        jit_exec_arg: list[Any],
        **boundary_options: Any,
    ) -> None:
        """Raise when an argument produced no usable JIT signature: neither a
        known type nor an adapter yielded MLIR types and execution values. The
        ``boundary_options`` are the call's: under ``compile_only`` a type
        with no execution value (``compile()``'s type-only placeholder, a
        dtype class) passes."""
        if jit_arg_type and (
            jit_exec_arg or not is_host or boundary_options.get("compile_only")
        ):
            return

        raise self.user_error(
            DiagId.ARG_UNSUPPORTED_TYPE,
            num=arg_index + 1,
            arg_name=arg_name,
            arg_type=type(arg).__name__,
            function_name=function_name,
        )

    def _flatten_jit_arg(
        self,
        arg: Any,
        arg_name: str,
        arg_index: int,
        function_name: str,
        *,
        is_host: bool,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Flatten one runtime argument into its execution values, SSA types
        and per-value attribute dicts.

        On the host side (``is_host``) the leaves are marshalled into owning
        ``c_void_p`` slots; at the kernel launch boundary they are already
        staged and their ``ir.Value`` s become the launch operands. The
        default knows the registered leaves and the containers of them; this
        is the override point for the values a DSL stages itself (a class of
        its own speaking a protocol, a whole object the DSL hands over as one
        execution value), paired with :meth:`_restore_argument`, which
        rebuilds them from the entry's block arguments, and
        :meth:`_is_meta_at_boundary`, which makes them runtime values.
        """
        values, attrs, treedef = tree_utils.tree_flatten(
            arg, return_ir_values=not is_host, root=arg_name
        )
        if not is_host:
            return list(values), [v.type for v in values], list(attrs)

        exec_args: list[Any] = []
        arg_types: list[Any] = []
        arg_attrs: list[Any] = []
        leaf_values = iter(values)
        for _, leaf in tree_utils.tree_leaves(treedef):
            if leaf.is_none or leaf.is_meta:
                continue
            value = next(leaf_values)
            entry = tree_utils.leaf_entry(type(value))
            if tree_utils.is_staged_leaf(value):
                raise self.user_error(
                    DiagId.ARG_UNSUPPORTED_TYPE,
                    num=arg_index + 1,
                    arg_name=arg_name,
                    arg_type=type(value).__name__,
                    function_name=function_name,
                )
            if entry is None or entry.marshal is None:
                raise self.user_error(
                    DiagId.ARG_UNSUPPORTED_TYPE,
                    num=arg_index + 1,
                    arg_name=arg_name,
                    function_name=function_name,
                    arg_type=type(value).__name__,
                    detail=": the DSL knows the type, but it has no host representation",
                )
            with JitArgAdapterRegistry.using_argument(arg_name, arg_index):
                exec_args.extend(entry.marshal(value))
            # One attribute dict per SSA value of the leaf, from the leaf's
            # ``attributes`` hook when it has one.
            leaf_types = tree_utils.leaf_ir_types(leaf)
            arg_types.extend(leaf_types)
            arg_attrs.extend(tree_utils.leaf_attributes(entry, value, len(leaf_types)))
        return exec_args, arg_types, arg_attrs

    def _generate_jit_func_args(
        self,
        func: Any,
        function_name: str,
        args: tuple[Any, ...] | list[Any],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        *,
        is_host: bool = True,
        **boundary_options: Any,
    ) -> JitFuncArgs:
        """Generate JIT function arguments: the SSA types, attribute dicts and
        execution values of every argument, through :meth:`_is_meta_at_boundary`,
        :meth:`_flatten_jit_arg` and :meth:`_check_unsupported_jit_arg`. The
        ``boundary_options`` are the call's (the core's: ``compile_only``) and
        reach the check."""
        positional_names = []
        kwonly_names = []
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwonly_names.append(name)
            else:
                positional_names.append(name)

        if len(args) != len(positional_names) or len(kwonlyargs) != len(kwonly_names):
            raise DSLRuntimeError(
                f"Input args {len(args)=} and kwonlyargs {len(kwonlyargs)=} must match "
                f"positional params {len(positional_names)=} and keyword-only params "
                f"{len(kwonly_names)=}"
            )

        jit_arg_types: list[Any] = []
        jit_arg_attrs: list[Any] = []
        jit_exec_args: list[Any] = []

        input_args = [*args, *kwonlyargs.values()]
        input_arg_names = [*positional_names, *kwonly_names]
        jit_adapted_args: list[Any] = [None] * len(input_args)
        for i, (arg_name, arg) in enumerate(zip(input_arg_names, input_args)):
            spec_ty = sig.parameters[arg_name].annotation

            # Retrieve markers from the annotated type that matches the arg
            annotation_markers = self._extract_annotation_markers(spec_ty, arg)

            log().debug("Processing [%d] Argument [%s : %s]", i, arg_name, spec_ty)

            cast_ty = (
                get_args(spec_ty)[0] if get_origin(spec_ty) is Annotated else spec_ty
            )
            # ``compile()``'s type-only placeholder: a dtype class for a staged
            # numeric parameter is neither cast nor adapted.
            placeholder = bool(
                boundary_options.get("compile_only")
            ) and self._is_type_placeholder(arg, spec_ty)
            # Implicitly convert into Numeric type if possible
            if (
                not placeholder
                and isinstance(cast_ty, t.NumericMeta)
                and not isinstance(arg, cast_ty)
            ):
                try:
                    arg = t.cast(arg, cast_ty)  # type: ignore[arg-type]
                except DSLBaseError as exc:
                    raise self.user_error(
                        DiagId.ARG_ANNOTATION_MISMATCH,
                        num=i + 1,
                        arg_name=arg_name,
                        expected=f"a `{cast_ty.__name__}`",
                        got=f"a `{type(arg).__name__}`",
                        cause=exc,
                    ) from exc
                jit_adapted_args[i] = arg

            # Adapt the argument before the Meta test, so the test sees the
            # adapted value. Keep the scope active through nested adaptations.
            with JitArgAdapterRegistry.using_scope(self._jit_arg_adapter_scope):
                adapted = None
                if isinstance(spec_ty, t.TypedPointer):
                    # The Pointer[T] annotation adapts a bare address or a buffer.
                    adapted = adapt_pointer_address(
                        arg, spec_ty, arg_name=arg_name, arg_index=i
                    )
                elif not isinstance(arg, ir.Value):
                    adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
                    if adapter is not None:
                        with JitArgAdapterRegistry.using_argument(arg_name, i):
                            adapted = adapter(arg)
                if adapted is not None:
                    arg = adapted
                    jit_adapted_args[i] = arg

                # Type safety check
                if spec_ty is not inspect.Parameter.empty:
                    err = self._validate_arg(arg, i, arg_name, spec_ty)
                    if err is not None:
                        raise err

                # A Python value (``None`` triple) is specialised on by the
                # trace; an MLIR op gets its types, attributes and values.
                jit_exec_arg: list[Any] | None = []
                jit_arg_type: list[Any] | None = []
                jit_arg_attr: list[Any] | None = []
                if placeholder:
                    # The type alone, no execution value.
                    jit_arg_type.append(arg.mlir_type)
                    jit_arg_attr.append(ir.DictAttr.get({}))
                elif self._is_meta_at_boundary(arg, spec_ty, arg_name, i, func):
                    jit_exec_arg = jit_arg_type = jit_arg_attr = None

                if jit_arg_type is not None:
                    if not placeholder:
                        exec_args, arg_types, arg_attrs = self._flatten_jit_arg(
                            arg, arg_name, i, function_name, is_host=is_host
                        )
                        jit_exec_arg.extend(exec_args)  # type: ignore[union-attr]
                        jit_arg_type.extend(arg_types)
                        jit_arg_attr.extend(arg_attrs)  # type: ignore[union-attr]

                    self._check_unsupported_jit_arg(
                        arg=arg,
                        arg_name=arg_name,
                        arg_index=i,
                        function_name=function_name,
                        is_host=is_host,
                        jit_arg_type=jit_arg_type,
                        jit_exec_arg=jit_exec_arg,  # type: ignore[arg-type]
                        **boundary_options,
                    )

            if jit_arg_type is not None:
                # Merge attributes from annotated markers (`__extract_mlir_attributes__`)
                # into every element of jit_arg_attr for this argument.
                if annotation_markers and jit_arg_attr:
                    extra = {
                        na.name: na.attr
                        for marker in annotation_markers
                        for attr_dict in self._marker_attributes(marker)
                        for na in attr_dict
                    }
                    if extra:
                        jit_arg_attr = [
                            ir.DictAttr.get({na.name: na.attr for na in d} | extra)
                            for d in jit_arg_attr
                        ]

                jit_exec_args.extend(jit_exec_arg)  # type: ignore[arg-type]
                jit_arg_types.extend(jit_arg_type)
                jit_arg_attrs.extend(jit_arg_attr)  # type: ignore[arg-type]

        return JitFuncArgs(
            jit_exec_args, jit_arg_types, jit_arg_attrs, jit_adapted_args
        )

    def generate_mlir_function_types(
        self,
        func: Any,
        function_name: str,
        args: tuple[Any, ...] | list[Any],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        **boundary_options: Any,
    ) -> JitFuncArgs:
        """Convert input arguments to the MLIR function signature and the
        marshalled execution arguments. The ``boundary_options`` reach
        :meth:`_generate_jit_func_args` and :meth:`_check_unsupported_jit_arg`
        (the core's: ``compile_only``, under which a type-only placeholder has
        a type and no execution value; a DSL adds its own)."""

        result = self._generate_jit_func_args(
            func, function_name, args, kwonlyargs, sig, is_host=True, **boundary_options
        )

        if not boundary_options.get("compile_only") and len(result.values) != len(
            result.types
        ):
            raise DSLRuntimeError(
                "expects the same number of arguments and function parameters",
                context={"values": len(result.values), "types": len(result.types)},
            )

        return result

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
        """The MLIR location of ``location`` (the decorator's by default): a
        plain file location, so that the NameLoc wrapping of traceback
        locations (``enter_traceback_locations``) carries only the names a DSL
        puts in ``ir.Location.current`` itself."""
        if location is None:
            location = self.decorator_location
        if location is None:
            return ir.Location.unknown()
        return ir.Location.file(location.filename, location.lineno, location.col_offset)

    def _enter_loc_tracebacks(self) -> Any:
        """Turn on MLIR's traceback locations for the trace when ``debuginfo``
        is set (the user's line) or ``<PREFIX>_LOC_TRACEBACKS=N`` asks for a
        call chain of N frames; ``debug`` keeps the DSL's own frames. Returns
        the context manager to exit, or None."""
        depth = self.envar.loc_tracebacks
        if depth <= 0 and self.envar.debuginfo:
            depth = 1
        return enter_traceback_locations(
            depth, include_dsl_frames=bool(self.envar.debug)
        )

    # ===------------------------------------------------------------------===
    # Compile, hash, dump
    # ===------------------------------------------------------------------===

    def _compile_options(self) -> dict[str, Any]:
        """The keywords :meth:`prepare_compile` fixes on a compile step for
        its :meth:`compile_and_jit` or :meth:`compile_module`: the compiler
        plugin's ``compile`` keywords (the verifier, IR printing into a tree,
        pass profiling, session options of its own), evaluated when the step
        is prepared so the step never reads the DSL's settings later. They
        win over what :meth:`_compile_keywords` passes on its own. The
        default adds nothing."""
        return {}

    def _compile_keywords(self, options: dict[str, Any]) -> dict[str, Any]:
        """The keywords of one compile through the compiler plugin: the pass
        profiling and debug-info settings, the remark options, then
        ``options`` (a step's ``compile_options``), which win."""
        return {
            "enable_pass_profiling": self.envar.enable_pass_profiling,
            "enable_debug_info": self.envar.debuginfo,
            **self._remark_options(),
            **options,
        }

    def compile_and_jit(
        self,
        module: ir.Module,
        pipeline: str,
        shared_libs: list[str],
        function_name: str = "",
        **options: Any,
    ) -> Any:
        """
        Compile and JIT an MLIR module through the compiler plugin.

        :param options: Further keywords of the plugin's ``compile_and_jit``
            (a step's ``compile_options``), over the DSL's own
        :return: The ``ExecutionEngine`` holding the compiled module.
        """
        # This method is also a direct entry point (not only reached through
        # generate_mlir), so it owns a compile boundary of its own; when
        # nested inside generate_mlir the depth counter folds it away.
        profiler.begin_compile()
        try:
            return self._compiler.compile_and_jit(
                module,
                pipeline,
                shared_libs=shared_libs,
                after_lowering=partial(
                    self._after_lowering, function_name=function_name
                ),
                **self._compile_keywords(options),
            )
        except DSLBaseError:
            raise
        except Exception as e:
            raise DSLRuntimeError(
                "compilation failed", context={"function": function_name}, cause=e
            ) from e
        finally:
            profiler.finish_compile()

    def compile_module(
        self,
        module: ir.Module,
        pipeline: str,
        function_name: str = "",
        **options: Any,
    ) -> ir.Module:
        """Lower ``module`` through the compiler plugin's pipeline without
        building an engine (a ``CompileStep(build_engine=False)``); the
        adapters' ``after_lowering`` run on the lowered module as in
        :meth:`compile_and_jit`.

        :param options: Further keywords of the plugin's ``compile``
        :return: The lowered module (``module`` itself with the shipped
            plugin, which lowers in place)
        """
        profiler.begin_compile()
        try:
            lowered = self._compiler.compile(
                module, pipeline, **self._compile_keywords(options)
            )
            if lowered is None:
                lowered = module
            self._after_lowering(lowered, function_name)
            return lowered
        except DSLBaseError:
            raise
        except Exception as e:
            raise DSLRuntimeError(
                "compilation failed", context={"function": function_name}, cause=e
            ) from e
        finally:
            profiler.finish_compile()

    def _after_lowering(self, module: ir.Module, function_name: str) -> None:
        """Run every adapter's ``after_lowering`` on the lowered ``module`` of
        the entry ``function_name``."""
        for plugin in self.plugins.adapters:
            plugin.after_lowering(self, module, function_name)

    def jit_lowered_module(
        self, module: ir.Module, shared_libs: list[str], function_name: str = ""
    ) -> Any:
        """Build the ``ExecutionEngine`` of an already lowered ``module`` (a file
        cache hit): the pass pipeline is skipped."""
        profiler.begin_compile()
        try:
            return self._compiler.jit(module, shared_libs=shared_libs)
        except DSLBaseError:
            raise
        except Exception as e:
            raise DSLRuntimeError(
                "compilation failed", context={"function": function_name}, cause=e
            ) from e
        finally:
            profiler.finish_compile()

    def get_shared_libs(self, extra_link_libs: tuple[str, ...] = ()) -> list[str]:
        """The runtime libraries handed to the ``ExecutionEngine``: the
        ``<PREFIX>_LIBS`` paths, every plugin's ``shared_libs()`` and
        ``extra_link_libs``, canonical and without duplicates."""
        shared_libs: list[str] = []
        support_libs = self.envar.shared_libs
        if support_libs is not None:
            if os.name == "nt":
                # Accept POSIX-style ':' separators too (lit RUN lines share
                # them across platforms), but a ':' followed by a path
                # separator is a drive colon, not a separator.
                shared_libs.extend(
                    p for p in re.split(r";|:(?![\\/])", support_libs) if p
                )
            else:
                shared_libs.extend(support_libs.split(os.pathsep))

        shared_libs.extend(self.shared_libs())
        for plugin in self.plugins:
            shared_libs.extend(plugin.shared_libs())
        shared_libs.extend(extra_link_libs)
        return list(_normalize_shared_library_paths(shared_libs))

    def get_version(self) -> "hashlib._Hash":
        """A hash of the toolchain identity (the MLIR binaries in use) and the
        framework identity (the DSL package's own sources)."""
        version_hash = hashlib.sha256()
        version_hash.update(repr(toolchain_identity()).encode())
        version_hash.update(framework_identity().encode())
        return version_hash

    def get_module_hash(
        self,
        module: ir.Module,
        function_name: str,
        *,
        pipeline: str = "",
        extra_link_libs: tuple[str, ...] = (),
    ) -> str:
        """The compile cache key of ``module``: a hash of its bytecode, the
        ``affects_compile`` environment settings, the DSL's own
        :meth:`_cache_key_extra`, the pipeline, the extra link libraries and
        the toolchain identity (:meth:`get_version`)."""
        s = io.BytesIO()
        module.operation.write_bytecode(s)
        s.write(self.envar.cache_key_str().encode())
        s.write(self._cache_key_extra())
        s.write(b"\0pipeline\0")
        s.write(pipeline.encode())
        for lib in extra_link_libs:
            s.write(b"\0extra_link_lib\0")
            s.write(os.fsencode(lib))
        hash_obj = self.get_version().copy()
        hash_obj.update(s.getvalue())
        module_hash = hash_obj.hexdigest()

        # Hex-encoding the whole bytecode buffer is per-compile waste unless
        # DEBUG logging is actually on.
        if log().isEnabledFor(logging.DEBUG):
            log().debug("Bytecode=[%s]", s.getvalue().hex())
        log().info(
            "Function=[%s] Computed module_hash=[%s]", function_name, module_hash
        )
        return module_hash

    def _cache_key_extra(self) -> bytes:
        """Bytes :meth:`get_module_hash` adds to the compile cache key of
        every module, after the environment's ``affects_compile`` settings:
        whatever else the compiled artifact depends on in this DSL (compile
        options of its own, the options of its pipeline). The default adds
        nothing."""
        return b""

    def _inspection_pipeline(self, passes: str) -> str:
        """Frame ``passes`` for IR-inspection dumps."""
        passes = passes.strip()
        if passes.startswith("builtin.module("):
            return passes
        return f"builtin.module({passes})"

    def _save_ir(self, module: ir.Module, label: str) -> str:
        """Write ``module`` as text under ``<PREFIX>_CACHE_DIR`` (atomically,
        through :func:`save_ir`) and return the path."""
        output_dir = get_default_generated_ir_path(self.name)
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = save_ir(
                self.name,
                module,
                label,
                output_dir,
                as_text=True,
                enable_debug_info=self.envar.debuginfo,
            )
        except OSError as e:
            raise DSLRuntimeError(
                "failed to save the generated IR", context={"dir": output_dir}, cause=e
            ) from e
        log().info("Saved IR [%s] to [%s]", label, path)
        return str(path)

    def _keep_ir_stages(self) -> tuple[str, ...]:
        """The IR dumps ``<PREFIX>_KEEP_IR`` asks :meth:`build_module` for:
        ``1``/``true`` is the raw module alone (``("raw",)``), otherwise the
        comma-separated stages ``raw``, ``clean`` (canonicalize and cse),
        ``after-<pass>`` and ``after-passes`` (the DSL's pipeline), every
        stage but the raw one taken on a clone. Empty when the setting is
        unset; an unknown stage is ``CONFIG_INVALID``."""
        raw = (self.envar.keep_ir or "").strip()
        lowered = raw.lower()
        if not lowered or lowered in ("0", "false", "no", "off"):
            return ()
        if lowered in ("1", "true", "yes", "on"):
            return ("raw",)
        stages = tuple(token.strip() for token in raw.split(",") if token.strip())
        for stage in stages:
            if stage not in ("raw", "clean", "after-passes") and not (
                stage.startswith("after-") and len(stage) > len("after-")
            ):
                raise DSLUserCodeValueError(
                    DiagId.CONFIG_INVALID,
                    var=f"{self.name}_KEEP_IR",
                    detail=f"`{stage}` is not a stage; expected `1`, or a "
                    "comma-separated list of `raw`, `clean`, `after-<pass>` and "
                    "`after-passes`",
                )
        return stages

    def _compile_clone_and_save(
        self,
        module: ir.Module,
        pipeline: str,
        label: str,
        *,
        enable_verifier: bool = True,
    ) -> str:
        """Run ``pipeline`` on a clone of ``module`` (a bytecode round trip, so
        the module itself, and what is cached under its hash, stays untouched)
        through the compiler plugin and save the result as the dump ``label``
        (:meth:`_save_ir`); returns the path."""
        buffer = io.BytesIO()
        module.operation.write_bytecode(buffer)
        clone = ir.Module.parse(buffer.getvalue(), context=module.context)
        self._compiler.compile(clone, pipeline, enable_verifier=enable_verifier)
        return self._save_ir(clone, label)

    def build_module(self, module: ir.Module, function_name: str) -> ir.Module:
        """
        Build the MLIR module: write the IR dumps the settings ask for, print
        the IR, verify the module (:meth:`_verify_module`, last) and return it
        """

        # The KEEP_IR dumps, one file per stage; the raw module (before any
        # pass) under the function's own name, the other stages on clones.
        self.dump_mlir_paths = {}
        for stage in self._keep_ir_stages():
            if stage == "raw":
                path = self._save_ir(module, function_name)
                self.dump_mlir_path = self.dump_mlir_raw_path = path
            elif stage == "clean":
                path = self._compile_clone_and_save(
                    module,
                    self._inspection_pipeline("canonicalize,cse"),
                    f"{function_name}_clean",
                )
            elif stage == "after-passes":
                path = self._compile_clone_and_save(
                    module, self._get_pipeline(None), f"{function_name}_after_passes"
                )
            else:
                pass_name = stage[len("after-") :]
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", pass_name).strip("_")
                path = self._compile_clone_and_save(
                    module,
                    self._inspection_pipeline(pass_name),
                    f"{function_name}_after_{safe_name}",
                )
            self.dump_mlir_paths[stage] = path

        if self.envar.print_ir:
            print("\n//===--- ------ Generated IR ------ ---====\n")
            module.operation.print(enable_debug_info=self.envar.debuginfo)
            print("\n//===--- --- End of Generated IR -- ---====\n")

        # Print IR after applying the passes, on a clone
        if self.envar.print_ir_after_passes:
            self._compiler.print_ir_after_passes(
                module,
                self.envar.print_ir_after_passes,
                enable_debug_info=self.envar.debuginfo,
            )

        # Save IR after applying the passes
        if self.envar.keep_ir_after_passes:
            pipeline = self._inspection_pipeline(self.envar.keep_ir_after_passes)
            self._compiler.compile(module, pipeline)
            self._save_ir(module, f"{function_name}_after_pass")

        self._verify_module(module)
        return module

    def _verify_module(self, module: ir.Module) -> None:
        """Verify the finished ``module``, last in :meth:`build_module`: the
        MLIR verifier, a failure being an ``IR verification failed``
        ``DSLRuntimeError``. A DSL with invariants of its own on the traced
        module (an attribute it forbids, an op its lowering cannot take)
        checks them here, around ``super()``'s."""
        try:
            module.operation.verify()
        except Exception as e:
            raise DSLRuntimeError("IR verification failed", cause=e) from e

    # ===------------------------------------------------------------------===
    # Return values
    # ===------------------------------------------------------------------===

    def _return_values(
        self,
        entry: DecoratorPlugin,
        result: Any,
        sig: inspect.Signature,
        loc: Any,
        *,
        func: Callable[..., Any],
    ) -> tuple[list[ir.Value], _ResultSpec | None]:
        """Turn the traced return value into what the entry returns.
        Numeric leaves and frozen records/tuples/``@struct``s of them are
        accepted; the decorator plugin's ``pack_results`` decides how the
        leaves travel (the shipped ``func.Jit`` packs several into one struct).
        ``func`` is the traced function, for a DSL's own return policy or
        diagnostics (the core reads only the return annotation of ``sig``).
        """
        if result is None:
            if sig.return_annotation not in (inspect.Signature.empty, None):
                raise self.user_error(
                    DiagId.TYPE_RETURN_MISMATCH,
                    got="`None`",
                    detail=" while it declares a return type",
                )
            return [], None
        if isinstance(result, (bool, int, float)):
            result = t.Numeric._from_python_value(result)

        _, _, treedef = tree_utils.tree_flatten(result, return_ir_values=False)
        leaves = [
            leaf
            for _, leaf in tree_utils.tree_leaves(treedef)
            if not leaf.is_none and not leaf.is_meta
        ]
        for leaf in leaves:
            cls = leaf.node_metadata.cls
            if not issubclass(cls, t.Numeric):
                raise self.user_error(
                    DiagId.TYPE_RETURN_MISMATCH,
                    got=f"a `{cls.__name__}`",
                    detail=(
                        " (a `Pointer` result is memory: write through the pointer instead)"
                        if issubclass(cls, t.Pointer)
                        else ""
                    ),
                )
        if not leaves:
            raise self.user_error(
                DiagId.TYPE_RETURN_MISMATCH, got=f"a `{type(result).__name__}`"
            )
        values, _, _ = tree_utils.tree_flatten(result, return_ir_values=True)
        ret_values, slot = entry.pack_results(
            values, [leaf.prototype for leaf in leaves], loc=loc
        )
        return ret_values, _ResultSpec(treedef, slot, entry)

    def _result_from_ctypes(self, spec: _ResultSpec, raw: Any) -> Any:
        """Rebuild the Python return value from the filled result slot."""
        leaves = [
            leaf
            for _, leaf in tree_utils.tree_leaves(spec.treedef)
            if not leaf.is_none and not leaf.is_meta
        ]
        values = iter(
            spec.entry.unpack_result(
                spec.slot, raw, [leaf.prototype for leaf in leaves]
            )
        )
        return self._restore_tree(spec.treedef, lambda leaf: next(values))

    # ===------------------------------------------------------------------===
    # IR Tracing
    # ===------------------------------------------------------------------===

    def generate_original_ir(
        self,
        entry: DecoratorPlugin,
        func: Callable[..., Any],
        function_name: str,
        func_types: list[Any],
        arg_attrs: list[Any],
        container_attrs: dict[str, Any],
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        location: DSLLocation | None = None,
        *,
        no_cache: bool = False,
        pipeline: str = "",
        extra_link_libs: tuple[str, ...] = (),
        trace_hooks: Sequence[Callable[..., Any]] = (),
    ) -> tuple[ir.Module, str | None, Any, _ResultSpec | None]:
        """Trace ``func`` into the op the decorator plugin ``entry`` builds, in
        a fresh module. This runs with the DSL active (``run``), so a decorated
        function called inside the body is launched into this trace, not
        called. ``trace_hooks`` are the call's ``trace_hooks=`` callables,
        ``hook(dsl, module, function_name)``, run on the finished module
        after the adapters added to it and before it is hashed.

        :return: The verified module, its hash (None under ``no_cache``), the
            trace's Python result and the result slot description
        """

        def build_ir_module() -> tuple[ir.Module, Any, _ResultSpec | None]:
            loc = self.get_ir_location(location)
            module = ir.Module.create(loc=loc)

            self._deferred_calls = []
            try:
                with ir.InsertionPoint(module.body):
                    return build_in_module(module, loc)
            finally:
                self._deferred_calls = None

        def build_in_module(
            module: ir.Module, loc: Any
        ) -> tuple[ir.Module, Any, _ResultSpec | None]:
            if True:  # the body of the trace, inside the module's insertion point
                # A plugin with per-trace state (a kernel container, the
                # launches it expects) sets it up before the entry is built.
                for plugin in self.plugins.decorators:
                    plugin.before_trace(self, module, loc=loc, attrs=container_attrs)

                # The entry is the decorator plugin's (``func.Jit``: a
                # ``func.func`` with the C interface); its result types are set
                # after the trace.
                func_op, entry_block, result = self.trace_body(
                    entry,
                    function_name,
                    func,
                    args,
                    kwonlyargs,
                    sig,
                    func_types,
                    arg_attrs,
                    loc=loc,
                )
                with ir.InsertionPoint(entry_block):
                    ret_values, result_spec = self._return_values(
                        entry, result, sig, loc, func=func
                    )
                    entry.generate_return(func_op, ret_values, loc=loc)
                # A prepared call the body never issued is the user's mistake.
                self._check_deferred_calls()
                for plugin in self.plugins.decorators:
                    plugin.after_trace(self, module, function_name)

            return module, result, result_spec

        # Build IR module, then finalize it (hash, verify). The whole closure
        # is one unit so the profiler's build phase covers the finalize steps
        # too, not just the trace.
        def build_and_finalize() -> (
            tuple[ir.Module, str | None, Any, _ResultSpec | None]
        ):
            module, result, result_spec = self._maybe_profile(build_ir_module)()
            # Plugins add to the module before the hash, so what they add is
            # part of the cached artifact.
            for plugin in self.plugins.adapters:
                plugin.attach_to_module(
                    self, module, function_name, sig, args, kwonlyargs
                )
            # The call's hooks see the finished module last, outside the
            # module's insertion point; what they add is hashed too.
            for hook in trace_hooks:
                hook(self, module, function_name)
            module_hash = None
            if not no_cache:
                module_hash = self.get_module_hash(
                    module,
                    function_name,
                    pipeline=pipeline,
                    extra_link_libs=extra_link_libs,
                )
            return (
                self.build_module(module, function_name),
                module_hash,
                result,
                result_spec,
            )

        build_and_finalize = profiler.profile_build(build_and_finalize)
        return build_and_finalize()

    def _maybe_profile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``fn`` with the JIT timer when profiling is enabled.

        ``self.profiler`` only exists when ``jit_time_profiling`` is set (see
        ``__init__``), so it is referenced only on that branch.
        """
        return self.profiler(fn) if self.envar.jit_time_profiling else fn

    def prepare_compile(
        self,
        module: ir.Module,
        module_hash: str | None,
        function_name: str,
        pipeline: str,
        sig: inspect.Signature,
        no_cache: bool,
        *,
        result_spec: _ResultSpec | None = None,
        extra_link_libs: tuple[str, ...] = (),
        func: Callable[..., Any] | None = None,
        func_type: Any = None,
        build_engine: bool = True,
        post_processors: Sequence[Callable[[Any], None]] = (),
    ) -> CompileStep:
        """The compile step of ``module`` (:class:`CompileStep`), ready to run;
        :meth:`compile_and_cache` is ``prepare_compile(...).run()``. The DSL's
        :meth:`_compile_options` are evaluated here, under the caller's
        scope, so a step run later or on another thread compiles with the
        settings of the call that prepared it.

        :param module_hash: The compile cache key, None under ``no_cache``
        :param func: The decorated function whose lifetime the cache entry
            follows
        :param func_type: The compiled-function class the plugin's ``load``
            builds, its default when None
        :param build_engine: False lowers without an engine (see
            :class:`CompileStep`)
        :param post_processors: Callables run on the compiled function after
            the plugins wrapped it, once per compile
        """
        return self._compile_step_class(
            dsl=self,
            module=module,
            module_hash=module_hash,
            function_name=function_name,
            pipeline=pipeline,
            sig=sig,
            no_cache=no_cache,
            result_spec=result_spec,
            extra_link_libs=tuple(extra_link_libs),
            func=func,
            func_type=func_type,
            build_engine=build_engine,
            compile_options=dict(self._compile_options()),
            post_processors=list(post_processors),
        )

    def compile_and_cache(
        self,
        module: ir.Module,
        module_hash: str | None,
        function_name: str,
        pipeline: str,
        sig: inspect.Signature,
        no_cache: bool,
        *,
        result_spec: _ResultSpec | None = None,
        extra_link_libs: tuple[str, ...] = (),
        func: Callable[..., Any] | None = None,
    ) -> Any:
        """Compile ``module`` and cache the callable the compiler plugin loads:
        the step :meth:`prepare_compile` builds, run now.

        The lowered module goes into an ``ExecutionEngine`` over
        ``get_shared_libs``; the entry is the packed ``_mlir_<name>`` wrapper.

        With file caching on (``DISABLE_FILE_CACHING`` unset) the lowered module
        is looked up under ``module_hash`` in the cache directory first and, on a
        miss, dumped there after the pipeline ran (bytecode + CRC32), so a later
        process skips the pass pipeline and only builds the engine.
        """
        return self.prepare_compile(
            module,
            module_hash,
            function_name,
            pipeline,
            sig,
            no_cache,
            result_spec=result_spec,
            extra_link_libs=extra_link_libs,
            func=func,
        ).run()

    def _load_options(self, step: CompileStep) -> dict[str, Any]:
        """Further keywords of the compiler plugin's ``load`` for ``step``.
        ``CompileStep.build`` passes the profiling flag, the result slot,
        ``func_type`` and the ``<PREFIX>_LIBS`` name itself; what a DSL
        returns here wins over them. The default adds nothing."""
        return {}

    def post_compilation_cleanup(self) -> None:
        """Clean up some internal state after one compilation is completed."""
        self.decorator_location = None

    def generate_mlir(
        self,
        entry: DecoratorPlugin,
        func: Callable[..., Any],
        function_name: str,
        container_attrs: dict[str, Any],
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        pipeline: str | None,
        no_cache: bool,
        compile_only: bool,
        location: DSLLocation | None = None,
        extra_link_libs: tuple[str, ...] = (),
        trace_hooks: Sequence[Callable[..., Any]] = (),
        *,
        setup: "BaseDSL._CompilationSetup | None" = None,
    ) -> Any:
        """Trace ``func`` into the op the decorator plugin ``entry`` builds in
        an MLIR module, compile it through the compiler plugin (or take the
        cached function) and run it.

        The call's hooks run in this order: :meth:`_configure_compilation` is
        entered around the trace and the compile, :meth:`after_trace_result`
        sees the finished trace, :meth:`emit_artifact` may end the call with
        a value of its own or a :class:`PendingCompile`, and
        :meth:`invoke_compiled` runs the compiled function and builds the
        result through :meth:`_call_result`. ``setup`` is the call's
        ``_CompilationSetup`` (:meth:`run` passes it); a direct caller may
        leave it out, and one without options is built from the arguments.

        :return: The trace's Python result under ``<PREFIX>_DRYRUN`` or on a
            DSL without a compiler, the compiled function under
            ``compile_only``, what :meth:`emit_artifact` returned when it
            returned something, else the call's result
        """
        if setup is None:
            setup = self._CompilationSetup(
                function_name=function_name,
                pipeline=pipeline,
                container_attrs=container_attrs,
                no_cache=no_cache,
                extra_link_libs=tuple(extra_link_libs),
                compile_only=compile_only,
                canonicalized_args=args,
                canonicalized_kwargs=kwonlyargs,
                sig=sig,
                location=location,
                trace_hooks=tuple(trace_hooks),
            )
        # The remark session owns the context's remark engine for the whole
        # call, so trace-time remarks and the passes' remarks share one stream.
        # The stack holds the context, the location and the session, so a
        # deferred step (``PendingCompile``) can take them over.
        with ExitStack() as scope:
            ctx = scope.enter_context(ir.Context())
            scope.enter_context(self.get_ir_location(location))
            remark_session = scope.enter_context(self._remark_session(ctx))
            # Each MLIR context keeps a thread pool alive; cached compilations
            # keep their context, so threading is disabled to bound the count.
            ctx.enable_multithreading(False)
            self.register_dialects(ctx)
            for plugin in self.plugins:
                plugin.register_dialects(ctx)
            self.collected_remarks = remark_session.remarks
            # Optional: capture full Python call stacks on every MLIR op.
            loc_tracebacks = self._enter_loc_tracebacks()

            profiler.begin_compile()
            try:
                with self._configure_compilation(setup):
                    trace = self._trace_call(
                        entry,
                        func,
                        function_name,
                        container_attrs,
                        args,
                        kwonlyargs,
                        sig,
                        pipeline,
                        location=location,
                        no_cache=no_cache,
                        extra_link_libs=extra_link_libs,
                        trace_hooks=trace_hooks,
                        compile_only=compile_only,
                    )
                    self.after_trace_result(trace, setup)

                    # A dry run generates the IR and stops; so does a call on a
                    # DSL without a compiler plugin (trace-only: the IR is the
                    # product). An explicit ``compile()`` on such a DSL falls
                    # through to the missing-compiler error below.
                    if self.envar.dryrun or (
                        self.plugins.compiler is None and not compile_only
                    ):
                        return trace.result

                    # A DSL may end the call here with an artifact of its own,
                    # or hand the compile back as a step to run once the call
                    # scope is left; the step takes the context over.
                    artifact = self.emit_artifact(trace, setup)
                    if artifact is not None:
                        if (
                            isinstance(artifact, PendingCompile)
                            and artifact.scope is None
                        ):
                            artifact.scope = scope.pop_all()
                        return artifact

                    module_hash = trace.module_hash
                    cached_jit_func = (
                        None if no_cache else self.jit_cache.get(module_hash)
                    )

                    if (
                        no_cache
                        or cached_jit_func is None
                        or cached_jit_func.capi_func is None
                    ):
                        # no cache or cache miss, do ir generation/compilation/jit engine
                        jit_function = self.compile_and_cache(
                            trace.module,
                            module_hash,
                            function_name,
                            trace.pipeline,
                            sig,
                            no_cache,
                            result_spec=trace.result_spec,
                            extra_link_libs=extra_link_libs,
                            func=func,
                        )
                    else:
                        self.cache_hits += 1
                        log().info(
                            "JIT cache hit IN-MEMORY function=[%s] module_hash=[%s]",
                            function_name,
                            module_hash,
                        )
                        jit_function = cached_jit_func

            finally:
                if loc_tracebacks is not None:
                    try:
                        loc_tracebacks.__exit__(None, None, None)
                    except Exception:
                        pass
                self.post_compilation_cleanup()
                # Diagnostics last: a profiler failure must not skip the cleanup.
                profiler.finish_compile()

        # If compile_only is set, bypass execution return the jit_executor directly
        if compile_only:
            # The returned function is specialized on this call's Meta values;
            # a later call with other ones is a diagnostic, not a silent reuse.
            execution_args = getattr(jit_function, "execution_args", None)
            if execution_args is not None:
                execution_args.set_meta_values(
                    {
                        param.name: arg
                        for param, arg in zip(sig.parameters.values(), args)
                        if self._is_meta_argument(arg, param.annotation)
                    }
                )
                # The host shape of every argument, adapted as the trace saw it.
                execution_args.set_shapes(
                    {
                        param.name: self._compiled_argument_shape(arg, param.annotation)
                        for param, arg in zip(sig.parameters.values(), trace.trace_args)
                    }
                )
            return jit_function

        return self.invoke_compiled(jit_function, trace)

    def _trace_call(
        self,
        entry: DecoratorPlugin,
        func: Callable[..., Any],
        function_name: str,
        container_attrs: dict[str, Any],
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        pipeline: str | None,
        *,
        location: DSLLocation | None,
        no_cache: bool,
        extra_link_libs: tuple[str, ...],
        trace_hooks: Sequence[Callable[..., Any]],
        compile_only: bool = False,
    ) -> _TraceResult:
        """The trace half of :meth:`generate_mlir`, inside the call's context:
        translate the arguments (``compile_only`` reaches the boundary as an
        option), trace the body into the entry's module, check the arguments
        and describe what the trace produced."""
        # Convert input arguments to MLIR arguments
        mlir_func_args = self.generate_mlir_function_types(
            func, function_name, args, kwonlyargs, sig, compile_only=compile_only
        )
        exe_args = mlir_func_args.values
        adapted_args = mlir_func_args.adapted_python_args
        # The body is traced with the adapted values, so an adapter runs once.
        n_positional = len(args)
        trace_args = tuple(
            original if adapted is None else adapted
            for original, adapted in zip(args, adapted_args[:n_positional])
        )
        trace_kwargs = {
            name: (original if adapted is None else adapted)
            for (name, original), adapted in zip(
                kwonlyargs.items(), adapted_args[n_positional:]
            )
        }
        pipeline = self._get_pipeline(pipeline)

        # Generate original ir module and its hash value.
        module, module_hash, result, result_spec = self.generate_original_ir(
            entry,
            func,
            function_name,
            mlir_func_args.types,
            mlir_func_args.attributes,
            container_attrs,
            trace_args,
            trace_kwargs,
            sig,
            location=location,
            no_cache=no_cache,
            pipeline=pipeline,
            extra_link_libs=extra_link_libs,
            trace_hooks=trace_hooks,
        )
        for plugin in self.plugins.decorators:
            plugin.check_arguments(
                self, sig, args, kwonlyargs, adapted_args, function_name
            )
        return _TraceResult(
            module=module,
            module_hash=module_hash,
            result=result,
            result_spec=result_spec,
            function_name=function_name,
            pipeline=pipeline,
            sig=sig,
            args=args,
            kwonlyargs=kwonlyargs,
            trace_args=trace_args,
            exe_args=exe_args,
            extra_link_libs=tuple(extra_link_libs),
        )

    def invoke_compiled(self, jit_function: Any, trace: _TraceResult) -> Any:
        """Run the compiled ``jit_function`` for the call ``trace`` describes
        and build the call's result (:meth:`_call_result`). A function that
        prefers the Python arguments is called with them; the packed entry
        takes the marshalled ``exe_args``. A DSL's deferred step (:class:`PendingCompile`) calls
        this once its compile is done."""
        if getattr(jit_function, "prefers_python_args", False):
            return jit_function(*trace.args, **trace.kwonlyargs)
        raw = jit_function.run_compiled_program(trace.exe_args)
        return self._call_result(jit_function, raw, trace)

    def _call_result(self, jit_function: Any, raw: Any, trace: _TraceResult) -> Any:
        """The call's result from ``raw``, what the packed entry stored in the
        result slot: rebuilt through the slot description the compiled
        function carries (``result_spec``), None for a function without one.
        A DSL whose calls return something else (a status word it checks,
        the traced value itself) overrides this."""
        result_spec = getattr(jit_function, "result_spec", None)
        if result_spec is None:
            return None
        return self._result_from_ctypes(result_spec, raw)

    # ===------------------------------------------------------------------===
    # Preprocessor
    # ===------------------------------------------------------------------===

    @staticmethod
    def _inject_closure_cells(
        original_function: Any, exec_globals: dict[str, Any]
    ) -> None:
        """Inject closure cell values into *exec_globals*.

        When a decorated function captures variables from an enclosing scope,
        those names are absent from ``__globals__``.  The AST preprocessor
        re-parses the source and ``exec()``s it, which requires those names
        to be resolvable in *exec_globals*.
        """
        if original_function.__closure__:
            for name, cell in zip(
                original_function.__code__.co_freevars,
                original_function.__closure__,
            ):
                try:
                    exec_globals[name] = cell.cell_contents
                except ValueError:
                    # Cell may be empty if the variable was never assigned
                    # in the enclosing scope; safe to skip.
                    pass

    def run_preprocessor(self, original_function: Any) -> Any:
        """Rewrite ``original_function`` through the AST preprocessor.

        :return: The rewritten function whose code object replaces the
            original's, or a false value when the preprocessor left it alone
        """
        # Preprocessing runs before jit_wrapper enters its call-time context.
        with active_dsl(self):
            return self._run_preprocessor_impl(original_function)

    def _run_preprocessor_impl(self, original_function: Any) -> Any:
        function_name = original_function.__name__
        self.traced_function = original_function
        log().info("Started preprocessing [%s]", function_name)
        exec_globals: dict[str, Any] = {}
        if original_function.__globals__ is not None:
            exec_globals.update(original_function.__globals__)
        self._inject_closure_cells(original_function, exec_globals)
        with self.preprocessor.get_session() as preprocessor_session:
            transformed_ast = preprocessor_session.transform(
                original_function, exec_globals
            )
            if original_function not in preprocessor_session.processed_functions:
                # The function carries no decorator the rewrite recognises:
                # nothing was rewritten, so no code object is swapped in.
                log().info("[%s] left as it is: no DSL decorator", function_name)
                original_function._preprocessed = True
                return None
            if self.envar.debug:
                log().info(
                    "# Printing unparsed AST after preprocess of func=`%s`",
                    function_name,
                )
                self.preprocessor.print_ast(transformed_ast)
            file_name = inspect.getsourcefile(original_function)
            try:
                code_object = compile(
                    transformed_ast, filename=file_name or "<unknown>", mode="exec"
                )
            except (SyntaxError, ValueError, TypeError) as e:
                raise DSLRuntimeError(
                    f"the preprocessed source of `{function_name}` does not compile",
                    cause=e,
                ) from e

            original_function._preprocessed = True

            return preprocessor_session.exec(
                original_function.__name__, original_function, code_object, exec_globals
            )

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
            raise self.user_error(
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
                raise self.user_error(
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
            raise self.user_error(
                DiagId.SCOPE_UNBOUND_NAME_IN_TRACE,
                var=getattr(e, "name", None) or "<annotation>",
                function_name=getattr(func, "__name__", "<function>"),
                cause=e,
            ) from e

    @staticmethod
    def _expand_varargs_varkw(
        canonicalized_args: tuple,
        canonicalized_kwargs: dict,
        signature: inspect.Signature,
    ) -> inspect.Signature:
        """
        Expands ``*args`` and ``**kwargs`` into concrete named parameters in the function's signature.

        Extra positional arguments get synthetic names ``_vararg_<i>``; extra
        keyword arguments become keyword-only parameters, so downstream
        components expecting fixed-arity signatures can function as usual.
        Order: positional -> *args -> keyword-only/default -> **kwargs.
        """
        new_params = []
        visited_kwonly_args = set()
        for idx, (name, param) in enumerate(signature.parameters.items()):
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                new_params.append(param)
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                for vararg_idx in range(idx, len(canonicalized_args)):
                    new_params.append(
                        inspect.Parameter(
                            name=f"_vararg_{vararg_idx}",
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    )
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                new_params.append(param)
                visited_kwonly_args.add(name)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                for kwarg_name in canonicalized_kwargs:
                    if kwarg_name not in visited_kwonly_args:
                        new_params.append(
                            inspect.Parameter(
                                name=kwarg_name, kind=inspect.Parameter.KEYWORD_ONLY
                            )
                        )
            else:
                raise DSLRuntimeError(f"Invalid parameter kind: {param.kind}")

        return signature.replace(parameters=new_params)

    # ===------------------------------------------------------------------===
    # The call pipeline (DecoratorPlugin.call)
    # ===------------------------------------------------------------------===

    @dataclass
    class _CompilationSetup:
        """Shared pre-IR-generation state of a host compilation: what
        :meth:`_prepare_compilation` made of the call, handed to every hook
        of the call. ``options`` is what :meth:`_pop_call_keywords` took out
        of the call's keywords, the core's and the DSL's own."""

        function_name: str
        pipeline: str | None
        container_attrs: dict[str, Any]
        no_cache: bool
        extra_link_libs: tuple[str, ...]
        compile_only: bool
        canonicalized_args: tuple[Any, ...]
        canonicalized_kwargs: dict[str, Any]
        sig: inspect.Signature
        location: DSLLocation | None
        trace_hooks: tuple[Callable[..., Any], ...]
        options: dict[str, Any] = field(default_factory=dict)

    def _pop_call_keywords(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Take the DSL's keywords out of a call's ``kwargs`` and return them:
        ``pipeline``, ``container_attrs``, ``no_cache``, ``extra_link_libs``,
        ``compile_only`` and ``trace_hooks``, each at its default when
        absent. They are the DSL's, not the function's: a reserved name
        shadows a parameter of the same name. A DSL with call keywords of its
        own pops them here after ``super()``'s and adds them to the dict,
        which reaches every hook of the call as ``setup.options``."""
        return {
            "pipeline": kwargs.pop("pipeline", None),
            "container_attrs": kwargs.pop("container_attrs", {}),
            "no_cache": kwargs.pop("no_cache", False),
            "extra_link_libs": kwargs.pop("extra_link_libs", ()),
            "compile_only": kwargs.pop("compile_only", False),
            "trace_hooks": kwargs.pop("trace_hooks", None),
        }

    def _resolve_function_name(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        sig: inspect.Signature,
        options: dict[str, Any],
    ) -> str:
        """The symbol name of this call's entry: :meth:`mangle_name` of the
        function's name over the canonical ``args`` by default. A DSL that
        names its entries from a call keyword of its own (``options``) or by
        another scheme does it here."""
        return self.mangle_name(func.__name__, args, sig)

    def _disable_cache_reason(self, options: dict[str, Any]) -> str | None:
        """Why this call must not use the compile cache, or None when it may:
        by default a call that compiles only (``compile_only``) is compiled
        afresh and not cached. A DSL adds its own reasons (a setting that
        dumps the IR of every compile, for one); the reason is logged."""
        if options.get("compile_only"):
            return "the call compiles only (`compile_only=True`)"
        return None

    def _prepare_compilation(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> "_CompilationSetup":
        """Extract kwargs, canonicalize args and mangle the name.

        The call keywords ``pipeline``, ``container_attrs``, ``no_cache``,
        ``extra_link_libs``, ``compile_only`` and ``trace_hooks`` are the
        DSL's (:meth:`_pop_call_keywords`), popped before the remaining
        arguments bind to the function's signature. ``trace_hooks`` are
        callables ``hook(dsl, module, function_name)`` run on this call's
        traced module once it is complete. The entry is named by
        :meth:`_resolve_function_name`; :meth:`_disable_cache_reason` may
        turn the cache off for the call.
        """
        function_name = func.__name__
        self.traced_function = func

        options = self._pop_call_keywords(kwargs)
        pipeline = options.get("pipeline")
        container_attrs = options.get("container_attrs", {})
        no_cache = options.get("no_cache", False) or self.envar.no_cache
        extra_link_libs_arg = options.get("extra_link_libs", ())
        if isinstance(extra_link_libs_arg, (str, bytes, os.PathLike)):
            extra_link_libs_arg = (extra_link_libs_arg,)
        extra_link_libs = _normalize_shared_library_paths(
            os.fspath(lib) for lib in extra_link_libs_arg
        )
        compile_only = options.get("compile_only", False)
        trace_hooks = tuple(options.get("trace_hooks") or ())
        for hook in trace_hooks:
            if not callable(hook):
                raise self.user_error(
                    DiagId.CALL_ARGUMENTS,
                    function_name=function_name,
                    detail=f"`trace_hooks` takes callables, not a `{type(hook).__name__}`",
                )

        if not no_cache:
            reason = self._disable_cache_reason(options)
            if reason:
                no_cache = True
                log().info("Cache is disabled: %s", reason)

        # Get signature of the function
        sig = self._get_signature(func)

        # Get bound arguments
        bound_args = self._get_function_bound_args(sig, function_name, *args, **kwargs)

        # Check the number of arguments
        has_varargs = self._check_arg_count(sig, bound_args, function_name)

        # Canonicalize the input arguments
        canonicalized_args, canonicalized_kwargs = self._canonicalize_args(bound_args)

        # Expand *args/**kwargs into concrete named parameters
        if has_varargs:
            sig = self._expand_varargs_varkw(
                canonicalized_args, canonicalized_kwargs, sig
            )
        function_name = self._resolve_function_name(
            func, canonicalized_args, sig, options
        )

        if not self.envar.debuginfo:
            self.decorator_location = None

        return self._CompilationSetup(
            function_name=function_name,
            pipeline=pipeline,
            container_attrs=container_attrs,
            no_cache=no_cache,
            extra_link_libs=extra_link_libs,
            compile_only=compile_only,
            canonicalized_args=canonicalized_args,
            canonicalized_kwargs=canonicalized_kwargs,
            sig=sig,
            location=self.decorator_location,
            trace_hooks=trace_hooks,
            options=options,
        )

    def _configure_compilation(self, setup: "_CompilationSetup") -> Any:
        """The context the trace and the compile of a call run in, inside the
        call's MLIR context: ``nullcontext()`` by default. A DSL that applies
        per-call settings, checks the call's options against its pipeline or
        installs a trace-time filter opens them here. A deferred step
        (:class:`PendingCompile`) runs after it was left."""
        return nullcontext()

    def after_trace_result(
        self, trace: _TraceResult, setup: "_CompilationSetup"
    ) -> None:
        """Runs once the trace is complete and its arguments checked, before
        anything is compiled or the cache consulted: a DSL reads what the
        trace produced (its module, its cache key, what the body returned)
        and prepares its compile from it. No-op by default."""

    def emit_artifact(self, trace: _TraceResult, setup: "_CompilationSetup") -> Any:
        """A DSL's exit from the call after the trace, before the compile
        cache is consulted. None, the default, continues to compile and
        invoke; any other value is the call's result, with no compile and no
        ``load`` (the module's text, a handle on an artifact the DSL built
        itself); a :class:`PendingCompile` is a compile step the DSL defines
        itself, run by the decorated function's wrapper once the call scope
        is left (a DSL whose entry has no host symbol compiles its container
        and never loads, for one). Not reached under ``<PREFIX>_DRYRUN``."""
        return None

    def run(
        self,
        entry: DecoratorPlugin,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """The pipeline of a call from Python (``DecoratorPlugin.call``): one
        call of the decorated ``func``, with the DSL already active.

        1. Translates the arguments (host buffers -> ``Pointer`` through the
           adapter plugins, ``float`` -> ``f32``, ...) and traces the body into
           the op ``entry`` builds
        2. Compiles and JITs the MLIR module through the compiler role (cached);
           a DSL without one, or a dry run, returns the trace result here
        3. Invokes the compiled function and rebuilds its result

        The call's keywords, name and cache policy come from
        :meth:`_prepare_compilation`; the hooks around the trace and the
        compile are :meth:`generate_mlir`'s.
        """
        # The DSL is active for the whole call: its types emit, its settings
        # are read, and a decorated function called inside the trace sees an
        # open trace (``in_trace``) and is launched into it.
        with active_dsl(self):
            setup = self._prepare_compilation(func, *args, **kwargs)

            log().debug("Generating MLIR for function '%s'", setup.function_name)
            return self.generate_mlir(
                entry,
                func,
                setup.function_name,
                setup.container_attrs,
                setup.canonicalized_args,
                setup.canonicalized_kwargs,
                setup.sig,
                setup.pipeline,
                setup.no_cache,
                setup.compile_only,
                location=setup.location,
                extra_link_libs=setup.extra_link_libs,
                trace_hooks=setup.trace_hooks,
                setup=setup,
            )

    # ===------------------------------------------------------------------===
    # Services for the decorators plugins add
    # ===------------------------------------------------------------------===

    def bind_arguments(
        self,
        func: Callable[..., Any],
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_host: bool = False,
        **boundary_options: Any,
    ) -> tuple[inspect.Signature, tuple[Any, ...], dict[str, Any], JitFuncArgs]:
        """Bind one call of ``func`` for a plugin's decorator.

        :return: The signature, the canonical positional and keyword-only
            arguments (defaults applied), and their IR operands, types and
            attribute dicts from the argument boundary; ``is_host=False`` is the
            device side (a kernel's arguments are the host trace's values);
            the ``boundary_options`` are forwarded to the boundary
        """
        signature = self._get_signature(func)
        self.traced_function = func
        bound = self._get_function_bound_args(signature, name, *args, **kwargs)
        self._check_arg_count(signature, bound, name)
        canonical_args, canonical_kwargs = self._canonicalize_args(bound)
        jit_args = self._generate_jit_func_args(
            func,
            name,
            canonical_args,
            canonical_kwargs,
            signature,
            is_host=is_host,
            **boundary_options,
        )
        if not len(jit_args.values) == len(jit_args.types) == len(jit_args.attributes):
            raise DSLRuntimeError(
                "the argument boundary produced mismatched operands, types and attributes",
                context={
                    "function": name,
                    "values": len(jit_args.values),
                    "types": len(jit_args.types),
                    "attributes": len(jit_args.attributes),
                },
            )
        return signature, canonical_args, canonical_kwargs, jit_args

    def _register_deferred_call(self, call: Any) -> None:
        """Remember a prepared call (``DeferredDecoratorCall``) made inside the open
        trace, with the user's call site for its diagnostic; outside a trace
        nothing is remembered (issuing it there is ``CALL_OUTSIDE_JIT``)."""
        if self._deferred_calls is not None:
            call.creation_loc = find_user_source_location()
            self._deferred_calls.append(call)

    def _check_deferred_calls(self) -> None:
        """Report the first prepared call of the open trace that was never
        issued (``CALL_NEVER_ISSUED``), at the line that prepared it."""
        for call in self._deferred_calls or ():
            if not call.issued:
                filename, lineno, col, end_col = call.creation_loc
                raise self.user_error(
                    DiagId.CALL_NEVER_ISSUED,
                    filename=filename,
                    lineno=lineno,
                    col_offset=col,
                    end_col_offset=end_col,
                    function_name=call.name,
                    decorator=f"@{call.plugin.decorator_name}",
                    issue=call.plugin.issue_example,
                )

    def trace_body(
        self,
        entry: Any,
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
        """Build the function op ``entry`` describes and trace ``func`` into it.

        ``entry`` is the decorator plugin whose function is traced, through
        its ``generate_func_op(name, arg_types, arg_attrs, loc) -> (op,
        entry_block)``: ``func.Jit`` for ``@jit``, a kernels plugin for
        ``@kernel``. The body runs
        with the insertion point in the entry block and the block arguments
        bound to the Python parameters; the caller appends the terminator. An
        entry with a ``body_scope(dsl, func, *, signature)`` context manager
        (optional) has the body run inside it, with the arguments bound.

        :return: The op, its entry block and the trace's Python result
        """
        func_op, entry_block = entry.generate_func_op(
            name, list(arg_types), list(arg_attrs), loc=loc
        )
        log().debug("Generated function op [%s]", func_op)
        with ir.InsertionPoint(entry_block):
            ir_args, ir_kwargs = self.generate_execution_arguments(
                args, kwonlyargs, entry_block, sig
            )
            body_scope = getattr(entry, "body_scope", None)
            scope = (
                nullcontext()
                if body_scope is None
                else body_scope(self, func, signature=sig)
            )
            try:
                with scope:
                    result = func(*ir_args, **ir_kwargs)
            except NameError as name_error:
                # The innermost frame of the traceback is the user's line.
                tb = name_error.__traceback__
                err_filename = err_lineno = None
                while tb is not None:
                    err_filename = tb.tb_frame.f_code.co_filename
                    err_lineno = tb.tb_lineno
                    tb = tb.tb_next
                self.on_trace_name_error(func, name_error, err_filename, err_lineno)
                raise
        return func_op, entry_block, result
