# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The DSL base class :class:`BaseDSL`.

A sub-DSL inherits :class:`BaseDSL` and names its plugins in one ``Plugins``
record on the class. The base owns what is the same for every DSL: the
``@jit`` decorator and the decorator machinery plugins build on
(:meth:`BaseDSL.make_decorator`, :meth:`BaseDSL.jit_runner`), the AST
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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import wraps
from types import UnionType
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin

from ... import ir
from ..core.common import DSLBaseError, DSLRuntimeError, DSLUserCodeError, active_dsl
from ..core.diagnostics import DiagId
from ..core.env_manager import EnvironmentVarManager
from .plugin import DecoratorPlugin, Plugins, _NoRemarkSession
from ..types import typing as t
from ..util import profiler
from ..util.profiler import timer
from ..util import tree_utils
from ..util.logger import log
from .arguments import JitArgAdapterRegistry, adapt_pointer_address
from .staging import Executor, _is_mlir_op_leaf, is_argument_meta
from .user_op import enter_traceback_locations
from ..util.cache import (
    dump_cache_to_path,
    get_default_generated_ir_path,
    JitCacheDict,
    load_cache_from_path,
    read_bytecode_and_check_crc32,
    toolchain_identity,
    write_bytecode_with_crc32,
)

__all__ = [
    "BaseDSL",
    "DSLLocation",
    "DSLSingletonMeta",
    "JitFuncArgs",
]

# =============================================================================
# Global Variables
# =============================================================================

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


# =============================================================================
# Main DSL Class
# =============================================================================


class DSLSingletonMeta(type):
    """
    Metaclass implementing the Singleton pattern for DSL classes.

    One instance exists per concrete subclass, kept in ``_instances``.
    Requesting ``BaseDSL`` itself returns the first concrete instance;
    ``clear_instances`` serves tests.
    """

    _instances: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            log().info("DSLSingletonMeta __call__ for %s", cls)
            if cls is BaseDSL:
                if not cls._instances:
                    raise DSLRuntimeError(
                        "Need to initialize a concrete subclass of BaseDSL first"
                    )
                return next(iter(cls._instances.values()))
            elif cls not in cls._instances:
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
        """A missing public class attribute is usually a decorator the record
        does not provide (``@MyDSL.jit`` on a record without ``func.Jit()``):
        say so, as an ``AttributeError`` so ``getattr`` defaults still work."""
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
            "`plugins = Plugins(..., decorators=[func.Jit()])` for `@jit`."
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


class _PluginDecorator:
    """A plugin's decorator as a class attribute: ``__get__`` binds the DSL
    class it is accessed on, one function per class (so ``Sub.kernel is
    Sub.kernel`` and ``m.kernel is MyDSL.kernel`` hold)."""

    def __init__(self, kind: str, on_call: Callable[..., Any]) -> None:
        self.kind = kind
        self.on_call = on_call
        self._bound: dict[type, Callable[..., Any]] = {}

    def __set_name__(self, owner: type, name: str) -> None:
        self.kind = name

    def __get__(self, obj: Any, owner: type | None = None) -> Callable[..., Any]:
        cls = owner if owner is not None else type(obj)
        bound = self._bound.get(cls)
        if bound is None:
            kind, on_call = self.kind, self.on_call

            def decorator(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
                return BaseDSL.jit_runner(
                    cls,
                    kind,
                    on_call,
                    BaseDSL.get_location_from_frame(
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
    _jit_arg_adapter_scope: ClassVar[str | None] = None
    # Optional compiler-recognized component inserted by a DSL's name mangler.
    _name_mangling_prefix: ClassVar[str] = ""
    #: What the DSL is made of: one plugin per role (``type_ops``,
    #: ``ast_preprocessor``, ``compiler``) and
    #: the families (``decorators``, ``adapters``). ``__init__``
    #: resolves the record once per
    #: instance: a plugin whose ``available()`` is False is dropped (and listed
    #: in ``unavailable_plugins``), the rest are copied and installed in record
    #: order. The base names nothing; a sub-DSL assembles itself here.
    plugins: ClassVar[Plugins] = Plugins()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install the decorators the class's plugins add (``@MyDSL.kernel``):
        every plugin of a record defined on the class itself is asked for its
        ``decorators``; an inherited record was installed on the parent."""
        super().__init_subclass__(**kwargs)
        record = cls.__dict__.get("plugins")
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

    def _remark_session(self, context: ir.Context) -> Any:
        """The compiler plugin's remark session for ``context`` under the
        ``REMARKS``/``REMARKS_POLICY``/``REMARKS_OUTPUT`` settings; a session
        collecting nothing for a DSL without a compiler (a trace-only DSL)."""
        compiler = self.plugins.compiler
        if compiler is None:
            return _NoRemarkSession()
        return compiler.remark_session(
            context,
            remark_filter=self.envar.remarks,
            remark_policy=self.envar.remarks_policy,
            remark_output=self.envar.remarks_output,
        )

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
        self.jit_cache: JitCacheDict = JitCacheDict(
            max_elems=0 if self.envar.no_cache else self.envar.jit_cache_max_elems
        )
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.file_cache_hits: int = 0
        # The structured remarks of the last compile (``RemarkSession.remarks``).
        self.collected_remarks: list[dict[str, Any]] = []

        # set warning
        if self.envar.warnings_ignore:
            warnings.filterwarnings("ignore")

        # Path of the dumped MLIR file; set by build_module when KEEP_IR is active.
        self.dump_mlir_path: Any = None
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
            if ast_preprocessor.preprocessor_class is not None:
                self.preprocessor = ast_preprocessor.preprocessor_class(
                    dsl_package_name or ast_preprocessor.helpers_package(),
                    closure_check=ast_preprocessor.closure_check,
                )
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

    # =========================================================================
    # Decorators
    # =========================================================================

    @staticmethod
    def _lazy_initialize_dsl(func: Any) -> None:
        """
        Lazy initialization of DSL object if has not been initialized
        """
        if hasattr(func, "_dsl_cls"):
            func._dsl_object = func._dsl_cls._get_dsl()
            delattr(func, "_dsl_cls")

    @staticmethod
    def _preprocess_and_replace_code(func: Any) -> None:
        """
        Run ast transformation and replace the function's code object
        """
        # Ensure the DSL instance is materialized before touching _dsl_object
        BaseDSL._lazy_initialize_dsl(func)
        # Update the decorator location to the new function
        func._dsl_object.decorator_location = func._decorator_location

        if getattr(func, "_preprocessed", False) is True:
            return
        if not func._dsl_object.enable_preprocessor:
            func._preprocessed = True
            return

        fcn_ptr = func._dsl_object.run_preprocessor(func)
        if fcn_ptr:
            func.__code__ = (
                fcn_ptr.__code__
                if not isinstance(fcn_ptr, staticmethod)
                else fcn_ptr.__func__.__code__
            )

    @staticmethod
    def jit_runner(
        cls: type["BaseDSL"],
        kind: str,
        on_call: Callable[..., Any],
        location: DSLLocation,
        *decorator_args: Any,
        **decorator_kwargs: Any,
    ) -> Any:
        """
        The decorator machinery shared by every decorator a plugin adds
        (``@jit`` included): validate the target and wrap the function so that
        a call materialises the DSL instance, rewrites the function once (when
        the DSL preprocesses) and hands the call to ``on_call(dsl, func,
        *args, **kwargs)``, the plugin's dispatch, with the DSL active. The
        decorators take no options.

        ``location`` is the user's call site, already resolved to a value by
        the caller via :meth:`get_location_from_frame`: the returned decorator
        outlives this call, so it takes a location rather than a frame.
        """
        log().info("jit_runner")

        def jit_runner_decorator(func: Any) -> Any:
            decorator_kind = kind
            if not inspect.isfunction(func):
                raise DSLUserCodeError(
                    DiagId.CALL_NOT_CALLABLE,
                    decorator=f"@{decorator_kind}",
                    got=f"a `{type(func).__name__}`",
                )
            if decorator_kwargs:
                unknown = sorted(decorator_kwargs)
                raise DSLUserCodeError(
                    DiagId.CALL_ARGUMENTS,
                    function_name=f"@{decorator_kind}",
                    detail=f"no option is named `{unknown[0]}`",
                )
            func._dsl_cls = cls
            # The decorator the function carries ("jit", "kernel", ...).
            func._decorator_kind = decorator_kind
            func._decorator_location = location

            @wraps(func)
            def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                BaseDSL._preprocess_and_replace_code(func)
                # No DSL is made active here: the plugin's dispatch reads the
                # current one to tell a call from Python (none active; ``run``
                # then activates this DSL) from a call inside a trace.
                return on_call(func._dsl_object, func, *args, **kwargs)

            return jit_wrapper

        if len(decorator_args) == 1 and callable(decorator_args[0]):
            return jit_runner_decorator(decorator_args[0])
        else:
            return jit_runner_decorator

    @classmethod
    def make_decorator(cls, kind: str, on_call: Callable[..., Any]) -> Any:
        """The decorator ``@<DSL>.<kind>`` a plugin adds through ``decorators``.

        Bare or as ``@kind()``, with the AST rewrite and the lazy DSL
        instance; a call of the decorated function runs ``on_call(dsl, func,
        *args, **kwargs)``, the plugin's dispatch, with the DSL active. It
        binds the class it is accessed on, so ``@Variant.kernel`` on a subclass
        traces into the variant's instance, not the defining class's. Every
        decorator a DSL has, ``@jit`` included, comes from a plugin this way.
        """
        return _PluginDecorator(kind, on_call)

    # =========================================================================
    # Pipeline
    # =========================================================================

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

    # =========================================================================
    # Staging
    # =========================================================================

    def is_mlir_op(self, value: Any) -> bool:
        """Whether one value is an MLIR op rather than a Python value, for this
        DSL. ``mlir.dsl.is_mlir_op`` calls it for each leaf of a tuple, list or
        frozen record (the containers are walked for you). The default: a raw
        SSA value or a registered leaf whose payload is one. A sub-DSL with its
        own value model overrides it."""
        return _is_mlir_op_leaf(value)

    # =========================================================================
    # Name mangling
    # =========================================================================

    def _is_meta_argument(self, arg: Any, spec_ty: Any) -> bool:
        """True if the boundary will treat ``arg`` as a compile-time value: no
        DSL annotation, no leaf inside it and no registered adapter for it."""
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
            return JitArgAdapterRegistry.get_registered_adapter(arg) is None

    def mangle_name(
        self, function_name: str, args: tuple[Any, ...], sig: inspect.Signature
    ) -> str:
        """Does simple name mangling: the Meta arguments are folded into the
        symbol, so every specialisation of a function gets its own name."""

        # sig.parameters maybe longer than args, but since canonicalized_args
        # only contains positional arguments, we can rely on zip to truncate
        for param, arg in zip(sig.parameters.values(), args):
            if not self._is_meta_argument(arg, param.annotation):
                continue
            if isinstance(arg, type):
                function_name = f"{function_name}_{arg.__name__}"
            elif isinstance(arg, (list, tuple)):
                function_name = f"{function_name}_{'_'.join(map(str, arg))}"
            else:
                function_name = f"{function_name}_{arg}"
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

    # =========================================================================
    # Block arguments -> traced Python arguments
    # =========================================================================

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
            # supplies the owning-function context ``is_argument_meta`` uses to
            # detect reserved ``self``/``cls`` parameters.
            if is_argument_meta(arg, arg_spec, arg_name, idx, self.traced_function):
                ir_arg: Any = arg
            else:
                with JitArgAdapterRegistry.using_scope(self._jit_arg_adapter_scope):
                    base_spec = (
                        get_args(arg_spec)[0]
                        if get_origin(arg_spec) is Annotated
                        else arg_spec
                    )
                    if isinstance(base_spec, t.NumericMeta) and not isinstance(
                        arg, base_spec
                    ):
                        # Numeric type coercion for a staged argument: the block arg
                        # already has the target MLIR type. Wrap it directly
                        # instead of casting the caller's value, which would
                        # emit casts referencing SSA values from an outer region.
                        ir_arg = base_spec(fop_args[iv_block_args])
                        iv_block_args += 1
                    else:
                        _, _, treedef = tree_utils.tree_flatten(
                            arg, return_ir_values=False
                        )
                        n_args = sum(
                            len(tree_utils.leaf_ir_types(leaf))
                            for _, leaf in tree_utils.tree_leaves(treedef)
                        )
                        blk_args = fop_args[iv_block_args : iv_block_args + n_args]
                        ir_arg = self._restore_from_block_args(treedef, blk_args)
                        iv_block_args += n_args

            return ir_arg, iv_block_args

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

    # =========================================================================
    # Host argument boundary
    # =========================================================================

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
        return DSLUserCodeError(
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
    ) -> None:
        """Raise when an argument produced no usable JIT signature: neither a
        known type nor an adapter yielded MLIR types and execution values."""
        if jit_arg_type and (jit_exec_arg or not is_host):
            return

        raise DSLUserCodeError(
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

        On the host side the leaves are marshalled into owning ``c_void_p``
        slots; at the kernel launch boundary they are already staged and their
        ``ir.Value`` s become the launch operands.
        """
        values, attrs, treedef = tree_utils.tree_flatten(
            arg, return_ir_values=not is_host, root=arg_name
        )
        if not is_host:
            return list(values), [v.type for v in values], list(attrs)

        exec_args: list[Any] = []
        arg_types: list[Any] = []
        leaf_values = iter(values)
        for _, leaf in tree_utils.tree_leaves(treedef):
            if leaf.is_none or leaf.is_meta:
                continue
            value = next(leaf_values)
            entry = tree_utils.leaf_entry(type(value))
            if tree_utils.is_staged_leaf(value):
                raise DSLUserCodeError(
                    DiagId.ARG_UNSUPPORTED_TYPE,
                    num=arg_index + 1,
                    arg_name=arg_name,
                    arg_type=type(value).__name__,
                    function_name=function_name,
                )
            if entry is None or entry.marshal is None:
                raise DSLUserCodeError(
                    DiagId.ARG_UNSUPPORTED_TYPE,
                    num=arg_index + 1,
                    arg_name=arg_name,
                    function_name=function_name,
                    arg_type=type(value).__name__,
                    detail=": the DSL knows the type, but it has no host representation",
                )
            with JitArgAdapterRegistry.using_argument(arg_name, arg_index):
                exec_args.extend(entry.marshal(value))
            arg_types.extend(tree_utils.leaf_ir_types(leaf))
        return exec_args, arg_types, [ir.DictAttr.get({})] * len(arg_types)

    def _generate_jit_func_args(
        self,
        func: Any,
        function_name: str,
        args: tuple[Any, ...] | list[Any],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        *,
        is_host: bool = True,
    ) -> JitFuncArgs:
        """Generate JIT function arguments."""
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
            # Implicitly convert into Numeric type if possible
            if isinstance(cast_ty, t.NumericMeta) and not isinstance(arg, cast_ty):
                try:
                    arg = t.cast(arg, cast_ty)  # type: ignore[arg-type]
                except DSLBaseError as exc:
                    raise DSLUserCodeError(
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
                if is_argument_meta(arg, spec_ty, arg_name, i, func):
                    jit_exec_arg = jit_arg_type = jit_arg_attr = None

                if jit_arg_type is not None and len(jit_arg_type) == 0:
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
                    )

            if jit_arg_type is not None:
                # Merge attributes from annotated markers (e.g. grid_constant)
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
    ) -> JitFuncArgs:
        """Convert input arguments to the MLIR function signature and the
        marshalled execution arguments."""

        result = self._generate_jit_func_args(
            func, function_name, args, kwonlyargs, sig, is_host=True
        )

        if len(result.values) != len(result.types):
            raise DSLRuntimeError(
                "expects the same number of arguments and function parameters",
                context={"values": len(result.values), "types": len(result.types)},
            )

        return result

    # =========================================================================
    # Locations
    # =========================================================================

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
        """
        Get python location information and generate MLIR location
        """
        if location is None:
            location = self.decorator_location
        if location is None:
            return ir.Location.unknown()

        file_loc = ir.Location.file(
            location.filename, location.lineno, location.col_offset
        )
        return ir.Location.name(location.function_name, childLoc=file_loc)

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

    # =========================================================================
    # Compile, hash, dump
    # =========================================================================

    def compile_and_jit(
        self,
        module: ir.Module,
        pipeline: str,
        shared_libs: list[str],
        function_name: str = "",
    ) -> Any:
        """
        Compile and JIT an MLIR module through the compiler plugin.

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
                enable_pass_profiling=self.envar.enable_pass_profiling,
                enable_debug_info=self.envar.debuginfo,
                remark_filter=self.envar.remarks,
                remark_policy=self.envar.remarks_policy,
                remark_output=self.envar.remarks_output,
                after_lowering=self._after_lowering,
            )
        except DSLBaseError:
            raise
        except Exception as e:
            raise DSLRuntimeError(
                "compilation failed", context={"function": function_name}, cause=e
            ) from e
        finally:
            profiler.finish_compile()

    def _after_lowering(self, module: ir.Module) -> None:
        """Run every adapter's ``after_lowering`` on the lowered ``module``."""
        for plugin in self.plugins.adapters:
            plugin.after_lowering(self, module)

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
        """A hash of the toolchain identity, the MLIR binaries in use."""
        version_hash = hashlib.sha256()
        version_hash.update(repr(toolchain_identity()).encode())
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
        ``affects_compile`` environment settings, the pipeline, the extra link
        libraries and the toolchain identity (:meth:`get_version`)."""
        s = io.BytesIO()
        module.operation.write_bytecode(s)
        s.write(self.envar.cache_key_str().encode())
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

    def _inspection_pipeline(self, passes: str) -> str:
        """Frame ``passes`` for IR-inspection dumps."""
        passes = passes.strip()
        if passes.startswith("builtin.module("):
            return passes
        return f"builtin.module({passes})"

    def _save_ir(self, module: ir.Module, label: str) -> str:
        """Write ``module`` under ``<PREFIX>_CACHE_DIR`` and return the path."""
        output_dir = get_default_generated_ir_path(self.name)
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"{label}.mlir")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    module.operation.get_asm(enable_debug_info=self.envar.debuginfo)
                )
        except OSError as e:
            raise DSLRuntimeError(
                "failed to save the generated IR", context={"dir": output_dir}, cause=e
            ) from e
        log().info("Saved IR [%s] to [%s]", label, path)
        return path

    def build_module(self, module: ir.Module, function_name: str) -> ir.Module:
        """
        Build the MLIR module, verify and return the module
        """

        # Save IR in a file (raw, before any passes) -- triggered by KEEP_IR
        if self.envar.keep_ir:
            self.dump_mlir_path = self._save_ir(module, function_name)

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

        # Verify the module
        try:
            module.operation.verify()
        except Exception as e:
            raise DSLRuntimeError("IR verification failed", cause=e) from e

        return module

    # =========================================================================
    # Return values
    # =========================================================================

    def _return_values(
        self, entry: DecoratorPlugin, result: Any, sig: inspect.Signature, loc: Any
    ) -> tuple[list[ir.Value], _ResultSpec | None]:
        """Turn the traced return value into what the entry returns.
        Numeric leaves and frozen records/tuples/``@struct``s of them are
        accepted; the decorator plugin's ``pack_results`` decides how the
        leaves travel (the shipped ``func.Jit`` packs several into one struct).
        """
        if result is None:
            if sig.return_annotation not in (inspect.Signature.empty, None):
                raise DSLUserCodeError(
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
                raise DSLUserCodeError(
                    DiagId.TYPE_RETURN_MISMATCH,
                    got=f"a `{cls.__name__}`",
                    detail=(
                        " (a `Pointer` result is memory: write through the pointer instead)"
                        if issubclass(cls, t.Pointer)
                        else ""
                    ),
                )
        if not leaves:
            raise DSLUserCodeError(
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

    # =========================================================================
    # Tracing
    # =========================================================================

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
    ) -> tuple[ir.Module, str | None, Any, _ResultSpec | None]:
        """Trace ``func`` into the op the decorator plugin ``entry`` builds, in
        a fresh module. This runs with the DSL active (``run``), so a decorated
        function called inside the body is launched into this trace, not
        called.

        :return: The verified module, its hash (None under ``no_cache``), the
            trace's Python result and the result slot description
        """

        def build_ir_module() -> tuple[ir.Module, Any, _ResultSpec | None]:
            loc = self.get_ir_location(location)
            module = ir.Module.create(loc=loc)

            with ir.InsertionPoint(module.body):
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
                        entry, result, sig, loc
                    )
                    entry.generate_return(func_op, ret_values, loc=loc)
                for plugin in self.plugins.decorators:
                    plugin.after_trace(self, module)

            return module, result, result_spec

        # Build IR module, then finalize it (hash, verify). The whole closure
        # is one unit so the profiler's build phase covers the finalize steps
        # too, not just the trace.
        def build_and_finalize() -> (
            tuple[ir.Module, str | None, Any, _ResultSpec | None]
        ):
            module, result, result_spec = self._maybe_profile(build_ir_module)()
            # Plugins add to the module before the hash, so what they add is
            # part of the cached artifact (the TVM-FFI wrapper, for one).
            for plugin in self.plugins.adapters:
                plugin.attach_to_module(
                    self, module, function_name, sig, args, kwonlyargs
                )
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
        """Compile ``module`` and cache the callable the compiler plugin loads.

        The lowered module goes into an ``ExecutionEngine`` over
        ``get_shared_libs``; the entry is the packed ``_mlir_<name>`` wrapper.

        With file caching on (``DISABLE_FILE_CACHING`` unset) the lowered module
        is looked up under ``module_hash`` in the cache directory first and, on a
        miss, dumped there after the pipeline ran (bytecode + CRC32), so a later
        process skips the pass pipeline and only builds the engine.
        """
        file_cache_enabled = (
            not no_cache
            and module_hash is not None
            and not self.envar.disable_file_caching
        )
        load_from_file_cache = False
        if file_cache_enabled:
            cached_module = load_cache_from_path(
                self.name,
                module_hash,
                path=get_default_generated_ir_path(self.name),
                bytecode_reader=read_bytecode_and_check_crc32,
            )
            if cached_module is not None:
                self.file_cache_hits += 1
                log().info(
                    "JIT cache hit IN-FILE function=[%s] module_hash=[%s]",
                    function_name,
                    module_hash,
                )
                module = cached_module
                load_from_file_cache = True
        if not load_from_file_cache:
            self.cache_misses += 1
            log().info(
                "JIT cache miss function=[%s] module_hash=[%s]",
                function_name,
                module_hash,
            )
        shared_libs = self.get_shared_libs(extra_link_libs)
        if load_from_file_cache:
            engine = self._maybe_profile(self.jit_lowered_module)(
                module, shared_libs, function_name
            )
        else:
            engine = self._maybe_profile(self.compile_and_jit)(
                module, pipeline, shared_libs, function_name
            )
        # Binding the entry materializes the JIT'd code (ORC compiles lazily),
        # so it belongs to the jit phase together with engine construction.
        load = profiler.timed("jit")(self._maybe_profile(self._compiler.load))
        jit_function = load(
            module,
            engine,
            function_name,
            sig,
            jit_time_profiling=self.envar.jit_time_profiling,
            result_ctype=result_spec.slot if result_spec is not None else None,
        )
        jit_function.result_spec = result_spec  # type: ignore[attr-defined]
        execution_args = getattr(jit_function, "execution_args", None)
        if execution_args is not None:
            execution_args.set_adapter_scope(self._jit_arg_adapter_scope)
        # Decorator plugins record what the trace did; adapters then wrap the
        # function for another ABI (their symbols are in the engine now).
        for plugin in self.plugins.decorators:
            plugin.finish_compiled_function(self, jit_function)
        for plugin in self.plugins.adapters:
            jit_function = plugin.wrap_compiled_function(self, jit_function)

        if not no_cache:
            # The entry lives as long as the decorated function does.
            self.jit_cache.set(module_hash, jit_function, func)
        if file_cache_enabled and not load_from_file_cache:
            dump_cache_to_path(
                self.name,
                jit_function,
                module_hash,
                path=get_default_generated_ir_path(self.name),
                bytecode_writer=lambda f: write_bytecode_with_crc32(
                    f, jit_function.ir_module
                ),
            )
        return jit_function

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
    ) -> Any:
        """Trace ``func`` into the op the decorator plugin ``entry`` builds in
        an MLIR module, compile it through the compiler plugin (or take the
        cached function) and run it.

        :return: The trace's Python result under ``<PREFIX>_DRYRUN`` or on a
            DSL without a compiler, the compiled function under
            ``compile_only``, else the call's result
        """
        # The remark session owns the context's remark engine for the whole
        # call, so trace-time remarks and the passes' remarks share one stream.
        with ir.Context() as ctx, self.get_ir_location(location), self._remark_session(
            ctx
        ) as remark_session:
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
                # Convert input arguments to MLIR arguments
                mlir_func_args = self.generate_mlir_function_types(
                    func, function_name, args, kwonlyargs, sig
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
                )
                for plugin in self.plugins.decorators:
                    plugin.check_arguments(
                        self, sig, args, kwonlyargs, adapted_args, function_name
                    )

                # A dry run generates the IR and stops; so does a call on a DSL
                # without a compiler plugin (trace-only: the IR is the product).
                # An explicit ``compile()`` on such a DSL falls through to the
                # missing-compiler error below.
                if self.envar.dryrun or (
                    self.plugins.compiler is None and not compile_only
                ):
                    return result

                cached_jit_func = None if no_cache else self.jit_cache.get(module_hash)

                if (
                    no_cache
                    or cached_jit_func is None
                    or cached_jit_func.capi_func is None
                ):
                    # no cache or cache miss, do ir generation/compilation/jit engine
                    jit_function = self.compile_and_cache(
                        module,
                        module_hash,
                        function_name,
                        pipeline,
                        sig,
                        no_cache,
                        result_spec=result_spec,
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
                        param.name: tree_utils.tree_flatten(
                            arg, return_ir_values=False
                        )[2]
                        for param, arg in zip(sig.parameters.values(), trace_args)
                    }
                )
            return jit_function

        # Run the compiled program. A function that prefers the Python
        # arguments (the TVM-FFI export marshals them itself) is called with
        # them; the packed entry takes the marshalled ``exe_args``.
        if getattr(jit_function, "prefers_python_args", False):
            return jit_function(*args, **kwonlyargs)
        raw = jit_function.run_compiled_program(exe_args)
        result_spec = getattr(jit_function, "result_spec", None)
        if result_spec is None:
            return None
        return self._result_from_ctypes(result_spec, raw)

    # =========================================================================
    # Preprocessor
    # =========================================================================

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

    # =========================================================================
    # Signature binding
    # =========================================================================

    def _get_function_bound_args(
        self, sig: inspect.Signature, function_name: str, *args: Any, **kwargs: Any
    ) -> inspect.BoundArguments:
        """
        Binds provided arguments to a function's signature and applies default values.

        E.g. given `def foo(a, b=2, c=3)` called as `foo(a=1, c=4)`, the
        returned BoundArguments has args = `[1]` and kwargs = `{'b': 2, 'c': 4}`.
        """
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
        """
        Canonicalize the input arguments so that returned args only contain
        positional arguments and kwargs only contain keyword arguments.
        """
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
        """
        Returns the signature for a given function, handling PEP-563
        (postponed evaluation of type annotations) via eval_str=True.
        """
        try:
            return inspect.signature(func, eval_str=True)
        except NameError as e:
            raise DSLUserCodeError(
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

    # =========================================================================
    # The call pipeline (DecoratorPlugin.call)
    # =========================================================================

    @dataclass
    class _CompilationSetup:
        """Shared pre-IR-generation state of a host compilation."""

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

    def _prepare_compilation(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> "_CompilationSetup":
        """Extract kwargs, canonicalize args and mangle the name.

        The call keywords ``pipeline``, ``container_attrs``, ``no_cache``,
        ``extra_link_libs`` and ``compile_only`` are the DSL's, popped before
        the remaining arguments bind to the function's signature.
        """
        function_name = func.__name__
        self.traced_function = func

        pipeline = kwargs.pop("pipeline", None)
        container_attrs = kwargs.pop("container_attrs", {})
        no_cache = kwargs.pop("no_cache", False) or self.envar.no_cache
        extra_link_libs_arg = kwargs.pop("extra_link_libs", ())
        if isinstance(extra_link_libs_arg, (str, bytes, os.PathLike)):
            extra_link_libs_arg = (extra_link_libs_arg,)
        extra_link_libs = _normalize_shared_library_paths(
            os.fspath(lib) for lib in extra_link_libs_arg
        )
        compile_only = kwargs.pop("compile_only", False)

        if not no_cache and compile_only:
            no_cache = True
            log().info("Cache is disabled as user wants to compile only.")

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
        function_name = self.mangle_name(function_name, canonicalized_args, sig)

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
        )

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
            )

    # =========================================================================
    # Services for the decorators plugins add
    # =========================================================================

    def bind_arguments(
        self,
        func: Callable[..., Any],
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_host: bool = False,
    ) -> tuple[inspect.Signature, tuple[Any, ...], dict[str, Any], JitFuncArgs]:
        """Bind one call of ``func`` for a plugin's decorator.

        :return: The signature, the canonical positional and keyword-only
            arguments (defaults applied), and their IR operands, types and
            attribute dicts from the argument boundary; ``is_host=False`` is the
            device side (a kernel's arguments are the host trace's values)
        """
        signature = self._get_signature(func)
        self.traced_function = func
        bound = self._get_function_bound_args(signature, name, *args, **kwargs)
        self._check_arg_count(signature, bound, name)
        canonical_args, canonical_kwargs = self._canonicalize_args(bound)
        jit_args = self._generate_jit_func_args(
            func, name, canonical_args, canonical_kwargs, signature, is_host=is_host
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
        bound to the Python parameters; the caller appends the terminator.

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
            try:
                result = func(*ir_args, **ir_kwargs)
            except NameError as name_error:
                # Extract the source location from the NameError traceback.
                tb = name_error.__traceback__
                err_filename = err_lineno = None
                while tb is not None:
                    err_filename = tb.tb_frame.f_code.co_filename
                    err_lineno = tb.tb_lineno
                    tb = tb.tb_next
                raise DSLUserCodeError(
                    DiagId.SCOPE_UNBOUND_NAME_IN_TRACE,
                    filename=err_filename,
                    lineno=err_lineno,
                    cause=name_error,
                    var=getattr(name_error, "name", None) or f"<in {func.__name__}>",
                    function_name=func.__name__,
                ) from name_error
        return func_op, entry_block, result
