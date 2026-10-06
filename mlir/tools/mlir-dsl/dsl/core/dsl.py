# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The DSL base class :class:`BaseDSL` and its kernel launch driver.

A sub-DSL inherits :class:`BaseDSL`, sets its ``ClassVar`` knobs and its
``plugins`` list and passes its compiler provider to ``__init__``. The base
handles the mechanics that are the same for every dialect: the ``@jit`` and
``@kernel`` decorators, the AST preprocessor run, the host argument boundary,
tracing the body into the host entry the dialect plugin builds (``func.func`` in the LLVM world),
the pass pipeline, the in-memory and on-disk compile caches, the packed
invocation through the ``ExecutionEngine`` and the kernel launch driver that a
``gpu`` plugin's kernel generation helper fills in.
"""

import copy
import hashlib
import inspect
import io
import logging
import os
import re
import threading
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from types import UnionType
from typing import Annotated, Any, ClassVar, NamedTuple, Union, get_args, get_origin

from ... import ir
from ..core.common import DSLBaseError, DSLRuntimeError, DSLUserCodeError, active_dsl
from ..core.diagnostics import DiagId, find_user_source_location
from ..core.env_manager import EnvironmentVarManager
from ..core.plugin import DialectPlugin, ASTPreprocessorPlugin, Plugin
from ..types import typing as t
from ..util import phase_profiler
from ..util import tree_utils
from ..util.logger import log
from ..runtime.jit_arg_adapters import (
    JitArgAdapterRegistry,
    adapt_pointer_address,
    is_argument_meta,
)
from .executor import Executor
from ..util.cache_helpers import (
    dump_cache_to_path,
    get_default_generated_ir_path,
    load_cache_from_path,
    read_bytecode_and_check_crc32,
    write_bytecode_with_crc32,
)
from ..compiler.jit_executor import (
    JitCacheDict,
    JitCompiledFunction,
    lookup_packed_function,
)
from ..util.cache_key import toolchain_identity
from ..util.timing import timer

__all__ = [
    "BaseDSL",
    "DSLLocation",
    "DSLSingletonMeta",
    "JitFuncArgs",
    "KernelLauncher",
    "KernelReturns",
    "LaunchConfig",
    "_KernelGenHelper",
]

# =============================================================================
# Global Variables
# =============================================================================

# Characters stripped from a mangled function name, and the precomputed
# translation table (mangle_name runs per compile, plus once per kernel trace).
_MANGLE_UNWANTED_CHARS = r"'-![]#,.<>()\":{}=%?@;"
_MANGLE_TRANSLATION_TABLE = str.maketrans("", "", _MANGLE_UNWANTED_CHARS)

# The pass list of the host entry itself: the dialect plugins (and the DSL's
# default dialects) lower the IR they emit, host entry included, in install
# order; the core
# then lowers the ``func`` entry and reconciles the casts.
_CORE_PIPELINE_PASSES: tuple[str, ...] = ("reconcile-unrealized-casts",)


class KernelReturns(NamedTuple):
    """Result of ``kernel_launcher``'s ``kernel_wrapper``."""

    kernel_func_ret: Any
    launch_op_ret: Any


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


@dataclass(frozen=True)
class DSLLocation:
    """
    Python source location of DSL code, used to annotate the generated IR.

    ``caller_locs`` is an optional tuple of (filename, lineno) pairs for the
    callsite chain.
    """

    filename: str
    lineno: int
    col_offset: int
    function_name: str
    caller_locs: tuple = ()


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
    shape of the traced return and the host-side slot descriptor of the dialect
    plugin's host helper (the LLVM world: a ``ctypes`` type)."""

    treedef: Any
    slot: Any


@dataclass
class LaunchConfig:
    """Grid, block and optional cluster dimensions plus dynamic shared memory
    of one kernel launch.

    Dimensions accept Python ints or staged integers and are padded to three
    entries; their type and count are validated by the kernel generation
    helper that emits the launch. ``async_deps`` is kept for signature
    fidelity and must be empty: launches are synchronous.
    """

    cluster: list[Any] | None = None
    grid: list[Any] = field(default_factory=lambda: [1, 1, 1])
    block: list[Any] = field(default_factory=lambda: [1, 1, 1])
    smem: int | None = None
    async_deps: list[Any] = field(default_factory=list)

    @staticmethod
    def _check_and_canonicalize_dim(dim: Any, name: str) -> list[Any]:
        """Return ``dim`` (a scalar or a sequence) as a list padded with 1s to
        three entries; a longer list is left for the launch to diagnose under
        ``name``."""
        if not isinstance(dim, (list, tuple)):
            dim = [dim]
        return list(dim) + [1] * (3 - len(dim))

    def __post_init__(self) -> None:
        self.grid = self._check_and_canonicalize_dim(self.grid, "grid")
        self.block = self._check_and_canonicalize_dim(self.block, "block")
        if self.cluster is not None:
            self.cluster = self._check_and_canonicalize_dim(self.cluster, "cluster")


class _KernelGenHelper(ABC):
    """Generates the kernel function op, its terminator and the launch op.

    A dialect plugin subclasses it and installs the subclass as the DSL's
    ``kernel_gen_helper``; ``kernel_launcher`` instantiates one per kernel
    trace. ``diag_ids`` names the plugin's namespaced diagnostic catalogue
    (the ``LAUNCH_*`` codes): the launch driver of this module raises those
    codes through it, so the core imports no plugin catalogue.
    """

    diag_ids: ClassVar[Any] = None

    def __init__(self) -> None:
        self.func_op: Any = None
        self.func_type: Any = None

    @abstractmethod
    def generate_func_op(
        self,
        arg_types: list[Any],
        arg_attrs: list[Any],
        kernel_name: str,
        loc: Any = None,
    ) -> Any:
        if arg_types is None:
            raise DSLRuntimeError("Invalid arg_types!")
        if not kernel_name:
            raise DSLRuntimeError("kernel name is empty")

    @abstractmethod
    def generate_func_ret_op(self) -> None:
        pass

    @abstractmethod
    def generate_launch_op(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def get_func_body_start(self) -> Any:
        pass

    # -- the kernel container ------------------------------------------------

    @classmethod
    def build_container(cls, attrs: dict[str, Any], loc: Any = None) -> Any:
        """Create the op holding this target's kernels at the current
        insertion point (the gpu plugin: ``gpu.module @kernels`` plus the
        ``gpu.container_module`` marker on the host module), or None when
        the target's kernels live in the host module itself."""
        return None

    @classmethod
    def container_insertion_point(
        cls, container: Any, module: Any
    ) -> ir.InsertionPoint:
        """Where a kernel of this trace is emitted: inside ``container`` when
        there is one, else at the start of the host ``module``."""
        return ir.InsertionPoint.at_block_begin(module.body)

    @classmethod
    def prune_empty_containers(cls, module: Any) -> None:
        """Drop a container that received no kernel, after the trace."""

    @classmethod
    def kernel_symbol(cls, kernel_name: str) -> ir.Attribute:
        """The symbol a launch refers to."""
        return ir.FlatSymbolRefAttr.get(kernel_name)


class _NoRemarkSession:
    """The remark session of a DSL that has no compiler: collects nothing."""

    remarks: list[dict[str, Any]] = []

    def __enter__(self) -> "_NoRemarkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _HostGenHelper(ABC):
    """Builds the host entry of a ``@jit`` function and its result slot.

    A dialect plugin subclasses it and installs the subclass as the DSL's
    ``host_gen_helper``; one instance serves one trace. The core decides what
    is returned (the leaves of the traced return value, numeric only); the
    helper decides the function op, how several leaves travel back to the
    host (the LLVM world packs them into one ``!llvm.struct`` read through
    ``ctypes``) and how a raw slot value becomes each leaf's Python value.
    """

    def __init__(self) -> None:
        self.func_op: Any = None

    @abstractmethod
    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> Any:
        """Create the entry with ``arg_types`` and no results; return its entry block."""

    @abstractmethod
    def generate_return(self, values: list[Any], loc: Any = None) -> None:
        """Emit the terminator returning ``values`` and fix the entry's result types."""

    @abstractmethod
    def pack_results(
        self, values: list[Any], prototypes: list[Any], loc: Any = None
    ) -> tuple[list[Any], Any]:
        """The values the entry returns for the result leaves ``values`` (of
        dtypes ``prototypes``) and the host-side slot descriptor, None for no
        result."""

    @abstractmethod
    def unpack_result(self, slot: Any, raw: Any, prototypes: list[Any]) -> list[Any]:
        """The Python value of each result leaf from the filled slot ``raw``."""


class KernelLauncher:
    """Bound kernel arguments awaiting their launch inside a ``@jit`` body::

    kernel(arg1, arg2).launch(LaunchConfig(grid=[1, 1, 1], block=[1, 1, 1]))
    kernel(arg1, arg2).launch(grid=[1, 1, 1], block=[1, 1, 1])
    """

    def __init__(
        self,
        dsl: "BaseDSL",
        kernelGenHelper: type[_KernelGenHelper],
        funcBody: Callable[..., None],
        /,
        *func_args: Any,
        **func_kwargs: Any,
    ) -> None:
        self.dsl = dsl
        self.kernelGenHelper = kernelGenHelper
        self.funcBody = funcBody
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self._launch_name: str | None = None

        # While a host body is being traced, register so an un-launched call is
        # reported (see `_track_deferred_kernel_launches`); capture the call
        # site now, while the user's frame is live, for the diagnostic's caret.
        self._launched = False
        self._creation_loc: tuple[Any, Any, Any, Any] = (None, None, None, None)
        if dsl._pending_launches is not None:
            self._creation_loc = find_user_source_location()
            dsl._pending_launches.append(self)

        self._check_func_args(funcBody, *func_args, **func_kwargs)

    def _check_func_args(
        self, funcBody: Any, *func_args: Any, **func_kwargs: Any
    ) -> None:
        # func_args and func_kwargs should match funcBody's signature.
        try:
            inspect.signature(funcBody).bind(*func_args, **func_kwargs)
        except TypeError as e:
            raise DSLUserCodeError(
                DiagId.CALL_SIGNATURE_MISMATCH,
                provided=len(func_args),
                provided_kw=len(func_kwargs),
                cause=e,
            ) from e

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """Emit the kernel and its launch at the current insertion point.

        Accepts one :class:`LaunchConfig` or its constructor arguments.
        """
        kernel_name = getattr(self.funcBody, "__name__", "<kernel>")
        # No active MLIR context means there is no @jit compilation in
        # progress to emit the launch into.
        if ir.Context.current is None:
            raise DSLUserCodeError(
                self.dsl._launch_diag("LAUNCH_OUTSIDE_JIT"), kernel_name=kernel_name
            )
        if self._launched:
            raise DSLUserCodeError(
                self.dsl._launch_diag("LAUNCH_ALREADY_ISSUED"), kernel_name=kernel_name
            )
        # A launch is being issued: this launcher is no longer a dangling
        # `my_kernel(...)` call (see `_track_deferred_kernel_launches`).
        self._launched = True

        if len(args) == 1 and not kwargs and isinstance(args[0], LaunchConfig):
            config = args[0]
        else:
            config = self.dsl.LaunchConfig(*args, **kwargs)
        if config.async_deps:
            raise DSLUserCodeError(
                self.dsl._launch_diag("LAUNCH_STREAM_UNSUPPORTED"),
                kernel_name=kernel_name,
            )

        kernel_generator = self.dsl.kernel_launcher(
            requiredArgs=["config"], kernelGenHelper=self.kernelGenHelper
        )(self.funcBody)
        ret, name = kernel_generator(*self.func_args, **self.func_kwargs, config=config)
        self.dsl.kernel_info[name] = config
        self.dsl.launch_inner_count += 1
        self._launch_name = name
        return ret.launch_op_ret

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.launch(*args, **kwargs)


class _DefaultDialects:
    """Descriptor resolving ``BaseDSL.default_dialects`` once, on first access,
    so the core imports the default dialect plugins only when a DSL is built."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.owner, self.attr = owner, name

    def __get__(self, obj: Any, cls: type | None = None) -> tuple[DialectPlugin, ...]:
        from ..plugins.dialects.llvm import LlvmDialectPlugin
        from ..plugins.dialects.scf import ScfDialectPlugin

        defaults = (ScfDialectPlugin(), LlvmDialectPlugin())
        setattr(self.owner, self.attr, defaults)
        return defaults


class BaseDSL(metaclass=DSLSingletonMeta):
    """The base of every DSL: one singleton instance per concrete subclass.

    The class attributes below are the sub-DSL knobs; the instance attributes
    set in ``__init__`` are the per-DSL state (environment manager, caches,
    executors, installed plugins). ``@MyDSL.jit`` and ``@MyDSL.kernel``
    decorate functions for the subclass; both dispatch through
    :meth:`jit_runner` to the instance's ``_func`` or ``_kernel_helper``.
    """

    # The op holding the kernels of the current trace (the kernel helper's
    # ``build_container``) and the module being traced.
    kernel_container: Any = None
    current_module: Any = None
    _env_class: type[EnvironmentVarManager] = EnvironmentVarManager
    _jit_arg_adapter_scope: ClassVar[str] = JitArgAdapterRegistry.GPU_DIALECT_SCOPE
    # Optional compiler-recognized component inserted by a DSL's name mangler.
    _name_mangling_prefix: ClassVar[str] = ""
    # Plugins listed on the class; ``__init__`` installs a shallow copy of
    # each on the instance as its last step.
    plugins: ClassVar[Sequence[Plugin]] = ()
    # The AST preprocessor used when ``plugins`` lists no ``ASTPreprocessorPlugin``: the DSL's
    # own AST preprocessor (``MlirDSL``: the scf preprocessor). None preprocesses
    # nothing, so such a DSL uses the explicit builders only.
    default_ast_preprocessor: ClassVar[ASTPreprocessorPlugin | None] = None
    # The dialect plugins installed when ``plugins`` lists none with an
    # ``emitter``: the IR world of the DSL's types. The core's default is
    # MLIR's own, ``scf`` for control flow and the LLVM world for the types,
    # resolved on first use so the core imports no plugin at import time. A
    # sub-DSL whose dialect plugin brings an emitter gets none of them.
    default_dialects: ClassVar[Sequence[DialectPlugin]] = _DefaultDialects()  # type: ignore[assignment]

    LaunchConfig = LaunchConfig
    _KernelGenHelper = _KernelGenHelper

    def _remark_session(self, context: ir.Context) -> Any:
        """The compiler provider's remark session for ``context`` under the
        ``REMARKS``/``REMARKS_POLICY``/``REMARKS_OUTPUT`` settings; a session
        collecting nothing for a DSL without a compiler (a trace-only DSL)."""
        if self.compiler_provider is None:
            return _NoRemarkSession()
        return self._compiler.remark_session(
            context,
            remark_filter=self.envar.remarks,
            remark_policy=self.envar.remarks_policy,
            remark_output=self.envar.remarks_output,
        )

    def _is_supported_arch(self) -> None:
        return

    def __init__(
        self,
        *,
        name: str,
        dsl_package_name: list[str],
        pass_sm_arch_name: str,
        compiler_provider: Any = None,
        device_compilation_only: bool = False,
        preprocess: bool = False,
    ) -> None:
        """
        Initialize the DSL with its providers and environment settings.

        :param name: Name of the DSL; the environment variable prefix
            (``<name>_DRYRUN``, ...) and the logging label
        :param dsl_package_name: The DSL's package path, used by the
            preprocessor to recognise its own symbols
        :param compiler_provider: The compiler running the pass pipeline and
            executing the module; the first dialect plugin bringing one
            (the LLVM world's ``Compiler``) when None
        :param pass_sm_arch_name: The pipeline option that names the target
            architecture (appended to ``<name>_PIPELINE``)
        :param device_compilation_only: Trace device code only
        :param preprocess: Enable the AST preprocessor

        Reads the environment through ``EnvironmentVarManager``, configures
        warnings and logging, and installs the plugins listed on the class
        last, so the subclass's own ``__init__`` still runs after them and wins.
        """
        # Enforcing initialization of instance variables
        if not all([name, pass_sm_arch_name]):
            raise DSLRuntimeError(
                "All required parameters must be provided and non-empty"
            )

        self.name: str = name
        self.compiler_provider: Any = compiler_provider
        self.pass_sm_arch_name: str = pass_sm_arch_name
        self.decorator_location: DSLLocation | None = None
        self.no_cache: bool = False
        self.device_compilation_only: bool = device_compilation_only
        self.num_kernels: int = 0
        # Read environment variables
        self.envar: EnvironmentVarManager = self._create_environment_manager()
        self.enable_preprocessor: bool = preprocess and bool(
            self.envar.ast_preprocessor
        )
        # This cache uses hash of original ir and env as key. Enabled by default
        self.jit_cache: JitCacheDict = JitCacheDict(
            max_elems=0 if self.envar.no_cache else self.envar.jit_cache_max_elems
        )
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.file_cache_hits: int = 0
        # The structured remarks of the last compile (``_RemarkSession.remarks``).
        self.collected_remarks: list[dict[str, Any]] = []

        self.host_jit_decorator_name: str = f"@{BaseDSL.jit.__name__}"
        self.device_jit_decorator_name: str = f"@{BaseDSL.kernel.__name__}"

        # set warning
        if self.envar.warnings_ignore:
            warnings.filterwarnings("ignore")

        # kernel info contains per kernel info including symbol string and
        # launch config. It's valid until the compilation is done.
        self.kernel_info: OrderedDict[str, Any] = OrderedDict()
        # used to generate unique name for gpu.launch
        self.launch_inner_count: int = 0
        # Path of the dumped MLIR file; set by build_module when KEEP_IR is active.
        self.dump_mlir_path: Any = None
        # The function being traced; set on every path that reaches the
        # argument boundary (``_prepare_compilation``, ``kernel_launcher``).
        self.funcBody: Callable[..., Any] | None = None
        # KernelLaunchers built during the current host trace (see
        # `_track_deferred_kernel_launches`); None when no host body is traced.
        self._pending_launches: list[KernelLauncher] | None = None
        # The control-flow executors of this instance, filled from the AST preprocessor plugin
        # plugin's ``executors`` once the plugins are installed.
        self.executor: Executor = Executor()
        # The kernel generation helper class a plugin installs; None until then.
        self.kernel_gen_helper: type[_KernelGenHelper] | None = None
        self.host_gen_helper: type[_HostGenHelper] | None = None
        log().info("Initializing %s DSL", name)

        if self.envar.jit_time_profiling:
            self.profiler: Any = timer(enable=True)

        # Install a per-instance shallow copy of every plugin listed on the
        # class, last: a plugin instance listed on two classes must not carry
        # another DSL's back-reference, and the subclass's own ``__init__``
        # (its ``set_functions`` call) resumes after this and wins.
        self.plugins: tuple[Plugin, ...] = tuple(
            copy.copy(plugin) for plugin in type(self).plugins
        )
        for plugin in self.plugins:
            plugin.install(self)
        # A listed dialect plugin with an emitter owns the IR of the type system;
        # otherwise the class's default dialects are installed alongside.
        self.default_dialect_plugins: tuple[DialectPlugin, ...] = ()
        if not any(
            isinstance(plugin, DialectPlugin) and plugin.emitter is not None
            for plugin in self.plugins
        ):
            self.default_dialect_plugins = tuple(
                copy.copy(plugin) for plugin in type(self).default_dialects
            )
            for plugin in self.default_dialect_plugins:
                plugin.install(self)
        # The emitter the scalar types ask for their SSA types and ops.
        self.emitter: Any = next(
            (
                plugin.emitter
                for plugin in self._dialect_plugins()
                if plugin.emitter is not None
            ),
            None,
        )
        # The first dialect plugin with kernels serves ``@kernel``, the first
        # with a host entry serves ``@jit``, the first with a compiler compiles
        # (unless the constructor was given one).
        self.kernel_gen_helper = next(
            (
                plugin.kernel_gen_helper
                for plugin in self._dialect_plugins()
                if plugin.kernel_gen_helper is not None
            ),
            None,
        )
        self.host_gen_helper = next(
            (
                plugin.host_gen_helper
                for plugin in self._dialect_plugins()
                if plugin.host_gen_helper is not None
            ),
            None,
        )
        if self.compiler_provider is None:
            self.compiler_provider = next(
                (
                    plugin.compiler_provider
                    for plugin in self._dialect_plugins()
                    if plugin.compiler_provider is not None
                ),
                None,
            )
        # The first listed AST preprocessor plugin (else the class's default AST preprocessor)
        # supplies the executors and the preprocessor.
        self.ast_preprocessor: ASTPreprocessorPlugin | None = next(
            (
                plugin
                for plugin in self.plugins
                if isinstance(plugin, ASTPreprocessorPlugin)
            ),
            None,
        )
        if (
            self.ast_preprocessor is None
            and type(self).default_ast_preprocessor is not None
        ):
            self.ast_preprocessor = copy.copy(type(self).default_ast_preprocessor)
            self.ast_preprocessor.install(self)
        if self.ast_preprocessor is not None:
            self.executor.set_functions(**self.ast_preprocessor.executors(self))
        if preprocess:
            if (
                self.ast_preprocessor is None
                or self.ast_preprocessor.preprocessor_class is None
            ):
                raise DSLRuntimeError(
                    "the DSL preprocesses (preprocess=True) but lists no AST preprocessor "
                    "plugin with a preprocessor; add one such as `ScfASTPreprocessorPlugin()` "
                    "to `plugins`",
                    context={"dsl": name},
                )
            self.preprocessor: Any = self.ast_preprocessor.preprocessor_class(
                dsl_package_name,
                warnings_ignore=self.envar.warnings_ignore,
                closure_check=self.ast_preprocessor.closure_check,
            )
            self.package_name = dsl_package_name

    @property
    def _compiler(self) -> Any:
        """The compiler of this DSL; an error when no plugin brought one."""
        if self.compiler_provider is None:
            raise DSLRuntimeError(
                "this DSL has no compiler: pass `compiler_provider=` to `BaseDSL.__init__` "
                "or list a dialect plugin that brings one (the LLVM world's "
                "`LlvmDialectPlugin`)",
                context={"dsl": self.name},
            )
        return self.compiler_provider

    def _host_gen_helper(self) -> _HostGenHelper:
        """One host-entry helper for one trace, from the dialect plugins."""
        if self.host_gen_helper is None:
            raise DSLRuntimeError(
                "no dialect plugin of this DSL builds a host entry for `@jit`: list one "
                "with a `host_gen_helper` (the LLVM world's `LlvmDialectPlugin`, or "
                "`LlvmHostGenHelper` reused by your dialect plugin)",
                context={"dsl": self.name},
            )
        return self.host_gen_helper()

    def _create_environment_manager(self) -> EnvironmentVarManager:
        """Create the environment manager for this DSL's prefix."""
        return self._env_class(self.name)

    def print_warning(self, message: str) -> None:
        """Log and emit ``message`` as a ``UserWarning`` unless warnings are
        ignored (``<PREFIX>_WARNINGS_IGNORE``)."""
        if self.envar.warnings_ignore:
            return
        log().warning("Warning: %s", message)
        warnings.warn(message, UserWarning)

    @classmethod
    def _get_dsl(cls) -> Any:
        """The instance of ``cls``; the singleton metaclass builds it once."""
        return cls()  # type: ignore[call-arg]

    def _plugin_named(self, name: str) -> Plugin | None:
        """Return the installed plugin called ``name``, or None."""
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None

    def _dialect_plugins(self) -> list[DialectPlugin]:
        """The installed plugins that bring a dialect or a target, the listed
        ones first and the class's default dialects last."""
        plugins = [p for p in self.plugins if isinstance(p, DialectPlugin)]
        plugins.extend(getattr(self, "default_dialect_plugins", ()))
        return plugins

    def _launch_diag(self, code: str) -> Any:
        """The launch diagnostic ``code`` of the installed kernel helper's catalogue."""
        catalogue = getattr(self.kernel_gen_helper, "diag_ids", None)
        if catalogue is None or not hasattr(catalogue, code):
            raise DSLRuntimeError(
                f"the kernel generation helper declares no `{code}` diagnostic",
                context={"helper": repr(self.kernel_gen_helper)},
            )
        return getattr(catalogue, code)

    # =========================================================================
    # Decorators
    # =========================================================================

    @staticmethod
    def _can_preprocess(**dkwargs: Any) -> bool:
        """
        Check if AST transformation is enabled or not for `jit` and `kernel` decorators.
        """
        return dkwargs.pop("preprocess", True)

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

        # Keep "already preprocessed" separate from "preprocessing is disabled".
        # The latter is a hard opt-out.
        if getattr(func, "_preprocess_enabled", True) is False:
            func._preprocessed = True
            return
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
        executor_name: str,
        location: DSLLocation,
        *dargs: Any,
        **dkwargs: Any,
    ) -> Any:
        """
        Decorator to mark a function for JIT compilation.

        ``location`` is the user's call site, already resolved to a value by
        the caller via :meth:`get_location_from_frame`: the returned decorator
        outlives this call, so it takes a location rather than a frame.
        """
        log().info("jit_runner")

        def jit_runner_decorator(func: Any) -> Any:
            decorator_kind = "kernel" if executor_name == "_kernel_helper" else "jit"
            if not inspect.isfunction(func):
                raise DSLUserCodeError(
                    DiagId.CALL_NOT_CALLABLE,
                    decorator=f"@{decorator_kind}",
                    got=f"a `{type(func).__name__}`",
                )
            unknown = sorted(set(dkwargs) - {"preprocess"})
            if unknown:
                raise DSLUserCodeError(
                    DiagId.CALL_UNEXPECTED_KWARG, argument_name=unknown[0]
                )
            # Run preprocessor that alters AST
            preprocess_enabled = BaseDSL._can_preprocess(**dkwargs)
            func._dsl_cls = cls
            # Distinguish @jit-decorated targets (executor ``_func``, the host
            # wrapper) from @kernel-decorated targets (executor
            # ``_kernel_helper``, the KernelLauncher).
            func._decorator_kind = decorator_kind
            func._decorator_location = location
            func._preprocess_enabled = preprocess_enabled
            if not hasattr(func, "_preprocessed") and not preprocess_enabled:
                func._preprocessed = True

            @wraps(func)
            def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                BaseDSL._preprocess_and_replace_code(func)

                with active_dsl(func._dsl_object):
                    return getattr(func._dsl_object, executor_name)(
                        func, *args, **kwargs
                    )

            return jit_wrapper

        if len(dargs) == 1 and callable(dargs[0]):
            return jit_runner_decorator(dargs[0])
        else:
            return jit_runner_decorator

    @classmethod
    def jit(cls, *dargs: Any, **dkwargs: Any) -> Any:
        """
        Decorator to mark a function for JIT compilation for Host code.

        Used bare (``@jit``) or with keywords (``@jit(preprocess=False)``);
        every call of the decorated function traces, compiles (cached) and
        runs it.
        """
        return BaseDSL.jit_runner(
            cls,
            "_func",
            BaseDSL.get_location_from_frame(
                inspect.currentframe().f_back  # type: ignore[union-attr]
            ),
            *dargs,
            **dkwargs,
        )

    @classmethod
    def kernel(cls, *dargs: Any, **dkwargs: Any) -> Any:
        """
        Decorator to mark a function for JIT compilation for GPU.

        Calling the decorated function returns a :class:`KernelLauncher`;
        ``.launch(...)`` inside a ``@jit`` body traces and launches the kernel.
        """
        return BaseDSL.jit_runner(
            cls,
            "_kernel_helper",
            BaseDSL.get_location_from_frame(
                inspect.currentframe().f_back  # type: ignore[union-attr]
            ),
            *dargs,
            **dkwargs,
        )

    # =========================================================================
    # GPU module and kernel helper seams (filled by the gpu plugin)
    # =========================================================================

    def _kernel_helper(self, func: Any, *args: Any, **kwargs: Any) -> KernelLauncher:
        """
        Helper function to handle kernel generation logic

        Calling a ``@kernel`` returns a :class:`KernelLauncher`; the kernel is
        traced and launched when its ``launch`` runs inside a ``@jit`` body.
        Needs a dialect plugin providing kernels (the ``gpu`` plugin).
        """
        if self.kernel_gen_helper is None:
            raise DSLUserCodeError(
                DiagId.CALL_PLUGIN_REQUIRED,
                name=self.device_jit_decorator_name,
                plugin="gpu",
                plugin_class="GpuPlugin",
            )
        return KernelLauncher(self, self.kernel_gen_helper, func, *args, **kwargs)

    def _enter_kernel_container(self) -> ir.InsertionPoint:
        """The insertion point for a kernel of the current trace."""
        if self.kernel_gen_helper is None:
            raise DSLRuntimeError(
                "no kernel helper is installed", context={"dsl": self.name}
            )
        return self.kernel_gen_helper.container_insertion_point(
            self.kernel_container, self.current_module
        )

    @contextmanager
    def _track_deferred_kernel_launches(self) -> Generator[None, None, None]:
        """Scope the host-function body: a ``@kernel`` call returns a deferred
        launcher, so a bare ``my_kernel(...)`` statement compiles to nothing;
        on clean exit any launcher never launched is a mistake."""
        outer, self._pending_launches = self._pending_launches, []
        try:
            yield
            pending = self._pending_launches
        finally:
            self._pending_launches = outer
        for launcher in pending:
            if not launcher._launched:
                filename, lineno, col, end_col = launcher._creation_loc
                raise DSLUserCodeError(
                    self._launch_diag("LAUNCH_NEVER_ISSUED"),
                    filename=filename,
                    lineno=lineno,
                    col_offset=col,
                    end_col_offset=end_col,
                    kernel_name=getattr(launcher.funcBody, "__name__", "<kernel>"),
                )

    # =========================================================================
    # Pipeline
    # =========================================================================

    def _get_pipeline(self, pipeline: str | None) -> str:
        """
        Get the pipeline from the other configuration options.

        An explicit ``pipeline`` (a call keyword) is used as given, the
        ``<PREFIX>_PIPELINE`` environment variable with the arch option
        appended, otherwise the plugins' passes (install order) followed by the
        core list, wrapped as ``builtin.module(...)``.
        """
        if pipeline is not None:
            return pipeline
        if self.envar.pipeline is not None:
            if self.envar.arch:
                return self.preprocess_pipeline(self.envar.pipeline, self.envar.arch)
            return self.envar.pipeline

        passes: list[str] = []
        for plugin in self._dialect_plugins():
            passes.extend(plugin.pipeline_passes())
        passes.extend(_CORE_PIPELINE_PASSES)
        return "builtin.module(" + ",".join(passes) + ")"

    def preprocess_pipeline(self, pipeline: str, arch: str) -> str:
        """Append the architecture option (``<pass_sm_arch_name>=<arch>``) to
        the ``<PREFIX>_PIPELINE`` string, merging into an existing ``{...}``
        option block when the pipeline has one."""
        options = {
            self.pass_sm_arch_name: arch,
        }

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

    def _generate_execution_arguments_for_known_types(
        self,
        arg: Any,
        arg_spec: Any,
        arg_name: str,
        i: int,
        fop_args: list[Any],
        iv_block_args: int,
    ) -> tuple[list[Any], int]:
        """
        Generate MLIR arguments for known types.

        Sub-DSLs can override this method to handle types that are not
        natively supported by the Base DSL.
        """
        ir_arg = []
        # ``self.funcBody`` is the function currently being traced; it supplies
        # the owning-function context that ``is_argument_meta`` uses to
        # detect reserved ``self``/``cls`` parameters.
        if is_argument_meta(
            arg,
            arg_spec,
            arg_name,
            i,
            self.funcBody,
        ):
            ir_arg.append(arg)

        return ir_arg, iv_block_args

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

            ir_arg, iv_block_args = self._generate_execution_arguments_for_known_types(
                arg, arg_spec, arg_name, idx, fop_args, iv_block_args
            )

            if not ir_arg:
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
            else:
                ir_arg = ir_arg[0]

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

    def _generate_jit_func_args_for_known_types(
        self,
        func: Any,
        arg: Any,
        arg_name: str,
        arg_spec: Any,
        arg_index: int,
        *,
        is_host: bool = True,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        """
        Generate JIT function arguments for known types.

        Sub-DSLs can override this method to handle types that are not
        natively supported by the Base DSL. ``None`` triples mark a Meta
        argument, which the trace specialises on.
        """

        jit_arg_type: list[Any] | None = []
        jit_arg_attr: list[Any] | None = []
        jit_exec_arg: list[Any] | None = []

        if is_argument_meta(
            arg,
            arg_spec,
            arg_name,
            arg_index,
            func,
        ):
            jit_exec_arg = jit_arg_type = jit_arg_attr = None

        return jit_exec_arg, jit_arg_type, jit_arg_attr

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
        spec_ty: Any,
        arg_name: str,
        arg_index: int,
        func: Any,
        function_name: str,
        is_host: bool,
        compile_only: bool,
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
            phase_label=("JitArgument" if is_host else "DynamicExpression"),
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
                    phase_label="JitArgument",
                    function_name=function_name,
                )
            if entry is None or entry.marshal is None:
                raise DSLUserCodeError(
                    DiagId.ARG_NOT_MARSHALABLE,
                    arg_name=arg_name,
                    arg_type=type(value).__name__,
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
        compile_only: bool = False,
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

                (
                    jit_exec_arg,
                    jit_arg_type,
                    jit_arg_attr,
                ) = self._generate_jit_func_args_for_known_types(
                    func, arg, arg_name, spec_ty, i, is_host=is_host
                )

                if jit_arg_type is not None and len(jit_arg_type) == 0:
                    exec_args, arg_types, arg_attrs = self._flatten_jit_arg(
                        arg, arg_name, i, function_name, is_host=is_host
                    )
                    jit_exec_arg.extend(exec_args)  # type: ignore[union-attr]
                    jit_arg_type.extend(arg_types)
                    jit_arg_attr.extend(arg_attrs)  # type: ignore[union-attr]

                    self._check_unsupported_jit_arg(
                        arg=arg,
                        spec_ty=spec_ty,
                        arg_name=arg_name,
                        arg_index=i,
                        func=func,
                        function_name=function_name,
                        is_host=is_host,
                        compile_only=compile_only,
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
        compile_only: bool = False,
    ) -> JitFuncArgs:
        """Convert input arguments to the MLIR function signature and the
        marshalled execution arguments."""

        result = self._generate_jit_func_args(
            func,
            function_name,
            args,
            kwonlyargs,
            sig,
            is_host=True,
            compile_only=compile_only,
        )

        if len(result.values) != len(result.types):
            raise DSLRuntimeError(
                "expects the same number of arguments and function parameters",
                context={"values": len(result.values), "types": len(result.types)},
            )

        return result

    def _check_buffer_kinds(
        self,
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        adapted_args: list[Any],
        sig: inspect.Signature,
        function_name: str,
    ) -> None:
        """Reject a buffer whose side does not match what the trace did.

        A host buffer (a numpy array, a CPU tensor) passed to a call whose
        trace launched a kernel is the plugin's ``LAUNCH_HOST_BUFFER``; a
        device buffer passed to a call whose trace launched nothing is
        ``ARG_DEVICE_BUFFER_ON_HOST``. A bare address (kind ``unknown``) is
        never checked.
        """
        launched = self.launch_inner_count > 0
        input_args = [*args, *kwonlyargs.values()]
        for name, original, adapted in zip(sig.parameters, input_args, adapted_args):
            arg = original if adapted is None else adapted
            if not tree_utils.contains_leaf(arg):
                continue
            values, _, _ = tree_utils.tree_flatten(arg, return_ir_values=False)
            for value in values:
                kind = getattr(value, "kind", "unknown")
                if not isinstance(value, t.Pointer) or kind == "unknown":
                    continue
                if kind == "host" and launched:
                    raise DSLUserCodeError(
                        self._launch_diag("LAUNCH_HOST_BUFFER"),
                        arg_name=name,
                        arg_type=type(original).__name__,
                        function_name=function_name,
                    )
                if kind == "device" and not launched:
                    raise DSLUserCodeError(
                        DiagId.ARG_DEVICE_BUFFER_ON_HOST,
                        arg_name=name,
                        arg_type=type(original).__name__,
                        function_name=function_name,
                    )

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
            caller_locs=(),
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
        loc = ir.Location.name(location.function_name, childLoc=file_loc)

        if location.caller_locs:
            caller_ir_locs = [
                ir.Location.file(fn, ln, 0) for fn, ln in location.caller_locs
            ]
            loc = ir.Location.callsite(loc, caller_ir_locs)

        return loc

    def _enter_loc_tracebacks(self) -> Any:
        """Enter the loc-tracebacks context if enabled, returning it (or None).

        Enable via <PREFIX>_LOC_TRACEBACKS=N (e.g. 128 for full stacks).
        """
        depth = self.envar.loc_tracebacks
        if depth <= 0:
            return None
        try:
            loc_tracebacks = ir.loc_tracebacks(max_depth=depth)
        except (ValueError, TypeError, AttributeError):
            # Bindings without the feature, or an unsupported depth.
            return None
        loc_tracebacks.__enter__()
        return loc_tracebacks

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
        Compile and JIT an MLIR module through the compiler provider.

        :return: The ``ExecutionEngine`` holding the compiled module.
        """
        # This method is also a direct entry point (not only reached through
        # generate_mlir), so it owns a compile boundary of its own; when
        # nested inside generate_mlir the depth counter folds it away.
        phase_profiler.begin_compile()
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
            )
        except DSLBaseError:
            raise
        except Exception as e:
            raise DSLRuntimeError(
                "compilation failed", context={"function": function_name}, cause=e
            ) from e
        finally:
            phase_profiler.finish_compile()

    def jit_lowered_module(
        self, module: ir.Module, shared_libs: list[str], function_name: str = ""
    ) -> Any:
        """Build the ``ExecutionEngine`` of an already lowered ``module`` (a file
        cache hit): the pass pipeline is skipped."""
        phase_profiler.begin_compile()
        try:
            return self._compiler.jit(module, shared_libs=shared_libs)
        except DSLBaseError:
            raise
        except Exception as e:
            raise DSLRuntimeError(
                "compilation failed", context={"function": function_name}, cause=e
            ) from e
        finally:
            phase_profiler.finish_compile()

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

    def get_return_types(self) -> list[Any]:
        """
        Get the return types of the host function (DkgDSL's seam).

        The base does not consult it: the entry is created as ``void`` and,
        once the body is traced, its function type is rewritten with the one
        result the return packs (:meth:`_return_values`).
        """
        return []

    def _return_values(
        self, helper: _HostGenHelper, result: Any, sig: inspect.Signature, loc: Any
    ) -> tuple[list[ir.Value], _ResultSpec | None]:
        """Turn the traced return value into what the entry returns.
        Numeric leaves and frozen records/tuples/``@struct``s of them are
        accepted; the host helper decides how the leaves travel (the LLVM
        world packs several into one ``!llvm.struct``).
        """
        if result is None:
            if sig.return_annotation not in (inspect.Signature.empty, None):
                raise DSLUserCodeError(DiagId.TYPE_RETURN_NONE)
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
        ret_values, slot = helper.pack_results(
            values, [leaf.prototype for leaf in leaves], loc=loc
        )
        return ret_values, _ResultSpec(treedef, slot)

    def _result_from_ctypes(self, spec: _ResultSpec, raw: Any) -> Any:
        """Rebuild the Python return value from the filled result slot."""
        leaves = [
            leaf
            for _, leaf in tree_utils.tree_leaves(spec.treedef)
            if not leaf.is_none and not leaf.is_meta
        ]
        values = iter(
            self._host_gen_helper().unpack_result(
                spec.slot, raw, [leaf.prototype for leaf in leaves]
            )
        )
        return self._restore_tree(spec.treedef, lambda leaf: next(values))

    # =========================================================================
    # Tracing
    # =========================================================================

    def generate_original_ir(
        self,
        funcBody: Callable[..., Any],
        function_name: str,
        func_types: list[Any],
        arg_attrs: list[Any],
        gpu_module_attrs: dict[str, Any],
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        location: DSLLocation | None = None,
        *,
        no_cache: bool = False,
        pipeline: str = "",
        extra_link_libs: tuple[str, ...] = (),
    ) -> tuple[ir.Module, str | None, Any, _ResultSpec | None]:
        """Trace ``funcBody`` into a ``func.func`` host entry of a fresh module.

        :return: The verified module, its hash (None under ``no_cache``), the
            trace's Python result and the result slot description
        """

        def build_ir_module() -> tuple[ir.Module, Any, _ResultSpec | None]:
            loc = self.get_ir_location(location)
            module = ir.Module.create(loc=loc)

            with ir.InsertionPoint(module.body):
                # The kernel container is built up front and pruned after the
                # trace when it received no kernel.
                self.current_module = module
                self.kernel_container = (
                    self.kernel_gen_helper.build_container(gpu_module_attrs, loc=loc)
                    if self.kernel_gen_helper is not None
                    else None
                )

                # The host entry is the dialect plugin's (the LLVM world: a
                # ``func.func`` with the C interface); its result types are
                # set after the trace.
                helper = self._host_gen_helper()
                entry_block = helper.generate_func_op(
                    function_name, list(func_types), list(arg_attrs), loc=loc
                )
                log().debug("Generated Function OP [%s]", helper.func_op)
                with ir.InsertionPoint(entry_block):
                    ir_args, ir_kwargs = self.generate_execution_arguments(
                        args, kwonlyargs, entry_block, sig
                    )
                    # Call user function body
                    try:
                        with self._track_deferred_kernel_launches():
                            result = funcBody(*ir_args, **ir_kwargs)
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
                            var=getattr(name_error, "name", None)
                            or f"<in {funcBody.__name__}>",
                            function_name=funcBody.__name__,
                        ) from name_error
                    ret_values, result_spec = self._return_values(
                        helper, result, sig, loc
                    )
                    helper.generate_return(ret_values, loc=loc)
                if self.kernel_gen_helper is not None:
                    self.kernel_gen_helper.prune_empty_containers(module)

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
            for plugin in self.plugins:
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

        build_and_finalize = phase_profiler.profile_build(build_and_finalize)
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
        func_type: Callable[..., JitCompiledFunction] = JitCompiledFunction,
        *,
        result_spec: _ResultSpec | None = None,
        extra_link_libs: tuple[str, ...] = (),
        funcBody: Callable[..., Any] | None = None,
    ) -> JitCompiledFunction:
        """Compile ``module`` and cache the resulting :class:`JitCompiledFunction`.

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
        # The lookup materializes the JIT'd code (ORC compiles lazily), so it
        # belongs to the jit phase together with engine construction.
        lookup = phase_profiler.timed("jit")(
            self._maybe_profile(lookup_packed_function)
        )
        capi_func = lookup(engine, function_name)

        jit_function = func_type(
            module,
            engine,
            capi_func,
            sig,
            function_name,
            self.kernel_info,
            jit_time_profiling=self.envar.jit_time_profiling,
            result_ctype=result_spec.slot if result_spec is not None else None,
            has_kernels=self.num_kernels > 0,
        )
        jit_function.result_spec = result_spec  # type: ignore[attr-defined]
        if isinstance(jit_function, JitCompiledFunction):
            jit_function.execution_args.set_adapter_scope(self._jit_arg_adapter_scope)
        # An export plugin wraps the function (its symbols are in the engine now).
        for plugin in self.plugins:
            jit_function = plugin.wrap_compiled_function(self, jit_function)

        if not no_cache:
            self.jit_cache.set(module_hash, jit_function, funcBody)
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
        self.kernel_info = OrderedDict()
        self.launch_inner_count = 0
        self.num_kernels = 0
        self.decorator_location = None
        self.kernel_container = None

    def generate_mlir(
        self,
        funcBody: Callable[..., Any],
        function_name: str,
        gpu_module_attrs: dict[str, Any],
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        sig: inspect.Signature,
        pipeline: str | None,
        no_cache: bool,
        compile_only: bool,
        location: DSLLocation | None = None,
        extra_link_libs: tuple[str, ...] = (),
    ) -> Any:
        """Trace ``funcBody`` into an MLIR module, compile it through the
        compiler provider (or take the cached function) and run it.

        :return: The trace's Python result under ``<PREFIX>_DRYRUN``, the
            compiled function under ``compile_only``, else the call's result
        """
        # Check current DSL build supports target arch
        self._is_supported_arch()

        # The remark session owns the context's remark engine for the whole
        # call, so trace-time remarks and the passes' remarks share one stream.
        with ir.Context() as ctx, self.get_ir_location(location), self._remark_session(
            ctx
        ) as remark_session:
            # Each MLIR context keeps a thread pool alive; cached compilations
            # keep their context, so threading is disabled to bound the count.
            ctx.enable_multithreading(False)
            for plugin in self._dialect_plugins():
                plugin.register_dialects(ctx)
            self.collected_remarks = remark_session.remarks
            # Optional: capture full Python call stacks on every MLIR op.
            loc_tracebacks = self._enter_loc_tracebacks()

            phase_profiler.begin_compile()
            try:
                # Convert input arguments to MLIR arguments
                mlir_func_args = self.generate_mlir_function_types(
                    funcBody, function_name, args, kwonlyargs, sig, compile_only
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
                    funcBody,
                    function_name,
                    mlir_func_args.types,
                    mlir_func_args.attributes,
                    gpu_module_attrs,
                    trace_args,
                    trace_kwargs,
                    sig,
                    location=location,
                    no_cache=no_cache,
                    pipeline=pipeline,
                    extra_link_libs=extra_link_libs,
                )
                self._check_buffer_kinds(
                    args, kwonlyargs, adapted_args, sig, function_name
                )

                # dryrun is used to only generate IR
                if self.envar.dryrun:
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
                        funcBody=funcBody,
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
                phase_profiler.finish_compile()

        # If compile_only is set, bypass execution return the jit_executor directly
        if compile_only:
            # The returned function is specialized on this call's Meta values;
            # a later call with other ones is a diagnostic, not a silent reuse.
            if isinstance(jit_function, JitCompiledFunction):
                jit_function.execution_args.set_meta_values(
                    {
                        param.name: arg
                        for param, arg in zip(sig.parameters.values(), args)
                        if self._is_meta_argument(arg, param.annotation)
                    }
                )
                # The host shape of every argument, adapted as the trace saw it.
                jit_function.execution_args.set_shapes(
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
        self.funcBody = original_function
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
        self, sig: inspect.Signature, func_name: str, *args: Any, **kwargs: Any
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
                DiagId.CALL_SIGNATURE_MISMATCH,
                provided=len(args),
                provided_kw=len(kwargs),
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
        """Raise ``CALL_MISSING_ARG`` for a parameter without default that
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
                    DiagId.CALL_MISSING_ARG,
                    missing=f"`{param.name}`",
                    function_name=function_name,
                )
        return has_varargs

    def _get_signature(self, funcBody: Callable[..., Any]) -> inspect.Signature:
        """
        Returns the signature for a given function, handling PEP-563
        (postponed evaluation of type annotations) via eval_str=True.
        """
        try:
            return inspect.signature(funcBody, eval_str=True)
        except NameError as e:
            raise DSLUserCodeError(
                DiagId.SCOPE_UNBOUND_NAME_IN_TRACE,
                var=getattr(e, "name", None) or "<annotation>",
                function_name=getattr(funcBody, "__name__", "<function>"),
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
    # Host entry: @jit
    # =========================================================================

    @dataclass
    class _CompilationSetup:
        """Shared pre-IR-generation state of a host compilation."""

        function_name: str
        pipeline: str | None
        gpu_module_attrs: dict[str, Any]
        no_cache: bool
        extra_link_libs: tuple[str, ...]
        compile_only: bool
        canonicalized_args: tuple[Any, ...]
        canonicalized_kwargs: dict[str, Any]
        sig: inspect.Signature
        location: DSLLocation | None

    def _prepare_compilation(
        self, funcBody: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> "_CompilationSetup":
        """Extract kwargs, canonicalize args and mangle the name.

        The call keywords ``pipeline``, ``gpu_module_attrs``, ``no_cache``,
        ``extra_link_libs`` and ``compile_only`` are the DSL's, popped before
        the remaining arguments bind to the function's signature.
        """
        function_name = funcBody.__name__
        self.funcBody = funcBody

        pipeline = kwargs.pop("pipeline", None)
        gpu_module_attrs = kwargs.pop("gpu_module_attrs", {})
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
        sig = self._get_signature(funcBody)

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
            gpu_module_attrs=gpu_module_attrs,
            no_cache=no_cache,
            extra_link_libs=extra_link_libs,
            compile_only=compile_only,
            canonicalized_args=canonicalized_args,
            canonicalized_kwargs=canonicalized_kwargs,
            sig=sig,
            location=self.decorator_location,
        )

    def _func(self, funcBody: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """The ``@jit`` executor: one call of the decorated ``funcBody``.

        1. Translates the arguments (numpy arrays -> ``Pointer``, ``float`` ->
           ``f32``, ...) and traces the body into the host entry
        2. Compiles and JITs the MLIR module (cached)
        3. Invokes the compiled function and rebuilds its result
        """
        # Keep this guard even though jit_wrapper also enters the DSL context:
        # compile/device paths may call _func directly.
        with active_dsl(self):
            return self._func_impl(funcBody, *args, **kwargs)

    def _func_impl(
        self, funcBody: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if ir.Context.current is not None and ir.InsertionPoint.current is not None:
            # A nested call inside a trace runs its (preprocessed) body inline.
            return funcBody(*args, **kwargs)

        setup = self._prepare_compilation(funcBody, *args, **kwargs)

        log().debug("Generating MLIR for function '%s'", setup.function_name)
        return self.generate_mlir(
            funcBody,
            setup.function_name,
            setup.gpu_module_attrs,
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
    # Device entry: @kernel
    # =========================================================================

    def generate_kernel_operands_and_types(
        self,
        kernel_func: Callable[..., Any],
        kernel_name: str,
        signature: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Generate the operands and types for the kernel function
        """
        log().debug(
            "Processing GPU kernel call in [%s] mode",
            (
                f"Only {self.device_jit_decorator_name}"
                if self.device_compilation_only
                else f"{self.host_jit_decorator_name} + {self.device_jit_decorator_name}"
            ),
        )

        if self.device_compilation_only:
            return [], [], []

        _kernel_args = self._generate_jit_func_args(
            kernel_func, kernel_name, args, kwargs, signature, is_host=False
        )
        if (
            not len(_kernel_args.values)
            == len(_kernel_args.types)
            == len(_kernel_args.attributes)
        ):
            raise DSLRuntimeError(
                "Size of kernel_operands, kernel_arg_types and kernel_arg_attrs must be equal"
            )
        return _kernel_args.values, _kernel_args.types, _kernel_args.attributes

    def _collect_kernel_launch_args(
        self,
        kernel_name: str,
        kwargs: dict[str, Any],
        requiredArgs: list[str],
        optionalArgs: list[str],
    ) -> tuple[Any, Any]:
        """Pop the named launch args out of ``kwargs`` into required/optional
        namedtuples (None when the corresponding name list is empty)."""

        def extract_args(argNames: list[str], assertIfNone: bool = False) -> list[Any]:
            extracted = []
            for name in argNames:
                value = kwargs.pop(name, None)
                if assertIfNone and value is None:
                    raise DSLRuntimeError(
                        f"the launch of `{kernel_name}` misses its `{name}` argument"
                    )
                extracted.append(value)
            return extracted

        RequiredArgs = namedtuple("RequiredArgs", requiredArgs)  # type: ignore[misc]
        req_args = (
            RequiredArgs._make(extract_args(requiredArgs, assertIfNone=True))
            if requiredArgs
            else None
        )
        OptionalArgs = namedtuple("OptionalArgs", optionalArgs)  # type: ignore[misc]
        opt_args = (
            OptionalArgs._make(extract_args(optionalArgs)) if optionalArgs else None
        )
        return req_args, opt_args

    def kernel_launcher(self, *dargs: Any, **dkwargs: Any) -> Any:
        """
        Decorator generating a kernel: the kernel function op with its traced
        body inside the ``gpu.module`` and the launch op at the call site.

        Decorator keywords (default in ``<>``):

        - ``requiredArgs <[]>``: launch arguments that must be present,
          collected as a namedtuple
        - ``optionalArgs <[]>``: launch arguments that may be present,
          collected as a namedtuple
        - ``unitAttrNames <[]>``: names of ``ir.UnitAttr`` to set on the
          kernel function op
        - ``valueAttrDict <{}>``: names and values of ``ir.Attribute`` to set
          on the kernel function op
        - ``kernelGenHelper <None>``: the mandatory kernel generation helper
          class (derived from :class:`_KernelGenHelper`)

        :return: The decorated function; calling it returns a
            :class:`KernelReturns` ``(kernel_func_ret, launch_op_ret)`` and
            the mangled kernel name
        """

        def decorator(funcBody: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(funcBody)
            def kernel_wrapper(*args: Any, **kwargs: Any) -> Any:
                requiredArgs = dkwargs.get("requiredArgs", [])
                optionalArgs = dkwargs.get("optionalArgs", [])
                unitAttrNames = dkwargs.get("unitAttrNames", [])
                valueAttrDict = dkwargs.get("valueAttrDict", {})
                kernelGenHelper = dkwargs.get("kernelGenHelper", None)
                launch_loc = kwargs.pop("_launch_loc", None)

                kernel_name = funcBody.__name__
                signature = self._get_signature(funcBody)
                self.funcBody = funcBody

                # Give each kernel a unique name. (The same kernel may be
                # called multiple times, resulting in multiple kernel traces.)
                kernel_name = f"kernel_{self.mangle_name(kernel_name, args, signature)}_{self.num_kernels}"
                self.num_kernels += 1

                # Pop the named launch args out of kwargs into the
                # required/optional tuples before the kernel's signature binds.
                req_args, opt_args = self._collect_kernel_launch_args(
                    kernel_name, kwargs, requiredArgs, optionalArgs
                )
                if kernelGenHelper is None:
                    raise DSLRuntimeError(
                        "kernelGenHelper should be explicitly specified!"
                    )

                # Get bound arguments
                bound_args = self._get_function_bound_args(
                    signature, kernel_name, *args, **kwargs
                )

                # check arguments
                self._check_arg_count(signature, bound_args, kernel_name)

                # Canonicalize the input arguments
                canonicalized_args, canonicalized_kwargs = self._canonicalize_args(
                    bound_args
                )

                (
                    kernel_operands,
                    kernel_types,
                    kernel_arg_attrs,
                ) = self.generate_kernel_operands_and_types(
                    funcBody,
                    kernel_name,
                    signature,
                    canonicalized_args,
                    canonicalized_kwargs,
                )

                loc = self.get_ir_location()
                with self._enter_kernel_container():
                    log().debug("Generating device kernel")
                    if self.device_compilation_only:
                        # Convert input arguments to MLIR arguments
                        _kernel_mlir_args = self.generate_mlir_function_types(
                            funcBody,
                            kernel_name,
                            canonicalized_args,
                            canonicalized_kwargs,
                            signature,
                        )
                        kernel_types = _kernel_mlir_args.types
                        kernel_arg_attrs = _kernel_mlir_args.attributes

                    helper = kernelGenHelper()
                    fop = helper.generate_func_op(
                        kernel_types, kernel_arg_attrs, kernel_name, loc
                    )
                    log().debug("Kernel function op: %s", fop)
                    for attr in unitAttrNames:
                        fop.attributes[attr] = ir.UnitAttr.get()
                    for key, val in valueAttrDict.items():
                        fop.attributes[key] = val

                    fop.sym_visibility = ir.StringAttr.get("public")
                    with ir.InsertionPoint(helper.get_func_body_start()):
                        ir_args, ir_kwargs = self.generate_execution_arguments(
                            canonicalized_args, canonicalized_kwargs, fop, signature
                        )
                        log().debug(
                            "IR arguments - args: %s ; kwargs: %s", ir_args, ir_kwargs
                        )
                        kernel_ret = funcBody(*ir_args, **ir_kwargs)
                        if kernel_ret is not None:
                            raise DSLUserCodeError(
                                DiagId.TYPE_RETURN_MISMATCH,
                                got=f"a `{type(kernel_ret).__name__}`",
                                detail=" from a kernel",
                            )
                        helper.generate_func_ret_op()

                # The call site: the launch op referring to the kernel symbol.
                kernel_sym = helper.kernel_symbol(kernel_name)
                setattr(funcBody, "_dsl_kernel_sym", kernel_sym)
                setattr(funcBody, "_dsl_kernel_name", kernel_name)
                launch_ret = helper.generate_launch_op(
                    kernelSym=kernel_sym,
                    kernelOperands=kernel_operands,
                    requiredArgs=req_args,
                    optionalArgs=opt_args,
                    loc=loc,
                    launch_loc=launch_loc or loc,
                )

                result = KernelReturns(
                    kernel_func_ret=kernel_ret, launch_op_ret=launch_ret
                )
                log().debug("Kernel result: %s, kernel name: %s", result, kernel_name)
                return result, kernel_name

            return kernel_wrapper

        if len(dargs) == 1 and callable(dargs[0]):
            return decorator(dargs[0])
        else:
            return decorator
