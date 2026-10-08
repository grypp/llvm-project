# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The plugins of :class:`BaseDSL`: the roles a DSL fills, the families that
extend it, and the record naming them.

A plugin fills or extends a core role; a module emits ops. The core defines
what it calls, and every plugin says where it connects by its base class:

* the **roles**, one plugin each, which the core calls on its own behalf:
  ``type_ops`` (:class:`TypeOpsPlugin`, the MLIR types of the core dtypes and
  the ops behind their operators), ``func_entry`` (:class:`FuncEntryPlugin`,
  the host entry of a ``@jit`` function and its result slot),
  ``ast_preprocessor`` (:class:`ASTPreprocessorPlugin`, the rewrite of native
  control flow and the executors it calls) and ``compiler``
  (:class:`CompilerPlugin`, the pipeline run, the engine, the invocation);
* the **families**, any number each, which extend the DSL at one fixed point:
  ``decorators`` (:class:`DecoratorPlugin`: a decorator such as ``@kernel``
  and its launcher, with hooks around every trace) and ``adapters``
  (:class:`AdapterPlugin`: the host boundary in both directions, host objects
  becoming arguments and the compiled entry exposed through another ABI).

Every plugin shares one lifecycle (``available``, ``install``,
``shared_libs``, ``register_dialects``); a role or family adds its contract
on top, and each hook of the core lives on exactly one of them. A DSL names
its plugins once, as a :class:`Plugins` record on the class::

    class MyDSL(BaseDSL):
        plugins = Plugins(
            type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
            func_entry=func.Entry(),
            ast_preprocessor=scf.ASTPreprocessor(),
            compiler=execution_engine.Compiler(),
            decorators=[gpu.Kernels()],
            adapters=[numpy.NumpyPlugin(), pytorch.PyTorchPlugin(), tvm_ffi.TvmFfiPlugin()],
        )

``BaseDSL.__init_subclass__`` installs the decorators of the record on the
class (``@MyDSL.kernel``); ``BaseDSL.__init__`` resolves the record once per
instance (:meth:`Plugins.resolve`): a plugin whose ``available()`` is False
is dropped and remembered in ``dsl.unavailable_plugins``, the others are
copied and installed in record order (roles, then decorators, then
adapters). A variant DSL is ``dataclasses.replace(Base.plugins,
decorators=(), adapters=())``. The base names nothing: there is no default
world in the core, and no decorator but ``@jit``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from ... import ir
from .common import DSLRuntimeError
from .mlir_op import OpEmitter

if TYPE_CHECKING:
    from .dsl import BaseDSL

__all__ = [
    "ASTPreprocessorPlugin",
    "AdapterPlugin",
    "CompilerPlugin",
    "DecoratorPlugin",
    "FuncEntryPlugin",
    "Plugin",
    "Plugins",
    "TypeOpsPlugin",
]


class Plugin:
    """The lifecycle every plugin shares; a role or family base adds the contract.

    ``role`` names the record field a subclass fills: a role (``"type_ops"``,
    ...) or a family (``"decorators"``, ``"adapters"``). A bare
    ``Plugin`` fits no field. A leaf type a plugin brings registers at import
    of the plugin's module (``register_leaf``), never in :meth:`install`, which
    runs once per DSL instance.
    """

    #: The record field this plugin fills; set by the role and family bases.
    role: ClassVar[str] = ""
    name: str = ""
    dsl: BaseDSL | None = None

    @classmethod
    def available(cls) -> bool:
        """Whether the plugin can serve here: its optional dependency is
        importable, its dialect bindings are built, its runtime library is
        findable, ... Probed, never imported, so an absent dependency costs
        nothing; a plugin that is not available is dropped from the record."""
        return True

    def install(self, dsl: BaseDSL) -> None:
        """Bind to ``dsl``: set ``self.dsl`` and reach its seams (an
        environment variable such as ``<PREFIX>_ARCH``, a check the plugin
        runs once per DSL instance)."""
        self.dsl = dsl

    def shared_libs(self) -> list[str]:
        """Library paths handed to the ``ExecutionEngine``; never bound here."""
        return []

    def pipeline_options(self) -> dict[str, str]:
        """Pass options merged into a ``<PREFIX>_PIPELINE`` override, option
        name to value (the gpu kernels plugin: its ``chip_option`` set to the
        architecture). A DSL's own ``pipeline()`` spells its options itself,
        so the default has none."""
        return {}

    def register_dialects(self, context: ir.Context) -> None:
        """Register the plugin's dialects on the trace ``context``.

        Called by ``BaseDSL.generate_mlir`` right after the trace context is
        created and before any op is built (after the DSL's own
        ``register_dialects``); upstream dialects are on every context already,
        so the default does nothing.
        """

    def _unsupported(self, what: str) -> Any:
        raise DSLRuntimeError(
            f"`{type(self).__name__}` ({self.role or 'plugin'}) does not implement `{what}`"
        )


# =============================================================================
# The roles: one plugin each, called by the core on its own behalf
# =============================================================================


class TypeOpsPlugin(Plugin, OpEmitter):
    """The ``type_ops`` role: the MLIR types of the core dtypes and the ops
    behind their operators.

    The :class:`OpEmitter` hooks answer, for ``Numeric``, ``Vector`` and
    ``Pointer``, which MLIR type a dtype has (``mlir_type``/``scalar_type``,
    ``vector_type``/``vector_shape``, ``pointer_type``/``pointer_space``) and
    which op implements each operation (``add`` ... ``cmp``, ``from_elements``
    ... ``reduce``, ``load`` ... ``addrspacecast``). The type promotion itself
    runs in the core before any hook is called; a type-ops plugin is stateless.
    """

    role: ClassVar[str] = "type_ops"


class FuncEntryPlugin(Plugin):
    """The ``func_entry`` role: the host entry of a ``@jit`` function.

    The core decides what is returned (the numeric leaves of the traced return
    value); the plugin decides the function op, how several leaves travel back
    to the host and how a raw slot value becomes each leaf's Python value. The
    plugin is stateless: ``generate_func_op`` returns the op it built and
    ``generate_return`` takes it back.
    """

    role: ClassVar[str] = "func_entry"

    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> tuple[Any, ir.Block]:
        """Create the entry ``name`` with ``arg_types`` and no results; return
        ``(func_op, entry_block)``."""
        return self._unsupported("generate_func_op")

    def generate_return(self, func_op: Any, values: list[Any], loc: Any = None) -> None:
        """Emit the terminator returning ``values`` from ``func_op`` and fix
        the entry's result types."""
        self._unsupported("generate_return")

    def pack_results(
        self, values: list[Any], prototypes: list[Any], loc: Any = None
    ) -> tuple[list[Any], Any]:
        """The values the entry returns for the result leaves ``values`` (of
        dtypes ``prototypes``) and the host-side slot descriptor, None for no
        result."""
        return self._unsupported("pack_results")

    def unpack_result(self, slot: Any, raw: Any, prototypes: list[Any]) -> list[Any]:
        """The Python value of each result leaf from the filled slot ``raw``."""
        return self._unsupported("unpack_result")


class ASTPreprocessorPlugin(Plugin):
    """The ``ast_preprocessor`` role: Python syntax mapped onto the executors.

    It provides the preprocessor that rewrites native ``for``/``if``/``while``
    into region functions and the executors (the preprocessor's helper
    callbacks) that run or stage those regions at trace time.
    ``BaseDSL.__init__`` installs its executors and, when the DSL preprocesses,
    builds its preprocessor. ``@jit(preprocess=False)`` and
    ``<PREFIX>_AST_PREPROCESSOR=0`` still bypass the rewrite.
    """

    role: ClassVar[str] = "ast_preprocessor"
    #: The ``DSLPreprocessor`` (sub)class performing the rewrite; None for a
    #: plugin that only supplies executors.
    preprocessor_class: Any = None
    #: Reject a nested function that captures a variable and is called from a
    #: staged region (``SCOPE_CLOSURE_CAPTURE``); a plugin whose executors
    #: support such captures sets this False and the preprocessor emits no check.
    closure_check: bool = True

    def executors(self, dsl: BaseDSL) -> dict[str, Any]:
        """The keyword arguments of ``Executor.set_functions``: the DSL's
        ``loop_execute_range_dynamic``, ``if_dynamic``, ``while_dynamic``,
        ``compare_executor``, ``builtin_redirector`` and ``ifexp_dynamic``."""
        return {}


class _NoRemarkSession:
    """The remark session of a DSL without a compiler plugin: collects nothing."""

    def __init__(self) -> None:
        self.remarks: list[dict[str, Any]] = []

    def __enter__(self) -> "_NoRemarkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class CompilerPlugin(Plugin):
    """The ``compiler`` role: lowering and invocation of a traced module.

    :meth:`compile` runs the pipeline, :meth:`jit` builds the engine,
    :meth:`compile_and_jit` both, :meth:`load` binds an entry into a callable,
    :meth:`remark_session` collects remarks and :meth:`print_ir_after_passes`
    prints IR on a clone. Lowering and invocation are both the plugin's, so a
    backend that does not produce an ``ExecutionEngine`` replaces the whole
    object. A DSL without a compiler traces only.
    """

    role: ClassVar[str] = "compiler"

    def remark_session(
        self,
        context: ir.Context,
        *,
        remark_filter: str = "",
        remark_policy: str = "all",
        remark_output: str = "",
    ) -> Any:
        """A remark session for ``context``; ``with session:`` owns the engine
        and the diagnostic handler, ``session.remarks`` holds the records."""
        return _NoRemarkSession()

    def compile(self, module: ir.Module, pipeline: str, **options: Any) -> None:
        """Run ``pipeline`` on ``module`` in place."""
        self._unsupported("compile")

    def jit(self, module: ir.Module, **options: Any) -> Any:
        """Build the engine of an already lowered ``module``."""
        return self._unsupported("jit")

    def compile_and_jit(self, module: ir.Module, pipeline: str, **options: Any) -> Any:
        """Lower ``module`` and build its engine."""
        return self._unsupported("compile_and_jit")

    def load(
        self,
        module: ir.Module,
        engine: Any,
        function_name: str,
        signature: Any,
        **options: Any,
    ) -> Any:
        """Bind the compiled entry ``function_name`` into a callable."""
        return self._unsupported("load")

    def print_ir_after_passes(
        self, module: ir.Module, pipeline: str, **options: Any
    ) -> None:
        """Print the IR after each pass of ``pipeline`` on a clone of ``module``."""
        self._unsupported("print_ir_after_passes")


# =============================================================================
# The families: any number each, extending the DSL at one fixed point
# =============================================================================


class DecoratorPlugin(Plugin):
    """The ``decorators`` family: a decorator and what calling it does.

    ``decorators`` returns the decorators the plugin adds to the DSL class;
    ``BaseDSL.__init_subclass__`` installs them (``@MyDSL.kernel``). Build one
    with ``dsl_cls.make_decorator(name, on_call)``: the core wrapper handles
    the lazy instance, the AST preprocessing and the active-DSL context and
    hands the call to ``on_call(dsl, func, *args, **kwargs)``, the plugin's
    launcher. The hooks run around every ``@jit`` trace, so the plugin owns
    its per-trace state (a kernel container, the launches it expects) and its
    rules at the host boundary; ``finish_compiled_function`` records what the
    trace did on the compiled function.
    """

    role: ClassVar[str] = "decorators"

    def decorators(self, dsl_cls: type) -> dict[str, Callable[..., Any]]:
        """The decorators this plugin adds to the DSL class, by name."""
        return self._unsupported("decorators")

    def before_trace(
        self, dsl: BaseDSL, module: ir.Module, *, loc: Any, attrs: dict[str, Any]
    ) -> None:
        """Set up per-trace state before the host entry of a ``@jit`` trace is
        built, with the insertion point at the body of the fresh ``module``.
        ``attrs`` are the attributes the call passed as ``container_attrs``."""

    def after_trace(self, dsl: BaseDSL, module: ir.Module) -> None:
        """Finish per-trace state after the host body was traced, before the
        module is hashed (prune an empty container, report a kernel call that
        was never launched)."""

    def check_arguments(
        self,
        dsl: BaseDSL,
        sig: Any,
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        adapted: list[Any],
        function_name: str,
    ) -> None:
        """Validate the call's arguments against what the trace did (a host
        buffer handed to a call that launched a kernel)."""

    def finish_compiled_function(self, dsl: BaseDSL, jit_function: Any) -> None:
        """Record on the compiled function what the trace did (the kernels it
        launched), before any adapter wraps it."""


class AdapterPlugin(Plugin):
    """The ``adapters`` family: the host boundary, in both directions.

    Inbound, ``register`` adds the plugin's argument adapters to the registry
    when the DSL is constructed; from then on a host object of the adapted
    type (a ``torch.Tensor``, anything speaking DLPack) meets a ``Pointer[T]``
    parameter like a NumPy array does, and the plugin's dependency is imported
    only when such an object arrives. Outbound, a plugin exposing the compiled
    entry through another ABI (TVM-FFI) sees the traced module before it is
    hashed (``attach_to_module``), the lowered module before the engine is
    built (``after_lowering``) and the compiled function before it is cached
    (``wrap_compiled_function``); what it adds to a module is part of the
    cached artifact, and a file cache hit skips the pipeline and
    ``after_lowering`` alike. A plugin implements the half it needs.
    """

    role: ClassVar[str] = "adapters"

    def register(self, dsl: BaseDSL) -> None:
        """Register the plugin's argument adapters with ``JitArgAdapterRegistry``."""

    def attach_to_module(
        self,
        dsl: BaseDSL,
        module: ir.Module,
        function_name: str,
        sig: Any,
        trace_args: tuple[Any, ...],
        trace_kwargs: dict[str, Any],
    ) -> None:
        """Add to the traced ``module`` once the host entry ``function_name``
        is built, given the Python signature and the trace arguments (Meta
        values, or the adapted leaves) it was built for."""

    def after_lowering(self, dsl: BaseDSL, module: ir.Module) -> None:
        """Edit the lowered ``module`` after the pass pipeline, before the
        engine is built."""

    def wrap_compiled_function(self, dsl: BaseDSL, jit_function: Any) -> Any:
        """Replace or decorate the compiled function before it is cached; the
        default returns it unchanged."""
        return jit_function


# =============================================================================
# The record
# =============================================================================

_ROLE_TYPES: dict[str, type[Plugin]] = {
    "type_ops": TypeOpsPlugin,
    "func_entry": FuncEntryPlugin,
    "ast_preprocessor": ASTPreprocessorPlugin,
    "compiler": CompilerPlugin,
}
_FAMILY_TYPES: dict[str, type[Plugin]] = {
    "decorators": DecoratorPlugin,
    "adapters": AdapterPlugin,
}


def _describe(value: Any) -> str:
    if isinstance(value, type):
        return f"the class `{value.__name__}` (name an instance: `{value.__name__}()`)"
    return f"a `{type(value).__name__}`"


def _misplaced(plugin: Plugin, field_name: str) -> str:
    where = plugin.role
    if where in _ROLE_TYPES:
        hint = f"name it as `Plugins({where}=...)`"
    elif where in _FAMILY_TYPES:
        hint = f"list it under `{where}`"
    else:
        hint = (
            "a bare `Plugin` fits no field: subclass DecoratorPlugin or AdapterPlugin"
        )
    return (
        f"`{type(plugin).__name__}` does not belong in `Plugins.{field_name}`: {hint}"
    )


@dataclass(frozen=True)
class Plugins:
    """What a DSL is made of: one plugin per role, any number per family.

    Every role field takes an instance of the role's plugin class (or None for
    a role the DSL does not fill); the two family fields take sequences of
    their family's plugins. A plugin in the wrong field is an error. Iterating the
    record yields the role plugins in field order, then the families in order:
    the order plugins are installed and their shared hooks are called in.
    """

    type_ops: TypeOpsPlugin | None = None
    func_entry: FuncEntryPlugin | None = None
    ast_preprocessor: ASTPreprocessorPlugin | None = None
    compiler: CompilerPlugin | None = None
    decorators: Sequence[DecoratorPlugin] = ()
    adapters: Sequence[AdapterPlugin] = ()

    ROLES: ClassVar[tuple[str, ...]] = tuple(_ROLE_TYPES)
    FAMILIES: ClassVar[tuple[str, ...]] = tuple(_FAMILY_TYPES)

    def __post_init__(self) -> None:
        for role, base in _ROLE_TYPES.items():
            value = getattr(self, role)
            if value is not None and not isinstance(value, base):
                if isinstance(value, Plugin):
                    raise DSLRuntimeError(_misplaced(value, role))
                raise DSLRuntimeError(
                    f"`Plugins.{role}` takes a `{base.__name__}` instance, not {_describe(value)}"
                )
        for family, base in _FAMILY_TYPES.items():
            value = getattr(self, family)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise DSLRuntimeError(
                    f"`Plugins.{family}` takes a sequence of `{base.__name__}`, not {_describe(value)}"
                )
            plugins = tuple(value)
            for plugin in plugins:
                if not isinstance(plugin, base):
                    if isinstance(plugin, Plugin):
                        raise DSLRuntimeError(_misplaced(plugin, family))
                    raise DSLRuntimeError(
                        f"`Plugins.{family}` takes `{base.__name__}` instances, not {_describe(plugin)}"
                    )
            object.__setattr__(self, family, plugins)

    def __iter__(self):
        for role in self.ROLES:
            plugin = getattr(self, role)
            if plugin is not None:
                yield plugin
        for family in self.FAMILIES:
            yield from getattr(self, family)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def named(self, name: str) -> Plugin | None:
        """The plugin called ``name`` (a role or a family member), or None."""
        for plugin in self:
            if plugin.name == name:
                return plugin
        return None

    def resolve(self) -> tuple["Plugins", dict[str, str]]:
        """The record of one DSL instance: a shallow copy of every available
        plugin (a plugin instance named on two classes must not carry another
        DSL's back-reference), and the plugins dropped because ``available()``
        said no, as ``{field: class name}``."""
        unavailable: dict[str, str] = {}
        fields: dict[str, Any] = {}
        for role in self.ROLES:
            plugin = getattr(self, role)
            if plugin is None:
                fields[role] = None
            elif not type(plugin).available():
                unavailable[role] = type(plugin).__name__
                fields[role] = None
            else:
                fields[role] = copy.copy(plugin)
        for family in self.FAMILIES:
            kept: list[Plugin] = []
            for plugin in getattr(self, family):
                if type(plugin).available():
                    kept.append(copy.copy(plugin))
                else:
                    unavailable[
                        f"{family}[{plugin.name or type(plugin).__name__}]"
                    ] = type(plugin).__name__
            fields[family] = tuple(kept)
        return Plugins(**fields), unavailable
