# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plug-in protocols of :class:`BaseDSL`.

A :class:`Plugin` is an optional extension a DSL class lists
(``plugins = [TvmFfiPlugin()]``) without subclassing it: an argument adapter
(``pytorch``), an ABI exporter (``tvm_ffi``), a runtime library. A
:class:`DialectPlugin` is a plugin that brings a dialect or a target: it
registers dialects on the trace context, contributes passes ahead of the core
list and may provide kernel generation for ``@kernel`` (``gpu``). A
:class:`ASTPreprocessorPlugin` is the DSL's AST preprocessor: the preprocessor rewriting
native control flow and the executors its generated code calls.
``BaseDSL.__init__`` installs a shallow copy of each listed plugin as its last
step, so the subclass's own ``__init__`` still runs after them and wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ... import ir

if TYPE_CHECKING:
    from .dsl import BaseDSL

__all__ = ["DialectPlugin", "ASTPreprocessorPlugin", "Plugin"]


class Plugin:
    """One optional extension of a DSL instance; every default is a no-op.

    A leaf type the plugin brings registers at import of the plugin's module
    (``register_leaf``), never in :meth:`install`, which runs once per DSL
    instance.
    """

    name: str = ""
    dsl: BaseDSL | None = None

    @classmethod
    def available(cls) -> bool:
        """Whether a DSL class should list the plugin: its optional dependency
        is importable, its runtime library is findable, ... Probed, never
        imported, so an absent dependency costs nothing."""
        return True

    def install(self, dsl: BaseDSL) -> None:
        """Bind to ``dsl``: set ``self.dsl`` and reach its seams.

        A plugin may call ``dsl.executor.set_functions``, register an argument
        adapter or add its DiagId catalogue or environment-variable subclass.
        """
        self.dsl = dsl

    def shared_libs(self) -> list[str]:
        """Library paths handed to the ``ExecutionEngine``; never bound here."""
        return []

    def attach_to_module(
        self,
        dsl: BaseDSL,
        module: ir.Module,
        function_name: str,
        sig: Any,
        trace_args: tuple[Any, ...],
        trace_kwargs: dict[str, Any],
    ) -> None:
        """Add to the traced ``module`` before it is hashed and compiled.

        Called by ``BaseDSL.generate_original_ir`` once the host entry
        ``function_name`` is built, with the Python signature and the trace
        arguments (Meta values, or the adapted leaves) it was built for. What
        is added here is part of the cached artifact; an export plugin emits
        its ABI wrapper for the host entry here.
        """

    def wrap_compiled_function(self, dsl: BaseDSL, jit_function: Any) -> Any:
        """Replace or decorate the compiled function before it is cached.

        Called by ``BaseDSL.compile_and_cache`` with the engine built, so a
        plugin can look up its own symbols; the default returns the function
        unchanged.
        """
        return jit_function


class DialectPlugin(Plugin):
    """A plugin bringing a dialect or a target to the DSL.

    Besides the :class:`Plugin` hooks it registers dialects on the trace
    context, contributes passes that run ahead of the core list, and may
    provide the kernel generation helper behind ``@kernel``: the first listed
    dialect plugin with a ``kernel_gen_helper`` serves the DSL's kernels.
    """

    #: The ``_KernelGenHelper`` subclass emitting this target's kernel function,
    #: terminator and launch op; None for a dialect without kernels.
    kernel_gen_helper: Any = None
    #: The ``OpEmitter`` behind the type system: the SSA types of the core
    #: types and the ops implementing their operators. The first listed
    #: dialect plugin with an emitter owns the IR of ``Int32(6) + a``; without
    #: one the DSL class's ``default_dialects`` (``scf`` and the LLVM world)
    #: are installed.
    emitter: Any = None
    #: The ``_HostGenHelper`` subclass building the host entry of a ``@jit``
    #: function, its return and the result slot the host reads (the LLVM
    #: world: ``func.func`` with the C interface, results packed into one
    #: ``!llvm.struct``). The first dialect plugin with one serves ``@jit``.
    host_gen_helper: Any = None
    #: The compiler running the pass pipeline and executing the module (the
    #: LLVM world: ``Compiler(passmanager, execution_engine)``), used when the
    #: DSL's constructor is given none.
    compiler_provider: Any = None

    def register_dialects(self, context: ir.Context) -> None:
        """Register the plugin's dialects on the trace ``context``.

        Called by ``BaseDSL.generate_mlir`` right after the trace context is
        created and before any op is built; upstream dialects are on every
        context already, so the default does nothing.
        """

    def pipeline_passes(self) -> list[str]:
        """Passes composed ahead of the core pass list, in install order."""
        return []


class ASTPreprocessorPlugin(Plugin):
    """The AST preprocessor of a DSL.

    It provides the preprocessor that rewrites native ``for``/``if``/``while``
    into region functions and the executors (the preprocessor's helper callbacks)
    that run or stage those regions at trace time. The first listed AST preprocessor
    plugin serves the DSL: ``BaseDSL.__init__`` installs its executors and,
    when the DSL preprocesses, builds its preprocessor. ``@jit(preprocess=False)``
    and ``<PREFIX>_AST_PREPROCESSOR=0`` still bypass the rewrite.
    """

    #: The ``DSLPreprocessor`` (sub)class performing the rewrite; None for a
    #: plugin that only supplies executors.
    preprocessor_class: Any = None
    #: Reject a nested function that captures a variable and is called from a
    #: staged region (``SCOPE_CLOSURE_CAPTURE``); a preprocessor plugin whose executors
    #: support such captures sets this False and the preprocessor emits no check.
    closure_check: bool = True

    def executors(self, dsl: BaseDSL) -> dict[str, Any]:
        """The keyword arguments of ``Executor.set_functions``: the DSL's
        ``is_dynamic_expression``, ``loop_execute_range_dynamic``,
        ``if_dynamic``, ``while_dynamic``, ``compare_executor``,
        ``builtin_redirector`` and ``ifexp_dynamic``."""
        return {}
