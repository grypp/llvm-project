# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``compiler`` plugin over MLIR's pass manager and execution engine.

:class:`Compiler` compiles a traced module with ``mlir.passmanager`` and
executes it through ``mlir.execution_engine``; both modules are imported on
first use, so a DSL that names this plugin pays for the engine only when it
compiles, and ``available()`` says whether the bindings carry it at all. The
pass pipeline is data: ``BaseDSL.pipeline`` is the sub-DSL's pass list and
:meth:`Compiler.compile` runs whatever string it is given. The optimization
remarks of a compile are ``core/remarks.py``'s business
(``Compiler.remark_session`` hands out one ``RemarkSession`` per compile).
"""

import collections.abc
import importlib.util
import os
import sys
from typing import Any

from .... import ir
from ...core.common import DSLRuntimeError, DSLUserCodeError
from ...core.diagnostics import DiagId
from ...core.plugin import CompilerPlugin
from ...core.remarks import RemarkSession, error_diagnostics, remarks_available
from ...util import profiler

__all__ = [
    "Compiler",
    "OptLevel",
    "remarks_available",
]


# =============================================================================
# Compiler Class
# =============================================================================


class Compiler(CompilerPlugin):
    """The ``compiler`` plugin over MLIR's pass manager and execution engine.

    :meth:`compile` runs the pipeline, :meth:`jit` builds the engine,
    :meth:`compile_and_jit` both, :meth:`load` binds an entry into a callable,
    :meth:`remark_session` collects remarks and :meth:`print_ir_after_passes`
    prints IR on a clone. Lowering and invocation are both the plugin's, so a
    backend that does not produce an ``ExecutionEngine`` replaces the whole
    object. ``mlir.passmanager`` and ``mlir.execution_engine`` are imported on
    first use; a build without the engine makes the plugin unavailable, and a
    DSL naming it then traces only.
    """

    name = "execution_engine"

    def __init__(self) -> None:
        self._passmanager: Any = None
        self._execution_engine: Any = None
        # The structured remarks of the last ``compile`` on this plugin.
        self.collected_remarks: list[dict[str, Any]] = []

    @classmethod
    def available(cls) -> bool:
        """Whether the bindings carry the pass manager and the execution
        engine (``MLIR_ENABLE_EXECUTION_ENGINE``): a probe of the package,
        importing neither (the engine is imported at the first compile)."""
        root = __package__.rsplit(".", 3)[0]  # the `mlir` package
        return all(
            importlib.util.find_spec(f"{root}.{module}") is not None
            for module in ("execution_engine", "passmanager")
        )

    @property
    def passmanager(self) -> Any:
        """The ``mlir.passmanager`` module, imported on first use."""
        if self._passmanager is None:
            from .... import passmanager

            self._passmanager = passmanager
        return self._passmanager

    @property
    def execution_engine(self) -> Any:
        """The ``mlir.execution_engine`` module, imported on first use."""
        if self._execution_engine is None:
            from .... import execution_engine

            self._execution_engine = execution_engine
        return self._execution_engine

    def remark_session(
        self,
        context: ir.Context,
        *,
        remark_filter: str = "",
        remark_policy: str = "all",
        remark_output: str = "",
    ) -> RemarkSession:
        """A remark session for ``context``; ``with session:`` owns the engine
        and the diagnostic handler, ``session.remarks`` holds the records."""
        return RemarkSession(
            context,
            remark_filter=remark_filter,
            remark_policy=remark_policy,
            remark_output=remark_output,
        )

    def compile(
        self,
        module: ir.Module,
        pipeline: str,
        *,
        enable_ir_printing: bool = False,
        print_ir_tree_dir: str = "",
        enable_pass_profiling: bool = False,
        enable_debug_info: bool = False,
        enable_verifier: bool = True,
        remark_filter: str = "",
        remark_policy: str = "all",
        remark_output: str = "",
    ) -> ir.Module:
        """Compiles the module by invoking the pipeline and returns it.

        The pass manager always runs ``pipeline`` as given; composing it is
        the DSL's job. Subclasses overriding this method should return the
        compiled module so ``compile_and_jit`` callers can retain the
        finalized IR.

        :param module: The module to lower in place
        :param pipeline: A textual pass pipeline, ``builtin.module(...)``
        :param enable_ir_printing: Print the IR after every pass to stderr
        :param print_ir_tree_dir: Dump the IR after every pass into this tree
        :param enable_pass_profiling: Print the pass manager's timing report
        :param enable_debug_info: Print locations in the IR dumps
        :param enable_verifier: Verify the module between passes
        :param remark_filter: Regular expression over remark categories
        :param remark_policy: ``"all"`` or ``"final"``
        :param remark_output: ``.yaml``/``.bitstream`` path for the remarks
        :return: The lowered ``module``
        """
        context = module.context
        session = self.remark_session(
            context,
            remark_filter=remark_filter,
            remark_policy=remark_policy,
            remark_output=remark_output,
        )
        try:
            pm = self.passmanager.PassManager.parse(pipeline, context=context)
            if enable_pass_profiling or profiler.deep():
                pm.enable_timing()
            if print_ir_tree_dir:
                os.makedirs(print_ir_tree_dir, exist_ok=True)
                pm.enable_ir_printing(
                    tree_printing_dir_path=print_ir_tree_dir,
                    enable_debug_info=enable_debug_info,
                )
            elif enable_ir_printing:
                pm.enable_ir_printing(enable_debug_info=enable_debug_info)
            if (
                enable_ir_printing
                or enable_pass_profiling
                or print_ir_tree_dir
                or profiler.deep()
            ):
                context.enable_multithreading(False)
            pm.enable_verifier(enable_verifier)

            # The session enables the engine (unless an enclosing one owns it),
            # keeps the handler attached across the run and flushes
            # ``policy="final"`` remarks on exit, handler still attached.
            with session:
                profiler.begin_mlir_phase()
                pm.run(module.operation)
                # MLIR prints the enable_timing report when the pass manager is
                # destroyed; drop the last reference here so it lands inside the
                # profiler's capture window, then close the phase.
                del pm
                profiler.end_mlir_phase()
            self.collected_remarks = session.remarks
        except ValueError as exc:
            raise DSLRuntimeError(
                "failed to parse the pass pipeline",
                context={"pipeline": pipeline},
                cause=exc,
            ) from exc
        except ir.MLIRError as exc:
            raise DSLRuntimeError(
                "MLIR pass pipeline failed",
                context={
                    "pipeline": pipeline,
                    "diagnostics": error_diagnostics(exc, session),
                },
                cause=exc,
            ) from exc
        finally:
            # Restore the captured fd / emit the report even if the pipeline
            # raised before end_mlir_phase ran above (idempotent when already
            # closed or when the profiler is off).
            profiler.end_mlir_phase()

        return module

    @profiler.timed("jit")
    def jit(
        self,
        module: ir.Module,
        opt_level: int = 2,
        shared_libs: collections.abc.Sequence[str] = (),
    ) -> Any:
        """Wraps the module in a JIT execution engine.

        :param module: A module lowered to the LLVM dialect
        :param opt_level: LLVM optimization level, 0 to 3
        :param shared_libs: Library paths the engine loads for the generated
            code to call into (the DSL itself never binds them)
        :return: An ``ExecutionEngine``
        :raises DSLUserCodeError: ``CONFIG_INVALID``
        :raises DSLRuntimeError: The engine could not be built (a module that
            is not fully lowered to the LLVM dialect, a missing library)
        """
        OptLevel(opt_level)
        try:
            return self.execution_engine.ExecutionEngine(
                module, opt_level=opt_level, shared_libs=list(shared_libs)
            )
        except (RuntimeError, ir.MLIRError) as exc:
            raise DSLRuntimeError(
                "failed to create the execution engine",
                context={"shared_libs": ", ".join(shared_libs)},
                cause=exc,
            ) from exc

    def load(
        self,
        module: ir.Module,
        engine: Any,
        function_name: str,
        signature: Any,
        *,
        jit_time_profiling: bool = False,
        result_ctype: Any = None,
        func_type: Any = None,
    ) -> Any:
        """Bind the compiled entry ``function_name`` of ``engine`` into a
        callable Python object: the invocation half of the plugin.

        The ``ExecutionEngine`` defines a packed ``_mlir_<name>(void**)``
        wrapper for every public function; :func:`lookup_packed_function`
        resolves it and :class:`JitCompiledFunction` marshals each call into
        ``c_void_p`` slots and reads the result back through ``result_ctype``,
        the slot type the entry (the decorator plugin, ``func.Jit``) declared. A
        compiler plugin for another backend returns its own callable here; the
        DSL relies only on ``__call__``, ``ir_module``, ``function_name`` and,
        when present,
        ``execution_args`` (adapter scope, Meta values and shapes of
        ``compile()``).

        :param func_type: The compiled-function class, :class:`JitCompiledFunction`
            by default
        """
        from .jit_executor import JitCompiledFunction, lookup_packed_function

        entry = lookup_packed_function(engine, function_name)
        make = func_type if func_type is not None else JitCompiledFunction
        return make(
            module,
            engine,
            entry,
            signature,
            function_name,
            jit_time_profiling,
            result_ctype=result_ctype,
        )

    def compile_and_jit(
        self,
        module: ir.Module,
        pipeline: str,
        shared_libs: collections.abc.Sequence[str] = (),
        opt_level: int = 2,
        *,
        enable_ir_printing: bool = False,
        print_ir_tree_dir: str = "",
        enable_pass_profiling: bool = False,
        enable_debug_info: bool = False,
        enable_verifier: bool = True,
        remark_filter: str = "",
        remark_policy: str = "all",
        remark_output: str = "",
        return_module: bool = False,
        after_lowering: collections.abc.Callable[[ir.Module], None] | None = None,
    ) -> Any:
        """Compile the module (see :meth:`compile`) and wrap it in an engine
        (see :meth:`jit`).

        :param return_module: Also return the lowered module
        :param after_lowering: Called with the lowered module before the
            engine is built (the DSL runs its plugins' ``after_lowering`` here)
        :return: The engine, or an ``(engine, compiled module)`` pair when
            ``return_module`` is set
        """
        compiled_module = self.compile(
            module,
            pipeline,
            enable_ir_printing=enable_ir_printing,
            print_ir_tree_dir=print_ir_tree_dir,
            enable_pass_profiling=enable_pass_profiling,
            enable_debug_info=enable_debug_info,
            enable_verifier=enable_verifier,
            remark_filter=remark_filter,
            remark_policy=remark_policy,
            remark_output=remark_output,
        )

        if after_lowering is not None:
            after_lowering(compiled_module)
        engine = self.jit(compiled_module, opt_level, shared_libs)
        if return_module:
            return engine, compiled_module
        return engine

    def print_ir_after_passes(
        self,
        module: ir.Module,
        passes: str,
        *,
        enable_debug_info: bool = False,
        enable_verifier: bool = True,
    ) -> None:
        """Run ``passes`` on a clone of ``module`` and print the result to stderr.

        Backs ``<PREFIX>_PRINT_IR_AFTER_PASSES``: the module itself is left
        untouched, so what is cached under its hash is the unlowered IR.

        :param passes: A comma-separated pass list, or a full
            ``builtin.module(...)`` pipeline
        """
        module_clone = ir.Module.parse(
            module.operation.get_asm(enable_debug_info=True), context=module.context
        )
        pipeline = passes.strip()
        if not pipeline.startswith("builtin.module("):
            pipeline = f"builtin.module({pipeline})"
        self.compile(module_clone, pipeline, enable_verifier=enable_verifier)
        print(f"\n//===--- IR after passes: {passes} ---===\n", file=sys.stderr)
        print(
            module_clone.operation.get_asm(enable_debug_info=enable_debug_info),
            file=sys.stderr,
        )
        print("\n//===--- End of IR after passes ---===\n", file=sys.stderr)


# =============================================================================
# Options
# =============================================================================


class OptLevel:
    """The LLVM optimization level of the JIT engine, an ``int`` in ``0..3``;
    constructing one validates it.

    :raises DSLUserCodeError: ``CONFIG_INVALID`` for anything else
        (a ``bool`` and a ``float`` included)
    """

    def __init__(self, value: int) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 3
        ):
            raise DSLUserCodeError(
                DiagId.CONFIG_INVALID,
                var="opt-level",
                detail=f"the optimization level must be an integer between 0 and 3, but got {value!r}",
            )
        self.value = value
