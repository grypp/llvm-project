# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This module provides a class that compiles generated IR using MLIR's PassManager
and executes it using MLIR's ExecutionEngine.

The pass pipeline is data: ``BaseDSL._get_pipeline`` composes the plugin passes
and the core list, and :meth:`Compiler.compile` runs whatever string it is
given. The optimization remarks of a compile are ``remarks.py``'s business
(``Compiler.remark_session`` hands out one ``_RemarkSession`` per compile).
"""

import collections.abc
import os
import sys
import types
from typing import Any

from ... import ir
from ..core.common import DSLRuntimeError, DSLUserCodeError
from ..core.diagnostics import DiagId
from ..util import phase_profiler
from ..util.logger import log
from .remarks import _RemarkSession, _error_diagnostics, remarks_available

__all__ = [
    "Compiler",
    "CompileOption",
    "OptLevel",
    "PostCompileHookContext",
    "remarks_available",
]


# =============================================================================
# Compiler Class
# =============================================================================


class Compiler:
    """Compiler class for compiling and building MLIR modules.

    :param passmanager: The ``mlir.passmanager`` module
    :param execution_engine: The ``mlir.execution_engine`` module
    """

    def __init__(self, passmanager: Any, execution_engine: Any) -> None:
        self.passmanager = passmanager
        self.execution_engine = execution_engine
        self._post_compile_hook: collections.abc.Callable[[Any], None] | None = None
        # The structured remarks of the last ``compile`` on this provider.
        self.collected_remarks: list[dict[str, Any]] = []

    def remark_session(
        self,
        context: ir.Context,
        *,
        remark_filter: str = "",
        remark_policy: str = "all",
        remark_output: str = "",
    ) -> _RemarkSession:
        """A remark session for ``context``; ``with session:`` owns the engine
        and the diagnostic handler, ``session.remarks`` holds the records."""
        return _RemarkSession(
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
            if enable_pass_profiling or phase_profiler.deep():
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
                or phase_profiler.deep()
            ):
                context.enable_multithreading(False)
            pm.enable_verifier(enable_verifier)

            # The session enables the engine (unless an enclosing one owns it),
            # keeps the handler attached across the run and flushes
            # ``policy="final"`` remarks on exit, handler still attached.
            with session:
                phase_profiler.begin_mlir_phase()
                pm.run(module.operation)
                # MLIR prints the enable_timing report when the pass manager is
                # destroyed; drop the last reference here so it lands inside the
                # profiler's capture window, then close the phase.
                del pm
                phase_profiler.end_mlir_phase()
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
                    "diagnostics": _error_diagnostics(exc, session),
                },
                cause=exc,
            ) from exc
        finally:
            # Restore the captured fd / emit the report even if the pipeline
            # raised before end_mlir_phase ran above (idempotent when already
            # closed or when the profiler is off).
            phase_profiler.end_mlir_phase()

        if self._post_compile_hook:
            self._post_compile_hook(module)
        return module

    @phase_profiler.timed("jit")
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
        :raises DSLUserCodeError: ``CONFIG_INVALID_OPT_LEVEL``
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
    ) -> Any:
        """Compile the module (see :meth:`compile`) and wrap it in an engine
        (see :meth:`jit`).

        :param return_module: Also return the lowered module
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


class PostCompileHookContext:
    """Install ``hook`` as the compiler's post-compile hook for a block.

    The hook is called with the lowered module after every
    :meth:`Compiler.compile`; the previous hook is restored on exit.

    :param compiler: The compiler to hook
    :param hook: A callable taking the lowered ``ir.Module``
    """

    def __init__(
        self,
        compiler: Compiler,
        hook: collections.abc.Callable[[Any], None],
    ) -> None:
        self.compiler = compiler
        self.hook = hook
        self.prev_post_compile_hook: collections.abc.Callable[[Any], None] | None = None

    def __enter__(self) -> "PostCompileHookContext":
        self.prev_post_compile_hook = self.compiler._post_compile_hook
        self.compiler._post_compile_hook = self.hook
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.compiler._post_compile_hook = self.prev_post_compile_hook


# =============================================================================
# Compile options
# =============================================================================


class CompileOption:
    """Base class for compile options.

    ``_option_name`` is the pipeline-string token; ``None`` marks an option
    that is not a pipeline option (``serialize()`` returns ``""``).
    """

    _option_name: "str | None" = None

    def __init__(self, val: Any) -> None:
        self._value: Any = val

    def serialize(self) -> str:
        """The ``name=value`` pipeline token, or ``""`` for a non-pipeline option."""
        if self.__class__._option_name is None:
            return ""
        return f"{self.__class__._option_name}={self._value}"

    @property
    def value(self) -> Any:
        """The option's value."""
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value = value


class OptLevel(CompileOption):
    """The LLVM optimization level of the JIT engine, an ``int`` in ``0..3``.

    :raises DSLUserCodeError: ``CONFIG_INVALID_OPT_LEVEL`` for anything else
        (a ``bool`` and a ``float`` included)
    """

    _option_name = "opt-level"

    def __init__(self, val: int) -> None:
        if isinstance(val, bool) or not isinstance(val, int) or val < 0 or val > 3:
            raise DSLUserCodeError(DiagId.CONFIG_INVALID_OPT_LEVEL, val=repr(val))
        super().__init__(val)
