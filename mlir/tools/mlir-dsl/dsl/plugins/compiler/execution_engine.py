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
A DSL that reports its compiles differently subclasses :class:`Compiler`:
:meth:`Compiler.remark_session` takes the session options ``compile`` is
given, :meth:`Compiler.pipeline_failed` decides the error of a failed
pipeline and :meth:`Compiler.after_pipeline` runs inside the session once the
passes are done.
"""

import collections.abc
import ctypes
import importlib.util
import os
import platform
import sys
from typing import Any

from .... import ir
from ...core.common import DSLBaseError, DSLRuntimeError, DSLUserCodeValueError
from ...core.diagnostics import DiagId
from ...core.plugin import CompilerPlugin
from ...util.logger import log

__all__ = [
    "Compiler",
    "OptLevel",
    "remarks_available",
]


# =============================================================================
# The C stdout of compiled code
# =============================================================================


# On Windows the C runtime that compiled code prints through (``printf`` from
# the JIT'd module resolves to the process's ``msvcrt``) fully buffers its
# stdout when that is a pipe, so a line printed by compiled code would show up
# after Python's later prints, or not at all before the process exits. ``jit``
# switches that stream to unbuffered, once per process. ``msvcrt`` exposes its
# ``FILE`` table through ``__iob_func``; stdout is the second entry, one
# ``FILE`` (48 bytes on x64; the layout differs on x86, hence the machine
# check) into it, and ``_fileno`` confirms the entry before ``setvbuf`` runs.
_IONBF = 4
_MSVCRT_FILE_SIZE_X64 = 48
_jit_c_stdout_unbuffered = False


def _unbuffer_jit_c_stdout() -> None:
    """Make the C stdout compiled code prints through unbuffered (Windows x64
    only; a no-op elsewhere and after the first call)."""
    global _jit_c_stdout_unbuffered
    if _jit_c_stdout_unbuffered:
        return
    _jit_c_stdout_unbuffered = True
    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        return
    try:
        crt = ctypes.cdll.msvcrt
        crt.__iob_func.restype = ctypes.c_void_p
        crt.__iob_func.argtypes = []
        crt._fileno.restype = ctypes.c_int
        crt._fileno.argtypes = [ctypes.c_void_p]
        crt.setvbuf.restype = ctypes.c_int
        crt.setvbuf.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        iob = crt.__iob_func()
        if not iob:
            return
        c_stdout = ctypes.c_void_p(iob + _MSVCRT_FILE_SIZE_X64)
        if crt._fileno(c_stdout) != 1:
            return
        crt.setvbuf(c_stdout, None, _IONBF, 0)
    except (AttributeError, OSError) as exc:
        log().debug("could not unbuffer the C stdout of compiled code: %s", exc)


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
        :return: The lowered ``module``
        """
        context = module.context
        session = self.remark_session(
            context,
        )
        try:
            pm = self.passmanager.PassManager.parse(pipeline, context=context)
            if enable_pass_profiling:
                pm.enable_timing()
            if print_ir_tree_dir:
                os.makedirs(print_ir_tree_dir, exist_ok=True)
                pm.enable_ir_printing(
                    tree_printing_dir_path=print_ir_tree_dir,
                    enable_debug_info=enable_debug_info,
                )
            elif enable_ir_printing:
                pm.enable_ir_printing(enable_debug_info=enable_debug_info)
            if enable_ir_printing or enable_pass_profiling or print_ir_tree_dir:
                context.enable_multithreading(False)
            pm.enable_verifier(enable_verifier)

            # The session enables the engine (unless an enclosing one owns it),
            # keeps the handler attached across the run and flushes
            # ``policy="final"`` remarks on exit, handler still attached.
            with session:
                pm.run(module.operation)
                # MLIR prints the enable_timing report when the pass manager is
                # destroyed; drop the last reference here so it lands inside the
                # profiler's capture window, then close the phase.
                del pm
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
                },
                cause=exc,
            ) from exc

        return module



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
        _unbuffer_jit_c_stdout()
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
        libs_var: str = "",
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

        :param engine: The ``ExecutionEngine``; None for a module that was
            lowered without one (``CompileStep(build_engine=False)``): no
            entry is bound, and the returned object holds the lowered module
            but refuses to be called
        :param func_type: The compiled-function class, :class:`JitCompiledFunction`
            by default
        :param libs_var: The runtime-library setting (``<PREFIX>_LIBS``) a
            failed lookup is blamed on, as ``CONFIG_INVALID``; an internal
            error when empty (see :func:`lookup_packed_function`)
        """
        from .jit_executor import JitCompiledFunction, lookup_packed_function

        entry = (
            None
            if engine is None
            else lookup_packed_function(engine, function_name, libs_var=libs_var)
        )
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
        )

        if after_lowering is not None:
            after_lowering(compiled_module)
        engine = self.jit(compiled_module, opt_level, shared_libs)
        if return_module:
            return engine, compiled_module
        return engine



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
            raise DSLUserCodeValueError(
                DiagId.CONFIG_INVALID,
                var="opt-level",
                detail=f"the optimization level must be an integer between 0 and 3, but got {value!r}",
            )
        self.value = value
