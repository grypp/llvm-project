# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The generic half of a kernels plugin: the ``@kernel`` decorator, its
launcher and the entry protocol a target implements.

:class:`KernelsPlugin` is a ``DecoratorPlugin`` whose ``decorator_name`` is
``kernel``. Calling a decorated function, from Python or inside a trace,
returns a :class:`KernelLauncher`; ``.launch(...)`` inside a ``@jit`` body
traces the kernel into the plugin's container and emits the launch at the
call site (outside one it is ``LAUNCH_OUTSIDE_JIT``). The
plugin owns the kernel state of a trace: the container (``before_trace``/``after_trace``), the launches it expects (a call
never launched is ``LAUNCH_NEVER_ISSUED``), the buffer-kind rule at the host
boundary (``check_arguments``) and the kernel records handed to the compiled
function (``finish_compiled_function``). A target subclass implements the entry
protocol: ``launch_config`` (what ``.launch(...)`` accepts: the gpu target's
``LaunchConfig``), ``generate_func_op``, ``generate_return``, ``generate_launch``
and the container hooks, with the ``LAUNCH_*`` codes in its ``diag_ids``
catalogue. Nothing here is CUDA-shaped: the grid and block live in the target.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from ..... import ir
from ....core.common import DSLRuntimeError, DSLUserCodeError, in_trace
from ....core.diagnostics import DiagId, find_user_source_location
from ....core.plugin import DecoratorPlugin
from ....util import tree_utils
from ....types import typing as t

if TYPE_CHECKING:
    from ....core.dsl import BaseDSL

__all__ = ["KernelLauncher", "KernelsPlugin"]


class KernelLauncher:
    """Bound kernel arguments awaiting their launch inside a ``@jit`` body;
    ``.launch(...)`` takes the target's launch configuration (the gpu target:
    a ``LaunchConfig`` or its fields)::

        kernel(arg1, arg2).launch(LaunchConfig(grid=[1, 1, 1], block=[1, 1, 1]))
        kernel(arg1, arg2).launch(grid=[1, 1, 1], block=[1, 1, 1])
    """

    def __init__(
        self,
        dsl: "BaseDSL",
        plugin: "KernelsPlugin",
        func: Callable[..., None],
        /,
        *func_args: Any,
        **func_kwargs: Any,
    ) -> None:
        self.dsl = dsl
        self.plugin = plugin
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs

        # While a host body is being traced, register so an un-launched call is
        # reported; capture the call site now, while the user's frame is live,
        # for the diagnostic's caret.
        self._launched = False
        self._creation_loc: tuple[Any, Any, Any, Any] = (None, None, None, None)
        if plugin._pending is not None:
            self._creation_loc = find_user_source_location()
            plugin._pending.append(self)

        try:
            inspect.signature(func).bind(*func_args, **func_kwargs)
        except TypeError as e:
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=getattr(func, "__name__", "the function"),
                detail=f"{len(func_args)} positional and {len(func_kwargs)} keyword argument(s) do not bind to its parameters ({e})",
                cause=e,
            ) from e

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """Emit the kernel and its launch at the current insertion point; the
        arguments are what the plugin's ``launch_config`` accepts."""
        kernel_name = getattr(self.func, "__name__", "<kernel>")
        # No open trace means there is no host body to emit the launch into.
        if not in_trace():
            raise DSLUserCodeError(
                self.plugin.diag("LAUNCH_OUTSIDE_JIT"), kernel_name=kernel_name
            )
        if self._launched:
            raise DSLUserCodeError(
                self.plugin.diag("LAUNCH_ALREADY_ISSUED"), kernel_name=kernel_name
            )
        # A launch is being issued: this launcher is no longer a dangling
        # `my_kernel(...)` call.
        self._launched = True

        config = self.plugin.launch_config(self.func, *args, **kwargs)
        return self.plugin.emit_launch(
            self.dsl, self.func, self.func_args, self.func_kwargs, config
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.launch(*args, **kwargs)


class KernelsPlugin(DecoratorPlugin):
    """A plugin adding ``@<decorator_name>`` and the launch of what it marks.

    Calling a kernel, from Python or inside a trace, prepares a launch (both
    :meth:`call` and :meth:`launch` return a :class:`KernelLauncher`); issuing
    it emits the kernel and its launch op into the open trace. A target
    subclass implements the entry protocol and names its diagnostic catalogue
    in ``diag_ids`` (the ``LAUNCH_*`` codes the launcher raises through it).
    The plugin instance installed on a DSL keeps the per-trace state: the
    container, the launchers created during the host trace, the kernel records
    of the last trace (``kernel_info``, ``num_kernels``, ``launch_count``).
    """

    decorator_name: ClassVar[str] = "kernel"
    diag_ids: ClassVar[Any] = None

    def __init__(self) -> None:
        self.kernel_info: "OrderedDict[str, Any]" = OrderedDict()
        self.num_kernels = 0
        self.launch_count = 0
        self._pending: list[KernelLauncher] | None = None
        self._module: Any = None
        self._container: Any = None

    # -- what a call does: a deferred launch -------------------------------------

    def call(self, dsl: "BaseDSL", func: Any, *args: Any, **kwargs: Any) -> Any:
        """A kernel called from plain Python prepares its launch like one called
        inside a trace; issuing it there is ``LAUNCH_OUTSIDE_JIT``."""
        return KernelLauncher(dsl, self, func, *args, **kwargs)

    def launch(self, dsl: "BaseDSL", func: Any, *args: Any, **kwargs: Any) -> Any:
        """A kernel called inside a host trace: the deferred launcher, whose
        ``.launch(config)`` emits the kernel and its launch op."""
        return KernelLauncher(dsl, self, func, *args, **kwargs)

    def diag(self, code: str) -> Any:
        """The launch diagnostic ``code`` of the target's catalogue."""
        catalogue = self.diag_ids
        if catalogue is None or not hasattr(catalogue, code):
            raise DSLRuntimeError(
                f"the kernels plugin declares no `{code}` diagnostic",
                context={"plugin": type(self).__name__},
            )
        return getattr(catalogue, code)

    # -- per-trace state ---------------------------------------------------------

    def before_trace(
        self, dsl: "BaseDSL", module: ir.Module, *, loc: Any, attrs: dict[str, Any]
    ) -> None:
        self.kernel_info = OrderedDict()
        self.num_kernels = 0
        self.launch_count = 0
        self._module = module
        self._container = self.build_container(dict(attrs or {}), loc=loc)
        self._pending = []

    def after_trace(self, dsl: "BaseDSL", module: ir.Module) -> None:
        pending, self._pending = self._pending or [], None
        for launcher in pending:
            if not launcher._launched:
                filename, lineno, col, end_col = launcher._creation_loc
                raise DSLUserCodeError(
                    self.diag("LAUNCH_NEVER_ISSUED"),
                    filename=filename,
                    lineno=lineno,
                    col_offset=col,
                    end_col_offset=end_col,
                    kernel_name=getattr(launcher.func, "__name__", "<kernel>"),
                )
        self.prune_empty_containers(module)
        self._module = self._container = None

    def check_arguments(
        self,
        dsl: "BaseDSL",
        sig: Any,
        args: tuple[Any, ...],
        kwonlyargs: dict[str, Any],
        adapted: list[Any],
        function_name: str,
    ) -> None:
        """Reject a buffer whose side does not match what the trace did.

        A host buffer (a numpy array, a CPU tensor) passed to a call whose
        trace launched a kernel is ``LAUNCH_HOST_BUFFER``; a device buffer
        passed to a call whose trace launched nothing is ``ARG_BUFFER_INVALID``.
        A bare address (kind ``unknown``) is never checked.
        """
        launched = self.launch_count > 0
        input_args = [*args, *kwonlyargs.values()]
        for name, original, value in zip(sig.parameters, input_args, adapted):
            arg = original if value is None else value
            if not tree_utils.contains_leaf(arg):
                continue
            values, _, _ = tree_utils.tree_flatten(arg, return_ir_values=False)
            for leaf in values:
                kind = getattr(leaf, "kind", "unknown")
                if not isinstance(leaf, t.Pointer) or kind == "unknown":
                    continue
                if kind == "host" and launched:
                    raise DSLUserCodeError(
                        self.diag("LAUNCH_HOST_BUFFER"),
                        arg_name=name,
                        arg_type=type(original).__name__,
                        function_name=function_name,
                    )
                if kind == "device" and not launched:
                    raise DSLUserCodeError(
                        DiagId.ARG_BUFFER_INVALID,
                        arg_name=name,
                        arg_type=type(original).__name__,
                        detail=f"it is a device buffer, but `{function_name}` runs on the host (its trace launched no kernel)",
                    )

    def finish_compiled_function(self, dsl: "BaseDSL", jit_function: Any) -> None:
        """Record the kernels of the trace on the compiled function."""
        jit_function.kernel_info = dict(self.kernel_info)
        jit_function.has_kernels = self.num_kernels > 0

    # -- the launch driver -------------------------------------------------------

    def emit_launch(
        self,
        dsl: "BaseDSL",
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: Any,
    ) -> Any:
        """Trace ``func`` as a kernel of the current host trace and emit its
        launch at the current insertion point.

        :param config: What ``launch_config`` returned
        :return: What ``generate_launch`` returned
        """
        base = getattr(func, "__name__", "<kernel>")
        signature, cargs, ckwargs, jit_args = dsl.bind_arguments(
            func, base, args, kwargs, is_host=False
        )
        # Every launch traces the kernel anew, so each gets its own symbol.
        name = f"kernel_{dsl.mangle_name(base, cargs, signature)}_{self.num_kernels}"
        self.num_kernels += 1
        loc = dsl.get_ir_location()
        with self.container_insertion_point(self._container, self._module):
            op, block, result = dsl.trace_body(
                self,
                name,
                func,
                cargs,
                ckwargs,
                signature,
                jit_args.types,
                jit_args.attributes,
                loc=loc,
            )
            if result is not None:
                raise DSLUserCodeError(
                    DiagId.TYPE_RETURN_MISMATCH,
                    got=f"a `{type(result).__name__}`",
                    detail=" from a kernel",
                )
            with ir.InsertionPoint(block):
                self.generate_return(op, [], loc=loc)
        symbol = self.kernel_symbol(name)
        ret = self.generate_launch(op, symbol, jit_args.values, config, loc=loc)
        self.kernel_info[name] = config
        self.launch_count += 1
        return ret

    # -- the entry protocol a target implements ----------------------------------

    def launch_config(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """The launch configuration behind ``kernel(...).launch(*args,
        **kwargs)``: whatever this target's ``generate_launch`` reads (the gpu
        target: a ``LaunchConfig`` or its fields). ``func`` is the kernel, for
        diagnostics."""
        return self._unsupported("launch_config")

    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> tuple[Any, ir.Block]:
        """Create the kernel function ``name`` with ``arg_types`` at the current
        insertion point; return ``(op, entry_block)``."""
        return self._unsupported("generate_func_op")

    def generate_return(self, op: Any, values: list[Any], loc: Any = None) -> None:
        """Terminate the kernel body at the current insertion point; a kernel
        returns nothing, so ``values`` is empty."""
        self._unsupported("generate_return")

    def generate_launch(
        self,
        op: Any,
        symbol: ir.Attribute,
        operands: list[Any],
        config: Any,
        *,
        loc: Any = None,
    ) -> Any:
        """Emit the launch of kernel ``op`` (referred to as ``symbol``) with
        ``operands`` under ``config`` (what ``launch_config`` returned) at the
        current insertion point."""
        return self._unsupported("generate_launch")

    def build_container(self, attrs: dict[str, Any], loc: Any = None) -> Any:
        """Create the op holding this target's kernels at the current insertion
        point (the gpu target: ``gpu.module @kernels`` plus the
        ``gpu.container_module`` marker on the host module), or None when the
        kernels live in the host module itself."""
        return None

    def container_insertion_point(
        self, container: Any, module: Any
    ) -> ir.InsertionPoint:
        """Where a kernel of this trace is emitted: inside ``container`` when
        there is one, else at the start of the host ``module``."""
        return ir.InsertionPoint.at_block_begin(module.body)

    def prune_empty_containers(self, module: Any) -> None:
        """Drop a container that received no kernel, after the trace."""

    def kernel_symbol(self, kernel_name: str) -> ir.Attribute:
        """The symbol a launch refers to."""
        return ir.FlatSymbolRefAttr.get(kernel_name)
