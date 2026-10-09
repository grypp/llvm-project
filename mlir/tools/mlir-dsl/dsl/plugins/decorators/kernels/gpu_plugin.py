# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The gpu kernels plugin: ``@kernel`` as ``gpu.func`` + ``gpu.launch_func``.

:class:`KernelsPlugin` is the target-neutral half: a ``DecoratorPlugin`` whose
``decorator_name`` is ``kernel``. Calling a decorated function, from Python or
inside a trace, returns a :class:`KernelLauncher`, a core ``DeferredDecoratorCall``;
``.launch(...)`` inside a ``@jit`` body issues it: the plugin traces the kernel
into its container and emits the launch at the call site. The core keeps the
deferred-call bookkeeping (never issued, issued twice, issued outside a
trace); the plugin owns the container (``before_trace``/``after_trace``), the
buffer-kind rule at the host boundary (``check_arguments``) and the kernel
records handed to the compiled function (``finish_compiled_function``).

:class:`Kernels` is the CUDA target over the ``gpu`` dialect: it builds the
``gpu.module`` container of a trace, the kernel function (``gpu.func`` with
``gpu.kernel``, ``known_block_size`` when the block is static) and a
synchronous ``gpu.launch_func`` guarded against zero-sized staged dimensions;
what ``.launch(...)`` accepts is its :class:`LaunchConfig`. It names the
``libmlir_cuda_runtime`` library for the execution engine (never bound here)
and reads ``<PREFIX>_ARCH``, the chip MLIR's lowering compiles for, only when a
launch is compiled: tracing needs no target, and the plugin never interprets
the chip name. The sub-DSL spells MLIR's ``gpu-lower-to-nvvm-pipeline`` with
the plugin's ``chip_option`` in its own ``pipeline()``. The DSL owns no CUDA runtime code and exposes no kernel-body
index helpers: a kernel body is the same language as a host body (loads,
stores, loops). The module imports without the gpu bindings; ``available()``
says whether the plugin can serve.
"""

from __future__ import annotations

import enum
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

from ..... import _mlir_libs, ir
from ....core.common import DSLRuntimeError, DSLUserCodeError
from ....core.diagnostics import USAGE, DiagCatalog, DiagId, classify
from ....core.mlir_op import current_emitter
from ....core.plugin import DecoratorPlugin, DeferredDecoratorCall
from ....types.typing import Boolean, Int32, Int64, Integer, Numeric, Pointer
from ....util import tree_utils
from ....util.logger import log

try:
    from .....dialects import gpu, nvvm, scf
except ImportError:  # the gpu bindings are not built into this MLIR
    gpu = nvvm = scf = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ....core.dsl import BaseDSL

__all__ = [
    "GpuDiagId",
    "KernelLauncher",
    "Kernels",
    "KernelsPlugin",
    "LaunchConfig",
    "check_arch",
    "cuda_runtime_library",
]


# =============================================================================
# Diagnostics
# =============================================================================


class GpuDiagId(DiagCatalog, enum.Enum):
    """Launch and target diagnostics, rendered ``error[gpu:CODE]``; member name ==
    stable code, value == (message, fixes). The generic ones (a prepared call
    never issued, issued twice, issued outside a trace) are the core's
    ``CALL_*`` codes: the ``DeferredDecoratorCall`` raises them.
    """

    namespace = enum.nonmember("gpu")

    LAUNCH_INVALID_DIMENSION = (
        "Element {idx} of the `{name}` launch argument is a `{arg_type}`{detail}, but "
        "every entry must be a non-negative integer (`int` or a staged `Integer`).",
        ("Pass integers for every entry of `{name}`, e.g. `{name}=[2, 1, 1]`.",),
    )
    LAUNCH_INVALID_GRID = (
        "The `{name}` launch argument can have at most 3 entries, but it has {count}.",
        ("Pass at most three values for `{name}`, e.g. `{name}=[128, 1, 1]`.",),
    )
    LAUNCH_HOST_BUFFER = (
        "Argument `{arg_name}` is a host buffer (`{arg_type}`), but this function "
        "launches a device kernel: device kernels take device addresses.",
        (
            "Allocate `{arg_name}` on the device with your framework and pass that tensor.",
        ),
    )
    LAUNCH_STREAM_UNSUPPORTED = (
        "Kernel `{kernel_name}` is launched with `async_deps`, but launches are "
        "synchronous: no stream can be passed.",
        ("Leave `async_deps` empty; the launch completes before the call returns.",),
    )
    CONFIG_MISSING_ARCH = (
        "No target chip is set for the gpu kernels this function launches: `{var}` "
        "must name the chip MLIR's gpu lowering compiles for.",
        ("Set the environment variable `{var}=<chip>` before the DSL is first used.",),
    )


classify(
    GpuDiagId,
    USAGE,
    "kernel launch",
    "LAUNCH_INVALID_DIMENSION",
    "LAUNCH_INVALID_GRID",
    "LAUNCH_HOST_BUFFER",
    "LAUNCH_STREAM_UNSUPPORTED",
)
classify(GpuDiagId, USAGE, "compile options", "CONFIG_MISSING_ARCH")


# =============================================================================
# Launch configuration
# =============================================================================


@dataclass
class LaunchConfig:
    """Grid, block and optional cluster dimensions plus dynamic shared memory
    of one ``gpu.launch_func``: what ``kernel(...).launch(...)`` takes under
    this plugin, whole or as its fields.

    Dimensions accept Python ints or staged integers and are padded to three
    entries; their type and count are validated when the launch is emitted.
    ``async_deps`` is kept for signature fidelity and must be empty: launches
    are synchronous.
    """

    cluster: list[Any] | None = None
    grid: list[Any] = field(default_factory=lambda: [1, 1, 1])
    block: list[Any] = field(default_factory=lambda: [1, 1, 1])
    smem: int | None = None
    async_deps: list[Any] = field(default_factory=list)

    @staticmethod
    def _pad_dim(dim: Any) -> list[Any]:
        """Return ``dim`` (a scalar or a sequence) as a list padded with 1s to
        three entries; a longer list is left for the launch to diagnose."""
        if not isinstance(dim, (list, tuple)):
            dim = [dim]
        return list(dim) + [1] * (3 - len(dim))

    def __post_init__(self) -> None:
        self.grid = self._pad_dim(self.grid)
        self.block = self._pad_dim(self.block)
        if self.cluster is not None:
            self.cluster = self._pad_dim(self.cluster)


# =============================================================================
# Target and runtime library
# =============================================================================

_CUDA_RUNTIME_STEM = "mlir_cuda_runtime"


def _find_cuda_runtime(libs: Optional[str] = None) -> Optional[str]:
    """Return the path of ``libmlir_cuda_runtime``, or ``None``: a path lookup only.

    The entry naming it in ``libs`` (the DSL's ``<PREFIX>_LIBS`` value, passed by
    the caller), else ``MLIR_CUDA_RUNTIME``, else the copy next to ``mlir._mlir_libs``.
    """
    libs = libs or ""
    entries = (
        re.split(r";|:(?![\\/])", libs) if os.name == "nt" else libs.split(os.pathsep)
    )
    for entry in filter(None, entries):
        path = Path(entry).expanduser()
        if _CUDA_RUNTIME_STEM in path.name and path.is_file():
            return str(path)
    explicit = Path(os.environ.get("MLIR_CUDA_RUNTIME", "")).expanduser()
    if explicit.name and explicit.is_file():
        return str(explicit.resolve())
    lib_dir = Path(_mlir_libs.__file__).parent
    for name in (
        f"lib{_CUDA_RUNTIME_STEM}.so",
        f"lib{_CUDA_RUNTIME_STEM}.dylib",
        f"{_CUDA_RUNTIME_STEM}.dll",
    ):
        if (lib_dir / name).is_file():
            return str(lib_dir / name)
    return None


def cuda_runtime_library(libs: Optional[str] = None) -> Optional[str]:
    """The path of ``libmlir_cuda_runtime`` for the ``ExecutionEngine``, if found
    (the ``libs`` paths, then ``MLIR_CUDA_RUNTIME``, then the bindings' library
    directory); the generated host code calls into it, the DSL never does. A
    DSL with gpu kernels returns it from ``shared_libs``."""
    return _find_cuda_runtime(libs)


def check_arch(arch: Any, *, var: str = "<PREFIX>_ARCH") -> str:
    """Return ``arch`` when a target chip is set.

    The plugin does not interpret the name: it is handed to MLIR's gpu lowering
    as the ``chip_option``, which validates it when a launch is compiled.

    :param arch: The value of ``var``; ``None`` and ``""`` read as unset
    :param var: The environment variable named in the diagnostic (the
        DSL's ``<PREFIX>_ARCH``)
    :return: ``arch`` unchanged
    :raises DSLUserCodeError: ``gpu:CONFIG_MISSING_ARCH`` when unset
    """
    if not isinstance(arch, str) or arch == "":
        raise DSLUserCodeError(GpuDiagId.CONFIG_MISSING_ARCH, var=var)
    return arch


# =============================================================================
# Launch dimensions
# =============================================================================

_Dimension = Union[int, Integer, ir.Value]


def _invalid_dimension(
    idx: int, name: str, arg_type: str, detail: str = ""
) -> DSLUserCodeError:
    """The ``LAUNCH_INVALID_DIMENSION`` error for entry ``idx`` of ``name``."""
    return DSLUserCodeError(
        GpuDiagId.LAUNCH_INVALID_DIMENSION,
        idx=idx,
        name=name,
        arg_type=arg_type,
        detail=detail,
    )


def _dimensions(dimensions: Any, name: str) -> tuple[_Dimension, ...]:
    """Validate a launch dimension argument into three entries: an ``int``, an
    ``Integer`` or an integer ``ir.Value``, or a sequence of one to three of them,
    padded with ``1``. A Python-valued ``Integer`` folds to its ``int``; a staged one
    is kept so its signedness reaches the ``i64`` promotion. ``bool``/``Boolean``,
    floats, negative sizes and a zero ``block`` are rejected."""
    if isinstance(dimensions, (int, Numeric, ir.Value)):
        dimensions = (dimensions,)
    elif isinstance(dimensions, (tuple, list)):
        dimensions = tuple(dimensions)
    else:
        raise _invalid_dimension(0, name, type(dimensions).__name__)
    if not 1 <= len(dimensions) <= 3:
        raise DSLUserCodeError(
            GpuDiagId.LAUNCH_INVALID_GRID, name=name, count=len(dimensions)
        )
    dimensions = dimensions + (1,) * (3 - len(dimensions))

    entries: list[_Dimension] = []
    for idx, dimension in enumerate(dimensions):
        arg_type = type(dimension).__name__
        if isinstance(dimension, Integer) and not isinstance(dimension, Boolean):
            if not isinstance(dimension.value, ir.Value):
                dimension = int(dimension.value)
        elif isinstance(dimension, ir.Value):
            if (
                not isinstance(dimension.type, ir.IntegerType)
                or dimension.type.width == 1
            ):
                raise _invalid_dimension(idx, name, str(dimension.type))
        if isinstance(dimension, (bool, Boolean)) or not isinstance(
            dimension, (int, Integer, ir.Value)
        ):
            raise _invalid_dimension(idx, name, arg_type)
        if isinstance(dimension, int) and (
            dimension < 0 or (dimension == 0 and name == "block")
        ):
            raise _invalid_dimension(
                idx, name, arg_type, f" with the value {dimension}"
            )
        entries.append(dimension)
    return tuple(entries)


def _launch_index_values(
    entries: tuple[_Dimension, ...], *, loc: Any
) -> tuple[list[ir.Value], list[ir.Value]]:
    """Promote validated entries to ``i64`` (what the LLVM translation of the launch
    expects): an ``int`` becomes ``arith.constant``, a staged entry goes through
    ``cast`` by its signedness and is also listed in ``dynamic`` for the guard."""
    values: list[ir.Value] = []
    dynamic: list[ir.Value] = []
    for entry in entries:
        if isinstance(entry, int):
            values.append(current_emitter().const(entry, Int64, loc=loc))
            continue
        signed = type(entry).signed if isinstance(entry, Integer) else None
        values.append(current_emitter().cast(entry, Int64, signed=signed, loc=loc))
        dynamic.append(values[-1])
    return values, dynamic


def _smem_operand(smem: Any, *, loc: Any) -> Optional[ir.Value]:
    """The ``i32`` dynamic shared memory operand, or ``None`` when unset."""
    if smem is None:
        return None
    if isinstance(smem, bool) or not isinstance(smem, (int, Integer, ir.Value)):
        raise DSLUserCodeError(
            DiagId.ARG_NOT_NUMERIC, arg_name="smem", arg_type=type(smem).__name__
        )
    if isinstance(smem, int):
        return current_emitter().const(smem, Int32, loc=loc)
    return current_emitter().cast(smem, Int32, loc=loc)


# =============================================================================
# The prepared call and the target-neutral plugin
# =============================================================================


class KernelLauncher(DeferredDecoratorCall):
    """A prepared kernel call; ``.launch(...)`` is its one verb and issues the
    launch with what the plugin's ``launch_config`` accepts (the gpu plugin: a
    ``LaunchConfig`` or its fields)::

        kernel(arg1, arg2).launch(LaunchConfig(grid=[1, 1, 1], block=[1, 1, 1]))
        kernel(arg1, arg2).launch(grid=[1, 1, 1], block=[1, 1, 1])
    """

    launch = DeferredDecoratorCall.issue


class KernelsPlugin(DecoratorPlugin):
    """The target-neutral half of a kernels plugin: ``@<decorator_name>`` and
    the launch of what it marks.

    Calling a kernel, from Python or inside a trace, prepares the launch (both
    :meth:`call` and :meth:`launch` return a :class:`KernelLauncher`, a
    ``DeferredDecoratorCall``); issuing it runs :meth:`emit`, which traces the kernel
    into the plugin's container and emits the launch op. The core keeps the
    deferred-call bookkeeping. A target subclass implements the entry protocol
    (``launch_config``, ``generate_func_op``, ``generate_return``,
    ``generate_launch``, the container hooks) and names its diagnostic
    catalogue in ``diag_ids``. The plugin instance installed on a DSL keeps
    the per-trace state: the container and the kernel records of the last
    trace (``kernel_info``, ``num_kernels``, ``launch_count``).
    """

    decorator_name: ClassVar[str] = "kernel"
    diag_ids: ClassVar[Any] = None

    def __init__(self) -> None:
        self.kernel_info: "OrderedDict[str, Any]" = OrderedDict()
        self.num_kernels = 0
        self.launch_count = 0
        self._module: Any = None
        self._container: Any = None

    # -- what a call does: a deferred launch -------------------------------------

    def call(self, dsl: "BaseDSL", func: Any, *args: Any, **kwargs: Any) -> Any:
        """A kernel called from plain Python prepares its launch like one called
        inside a trace; issuing it there is ``CALL_OUTSIDE_JIT``."""
        return KernelLauncher(dsl, self, func, args, kwargs)

    def launch(self, dsl: "BaseDSL", func: Any, *args: Any, **kwargs: Any) -> Any:
        """A kernel called inside a host trace: the prepared launch, whose
        ``.launch(...)`` emits the kernel and its launch op."""
        return KernelLauncher(dsl, self, func, args, kwargs)

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

    def after_trace(self, dsl: "BaseDSL", module: ir.Module) -> None:
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
                if not isinstance(leaf, Pointer) or kind == "unknown":
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

    # -- the launch driver: issuing the prepared call ----------------------------

    def emit(
        self,
        dsl: "BaseDSL",
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *issue_args: Any,
        **issue_kwargs: Any,
    ) -> Any:
        """Trace ``func`` as a kernel of the current host trace and emit its
        launch at the current insertion point; ``issue_args``/``issue_kwargs``
        are what the user passed to ``.launch(...)``, read by ``launch_config``.

        :return: What ``generate_launch`` returned
        """
        config = self.launch_config(func, *issue_args, **issue_kwargs)
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


# =============================================================================
# The plugin
# =============================================================================


class Kernels(KernelsPlugin):
    """The kernels plugin for CUDA kernels over the ``gpu`` dialect.

    :param chip_option: The option of MLIR's ``gpu-lower-to-nvvm-pipeline`` that
        names the target architecture; the sub-DSL spells that pipeline in its
        own ``pipeline()`` with it, and :meth:`pipeline_options` merges it into
        a ``<PREFIX>_PIPELINE`` override. The plugin lists no pass itself.
    """

    name = "gpu"
    # ``KernelsPlugin.diag`` resolves the ``LAUNCH_*`` codes through it.
    diag_ids: ClassVar[Any] = GpuDiagId
    issue_example: ClassVar[str] = ".launch(grid=[...], block=[...])"

    def __init__(self, *, chip_option: str = "cubin-chip") -> None:
        super().__init__()
        self.chip_option = chip_option

    @classmethod
    def available(cls) -> bool:
        """Whether the ``gpu`` and ``nvvm`` dialect bindings are built."""
        return gpu is not None and nvvm is not None

    def pipeline_options(self) -> dict[str, str]:
        """``{chip_option: <PREFIX>_ARCH}`` for a ``<PREFIX>_PIPELINE`` override
        when an architecture is set, else nothing."""
        arch = self.dsl.envar.arch if self.dsl is not None else None
        return {self.chip_option: arch} if arch else {}

    def shared_libs(self) -> list[str]:
        """The CUDA runtime library for the kernels' host code, when found."""
        libs = self.dsl.envar.shared_libs if self.dsl is not None else None
        path = cuda_runtime_library(libs)
        return [] if path is None else [path]

    # -- the container: ``gpu.module @kernels`` --------------------------------

    def build_container(self, attrs: dict[str, Any], loc: Any = None) -> Any:
        host_module = ir.InsertionPoint.current.block.owner.operation
        # ``gpu.launch_func``'s verifier demands the marker on the host module.
        host_module.attributes["gpu.container_module"] = ir.UnitAttr.get()
        container = gpu.GPUModuleOp(ir.StringAttr.get("kernels"), loc=loc)
        container.bodyRegion.blocks.append()
        for attr_name, attr in attrs.items():
            container.attributes[attr_name] = (
                attr if isinstance(attr, ir.Attribute) else ir.Attribute.parse(attr)
            )
        return container

    def container_insertion_point(
        self, container: Any, module: Any
    ) -> ir.InsertionPoint:
        if container is None:
            raise DSLRuntimeError("no gpu.module was built for this trace")
        return ir.InsertionPoint(container.bodyRegion.blocks[0])

    def prune_empty_containers(self, module: Any) -> None:
        """Erase the ``gpu.module`` ops without kernels, and the
        ``gpu.container_module`` marker when none remains."""
        remaining = 0

        def visit(op: Any) -> ir.WalkResult:
            nonlocal remaining
            if op.name != "gpu.module":
                return ir.WalkResult.ADVANCE
            if (
                len(op.regions) == 0
                or len(op.regions[0].blocks) == 0
                or len(op.regions[0].blocks[0].operations) == 0
            ):
                op.erase()
            else:
                remaining += 1
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        attributes = module.operation.attributes
        if remaining == 0 and "gpu.container_module" in attributes:
            del attributes["gpu.container_module"]

    def kernel_symbol(self, kernel_name: str) -> ir.Attribute:
        return ir.SymbolRefAttr.get(["kernels", kernel_name])

    # -- the entry protocol ------------------------------------------------------

    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> tuple[Any, ir.Block]:
        """Build ``gpu.func @name(arg_types) attributes {gpu.kernel}`` at the
        current insertion point with its entry block; ``arg_attrs`` is one
        ``ir.DictAttr`` (or ``dict``) per argument."""
        arg_types = list(arg_types)
        attrs = [
            (
                attr
                if isinstance(attr, ir.Attribute)
                else ir.DictAttr.get(dict(attr or {}))
            )
            for attr in (arg_attrs if arg_attrs is not None else [{}] * len(arg_types))
        ]
        fop = gpu.GPUFuncOp(
            ir.FunctionType.get(arg_types, []),
            name,
            arg_attrs=ir.ArrayAttr.get(attrs),
            kernel=True,
            loc=loc,
        )
        fop.sym_visibility = ir.StringAttr.get("public")
        log().debug("gpu.func @%s(%s)", name, ", ".join(map(str, arg_types)))
        return fop, fop.add_entry_block()

    def generate_return(self, op: Any, values: list[Any], loc: Any = None) -> None:
        """Terminate the kernel body with a ``gpu.return``."""
        gpu.ReturnOp([], loc=loc)

    def launch_config(self, func: Any, *args: Any, **kwargs: Any) -> LaunchConfig:
        """What ``.launch(...)`` accepts under this plugin: one :class:`LaunchConfig`
        or its constructor arguments; a launch is synchronous, so ``async_deps``
        must be empty. The core never sees these names."""
        if len(args) == 1 and not kwargs and isinstance(args[0], LaunchConfig):
            config = args[0]
        else:
            config = LaunchConfig(*args, **kwargs)
        if config.async_deps:
            raise DSLUserCodeError(
                GpuDiagId.LAUNCH_STREAM_UNSUPPORTED,
                kernel_name=getattr(func, "__name__", "<kernel>"),
            )
        return config

    def generate_launch(
        self,
        op: Any,
        symbol: ir.Attribute,
        operands: list[Any],
        config: LaunchConfig,
        *,
        loc: Any = None,
    ) -> None:
        """Emit the synchronous ``gpu.launch_func`` of ``op`` for ``config``.

        A static zero grid emits nothing; staged dimensions are guarded by
        ``scf.if (all dims > 0)``. The launch has no result.
        """
        launch_loc = loc
        kernel_name = op.name.value
        if self.dsl.plugins.compiler is not None and not self.dsl.envar.dryrun:
            # A launch that will be compiled needs the target; a trace does not.
            check_arch(self.dsl.envar.arch, var=f"{self.dsl.envar.prefix}_ARCH")

        grid = _dimensions(config.grid, "grid")
        block = _dimensions(config.block, "block")
        cluster = (
            None if config.cluster is None else _dimensions(config.cluster, "cluster")
        )
        if all(isinstance(entry, int) for entry in block):
            op.attributes[
                gpu.GPUFuncOp.KNOWN_BLOCK_SIZE_ATTR_NAME
            ] = ir.DenseI32ArrayAttr.get(list(block))
        if any(entry == 0 for entry in grid if isinstance(entry, int)):
            log().debug(
                "kernel %s: static zero grid %s, launch skipped", kernel_name, grid
            )
            return None

        grid_values, dynamic = _launch_index_values(grid, loc=loc)
        block_values, block_dynamic = _launch_index_values(block, loc=loc)
        dynamic += block_dynamic
        cluster_values = None
        if cluster is not None:
            values, cluster_dynamic = _launch_index_values(cluster, loc=loc)
            cluster_values = tuple(values)
            dynamic += cluster_dynamic
        smem = _smem_operand(config.smem, loc=loc)

        def launch_op() -> None:
            gpu.launch_func(
                symbol,
                tuple(grid_values),
                tuple(block_values),
                list(operands),
                dynamic_shared_memory_size=smem,
                cluster_size=cluster_values,
                loc=launch_loc,
            )

        if dynamic:
            # CUDA rejects a zero-sized launch: skip it at run time when any
            # staged dimension is zero.
            zero = current_emitter().const(0, Int64, loc=loc)
            condition = current_emitter().cmp("gt", dynamic[0], zero, loc=loc)
            for dimension in dynamic[1:]:
                condition = current_emitter().and_(
                    condition,
                    current_emitter().cmp("gt", dimension, zero, loc=loc),
                    loc=loc,
                )
            guard = scf.IfOp(condition, loc=loc)
            with ir.InsertionPoint(guard.then_block):
                launch_op()
                scf.YieldOp([], loc=loc)
        else:
            launch_op()
        return None
