# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The gpu kernels plugin: ``@kernel`` as ``gpu.func`` + ``gpu.launch_func``.

``Kernels`` is a :class:`KernelsPlugin` listed in a DSL's ``Plugins(decorators=[...])``:
it adds the ``@kernel`` decorator, builds the ``gpu.module`` container of a
trace, the kernel function (``gpu.func`` with ``gpu.kernel``,
``known_block_size`` when the block is static) and a synchronous
``gpu.launch_func`` guarded against zero-sized staged dimensions. It names the
``libmlir_cuda_runtime`` library for the execution engine (never bound here)
and checks ``<PREFIX>_ARCH`` when it is installed; the sub-DSL spells MLIR's
``gpu-lower-to-nvvm-pipeline`` with the plugin's ``chip_option`` in its own
``pipeline()``. The DSL owns no CUDA runtime code. The kernel-body index
helpers are ``indices.py`` beside this module. The module imports without the gpu
bindings; ``available()`` says whether the plugin can serve.
"""

from __future__ import annotations

import enum
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

from ...... import _mlir_libs, ir
from .....core.common import DSLRuntimeError, DSLUserCodeError
from .....core.diagnostics import USAGE, DiagCatalog, DiagId, classify
from .....core.mlir_op import current_emitter
from ..launch import KernelsPlugin, LaunchConfig
from .....types.typing import Boolean, Int32, Int64, Integer, Numeric
from .....util.logger import log

try:
    from ......dialects import gpu, nvvm, scf
except ImportError:  # the gpu bindings are not built into this MLIR
    gpu = nvvm = scf = None  # type: ignore[assignment]

__all__ = [
    "GpuDiagId",
    "Kernels",
    "check_arch",
    "cuda_runtime_library",
]

if gpu is not None:
    # The kernel-body ops (``thread_idx`` ...) ride along with the plugin.
    from .indices import (
        GridConstant,
        block_dim,
        block_idx,
        grid_constant,
        grid_dim,
        thread_idx,
    )

    __all__ += [
        "GridConstant",
        "block_dim",
        "block_idx",
        "grid_constant",
        "grid_dim",
        "thread_idx",
    ]


# =============================================================================
# Diagnostics
# =============================================================================


class GpuDiagId(DiagCatalog, enum.Enum):
    """Launch and target diagnostics, rendered ``error[gpu:CODE]``; member name ==
    stable code, value == (message, fixes). Raised here and by the launcher in
    ``launch.py`` (``KernelLauncher``/``KernelsPlugin``) through ``KernelsPlugin.diag``.
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
    LAUNCH_OUTSIDE_JIT = (
        "Kernel `{kernel_name}` is being launched from plain Python, but a kernel can "
        "only be launched from inside a function decorated with `@jit`.",
        ("Wrap the launch in a host function decorated with `@jit` and call that.",),
    )
    LAUNCH_NEVER_ISSUED = (
        "Kernel `{kernel_name}` was called but never launched. Calling a `@kernel` "
        "function only prepares a launch; the kernel does not run until "
        "`.launch(...)` is called on the result.",
        (
            "Launch the kernel, e.g. `{kernel_name}(...).launch(grid=[...], block=[...])`.",
            "If the call is not needed, remove it.",
        ),
    )
    LAUNCH_ALREADY_ISSUED = (
        "Kernel `{kernel_name}` is launched twice from one prepared call; "
        "`{kernel_name}(...)` runs once.",
        (
            "Call `{kernel_name}(...)` again for a second launch, one `.launch(...)` each.",
        ),
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
    CONFIG_UNSUPPORTED_ARCH = (
        "The GPU architecture `{arch}` is not a CUDA target this DSL can compile for; "
        "`{var}` must name one such as `sm_80` or `sm_90a`.",
        ("Set the environment variable `{var}=<arch>`, e.g. `{var}=sm_90a`.",),
    )


classify(
    GpuDiagId,
    USAGE,
    "kernel launch",
    "LAUNCH_INVALID_DIMENSION",
    "LAUNCH_INVALID_GRID",
    "LAUNCH_OUTSIDE_JIT",
    "LAUNCH_NEVER_ISSUED",
    "LAUNCH_ALREADY_ISSUED",
    "LAUNCH_HOST_BUFFER",
    "LAUNCH_STREAM_UNSUPPORTED",
)
classify(GpuDiagId, USAGE, "compile options", "CONFIG_UNSUPPORTED_ARCH")


# =============================================================================
# Target and runtime library
# =============================================================================

_CUDA_RUNTIME_STEM = "mlir_cuda_runtime"
_ARCH_PATTERN = re.compile(r"sm_[0-9]+[af]?")


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
    """Return ``arch`` when it names a CUDA architecture (``sm_[0-9]+[af]?``).

    :param arch: The value of ``var``; ``None`` and ``""`` read as unset
    :param var: The environment variable named in the diagnostic (the
        DSL's ``<PREFIX>_ARCH``)
    :return: ``arch`` unchanged
    :raises DSLUserCodeError: ``gpu:CONFIG_UNSUPPORTED_ARCH`` for anything else
    """
    if not isinstance(arch, str) or _ARCH_PATTERN.fullmatch(arch) is None:
        raise DSLUserCodeError(
            GpuDiagId.CONFIG_UNSUPPORTED_ARCH,
            arch="<unset>" if arch is None or arch == "" else arch,
            var=var,
        )
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
    # The launcher raises the ``LAUNCH_*`` codes through it.
    diag_ids: ClassVar[Any] = GpuDiagId

    def __init__(self, *, chip_option: str = "cubin-chip") -> None:
        super().__init__()
        self.chip_option = chip_option

    @classmethod
    def available(cls) -> bool:
        """Whether the ``gpu`` and ``nvvm`` dialect bindings are built."""
        return gpu is not None and nvvm is not None

    def install(self, dsl: Any) -> None:
        """Bind to ``dsl``: a set ``<PREFIX>_ARCH`` must name a CUDA architecture."""
        super().install(dsl)
        if dsl.envar.arch:
            check_arch(dsl.envar.arch, var=f"{dsl.envar.prefix}_ARCH")

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

    def generate_return(self, op: Any, loc: Any = None) -> None:
        """Terminate the kernel body with a ``gpu.return``."""
        gpu.ReturnOp([], loc=loc)

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

        def emit_launch() -> None:
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
                emit_launch()
                scf.YieldOp([], loc=loc)
        else:
            emit_launch()
        return None
