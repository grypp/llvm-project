# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``gpu`` dialect plugin: CUDA kernels as ``gpu.func`` + ``gpu.launch_func``.

``GpuPlugin`` brings the target (a ``gpu.module`` lowered by
``gpu-lower-to-nvvm-pipeline`` for ``{prefix}_ARCH``) and the path of
``libmlir_cuda_runtime`` (handed to the ``ExecutionEngine``, never bound here).
``GpuKernelGenHelper`` emits the kernel function and a synchronous launch
guarded against zero-sized staged dimensions; the index helpers read the NVVM
special registers as ``Int32``. The DSL owns no CUDA runtime code.
"""

from __future__ import annotations

import enum
import os
import re
from pathlib import Path
from typing import Annotated, Any, ClassVar, Optional, TypeAlias, TypeVar, Union

from .... import _mlir_libs, ir
from ....dialects import gpu, nvvm, scf
from .llvm import arith as arith_helper
from ...core.user_op import dsl_user_op
from ...core.common import DSLRuntimeError, DSLUserCodeError, get_current_dsl
from ...core.diagnostics import DiagCatalog, DiagId
from ...core.dsl import BaseDSL, _KernelGenHelper
from ...core.plugin import DialectPlugin
from ...types.typing import Boolean, Int32, Int64, Integer, Numeric
from ...util.logger import log

__all__ = [
    "GridConstant",
    "grid_constant",
    "GpuDiagId",
    "GpuKernelGenHelper",
    "GpuPlugin",
    "available",
    "block_dim",
    "block_idx",
    "check_arch",
    "grid_dim",
    "thread_idx",
]


# =============================================================================
# Diagnostics
# =============================================================================


# =============================================================================
# Kernel-argument annotation markers
# =============================================================================

TY = TypeVar("TY")


class _GridConstantMarker:
    """``Annotated`` marker that tags a kernel argument as a grid constant.

    ``BaseDSL._extract_annotation_markers`` turns it into one
    ``{cuda.grid_constant}`` argument attribute on the generated kernel.
    """

    def __extract_mlir_attributes__(self) -> list:
        return [ir.DictAttr.get({"cuda.grid_constant": ir.UnitAttr.get()})]


# The one marker instance: ``Annotated[Int32, grid_constant]``.
grid_constant: _GridConstantMarker = _GridConstantMarker()

# ``GridConstant[T]`` spells ``Annotated[T, grid_constant]``.
GridConstant: TypeAlias = Annotated[TY, grid_constant]


class GpuDiagId(DiagCatalog, enum.Enum):
    """Launch and target diagnostics, rendered ``error[gpu:CODE]``; member name ==
    stable code, value == (message, fixes). Raised here and by ``BaseDSL``'s launch path.
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


# =============================================================================
# Target and runtime library
# =============================================================================

# ``available`` runs in the DSL class body, before any environment manager
# exists; the manager exposes the same ``ARCH``/``LIBS`` under its prefix.
_ENV_PREFIX = "MLIR_DSL"
_CUDA_RUNTIME_STEM = "mlir_cuda_runtime"
_ARCH_PATTERN = re.compile(r"sm_[0-9]+[af]?")


def _find_cuda_runtime(libs: Optional[str] = None) -> Optional[str]:
    """Return the path of ``libmlir_cuda_runtime``, or ``None``: a path lookup only.

    The entry naming it in ``libs`` (a ``{prefix}_LIBS`` value; ``MLIR_DSL_LIBS``
    when ``None``), else ``MLIR_CUDA_RUNTIME``, else the copy next to ``mlir._mlir_libs``.
    """
    if libs is None:
        libs = os.environ.get(f"{_ENV_PREFIX}_LIBS", "")
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


def available() -> bool:
    """Whether a DSL class should list ``GpuPlugin()``: the runtime library is
    findable or ``MLIR_DSL_ARCH`` is set. Only the listing is conditional."""
    return _find_cuda_runtime() is not None or bool(
        os.environ.get(f"{_ENV_PREFIX}_ARCH")
    )


def check_arch(arch: Any, *, var: str = f"{_ENV_PREFIX}_ARCH") -> str:
    """Return ``arch`` when it names a CUDA architecture (``sm_[0-9]+[af]?``).

    :param arch: The value of ``var``; ``None`` and ``""`` read as unset
    :param var: The environment variable named in the diagnostic
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
# Kernel-body index helpers
# =============================================================================

_SREGS = {
    "thread_idx": "tid",
    "block_idx": "ctaid",
    "block_dim": "ntid",
    "grid_dim": "nctaid",
}


def _in_kernel_body() -> bool:
    """Whether the current insertion point sits inside a ``gpu.func``."""
    if ir.Context.current is None:
        return False
    try:
        op = ir.InsertionPoint.current.block.owner
    except ValueError:
        return False
    while op is not None:
        if op.operation.name == gpu.GPUFuncOp.OPERATION_NAME:
            return True
        op = op.operation.parent
    return False


def _sreg_indices(api: str, *, loc: Any, ip: Any) -> tuple[Int32, Int32, Int32]:
    """Read the ``x``/``y``/``z`` special register of ``api`` as ``Int32``."""
    if not _in_kernel_body():
        decorator = getattr(get_current_dsl(), "device_jit_decorator_name", "@kernel")
        raise DSLUserCodeError(
            DiagId.CALL_OUTSIDE_JIT, api=f"{api}()", decorator=decorator
        )
    x, y, z = (
        Int32(getattr(nvvm, f"read_ptx_sreg_{_SREGS[api]}_{axis}")(loc=loc, ip=ip))
        for axis in "xyz"
    )
    return x, y, z


@dsl_user_op
def thread_idx(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` index of this thread in its block (``%tid``)."""
    return _sreg_indices("thread_idx", loc=loc, ip=ip)


@dsl_user_op
def block_idx(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` index of this block in the grid (``%ctaid``)."""
    return _sreg_indices("block_idx", loc=loc, ip=ip)


@dsl_user_op
def block_dim(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` size of a block in threads (``%ntid``)."""
    return _sreg_indices("block_dim", loc=loc, ip=ip)


@dsl_user_op
def grid_dim(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` size of the grid in blocks (``%nctaid``)."""
    return _sreg_indices("grid_dim", loc=loc, ip=ip)


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
            values.append(arith_helper.const(entry, Int64, loc=loc))
            continue
        signed = type(entry).signed if isinstance(entry, Integer) else None
        values.append(arith_helper.cast(entry, Int64, signed=signed, loc=loc))
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
        return arith_helper.const(smem, Int32, loc=loc)
    return arith_helper.cast(smem, Int32, loc=loc)


# =============================================================================
# Kernel generation helper
# =============================================================================


class GpuKernelGenHelper(_KernelGenHelper):
    """``_KernelGenHelper`` over the ``gpu`` dialect: ``gpu.func`` (``gpu.kernel``,
    ``known_block_size`` when the block is static) inside the DSL's ``gpu.module``,
    and a synchronous ``gpu.launch_func`` with ``i64`` dimensions wrapped in an
    ``scf.if`` when any is staged. ``kernel_launcher``'s default helper once
    ``GpuPlugin`` is installed; a sub-DSL overrides the four methods for its own op."""

    # The launch driver of ``core/dsl.py`` raises the ``LAUNCH_*`` codes through it.
    diag_ids: ClassVar[Any] = GpuDiagId

    # -- the kernel container: ``gpu.module @kernels`` -------------------------

    @classmethod
    def build_container(cls, attrs: dict[str, Any], loc: Any = None) -> Any:
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

    @classmethod
    def container_insertion_point(
        cls, container: Any, module: Any
    ) -> ir.InsertionPoint:
        if container is None:
            raise DSLRuntimeError("no gpu.module was built for this trace")
        return ir.InsertionPoint(container.bodyRegion.blocks[0])

    @classmethod
    def prune_empty_containers(cls, module: Any) -> None:
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

    @classmethod
    def kernel_symbol(cls, kernel_name: str) -> ir.Attribute:
        return ir.SymbolRefAttr.get(["kernels", kernel_name])

    def __init__(self, dsl: Optional[BaseDSL] = None) -> None:
        super().__init__()
        self.dsl = dsl if dsl is not None else get_current_dsl()
        if self.dsl is None:
            raise DSLRuntimeError("GpuKernelGenHelper needs a DSL, but none is tracing")
        self.arg_types: list[ir.Type] = []
        self.entry_block: Optional[ir.Block] = None

    def generate_func_op(
        self,
        arg_types: list[Any],
        arg_attrs: list[Any],
        kernel_name: str,
        loc: Any = None,
    ) -> Any:
        """Build ``gpu.func @kernel_name(arg_types) attributes {gpu.kernel}`` at the current
        insertion point; ``arg_attrs`` is one ``ir.DictAttr`` (or ``dict``) per argument.
        """
        super().generate_func_op(arg_types, arg_attrs, kernel_name, loc)
        self.arg_types = list(arg_types)
        attrs = [
            (
                attr
                if isinstance(attr, ir.Attribute)
                else ir.DictAttr.get(dict(attr or {}))
            )
            for attr in (
                arg_attrs if arg_attrs is not None else [{}] * len(self.arg_types)
            )
        ]
        self.func_type = ir.FunctionType.get(self.arg_types, [])
        self.func_op = gpu.GPUFuncOp(
            self.func_type,
            kernel_name,
            arg_attrs=ir.ArrayAttr.get(attrs),
            kernel=True,
            loc=loc,
        )
        log().debug(
            "gpu.func @%s(%s)", kernel_name, ", ".join(map(str, self.arg_types))
        )
        return self.func_op

    def generate_func_ret_op(self, loc: Any = None, ip: Any = None) -> Any:
        """Terminate the kernel body with a ``gpu.return``."""
        return gpu.ReturnOp([], loc=loc, ip=ip)

    def get_func_body_start(self) -> ir.Block:
        """The kernel's entry block, created on first use; its arguments are the
        block args ``kernel_launcher`` hands to ``tree_unflatten``."""
        if self.func_op is None:
            raise DSLRuntimeError(
                "generate_func_op must run before get_func_body_start"
            )
        if self.entry_block is None:
            self.entry_block = self.func_op.add_entry_block()
        return self.entry_block

    def generate_launch_op(self, *args: Any, **kwargs: Any) -> None:
        """Emit the synchronous ``gpu.launch_func`` for ``requiredArgs.config``.

        Keywords as ``kernel_launcher`` passes them: ``kernelSym``, ``kernelOperands``,
        ``requiredArgs`` (its ``config`` is the ``LaunchConfig``), ``optionalArgs``,
        ``loc``, ``launch_loc``. A static zero grid emits nothing; staged dimensions
        are guarded by ``scf.if (all dims > 0)``. The launch has no result.
        """
        kernel_sym = kwargs.get("kernelSym")
        kernel_operands = kwargs.get("kernelOperands")
        config = getattr(kwargs.get("requiredArgs"), "config", None)
        loc = kwargs.get("loc")
        launch_loc = kwargs.get("launch_loc") or loc
        if kernel_sym is None or kernel_operands is None or self.func_op is None:
            raise DSLRuntimeError(
                "generate_launch_op needs kernelSym and kernelOperands after generate_func_op"
            )
        if (
            config is None
            or not hasattr(config, "grid")
            or not hasattr(config, "block")
        ):
            raise DSLRuntimeError(
                f"generate_launch_op expects a LaunchConfig, got {type(config).__name__}"
            )
        kernel_name = self.func_op.name.value
        check_arch(self.dsl.envar.arch, var=f"{self.dsl.envar.prefix}_ARCH")
        if getattr(config, "async_deps", None):
            raise DSLUserCodeError(
                GpuDiagId.LAUNCH_STREAM_UNSUPPORTED, kernel_name=kernel_name
            )

        grid = _dimensions(config.grid, "grid")
        block = _dimensions(config.block, "block")
        cluster = getattr(config, "cluster", None)
        cluster = None if cluster is None else _dimensions(cluster, "cluster")
        if all(isinstance(entry, int) for entry in block):
            self.func_op.attributes[
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
        smem = _smem_operand(getattr(config, "smem", None), loc=loc)

        def emit_launch() -> None:
            gpu.launch_func(
                kernel_sym,
                tuple(grid_values),
                tuple(block_values),
                list(kernel_operands),
                dynamic_shared_memory_size=smem,
                cluster_size=cluster_values,
                loc=launch_loc,
            )

        if dynamic:
            # CUDA rejects a zero-sized launch: skip it at run time when any
            # staged dimension is zero.
            zero = arith_helper.const(0, Int64, loc=loc)
            condition = arith_helper.cmp("gt", dynamic[0], zero, loc=loc)
            for dimension in dynamic[1:]:
                condition = arith_helper.and_(
                    condition, arith_helper.cmp("gt", dimension, zero, loc=loc), loc=loc
                )
            guard = scf.IfOp(condition, loc=loc)
            with ir.InsertionPoint(guard.then_block):
                emit_launch()
                scf.YieldOp([], loc=loc)
        else:
            emit_launch()
        return None


# =============================================================================
# Plugin
# =============================================================================


class GpuPlugin(DialectPlugin):
    """The ``gpu`` dialect plugin: target, pipeline and runtime library. Listed on
    the DSL class, ``plugins = [GpuPlugin()] if available() else []``. ``install``
    provides ``GpuKernelGenHelper`` as the DSL's kernel
    generation helper, and checks ``{prefix}_ARCH`` when set; unset (or empty) is diagnosed
    at the first launch, so a CPU-only run constructs. ``register_dialects`` stays
    the no-op: ``gpu`` and ``nvvm`` are on every context."""

    name = "gpu"
    kernel_gen_helper = GpuKernelGenHelper

    @classmethod
    def available(cls) -> bool:
        return available()

    def install(self, dsl: BaseDSL) -> None:
        super().install(dsl)
        arch = dsl.envar.arch
        if arch:
            check_arch(arch, var=f"{dsl.envar.prefix}_ARCH")
        log().debug("gpu plugin installed on %s [arch=%s]", dsl.name, arch)

    def pipeline_passes(self) -> list[str]:
        """``gpu-lower-to-nvvm-pipeline{<pass_sm_arch_name>=<arch>}``, composed ahead of
        the core list so the pipeline sees the kernel body in its high-level form.
        Empty while ``{prefix}_ARCH`` is unset: such a trace holds no kernel (a
        launch has already raised ``CONFIG_UNSUPPORTED_ARCH``), so the core list
        alone lowers it."""
        if self.dsl is None:
            raise DSLRuntimeError("GpuPlugin is not installed on a DSL")
        arch = self.dsl.envar.arch
        if not arch:
            return []
        return [f"gpu-lower-to-nvvm-pipeline{{{self.dsl.pass_sm_arch_name}={arch}}}"]

    def shared_libs(self) -> list[str]:
        """The path of ``libmlir_cuda_runtime`` for the ``ExecutionEngine``, if found;
        the generated host code calls into it, the DSL never does."""
        libs = self.dsl.envar.shared_libs if self.dsl is not None else None
        path = _find_cuda_runtime(libs)
        return [] if path is None else [path]
