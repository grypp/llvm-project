# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The TVM-FFI export plugin: every compiled function also under the TVM-FFI ABI.

With ``{prefix}_ENABLE_TVM_FFI`` set the plugin adds ``llvm.func
@__tvm_ffi_<name>`` next to each host entry (``tvm_ffi_builder``), a wrapper
that decodes and checks the ``TVMFFIAny`` arguments (type, value of the
compile-time constants) and calls the entry; after the JIT the compiled
function becomes a :class:`TvmFfiJitCompiledFunction` whose calls go through
``tvm_ffi.Function``, so the same symbol serves Python, PyTorch (DLPack) and
any other TVM-FFI host without the DSL's marshalling. The ``tvm_ffi`` package
is imported lazily; without it the plugin is inert, and with the variable set
but the package missing construction raises ``CONFIG_MISSING_TVM_FFI``.

Exported parameters (in signature order): a DSL scalar annotation
(``a: Int32``) is a ``spec.Var``; a ``Pointer[T]`` annotation or an adapted
``Pointer`` argument is a ``spec.DataPointer`` (the Python call hands the
buffer's address, so numpy arrays and torch tensors work through the usual
adapters); a Meta argument whose value is an int, bool, float or None is a
``spec.Const*`` that the wrapper asserts and does not forward. Anything else
(a struct or dataclass argument, another Meta type, keyword-only parameters,
a pointer in a non-zero address space, a non-scalar result) leaves that
function on the packed path with a warning; the compile never fails because
of the export. Keyword arguments are not supported by the exported call (v1);
a call with keywords takes the packed path.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, get_args, get_origin

from ..... import ir
from .....dialects import llvm
from ....compiler.jit_executor import JitCompiledFunction
from ....core.common import DSLBaseError, DSLUserCodeError
from ....core.diagnostics import DiagId
from ....core.plugin import Plugin
from ....runtime.jit_arg_adapters import (
    JitArgAdapterRegistry,
    is_reserved_python_func_arg,
)
from . import spec
from .call_provider import DirectCallProvider
from .tvm_ffi_builder import attach_ffi_func
from ....types import typing as t
from ....util.logger import log

if TYPE_CHECKING:
    from ....core.dsl import BaseDSL

__all__ = [
    "NumericToTVMFFIDtype",
    "TvmFfiJitCompiledFunction",
    "TvmFfiPlugin",
    "available",
    "tvm_ffi_symbol",
]

TVM_FFI_SYMBOL_PREFIX = "__tvm_ffi_"


def available() -> bool:
    """Whether the ``tvm_ffi`` package can be imported (never imported here)."""
    return importlib.util.find_spec("tvm_ffi") is not None


def tvm_ffi_symbol(function_name: str) -> str:
    """The symbol of the TVM-FFI wrapper of ``function_name``."""
    return TVM_FFI_SYMBOL_PREFIX + function_name


# DSL scalar type -> tvm_ffi dtype spelling (DkgDSL's ``NumericToTVMFFIDtype``).
# Only the types whose MLIR type the wrapper's decoders produce are listed:
# the wrapper narrows ints to the dtype's width and reads floats as f16/bf16/
# f32/f64; sub-byte and 128-bit integers, tf32 and the 8-bit floats are not
# exportable scalars.
NumericToTVMFFIDtype: dict[type, str] = {
    t.Boolean: "bool",
    t.Int8: "int8",
    t.Int16: "int16",
    t.Int32: "int32",
    t.Int64: "int64",
    t.Uint8: "uint8",
    t.Uint16: "uint16",
    t.Uint32: "uint32",
    t.Uint64: "uint64",
    t.Float16: "float16",
    t.BFloat16: "bfloat16",
    t.Float32: "float32",
    t.Float64: "float64",
}


@dataclass
class _ExportPlan:
    """What the wrapper of one compiled function expects from a Python call.

    :ivar params: The ``spec`` parameters, in signature order
    :ivar pointer_positions: Positions whose Python argument is a buffer,
        handed over as its address
    :ivar const_positions: Positions whose Python argument is a compile-time
        constant the wrapper asserts
    :ivar result_dtype: The DSL scalar type the result is returned as (the
        packed path's type); None for a ``void`` function
    :ivar result_type: The host entry's MLIR result type, or None
    """

    params: list[Any]
    pointer_positions: set[int] = field(default_factory=set)
    const_positions: set[int] = field(default_factory=set)
    result_dtype: type | None = None
    result_type: ir.Type | None = None


class _NotExportable(Exception):
    """Raised while planning when a function cannot take the TVM-FFI path."""


def _unwrap_annotation(annotation: Any) -> Any:
    """``annotation`` without an ``Annotated[...]`` wrapper."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _find_host_entry(module: ir.Module, function_name: str) -> ir.Operation | None:
    """The ``llvm.func``/``func.func`` named ``function_name`` in ``module``."""
    for op in module.body.operations:
        operation = op.operation
        if operation.name not in ("llvm.func", "func.func"):
            continue
        if "sym_name" not in operation.attributes:
            continue
        if ir.StringAttr(operation.attributes["sym_name"]).value == function_name:
            return operation
    return None


def _entry_types(entry: ir.Operation) -> tuple[list[ir.Type], ir.Type | None]:
    """``(input types, result type or None)`` of an ``llvm.func``/``func.func``.

    :raises _NotExportable: A ``func.func`` with more than one result
    """
    function_type = ir.TypeAttr(entry.attributes["function_type"]).value
    if entry.name == "func.func":
        fn = ir.FunctionType(function_type)
        results = list(fn.results)
        if len(results) > 1:
            raise _NotExportable("more than one result")
        return list(fn.inputs), results[0] if results else None
    fn = llvm.FunctionType(function_type)
    return_type = fn.return_type
    if str(return_type) == "!llvm.void":
        return_type = None
    return list(fn.inputs), return_type


def _result_dtype(result_type: ir.Type | None, return_annotation: Any) -> type | None:
    """The DSL scalar type the wrapper's result is returned as.

    The Python return annotation decides when it is an exportable scalar of
    the entry's MLIR type (MLIR integers are signless, so ``i8`` alone cannot
    tell ``Uint8`` from ``Int8``); otherwise the first table entry of that
    MLIR type is used.

    :raises _NotExportable: The result is not an exportable scalar
    """
    if result_type is None:
        return None
    annotation = _unwrap_annotation(return_annotation)
    if (
        isinstance(annotation, t.NumericMeta)
        and annotation in NumericToTVMFFIDtype
        and str(annotation.mlir_type) == str(result_type)
    ):
        return annotation
    for dtype in NumericToTVMFFIDtype:
        if str(dtype.mlir_type) == str(result_type):
            return dtype
    raise _NotExportable(f"result type {result_type} is not an exportable scalar")


class TvmFfiJitCompiledFunction(JitCompiledFunction):
    """A compiled function called through its TVM-FFI wrapper.

    ``tvm_ffi_function`` is the ``tvm_ffi.Function`` over the packed
    ``_mlir___tvm_ffi_<name>`` symbol of the engine; ``__call__`` hands it the
    Python arguments, buffers as addresses, and returns the result as the DSL
    scalar the packed path returns. A call with keyword arguments takes the
    packed path.

    :param tvm_ffi_function: The ``tvm_ffi.Function`` of the wrapper
    :param export_plan: The plan the wrapper was built from
    """

    prefers_python_args: ClassVar[bool] = True

    def __init__(
        self,
        *args: Any,
        tvm_ffi_function: Any = None,
        export_plan: _ExportPlan | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tvm_ffi_function = tvm_ffi_function
        self._export_plan = export_plan

    @classmethod
    def from_compiled(
        cls,
        compiled: JitCompiledFunction,
        tvm_ffi_function: Any,
        export_plan: _ExportPlan,
    ) -> "TvmFfiJitCompiledFunction":
        """Rebuild ``compiled`` as a TVM-FFI calling function.

        The engine, the packed entry, the signature, the result spec and the
        adapter scope are carried over, so the packed path stays available.
        """
        wrapped = cls(
            compiled.ir_module,
            compiled.engine,
            compiled.capi_func,
            compiled.execution_args.original_signature,
            compiled.function_name,
            compiled.kernel_info,
            compiled.jit_time_profiling,
            result_ctype=compiled.result_ctype,
            has_kernels=compiled.has_kernels,
            tvm_ffi_function=tvm_ffi_function,
            export_plan=export_plan,
        )
        wrapped.result_spec = getattr(compiled, "result_spec", None)  # type: ignore[attr-defined]
        wrapped.execution_args.set_adapter_scope(
            compiled.execution_args._jit_arg_adapter_scope
        )
        return wrapped

    def _tvm_ffi_args(
        self, plan: _ExportPlan, args: tuple[Any, ...]
    ) -> tuple[list[Any], list[Any]]:
        """The call arguments for the wrapper and the objects to keep alive.

        A buffer at a pointer position is adapted (numpy, torch, ...) and
        handed over as its address; a host DSL scalar as its Python value.
        """
        converted = list(args)
        keepalive: list[Any] = []
        with JitArgAdapterRegistry.using_scope(
            self.execution_args._jit_arg_adapter_scope
        ):
            for position in plan.pointer_positions:
                if position >= len(converted):
                    break
                arg = converted[position]
                pointer = arg
                if not isinstance(pointer, t.Pointer):
                    adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
                    pointer = adapter(arg) if adapter is not None else None
                if isinstance(pointer, t.Pointer) and pointer.address is not None:
                    keepalive.append(pointer)
                    converted[position] = pointer.address
        for position, arg in enumerate(converted):
            if isinstance(arg, t.Numeric) and not arg.is_staged:
                converted[position] = arg.value
        return converted, keepalive

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapper; a rejected argument raises ``CALL_TVM_FFI_ARGS``."""
        plan = self._export_plan
        if kwargs or self.tvm_ffi_function is None or plan is None:
            return super().__call__(*args, **kwargs)
        ffi_args, keepalive = self._tvm_ffi_args(plan, args)
        try:
            result = self.tvm_ffi_function(*ffi_args)
        except DSLBaseError:
            raise
        except Exception as exc:
            raise DSLUserCodeError(
                DiagId.CALL_TVM_FFI_ARGS,
                function_name=self.function_name,
                detail=str(exc),
                cause=exc,
            ) from exc
        finally:
            del keepalive
        dtype = plan.result_dtype
        if dtype is None or result is None:
            return result
        return dtype(result)


class TvmFfiPlugin(Plugin):
    """The TVM-FFI export plugin, listed on the DSL class when the package is
    importable (``available()``). Inert unless ``{prefix}_ENABLE_TVM_FFI`` is set.
    """

    name = "tvm_ffi"
    enabled: bool = False

    @classmethod
    def available(cls) -> bool:
        return available()

    def install(self, dsl: BaseDSL) -> None:
        """Read ``{prefix}_ENABLE_TVM_FFI``.

        :raises DSLUserCodeError: ``CONFIG_MISSING_TVM_FFI`` when it is set but
            the ``tvm_ffi`` package is not importable
        """
        super().install(dsl)
        self.enabled = bool(dsl.envar.enable_tvm_ffi)
        self._plans: dict[str, _ExportPlan] = {}
        if self.enabled and not available():
            raise DSLUserCodeError(
                DiagId.CONFIG_MISSING_TVM_FFI, var=f"{dsl.envar.prefix}_ENABLE_TVM_FFI"
            )
        log().debug(
            "tvm_ffi plugin installed on %s [enabled=%s]", dsl.name, self.enabled
        )

    def shared_libs(self) -> list[str]:
        """``libtvm_ffi``: the wrapper calls its error and stream entry points."""
        if not self.enabled:
            return []
        from tvm_ffi import libinfo

        return [libinfo.find_libtvm_ffi()]

    # -- planning -----------------------------------------------------------------

    def _plan(
        self,
        dsl: BaseDSL,
        module: ir.Module,
        function_name: str,
        sig: inspect.Signature,
        trace_args: tuple[Any, ...],
        trace_kwargs: dict[str, Any],
    ) -> _ExportPlan:
        """Map the Python signature and trace arguments to ``spec`` parameters.

        :raises _NotExportable: With the reason the function keeps the packed
            path (see the module docstring for the exportable shapes)
        """
        if trace_kwargs:
            raise _NotExportable("keyword-only parameters are not exported")
        entry = _find_host_entry(module, function_name)
        if entry is None:
            raise _NotExportable("host entry not found in the module")
        input_types, result_type = _entry_types(entry)
        plan = _ExportPlan(params=[], result_type=result_type)
        plan.result_dtype = _result_dtype(result_type, sig.return_annotation)
        operand_types: list[str] = []
        parameters = list(sig.parameters.values())
        if len(trace_args) != len(parameters):
            raise _NotExportable("the call does not bind every parameter positionally")
        for position, (parameter, arg) in enumerate(zip(parameters, trace_args)):
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise _NotExportable(f"parameter `{parameter.name}` is not positional")
            name = parameter.name
            annotation = _unwrap_annotation(parameter.annotation)
            if is_reserved_python_func_arg(
                position, name, None
            ) or dsl._is_meta_argument(arg, annotation):
                plan.const_positions.add(position)
                if arg is None:
                    plan.params.append(spec.ConstNone(name))
                elif isinstance(arg, bool):
                    plan.params.append(spec.ConstBool(name, arg))
                elif isinstance(arg, int):
                    plan.params.append(spec.ConstInt(name, arg))
                elif isinstance(arg, float):
                    plan.params.append(spec.ConstFloat(name, arg))
                else:
                    raise _NotExportable(
                        f"compile-time argument `{name}` of type {type(arg).__name__}"
                    )
                continue
            scalar: type | None = None
            if isinstance(annotation, t.NumericMeta):
                scalar = annotation
            elif isinstance(arg, t.Numeric) and annotation is inspect.Parameter.empty:
                scalar = type(arg)
            if scalar is not None:
                dtype = NumericToTVMFFIDtype.get(scalar)
                if dtype is None:
                    raise _NotExportable(f"scalar type {scalar.__name__} of `{name}`")
                plan.params.append(spec.Var(name, dtype))
                operand_types.append(str(scalar.mlir_type))
                continue
            if isinstance(annotation, t.TypedPointer) or isinstance(arg, t.Pointer):
                space = (
                    annotation.space
                    if isinstance(annotation, t.TypedPointer)
                    else arg.space
                )
                if space != 0:
                    raise _NotExportable(
                        f"`{name}` is a pointer in address space {space}"
                    )
                plan.params.append(spec.DataPointer(name))
                plan.pointer_positions.add(position)
                operand_types.append(str(llvm.PointerType.get()))
                continue
            raise _NotExportable(f"parameter `{name}` ({type(arg).__name__})")
        entry_types = [str(ty) for ty in input_types]
        if operand_types != entry_types:
            raise _NotExportable(
                f"operands {operand_types} do not match the entry {entry_types}"
            )
        return plan

    def attach_to_module(
        self,
        dsl: BaseDSL,
        module: ir.Module,
        function_name: str,
        sig: Any,
        trace_args: tuple[Any, ...],
        trace_kwargs: dict[str, Any],
    ) -> None:
        """Add the wrapper for ``function_name`` when the signature is exportable.

        A function that is not exportable keeps the packed entry; the reason
        is logged as a warning and the compile goes on.
        """
        if not self.enabled:
            return
        with module.context, module.operation.location:
            try:
                plan = self._plan(
                    dsl, module, function_name, sig, trace_args, trace_kwargs
                )
            except _NotExportable as reason:
                log().warning(
                    "tvm_ffi: `%s` is not exported (%s); it keeps the packed entry",
                    function_name,
                    reason,
                )
                return
            result_signed = True
            if isinstance(plan.result_dtype, t.IntegerMeta):
                result_signed = bool(plan.result_dtype.signed)
            provider = DirectCallProvider(
                function_name,
                result_type=plan.result_type,
                result_signed=result_signed,
                callee_kind="func",
            )
            attach_ffi_func(
                module,
                function_name,
                plan.params,
                provider,
                fn_display_name=function_name,
            )
        self._plans[function_name] = plan
        log().debug("tvm_ffi: attached %s", tvm_ffi_symbol(function_name))

    def wrap_compiled_function(self, dsl: BaseDSL, jit_function: Any) -> Any:
        """Return ``jit_function`` as a :class:`TvmFfiJitCompiledFunction`.

        The wrapper symbol is looked up in the engine; when it is missing the
        compiled function is returned unchanged (with a warning).
        """
        plan = self._plans.get(getattr(jit_function, "function_name", ""))
        if (
            not self.enabled
            or plan is None
            or not isinstance(jit_function, JitCompiledFunction)
        ):
            return jit_function
        engine = jit_function.engine
        if engine is None:
            return jit_function
        import tvm_ffi

        symbol = tvm_ffi_symbol(jit_function.function_name)
        try:
            address = engine.raw_lookup(symbol)
        except Exception as exc:  # a missing wrapper keeps the packed entry
            log().warning("tvm_ffi: lookup of `%s` failed (%s)", symbol, exc)
            return jit_function
        if not address:
            log().warning("tvm_ffi: `%s` has no address in the engine", symbol)
            return jit_function
        function = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            address, keep_alive_object=engine
        )
        return TvmFfiJitCompiledFunction.from_compiled(jit_function, function, plan)
