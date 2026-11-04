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
but the package missing construction raises ``CONFIG_INVALID``.

Exported parameters (in signature order): a DSL scalar annotation
(``a: Int32``) is a ``spec.Var``; a ``Pointer[T]`` annotation or an adapted
``Pointer`` argument is a ``spec.DataPointer`` (the Python call hands the
buffer's address, so numpy arrays and torch tensors work through the usual
adapters); a Meta argument whose value is an int, bool, float or None is a
``spec.Const*`` that the wrapper asserts and does not forward. Keyword-only
parameters follow the positional ones, which is the entry's operand order.
Anything else (a struct or dataclass argument, another Meta type, a pointer
in a non-zero address space, a non-scalar result) leaves that function on
the packed path with a warning; the compile never fails because of the
export.

A call with keyword arguments, or one that leaves a defaulted parameter out,
goes through a wrapper that ``tvm_ffi.utils.kwargs_wrapper`` (apache-tvm-ffi
0.1.11 and newer) builds from the plan's names and defaults
(:func:`make_kwargs_wrapper`); without that utility such a function keeps the
packed path. Keyword arguments are scalars in this version: a buffer passed
by keyword takes the packed path.

A DSL with another calling convention, or whose host entry only takes its
final shape after lowering, subclasses the plugin: :meth:`TvmFfiPlugin.plan`,
:meth:`TvmFfiPlugin.make_call_provider` and ``TvmFfiPlugin.stage``
(``"traced"`` or ``"lowered"``) are its three hooks.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    get_args,
    get_origin,
)

from ..... import ir
from .....dialects import llvm
from ...compiler.jit_executor import JitCompiledFunction
from ....core.common import DSLBaseError, DSLUserCodeError
from ....core.diagnostics import DiagId
from .diagnostics import TvmFfiDiagId
from ....core.plugin import AdapterPlugin
from ....core.arguments import JitArgAdapterRegistry
from ....core.staging import is_reserved_python_func_arg
from . import spec
from .call_provider import DirectCallProvider
from .mlir_builder import MLIRBuilder
from .tvm_ffi_builder import CallProvider, attach_ffi_func
from ....types import typing as t
from ....util.logger import log

if TYPE_CHECKING:
    from ....core.dsl import BaseDSL

__all__ = [
    "ExportPlan",
    "NotExportable",
    "NumericToTVMFFIDtype",
    "TvmFfiJitCompiledFunction",
    "TvmFfiPlugin",
    "available",
    "kwargs_wrapper_available",
    "make_kwargs_wrapper",
    "tvm_ffi_symbol",
]

TVM_FFI_SYMBOL_PREFIX = "__tvm_ffi_"


def available() -> bool:
    """Whether the ``tvm_ffi`` package can be imported (never imported here)."""
    return importlib.util.find_spec("tvm_ffi") is not None


def kwargs_wrapper_available() -> bool:
    """Whether ``tvm_ffi.utils.kwargs_wrapper`` exists (apache-tvm-ffi 0.1.11+).

    Asked when a compiled function is wrapped, not before: locating the
    submodule imports ``tvm_ffi`` itself, which :func:`available` never does.
    """
    if not available():
        return False
    try:
        return importlib.util.find_spec("tvm_ffi.utils.kwargs_wrapper") is not None
    except ImportError:
        return False


def make_kwargs_wrapper(
    function: Callable[..., Any],
    plan: Any,
    *,
    map_dataclass_to_tuple: Sequence[str] = (),
) -> Callable[..., Any]:
    """``function`` behind the Python calling convention ``plan`` describes.

    The wrapper takes the positional parameters by position or name and the
    keyword-only ones by name, fills in the defaults and calls ``function``
    with every argument in order (``tvm_ffi.utils.kwargs_wrapper``).

    :param function: A callable taking all its arguments positionally (a
        ``tvm_ffi.Function``)
    :param plan: Anything with ``arg_names``, ``arg_defaults`` (right-aligned
        to the names, as ``__defaults__``), ``kwonly_names`` and
        ``kwonly_defaults``: an export plan, or a DSL's own record of a
        signature
    :param map_dataclass_to_tuple: Names of the arguments the wrapper unpacks
        from a dataclass into a tuple
    :raises DSLUserCodeError: ``CONFIG_INVALID`` when the installed
        ``tvm_ffi`` has no ``utils.kwargs_wrapper``
    """
    if not kwargs_wrapper_available():
        raise DSLUserCodeError(
            DiagId.CONFIG_INVALID,
            var="apache-tvm-ffi",
            detail="keyword arguments through the exported function need `tvm_ffi.utils.kwargs_wrapper` (version 0.1.11 or newer)",
        )
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper as make_wrapper

    return make_wrapper(
        function,
        list(plan.arg_names),
        tuple(plan.arg_defaults),
        kwonly_names=list(plan.kwonly_names),
        kwonly_defaults=dict(plan.kwonly_defaults),
        map_dataclass_to_tuple=list(map_dataclass_to_tuple) or None,
    )


def tvm_ffi_symbol(function_name: str) -> str:
    """The symbol of the TVM-FFI wrapper of ``function_name``."""
    return TVM_FFI_SYMBOL_PREFIX + function_name


def _host_value(arg: Any) -> Any:
    """``arg`` as the wrapper decodes it: a host DSL scalar as its value."""
    if isinstance(arg, t.Numeric) and not arg.is_staged:
        return arg.value
    return arg


# DSL scalar type -> tvm_ffi dtype spelling.
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
class ExportPlan:
    """What the wrapper of one compiled function expects from a Python call.

    :ivar params: The ``spec`` parameters, in signature order
    :ivar pointer_positions: Positions whose Python argument is a buffer,
        handed over as its address
    :ivar const_positions: Positions whose Python argument is a compile-time
        constant the wrapper asserts
    :ivar result_dtype: The DSL scalar type the result is returned as (the
        packed path's type); None for a ``void`` function
    :ivar result_type: The host entry's MLIR result type, or None
    :ivar arg_names: The positional parameter names, in signature order
    :ivar arg_defaults: The defaults of the trailing positional parameters
        that have one (right-aligned to ``arg_names``, as ``__defaults__``)
    :ivar kwonly_names: The keyword-only parameter names, in signature order;
        their positions follow the positional ones
    :ivar kwonly_defaults: The defaults of the keyword-only parameters that
        have one
    """

    params: list[Any]
    pointer_positions: set[int] = field(default_factory=set)
    const_positions: set[int] = field(default_factory=set)
    result_dtype: type | None = None
    result_type: ir.Type | None = None
    arg_names: list[str] = field(default_factory=list)
    arg_defaults: tuple[Any, ...] = ()
    kwonly_names: list[str] = field(default_factory=list)
    kwonly_defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_kwargs_wrapper(self) -> bool:
        """Whether a Python call may name or leave out an argument."""
        return bool(self.kwonly_names or self.arg_defaults)


class NotExportable(Exception):
    """Raised while planning when a function cannot take the TVM-FFI path."""


# The earlier, private spellings of the plan and its exception.
_ExportPlan = ExportPlan
_NotExportable = NotExportable


def _unwrap_annotation(annotation: Any) -> Any:
    """``annotation`` without an ``Annotated[...]`` wrapper."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _find_host_entry(module: ir.Module, function_name: str) -> ir.Operation | None:
    """The ``llvm.func``/``func.func`` named ``function_name`` in ``module``."""
    return MLIRBuilder.find_func_in_module(
        module, function_name, op_names=("llvm.func", "func.func")
    )


def _entry_types(entry: ir.Operation) -> tuple[list[ir.Type], ir.Type | None]:
    """``(input types, result type or None)`` of an ``llvm.func``/``func.func``.

    :raises NotExportable: A ``func.func`` with more than one result
    """
    function_type = ir.TypeAttr(entry.attributes["function_type"]).value
    if entry.name == "func.func":
        fn = ir.FunctionType(function_type)
        results = list(fn.results)
        if len(results) > 1:
            raise NotExportable("more than one result")
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

    :raises NotExportable: The result is not an exportable scalar
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
    raise NotExportable(f"result type {result_type} is not an exportable scalar")


class TvmFfiJitCompiledFunction(JitCompiledFunction):
    """A compiled function called through its TVM-FFI wrapper.

    ``tvm_ffi_function`` is the ``tvm_ffi.Function`` over the packed
    ``_mlir___tvm_ffi_<name>`` symbol of the engine; ``__call__`` hands it the
    Python arguments, buffers as addresses, and returns the result as the DSL
    scalar the packed path returns. With a ``kwargs_wrapper`` every call goes
    through it, keywords and defaults included; without one a call with
    keyword arguments takes the packed path.

    :param tvm_ffi_function: The ``tvm_ffi.Function`` of the wrapper
    :param export_plan: The plan the wrapper was built from
    :param kwargs_wrapper: ``tvm_ffi_function`` behind the Python calling
        convention (:func:`make_kwargs_wrapper`), when the plan has
        keyword-only or defaulted parameters
    """

    prefers_python_args: ClassVar[bool] = True

    def __init__(
        self,
        *args: Any,
        tvm_ffi_function: Any = None,
        export_plan: ExportPlan | None = None,
        kwargs_wrapper: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tvm_ffi_function = tvm_ffi_function
        self.kwargs_wrapper = kwargs_wrapper
        self._export_plan = export_plan

    @classmethod
    def from_compiled(
        cls,
        compiled: JitCompiledFunction,
        tvm_ffi_function: Any,
        export_plan: ExportPlan,
        kwargs_wrapper: Callable[..., Any] | None = None,
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
            compiled.jit_time_profiling,
            result_ctype=compiled.result_ctype,
            tvm_ffi_function=tvm_ffi_function,
            export_plan=export_plan,
            kwargs_wrapper=kwargs_wrapper,
        )
        # What a kernels plugin recorded on the compiled function travels along.
        wrapped.kernel_info = compiled.kernel_info
        wrapped.has_kernels = compiled.has_kernels
        wrapped.result_spec = getattr(compiled, "result_spec", None)  # type: ignore[attr-defined]
        wrapped.execution_args.set_adapter_scope(
            compiled.execution_args._jit_arg_adapter_scope
        )
        return wrapped

    def _tvm_ffi_args(
        self, plan: ExportPlan, args: tuple[Any, ...]
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
                    continue
                arg = converted[position]
                pointer = arg
                if not isinstance(pointer, t.Pointer):
                    adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
                    pointer = adapter(arg) if adapter is not None else None
                if isinstance(pointer, t.Pointer) and pointer.address is not None:
                    keepalive.append(pointer)
                    converted[position] = pointer.address
        return [_host_value(arg) for arg in converted], keepalive

    @staticmethod
    def _pointer_by_keyword(plan: ExportPlan, kwargs: dict[str, Any]) -> bool:
        """Whether a keyword names a buffer position (not adapted by keyword)."""
        names = [*plan.arg_names, *plan.kwonly_names]
        return any(
            name in names and names.index(name) in plan.pointer_positions
            for name in kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapper; a rejected argument raises ``CALL_REJECTED``.

        The call goes through ``kwargs_wrapper`` when there is one (it binds
        the keywords and fills the defaults); otherwise a call with keywords
        takes the packed path, as does a buffer passed by keyword.
        """
        plan = self._export_plan
        if self.tvm_ffi_function is None or plan is None:
            return super().__call__(*args, **kwargs)
        target = self.kwargs_wrapper
        if kwargs and (target is None or self._pointer_by_keyword(plan, kwargs)):
            return super().__call__(*args, **kwargs)
        if target is None:
            target = self.tvm_ffi_function
        ffi_args, keepalive = self._tvm_ffi_args(plan, args)
        ffi_kwargs = {name: _host_value(value) for name, value in kwargs.items()}
        try:
            result = target(*ffi_args, **ffi_kwargs)
        except DSLBaseError:
            raise
        except Exception as exc:
            raise DSLUserCodeError(
                TvmFfiDiagId.CALL_REJECTED,
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


class TvmFfiPlugin(AdapterPlugin):
    """The TVM-FFI export plugin: the outbound half of the ``adapters`` family.

    The test DSL lists it unconditionally; the resolved record drops it when the
    ``tvm_ffi`` package is not importable (``available()``). Inert unless
    ``{prefix}_ENABLE_TVM_FFI`` is set.

    A DSL specialises the export through three hooks. :meth:`plan` maps the
    Python signature and the trace arguments of a host entry to the wrapper's
    parameters; :meth:`make_call_provider` emits the call from the wrapper
    into that entry (the shipped provider calls a ``func.func`` and returns a
    scalar); :attr:`stage` names the module the wrapper is added to: the
    traced one, before it is hashed (``"traced"``, the default: the wrapper is
    part of the cache key), or the lowered one, after the pass pipeline
    (``"lowered"``: the entry is an ``llvm.func`` by then, so the provider
    calls it as one). The plan is made on the traced module on every call,
    so a cache hit has it too; a lowered-stage wrapper is in the cached
    artifact, and a file cache hit serves it without attaching it again.
    """

    name = "tvm_ffi"
    enabled: bool = False
    stage: ClassVar[Literal["traced", "lowered"]] = "traced"

    @classmethod
    def available(cls) -> bool:
        return available()

    def install(self, dsl: BaseDSL) -> None:
        """Read ``{prefix}_ENABLE_TVM_FFI``.

        :raises DSLUserCodeError: ``CONFIG_INVALID`` when it is set but
            the ``tvm_ffi`` package is not importable
        """
        super().install(dsl)
        self.enabled = bool(dsl.envar.enable_tvm_ffi)
        self._plans: dict[str, ExportPlan] = {}
        if self.enabled and not available():
            raise DSLUserCodeError(
                DiagId.CONFIG_INVALID,
                var=f"{dsl.envar.prefix}_ENABLE_TVM_FFI",
                detail="it is set, but the `tvm_ffi` package is not installed (`pip install apache-tvm-ffi`)",
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

    def plan(
        self,
        dsl: BaseDSL,
        module: ir.Module,
        function_name: str,
        sig: inspect.Signature,
        trace_args: tuple[Any, ...],
        trace_kwargs: dict[str, Any],
    ) -> ExportPlan:
        """Map the Python signature and trace arguments to ``spec`` parameters.

        Called on the traced ``module`` once the host entry ``function_name``
        is built; a DSL with other parameter kinds (its own leaves, another
        pointer rule) overrides it.

        :raises NotExportable: With the reason the function keeps the packed
            path (see the module docstring for the exportable shapes)
        """
        entry = _find_host_entry(module, function_name)
        if entry is None:
            raise NotExportable("host entry not found in the module")
        input_types, result_type = _entry_types(entry)
        plan = ExportPlan(params=[], result_type=result_type)
        plan.result_dtype = _result_dtype(result_type, sig.return_annotation)
        operand_types: list[str] = []
        # The entry takes the keyword-only parameters behind the positional
        # ones, in signature order; the plan's positions follow that order.
        positional: list[inspect.Parameter] = []
        keyword_only: list[inspect.Parameter] = []
        for parameter in sig.parameters.values():
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keyword_only.append(parameter)
            elif parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(parameter)
            else:
                raise NotExportable(f"parameter `{parameter.name}` is variadic")
        if len(trace_args) != len(positional) or any(
            parameter.name not in trace_kwargs for parameter in keyword_only
        ):
            raise NotExportable("the call does not bind every parameter")
        parameters = [*positional, *keyword_only]
        values = [*trace_args, *(trace_kwargs[p.name] for p in keyword_only)]
        plan.arg_names = [parameter.name for parameter in positional]
        plan.arg_defaults = tuple(
            _host_value(parameter.default)
            for parameter in positional
            if parameter.default is not inspect.Parameter.empty
        )
        plan.kwonly_names = [parameter.name for parameter in keyword_only]
        plan.kwonly_defaults = {
            parameter.name: _host_value(parameter.default)
            for parameter in keyword_only
            if parameter.default is not inspect.Parameter.empty
        }
        for position, (parameter, arg) in enumerate(zip(parameters, values)):
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
                    raise NotExportable(
                        f"compile-time argument `{name}` of type {type(arg).__name__}"
                    )
                continue
            scalar: type | None = None
            if isinstance(annotation, t.NumericMeta):
                # An abstract annotation (`Numeric`) names no dtype; the
                # adapted argument carries the one the boundary assigned.
                scalar = (
                    type(arg)
                    if annotation.is_abstract and isinstance(arg, t.Numeric)
                    else annotation
                )
            elif isinstance(arg, t.Numeric) and annotation is inspect.Parameter.empty:
                scalar = type(arg)
            if scalar is not None:
                dtype = NumericToTVMFFIDtype.get(scalar)
                if dtype is None:
                    raise NotExportable(f"scalar type {scalar.__name__} of `{name}`")
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
                    raise NotExportable(
                        f"`{name}` is a pointer in address space {space}"
                    )
                plan.params.append(spec.DataPointer(name))
                plan.pointer_positions.add(position)
                operand_types.append(str(llvm.PointerType.get()))
                continue
            raise NotExportable(f"parameter `{name}` ({type(arg).__name__})")
        entry_types = [str(ty) for ty in input_types]
        if operand_types != entry_types:
            raise NotExportable(
                f"operands {operand_types} do not match the entry {entry_types}"
            )
        return plan

    # The earlier spelling of the planning step.
    _plan = plan

    def make_call_provider(self, plan: ExportPlan, function_name: str) -> CallProvider:
        """The provider that calls the host entry ``function_name`` from its
        wrapper: a ``DirectCallProvider`` over the plan's result type, signed
        as the result dtype says, calling a ``func.func``. A DSL whose entry
        has another shape (an ``llvm.func`` after lowering, a result handed
        back through an operand) returns its own."""
        result_signed = True
        if isinstance(plan.result_dtype, t.IntegerMeta):
            result_signed = bool(plan.result_dtype.signed)
        return DirectCallProvider(
            function_name,
            result_type=plan.result_type,
            result_signed=result_signed,
            callee_kind="func",
        )

    def _attach(self, module: ir.Module, function_name: str, plan: ExportPlan) -> None:
        """Add the wrapper of ``function_name`` to ``module`` from ``plan``."""
        with module.context, module.operation.location:
            attach_ffi_func(
                module,
                function_name,
                plan.params,
                self.make_call_provider(plan, function_name),
                fn_display_name=function_name,
            )
        log().debug("tvm_ffi: attached %s", tvm_ffi_symbol(function_name))

    def attach_to_module(
        self,
        dsl: BaseDSL,
        module: ir.Module,
        function_name: str,
        sig: Any,
        trace_args: tuple[Any, ...],
        trace_kwargs: dict[str, Any],
    ) -> None:
        """Plan the export of ``function_name`` and, in the ``"traced"`` stage,
        add its wrapper when the signature is exportable.

        A function that is not exportable keeps the packed entry; the reason
        is logged as a warning and the compile goes on. The plan is kept for
        :meth:`after_lowering` and :meth:`wrap_compiled_function`.
        """
        if not self.enabled:
            return
        with module.context, module.operation.location:
            try:
                plan = self.plan(
                    dsl, module, function_name, sig, trace_args, trace_kwargs
                )
            except NotExportable as reason:
                log().warning(
                    "tvm_ffi: `%s` is not exported (%s); it keeps the packed entry",
                    function_name,
                    reason,
                )
                return
        self._plans[function_name] = plan
        if self.stage == "traced":
            self._attach(module, function_name, plan)

    def after_lowering(
        self, dsl: BaseDSL, module: ir.Module, function_name: str
    ) -> None:
        """In the ``"lowered"`` stage, add the wrapper planned for
        ``function_name`` to the lowered ``module`` (the pipeline ran, the
        engine is built next). Not reached on a file cache hit: the cached
        module already holds the wrapper."""
        if not self.enabled or self.stage != "lowered":
            return
        plan = self._plans.get(function_name)
        if plan is None:
            return
        self._attach(module, function_name, plan)

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
        kwargs_wrapper = None
        if plan.needs_kwargs_wrapper:
            # Keywords and defaults need the kwargs utility; without it the
            # function keeps the packed entry, which binds them itself.
            if not kwargs_wrapper_available():
                log().warning(
                    "tvm_ffi: `%s` has keyword-only or defaulted parameters and "
                    "the installed tvm_ffi has no `utils.kwargs_wrapper`; it "
                    "keeps the packed entry",
                    jit_function.function_name,
                )
                return jit_function
            kwargs_wrapper = make_kwargs_wrapper(function, plan)
        return TvmFfiJitCompiledFunction.from_compiled(
            jit_function, function, plan, kwargs_wrapper=kwargs_wrapper
        )
