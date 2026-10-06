# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""LLVM-dialect type and operation helpers shared by the TVM-FFI builders.

:class:`MLIRTypeBuilder` caches the scalar, pointer and function types the
wrapper emits; :class:`MLIRBuilder` adds constants, comparisons, branches,
``getelementptr``, global strings, function declarations and ``alloca``
packing. Every statement builder expects an active insertion point in the
block it extends.
"""

from collections.abc import Sequence
from typing import Any, Optional

from ..... import ir
from .....dialects import llvm
from ....core.common import DSLRuntimeError


class MLIRTypeBuilder:
    """The LLVM-dialect types the builders use, created once per context."""

    BRANCH_WEIGHTS_LIKELY = (2000, 1)

    def __init__(self) -> None:
        self.i32_type = ir.IntegerType.get_signless(32)
        self.ui32_type = ir.IntegerType.get_unsigned(32)
        self.i64_type = ir.IntegerType.get_signless(64)
        self.i16_type = ir.IntegerType.get_signless(16)
        self.i8_type = ir.IntegerType.get_signless(8)
        self.i1_type = ir.IntegerType.get_signless(1)
        self.f16_type = ir.Type.parse("f16")
        self.bf16_type = ir.Type.parse("bf16")
        self.f32_type = ir.Type.parse("f32")
        self.f64_type = ir.Type.parse("f64")
        self.ptr_type = llvm.PointerType.get()
        # The bindings expose no constructor for the void type.
        self.void_type = ir.Type.parse("!llvm.void")
        self.llvm_internal_linkage = ir.Attribute.parse("#llvm.linkage<internal>")

    def ptr_type_with_address_space(
        self, address_space: Optional[int] = None
    ) -> ir.Type:
        """``!llvm.ptr<N>``; None or 0 is the generic ``!llvm.ptr``."""
        if address_space is None or address_space == 0:
            return self.ptr_type
        return llvm.PointerType.get(address_space=address_space)

    def as_attr(self, tp: ir.Type) -> ir.TypeAttr:
        """Wrap ``tp`` in a ``TypeAttr``."""
        return ir.TypeAttr.get(tp)

    def struct_type(
        self,
        *,
        name: Optional[str] = None,
        fields: Sequence[ir.Type] = (),
        packed: bool = False,
    ) -> ir.Type:
        """Get or create an LLVM struct type.

        :param name: The identified struct's name; None makes a literal struct,
            identified by its fields alone
        :param fields: The field types in order
        :param packed: No padding between fields (else fields are aligned)
        :return: The ``!llvm.struct`` type
        """
        if name is None:
            return llvm.StructType.get_literal(fields, packed=packed)
        return llvm.StructType.new_identified(name, fields, packed=packed)

    def func_type(self, *, params: Sequence[ir.Type] = (), ret: ir.Type) -> ir.Type:
        """The ``!llvm.func<ret (params)>`` type.

        :param params: The parameter types
        :param ret: The return type (``void_type`` for none)
        """
        # The bindings expose no constructor for the LLVM function type.
        return ir.Type.parse(
            "!llvm.func<{} ({})>".format(str(ret), ", ".join(map(str, params)))
        )


class MLIRBuilder(MLIRTypeBuilder):
    """Constants, expressions and statements of the LLVM dialect.

    Convention: every statement builder emits at the active insertion point.
    A builder that ends the current block returns the block to continue in
    without setting an insertion point there.
    """

    MLIR_DYNAMIC_INDEX = -(2**31)

    def __init__(self) -> None:
        super().__init__()
        self.module: Optional[ir.Module] = None
        # Global string content -> symbol, so repeated messages share a global.
        self.const_str_table: dict[str, str] = {}

    # -- constants ------------------------------------------------------------

    def integer_constant(self, tp: ir.Type, value: int) -> ir.Value:
        """An integer constant of type ``tp``."""
        return llvm.ConstantOp(tp, ir.IntegerAttr.get(tp, value)).res

    def i32(self, value: int) -> ir.Value:
        """An ``i32`` constant."""
        return self.integer_constant(self.i32_type, value)

    def ui32(self, value: int) -> ir.Value:
        """A ``ui32`` constant."""
        return self.integer_constant(self.ui32_type, value)

    def i1(self, value: int) -> ir.Value:
        """An ``i1`` constant."""
        return self.integer_constant(self.i1_type, value)

    def i8(self, value: int) -> ir.Value:
        """An ``i8`` constant."""
        return self.integer_constant(self.i8_type, value)

    def i16(self, value: int) -> ir.Value:
        """An ``i16`` constant."""
        return self.integer_constant(self.i16_type, value)

    def i64(self, value: int) -> ir.Value:
        """An ``i64`` constant."""
        return self.integer_constant(self.i64_type, value)

    # -- expressions ----------------------------------------------------------

    def fptrunc(self, value: ir.Value, res_type: ir.Type) -> ir.Value:
        """Truncate a float ``value`` to the narrower ``res_type``.

        Goes through ``f32`` so that no compiler-rt call (``__truncdfhf2``,
        ``__truncdfbf2``), which the JIT engine lacks, is emitted; ``bf16`` is
        the upper half of the ``f32`` bits.
        """
        if value.type == res_type:
            return value
        if res_type == self.bf16_type:
            if value.type != self.f32_type:
                value = llvm.fptrunc(res=self.f32_type, arg=value)
            v_i32 = llvm.bitcast(self.i32_type, value)
            v_shifted = llvm.lshr(v_i32, self.i32(16))
            v_i16 = llvm.trunc(
                self.i16_type, v_shifted, overflow_flags=llvm.IntegerOverflowFlags.none
            )
            return llvm.bitcast(self.bf16_type, v_i16)
        if res_type == self.f16_type and value.type != self.f32_type:
            value = llvm.fptrunc(res=self.f32_type, arg=value)
        return llvm.fptrunc(res=res_type, arg=value)

    def mul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """``lhs * rhs`` (integers, no overflow flags)."""
        return llvm.mul(lhs, rhs, overflow_flags=llvm.IntegerOverflowFlags.none)

    def not_equal(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """``lhs != rhs`` as an ``i1``."""
        return llvm.icmp(llvm.ICmpPredicate.ne, lhs, rhs)

    def equal(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """``lhs == rhs`` as an ``i1``."""
        return llvm.icmp(llvm.ICmpPredicate.eq, lhs, rhs)

    def or_(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Bitwise ``lhs | rhs`` (logical OR on ``i1``)."""
        return llvm.or_(lhs, rhs)

    def and_(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Bitwise ``lhs & rhs`` (logical AND on ``i1``)."""
        return llvm.and_(lhs, rhs)

    def not_(self, value: ir.Value) -> ir.Value:
        """Logical NOT of ``value``, truncated to ``i1`` first when wider."""
        if value.type != self.i1_type:
            value = llvm.trunc(
                res=self.i1_type,
                arg=value,
                overflow_flags=llvm.IntegerOverflowFlags.none,
            )
        return llvm.xor(value, self.i1(1))

    def i64_divisible_const(self, value: ir.Value, align_const: int) -> ir.Value:
        """Whether the ``i64`` ``value`` is a multiple of ``align_const``.

        A power of two tests the low bits with a mask; anything else uses an
        unsigned remainder.

        :param value: The ``i64`` value
        :param align_const: The positive divisor
        :return: The ``i1`` condition
        """
        is_power_of_two = (align_const > 0) and (align_const & (align_const - 1)) == 0
        if is_power_of_two:
            masked = llvm.and_(value, self.i64(align_const - 1))
            return self.equal(masked, self.i64(0))
        remainder = llvm.urem(value, self.i64(align_const))
        return self.equal(remainder, self.i64(0))

    # -- statements -----------------------------------------------------------

    def br(
        self, target_block: ir.Block, *, args: Optional[list[ir.Value]] = None
    ) -> None:
        """Branch to ``target_block``, passing ``args`` as its block arguments."""
        llvm.br(dest_operands=args if args is not None else [], dest=target_block)

    def cond_br(
        self,
        cond: ir.Value,
        true_block: ir.Block,
        false_block: ir.Block,
        *,
        branch_weights: Optional[tuple[int, int]] = None,
        true_dest_operands: Sequence[ir.Value] = (),
        false_dest_operands: Sequence[ir.Value] = (),
    ) -> None:
        """Branch on ``cond``.

        :param cond: The ``i1`` condition
        :param true_block: Taken when ``cond`` holds
        :param false_block: Taken otherwise
        :param branch_weights: ``(true_weight, false_weight)`` optimisation
            hint; a larger weight is the likelier branch
        :param true_dest_operands: Block arguments of ``true_block``
        :param false_dest_operands: Block arguments of ``false_block``
        :raises DSLRuntimeError: When ``branch_weights`` has not two entries
        """
        extra: dict[str, Any] = {}
        if branch_weights is not None:
            if len(branch_weights) != 2:
                raise DSLRuntimeError("branch_weights must have exactly 2 elements")
            extra["branch_weights"] = ir.DenseI32ArrayAttr.get(list(branch_weights))
        llvm.cond_br(
            cond,
            true_dest_operands=true_dest_operands,
            false_dest_operands=false_dest_operands,
            true_dest=true_block,
            false_dest=false_block,
            **extra,
        )

    def return_(self, ret: Optional[ir.Value] = None) -> None:
        """``llvm.return``, with ``ret`` when the function has a result."""
        llvm.return_(arg=ret)

    def address_of(self, name: str, tp: ir.Type) -> ir.Value:
        """The address of the global symbol ``name`` as ``tp``."""
        return llvm.AddressOfOp(tp, name).result

    def getelementptr(
        self,
        ptr: ir.Value,
        constant_indices: Sequence[int],
        elem_type: ir.Type,
        dynamic_indices: Sequence[ir.Value] = (),
    ) -> ir.Value:
        """``llvm.getelementptr`` into ``ptr`` viewed as ``elem_type``.

        :param ptr: The base pointer
        :param constant_indices: The index path; ``MLIR_DYNAMIC_INDEX`` marks a
            position taken from ``dynamic_indices`` instead
        :param elem_type: The type ``ptr`` points to
        :param dynamic_indices: The runtime indices, in order of their marks
        :return: The element pointer (generic address space)
        """
        return llvm.getelementptr(
            self.ptr_type,
            ptr,
            list(dynamic_indices),
            raw_constant_indices=ir.DenseI32ArrayAttr.get(list(constant_indices)),
            elem_type=elem_type,
            no_wrap_flags=[],
        )

    def offset_ptr_bytes(self, ptr: ir.Value, byte_offset: ir.Value) -> ir.Value:
        """Displace ``ptr`` by a runtime number of bytes."""
        return self.getelementptr(
            ptr,
            [self.MLIR_DYNAMIC_INDEX],
            self.i8_type,
            dynamic_indices=[byte_offset],
        )

    def define_global_string(self, content: str) -> str:
        """Define (once) a private constant global holding ``content``.

        :param content: The string, stored NUL-terminated
        :return: The symbol, ``__tvm_ffi__str_<n>``, unique in the module
        """
        if content in self.const_str_table:
            return self.const_str_table[content]
        symbol_index = len(self.const_str_table)
        symbol = f"__tvm_ffi__str_{symbol_index}"
        module = self.module
        if module is not None:
            while self._module_symbol_exists(module, symbol):
                symbol_index += 1
                symbol = f"__tvm_ffi__str_{symbol_index}"

        # The global is parsed from text, so the content is escaped for the
        # MLIR string literal.
        escaped_content = content.replace("\\", "\\\\").replace('"', '\\"')
        module_body = self.module.body  # type: ignore[union-attr]
        with ir.InsertionPoint(module_body):
            parsed_op = ir.Operation.parse(
                f'llvm.mlir.global private constant @{symbol}("{escaped_content}\\00")'
            )
            module_body.append(parsed_op)
            self.const_str_table[content] = symbol
        return symbol

    @staticmethod
    def _symbol_name(op: ir.Operation) -> Optional[str]:
        """The ``sym_name`` of a top-level ``op``, or None without one."""
        if "sym_name" not in op.attributes:
            return None
        return ir.StringAttr(op.attributes["sym_name"]).value

    def _module_symbol_exists(self, module: ir.Module, symbol: str) -> bool:
        """Whether a top-level operation of ``module`` is named ``symbol``."""
        return any(self._symbol_name(op) == symbol for op in module.body)

    # -- functions ------------------------------------------------------------

    def function(
        self,
        name: str,
        params_type: Sequence[ir.Type],
        ret_type: ir.Type,
        internal: bool = False,
        llvm_func_attrs: Sequence[str] = (),
    ) -> tuple[list[ir.Value], ir.Block]:
        """Create an ``llvm.func`` with an empty entry block.

        :param name: The symbol name
        :param params_type: The parameter types
        :param ret_type: The return type
        :param internal: Internal linkage; otherwise the function gets
            ``llvm.emit_c_interface`` so the execution engine exposes it
        :param llvm_func_attrs: LLVM function attributes (``"noinline"``, ...)
            passed through
        :return: The block arguments and the entry block
        """
        func_op = llvm.func(
            name,
            function_type=self.as_attr(
                self.func_type(ret=ret_type, params=params_type)
            ),
        )
        if internal:
            func_op.attributes["linkage"] = self.llvm_internal_linkage
        else:
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        if llvm_func_attrs:
            func_op.attributes["passthrough"] = ir.ArrayAttr.get(
                [ir.StringAttr.get(attr) for attr in llvm_func_attrs]
            )

        func_body: Any = func_op.body
        if func_body is None:
            raise DSLRuntimeError("Function body is None")
        entry_block = ir.Block.create_at_start(func_body)
        params = [
            entry_block.add_argument(param_type, ir.Location.unknown())
            for param_type in params_type
        ]
        return params, entry_block

    def declare_extern_func(
        self, name: str, params: Sequence[ir.Type], ret: ir.Type
    ) -> None:
        """Declare an external ``llvm.func`` (no body)."""
        func_op = llvm.func(
            name,
            function_type=self.as_attr(self.func_type(params=params, ret=ret)),
        )
        func_op.attributes["llvm.linkage"] = ir.StringAttr.get("external")

    def create_alloca(
        self, entry_block: ir.Block, alloca_type: ir.Type, array_size: int
    ) -> ir.Value:
        """``llvm.alloca`` of ``array_size`` x ``alloca_type`` at the top of ``entry_block``."""
        with ir.InsertionPoint(entry_block.operations[0]):
            return llvm.alloca(
                res=self.ptr_type,
                elem_type=alloca_type,
                array_size=self.i32(array_size),
            )

    def pack_values_to_alloca(
        self,
        current_block: ir.Block,
        entry_block: ir.Block,
        values: Sequence[ir.Value],
    ) -> tuple[ir.Type, ir.Value]:
        """Store ``values`` into a stack struct laid out in their order.

        :param current_block: Where the stores are emitted
        :param entry_block: Where the ``alloca`` is hoisted to
        :param values: The values to pack
        :return: The literal struct type and the ``alloca`` pointer
        """
        struct_type = self.struct_type(fields=[value.type for value in values])
        alloca = self.create_alloca(entry_block, struct_type, array_size=1)
        with ir.InsertionPoint(current_block):
            for index, value in enumerate(values):
                field_ptr = self.getelementptr(alloca, [0, index], struct_type)
                llvm.store(value, field_ptr)
        return (struct_type, alloca)

    def find_func_in_module(
        self, module: ir.Module, name: str
    ) -> Optional[ir.Operation]:
        """The ``llvm.func`` named ``name`` in ``module``, or None."""
        for op in module.body:
            if op.name == "llvm.func" and self._symbol_name(op) == name:
                return op
        return None
