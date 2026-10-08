# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s %t 2>&1 | FileCheck %s
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_DEBUGINFO=1 MLIR_DSL_VERIFY_TRACE=1 %PYTHON %s %t 2>&1 | FileCheck %s --check-prefix=DEBUGINFO
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_DEBUG=1 %PYTHON %s %t 2>&1 | FileCheck %s --check-prefix=DEBUG
# RUN: %if host-supports-jit %{ %PYTHON %s %t 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The DSL-owned helper layers: the `plugins/type_ops/arith`
# emitter's choice of op by signedness and float-ness (the `signed=` keyword,
# MLIR integers being signless), its literal rules and its conversion helpers
# (`cvtf`, `fptoi`/`itofp`, `int_to_int`, `cast`, `bitcast`) and `arith.pow`'s
# rejection of a staged int ** int; `dsl_user_op` locations under MLIR_DSL_DEBUGINFO (one
# `NameLoc(FileLineColLoc)` per op at the first user frame; the MLIR_DSL_DEBUG
# master switch keeps the closest frame and verifies at trace time),
# MLIR's traceback locations and trace-time verification; the `tree_utils` leaf registry,
# frozen-dataclass flattening, its rejections and the join type-stability rule;
# `is_mlir_op`; the `profiler` report modes (`%t` is the
# report file). The lowering of the emitted ops is MLIR's and is not checked.
import contextlib
import dataclasses
import operator
import sys

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import llvm, scf
from mlir.dsl import is_mlir_op as dyn
from mlir.dsl.plugins.type_ops import arith as A
from mlir.dsl.core.user_op import dsl_user_op
from mlir.dsl.core.common import active_dsl
from mlir.dsl.util import profiler
from mlir.dsl.util import tree_utils as tu
from mlir.extras import types as T


@contextlib.contextmanager
def function(name, arg_types, result_type):
    fty = llvm.FunctionType.get(result_type, arg_types)
    fn = llvm.LLVMFuncOp(name, ir.TypeAttr.get(fty))
    block = fn.body.blocks.append(*arg_types)
    # The types ask the tracing DSL's `type_ops` plugin for their MLIR types
    # and ops, so the hand-built body runs with the test DSL active.
    with ir.InsertionPoint(block), active_dsl(m.MlirTestDSL()):
        yield block.arguments
        llvm.ReturnOp(arg=block.arguments[0])
    print(fn)


def report(fn, *args, **kwargs):
    try:
        print("RESULT:", fn(*args, **kwargs))
    except m.DSLUserCodeError as e:
        print("ERROR:", e.diag_id.name)
    except m.DSLRuntimeError as e:
        print("INTERNAL:", e.message)


# =============================================================================
# The arith emitter: one op per operator, chosen by signedness and float-ness
# =============================================================================
with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        i1, i8, i32, i64 = T.bool(), T.i8(), T.i32(), T.i64()
        f16, bf16, f32 = T.f16(), T.bf16(), T.f32()
        v4f32 = ir.VectorType.get([4], f32)

        # The type helpers behind the emitter: `recast_type` keeps the shape.
        print(
            "RECAST:",
            A.recast_type(f32, f16),
            A.recast_type(v4f32, f16),
            A.recast_type(ir.VectorType.get([4], f32, scalable=[True]), i32),
        )
        print(
            "PREDICATES:",
            A.element_type(v4f32),
            A.is_scalar(v4f32),
            A.is_float_type(f16),
            A.is_integer_like_type(ir.IndexType.get()),
            A.is_integer_like_type(f32),
            A.is_narrow_precision(T.f8E4M3FN()),
            A.is_narrow_precision(f16),
        )
        # CHECK:      RECAST: f16 vector<4xf16> vector<[4]xi32>
        # CHECK-NEXT: PREDICATES: f32 False True True False True False

        with function("signed", [i32, i32], i32) as (a, b):
            A.add(a, b), A.floordiv(a, b), A.mod(a, b), A.shr(a, b)
            A.cmp("lt", a, b), A.minmax(a, b, is_min=True), A.truediv(a, b)
            A.neg(a), A.abs(a)
            report(A.pow, a, b)  # staged int ** int has no arith op
        # CHECK:       ERROR: TYPE_INT_POW_UNSUPPORTED
        # CHECK-LABEL: llvm.func @signed(
        # CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32)
        # CHECK:         arith.addi %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.floordivsi %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.remsi %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.shrsi %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.cmpi slt, %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.minsi %[[A]], %[[B]] : i32
        # CHECK-NEXT:    %[[FA:.+]] = arith.sitofp %[[A]] : i32 to f32
        # CHECK-NEXT:    %[[FB:.+]] = arith.sitofp %[[B]] : i32 to f32
        # CHECK-NEXT:    arith.divf %[[FA]], %[[FB]] : f32
        # CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0 : i32
        # CHECK-NEXT:    arith.subi %[[ZERO]], %[[A]] : i32
        # CHECK-NEXT:    math.absi %[[A]] : i32

        with function("unsigned", [i32, i32], i32) as (a, b):
            A.floordiv(a, b, signed=False), A.mod(a, b, signed=False)
            A.shr(a, b, signed=False), A.cmp("lt", a, b, signed=False)
            A.minmax(a, b, is_min=False, signed=False)
            A.truediv(a, b, signed=False)
        # CHECK-LABEL: llvm.func @unsigned(
        # CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32)
        # CHECK:         arith.divui %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.remui %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.shrui %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.cmpi ult, %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.maxui %[[A]], %[[B]] : i32
        # CHECK-NEXT:    arith.uitofp %[[A]] : i32 to f32
        # CHECK-NEXT:    arith.uitofp %[[B]] : i32 to f32

        with function("floating", [f32, f32, i32], f32) as (x, y, a):
            A.add(x, y), A.floordiv(x, y), A.mod(x, y), A.neg(x), A.abs(x)
            A.cmp("lt", x, y), A.cmp("ne", x, y)  # ordered, but `!=` unordered
            A.minmax(x, y, is_min=True), A.pow(x, y), A.pow(x, a), A.pow(a, x)
        # CHECK-LABEL: llvm.func @floating(
        # CHECK-SAME:    %[[X:[^:]+]]: f32, %[[Y:[^:]+]]: f32, %[[A:[^:]+]]: i32)
        # CHECK:         arith.addf %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    %[[Q:.+]] = arith.divf %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    math.floor %[[Q]] : f32
        # CHECK-NEXT:    arith.remf %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    arith.negf %[[X]] : f32
        # CHECK-NEXT:    math.absf %[[X]] : f32
        # CHECK-NEXT:    arith.cmpf olt, %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    arith.cmpf une, %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    arith.minimumf %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    math.powf %[[X]], %[[Y]] : f32
        # CHECK-NEXT:    math.fpowi %[[X]], %[[A]] : f32, i32
        # CHECK-NEXT:    %[[FA:.+]] = arith.sitofp %[[A]] : i32 to f32
        # CHECK-NEXT:    math.powf %[[FA]], %[[X]] : f32

        # `const`: an SSA value or a staged `Numeric` is returned as is; the
        # literal rule (bool/int/float -> i1/i32/f32) unless an `ir.Type`, a
        # `Numeric` class or a `Numeric` payload names the type; a float literal
        # for an integer type truncates; a shaped type splats; a canonical
        # unsigned literal is re-encoded into the signed range IntegerAttr takes
        # and the storage form is accepted as is.
        with function("const", [i32], i32) as (a,):
            print("CONST IDENTITY:", A.const(a) is a, A.const(m.Int32(a)) is a)
            A.const(True), A.const(5), A.const(2.5), A.const(3, f32), A.const(2.9, i32)
            A.const(9, m.Int8), A.const(m.Float16(0.5)), A.const(m.Uint32(0xFFFFFFFF))
            A.const(0xFFFFFFFF, m.Uint32), A.const(-1, m.Uint8), A.const(1.5, v4f32)
            A.const(0xFFFF, ir.VectorType.get([2], T.i16()), signed=False)
            report(A.const, 300, m.Uint8)
            report(A.const, "x")
            report(A.cmp, "bogus", a, a)
            report(A.cmp, operator.add, a, a)
        # CHECK:       CONST IDENTITY: True True
        # CHECK-NEXT:  INTERNAL: Unsigned integer literal 300 does not fit in unsigned i8
        # CHECK-NEXT:  INTERNAL: <class 'str'> is not supported
        # CHECK-NEXT:  INTERNAL: cmp: unsupported predicate 'bogus'
        # CHECK-NEXT:  INTERNAL: cmp: unsupported predicate <built-in function add>
        # CHECK-LABEL: llvm.func @const(
        # CHECK:         arith.constant true
        # CHECK-NEXT:    arith.constant 5 : i32
        # CHECK-NEXT:    arith.constant 2.500000e+00 : f32
        # CHECK-NEXT:    arith.constant 3.000000e+00 : f32
        # CHECK-NEXT:    arith.constant 2 : i32
        # CHECK-NEXT:    arith.constant 9 : i8
        # CHECK-NEXT:    arith.constant 5.000000e-01 : f16
        # CHECK-NEXT:    arith.constant -1 : i32
        # CHECK-NEXT:    arith.constant -1 : i32
        # CHECK-NEXT:    arith.constant -1 : i8
        # CHECK-NEXT:    arith.constant dense<1.500000e+00> : vector<4xf32>
        # CHECK-NEXT:    arith.constant dense<-1> : vector<2xi16>

        # The re-encoding itself, and `minmax` folding two Python scalars.
        enc = A._python_int_for_integer_attr
        print(
            "ENCODE:",
            enc(255, 8, signed=False),
            enc(-128, 8, signed=False),
            enc(2**64 - 1, 64, signed=False),
        )
        report(enc, 256, 8, signed=False)
        report(enc, 128, 8, signed=True)
        report(enc, 0, 0, signed=True)
        print("FOLD:", A.minmax(2, 3, is_min=True), A.minmax(2.5, -1.0, is_min=False))
        # CHECK:      ENCODE: -1 -128 -1
        # CHECK-NEXT: INTERNAL: Unsigned integer literal 256 does not fit in unsigned i8
        # CHECK-NEXT: INTERNAL: Signed integer literal 128 does not fit in i8
        # CHECK-NEXT: INTERNAL: Invalid integer width: 0
        # CHECK-NEXT: FOLD: 2 2.5

        # `minmax` with one Python scalar materializes it in the other's type;
        # `cmp` takes the `operator` function as the predicate name and equality
        # has no signedness; `select` is `arith.select`.
        with function("compare", [i32, f32, i1], i32) as (a, x, c):
            A.minmax(a, 3, is_min=True), A.minmax(2.5, x, is_min=False)
            A.cmp(operator.le, a, a), A.cmp("eq", a, a, signed=False)
            A.cmp(operator.eq, x, x), A.select(c, a, a), A.select(c, x, x)
        # CHECK-LABEL: llvm.func @compare(
        # CHECK-SAME:    %[[A:[^:]+]]: i32, %[[X:[^:]+]]: f32, %[[C:[^:]+]]: i1)
        # CHECK:         %[[C3:.+]] = arith.constant 3 : i32
        # CHECK-NEXT:    arith.minsi %[[A]], %[[C3]] : i32
        # CHECK-NEXT:    %[[C25:.+]] = arith.constant 2.500000e+00 : f32
        # CHECK-NEXT:    arith.maximumf %[[C25]], %[[X]] : f32
        # CHECK-NEXT:    arith.cmpi sle, %[[A]], %[[A]] : i32
        # CHECK-NEXT:    arith.cmpi eq, %[[A]], %[[A]] : i32
        # CHECK-NEXT:    arith.cmpf oeq, %[[X]], %[[X]] : f32
        # CHECK-NEXT:    arith.select %[[C]], %[[A]], %[[A]] : i32
        # CHECK-NEXT:    arith.select %[[C]], %[[X]], %[[X]] : f32

        # The conversion helpers: `cvtf` (the bf16<->f16 detour through f32,
        # E8M0 rounds upward, element-wise on a vector), `fptoi`/`itofp` by
        # signedness (`None` reads as signed, an i1 source is always
        # zero-extended), `int_to_int` (a signless source takes the
        # destination's signedness unless `src_signed` says; an unsigned side
        # extends by zero), `cast` with an `ir.Type` (signed by default) or a
        # `Numeric` destination and a raw or `Numeric` source, and `bitcast`.
        # A same-type conversion returns its operand.
        ptr = llvm.PointerType.get()
        with function("convert", [f32, f16, bf16, v4f32, i32, i8, i1, ptr], f32) as (
            x,
            h,
            bh,
            xv,
            a,
            c,
            b,
            p,
        ):
            print(
                "CAST IDENTITY:",
                A.cvtf(x, f32) is x,
                A.int_to_int(a, m.Uint32) is a,
                A.cast(x, m.Float32) is x,
                A.cast(a, i32) is a,
            )
            A.cvtf(h, f32), A.cvtf(x, f16), A.cvtf(h, bf16), A.cvtf(bh, f16)
            A.cvtf(x, T.f8E8M0FNU()), A.cvtf(x, T.f8E4M3FN()), A.cvtf(xv, f16)
            A.fptoi(x, None, i8), A.fptoi(x, False, i64)
            A.itofp(a, False, f16), A.itofp(b, True, f32)
            A.int_to_int(c, m.Int32), A.int_to_int(c, m.Int32, src_signed=False)
            A.int_to_int(c, m.Uint32, src_signed=True)
            A.int_to_int(b, m.Int32, src_signed=True)
            A.int_to_int(a, m.Int8, src_signed=False)
            A.cast(a, m.Float64), A.cast(x, m.Uint8), A.cast(x, i64)
            A.cast(x, i64, signed=False), A.cast(c, m.Int32, signed=False)
            A.cast(h, f32), A.cast(m.Int32(a), m.Float32)
            A.bitcast(x, i32), A.bitcast(xv, i32)
            A.pow(a, h)  # int ** narrow float: both sides become f32
            report(A.cast, p, m.Int32)
        # CHECK:       CAST IDENTITY: True True True True
        # CHECK-NEXT:  INTERNAL: cast from !llvm.ptr to i32 is not supported
        # CHECK-LABEL: llvm.func @convert(
        # CHECK-SAME:    %[[X:[^:]+]]: f32, %[[H:[^:]+]]: f16, %[[BH:[^:]+]]: bf16, %[[XV:[^:]+]]: vector<4xf32>, %[[A:[^:]+]]: i32, %[[C:[^:]+]]: i8, %[[B:[^:]+]]: i1, %{{[^:]+}}: !llvm.ptr)
        # CHECK:         arith.extf %[[H]] : f16 to f32
        # CHECK-NEXT:    arith.truncf %[[X]] : f32 to f16
        # CHECK-NEXT:    %[[W:.+]] = arith.extf %[[H]] : f16 to f32
        # CHECK-NEXT:    arith.truncf %[[W]] : f32 to bf16
        # CHECK-NEXT:    %[[WB:.+]] = arith.extf %[[BH]] : bf16 to f32
        # CHECK-NEXT:    arith.truncf %[[WB]] : f32 to f16
        # CHECK-NEXT:    arith.truncf %[[X]] upward : f32 to f8E8M0FNU
        # CHECK-NEXT:    arith.truncf %[[X]] : f32 to f8E4M3FN
        # CHECK-NEXT:    arith.truncf %[[XV]] : vector<4xf32> to vector<4xf16>
        # CHECK-NEXT:    arith.fptosi %[[X]] : f32 to i8
        # CHECK-NEXT:    arith.fptoui %[[X]] : f32 to i64
        # CHECK-NEXT:    arith.uitofp %[[A]] : i32 to f16
        # CHECK-NEXT:    arith.uitofp %[[B]] : i1 to f32
        # CHECK-NEXT:    arith.extsi %[[C]] : i8 to i32
        # CHECK-NEXT:    arith.extui %[[C]] : i8 to i32
        # CHECK-NEXT:    arith.extsi %[[C]] : i8 to i32
        # CHECK-NEXT:    arith.extui %[[B]] : i1 to i32
        # CHECK-NEXT:    arith.trunci %[[A]] : i32 to i8
        # CHECK-NEXT:    arith.sitofp %[[A]] : i32 to f64
        # CHECK-NEXT:    arith.fptoui %[[X]] : f32 to i8
        # CHECK-NEXT:    arith.fptosi %[[X]] : f32 to i64
        # CHECK-NEXT:    arith.fptoui %[[X]] : f32 to i64
        # CHECK-NEXT:    arith.extui %[[C]] : i8 to i32
        # CHECK-NEXT:    arith.extf %[[H]] : f16 to f32
        # CHECK-NEXT:    arith.sitofp %[[A]] : i32 to f32
        # CHECK-NEXT:    arith.bitcast %[[X]] : f32 to i32
        # CHECK-NEXT:    arith.bitcast %[[XV]] : vector<4xf32> to vector<4xi32>
        # CHECK-NEXT:    %[[PA:.+]] = arith.sitofp %[[A]] : i32 to f32
        # CHECK-NEXT:    %[[PH:.+]] = arith.extf %[[H]] : f16 to f32
        # CHECK-NEXT:    math.powf %[[PA]], %[[PH]] : f32
    module.operation.verify()


# =============================================================================
# dsl_user_op: source locations (MLIR's traceback locations), verification
# =============================================================================
# Locations come from MLIR's traceback locations, which `BaseDSL` turns on
# under DEBUGINFO (depth 1): an op carries the user line and column range that
# built it; the frames of the DSL packages, the bindings and the standard
# library are skipped, and DEBUG keeps the DSL frames instead so an op is
# attributed to the DSL line that built it. An explicit `loc=` always wins.
# A helper in a module of a DSL package: its frames are skipped by the walk.
# (The exec'd module has no file on disk, so the test registers its pseudo
# filename the way a package directory is registered.)
LIB_SRC = "def lib_twice(x):\n    return x + x\n"
lib_namespace = {"__name__": "mlir.dsl._test_lib"}
exec(compile(LIB_SRC, "<dsl-lib>", "exec"), lib_namespace)
lib_twice = lib_namespace["lib_twice"]
ir._globals.register_traceback_file_exclusion("<dsl-lib>")


@m.jit
def located(a: m.Int32, x: m.Float32) -> m.Float32:
    b = a + 1
    print("LOC addi:", b.value.owner.location)
    # DEBUGINFO: LOC addi: loc("located"("{{.*}}helpers.py":[[#@LINE-2]]:8 to :13))
    # DEBUG:     LOC addi: loc("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}}))
    # CHECK:     LOC addi: loc(unknown)
    r = lib_twice(x)
    print("LOC lib:", r.value.owner.location)
    # The op was built through a helper in a DSL package: the user line is
    # its caller, except under MLIR_DSL_DEBUG=1, which names the DSL line
    # that built the op (the scalar type ops).
    # DEBUGINFO: LOC lib: loc("located"("{{.*}}helpers.py":[[#@LINE-5]]:8 to :20))
    # DEBUG:     LOC lib: loc("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}}))
    # CHECK:     LOC lib: loc(unknown)
    y = a + x
    print("LOC promoted:", y.value.owner.operands[0].owner.location)
    # The `sitofp` that operand promotion emits inside the DSL gets the user
    # line too: every op built during the statement does, whichever DSL frame
    # builds it.
    # DEBUGINFO: LOC promoted: loc("located"("{{.*}}helpers.py":[[#@LINE-5]]:8 to :13))
    # DEBUG:     LOC promoted: loc("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}}))
    # CHECK:     LOC promoted: loc(unknown)
    c = b.__add__(1, loc=ir.Location.name("given"))  # a caller's `loc=` wins
    print("LOC explicit:", c.value.owner.location)
    # DEBUGINFO: LOC explicit: loc("given")
    # DEBUG:     LOC explicit: loc("given")
    # CHECK:     LOC explicit: loc("given")
    return m.Float32(c) + r


report(located, 1, 4.0)
# DEBUGINFO: RESULT: ?
# DEBUG:     RESULT: ?
# CHECK:     RESULT: ?
# EXEC:      RESULT: 11.0


@dsl_user_op
def bad_addi(a, b, *, loc=None, ip=None):
    """`arith.addi` on an i32 and an f32: rejected by the verifier."""
    operands = [a, b]
    return ir.Operation.create("arith.addi", [a.type], operands, loc=loc, ip=ip).result


@dsl_user_op
def bad_addi_op(a, b, *, loc=None, ip=None):
    """The same malformed op, returned as an `OpView`."""
    return ir.Operation.create("arith.addi", [a.type], [a, b], loc=loc, ip=ip).opview


@dsl_user_op
def bad_addi_at_start(a, b, *, loc=None, ip=None):
    """The malformed op inserted at the start of the block, not at its tail."""
    ip = ir.InsertionPoint.at_block_begin(ir.InsertionPoint.current.block)
    return ir.Operation.create("arith.addi", [a.type], [a, b], loc=loc, ip=ip).result


@dsl_user_op
def then_region(cond, *, loc=None, ip=None):
    """An `scf.if` whose body the caller fills in: a context manager."""
    return ir.InsertionPoint(scf.IfOp(cond, has_else=False, loc=loc, ip=ip).then_block)


@dsl_user_op
def no_loc_keyword(a):
    return a


@m.jit
def malformed(a: m.Int32, x: m.Float32) -> m.Int32:
    return m.Int32(bad_addi(a.ir_value(), x.ir_value()))


@m.jit
def malformed_opview(a: m.Int32, x: m.Float32) -> m.Int32:
    return m.Int32(bad_addi_op(a.ir_value(), x.ir_value()).result)


@m.jit
def malformed_at_start(a: m.Int32, x: m.Float32) -> m.Int32:
    b = a + 1  # the block's tail op before the builder runs
    return m.Int32(bad_addi_at_start(a.ir_value(), x.ir_value())) + b


@m.jit
def region_builder(a: m.Int32) -> m.Int32:
    with then_region((a > 0).ir_value()):  # the empty region is filled in here
        scf.YieldOp([])
    return a


@m.jit
def missing_loc(a: m.Int32) -> m.Int32:
    return m.Int32(no_loc_keyword(a.ir_value()))


# Trace-time verification (MLIR_DSL_VERIFY_TRACE=1, or the MLIR_DSL_DEBUG master
# switch) names the builder as soon as it returns; otherwise module verify
# reports. An `OpView` result is verified in either mode; a context manager
# (its body is filled in later) is left to module verify; an op inserted ahead
# of the block's tail is reached through the returned value. A builder without
# the `loc=` keyword is the wrapper's remaining diagnostic.
# DEBUGINFO: INTERNAL: Operation verification failed in 'bad_addi'
# DEBUGINFO: INTERNAL: Operation verification failed in 'bad_addi_op'
# DEBUGINFO: INTERNAL: Operation verification failed in 'bad_addi_at_start'
# DEBUGINFO: RESULT: ?
# DEBUG:     INTERNAL: Operation verification failed in 'bad_addi'
# DEBUG:     INTERNAL: Operation verification failed in 'bad_addi_op'
# DEBUG:     INTERNAL: Operation verification failed in 'bad_addi_at_start'
# DEBUG:     RESULT: ?
# CHECK:     INTERNAL: IR verification failed
# CHECK:     INTERNAL: Operation verification failed in 'bad_addi_op'
# CHECK:     INTERNAL: IR verification failed
# CHECK:     RESULT: ?
# CHECK:     INTERNAL: Function 'no_loc_keyword' decorated with @dsl_user_op does not accept the required 'loc' parameter.
# CHECK:     SUGGESTION: 1. Add 'loc=None' as a keyword-only parameter to no_loc_keyword:
report(malformed, 1, 2.0)
report(malformed_opview, 1, 2.0)
report(malformed_at_start, 1, 2.0)
report(region_builder, 1)
report(missing_loc, 1)
try:
    missing_loc(1)
except m.DSLRuntimeError as e:
    print("SUGGESTION:", e.suggestion[0])


# =============================================================================
# tree_utils: the leaf registry
# =============================================================================
class Pair:
    """Two i32 values: Python ints on the host, SSA values inside a trace."""

    def __init__(self, a, b):
        self.a, self.b = a, b


class SubPair(Pair):
    pass


m.register_leaf(
    Pair,
    ir_types=lambda p: [T.i32(), T.i32()],
    ir_values=lambda v: [v.a, v.b],
    from_ir_values=lambda p, vs: Pair(vs[0], vs[1]),
    marshal=lambda v: [m.Int32.marshal(v.a), m.Int32.marshal(v.b)],
)
entry = tu.leaf_entry(Pair)
print("REGISTRY:", entry.cls.__name__, tu.leaf_entry(SubPair) is entry)
print("IS_LEAF:", tu.is_leaf(Pair), tu.is_leaf(SubPair(1, 2)), tu.is_leaf(m.Int32(1)))
print("CONTAINS:", tu.contains_leaf((1, [Pair(1, 2)])), tu.contains_leaf({"k": 2}))
ref = tu.is_reference_leaf
print("REFERENCE:", ref(m.Pointer(0x1000)), ref(Pair(1, 2)))
# CHECK:      REGISTRY: Pair True
# CHECK-NEXT: IS_LEAF: True True True
# CHECK-NEXT: CONTAINS: True False
# CHECK-NEXT: REFERENCE: True False
# CHECK-NEXT: INTERNAL: leaf class `Pair` is already registered; a leaf registers once, at import of its module
again = {k: getattr(entry, k) for k in ("ir_types", "ir_values", "from_ir_values")}
report(m.register_leaf, Pair, **again)


@m.jit
def carry_pair(n: m.Int32) -> m.Int32:
    p = Pair(m.Int32(0).ir_value(), m.Int32(1).ir_value())
    for i in range(n):
        p = Pair((m.Int32(p.a) + i).ir_value(), (m.Int32(p.b) * 2).ir_value())
    return m.Int32(p.a) + m.Int32(p.b)


@m.jit
def pair_sum(p: Pair) -> m.Int32:
    return m.Int32(p.a) + m.Int32(p.b)


# A registered leaf is loop state (its two values are the iter_args) and a
# host argument (`marshal` hands over two i32).
# CHECK-LABEL: func.func @carry_pair(
# CHECK:         scf.for {{.*}} iter_args(%[[PA:.+]] = %{{.+}}, %[[PB:.+]] = %{{.+}}) -> (i32, i32)
# CHECK:           arith.addi %[[PA]], %{{.+}} : i32
# CHECK:           arith.muli %[[PB]], %{{.+}} : i32
# CHECK-LABEL: func.func @pair_sum(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32) -> i32
# CHECK:         arith.addi %[[A]], %[[B]] : i32
# EXEC:          RESULT: 22
# EXEC-NEXT:     RESULT: 7
report(carry_pair, 4)  # 0+1+2+3, 1*2**4
report(pair_sum, Pair(3, 4))


# =============================================================================
# tree_utils: flattening, rejections and the join comparison
# =============================================================================
@dataclasses.dataclass(frozen=True)
class State:
    acc: m.Int32
    count: int
    cfg: int
    pending: object = None


@dataclasses.dataclass(frozen=True)
class Pt:
    x: m.Float32
    y: m.Float32


@dataclasses.dataclass
class Mutable:
    acc: object


@dataclasses.dataclass(frozen=True)
class Cfg:
    stages: int


@dataclasses.dataclass(frozen=True)
class Lazy:
    a: m.Int32
    b: m.Int32 = dataclasses.field(init=False)  # never assigned


@dataclasses.dataclass(frozen=True)
class Doubled:
    a: m.Int32

    def __post_init__(self):  # a DSL value outside the fields
        object.__setattr__(self, "twice", self.a * 2)


@dataclasses.dataclass(frozen=True)
class Tagged:
    a: m.Int32

    def __post_init__(self):  # a Python value outside the fields
        object.__setattr__(self, "tag", "hello")


class MyTuple(tuple):
    pass


def describe(leaf):
    if isinstance(leaf, tu.PyTreeDef):
        return leaf.node_metadata.kind
    if leaf.is_none:
        return "None"
    if leaf.is_meta:
        return f"meta={leaf.meta!r}"
    proto = leaf.prototype
    if isinstance(proto, type):
        name = proto.__name__
    elif isinstance(proto, tuple) and isinstance(proto[0], type) and len(proto) == 3:
        name = f"{proto[0].__name__}[{proto[1].__name__}, {proto[2]}]"  # Vector
    else:
        name = f"<{type(proto).__name__}>"  # Pointer: (dtype, space)
    return f"{name}:{leaf.ir_type_str}"


def leaves(treedef):
    return [(path, describe(leaf)) for path, leaf in tu.tree_leaves(treedef)]


def expect(label, thunk):
    try:
        result = thunk()
    except m.DSLUserCodeError as e:
        print(f"{label}: {e.diag_id.name}")
    else:
        print(f"{label}: no error, {len(result[0])} values")


def compare(label, lhs, rhs):
    lt, rt = tu.tree_flatten(lhs)[2], tu.tree_flatten(rhs)[2]
    try:
        index = tu.check_tree_equal(lt, rt)
    except m.DSLRuntimeError:
        index = "n/a"
    print(f"{label}: {index}", repr(tu.describe_tree_difference(lt, rt, "state")))


with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        i32, f32, ptr = T.i32(), T.f32(), llvm.PointerType.get()
        v4f32, index = ir.VectorType.get([4], f32), ir.IndexType.get()
        v4i32 = ir.VectorType.get([4], i32)
        with function("trees", [i32, i32, f32, ptr, v4f32, v4i32, index], i32) as args:
            a, b, x, p, v, vi, idx = args
            n, u, fx = m.Int32(a), m.Uint32(a), m.Float32(x)

            # Leaves flatten to their SSA values; Python scalars (fields without
            # a DSL annotation) are META slots; `tree_unflatten` rebuilds fresh objects.
            state = State(acc=n, count=2, cfg=4)
            values, attrs, treedef = tu.tree_flatten(state)
            print("FLATTEN:", len(values), values[0] is a, treedef.paths)
            print("LEAVES:", leaves(treedef))
            rebuilt = tu.tree_unflatten(treedef, [b])
            print("REBUILT:", rebuilt is state, rebuilt.acc.value is b, rebuilt.cfg)
            # CHECK:      FLATTEN: 1 True ('.acc', '.count', '.cfg', '.pending')
            # CHECK-NEXT: LEAVES: [('.acc', 'Int32:i32'), ('.count', 'meta=2'), ('.cfg', 'meta=4'), ('.pending', 'None')]
            # CHECK-NEXT: REBUILT: False True 4
            vec, pointer = m.Vector.from_ir(v), m.Pointer(p, dtype=m.Float32)
            nested = (Pt(fx, fx), (n, 7), pointer, vec, Cfg(stages=3))
            values, _, treedef = tu.tree_flatten(nested)
            print("NESTED:", len(values), treedef.paths)
            print("KINDS:", [describe(c) for c in treedef.child_treedefs])
            print("PROTOTYPES:", [d for _, d in leaves(treedef)][2:])
            # CHECK-NEXT: NESTED: 5 ('[0].x', '[0].y', '[1][0]', '[1][1]', '[2]', '[3]', '[4].stages')
            # CHECK-NEXT: KINDS: ['dataclass', 'tuple', '<tuple>:!llvm.ptr', 'Vector[Float32, 4]:vector<4xf32>', 'dataclass']
            # CHECK-NEXT: PROTOTYPES: ['Int32:i32', 'meta=7', '<tuple>:!llvm.ptr', 'Vector[Float32, 4]:vector<4xf32>', 'meta=3']

            # `wrap_ir_value` claims a raw SSA value through the built-in or a
            # registered leaf; an unknown type is a user error.
            wrapped = [type(tu.wrap_ir_value(raw)).__name__ for raw in (a, x, p, v)]
            print("WRAP:", wrapped)
            expect("wrap index", lambda: (tu.wrap_ir_value(idx),))
            # CHECK-NEXT: WRAP: ['Int32', 'Float32', 'Pointer', 'Vector']
            # CHECK-NEXT: wrap index: TYPE_UNSUPPORTED_MLIR_TYPE

            # Rejections: the only containers are tuples, lists and frozen
            # records; a dict, set or tuple subclass holding DSL values is
            # refused, one holding Meta values only is a Meta value.
            expect("not frozen", lambda: tu.tree_flatten(Mutable(acc=n)))
            expect("meta-only not frozen", lambda: tu.tree_flatten(Mutable(acc="meta")))
            expect("list", lambda: tu.tree_flatten((fx, [n, 2])))
            expect("dict", lambda: tu.tree_flatten({"k": n}))
            expect("set", lambda: tu.tree_flatten({n}))
            expect("tuple subclass", lambda: tu.tree_flatten(MyTuple((n, 2))))
            expect("meta list", lambda: tu.tree_flatten((n, [1, 2])))
            expect("leaf twice", lambda: tu.tree_flatten((n, n, (n,))))
            cyclic = [n]
            cyclic.append(cyclic)
            expect("cycle", lambda: tu.tree_flatten((n, cyclic)))
            # CHECK-NEXT: not frozen: CONTAINER_INVALID_RECORD
            # CHECK-NEXT: meta-only not frozen: no error, 0 values
            # CHECK-NEXT: list: no error, 2 values
            # CHECK-NEXT: dict: CONTAINER_UNSUPPORTED
            # CHECK-NEXT: set: CONTAINER_UNSUPPORTED
            # CHECK-NEXT: tuple subclass: CONTAINER_UNSUPPORTED
            # CHECK-NEXT: meta list: no error, 1 values
            # CHECK-NEXT: leaf twice: no error, 3 values
            # CHECK-NEXT: cycle: CONTAINER_TOO_DEEP

            # A record is carried field by field: a field without a value
            # (`init=False`, never assigned) and an instance attribute holding
            # a DSL value are refused; a Python-valued extra attribute rides
            # along; a tree deeper than the recursion limit is a user error,
            # not a `RecursionError`; `tree_unflatten` checks the value count.
            lazy = object.__new__(Lazy)
            object.__setattr__(lazy, "a", n)
            expect("field unset", lambda: tu.tree_flatten(lazy))
            expect("extra leaf", lambda: tu.tree_flatten(Doubled(n)))
            values, _, treedef = tu.tree_flatten(Tagged(n))
            rebuilt = tu.tree_unflatten(treedef, [b])
            print("extra meta:", len(values), rebuilt.tag, rebuilt.a.value is b)
            deep = n
            for _ in range(sys.getrecursionlimit()):
                deep = (deep,)
            expect("too deep", lambda: tu.tree_flatten(deep))
            expect(
                "too deep (host)", lambda: tu.tree_flatten(deep, return_ir_values=False)
            )
            print("contains too deep:", end=" ")
            expect("", lambda: (tu.contains_leaf(deep),))
            _, _, pair_def = tu.tree_flatten((n, n))
            try:
                tu.tree_unflatten(pair_def, [a])
            except m.DSLRuntimeError:
                print("unflatten count: DSLRuntimeError")
            # CHECK-NEXT: field unset: CONTAINER_INVALID_RECORD
            # CHECK-NEXT: extra leaf: CONTAINER_INVALID_RECORD
            # CHECK-NEXT: extra meta: 1 hello True
            # CHECK-NEXT: too deep: CONTAINER_TOO_DEEP
            # CHECK-NEXT: too deep (host): CONTAINER_TOO_DEEP
            # CHECK-NEXT: contains too deep: : CONTAINER_TOO_DEEP
            # CHECK-NEXT: unflatten count: DSLRuntimeError

            # The join rule: leaves match on IR type and prototype (so Int32 and
            # Uint32 differ at the same `i32`), META slots on `==`, containers
            # on shape; the index of the first differing child is returned.
            compare("equal", State(n, 2, 4), State(n, 2, 4))
            compare("dtype", State(n, 2, 4), State(fx, 2, 4))
            compare("signedness", State(n, 2, 4), State(u, 2, 4))
            compare("meta", State(n, 2, 4), State(n, 3, 4))
            compare("none to leaf", State(n, 2, 4), State(n, 2, 4, n))
            compare("length", (n, n), (n, n, n))
            vi32, vu32 = m.Vector.from_ir(vi), m.Vector.from_ir(vi, dtype=m.Uint32)
            compare("vector", (vi32,), (vi32,))
            compare("vector signedness", (vi32,), (vu32,))
            compare("vector dtype", (vec,), (vi32,))
            # CHECK-NEXT: equal: -1 ''
            # CHECK-NEXT: dtype: 0 '`state.acc` changed from `Int32` (type `i32`) to `Float32` (type `f32`)'
            # CHECK-NEXT: signedness: 0 '`state.acc` changed from `Int32` (type `i32`) to `Uint32` (type `i32`)'
            # CHECK-NEXT: meta: 1 '`state.count` changed from a Python value `2` to a Python value `3`'
            # CHECK-NEXT: none to leaf: 3 '`state.pending` changed from `None` to `Int32` (type `i32`)'
            # CHECK-NEXT: length: n/a '`state` changed from `tuple` with 2 items to `tuple` with 3 items'
            # CHECK-NEXT: vector: -1 ''
            # CHECK-NEXT: vector signedness: 0 '`state[0]` changed from `Vector[Int32, 4]` (type `vector<4xi32>`) to `Vector[Uint32, 4]` (type `vector<4xi32>`)'
            # CHECK-NEXT: vector dtype: 0 '`state[0]` changed from `Vector[Float32, 4]` (type `vector<4xf32>`) to `Vector[Int32, 4]` (type `vector<4xi32>`)'

            # `is_mlir_op` tells an MLIR op from a Python value: true for a
            # raw SSA value or block-argument list, a leaf whose payload is an
            # SSA value and a tuple, list or frozen record holding one
            # anywhere; false for Python values, host-side leaves and the
            # containers it does not walk (dicts).
            host = (m.Int32(1), m.Pointer(0x1000), State(n, 2, 4), {"k": n}, [[]])
            print(
                "DYNAMIC:",
                dyn(n),
                dyn(a),
                dyn(args),
                dyn(pointer),
                dyn(vec),
                dyn(n + 1),
            )
            print("DYNAMIC containers:", dyn((1, [n])), *[dyn(h) for h in host])
            # CHECK-NEXT: DYNAMIC: True True True True True True
            # CHECK-NEXT: DYNAMIC containers: True False False True False False


# =============================================================================
# profiler: the default, deep and file report modes
# =============================================================================
def profiled_compile():
    profiler.begin_compile()
    profiler.profile_build(lambda: sum(range(1000)))()
    profiler.begin_mlir_phase()
    profiler.end_mlir_phase()
    profiler.finish_compile()


# The reports go to stderr as the phases end: keep stdout in step with them.
sys.stdout.flush()
profiler.configure("T_PROFILE_COMPILER", "1")
profiled_compile()
# The tracing section is emitted as soon as the build phase ends, the mlir
# section at the end of the compile; both name the variable.
# CHECK:      DSL compile profile — tracing  (T_PROFILE_COMPILER)
# CHECK:      build     : {{ *[0-9.]+}} ms   (IR trace + finalize; includes ast-build)
# CHECK-NOT:  deep mode
# CHECK:      DSL compile profile — mlir  (T_PROFILE_COMPILER)
# CHECK:      mlir      : {{ *[0-9.]+}} ms   (pass pipeline)
# CHECK-NOT:  per-pass
profiler.configure("T_PROFILE_COMPILER", "deep")
profiled_compile()
# CHECK:      DSL compile profile — tracing
# CHECK:      (deep mode: cProfile active — times inflated vs default mode)
# CHECK:      TOP FILES by self time
# CHECK:      WHICH FUNCTION is slow in 'build'
# CHECK:      mlir      : {{ *[0-9.]+}} ms   (pass pipeline; single-threaded for timing)
# CHECK:      (no MLIR per-pass report captured — pass pipeline did not run,
report_path = sys.argv[1]
profiler.configure("T_PROFILE_COMPILER", report_path)
profiled_compile()
with open(report_path, encoding="utf-8") as f:
    found = [line.strip() for line in f if "compile profile" in line]
    print("FILE:", found, flush=True)
# CHECK:      [T_PROFILE_COMPILER] compile profile written to {{.*}}
# CHECK:      FILE: ['DSL compile profile — tracing  (T_PROFILE_COMPILER)', 'DSL compile profile — mlir  (T_PROFILE_COMPILER)']
profiler.configure("T_PROFILE_COMPILER", "off")
profiled_compile()
print("OFF:", profiler.enabled(), profiler.deep(), flush=True)
# CHECK-NEXT: OFF: False False
