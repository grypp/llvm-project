# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC
# REQUIRES: host-supports-jit
# The explicit builders `for_`/`if_`/`while_`/`yield_` (Design 7.7) under
# `preprocess=False`, and the executors behind the rewrite (Design 7.3-7.5):
# the bound promotion table, the rejection of loop options, the join checks
# every region applies to its write_args (`PHASE_MUTATE_PYTHON`,
# `TYPE_UNSTABLE_JOIN`, `CONTAINER_STRUCTURE_CHANGED`) with the diagnostics
# they render, the ternary executor, the `and_`/`or_`/`not_`/`any_`/`all_`/
# `in_` helpers and the `max`/`min`/`any`/`all` builtin redirection. The
# `scf` ops themselves are the dialect's business and are not checked.
import numpy as np

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.dialects.scf import executors, in_
from mlir.dsl.plugins.ast_preprocessor.scf import _builtin_redirector


def report(fn, *args):
    try:
        print("RESULT:", fn.__name__, fn(*args))
    except m.DSLUserCodeError as e:
        print("ERROR:", fn.__name__, e.diag_id.name if e.diag_id else None)
        print(str(e))


@m.struct
class Pair:
    lo: m.Int32
    hi: m.Int32


# --- `for_` and `yield_` ----------------------------------------------------


@m.jit(preprocess=False)
def for_no_carry(n: m.Int32, out: m.Pointer[m.Int32]):
    for i in m.for_(n):  # `for_(stop)`: `iv` alone, the body terminated for the user
        out[i] = i * 2


# CHECK-LABEL: func.func @for_no_carry(
# CHECK-SAME:    %[[N:[^:]+]]: i32, %[[OUT:[^:]+]]: !llvm.ptr)
# CHECK-NOT:     index
# CHECK:         scf.for %[[I:.+]] = %{{.+}} to %[[N]] step %{{.+}} : i32 {
# CHECK:           llvm.store {{.*}} : i32, !llvm.ptr
# CHECK-NEXT:    }
buf = np.zeros(4, np.int32)
for_no_carry(4, buf)
# EXEC: RESULT: for_no_carry [0, 2, 4, 6]
print("RESULT: for_no_carry", buf.tolist())


@m.jit(preprocess=False)
def for_bounds(a: m.Int32, b: m.Int64) -> m.Int64:
    # `for_(start, stop, step, iter_args)`: the bounds and `iv` take the
    # promoted dtype, a Python step is materialised in it; one carry is the
    # `(iv, iter_arg, result)` triple and `yield_` takes a bare value.
    for i, acc_in, acc_out in m.for_(a, b, 2, [m.Int64(0)]):
        m.yield_(acc_in + i)
    return acc_out


# CHECK-LABEL: func.func @for_bounds(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i64) -> i64
# CHECK:         arith.extsi %[[A]] : i32 to i64
# CHECK:         %[[STEP:.+]] = arith.constant 2 : i64
# CHECK:         %[[R:.+]] = scf.for %{{.+}} = %{{.+}} to %[[B]] step %[[STEP]] iter_args(%{{.+}} = %{{.+}}) -> (i64) : i64 {
# CHECK:         return %[[R]] : i64
# EXEC:          RESULT: for_bounds 16
report(for_bounds, 1, 9)  # 1 + 3 + 5 + 7


@m.jit(preprocess=False)
def for_carries(n: m.Int32, f: m.Float32) -> m.Float32:
    for i in m.for_(n):
        m.yield_()  # nothing: an empty terminator, no second one added
    p = Pair(lo=m.Int32(0), hi=m.Int32(1))
    # Several carries, nested containers among them: `iter_args` and results
    # come back in the inputs' shape with their DSL types, a struct being one
    # SSA value; a Python literal in the yield takes the carry's dtype.
    for i, (p_in, (f_in, b_in)), (p_out, (f_out, b_out)) in m.for_(
        n, iter_args=[p, (f, m.Int64(0))]
    ):
        m.yield_([p_in.replace(lo=p_in.lo + i), (f_in * 2.0, b_in + 5)])
    print("RESTORED:", type(p_out).__name__, type(f_out).__name__, type(b_out).__name__)
    return f_out + m.Float32(p_out.lo) + m.Float32(b_out)


# CHECK:       RESTORED: Pair Float32 Int64
# CHECK-LABEL: func.func @for_carries(
# CHECK:         scf.for
# CHECK-NEXT:    }
# CHECK:         %[[R:.+]]:4 = scf.for {{.*}} -> (i32, i32, f32, i64) : i32 {
# CHECK:           %[[C:.+]] = arith.constant 5 : i64
# CHECK:           arith.addi %{{.+}}, %[[C]] : i64
# CHECK:           scf.yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : i32, i32, f32, i64
# CHECK:         arith.sitofp %[[R]]#0 : i32 to f32
# EXEC:          RESULT: for_carries 42.0
report(for_carries, 4, 1.0)  # 16.0 + 6 + 20


@m.jit(preprocess=False)
def for_missing_yield(n: m.Int32) -> m.Int32:
    for i, acc_in, acc_out in m.for_(n, iter_args=[m.Int32(0)]):
        unused = acc_in + i
    return acc_out


# CHECK: error:{{.*}}The body of this `for_` has `iter_args` but did not end with `yield_(...)`.
report(for_missing_yield, 3)


@m.jit(preprocess=False)
def for_wrong_yield(n: m.Int32) -> m.Int32:
    # `yield_` checks what it yields against the carries of the enclosing
    # builder: another type (an f32 for an i32 carry) or another shape is a
    # diagnostic at the `yield_`, not a verifier failure later.
    for i, acc_in, acc_out in m.for_(n, iter_args=[m.Int32(0)]):
        m.yield_([m.Float32(acc_in) + 1.0])
    return acc_out


# CHECK: error[CONTAINER_STRUCTURE_CHANGED]:{{.*}}`the carried values` has a different structure at the end of this `for_` than at the start (`the carries[0]` changed from `Int32` (type `i32`) to `Float32` (type `f32`))
report(for_wrong_yield, 3)


# --- `if_` ------------------------------------------------------------------


@m.jit(preprocess=False)
def if_store(n: m.Int32, out: m.Pointer[m.Int32]):
    def then():
        out[0] = n

    print("IF_RETURNS:", m.if_(n > 2, then))  # no results: `[]`, no else region


# CHECK:       IF_RETURNS: []
# CHECK-LABEL: func.func @if_store(
# CHECK:         scf.if %{{.+}} {
# CHECK:           llvm.store {{.*}} : i32, !llvm.ptr
# CHECK-NEXT:    }
# CHECK-NOT:     else
# CHECK:         return
buf = np.zeros(1, np.int32)
if_store(5, buf)
if_store(1, buf)
# EXEC: RESULT: if_store [5]
print("RESULT: if_store", buf.tolist())


@m.jit(preprocess=False)
def if_results(n: m.Int32, f: m.Float32) -> m.Float32:
    # `return_types` without an else: the else yields `input_args` unchanged;
    # the arms' values are cast to `return_types` (a Python literal becomes
    # the dtype, a `@struct` class is one result per field) and come back in that order.
    p = Pair(lo=n, hi=n)
    a, b, q = m.if_(
        n > 2,
        lambda x, y, s: (x * 2, 7, s.replace(lo=s.lo + 1)),
        input_args=[n, f, p],
        return_types=[m.Int32, m.Float32, Pair],
    )
    print("RESULT_TYPES:", type(a).__name__, type(b).__name__, type(q).__name__)
    return b + m.Float32(a + q.lo)


# CHECK:       RESULT_TYPES: Int32 Float32 Pair
# CHECK-LABEL: func.func @if_results(
# CHECK-SAME:    %[[N:[^:]+]]: i32, %[[F:[^:]+]]: f32) -> f32
# CHECK:         %[[R:.+]]:4 = scf.if %{{.+}} -> (i32, f32, i32, i32) {
# CHECK:           %[[C:.+]] = arith.constant 7.000000e+00 : f32
# CHECK:           scf.yield %{{.+}}, %[[C]], %{{.+}}, %[[N]] : i32, f32, i32, i32
# CHECK:         } else {
# CHECK:           scf.yield %[[N]], %[[F]], %[[N]], %[[N]] : i32, f32, i32, i32
# EXEC:          RESULT: if_results 23.0 3.5
print("RESULT: if_results", if_results(5, 1.5), if_results(1, 1.5))


@m.jit(preprocess=False)
def if_too_few(n: m.Int32) -> m.Int32:
    a, b = m.if_(n > 2, lambda: (n,), lambda: (n, n), return_types=[m.Int32, m.Int32])
    return a


# CHECK: error:{{.*}}An `if_` body returned 1 value(s), but `return_types` lists 2.
report(if_too_few, 4)


# --- `while_` and `WhileLoopContext` -----------------------------------------


@m.jit(preprocess=False)
def while_context(n: m.Int32, f: m.Float32) -> m.Float32:
    # `while_(inputs, cond)`: the context exposes the inputs' IR types, `with`
    # binds the after-block carries as DSL values, `.results` the results.
    loop = m.while_([n, f], lambda i, x: i > 0)
    print(
        "CONTEXT:",
        isinstance(loop, m.WhileLoopContext),
        [str(t) for t in loop.input_ir_types],
    )
    with loop as (i, x):
        print("CARRIES:", type(i).__name__, type(x).__name__)
        m.yield_([i - 1, x * 2.0])
    return loop.results[1]


# CHECK:       CONTEXT: True ['i32', 'f32']
# CHECK:       CARRIES: Int32 Float32
# CHECK-LABEL: func.func @while_context(
# CHECK-SAME:    %[[N:[^:]+]]: i32, %[[F:[^:]+]]: f32) -> f32
# CHECK:         %[[R:.+]]:2 = scf.while (%[[I:.+]] = %[[N]], %[[X:.+]] = %[[F]]) : (i32, f32) -> (i32, f32) {
# CHECK:           scf.condition(%{{.+}}) %[[I]], %[[X]] : i32, f32
# CHECK:         } do {
# CHECK:           scf.yield %{{.+}}, %{{.+}} : i32, f32
# CHECK:         return %[[R]]#1 : f32
# EXEC:          RESULT: while_context 8.0
report(while_context, 3, 1.0)


@m.jit(preprocess=False)
def while_python_cond(n: m.Int32) -> m.Int32:
    loop = m.while_([n], lambda i: False)  # a Python condition is a constant:
    with loop as (i,):  # the builders never fold (the rewrite does)
        m.yield_([i + 1])
    return loop.results[0]


# CHECK-LABEL: func.func @while_python_cond(
# CHECK:         %[[F:.+]] = arith.constant false
# CHECK:         scf.condition(%[[F]])
# EXEC:          RESULT: while_python_cond 7
report(while_python_cond, 7)


@m.jit(preprocess=False)
def while_wrong_yield(n: m.Int32) -> m.Int32:
    loop = m.while_([n, n], lambda i, j: i > 0)
    with loop as (i, j):
        m.yield_([i - 1])  # one value for two carries
    return loop.results[0]


# CHECK: error[CONTAINER_STRUCTURE_CHANGED]:{{.*}}`the carried values` has a different structure at the end of this `while_` than at the start (`the carries` changed from `list` with 2 items to `list` with 1 item)
report(while_wrong_yield, 3)


# --- Bound promotion of the staged loop (Design 7.3) ------------------------


@m.jit
def promotion(a: m.Int32, b: m.Int64, c: m.Int8, d: m.Int16) -> m.Int64:
    wide = m.Int64(0)
    for i in range(a, b):  # Int32 start + Int64 stop: an i64 loop
        wide += i
    narrow = m.Int16(0)
    for i in range(c, d, m.Int16(2)):  # Int8 + Int16 + Int16: i16, nothing wider
        narrow += i  # (a Python int bound would be Int32 and widen the loop)
    return wide + m.Int64(narrow)


# CHECK-LABEL: func.func @promotion(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i64, %[[C:[^:]+]]: i8, %[[D:[^:]+]]: i16) -> i64
# CHECK:         arith.extsi %[[A]] : i32 to i64
# CHECK:         scf.for %{{.+}} = %{{.+}} to %[[B]] step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (i64) : i64 {
# CHECK:         arith.extsi %[[C]] : i8 to i16
# CHECK:         scf.for %{{.+}} = %{{.+}} to %[[D]] step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (i16) : i16 {
# EXEC:          RESULT: promotion 19
report(promotion, 1, 5, 1, 7)  # (1 + 2 + 3 + 4) + (1 + 3 + 5)


# --- Loop options and bounds that are rejected -------------------------------


@m.jit
def unknown_option(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, pipelining=2):  # a sub-DSL option the base does not know
        acc += i
    return acc


# CHECK: ERROR: unknown_option CALL_UNEXPECTED_KWARG
# CHECK: error[CALL_UNEXPECTED_KWARG]:{{.*}}This call passes a keyword argument `pipelining`
report(unknown_option, 4)


@m.jit
def staged_unroll(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, unroll=n):  # (`unroll=True` is rejected as well)
        acc += i
    return acc


# CHECK: ERROR: staged_unroll PHASE_REQUIRES_CONSTANT
# CHECK: error[PHASE_REQUIRES_CONSTANT]:{{.*}}`unroll` requires a value known at compile time
report(staged_unroll, 4)


@m.jit
def float_stop(n: m.Float32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += 1
    return acc


# CHECK: ERROR: float_stop TYPE_LOOP_BOUND_NOT_INT
# CHECK: error[TYPE_LOOP_BOUND_NOT_INT]:{{.*}}The loop's `stop` is a `Float32`
report(float_stop, 4.0)


@m.jit
def negative_step(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n, 0, -1):
        acc += 1
    return acc


# CHECK: error:{{.*}}The loop's `step` is `-1`, but a loop controlled by a runtime value needs a positive step.
# CHECK: suggestion:{{.*}}a plain `range(...)` runs in Python
report(negative_step, 4)

with ir.Context():  # `unroll=1` means "do not unroll": the attribute says so
    print("LOOP_UNROLL:", executors.LoopUnroll(count=1))
# CHECK: LOOP_UNROLL: #llvm.loop_annotation<unroll = <disable = true, count = 1 : i32>>


# --- The join checks (`ScfGenerator`, Design 4, 7.5, 8) ----------------------


@m.jit
def meta_in_for(n: m.Int32) -> m.Int32:
    acc = 0
    for i in range(n):
        acc += i
    return acc


report(meta_in_for, 4)
# CHECK:      ERROR: meta_in_for PHASE_MUTATE_PYTHON
# CHECK:      error[PHASE_MUTATE_PYTHON]:{{.*}}`acc` is a Python value, but it is changed inside a for/while/if controlled by a runtime value (this `for`). Only a runtime value can change there
# CHECK:      -->{{.*}}executors_builders.py:[[#@LINE-8]]
# CHECK:      = note:{{.*}}region: for
# CHECK:      suggestion:{{.*}}Create `acc` as a runtime value of the matching type before the for/while/if
# CHECK-NEXT: `acc = Int32(0)`


@m.jit
def type_in_for(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc = m.Float32(acc) + 1.0
    return acc


# CHECK: ERROR: type_in_for TYPE_UNSTABLE_JOIN
# CHECK: error[TYPE_UNSTABLE_JOIN]:{{.*}}`acc` has type `Int32` on one path and `Float32` on another (in this `for`). `acc` must have one type wherever these paths come back together.
# CHECK: suggestion:{{.*}}Make every assignment to `acc` produce the same type.
report(type_in_for, 4)


@m.jit
def none_on_one_arm(n: m.Int32) -> m.Int32:
    if n > 2:
        y = n + 1  # the else arm leaves the if-born `y` as `None`
    return y


# CHECK: error[TYPE_UNSTABLE_JOIN]:{{.*}}`y` has type `Int32` on one path and `None` on another (between the `then` and `else` arms of this `if`).
report(none_on_one_arm, 4)


@m.jit
def tuple_length(n: m.Int32) -> m.Int32:
    t = (n, n)
    for i in range(n):
        t = (t[0] + i,)
    return t[0]


# CHECK: ERROR: tuple_length CONTAINER_STRUCTURE_CHANGED
# CHECK: error[CONTAINER_STRUCTURE_CHANGED]:{{.*}}`t` has a different structure at the end of this `for` than at the start (`t` changed from `tuple` with 2 items to `tuple` with 1 item).
# CHECK: suggestion:{{.*}}Assign `t` a value of the same type and structure on every branch and every iteration
report(tuple_length, 4)


@m.jit
def meta_disagrees(n: m.Int32) -> m.Int32:
    if n > 2:
        k = 1  # two arms set an if-born Meta to different Python values
    else:
        k = 2
    return n + k


# CHECK: error[CONTAINER_STRUCTURE_CHANGED]:{{.*}}`k` has a different structure at the end of this `if` than at the start (`k` changed from a Python value `1` to a Python value `2`).
report(meta_disagrees, 4)


@m.jit
def joins_pass(n: m.Int32, out: m.Pointer[m.Int32]) -> m.Int32:
    # An unchanged Meta, equal Metas and `None` on both arms, and a store-only
    # pointer all pass: nothing is carried for them.
    k = 3
    acc = m.Int32(0)
    for i in range(n):
        acc = acc + i * k
        out[i] = acc
    if n > 2:
        k2, y = 7, None
    else:
        k2, y = 7, None
    while acc > 100:
        acc = acc - k2
    print("PASSES:", k, k2, y)
    return acc


# CHECK:       PASSES: 3 7 None
# CHECK-LABEL: func.func @joins_pass(
# CHECK:         scf.for {{.*}} iter_args(%{{.+}} = %{{.+}}) -> (i32) : i32 {
# CHECK:         scf.if %{{.+}} {
# CHECK-NEXT:    }
# CHECK:         scf.while (%{{.+}} = %{{.+}}) : (i32) -> i32 {
# CHECK:           arith.constant 7 : i32
# EXEC:          RESULT: joins_pass 18
report(joins_pass, 4, np.zeros(4, np.int32))


@m.jit
def turns_staged(a: m.Int32) -> m.Int32:
    x = 0
    while x < 2:  # a Python bool first, a staged value later
        x = x + a
    return x


@m.jit
def ternary_lists(n: m.Int32) -> m.Int32:
    # The arms of a staged ternary are whole values: a one-element list stays
    # a list (not its element), a record is rebuilt on the join.
    xs = [n + 1] if n > 2 else [n - 1]
    pair = (n, n * 2) if n > 2 else (n * 3, n)
    print("TERNARY:", type(xs).__name__, len(xs), type(pair).__name__)
    return xs[0] + pair[1]


# CHECK:       TERNARY: list 1 tuple
# CHECK-LABEL: func.func @ternary_lists(
# CHECK:         scf.if %{{.+}} -> (i32) {
# CHECK:         scf.if %{{.+}} -> (i32, i32) {
# EXEC:          RESULT: ternary_lists 10
report(ternary_lists, 3)  # xs[0] = 4, pair[1] = 6


# CHECK: ERROR: turns_staged PHASE_DYNAMIC_TO_STATIC_BOOL
# CHECK: = note:{{.*}}the `while` condition was a Python value on the first evaluation and a runtime value on a later one
report(turns_staged, 3)


@m.jit(preprocess=False)
def direct_executors(n: m.Int32) -> m.Int32:
    # The executors called directly (a sub-DSL's view): a Python predicate
    # returns the write-back shape; a `None`-seeded slot takes the arms' type;
    # a Meta comparison chain is a Python bool.
    one = executors._if_execute_dynamic(False, lambda x: [x + 1], None, [n], 1, ["x"])
    x, y = executors._if_execute_dynamic(
        n > 1,
        lambda x, y: [x + 1, x],
        lambda x, y: [x - 1, x],
        [n, None],
        2,
        ["x", "y"],
    )
    meta = executors._compare_executor(1, [2, 3], ["<", "<"])
    r = executors._ifexp_execute_dynamic(n > 1, (n,), lambda v: v + 1, lambda v: v - 1)
    print("DIRECT:", one is n, type(y).__name__, meta, type(r).__name__)
    return y + r


# CHECK:       DIRECT: True Int32 True Int32
# CHECK-LABEL: func.func @direct_executors(
# CHECK:         %[[R:.+]]:2 = scf.if %{{.+}} -> (i32, i32) {
# CHECK:         scf.if %{{.+}} -> (i32) {
# EXEC:          RESULT: direct_executors 9
report(direct_executors, 4)


# --- The ternary executor ----------------------------------------------------


@m.jit
def ternary_tuple(a: m.Int32) -> m.Int32:
    t = (a, a + 1) if a > 2 else (a + 1, a)  # several results, restored as a tuple
    return t[0] * 10 + t[1]


# CHECK-LABEL: func.func @ternary_tuple(
# CHECK:         %[[R:.+]]:2 = scf.if %{{.+}} -> (i32, i32) {
# CHECK:         arith.muli %[[R]]#0
# EXEC:          RESULT: ternary_tuple 56 21
print("RESULT: ternary_tuple", ternary_tuple(5), ternary_tuple(1))


@m.jit
def ternary_mismatch(a: m.Int32) -> m.Int32:
    return a + 1 if a > 2 else m.Float32(1.0)


# CHECK: ERROR: ternary_mismatch TYPE_CONDITIONAL_BRANCH_MISMATCH
# CHECK: error[TYPE_CONDITIONAL_BRANCH_MISMATCH]:{{.*}}The two branches of this `x if cond else y` expression produce different types.
report(ternary_mismatch, 3)


@m.jit
def ternary_meta_arm(a: m.Int32) -> m.Int32:
    return 1 if a > 2 else a  # a Python value is not cast to the other arm


# CHECK: ERROR: ternary_meta_arm TYPE_CONDITIONAL_BRANCH_MISMATCH
report(ternary_meta_arm, 3)


# --- `and_`/`or_`/`not_`/`any_`/`all_`/`in_` (Design 7.5) -------------------

# On Meta values they are Python's operators; `any_`/`all_` always answer a
# folded `Boolean`.
print("META:", m.and_(1, 2, 3), m.or_(0, 0, 4), m.not_(0), in_(2, [1, 2]))
# CHECK: META: 3 4 True True
every, some = m.all_([True, 1, 2.5]), m.any_([0, False])
print("REDUCE:", type(every).__name__, bool(every), bool(some), bool(m.all_([])))
# CHECK: REDUCE: Boolean True False True


@m.jit(preprocess=False)
def helpers_staged(a: m.Int32, b: m.Int32) -> m.Int32:
    # On staged values every helper answers a `Boolean`; `and_`/`or_` fold the
    # leading Meta operands and reduce left to right; `in_` is `any_` of the
    # equalities.
    x, y = a > 1, b > 2
    print(
        "TYPES:",
        type(m.and_(x, y, b > 3)).__name__,
        type(m.not_(x)).__name__,
        type(m.all_([x, y])).__name__,
        type(in_(a, (1, 2))).__name__,
    )
    return (
        m.Int32(m.all_([x, y]))
        + m.Int32(in_(a, (1, 2))) * 2
        + m.and_(1, 2, a)
        + m.or_(0, a, 100)
    )


# CHECK:       TYPES: Boolean Boolean Boolean Boolean
# CHECK-LABEL: func.func @helpers_staged(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32) -> i32
# CHECK-DAG:     arith.cmpi eq, %[[A]], %{{.+}} : i32
# CHECK:         arith.select
# EXEC:          RESULT: helpers_staged 7 100
print("RESULT: helpers_staged", helpers_staged(2, 4), helpers_staged(0, 0))


@m.jit
def in_non_sequence(a: m.Int32) -> m.Boolean:
    return a in 5  # `in` needs a sequence when an operand is staged


# CHECK: ERROR: in_non_sequence UNSUP_COMPARISON_OPERATOR
# CHECK: error[UNSUP_COMPARISON_OPERATOR]:{{.*}}The comparison operator `in` is not supported in compiled code.
report(in_non_sequence, 3)


# --- `max`/`min`/`any`/`all` through `_builtin_redirector` (Design 7.3) ------


@m.jit
def builtin_max_min(a: m.Int32, b: m.Int32, k) -> m.Int32:
    # With a staged argument the builtins go to the DSL's `max`/`min` (a Python
    # operand becomes a constant; lists and several arguments reduce); with
    # Meta arguments only they stay Python's, keywords included.
    staged = max([a, b, 3]) * 10 + min(a, b)
    meta = max(k, 2) + min([k, 7], key=lambda v: -v)
    return staged + meta


# CHECK-LABEL: func.func @builtin_max_min_5(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32) -> i32
# CHECK:         arith.maxsi %[[A]], %[[B]] : i32
# CHECK:         arith.maxsi %{{.+}}, %{{.+}} : i32
# CHECK:         arith.minsi %[[A]], %[[B]] : i32
# CHECK-NOT:     arith.maxsi
# CHECK-NOT:     arith.minsi
# CHECK:         return
# EXEC:          RESULT: builtin_max_min 84
report(builtin_max_min, 2, 7, 5)  # 70 + 2 + 5 + 7


@m.jit
def builtin_any_all(a: m.Int32) -> m.Int32:
    r = m.Int32(0)
    if any([a == 1, a == 2]):  # `any_`/`all_` of staged Booleans
        r = r + 1
    if all([a > 0, a < 10]):
        r = r + 2
    return r


# CHECK-LABEL: func.func @builtin_any_all(
# CHECK:         arith.cmpi eq
# CHECK:         arith.cmpi eq
# CHECK:         scf.if
# CHECK:         arith.andi
# CHECK:         scf.if
# EXEC:          RESULT: builtin_any_all 3 2 0
print(
    "RESULT: builtin_any_all",
    builtin_any_all(2),
    builtin_any_all(5),
    builtin_any_all(20),
)


@m.jit
def max_keyword(a: m.Int32, b: m.Int32) -> m.Int32:
    return max(a, b, key=abs)


# CHECK: ERROR: max_keyword CALL_BUILTIN_KWARGS_UNSUPPORTED
# CHECK: error[CALL_BUILTIN_KWARGS_UNSUPPORTED]:{{.*}}`max` does not accept keyword arguments when one of its arguments is a runtime value.
report(max_keyword, 3, 4)


@m.jit(preprocess=False)
def other_builtin(a: m.Int32) -> m.Int32:
    return _builtin_redirector(len)([a])  # any other builtin with a staged argument


# CHECK: ERROR: other_builtin UNSUP_BUILTIN
# CHECK: error[UNSUP_BUILTIN]:{{.*}}The built-in function `len` is not allowed in compiled code when one of its arguments is a runtime value.
report(other_builtin, 3)
