# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
import mlir.mlir_dsl as m
from mlir.dsl.plugins.ast_preprocessor.scf import executors


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


@m.jit
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


@m.jit
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


@m.jit
def for_missing_yield(n: m.Int32) -> m.Int32:
    for i, acc_in, acc_out in m.for_(n, iter_args=[m.Int32(0)]):
        unused = acc_in + i
    return acc_out


# CHECK: error:{{.*}}The body of this `for_` has `iter_args` but did not end with `yield_(...)`.
report(for_missing_yield, 3)


@m.jit
def for_wrong_yield(n: m.Int32) -> m.Int32:
    # `yield_` checks what it yields against the carries of the enclosing
    # builder: another type (an f32 for an i32 carry) or another shape is a
    # diagnostic at the `yield_`, not a verifier failure later.
    for i, acc_in, acc_out in m.for_(n, iter_args=[m.Int32(0)]):
        m.yield_([m.Float32(acc_in) + 1.0])
    return acc_out


# CHECK: error[CONTAINER_STRUCTURE_CHANGED]:{{.*}}`the carried values` has a different structure at the end of this `for_` than at the start (`the carries[0]` changed from `Int32` (type `i32`) to `Float32` (type `f32`))
report(for_wrong_yield, 3)


@m.jit
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


@m.jit
def if_too_few(n: m.Int32) -> m.Int32:
    a, b = m.if_(n > 2, lambda: (n,), lambda: (n, n), return_types=[m.Int32, m.Int32])
    return a


# CHECK: error:{{.*}}An `if_` body returned 1 value(s), but `return_types` lists 2.
report(if_too_few, 4)


@m.jit
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


@m.jit
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


@m.jit
def while_wrong_yield(n: m.Int32) -> m.Int32:
    loop = m.while_([n, n], lambda i, j: i > 0)
    with loop as (i, j):
        m.yield_([i - 1])  # one value for two carries
    return loop.results[0]


# CHECK: error[CONTAINER_STRUCTURE_CHANGED]:{{.*}}`the carried values` has a different structure at the end of this `while_` than at the start (`the carries` changed from `list` with 2 items to `list` with 1 item)
report(while_wrong_yield, 3)


@m.jit
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
