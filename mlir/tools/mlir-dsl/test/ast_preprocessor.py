# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_AST_PREPROCESSOR=0 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=OFF
# REQUIRES: host-supports-jit
# The preprocessor's decisions (Design 7): which native construct becomes
# Python control flow at trace time (Meta) and which an `scf` region (staged);
# what a region carries; how `and`/`or`/`not`, comparison chains, `assert`,
# `bool()` and closures are rewritten; which early exits stay native Python
# and which are rejected; and the switch that turns the rewrite off. What the `scf` ops then
# mean is the dialect's business and is not checked here.
import contextlib

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.core.env_manager import EnvironmentVarManager


def report(fn, *args):
    try:
        print("RESULT:", fn.__name__, fn(*args))
    except AssertionError as e:
        print("ASSERTION:", fn.__name__, e)
    except m.DSLUserCodeError as e:
        print("ERROR:", fn.__name__, e.diag_id.name)


@m.struct
class Pair:
    lo: m.Int32
    hi: m.Int32


# --- The switch (Design 7.8) ------------------------------------------------
# `MLIR_DSL_AST_PREPROCESSOR=0` turns the rewrite off for the whole process,
# `@m.jit(preprocess=False)` for one function. Without it, native control flow
# on a staged value fails in plain Python; Meta control flow and the explicit
# builders work either way. The rest of this file needs the rewrite.
PREPROCESSOR_ON = EnvironmentVarManager("MLIR_DSL").ast_preprocessor
print("PREPROCESSOR:", PREPROCESSOR_ON, m.MlirDSL().enable_preprocessor)
# CHECK: PREPROCESSOR: True True
# OFF:   PREPROCESSOR: False False


@m.jit
def staged_loop(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


# CHECK: RESULT: staged_loop ?
# OFF:   ERROR: staged_loop PHASE_DYNAMIC_INDEX
report(staged_loop, 10)


@m.jit(preprocess=False)
def opted_out(n: m.Int32) -> m.Int32:
    r = m.Int32(0)
    if n > 3:
        r = n
    return r


# CHECK: ERROR: opted_out PHASE_DYNAMIC_TO_STATIC_BOOL
# OFF:   ERROR: opted_out PHASE_DYNAMIC_TO_STATIC_BOOL
report(opted_out, 10)


@m.jit
def meta_and_builders(n, k: m.Int32) -> m.Int32:
    acc = k
    for i in range(n):  # a Meta bound: Python under both settings
        acc += i
    for i, acc_in, acc_out in m.for_(0, k, 1, [acc]):
        m.yield_([acc_in + i])
    return acc_out


# CHECK: RESULT: meta_and_builders ?
# OFF:   RESULT: meta_and_builders ?
# EXEC:  RESULT: meta_and_builders 61
report(meta_and_builders, 4, 10)  # 16 + 0..9

if not PREPROCESSOR_ON:
    raise SystemExit(0)


# --- Loop kinds by bound (Design 7.3, 7.4, 7.6) -----------------------------


@m.jit
def meta_bound(n, k: m.Int32) -> m.Int32:
    # A bare `range` over a Meta bound is `builtins.range` (the options are
    # dropped) and a display is a Python iterable: Python loops at trace time,
    # the body traced once per iteration, a `break` under a Meta `if` Python's.
    acc = k
    for i in range(n, unroll=2):
        if i > 1:
            break
        acc = acc + i
    for a, b in ((1, 2), (3, 4)):
        acc = acc + a * b
    for i in m.range(2, n, unroll=4):  # the DSL's `range` stages even over Meta
        acc = acc + i  # bounds; `unroll=` is the loop annotation
    return acc


# CHECK:       #llvm.loop_unroll<count = 4 : i32>
# CHECK-LABEL: func.func @meta_bound_4(
# CHECK-NOT:     scf.for
# CHECK-COUNT-4: arith.addi
# CHECK-DAG:     %[[LB:.+]] = arith.constant 2 : i32
# CHECK-DAG:     %[[UB:.+]] = arith.constant 4 : i32
# CHECK:         scf.for %{{.+}} = %[[LB]] to %[[UB]]
# CHECK:         } {llvm.loop_annotation = #loop_annotation
# EXEC:          RESULT: meta_bound 30
report(meta_bound, 4, 10)  # 10 + 0 + 1 + 2 + 12 + 2 + 3


@m.jit
def staged(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    cnt = m.Int32(0)
    for i in range(n):  # a staged stop: `scf.for`, i32 bounds, no `index`
        for j in range(i, unroll_full=True):  # the bare `range` takes the
            acc += j  # options too; both carries thread through both loops
            cnt += 1
    return acc + cnt


# CHECK:       #llvm.loop_unroll<full = true>
# CHECK-LABEL: func.func @staged(
# CHECK-SAME:    %[[N:[^:]+]]: i32) -> i32
# CHECK-NOT:     index
# CHECK:         %[[OUT:.+]]:2 = scf.for %[[I:.+]] = %{{.+}} to %[[N]] step %{{.+}} iter_args(%[[A:.+]] = %{{.+}}, %[[C:.+]] = %{{.+}}) -> (i32, i32) : i32 {
# CHECK:           %[[IN:.+]]:2 = scf.for %{{.+}} = %{{.+}} to %[[I]] step %{{.+}} iter_args(%{{.+}} = %[[A]], %{{.+}} = %[[C]]) -> (i32, i32) : i32 {
# CHECK:           } {llvm.loop_annotation = #loop_annotation
# CHECK:           scf.yield %[[IN]]#0, %[[IN]]#1 : i32, i32
# CHECK:         arith.addi %[[OUT]]#0, %[[OUT]]#1 : i32
# EXEC:          RESULT: staged 20
report(staged, 5)  # acc = 0+0+1+3+6, cnt = 10


@m.jit
def meta_control(k, n: m.Int32) -> m.Int32:
    # Control flow on Meta values is Python's, decided at trace time. An `if`
    # or `while` that owns an early exit stays a Python statement whose
    # condition must be Meta (`early_exit_predicate`), so it may exit early.
    if k > 5:
        return n
    acc = n
    while k > 0:  # a Python loop over the Meta counter `k`
        acc = acc + k
        k -= 1
    # A `range` over Meta bounds is a Python loop: `break`, `continue` and
    # `else:` follow Python. (A body-born name still may not be read after any
    # loop: `last` is set before it.)
    last = 0
    for i in range(10):
        if i == 1:
            continue
        if i == 3:
            break
        last = i
        acc = acc + i
    else:
        acc = acc + 1000
    return acc + last


# CHECK-LABEL: func.func @meta_control_3(
# CHECK-NOT:     scf.
# CHECK-COUNT-6: arith.addi
# CHECK-NEXT:    return
# CHECK-LABEL: func.func @meta_control_9(
# CHECK-NOT:     arith.addi
# CHECK:         return
# EXEC:          RESULT: meta_control 20 10
print("RESULT: meta_control", meta_control(3, 10), meta_control(9, 10))


# --- Loop carries (Design 7.2, the write_args protocol) ---------------------


@m.jit
def carries(n: m.Int32) -> m.Float32:
    k = 3  # read in the body, never stored: a folded constant, not a carry
    a, b = m.Int32(0), m.Int32(1)
    total = m.Float32(0.0)
    p = Pair(lo=m.Int32(0), hi=m.Int32(0))
    t = (m.Int32(1), m.Float32(0.0))
    for i in range(n):  # every stored name is an iter_arg, in first-store order
        a, b = b, a + b  # a tuple-unpacking store: both carried
        total = total + m.Float32(k)
        p = p.replace(lo=p.lo + i)  # a struct and a tuple carry: flattened to
        t = (t[0] * 2, t[1] + 1.0)  # their leaves, restored after the loop
    return total + m.Float32(a + p.lo + t[0])


# CHECK-LABEL: func.func @carries(
# CHECK:         scf.for {{.*}} iter_args(%[[A:.+]] = %{{.+}}, %[[B:.+]] = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) -> (i32, i32, f32, i32, i32, i32, f32) : i32 {
# CHECK:           %[[S:.+]] = arith.addi %[[A]], %[[B]] : i32
# CHECK:           arith.constant 3.000000e+00 : f32
# CHECK:           scf.yield %[[B]], %[[S]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : i32, i32, f32, i32, i32, i32, f32
# EXEC:          RESULT: carries 1154.0
report(carries, 10)  # 30.0 + (55 + 45 + 1024)


@m.jit
def pointers(out: m.Pointer[m.Int32], p: m.Pointer[m.Int32], n: m.Int32) -> m.Int32:
    i = m.Int32(100)  # bound before the loop that rebinds it as its target
    for i in range(n):
        out[i] = i * 2  # a subscript store marks `out` mutated: no carry (Design 4)
        p[0] = i
        p = p + 1  # a rebound pointer is an ordinary carry of type !llvm.ptr
    return i  # the last induction value, or 100 when the loop ran zero times


# The live target is carried too (as `loop_carried_var_N`), seeded with 100.
# CHECK-LABEL: func.func @pointers(
# CHECK-SAME:    %[[OUT:[^:]+]]: !llvm.ptr, %[[P:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32) -> i32
# CHECK:         %[[INIT:.+]] = arith.constant 100 : i32
# CHECK:         %[[R:.+]]:2 = scf.for %[[I:.+]] = %{{.+}} to %[[N]] step %{{.+}} iter_args(%[[Q:.+]] = %[[P]], %{{.+}} = %[[INIT]]) -> (!llvm.ptr, i32) : i32 {
# CHECK:           llvm.getelementptr %[[OUT]][%[[I]]]
# CHECK:           %[[NEXT:.+]] = llvm.getelementptr %[[Q]][1]
# CHECK:           scf.yield %[[NEXT]], %[[I]] : !llvm.ptr, i32
# CHECK:         return %[[R]]#1 : i32
buf, buf2 = np.zeros(4, np.int32), np.full(4, -1, np.int32)
# EXEC: RESULT: pointers 3 100 [0 2 4 6] [0 1 2 3]
print("RESULT: pointers", pointers(buf, buf2, 4), pointers(buf, buf2, 0), buf, buf2)


# --- `if`/`elif`/`else` (Design 7.2, 7.4) -----------------------------------


@m.jit
def ifs(out: m.Pointer[m.Int32], n: m.Int32) -> m.Int32:
    if n > 5:  # staged: an `scf.if` whose results are the names the arms store
        r = n * 2  # born in both arms and read after: seeded `None`,
    else:
        r = n + 1  # the arms decide its type
    if n > 7:  # without else: the synthesised else yields the carry unchanged
        r = r + 100
    if n > 9:  # no stored name: no results, and the empty else is dropped
        out[0] = r
    return r


# CHECK-LABEL: func.func @ifs(
# CHECK-SAME:    %[[OUT:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32) -> i32
# CHECK:         %[[C:.+]] = arith.cmpi sgt, %[[N]], %{{.+}} : i32
# CHECK:         %[[R:.+]] = scf.if %[[C]] -> (i32) {
# CHECK:           arith.muli
# CHECK:         } else {
# CHECK:           arith.addi
# CHECK:         %[[R2:.+]] = scf.if %{{.+}} -> (i32) {
# CHECK:         } else {
# CHECK-NEXT:      scf.yield %[[R]] : i32
# CHECK:         scf.if %{{.+}} {
# CHECK-NOT:     -> (
# CHECK:           llvm.store
# CHECK-NOT:     } else {
# CHECK:         return %[[R2]] : i32
buf = np.zeros(1, np.int32)
# EXEC: RESULT: ifs 120 4 [120]
print("RESULT: ifs", ifs(buf, 10), ifs(buf, 3), buf)


@m.jit
def nesting(flag, n: m.Int32) -> m.Int32:
    r = n
    if n > 3:  # staged
        if flag:  # a Meta predicate runs one arm in Python, also inside the arm
            r = n + 100
        else:
            r = n + 200
    if flag:  # Meta
        if n > 50:  # staged inside the taken arm only
            r = r + 1000
    return r


# CHECK-LABEL: func.func @nesting_True(
# CHECK:         scf.if %{{.+}} -> (i32) {
# CHECK-NOT:       scf.if
# CHECK:           arith.constant 100 : i32
# CHECK:         } else {
# CHECK:         scf.if %{{.+}} -> (i32) {
# CHECK:           arith.constant 1000 : i32
# CHECK-LABEL: func.func @nesting_False(
# CHECK:         scf.if %{{.+}} -> (i32) {
# CHECK:           arith.constant 200 : i32
# CHECK:         } else {
# CHECK-NOT:     scf.if
# CHECK:         return
# EXEC:          RESULT: nesting 110 210 1160
print("RESULT: nesting", nesting(True, 10), nesting(False, 10), nesting(True, 60))


@m.jit
def elif_chain(flag, n: m.Int32) -> m.Int32:
    r = n
    if n > 100:  # staged
        r = n + 1
    elif n > 50:  # `elif` nests an `scf.if` in the else region ...
        r = n + 2
    elif flag:  # ... unless it is a Python decision
        r = n + 10
    else:
        r = n + 100
    return r


# CHECK-LABEL: func.func @elif_chain_True(
# CHECK:         scf.if %{{.+}} -> (i32) {
# CHECK:         } else {
# CHECK:           %[[R2:.+]] = scf.if %{{.+}} -> (i32) {
# CHECK:           } else {
# CHECK-NOT:         scf.if
# CHECK:             arith.constant 10 : i32
# CHECK-NOT:         arith.constant 100
# CHECK:           scf.yield %[[R2]] : i32
# EXEC:          RESULT: elif_chain 20 110 62 201
print(
    "RESULT: elif_chain",
    elif_chain(True, 10),
    elif_chain(False, 10),
    elif_chain(True, 60),
    elif_chain(False, 200),
)


# --- The ternary (Design 7.1, 7.5) ------------------------------------------


@m.jit
def ternary(flag, n: m.Int32) -> m.Int32:
    a = n * 2 if n > 5 else n + 1  # staged predicate: both arms, one result
    vals = [a + i if n > 2 else a - i for i in range(2)]  # `i` is a block arg
    return vals[0] * 2 if flag else vals[0] + vals[1]  # Meta predicate: one arm


# CHECK-LABEL: func.func @ternary_False(
# CHECK-COUNT-3: scf.if %{{.+}} -> (i32) {
# CHECK-NOT:     scf.if
# CHECK:         return
# EXEC:          RESULT: ternary 41 40
print("RESULT: ternary", ternary(False, 10), ternary(True, 10))


def ctx(_):
    return contextlib.nullcontext()


@m.jit
def hoisted_ternaries(out: m.Pointer[m.Int32], a: m.Int32) -> m.Int32:
    # The arm blocks are hoisted in front of the statement that holds the
    # ternary, also from a `with` item, a subscript target and a call argument.
    with ctx(a + 1 if a > 3 else a - 1):
        out[m.Int32(1) if a > 3 else m.Int32(0)] = a
    return m.max(a, a * 2 if a > 3 else a + 100)


# CHECK-LABEL: func.func @hoisted_ternaries(
# CHECK:         scf.if %{{.+}} -> (i32) {
# CHECK:         %[[IDX:.+]] = scf.if %{{.+}} -> (i32) {
# CHECK:         llvm.getelementptr %{{.+}}[%[[IDX]]]
# CHECK:         llvm.store
# CHECK:         %[[V:.+]] = scf.if %{{.+}} -> (i32) {
# CHECK:         arith.maxsi %{{.+}}, %[[V]] : i32
buf = np.zeros(2, np.int32)
# EXEC: RESULT: hoisted_ternaries 20 [ 0 10]
print("RESULT: hoisted_ternaries", hoisted_ternaries(buf, 10), buf)


# --- `and`/`or`, comparison chains, `assert`, `bool()` (Design 7.1, 7.5) ----

evaluations = []


def side_effect(value):
    evaluations.append(value)
    return True


@m.jit
def boolops(flag, a: m.Int32) -> m.Boolean:
    first = flag and side_effect("and-rhs")  # a Python-bool left operand
    second = flag or side_effect("or-rhs")  # short-circuits exactly as in Python
    # A staged right operand goes through `and_`; a chain is `compare_executor`
    # `and_`ing its links, seeded with True; a False left operand is False and
    # the right side is not traced.
    return first and 0 < a < 10


r_false = boolops(False, 5)
seen_false, evaluations = list(evaluations), []
r_true = boolops(True, 5)
print("SHORT_CIRCUIT:", seen_false, evaluations)
# CHECK-LABEL: func.func @boolops_False(
# CHECK-NOT:     cmpi
# CHECK:         return %false : i1
# CHECK-LABEL: func.func @boolops_True(
# CHECK:         %[[C1:.+]] = arith.cmpi sgt
# CHECK:         %[[A1:.+]] = arith.andi %true, %[[C1]] : i1
# CHECK:         %[[C2:.+]] = arith.cmpi slt
# CHECK:         %[[A2:.+]] = arith.andi %[[A1]], %[[C2]] : i1
# CHECK:         %[[A3:.+]] = arith.andi %true{{(_[0-9]+)?}}, %[[A2]] : i1
# CHECK:         return %[[A3]] : i1
# CHECK:       SHORT_CIRCUIT: ['or-rhs'] ['and-rhs']
# EXEC:          RESULT: boolops True False False
print("RESULT: boolops", r_true, boolops(True, 10), r_false)


@m.jit
def assert_and_bool(n, k: m.Int32) -> m.Int32:
    assert n > 0, "n must be positive"  # a Python assertion on a Meta value
    # `bool(x)` is Python's on a Meta value; `bool` as a name is untouched.
    if bool(n) and isinstance(n, int):  # Meta: a Python `if`, may `return`
        return k + n
    return k


# CHECK-LABEL: func.func @assert_and_bool_1(
# CHECK-NOT:     cmpi
# CHECK:         return
# EXEC:          RESULT: assert_and_bool 11
# EXEC:          ASSERTION: assert_and_bool n must be positive
report(assert_and_bool, 1, 10)
report(assert_and_bool, 0, 10)


@m.jit
def assert_staged(k: m.Int32) -> m.Int32:
    assert bool(k > 0)  # cannot be decided at trace time
    return k


# CHECK: ERROR: assert_staged PHASE_REQUIRES_CONSTANT
report(assert_staged, 10)


# --- Closures and nested functions (Design 7.1, 7.2) ------------------------


def make_scaler(k):
    @m.jit
    def scale(n: m.Int32) -> m.Int32:
        acc = m.Int32(0)
        for i in range(n):
            acc = acc + k  # a closure cell of the jit function: Meta 7
        return acc

    return scale


# CHECK-LABEL: func.func @scale(
# CHECK:         scf.for
# CHECK:           arith.constant 7 : i32
# EXEC:          RESULT: scale 28
report(make_scaler(7), 4)


@m.jit
def nested_functions(n: m.Int32) -> m.Int32:
    k = m.Int32(2)
    acc = m.Int32(0)

    def inc(x):
        return x + 2  # capture-free: may be called from a staged region

    def add_k(x):
        return x + k  # captures `k`: fine outside any staged region

    def bump():
        nonlocal acc
        if n > 3:  # a staged `if` in a nested function keeps the write-back
            acc = acc + 1

    for i in range(n):
        acc = inc(acc)
    bump()
    return add_k(acc)


# CHECK-LABEL: func.func @nested_functions(
# CHECK:         scf.for
# CHECK:         scf.if %{{.+}} -> (i32) {
# EXEC:          RESULT: nested_functions 13 4
print("RESULT: nested_functions", nested_functions(5), nested_functions(1))


@m.jit
def capture_in_region(n: m.Int32) -> m.Int32:
    k = m.Int32(2)

    def inc(x):
        return x + k

    acc = m.Int32(0)
    for i in range(n):
        acc = inc(acc)  # a capture inside a staged region (`closure_check`)
    return acc


try:
    capture_in_region(4)
except m.DSLUserCodeError as e:
    print(str(e))
# CHECK: error[SCOPE_CLOSURE_CAPTURE]:{{.*}} Function `inc` captures variable `k`
# CHECK: suggestion:{{.*}}Pass `k` to `inc` as an argument.


@m.jit
def staging_probe(n: m.Int32, k) -> m.Int32:
    # `is_dynamic_expr` tells a staged value from a Meta one inside a trace
    # (the active DSL's notion); outside a trace a host `Int32` is Meta too.
    print(
        "IS_DYNAMIC:",
        m.is_dynamic_expr(n),
        m.is_dynamic_expr(k),
        m.is_dynamic_expr(n + k),
        m.is_dynamic_expr(k + 1),
        m.is_dynamic_expr(m.Int32(1)),
    )
    return n


staging_probe(1, 2)
print("IS_DYNAMIC HOST:", m.is_dynamic_expr(7), m.is_dynamic_expr(m.Int32(3)))
# CHECK: IS_DYNAMIC: True False True False False
# CHECK: IS_DYNAMIC HOST: False False


# --- Early exits and loop `else` (Design 7.2) -------------------------------


@m.jit
def return_in_if(n: m.Int32) -> m.Int32:
    if n > 3:
        return n
    return n + 1


try:
    return_in_if(10)
except m.DSLUserCodeError as e:
    print(str(e))
# CHECK:      error[UNSUP_EARLY_EXIT]:{{.*}} Early exit (return) is not allowed in `return_in_if`.
# CHECK:      -->{{.*}}ast_preprocessor.py:[[#@LINE-9]]:9
# CHECK:      {{[0-9]+}} |         return n
# CHECK-NEXT: |{{.*}}^^^^^^^^
# CHECK:      suggestion:{{.*}}make it a Meta value


@m.jit
def for_else(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):  # `break`/`continue` are early exits of a staged loop too
        acc += i
    else:
        acc += 1  # and a loop `else:` has no staged meaning
    return acc


# CHECK: ERROR: for_else UNSUP_LOOP_ELSE
report(for_else, 10)


@m.jit
def native_arm_exits(n) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):  # Meta bound: the Python arm runs; an `if` owning a
        if i == 2:  # `continue`/`break` stays native (its test must be Meta)
            continue
        acc += i
    return acc


# CHECK: RESULT: native_arm_exits 13
# EXEC:  RESULT: native_arm_exits 13
report(native_arm_exits, 6)


@m.jit
def staged_arm_exit(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(4):  # Meta bound, but the exit's `if` tests a staged value
        if acc > n:
            break
        acc += i
    return acc


# CHECK:      ERROR: staged_arm_exit UNSUP_EARLY_EXIT
report(staged_arm_exit, 2)


@m.jit
def while_exits(k, n: m.Int32) -> m.Int32:
    acc = n
    while k > 0:  # a native `while` (it owns a `break`); `k` must stay Meta
        if k == 2:
            break
        acc = acc + k
        k -= 1
    while n > 0:  # a staged `while` cannot own a `break`
        if n == 1:
            break
        n = n - 1
    return acc


# CHECK: ERROR: while_exits UNSUP_EARLY_EXIT
report(while_exits, 4, 10)


@m.jit
def nested_function_return(n: m.Int32) -> m.Int32:
    r = n
    if n > 3:

        def helper(x):
            return x + 1  # a `return` of the nested function, not an early exit

        r = helper(n)
    return r


# CHECK: RESULT: nested_function_return ?
report(nested_function_return, 10)


# --- `while` (Design 7.2, 7.5) ----------------------------------------------


@m.jit
def whiles(n: m.Int32) -> m.Int32:
    k = 3
    acc = n
    while k > 0:  # a Python bool on the first evaluation: a Python loop
        acc = acc + k
        k -= 1
    steps = m.Int32(0)
    while acc > 0:  # staged: the stored names are carried, the before block
        acc = acc - 2  # yields the condition
        steps += 1
    return steps


# CHECK-LABEL: func.func @whiles(
# CHECK-NOT:     scf.while
# CHECK-COUNT-3: arith.addi
# CHECK:         %[[R:.+]]:2 = scf.while (%[[I:.+]] = %{{.+}}, %[[S:.+]] = %{{.+}}) : (i32, i32) -> (i32, i32) {
# CHECK:           %[[C:.+]] = arith.cmpi sgt, %[[I]], %{{.+}} : i32
# CHECK:           scf.condition(%[[C]]) %[[I]], %[[S]] : i32, i32
# CHECK:         } do {
# CHECK:           scf.yield %{{.+}}, %{{.+}} : i32, i32
# CHECK:         return %[[R]]#1 : i32
# EXEC:          RESULT: whiles 8 3
print("RESULT: whiles", whiles(9), whiles(0))  # 15 -> -1 in 8, 6 -> 0 in 3
