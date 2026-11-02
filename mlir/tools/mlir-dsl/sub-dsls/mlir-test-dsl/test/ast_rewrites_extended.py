# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The rewrite's handling of constructs beyond the basic statements: a DSL
# decorator reached through an alias or under another decorator, zero-argument
# `super()` called through a subclass, Python `range` semantics for a negative
# or runtime loop step, a name a staged `while` condition binds with `:=`, a
# `raise` guarded by a `try` inside a staged region, comprehension filters,
# `and`/`or` inside a comprehension's iterable, generator arguments of the
# redirected built-ins, the class-level range rules a preprocessor subclass
# sets (`STAGED_RANGE_KEYWORDS`, `BARE_RANGE_STAGES`), and the parked error of
# a speculative loop arm not holding the rewrite's frames alive.
import ast
import gc
import weakref

import mlir.mlir_dsl as m
from mlir.dsl.plugins.ast_preprocessor import DSLPreprocessor, scf
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith, llvm, vector


def report(fn, *args):
    try:
        print("RESULT:", fn.__name__, fn(*args))
    except m.DSLUserCodeError as e:
        print("ERROR:", fn.__name__, e.diag_id.name if e.diag_id else None)
        print(str(e))


def make(name, ast_preprocessor):
    class Custom(m.BaseDSL):
        plugins = m.Plugins(
            type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
            ast_preprocessor=ast_preprocessor,
            compiler=execution_engine.Compiler(),
            decorators=[func.Jit()],
        )

        def pipeline(self):
            return list(m.LOWER_TO_LLVM)

        def __init__(self):
            super().__init__(name=name)

    Custom.__name__ = name
    return Custom


# --- The decorator: an alias, an outer decorator, `super()` ------------------
# The rewrite recognises the decorator by what the name is bound to, so an
# alias of `m.jit` marks the function as well as `@m.jit` does.
my_jit = m.jit


@my_jit
def aliased(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @aliased(
# CHECK:         scf.for
# EXEC:          ALIASED: 10
print("ALIASED:", aliased(5))


# An outer decorator is Python's business: its expression is not rewritten
# (reading `_` is rejected inside a compiled body) and it is not applied again.
_ = "tag"


def deco(tag):
    def wrap(fn):
        wrap.tag = tag
        return fn

    return wrap


@deco(_)
@m.jit
def tagged(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @tagged(
# CHECK:         scf.for
# EXEC:          TAGGED: 6
print("TAGGED:", tagged(4))


# A zero-argument `super()` names the defining class, so a call through an
# instance of a further subclass does not recurse into the same method.
class Base:
    def scale(self, n):
        return n


class Child(Base):
    @m.jit
    def scale(self, n: m.Int32) -> m.Int32:
        return super().scale(n) * 2


class GrandChild(Child):
    pass


# `self` is a Python value, so the entry's name carries it.
# CHECK-LABEL: func.func @scale_
# CHECK:         arith.muli
# EXEC:          SUPER: 6
print("SUPER:", GrandChild().scale(3))


# --- Loop steps: Python's `range` semantics ----------------------------------
# A negative Python step swaps the bounds and negates the step in Python; the
# body sees the descending value (`start + stop - iv`), so the `scf.for` keeps
# a positive step and nothing is emitted for the sign.
@m.jit
def neg_literal(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, 0, -1):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @neg_literal(
# CHECK-NOT:     arith.cmpi
# CHECK:         scf.for
# CHECK:           arith.subi
# CHECK:           arith.addi
# EXEC:          NEG_LITERAL: 10
print("NEG_LITERAL:", neg_literal(4))


# A runtime step is tested when the program runs: the bounds, the step and the
# induction variable are selected by its sign.
@m.jit
def runtime_step(n: m.Int32, s: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, 0, s):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @runtime_step(
# CHECK:         arith.cmpi slt
# CHECK:         arith.select
# CHECK:         scf.for
# CHECK:           arith.select
# EXEC:          RUNTIME_STEP: 10 6 0
print("RUNTIME_STEP:", runtime_step(4, -1), runtime_step(4, -2), runtime_step(4, 1))


@m.jit
def neg_constant(n: m.Int32) -> m.Int32:
    acc = n
    for i in m.range(10, 0, -2):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @neg_constant(
# CHECK:         scf.for
# EXEC:          NEG_CONSTANT: 30
print("NEG_CONSTANT:", neg_constant(0))


@m.jit
def zero_step(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, 0, 0):
        acc = acc + i
    return acc


# A step of 0 stays an error, as in Python.
# CHECK: error:{{.*}}The loop's `step` is `0`
report(zero_step, 4)


# --- A name the `while` condition binds with `:=` -----------------------------
# The condition runs in the region's before block, so a name it first binds is
# not visible to the body: reading it there is refused with the fix, and a
# loop whose body never reads it traces as usual. A name bound before the loop
# and rebound by the `:=` is a write_arg like any other.
@m.jit
def walrus_read(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    while (half := n // 2) > 0:
        acc = acc + half
        n = half
    return acc


# CHECK: ERROR: walrus_read UNSUP_SYNTAX
# CHECK: error[UNSUP_SYNTAX]:{{.*}}Reading `half`, which the `while` condition binds with `:=`, inside the loop body is not supported in a compiled function: the condition runs in its own block, so the body never sees the name it binds; bind the name before the loop and update it in the body.
report(walrus_read, 9)


@m.jit
def walrus_unread(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    while (half := n // 2) > 0:
        n = n - 1
        acc = acc + 1
    return acc


# CHECK-LABEL: func.func @walrus_unread(
# CHECK:         scf.while
# EXEC:          WALRUS_UNREAD: 4
print("WALRUS_UNREAD:", walrus_unread(5))


@m.jit
def walrus_carried(k: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    while (k := k - 1) > 0:
        acc = acc + k
    return acc


# CHECK-LABEL: func.func @walrus_carried(
# CHECK:         scf.while
# EXEC:          WALRUS_CARRIED: 10
print("WALRUS_CARRIED:", walrus_carried(5))


# --- Early exits guarded by `try`, comprehension filters --------------------
# A `raise` that its own `try` catches never leaves the region, so the `if`
# is staged; a `raise` the handlers do not cover is still an early exit.
@m.jit
def try_guarded(c: m.Int32) -> m.Int32:
    r = m.Int32(0)
    if c > 0:
        try:
            raise ValueError("handled at trace time")
        except ValueError:
            r = c + 1
    return r


# CHECK-LABEL: func.func @try_guarded(
# CHECK:         scf.if
# EXEC:          TRY_GUARDED: 4 0
print("TRY_GUARDED:", try_guarded(3), try_guarded(-3))


@m.jit
def try_uncovered(c: m.Int32) -> m.Int32:
    r = m.Int32(0)
    if c > 0:
        try:
            raise ValueError("not handled here")
        except TypeError:
            r = c + 1
    return r


# CHECK: ERROR: try_uncovered UNSUP_EARLY_EXIT
report(try_uncovered, 3)


# A comprehension runs in Python, so its `if` clause must be a Python value:
# a staged one is refused at the clause, a Python one filters as usual.
@m.jit
def comp_filter_staged(n: m.Int32) -> m.Int32:
    xs = [1, 2, 3]
    ys = [x for x in xs if x > n]
    return m.Int32(len(ys))


# CHECK: ERROR: comp_filter_staged PHASE_REQUIRES_CONSTANT
# CHECK: error[PHASE_REQUIRES_CONSTANT]:{{.*}}The `if` clause of a comprehension requires a value known at compile time
report(comp_filter_staged, 1)


@m.jit
def comp_filter_python(n: m.Int32) -> m.Int32:
    xs = [1, 2, 3]
    ys = [n + x for x in xs if x > 1]
    return ys[0] * ys[1]


# CHECK-LABEL: func.func @comp_filter_python(
# CHECK:         arith.muli
# EXEC:          COMP_FILTER: 12
print("COMP_FILTER:", comp_filter_python(1))


# --- `and`/`or` where an assignment expression is not allowed ---------------
# The lowering evaluates the left operand once through an immediately called
# lambda, which is legal in a comprehension's iterable (a `:=` there is a
# SyntaxError); a `for` over `a or b` keeps working.
@m.jit
def comp_or(a: m.Int32) -> m.Int32:
    xs = []
    ys = [1, 2]
    vals = [a + v for v in (xs or ys)]
    return vals[0] * vals[1]


# CHECK-LABEL: func.func @comp_or(
# CHECK:         arith.addi
# CHECK:         arith.addi
# CHECK:         arith.muli
# EXEC:          COMP_OR: 12
print("COMP_OR:", comp_or(2))


@m.jit
def for_or(a: m.Int32) -> m.Int32:
    xs = []
    ys = [1, 2, 3]
    acc = a
    for v in xs or ys:
        acc = acc + v
    return acc


# CHECK-LABEL: func.func @for_or(
# CHECK:         arith.addi
# EXEC:          FOR_OR: 7
print("FOR_OR:", for_or(1))


# --- Generator arguments of the redirected built-ins, `bool()` --------------
# A generator hides its elements from the staged check: `max`/`min` receive it
# materialised, `any`/`all` fold it element by element (Python's short-circuit
# over Python elements, the DSL's reduction from the first staged one).
@m.jit
def max_gen(a: m.Int32, b: m.Int32) -> m.Int32:
    return max(x for x in (a, b))


# CHECK-LABEL: func.func @max_gen(
# CHECK:         arith.maxsi
# EXEC:          MAX_GEN: 7
print("MAX_GEN:", max_gen(3, 7))


@m.jit
def any_all_gen(a: m.Int32, b: m.Int32) -> m.Int32:
    r = m.Int32(0)
    if any(x > 0 for x in (a, b)):
        r = r + 1
    if all(x > 0 for x in (1, a)):  # `1 > 0` is Python's; the rest is staged
        r = r + 2
    if any(x > 0 for x in (1, b)):  # decided by Python at the first element
        r = r + 4
    return r


# The DSL's `any_` joins its elements with `__dsl_or__` (a select), `all_`
# with `__dsl_and__` (an `andi`); the third `any` is Python's `True`.
# CHECK-LABEL: func.func @any_all_gen(
# CHECK:         arith.cmpi sgt
# CHECK:         arith.cmpi sgt
# CHECK:         arith.select
# CHECK:         scf.if
# CHECK:         arith.andi
# CHECK:         scf.if
# CHECK-NOT:     scf.if
# CHECK:         return
# EXEC:          ANY_ALL_GEN: 7 4 5
print("ANY_ALL_GEN:", any_all_gen(3, -3), any_all_gen(-3, -3), any_all_gen(-3, 3))


@m.jit
def bool_staged(x: m.Int32) -> m.Int32:
    if bool(x):
        return m.Int32(1)
    return m.Int32(0)


# `bool()` goes through the DSL's builtin redirector; the shipped one keeps
# the conversion of a staged value an error.
# CHECK: ERROR: bool_staged PHASE_REQUIRES_CONSTANT
report(bool_staged, 3)


@m.jit
def bool_python(n: m.Int32, k) -> m.Int32:
    if bool(k):
        return n + 1
    return n


# CHECK-LABEL: func.func @bool_python
# EXEC:          BOOL_PYTHON: 4 3
print("BOOL_PYTHON:", bool_python(3, 1), bool_python(3, 0))


# --- The range rules of a preprocessor subclass -----------------------------
# A loop option the executor understands is named in `STAGED_RANGE_KEYWORDS`:
# the rewrite then stages the loop itself (no trace-time dispatch between a
# staged and a Python loop) and forwards the option. The default preprocessor
# keeps the dispatch for the unknown keyword and the `range` built at trace
# time forwards it as one of its `options`.
class TracedRangePreprocessor(DSLPreprocessor):
    STAGED_RANGE_KEYWORDS = DSLPreprocessor.STAGED_RANGE_KEYWORDS | {"trace"}


class TracedRangeASTPreprocessor(scf.ASTPreprocessor):
    name = "traced_range"
    preprocessor_class = TracedRangePreprocessor

    def executors(self, dsl):
        executors = super().executors(dsl)
        inner = executors["loop_execute_range_dynamic"]

        def traced(*args, trace=False, **kwargs):
            if trace:
                print("[trace]")
            return inner(*args, **kwargs)

        executors["loop_execute_range_dynamic"] = traced
        return executors


TracedRange = make("MLIR_DSL", TracedRangeASTPreprocessor())


@TracedRange.jit
def traced_loop(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, trace=True):
        acc = acc + i
    return acc


def rewrite_of(dsl, fn):
    """The source the DSL's preprocessor produces for a `@jit` function."""
    original = fn.__wrapped__
    with dsl.preprocessor.get_session() as session:
        tree = session.transform(original, dict(original.__globals__))
    return ast.unparse(tree)


print("DISPATCHED:", "materialize_for_iter" in rewrite_of(TracedRange(), traced_loop))
# CHECK: DISPATCHED: False
# CHECK: [trace]
# CHECK-LABEL: func.func @traced_loop(
# CHECK:         scf.for
# EXEC:          TRACED_LOOP: 10
print("TRACED_LOOP:", traced_loop(5))


@m.jit
def default_loop(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in m.range(n, trace=True):
        acc = acc + i
    return acc


print(
    "DISPATCHED:", "materialize_for_iter" in rewrite_of(m.MlirTestDSL(), default_loop)
)
# CHECK: DISPATCHED: True
# CHECK: ERROR: default_loop CALL_ARGUMENTS
report(default_loop, 5)


# With `BARE_RANGE_STAGES` a bare `range(...)` is the staged range whatever its
# bounds are: a loop over Python bounds becomes an `scf.for` with constant
# bounds instead of a Python loop.
class BareRangePreprocessor(DSLPreprocessor):
    BARE_RANGE_STAGES = True


class BareRangeASTPreprocessor(scf.ASTPreprocessor):
    name = "bare_range"
    preprocessor_class = BareRangePreprocessor


BareRange = make("MLIR_DSL", BareRangeASTPreprocessor())


@BareRange.jit
def bare_range(n: m.Int32) -> m.Int32:
    acc = n
    for i in range(3):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @bare_range(
# CHECK:         arith.constant 3 : i32
# CHECK:         scf.for
# EXEC:          BARE_RANGE: 4
print("BARE_RANGE:", bare_range(1))


@m.jit
def python_range(n: m.Int32) -> m.Int32:
    acc = n
    for i in range(3):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @python_range(
# CHECK-NOT:     scf.for
# CHECK:         return
# EXEC:          PYTHON_RANGE: 4
print("PYTHON_RANGE:", python_range(1))


# --- A parked speculative error does not hold the rewrite's frames ----------
# The dynamic arm of a dispatched loop is rewritten speculatively; an error it
# raises is parked until trace-time dispatch selects an arm. A loop that never
# runs leaves it parked for the process, so the error is stored without its
# traceback: the frames of the rewrite (here: a local of the preprocessor's
# frame) are released once the rewrite returns.
class Sentinel:
    pass


class ProbePreprocessor(DSLPreprocessor):
    probe = None

    def transform_for_loop(self, node, active_symbols, active_callables):
        sentinel = Sentinel()
        ProbePreprocessor.probe = weakref.ref(sentinel)
        return super().transform_for_loop(node, active_symbols, active_callables)


class ProbeASTPreprocessor(scf.ASTPreprocessor):
    name = "probe"
    preprocessor_class = ProbePreprocessor


Probe = make("MLIR_DSL", ProbeASTPreprocessor())


@Probe.jit
def parked(n: m.Int32, k) -> m.Int32:
    acc = n
    if k > 100:  # a Python condition, false here: the loop below never runs
        for i in range(k):
            if i == 2:
                break  # an early exit: the speculative staged arm is an error
            acc = acc + i
    return acc


# EXEC:  PARKED: 1
print("PARKED:", parked(1, 3))
gc.collect()
print("PARKED FRAME ALIVE:", ProbePreprocessor.probe() is not None)
# CHECK: PARKED FRAME ALIVE: False
# EXEC:  PARKED FRAME ALIVE: False
