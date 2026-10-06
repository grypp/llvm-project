Current mlir.dsl tutorial example verification

Verified on 2026-09-30 from this checkout; no implementation or test source changed.
Interpreter: /usr/bin/python3.12
PYTHONPATH=/home/gozen/work/llvm-project/build/tools/mlir/python_packages/mlir_core
All runs used PYTHONDONTWRITEBYTECODE=1 MLIR_DSL_NO_CACHE=1 MLIR_DSL_DISABLE_FILE_CACHING=1.
IR runs additionally used MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1.
Command pattern: /usr/bin/python3.12 mlir/test/python/dsl/<example>.py
FileCheck: build/bin/FileCheck mlir/test/python/dsl/<example>.py < <example>.ir.log
EXEC FileCheck when provided by the test: append --check-prefix=EXEC, input <example>.exec.log.

example       IR + CHECK    CPU execution       EXEC FileCheck
expression    PASS          7; 6.0              n/a (no EXEC directives)
cumsum        PASS          45 Int32            PASS
builders      PASS          45; 3 2; 5           PASS
axpy_pointer  PASS          True 2.5 512.5      PASS
core_only     PASS          7                    n/a (no EXEC directives)
meta_loop     PASS          6; 45; 6             PASS
diagnostics   PASS          not needed           n/a

All 12 requested executions returned status 0. All 6 requested IR FileChecks and
all 4 available EXEC FileChecks passed. Additional existing diagnostics test and
its FileCheck passed. results.json contains machine-readable status.

Slide-safe facts confirmed by emitted IR:
- expression's Int32 annotations become i32 runtime arguments; unannotated scale=3
  specializes @expression_3 and becomes arith.constant 3, not a runtime argument.
- cumsum initializes acc with m.Int32(0); staged range(n) emits scf.for with i32
  induction variable/bounds and iter_args for the accumulator; result at n=10 is 45.
- m.for_ explicit builder (preprocess=False) produces equivalent loop/carried value.
- unannotated bound: unrolled(4) emits constant 6 and no scf.for.
  annotated bound: staged(10) emits scf.for and returns 45.
  m.range forces scf.for even for unannotated Meta bound forced(4).
- axpy NumPy Float32 arrays adapt to !llvm.ptr; pointer subscription emits
  llvm.getelementptr, llvm.load and llvm.store with 4-byte alignment.
- core-only test observes no GPU/NVVM bindings imported and this pipeline:
  builtin.module(convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,
                 convert-arith-to-llvm,convert-math-to-llvm,reconcile-unrealized-casts)
  This is a deliberately constructed BaseDSL subclass, not a claim that importing
  the default mlir.dsl surface avoids all GPU imports.

Mutation handoff grounded in existing code (no PyIR behavior asserted):
- Scalar rebinding is already supported for explicitly staged values: acc =
  m.Int32(0); acc += i becomes an SSA iter_arg/result, not mutable Python state.
- Memory stores are already supported: out[i] = ... emits llvm.store; a pointer
  that is only a store base is treated as a memory side effect and need not become
  a loop carry. See mlir_dsl_ast_decorators.py:206-213.
- acc = 0 followed by acc += i in a staged loop is rejected with
  PHASE_MUTATE_PYTHON. diagnostics.ir.log includes the real diagnostic and remedy:
  initialize acc as Int32(0) before the loop.
- Fixed container structure and stable leaf types are checked at staged joins.
  This is source evidence, not additionally executed: mlir_dsl_ast_decorators.py
  ScfGenerator._check_leaf and _check_region_result (lines 298-354).
- Good bridge question for Amir: How should a Python DSL represent richer Python
  object mutation and aliasing across staged control flow while preserving SSA?
  Present the question only; the actual PyIR design and examples belong to Amir.
