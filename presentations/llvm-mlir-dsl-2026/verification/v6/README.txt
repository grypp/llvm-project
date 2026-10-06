v6 evidence — DkgDSL extraction tutorial, 2026-10-05

DESIGN SOURCE
The three dkg-extraction-*-audit.txt reports cite the current source under
/home/gozen/work/dkg4/DkgDSL. These are read-only source audits. DkgDSL and
Tile IR GPU execution were not attempted. Renamed m.Array/CompileCallable
facades illustrate extraction targets. No DkgDSL implementation files are
included here.

EXECUTED EXAMPLES
Interpreter: /usr/bin/python3.12
PYTHONPATH=/home/gozen/work/llvm-project/build/tools/mlir/python_packages/mlir_core
PYTHONDONTWRITEBYTECODE=1
MLIR_DSL_NO_CACHE=1 MLIR_DSL_DISABLE_FILE_CACHING=1 except the cache test.

Core smoke: BaseDSL example 12; custom complex value/operator 10.0;
explicit compile 12; literal coercion Int32 / Int64 / Float32. Execution and
PRINT_IR runs returned status 0. Exact commands and source fingerprints are
in mlir-tutorial-core-smoke.README.txt and .sources.json.

Runtime examples: affine 25; polymorphic dictionary 18; fused NumPy buffer
[6, 9, 12, 15]; compiled affine 31. Execution and PRINT_IR runs passed.
Frozen-record example: 45; rejection of mutable staged record confirmed.
Compile-before-call: JitCompiledFunction returned; execution 14.
Remarks: result 42 and structured TraceDecision record; FileCheck passed.
Cache: fresh process MISS then second process HIT; both hit the memory
cache. MISS/HIT FileChecks passed. Harness and logs are supplied.

Earlier verification/ logs remain evidence for unchanged cumsum/builders
examples. Their original 2026-09-30 date is retained.

MAINTAINER-REPORTED BASELINE
The author supplied: 14 DSL tests plus ir/remarks.py, 15/15 passing; new
C API remarks test passing; 40 files, 24.8k lines, 29 environment variables.
These are reported snapshot facts, not a claim this slide-editing turn
reran the entire suite or recomputed package statistics.

The publication clone retains an older source snapshot; run these examples
against the current original LLVM working tree identified above. The source
fingerprints and provenance files make that dependency explicit.
