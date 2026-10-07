# mlir.mlir_dsl examples

One concept per file, in reading order, each checks its own results. Every
file starts with `import mlir.mlir_dsl as m`, keeps the concept it shows in its
module docstring and in the comments at the point where it shows, and ends
with `<Concept>: passed` (or a clear `skipped` line when a dependency is
missing).

| File | Concept | What you learn |
|---|---|---|
| `01_staging.py` | Staging: runtime vs Meta values | an annotated parameter (`a: m.Int32`) is a staged IR value, an unannotated one is Meta and folds into the trace and the symbol name; `m.is_mlir_op`; a Python `if` on a Meta value leaves no `scf.if`; one compile per Meta value |
| `02_numeric_types.py` | Numeric types | dtype annotations, literals promoting in the operand's dtype, `m.Float32(a)` and `m.cast`, `m.Boolean` comparisons, host results as `Int32(16)` with `.value` / `int()` / `float()` |
| `03_control_flow.py` | Native control flow | `for` / `if` / `while` become `scf.for` / `scf.if` / `scf.while` when the bound or condition is staged and stay Python when it is Meta: unrolled loops, `return` / `break` / `continue` only under Meta conditions |
| `04_pointers.py` | Pointers over host buffers | `m.Pointer[T]` over NumPy arrays: one `!llvm.ptr`, the length as its own argument, `p[i]` load and store, `p + i`, contiguous slices, one compiled function for every length |
| `05_structs.py` | Structs | `@m.struct` and `m.make_struct` as frozen records of DSL-typed fields: field reads, `replace`, unpacking, nesting, struct arguments and results (one block argument per field, a returned tuple packed into one host result) |
| `06_vectors.py` | Vectors | `m.Vector([...])`, `m.Vector.splat`, lane-wise arithmetic with vectors, typed scalars and literals, a vector as loop carry, `sum()` and lane extraction |
| `07_dataclass_arguments.py` | Dataclass arguments and carries | frozen dataclasses as arguments (DSL-typed fields staged, Python-typed fields Meta), `dataclasses.replace` as the loop carry, non-frozen records rejected |
| `08_explicit_builders.py` | Explicit scf builders | `@m.jit(preprocess=False)` with `m.for_`, `m.if_`, `m.while_`, `m.yield_`: hand-threaded carries, the layer the preprocessor targets |
| `09_compile_and_cache.py` | Compilation and caches | `m.compile`, the in-memory cache keyed by the traced module (`cache_hits`, `cache_misses`), the on-disk cache shared between processes (`MLIR_DSL_CACHE_DIR`, `file_cache_hits`) |
| `10_diagnostics.py` | Diagnostics | four deliberate mistakes, each a `m.DSLUserCodeError` identified by its stable `e.diag_id` (`m.DiagId.*`); the rendered headline, caret, category and suggestion |
| `11_custom_dsl.py` | Building a sub-DSL | an `MlirTestDSL` variant whose `Plugins` record drops its `decorators` and `adapters` (CPU-only), a `BaseDSL` subclass with its own `name` and environment prefix that names its plugins from scratch (`TypeOps(scalars=arith, vectors=vector, memory=llvm)`, `func.Entry`, `scf.ASTPreprocessor`, `execution_engine.Compiler`) and its own `pipeline()`, `register_jit_arg_adapter` for a host type, `@kernel` without the gpu kernels plugin is `CALL_PLUGIN_REQUIRED` |
| `12_gpu_kernels.py` | GPU kernels | `@m.kernel` is the gpu kernels plugin's decorator (a `DecoratorPlugin` listed in the record), its function becomes `gpu.func`, `.launch(grid=, block=)` from a `@m.jit` host becomes `gpu.launch_func`, CUDA tensors adapt to device pointers, `MLIR_DSL_ARCH` is fixed before the import, bounds checks in the kernel |
| `13_tvm_ffi_export.py` | TVM-FFI export | the `tvm_ffi` adapter plugin (the outbound half of the host boundary): `MLIR_DSL_ENABLE_TVM_FFI=1`, `m.compile(...).tvm_ffi_function` as a plain `tvm_ffi.Function`, staged and Meta arguments in the exported signature, a different Meta value rejected |
| `14_type_ops_plugin.py` | The type-ops plugin behind the types | the `type_ops` role: a plugin subclassing the `TypeOps` composer, `mlir_type` makes `Int32` a rank-0 `tensor<i32>`, `scalar_type` maps it back, `const` builds the dialect's constant; the same `Int32(6) + a` traced under the test DSL and under the stand-in tile dialect, the pipeline it composes, and a plugin instance answering outside any trace |

Two examples depend on something the machine may not have: `12_gpu_kernels.py`
prints `GPU kernels: skipped` without a visible CUDA device (its dry run below
works anywhere), and `13_tvm_ffi_export.py` prints `TVM-FFI export: skipped`
without the `tvm_ffi` package (`pip install apache-tvm-ffi`).

## Run

Build MLIR with the Python bindings (see `../../../README.md` for the CMake options),
then:

```sh
export PYTHONPATH="$PWD/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
python mlir/tools/mlir-dsl/sub-dsls/mlir-dsl/examples/01_staging.py
```

Everything is controlled through `MLIR_DSL_*` environment variables plus a few
call keywords (`pipeline`, `no_cache`, `extra_link_libs`, `compile_only`,
`container_attrs`):

- `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` prints the traced IR instead of
  compiling and running it. For `12_gpu_kernels.py` add `MLIR_DSL_ARCH=sm_90`
  to trace the kernels on a machine without a GPU.
- The lowered module of each compiled function is kept on disk, so a second
  process skips the pass pipeline: in `MLIR_DSL_CACHE_DIR=<dir>` when set,
  otherwise under `$TMPDIR/<user>/mlir_dsl_cache`. A function loaded from there
  counts in `file_cache_hits`, not in `cache_misses`, so 01, 04 and 07 add the
  two to count compiled functions. `MLIR_DSL_DISABLE_FILE_CACHING=1` turns the
  file cache off (09 does, so that its in-memory counters stand alone),
  `MLIR_DSL_NO_CACHE=1` every cache.
- `MLIR_DSL_ARCH=sm_90` selects the CUDA target of the gpu kernels plugin (the
  decorator plugin behind `@kernel`); `12_gpu_kernels.py` sets it from the
  visible device before the import, the DSL itself detects nothing. The
  variable is read when the DSL is first used, so set it before the import.
- `MLIR_DSL_REMARKS=".*"` renders the compiler's remarks.

The full list is in `dsl/core/env_manager.py` of the tool.
