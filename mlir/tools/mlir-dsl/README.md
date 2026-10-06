# mlir-dsl

`mlir.dsl` is a Python DSL base layer for building staged, JIT-compiled
domain-specific languages on top of MLIR. A Python function decorated with
`@jit` is traced into MLIR, lowered through a pass pipeline and executed; the
core decides what is staged and how the host boundary works, plugins decide
which dialects the program becomes, and a sub-DSL is a `BaseDSL` subclass that
lists the plugins it wants. `mlir.mlir_dsl` is the reference sub-DSL (the
upstream dialects, lowered to LLVM) and the namespace programs are written
against:

```python
import mlir.mlir_dsl as m

@m.jit
def saxpy(a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32], n: m.Int32):
    for i in range(n):
        y[i] = a * x[i] + y[i]
```

## Layout

```
mlir/tools/mlir-dsl/
├── dsl/                             CORE, installed as `mlir.dsl`: the language, no dialect of its own
│   ├── core/                        BaseDSL (@jit/@kernel, staging, host boundary), the plugin
│   │                                protocols, OpEmitter protocol, Executor slots, diagnostics, env
│   ├── types/                       Int32/Float32/..., Pointer, @struct, Vector, max/min
│   ├── runtime/  compiler/  util/   argument adapters; pipeline/ExecutionEngine/caches; pytrees, profiler
│   └── plugins/                     the plugin library; the core imports none at import time
│       ├── dialects/                IR worlds: what the types emit and how it lowers
│       │   ├── llvm/                LlvmEmitter, LlvmDialectPlugin, func.func host entry, arith/math helpers
│       │   ├── scf/                 for_/if_/while_/yield_, the scf executors, ScfDialectPlugin
│       │   └── gpu.py               gpu/nvvm kernels and launches, thread_idx/...
│       ├── ast_preprocessor/        how Python syntax maps onto a dialect: the rewrite, helpers, ScfASTPreprocessorPlugin
│       └── thirdparty/              pytorch (tensor arguments), dlpack (any DLPack object; csrc/ nanobind), tvm_ffi (ABI export)
├── test/                            core and plugin tests (lit suite MLIR-DSL)
└── sub-dsls/
    └── mlir-dsl/                    the reference sub-DSL
        ├── mlir_dsl/                installed as `mlir.mlir_dsl`: MlirDSL, jit, kernel, compile, the namespace
        ├── test/                    its own lit suite (MLIR-DSL-MlirDSL)
        └── examples/                14 runnable examples, one concept per file
```

A sub-DSL of your own is the same shape as `sub-dsls/mlir-dsl/`: a package
that subclasses `BaseDSL`, lists plugins, and re-exports the names its users
write. It never imports `mlir.mlir_dsl`. A package directory carries its
Python name (`dsl/`, `mlir_dsl/`) because the build installs every file at its
path relative to the declared root inside the `mlir` package.

## The core emits no dialect

Everything the core types build goes through the `OpEmitter` protocol of the
active DSL's dialect plugin (`DialectPlugin.emitter`); the core itself imports
no MLIR dialect. The emitter answers, for the three kinds of core values:

| Value | Type hooks | Op hooks | LLVM world (`LlvmEmitter`) | a tile dialect |
|---|---|---|---|---|
| scalars (`Int32`, `Float32`, ...) | `mlir_type(dtype)`, `scalar_type(ir_type)` | `const`, `add`, `cmp`, `cast`, `minmax`, ... | `i32`, `arith`/`math` ops | a rank-0 tile, the tile ops |
| `Vector` | `vector_type(dtype, lanes)`, `vector_shape(ir_type)` | `from_elements`, `broadcast`, `extract`, `reduce` | `vector<N x T>`, `vector` ops | a rank-1 tile |
| `Pointer` | `pointer_type(dtype, space)`, `pointer_space(ir_type)` | `load` (scalar, `lanes`, `mask`), `store`, `ptr_add`, `inttoptr`, `ptrtoint`, `addrspacecast` | `!llvm.ptr`, `llvm` loads/stores/GEP, masked intrinsics | a rank-0 tile of pointers, `offset`, `load_ptr_tko` |

`@struct` needs no hook at all: it is a frozen record of DSL-typed fields and a
pytree, not an SSA aggregate. At every boundary (a `@jit` argument or result,
a loop carry, an `if_` result) it flattens to its fields and is rebuilt field
by field, so it works under every dialect.

The remaining LLVM-specific pieces are hooks of the dialect plugin too: the
host entry of a `@jit` function (`host_gen_helper`; the LLVM world builds a
`func.func` with the C interface and packs several results into one
`!llvm.struct` read through `ctypes`), the compiler (`compiler_provider`; the
LLVM world uses the `mlir` pass manager and execution engine) and the kernel
entry (`kernel_gen_helper`; the gpu plugin builds `gpu.func` and launches).
The core pass list is only `reconcile-unrealized-casts`; every dialect
plugin lowers its own world, host entry included (the LLVM world ends with
`convert-func-to-llvm`).

`BaseDSL.default_dialects` is `(ScfDialectPlugin(), LlvmDialectPlugin())`,
resolved on first use and installed only when no listed `DialectPlugin` brings
an emitter. So `MlirDSL` and a bare `BaseDSL` emit the upstream dialects, and a
sub-DSL whose dialect plugin brings an emitter emits that dialect instead with
the same source and the same types; `test/plugins_dialect.py` and example 14
trace one program under both. With no DSL active the types fall back to
`LlvmEmitter`, so they also work standalone.

## Plugins

Plugins are the extension mechanism for everything that is not the language
itself, in three families:

| Family | Protocol | Hooks | In tree |
|---|---|---|---|
| `plugins/dialects/` | `DialectPlugin(Plugin)` | `register_dialects`, `pipeline_passes`, `emitter`, `host_gen_helper`, `kernel_gen_helper`, `compiler_provider` | `LlvmDialectPlugin`, `ScfDialectPlugin`, `GpuPlugin` |
| `plugins/ast_preprocessor/` | `ASTPreprocessorPlugin(Plugin)` | `preprocessor_class`, `closure_check`, `executors(dsl)` | `ScfASTPreprocessorPlugin` (native `for`/`if`/`while` as `scf`) |
| `plugins/thirdparty/` | `Plugin` | `available`, `install`, `shared_libs`, `attach_to_module`, `wrap_compiled_function` | `PyTorchPlugin`, `DlpackPlugin`, `TvmFfiPlugin` |

A plugin is listed on a DSL class without subclassing anything else
(`plugins = [ScfASTPreprocessorPlugin(), GpuPlugin()]`); the same plugin
instance serves any number of sub-DSLs (each gets a shallow copy at install).
Importing `mlir.dsl.plugins` pulls in no plugin and no optional dependency
(CUDA runtime, `torch`, `tvm_ffi` live only under `plugins/`). The staging
decision and the host boundary stay with `BaseDSL`.

How DkgDSL's sub-DSLs map onto this: CuTe is `BaseDSL` plus the `scf` plugins,
`GpuPlugin` (or its own `DialectPlugin` for `nvvm`/`cute` ops), its PyIR
`ASTPreprocessorPlugin`, `DlpackPlugin`, keeping the LLVM world's emitter,
pointers and host entry; cuTile is `BaseDSL` plus one `DialectPlugin` whose
emitter answers rank-0/rank-1 tiles and `cuda_tile` ops for scalars, vectors
and pointers, with its own `host_gen_helper`/`kernel_gen_helper` and a
`compiler_provider` for the tile backend.

## Sub-DSL knobs

A sub-DSL is a subclass of `BaseDSL` (or of `MlirDSL` to start from its
plugin list). The class attributes it may set:

| Attribute | Default | Meaning |
|---|---|---|
| `plugins` | `BaseDSL`: none; `MlirDSL`: gpu/tvm_ffi/pytorch/dlpack when `available()` | The `Plugin`s installed on every instance, in order; `MlirDSL` resolves its list on first use, so an environment probe counts at that time. |
| `default_dialects`, or a `DialectPlugin` with an `emitter` listed in `plugins` | `(ScfDialectPlugin(), LlvmDialectPlugin())` | The IR world of the types and of control flow, with its host entry, compiler and lowering. A listed dialect plugin with an emitter replaces all of them. |
| `default_ast_preprocessor`, or an `ASTPreprocessorPlugin` listed in `plugins` | `BaseDSL`: none; `MlirDSL`: `ScfASTPreprocessorPlugin()` | The AST preprocessor and the executors that stage native control flow. `ScfASTPreprocessorPlugin(closure_check=False)` lets nested functions capture variables inside staged regions; a plugin subclass may replace the `DSLPreprocessor` or any executor. A DSL without one cannot preprocess and uses the explicit builders. |
| `_jit_arg_adapter_scope` | `"gpu"` | The adapter registry scope used at the host boundary. |
| `name`, `dsl_package_name`, `pass_sm_arch_name` (constructor) | required | The environment-variable prefix (`<name>_DRYRUN`, ...) and log label (`MlirDSL`: `MLIR_DSL`), the package the rewrite imports for `and_`/`or_`/..., the pipeline option that names the target architecture. |
| `compiler_provider` (constructor) | the first dialect plugin's | An explicit compiler for the DSL. |

What is staged and what is Meta is decided by the annotation alone: a DSL type
is a runtime argument, anything else is a compile-time value (there is no
`Constexpr`); `is_dynamic_expr(x)` tells the two apart inside a trace. Control
flow on a Meta value is Python's. An `if`/`while` that owns an early exit
(`return`, `raise`, `break`, `continue`) stays a Python statement, and its
condition must be Meta when the trace reaches it (`UNSUP_EARLY_EXIT`
otherwise). The on-disk cache of lowered modules is on by default under
`$TMPDIR/<user>/<name lower>_cache`; `<name>_CACHE_DIR` relocates it and
`<name>_DISABLE_FILE_CACHING=1` turns it off (a file-cache load counts as
`file_cache_hits`, not as a `cache_miss`).

## Configuration

CMake options:

| Option | Default | Effect |
|---|---|---|
| `MLIR_ENABLE_BINDINGS_PYTHON` | `OFF` | Required. Both packages are part of the `mlir` Python package. |
| `MLIR_ENABLE_PYTHON_DSL` | `ON` | Builds `mlir.dsl` and `mlir.mlir_dsl`; `OFF` leaves both packages and both lit suites out of the build. |
| `MLIR_INCLUDE_TESTS` | `ON` | Adds the lit suites `check-mlir-dsl` (`test/`, which also runs `check-mlir-dsl-mlir-dsl` for `sub-dsls/mlir-dsl/test/`); `check-mlir` depends on them. |
| `MLIR_ENABLE_EXECUTION_ENGINE` | `ON` | Needed to run compiled code; tests that execute require the lit feature `host-supports-jit` (also needs the host in `LLVM_TARGETS_TO_BUILD`). |
| `MLIR_ENABLE_CUDA_RUNNER` | `OFF` | Builds the CUDA runtime library the `gpu` plugin loads; `MlirDSL` lists the plugin when that library is found or `MLIR_DSL_ARCH` is set. |

A typical configuration:

```sh
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_ENABLE_PYTHON_DSL=ON \
  -DMLIR_ENABLE_EXECUTION_ENGINE=ON -DMLIR_ENABLE_CUDA_RUNNER=ON
cmake --build build --target MLIRPythonModules
ninja -C build check-mlir-dsl
export PYTHONPATH="$PWD/build/tools/mlir/python_packages/mlir_core"
python mlir/tools/mlir-dsl/sub-dsls/mlir-dsl/examples/01_staging.py
```

Lit features: `host-supports-jit` gates the executing RUN lines; `tvm_ffi`
(added when the test interpreter can import the `tvm_ffi` package) gates the
TVM-FFI export test.

Runtime configuration is read from `MLIR_DSL_*` environment variables (see
`core/env_manager.py`); `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` prints the IR
without compiling.
