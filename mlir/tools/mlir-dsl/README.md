# mlir-dsl

`mlir.dsl` is a Python DSL base layer for building staged, JIT-compiled
domain-specific languages on top of MLIR. A Python function decorated with
`@jit` is traced into MLIR, lowered through a pass pipeline and executed; the
core decides what is staged and how the host boundary works, plugins decide
which dialects the program becomes and how it compiles, and a sub-DSL is a
`BaseDSL` subclass that names its plugins in one record. `mlir.mlir_dsl` is
the reference sub-DSL (the upstream dialects, lowered to LLVM) and the
namespace programs are written against:

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
│   ├── core/                        BaseDSL (@jit; plugins add the other decorators), staging (Python value vs MLIR op), the host
│   │                                boundary (argument adapters), the plugin roles and the Plugins record,
│   │                                OpEmitter, diagnostics, env, remarks
│   ├── types/                       Int32/Float32/..., Pointer, @struct, Vector, max/min
│   ├── plugins/                     one folder per role, one per family; the core imports none at import time
│   │   ├── type_ops/                UpstreamDialectTypeOps(scalars=, vectors=, memory=) + the op modules arith (also the math ops), vector, llvm
│   │   ├── func_entry/func.py       Entry: the func.func host entry with the C interface
│   │   ├── ast_preprocessor/        the rewrite, its helpers; scf/: ASTPreprocessor, the scf builders and executors
│   │   ├── compiler/                Compiler: pass manager + ExecutionEngine + packed invoke; jit_executor
│   │   ├── decorators/kernels/      @kernel: launch.py (decorator, launcher, LaunchConfig); gpu/: Kernels + index ops
│   │   └── adapters/                the host boundary: numpy, pytorch, dlpack (arguments in), tvm_ffi (the entry out as another ABI)
│   └── util/                        pytrees, caches, profiler, logger
├── test/                            core and plugin tests (lit suite MLIR-DSL)
└── sub-dsls/
    └── mlir-dsl/                    the reference sub-DSL
        ├── dsl/                     installed as `mlir.mlir_dsl`: MlirTestDSL, jit, kernel, compile, the namespace
        ├── test/                    its own lit suite (MLIR-DSL-MlirTestDSL)
        └── examples/                14 runnable examples, one concept per file
```

A sub-DSL of your own is the same shape as `sub-dsls/mlir-dsl/`: a package
that subclasses `BaseDSL`, names its plugins in one `Plugins` record, and
re-exports the names its users write. It never imports `mlir.mlir_dsl`. The
core's directory is `dsl/` because the build installs every file at its path
relative to the declared root inside the `mlir` package; the sub-DSL's `dsl/`
directory is installed as `mlir.mlir_dsl` by a module target of its own, so a
sub-DSL's directory name is free.

## The core emits no dialect

Everything the core types build goes through the `OpEmitter` protocol of the
tracing DSL's `type_ops` plugin; the core itself imports no MLIR dialect, not
even lazily. The plugin answers, for the three kinds of core values:

| Value | Type hooks | Op hooks | `UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm)` (the test DSL) | a tile dialect |
|---|---|---|---|---|
| scalars (`Int32`, `Float32`, ...) | `mlir_type(dtype)`, `scalar_type(ir_type)` | `const`, `add`, `cmp`, `cast`, `minmax`, ... | `i32`, `arith`/`math` ops | a rank-0 tile, the tile ops |
| `Vector` | `vector_type(dtype, lanes)`, `vector_shape(ir_type)` | `from_elements`, `broadcast`, `extract`, `reduce` | `vector<N x T>`, `vector` ops | a rank-1 tile |
| `Pointer` | `pointer_type(dtype, space)`, `pointer_space(ir_type)` | `load` (scalar, `lanes`, `mask`), `store`, `ptr_add`, `inttoptr`, `ptrtoint`, `addrspacecast` | `!llvm.ptr`, `llvm` loads/stores/GEP, masked intrinsics | a rank-0 tile of pointers, `offset`, `load_ptr_tko` |

`@struct` needs no hook at all: it is a frozen record of DSL-typed fields and a
pytree, not an SSA aggregate. At every boundary (a `@jit` argument or result,
a loop carry, an `if_` result) it flattens to its fields and is rebuilt field
by field, so it works under every dialect.

A DSL names what it is made of in one `Plugins` record on its class, one
plugin per role and any number per family; the base names nothing, and the
core knows one decorator, `@jit`:

```python
class MlirTestDSL(BaseDSL):
    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),  # one dialect module per hook group
        func_entry=func.Entry(),                # the host entry of a @jit function and its result slot
        ast_preprocessor=scf.ASTPreprocessor(), # Python keywords -> the scf executors
        compiler=execution_engine.Compiler(),   # the pass manager and execution engine: lowering and invocation
        decorators=[gpu.Kernels(chip_option="cubin-chip")],  # adds @kernel and its launcher
        adapters=[numpy.NumpyPlugin(), pytorch.PyTorchPlugin(), dlpack.DlpackPlugin(), tvm_ffi.TvmFfiPlugin()],  # the host boundary, in and out
    )
    def pipeline(self): ...                     # the pass list; no plugin contributes a pass
    def register_dialects(self, ctx): ...       # out-of-tree dialects
```

A dialect that only adds operations (`math`, your own) is a module, not a
plugin: write the functions that build its ops (the shipped ones live beside
the plugin that emits them, `plugins/type_ops/arith.py` for one) and list its
lowering in `pipeline()`.
A dialect takes over a role of the core by being named in the record: a tile
dialect names a `type_ops` plugin answering rank-0 tiles, its own
`func_entry` and `compiler` when the shipped ones do not apply, and its own
kernels decorator plugin. `test/plugins_type_ops.py` and example 14 trace one program under two
type-ops plugins. Outside a trace there is no DSL and no `type_ops`, so
`Int32.mlir_type` is an error there; a plugin instance answers directly.

## Plugins

A plugin fills or extends a core role; a module emits ops. Every plugin says
where it connects by its base class. The roles are the fields of `Plugins`
that hold one plugin, which the core calls on its own behalf; the families are
the fields that hold any number, each extending the DSL at one fixed point:

| Role | Base class | Contract | In tree |
|---|---|---|---|
| `type_ops` | `TypeOpsPlugin` | the `OpEmitter` hooks above | the `UpstreamDialectTypeOps` composer over the op modules `arith`/`vector`/`llvm` beside it, every part optional (`plugins/type_ops/`) |
| `func_entry` | `FuncEntryPlugin` | `generate_func_op`, `generate_return`, `pack_results`, `unpack_result` | `func.Entry` (`plugins/func_entry/`) |
| `ast_preprocessor` | `ASTPreprocessorPlugin` | `preprocessor_class`, `closure_check`, `executors(dsl)` | `scf.ASTPreprocessor` (`plugins/ast_preprocessor/`) |
| `compiler` | `CompilerPlugin` | `compile`, `jit`, `compile_and_jit`, `load`, `remark_session`, `print_ir_after_passes` | `execution_engine.Compiler` (`plugins/compiler/`) |

| Family | Base class | Contract | In tree |
|---|---|---|---|
| `decorators` | `DecoratorPlugin` | `decorators()`, `before_trace`, `after_trace`, `check_arguments`, `finish_compiled_function` | `gpu.Kernels` (`plugins/decorators/kernels/`) |
| `adapters` | `AdapterPlugin` | inbound `register(dsl)`; outbound `attach_to_module`, `after_lowering`, `wrap_compiled_function` | `NumpyPlugin`, `PyTorchPlugin`, `DlpackPlugin`, `TvmFfiPlugin` (`plugins/adapters/`) |

`Plugin` itself keeps only the lifecycle every plugin shares: `available()` (a
class-level probe: is the optional dependency importable, are the dialect
bindings built), `install(dsl)`, `shared_libs()` and `register_dialects(ctx)`.
Each hook of the core lives on exactly one role or family, so a plugin's
connection is its base class, and the record names it a second time.
`BaseDSL.__init__` resolves the record once per instance: a plugin whose
`available()` is False is dropped and listed in `dsl.unavailable_plugins`, the
others are copied and installed in record order (roles, then decorators,
then adapters), and every hook loop runs over its own family. A variant
of a DSL is its record with a change: `replace(MlirTestDSL.plugins,
decorators=(), adapters=())` is a CPU-only DSL. The same plugin
instance may be named by any number of sub-DSLs (each gets a shallow copy).
Importing `mlir.dsl.plugins` pulls in no plugin and no optional dependency
(CUDA runtime, `torch`, `tvm_ffi`, the execution engine live only under
`plugins/`). The staging decision and the host boundary stay with `BaseDSL`.

Adding a decorator. The core's only decorator is `@jit`; every other one comes
from a `DecoratorPlugin`. Its `decorators(dsl_cls)` returns the decorator,
built with `dsl_cls.make_decorator(name, on_call)`, and
`BaseDSL.__init_subclass__` installs it on every class whose record names the
plugin, so `@MyDSL.kernel` exists exactly when the record says so. The core
wrapper handles the lazy instance, the AST preprocessing and the active-DSL
context and hands each call to `on_call(dsl, func, *args, **kwargs)`, the
plugin's launcher. The launcher builds its function with the shared services
`dsl.bind_arguments(...)` (the signature, canonical arguments and their IR
operands, types and attributes) and `dsl.trace_body(entry, ...)` (the function
op the entry protocol describes, with the body traced into it), and keeps its
per-trace state through `before_trace`, `after_trace` and `check_arguments`.
`plugins/decorators/kernels/launch.py` is the template: the `kernel` decorator,
the deferred `KernelLauncher`, `LaunchConfig` and the entry protocol a target
such as `gpu/` (its `__init__.py` holds `Kernels`) implements. The three shapes of a DSL are three records:

```python
class JitOnly(BaseDSL):      # @jit
    plugins = Plugins(type_ops=..., func_entry=func.Entry(), compiler=...)

class WithKernels(BaseDSL):  # @jit, @kernel
    plugins = Plugins(..., decorators=[gpu.Kernels(chip_option="cubin-chip")])

class WithMore(BaseDSL):     # @jit, @kernel, @task
    plugins = Plugins(..., decorators=[gpu.Kernels(), tasks.Tasks()])
```

How another sub-DSL would map onto this (a sketch; only the test DSL ships):
one over the upstream dialects keeps `UpstreamDialectTypeOps` and
`func.Entry` and brings its own kernels decorator plugin, preprocessor and
adapters; one over its own dialect names a `type_ops` plugin answering its
types and ops, with its own `func_entry`, `compiler` and kernels plugin.
`examples/14_type_ops_plugin.py` shows the second shape with a rank-0 tensor
stand-in and a placeholder pipeline.

## Sub-DSL knobs

A sub-DSL is a subclass of `BaseDSL` (or of `MlirTestDSL` to start from its
record). The class attributes it may set:

| Attribute | Default | Meaning |
|---|---|---|
| `plugins` | `BaseDSL`: `Plugins()`, nothing; `MlirTestDSL`: the record above | The `Plugins` record: one plugin per role and any number per family, resolved and installed once per instance. |
| `plugins.type_ops` | none (`MlirTestDSL`: `UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm)`) | The `TypeOpsPlugin` behind the types: their MLIR types and the ops of their operators, routed to the dialect modules. |
| `plugins.func_entry` | none (`MlirTestDSL`: `func.Entry()`) | The `FuncEntryPlugin` building the host entry of a `@jit` function and its result slot. |
| `plugins.ast_preprocessor` | none (`MlirTestDSL`: `scf.ASTPreprocessor()`) | The AST preprocessor and the executors that stage native control flow. `scf.ASTPreprocessor(closure_check=False)` lets nested functions capture variables inside staged regions; a subclass may replace the `DSLPreprocessor` or any executor. A DSL without one cannot preprocess and uses the explicit builders. |
| `plugins.compiler` | none (`MlirTestDSL`: `execution_engine.Compiler()`) | The `CompilerPlugin`: it runs `pipeline()`, builds the engine and loads the entry into the callable the DSL caches (`load`), so lowering and invocation are both its. A DSL without one traces only: a call returns the trace result, as under `<name>_DRYRUN`, and `compile()` raises. The execution engine is imported by this plugin at the first DSL construction, never when `mlir.dsl` is imported. |
| `plugins.decorators` | none (`MlirTestDSL`: `gpu.Kernels(chip_option="cubin-chip")` when the gpu bindings import) | The `DecoratorPlugin`s: each adds a decorator (`@kernel`) and its launcher, keeps its per-trace state and its rules at the host boundary; the gpu one also hands the CUDA runtime library to the engine and checks `<name>_ARCH`. |
| `plugins.adapters` | none (`MlirTestDSL`: `NumpyPlugin()`, `PyTorchPlugin()`, `DlpackPlugin()`, `TvmFfiPlugin()`) | The `AdapterPlugin`s: inbound, host objects (a `numpy.ndarray`, a `torch.Tensor`, anything speaking DLPack) becoming arguments at the boundary; the core adapts no host buffer itself; outbound, another ABI around the compiled entry, added to the traced module and wrapped around the compiled function. |
| `pipeline()` | `[]` (`MlirTestDSL`: the gpu lowering when an architecture is set, then its own `LOWER_TO_LLVM` list) | The pass list of the DSL, in order; no plugin publishes passes. |
| `_jit_arg_adapter_scope` | `None` | The adapter registry scope used at the host boundary; `None` is the common registry plus the single scope registering a type. |
| `name`, `dsl_package_name` (constructor) | `name` required, the other `None` | The environment-variable prefix (`<name>_DRYRUN`, ...) and log label (`MlirTestDSL`: `MLIR_DSL`); the package the rewrite imports for `and_`/`or_`/... (needed with `preprocess=True`). The pipeline option naming the target architecture is the gpu plugin's `chip_option`. |
| `register_dialects(ctx)` | nothing | Registers out-of-tree dialects on the trace context (a plugin registers its own through its hook). |

What is staged and what is Meta is decided by the annotation alone: a DSL type
is a runtime argument, anything else is a compile-time value (there is no
`Constexpr`); `is_mlir_op(x)` tells a Python value from an MLIR op inside a trace. Control
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
| `MLIR_ENABLE_PYTHON_DSL` | `OFF` | Builds `mlir.dsl` and `mlir.mlir_dsl` into the Python package and adds both lit suites; opt in with `-DMLIR_ENABLE_PYTHON_DSL=ON`. |
| `MLIR_INCLUDE_TESTS` | `ON` | Adds the lit suites `check-mlir-dsl` (`test/`, which also runs `check-mlir-dsl-mlir-dsl` for `sub-dsls/mlir-dsl/test/`); `check-mlir` depends on them. |
| `MLIR_ENABLE_EXECUTION_ENGINE` | `ON` | Needed to run compiled code; tests that execute require the lit feature `host-supports-jit` (also needs the host in `LLVM_TARGETS_TO_BUILD`). |
| `MLIR_ENABLE_CUDA_RUNNER` | `OFF` | Builds the CUDA runtime library the gpu kernels plugin hands to the engine when it finds it; `MlirTestDSL` names the plugin whenever the gpu bindings import, `MLIR_DSL_ARCH` selects the target. |

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
