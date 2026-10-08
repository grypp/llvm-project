# Does `mlir.dsl` fit MLIR's design?

An objective analysis of the DSL base layer under `mlir/tools/mlir-dsl`
against the upstream MLIR tree, written 2026-10-07. Three independent reviews
(upstream precedent, boundary audit of the code, upstream-review and packaging)
were followed by an adversarial pass that checked every claim against both
trees; seven claims were refuted and dropped, the rest is below. Line numbers
refer to the tree at the time of writing.

## The three premises

| Premise | What the upstream tree shows |
|---|---|
| MLIR is not end-to-end | True for the installed library: no driver, frontend, cache or configuration layer (`InitAllPasses.h:21-25`; `docs/Tutorials/MlirOpt.md:18-21` calls `mlir-opt` a testing and debugging utility; `JitRunner.h:48-53` lowers nothing unless the embedding tool supplies a transformer). Not true for the repository: Toy chapter 6 (`examples/toy/Ch6/toyc.cpp`), the GPU integration tests through `gpu-lower-to-nvvm-pipeline` and `mlir-runner`, and the Python decorator DSL `test/Examples/NVGPU/tools/nvdsl.py` are complete trace, compile and run flows; `nvdsl` has exactly our shape (trace into `func.func` with `llvm.emit_c_interface`, numpy at the boundary, `engine.invoke`). |
| MLIR has no default pipeline | No tool applies a pipeline by default (`MlirOptMain.cpp:297`), and the CPU lower-to-LLVM pipeline exists only in the test library (`test/lib/Dialect/LLVM/TestLowerToLLVM.cpp`). But six named pipelines are registered in `lib/RegisterAllPasses.cpp` and reach the Python bindings: `gpu-lower-to-nvvm-pipeline` (its own description: "The default pipeline lowers main dialects ... to NVVM"), `gpu-lower-to-rocdl-pipeline`, `gpu-lower-to-xevm-pipeline`, `sparsifier` ("The standard pipeline"), `buffer-deallocation-pipeline`, `tosa-to-linalg-pipeline`. We reuse the first one. |
| MLIR has plugins | True, but they are a different mechanism: shared-library pass and dialect plugins with a versioned C ABI (`Tools/Plugins/PassPlugin.h`, `DialectPlugin.h`, loaded by `mlir-opt --load-pass-plugin`), `DialectRegistry` extension points, and on the Python side the `_site_initialize_*` modules and the "composable modules downstream integrators include and re-export" design (`docs/Bindings/Python.md:118-131`). Ours are lifecycle hooks on a DSL instance. Our `Plugins` record shares a word with `mlir::DialectPlugin`/`PassPlugin` while being a different kind of thing. |

A fact all three reviews missed: upstream added `mlir/python/mlir/dialects/ext.py`
in January 2026, an installed, documented, tested Python DSL for defining
dialects through IRDL; its `Dialect.load()` even hard-codes `canonicalize, cse`.
A Python-level DSL layer is therefore not outside MLIR's design. That precedent
is about a thousand lines, dialect-definition scope, relocatable under
`MLIR_PYTHON_PACKAGE_PREFIX`, and free of runtime, cache and configuration
concerns.

## What fits

- The core imports no dialect (verified by grep over `dsl/core`, `dsl/types`,
  `dsl/util`, and by `test/core_only.py`); dialect modules live in the plugin
  folders and are imported when a DSL names the plugin.
- The emitter behind the types, the host entry behind `@jit` and the compiler
  are hooks with a working default and an exercised override
  (`sub-dsls/mlir-test-dsl/test/plugins_type_ops.py`, example 14 trace one program under two type-ops plugins).
- `Compiler` is a thin, idiomatic use of the bindings and almost method for
  method `test/Examples/NVGPU/tools/nvgpucompiler.py`; the host entry is
  `nvdsl`'s `func.func` with the C interface and a packed invoke.
- The pass pipeline is one explicit, sub-DSL-owned list (`BaseDSL.pipeline`,
  `MlirTestDSL.pipeline` prepending upstream's own gpu pipeline when an
  architecture is set), the shape upstream uses for its named pipelines; no
  dialect or plugin contributes a pass.
- The AST preprocessor has no upstream analogue but is a plugin with an
  explicit-builder bypass, and the `Executor` cleanly separates the
  virtualization of Python keywords from the dialect.
- numpy, the DLPack protocol and the nanobind extension are within the
  dependency surface the bindings already declare (`python/requirements.txt`).

## What deviates

Isolated, deliberate choices the design already contains:

- the defaults themselves (LLVM world, upstream dialects) behind overridable
  slots;
- the preprocessor as an optional plugin;
- torch, DLPack and TVM-FFI as probed, optional add-ons;
- environment and cache policy under one `<PREFIX>_` prefix.

Structural, meaning the current seams do not contain them:

1. **Invocation is hardwired.** `compile_and_cache` always looks up the
   execution engine's packed `_mlir_<name>(void**)` symbol and the JIT executor
   always fills a ctypes result slot, so `compiler_provider` is a hook for
   lowering only; the README's statement that the compiler is a hook is half
   true. A compiler that does not produce an `ExecutionEngine` cannot be
   plugged in today.
2. **A CUDA-shaped launch protocol lived in the core**: `LaunchConfig`, the
   launcher, `@kernel`, the `gpu_module_attrs` call keyword, and the mandatory
   constructor arguments `pass_sm_arch_name` and `dsl_package_name`, which a
   DSL without kernels still had to supply.
3. **The default world imported `mlir.execution_engine` at module import**
   (then `dsl/plugins/dialects/llvm/plugin.py`), an upstream-optional
   component, while `MLIR_ENABLE_PYTHON_DSL` defaulted to ON (both fixed
   below: the compiler plugin probes it lazily, the option defaults to OFF).
4. **Absolute `mlir.` names** in the sub-DSL, the DLPack plugin, the
   diagnostics frame registry and the gpu environment prefix break the
   documented package-prefix relocation that every hand-written upstream Python
   file honours.
5. **Placement and process**: a pure-Python package under `mlir/tools` pulled
   in cross-tree from `mlir/python/CMakeLists.txt`, a second module target that
   exists only for a directory name, `check-mlir` coupled to our suites,
   default-ON inclusion in every bindings build, a dependency on remark
   bindings that are not on `origin/main`, and no RFC, maintainer entry or
   status document.

A product layer with no counterpart in `python/mlir`: about thirty environment
variables (two of them plugin-specific), a
default-on bytecode cache under the temp directory that also skips
`after_lowering` on a hit, unconditional ANSI colour in diagnostics, 42 error
codes of which two belong to the TVM-FFI plugin, singleton DSL instances, and
a fresh per-call `ir.Context` with multithreading disabled.

Two README statements are wrong against the code: the examples README says
everything is controlled by environment variables and never by decorator
arguments, while `pipeline`, `gpu_module_attrs`, `no_cache`, `extra_link_libs`
and `compile_only` are call keywords; and the README overstates the compiler
hook (item 1).

## Verdict

The base layer, meaning the core, the LLVM world, the compiler and the
explicit `scf` builders, is in the spirit of the Toy, `nvdsl`, `standalone` and
`ext` precedents and could be argued upstream. `mlir.mlir_dsl` as an installed,
default-ON language, the default-on file cache, environment-first
configuration, the TVM-FFI reimplementation and the three-thousand-line AST
rewriter exceed what MLIR has ever installed; they need either opt-in
placement (`test/Examples` or an out-of-tree wheel) or an RFC that argues the
scope explicitly. Items 4 and 5 above, not the compiler layers, are what an
upstream reviewer would reject first.

## What to do, from our side

The analysis above describes the tree on the morning of 2026-10-07; this list
tracks what changed since.

Decided or already done (2026-10-07):

- [x] Passes out of dialect plugins; `BaseDSL.pipeline()` is the sub-DSL's
  pass list, `MlirTestDSL.pipeline()` prepends the gpu pipeline when an
  architecture is set; the `scf` dialect plugin is gone.
- [x] Replace role discovery through the plugin list by typed plugins in one
  `Plugins` record on the DSL class: the roles (`type_ops`,
  `ast_preprocessor`, `compiler`; `func_entry` was a role until `@jit` became
  the `func.Jit` decorator plugin), one plugin each, and the families
  (`decorators`, `adapters`), any number each; `Plugin` keeps the
  shared lifecycle only, each role or family adds its contract, and each hook
  of the core lives on exactly one of them. A dialect the DSL only emits ops
  from is a module in the folder of the plugin that emits it, not a plugin. `pipeline()` and
  `register_dialects()` stay methods of the DSL.
- [x] Decorators are plugins (`DecoratorPlugin`): the core knows no decorator,
  only the default name `jit`; `@jit` is the `func.Jit` plugin's.
  A plugin's `decorators()` is installed on the DSL class by
  `__init_subclass__`; the shared services `bind_arguments` and `trace_body`
  and the hooks `before_trace`, `after_trace`, `check_arguments`,
  `finish_compiled_function` are what a decorator needs from the core.
- [x] Typed plugin families replace the untyped extras: `dlpack` and
  `tvm_ffi` are adapters (the host boundary, inbound `register` and outbound
  `attach_to_module`/`wrap_compiled_function`), `gpu.Kernels` is a decorator
  plugin.
- [x] No default world in the core: `BaseDSL.plugins = Plugins()`,
  `BaseDSL.pipeline()` is empty and `current_emitter()` needs the tracing
  DSL's `type_ops` plugin. The core knows no dialect, not even lazily; the
  LLVM lowering is the sub-DSL's own `LOWER_TO_LLVM` list, published by no plugin
  and listed by the sub-DSL.

Code changes behind new hooks (structural items 1 to 3):

- [x] Give the compiler an invocation surface (`CompilerPlugin.load`: entry
  lookup and the callable) so a compiler plugin that does not produce an
  `ExecutionEngine` replaces lowering and invocation together; the core no
  longer imports the JIT executor.
- [x] The kernel protocol left the core entirely: `@kernel`, `LaunchConfig`,
  `KernelLauncher`, the launch driver and the kernel bookkeeping are the gpu
  kernels decorator plugin's (`plugins/decorators/kernels/`); `core/kernels.py`
  is gone, the `device_compilation_only` constructor flag with it, and the
  compiler's `load` no longer receives kernel facts. The architecture option
  is the gpu plugin's `chip_option`, merged into a `<PREFIX>_PIPELINE` override
  through `Plugin.pipeline_options()` (no DSL attribute); `dsl_package_name` is
  optional; the call keyword is `container_attrs`. The launch shape (grid, block, cluster, shared
  memory) is the gpu plugin's own, as another accelerator's would be.
- [x] `mlir.execution_engine` is imported by the compiler plugin only, at the
  first DSL construction (`execution_engine.Compiler.available()` probes it).

Product layer:

- [ ] File cache off by default, or an opt-in plugin; never skip
  `after_lowering` silently on a hit.
- [ ] Constructor and decorator arguments as the primary configuration
  surface, environment variables as the override layer; let plugins declare
  their own settings.
- [x] The gpu plugin reads the DSL's prefix (`dsl.envar.prefix`) instead of
  hard-coding `MLIR_DSL`.
- [x] Move the two TVM-FFI codes into a `TvmFfiDiagId` catalog in the plugin;
  plugin catalogs classify their own codes (`diagnostics.classify`), the core
  names none of them.
- [ ] Gate colour on `isatty` and `NO_COLOR`.
- [x] The core registers no host buffer type; the one inbound adapter is
  `dlpack` (numpy arrays and torch tensors speak DLPack), the separate numpy
  and pytorch adapters are gone. `ARCH` is documented
  as a target setting the plugins interpret.
- [x] An IR-only DSL (no compiler plugin) has its own test, `test/core_ir_only.py`.
- [ ] Allow explicit DSL instances next to the per-class default.

Packaging and process (structural items 4 and 5):

- [ ] Relative imports everywhere; derive the extension module name and the
  DSL package names from `__package__`.
- [x] `MLIR_ENABLE_PYTHON_DSL` defaults to OFF.
- [ ] Move the tests under
  `mlir/test/python/dsl` with one `lit.local.cfg`; stop coupling `check-mlir`
  to the DSL suites.
- [ ] Decide the home: `mlir/python/mlir/dsl` for the base layer, and either
  `test/Examples` status or an out-of-tree wheel for `mlir.mlir_dsl`, the
  pytorch and TVM-FFI plugins and the examples.
- [ ] Land the remark-engine bindings as their own upstream change first.
- [x] Fix the two README statements.
- [ ] Add a status document and a maintainer entry before any RFC.
