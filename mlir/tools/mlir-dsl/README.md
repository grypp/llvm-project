# mlir-dsl

`mlir.dsl` provides the decorator lifecycle and tracing services for Python
DSLs that build MLIR. A `BaseDSL` subclass names the decorators it provides
in a `Plugins` record. Each decorator plugin supplies the function-entry
operation and its terminator; the body uses existing MLIR Python builders.

```python
from mlir.dialects import gpu
from mlir.dsl import BaseDSL, Plugins
from mlir.dsl.plugins.decorators.jit import func

class PrintDSL(BaseDSL):
    plugins = Plugins(decorators=[func.Jit()])

    def __init__(self):
        super().__init__(name="PRINT_DSL")

@PrintDSL.jit
def hello(repetitions):
    for _ in range(repetitions):
        gpu.printf("hello\n")

hello(2)
```

Run with `PRINT_DSL_PRINT_IR=1` to print the verified module. The call traces
`hello` into a zero-argument `func.func` containing two `gpu.printf`
operations. The Python argument specializes IR construction. Calls to other
decorated functions inside the body inline their builders into the active
trace. Additional capabilities are introduced below. Emitting a print operation does not execute it.

Enable `MLIR_ENABLE_BINDINGS_PYTHON` and `MLIR_ENABLE_PYTHON_DSL` when building
MLIR, then run `check-mlir-dsl` to exercise the tracing tests.

## Available capabilities

Typed scalar parameters become SSA block arguments. Numeric operators dispatch through the selected `type_ops` emitter, so the same Python expression can target different dialects.

The scalar catalogue includes signed and unsigned integers, standard floating-point types and supported low-precision formats. Casts and promotions validate supported combinations.

Tuples, lists, frozen dataclasses and `@struct` records flatten at argument and result boundaries. Registered leaves allow downstream value classes to participate.

`dsl_user_op` preserves source locations around user operations. Diagnostics show source context, and per-DSL environment settings control tracing and logging.

`for_`, `if_`, `while_` and `yield_` construct structured control flow explicitly. Callback executors validate loop carries and branch results.

Typed `Pointer` values support address arithmetic, address spaces, scalar loads and stores through the LLVM memory emitter.

`Vector` supports construction, arithmetic, extraction and reduction. Pointer operations can load and store vectors, including masked accesses.

An optional compiler plugin runs the DSL-owned pass pipeline and invokes the resulting code through the execution engine. `compile()` returns a reusable compiled callable.

The DLPack adapter accepts supported buffers as pointer arguments and validates their dtype, layout and ownership.

The GPU decorator plugin provides `@kernel`, launch configuration and deferred launches. IR construction works without a GPU; execution requires a suitable target and runtime.

An optional preprocessor preserves source locations, defaults and closure cells while recompiling decorated Python functions. Direct calls through dialect-module aliases accept DSL operands.

Python `for` loops with staged bounds lower to structured loops. Loops over Python values execute during tracing.

Staged `if`, `elif` and `else` statements merge branch results. Unsupported early exits and incompatible result structures are diagnosed.

Staged `while` loops carry values between their condition and body regions.

The preprocessor handles conditional expressions, Boolean expressions, comparison chains and supported builtin redirection.

An in-memory compilation cache uses specialization and module identity to reuse compiled functions. Its size and disabling behavior are configurable.

Persistent caching stores lowered bytecode with integrity checks and toolchain identity. Invalid or corrupt entries are rebuilt.

IR dumping, pass inspection and profiling expose tracing, lowering and JIT activity.

The call keywords `pipeline`, `container_attrs`, `no_cache`, `extra_link_libs`
and `compile_only` belong to the framework. They do not bind to user Meta
parameters; use different names for compile-time keyword arguments.
