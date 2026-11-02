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
trace. Functions currently return `None`; no runtime arguments, compiler or
execution engine are required. Emitting a print operation does not execute it.

Enable `MLIR_ENABLE_BINDINGS_PYTHON` and `MLIR_ENABLE_PYTHON_DSL` when building
MLIR, then run `check-mlir-dsl` to exercise the tracing tests.

The call keywords `pipeline`, `container_attrs`, `no_cache`, `extra_link_libs`
and `compile_only` are reserved for the framework's compilation interface.
The tracing interface rejects these keywords, including values of `None` or
`False`; use different names for compile-time keyword arguments.
