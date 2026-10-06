# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s
# The reference sub-DSL: `import mlir.mlir_dsl as m` is one namespace over the
# core and the plugins `MlirDSL` is assembled from; `MlirDSL` keeps its AST
# preprocessor and the default dialects when a subclass lists no plugins, lists
# the optional plugins only when each is available, composes their passes
# ahead of the core's, and reads `MLIR_DSL_*`.
import sys

import mlir.mlir_dsl as m
from mlir.dsl.plugins.dialects import gpu as gpu_plugin

# The namespace: core names, plugin names, the DSL.
print("CORE:", m.Int32, m.Pointer, m.struct, m.Vector, m.BaseDSL.__name__)
print(
    "PLUGINS:",
    m.for_.__name__,
    m.range.__name__,
    m.ScfASTPreprocessorPlugin.__name__,
    m.LlvmEmitter.__name__,
)
print("DSL:", m.MlirDSL.__name__, m.jit.__name__, m.kernel.__name__, m.compile.__name__)
print(
    "GPU HELPERS:",
    hasattr(m, "thread_idx") == gpu_plugin.GpuPlugin.available()
    or hasattr(m, "thread_idx"),
)
# CHECK: CORE: Int32 <class 'mlir.dsl.types.typing.Pointer'> <function struct{{.*}}> <class 'mlir.dsl.types.vector.Vector'> BaseDSL
# CHECK: PLUGINS: for_ range ScfASTPreprocessorPlugin LlvmEmitter
# CHECK: DSL: MlirDSL jit kernel compile
# CHECK: GPU HELPERS: True


class CpuOnly(m.MlirDSL):
    plugins = []  # drops the optional plugins, not the language


dsl = CpuOnly()
print("AST:", type(dsl.ast_preprocessor).__name__, dsl.enable_preprocessor)
print("DIALECTS:", [p.name for p in dsl._dialect_plugins()], type(dsl.emitter).__name__)
print("PIPELINE:", dsl._get_pipeline(None))
print("PREFIX:", dsl.name, dsl.envar.dryrun)
# CHECK: AST: ScfASTPreprocessorPlugin True
# CHECK: DIALECTS: ['scf', 'llvm'] LlvmEmitter
# CHECK: PIPELINE: builtin.module(convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)
# CHECK: PREFIX: MLIR_DSL True

# The default plugin list holds the optional plugins that are available here.
names = [type(p).__name__ for p in m.MlirDSL.plugins]
print(
    "DEFAULTS:",
    all(
        n in ("GpuPlugin", "TvmFfiPlugin", "PyTorchPlugin", "DlpackPlugin")
        for n in names
    ),
    len(names) <= 4,
)
# CHECK: DEFAULTS: True True


@m.jit
def twice(a: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(a):
        acc = acc + 2
    return acc


print("TRACED:", twice(3))
# CHECK: TRACED: ?
