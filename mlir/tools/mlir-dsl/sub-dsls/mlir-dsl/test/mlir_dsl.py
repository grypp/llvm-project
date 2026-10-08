# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s
# The reference sub-DSL: `import mlir.mlir_dsl as m` is one namespace over the
# core and the plugins `MlirTestDSL` is assembled from; `MlirTestDSL` names its
# plugins in a `Plugins` record (a subclass keeps the language when it drops
# the decorator and adapter plugins), installs a family member only
# when it is available,
# composes the gpu lowering ahead of the LLVM one, and reads `MLIR_DSL_*`.
from dataclasses import replace

import mlir.mlir_dsl as m
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.decorators.kernels import gpu as g

# The namespace: core names, the control-flow names, the DSL and its record.
print("CORE:", m.Int32, m.Pointer, m.struct, m.Vector, m.BaseDSL.__name__)
print(
    "PLUGINS:",
    m.for_.__name__,
    m.range.__name__,
    type(m.MlirTestDSL.plugins.ast_preprocessor).__name__,
    type(m.MlirTestDSL.plugins.type_ops).__name__,
    type(m.MlirTestDSL.plugins.named("func")).__name__,
    type(m.MlirTestDSL.plugins.compiler).__name__,
)
print(
    "DSL:",
    m.MlirTestDSL.__name__,
    m.jit.__name__,
    m.kernel.__name__,
    m.compile.__name__,
)
# The index helpers are in the namespace exactly when the gpu bindings are built.
print("GPU HELPERS:", hasattr(m, "thread_idx") == g.Kernels.available())
# CHECK: CORE: Int32 <class 'mlir.dsl.types.typing.Pointer'> <function struct{{.*}}> <class 'mlir.dsl.types.vector.Vector'> BaseDSL
# CHECK: PLUGINS: for_ range ASTPreprocessor UpstreamDialectTypeOps Jit Compiler
# CHECK: DSL: MlirTestDSL jit kernel compile
# CHECK: GPU HELPERS: True


class CpuOnly(m.MlirTestDSL):
    # Drops the gpu kernels plugin and the adapters, not the language: `@jit`
    # is the `func.Jit` decorator plugin, so it stays in the list.
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit()], adapters=())


dsl = CpuOnly()
print("AST:", type(dsl.plugins.ast_preprocessor).__name__, dsl.enable_preprocessor)
print(
    "DIALECTS:",
    type(dsl.plugins.type_ops).__name__,
    type(dsl.plugins.named("func")).__name__,
)
print("ROLES:", [p.name for p in dsl.plugins], dsl.plugins.named("gpu"))
print("PIPELINE:", dsl._get_pipeline(None))
print("PREFIX:", dsl.name, dsl.envar.dryrun)
# CHECK: AST: ASTPreprocessor True
# CHECK: DIALECTS: UpstreamDialectTypeOps Jit
# CHECK: ROLES: ['arith+vector+llvm', 'scf', 'execution_engine', 'func'] None
# CHECK: PIPELINE: builtin.module(convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)
# CHECK: PREFIX: MLIR_DSL True

# The record lists four family members (two decorator plugins, two adapters);
# an instance installs the available ones and remembers the others.
FAMILIES = m.Plugins.FAMILIES
listed = [type(p).__name__ for f in FAMILIES for p in getattr(m.MlirTestDSL.plugins, f)]
installed = {p.name for f in FAMILIES for p in getattr(m.MlirTestDSL().plugins, f)}
dropped = {
    k for k in m.MlirTestDSL().unavailable_plugins if k.split("[")[0] in FAMILIES
}
print("DEFAULTS:", listed)
print(
    "INSTALLED:",
    installed <= {"func", "gpu", "tvm_ffi", "dlpack"},
    len(installed) + len(dropped) == 4,
)
# CHECK: DEFAULTS: ['Jit', 'Kernels', 'DlpackPlugin', 'TvmFfiPlugin']
# CHECK: INSTALLED: True True


@m.jit
def twice(a: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(a):
        acc = acc + 2
    return acc


print("TRACED:", twice(3))
# CHECK: TRACED: ?
