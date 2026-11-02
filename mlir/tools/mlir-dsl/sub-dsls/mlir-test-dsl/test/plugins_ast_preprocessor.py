
# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s
# Source/session/helper overrides and preprocessor switches need no staged loops.
import ast
import copy
import contextlib
import sys
import types
import mlir.mlir_dsl as m
from mlir.dsl.plugins.ast_preprocessor import DSLPreprocessor, scf
from mlir.dsl.plugins.ast_preprocessor import helpers
from mlir.dsl.plugins.ast_preprocessor.preprocessor import create_module_attribute
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith
seen=[]
extra=types.ModuleType("series_frontend_helpers")
for name in helpers.__all__: setattr(extra,name,getattr(helpers,name))
extra.enter=lambda name: seen.append(name)
sys.modules[extra.__name__]=extra
class Rewrite(DSLPreprocessor):
    parses=0
    def parse_function_source(self, fn):
        self.parses+=1
        return super().parse_function_source(fn)
    def post_visit_function_body(self, node):
        return [ast.Expr(ast.Call(create_module_attribute("enter"),[ast.Constant(node.name)],[]))]
class Frontend(scf.ASTPreprocessor):
    preprocessor_class=Rewrite
    helpers_module=extra.__name__
class Example(m.BaseDSL):
    plugins=m.Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith),ast_preprocessor=Frontend(),decorators=[func.Jit()])
    def __init__(self): super().__init__(name="FRONTEND_DSL")
alias=Example.jit
@alias
def plus_one(value:m.Int32)->m.Int32:
    return value+1
plus_one(3)
assert seen==["plus_one"]
assert Example().preprocessor.parses==1
class Base:
    def scale(self,n):return n
class Child(Base):
    @Example.jit
    def scale(self,n:m.Int32)->m.Int32:
        return super().scale(n)*2
class Grandchild(Child):pass
Grandchild().scale(3)
class Strict(DSLPreprocessor):
    DIAGNOSTICS={**DSLPreprocessor.DIAGNOSTICS,"global":(m.DiagId.UNSUP_SYNTAX,{"what":"module writes","detail":": denied here"})}
processor=Strict(["mlir","mlir_dsl"])
with processor.get_session():
    try: processor.visit(ast.Global(names=["bad"]))
    except m.DSLUserCodeError as error: assert "denied here" in str(error)
    else: raise AssertionError("custom syntax reason was ignored")
assert processor._session_data is None
