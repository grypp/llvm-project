import mlir.dsl as m
from mlir import execution_engine, passmanager, ir
from mlir.dialects import complex as cx


class MySubDSL(m.BaseDSL):
    plugins = []
    _jit_arg_adapter_scope = "mlir"
    _unannotated_arg_is_constexpr = True

    def __init__(self):
        super().__init__(
            name="MY_DSL",
            dsl_package_name=["mydsl"],
            compiler_provider=m.Compiler(passmanager, execution_engine),
            pass_sm_arch_name="cubin-chip",
            preprocess=False,
        )


@MySubDSL.jit
def scale(x: m.Int32, factor):
    return x * factor


class Complex64:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Complex64(cx.AddOp(self.value, other.value).result)


m.register_leaf(
    Complex64,
    ir_types=lambda p: [ir.ComplexType.get(ir.F32Type.get())],
    ir_values=lambda z: [z.value],
    from_ir_values=lambda p, values: Complex64(values[0]),
)


class ComplexPlugin(m.DialectPlugin):
    name = "complex"

    def pipeline_passes(self):
        return ["convert-complex-to-llvm"]


class ComplexDSL(m.MlirDSL):
    plugins = [ComplexPlugin()]


@ComplexDSL.jit
def magnitude(x: m.Float32, y: m.Float32):
    ty = ir.ComplexType.get(ir.F32Type.get())
    z = Complex64(cx.CreateOp(ty, x.ir_value(), y.ir_value()).result)
    twice = z + z
    return m.Float32(cx.AbsOp(twice.value).result)


print("SUBDSL", scale(4, 3))
print("CUSTOM_OP", magnitude(3.0, 4.0))
compiled = m.compile(scale, 0, 3)
print("COMPILE", compiled(4, 3))
print("LITERAL", type(m.as_numeric(5)).__name__, type(m.as_numeric(2**40)).__name__, type(m.as_numeric(1.5)).__name__)
