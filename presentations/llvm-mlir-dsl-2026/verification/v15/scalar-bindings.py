import ctypes as C
from mlir import ir, execution_engine as ee
from mlir import passmanager as pm
from mlir.dialects import arith, func

with ir.Context(), ir.Location.unknown():
    i32 = ir.IntegerType.get_signless(32)
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fn = func.FuncOp(
            "add", ([i32, i32], [i32]))
        fn.attributes["llvm.emit_c_interface"] = (
            ir.UnitAttr.get())
        entry = fn.add_entry_block()
        with ir.InsertionPoint(entry):
            a, b = fn.arguments
            result = arith.AddIOp(a, b).result
            func.ReturnOp([result])

    pipeline = (
        "builtin.module("
        "convert-arith-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts)"
    )
    pm.PassManager.parse(pipeline).run(
        module.operation)
    engine = ee.ExecutionEngine(module)

    a, b = C.c_int32(2), C.c_int32(3)
    out = C.c_int32()
    engine.invoke(
        "add", C.byref(a), C.byref(b),
        C.byref(out))
    print(out.value)  # 5
