# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Same CPU PyTorch elementwise add with raw MLIR bindings and mlir dsl.

Run from any directory with the MLIR Python package on PYTHONPATH:
  env PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=/home/gozen/work/llvm-project/build/tools/mlir/python_packages/mlir_core \
      MLIR_DSL_DISABLE_FILE_CACHING=1 /usr/bin/python3.12 bindings-vs-dsl.py

The raw path builds every operation with Python bindings, lowers it, creates
an ExecutionEngine, and explicitly packs tensor addresses for its C interface.
The DSL path uses the shipped PyTorch adapter and the same CPU tensor buffers.
Both are intentionally scalar loops: this example demonstrates frontend and
host-boundary mechanics, not tensor-add performance or a GPU kernel.
"""

import ctypes
import hashlib
import json
import pathlib
import sys

import torch
from mlir import ir
from mlir.dialects import arith, func, llvm, scf
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
import mlir.mlir_dsl as m

# The raw client supplies the lowering pipeline explicitly. It is equal to
# the shipped showcase DSL's current host pipeline, independently spelled out.
RAW_PIPELINE = "builtin.module(" + ",".join((
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-vector-to-llvm",
    "convert-arith-to-llvm",
    "convert-math-to-llvm",
    "convert-func-to-llvm",
    "reconcile-unrealized-casts",
)) + ")"


def check_inputs(a, b, out):
    for tensor in (a, b, out):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous()
        assert tensor.dim() == 1
    assert a.shape == b.shape == out.shape


def raw_add(a, b, out):
    """Generate, lower, JIT and call the bindings version, returning its IR."""
    check_inputs(a, b, out)
    # BEGIN RAW SLIDE EXCERPT -- imports, checks and RAW_PIPELINE are above.
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32, i32 = ir.F32Type.get(), ir.IntegerType.get_signless(32)
        ptr = llvm.PointerType.get()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("add", ([i32, ptr, ptr, ptr], []))
            fn.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            with ir.InsertionPoint(fn.add_entry_block()):
                n, x, y, z = fn.arguments
                lo, step = arith.ConstantOp(i32, 0), arith.ConstantOp(i32, 1)
                loop = scf.ForOp(lo, n, step)
                with ir.InsertionPoint(loop.body):
                    i = loop.induction_variable
                    addr = [llvm.GEPOp(ptr, p, [i], [-2147483648], f32, "None").result
                            for p in (x, y, z)]
                    lhs, rhs = [llvm.LoadOp(f32, p).result for p in addr[:2]]
                    llvm.StoreOp(arith.AddFOp(lhs, rhs).result, addr[2])
                    scf.YieldOp([])
                func.ReturnOp([])
        # END RAW IR CONSTRUCTION -- retain the human-readable generated IR.
        module.operation.verify()
        captured_ir = str(module)
        # BEGIN RAW COMPILE/CALL -- same context, immediately after construction.
        PassManager.parse(RAW_PIPELINE).run(module.operation)
        engine = ExecutionEngine(module)
        args = [ctypes.c_int32(a.numel())]
        args += [ctypes.c_void_p(t.data_ptr()) for t in (a, b, out)]
        engine.invoke("add", *[ctypes.byref(arg) for arg in args])
        # END RAW SLIDE EXCERPT
        return captured_ir


# BEGIN DSL SLIDE EXCERPT -- import mlir.mlir_dsl as m is above.
F32Ptr = m.Pointer[m.Float32]


@m.jit
def add(n: m.Int32, a: F32Ptr, b: F32Ptr, out: F32Ptr):
    for i in range(n):
        out[i] = a[i] + b[i]
# Call with CPU tensors: add(a.numel(), a, b, out)
# END DSL SLIDE EXCERPT


def source_identity(module_name):
    module = sys.modules.get(module_name)
    if module is None or not getattr(module, "__file__", None):
        return None
    path = pathlib.Path(module.__file__).resolve()
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    results = []
    first_ir = None
    for n in (0, 1, 4, 17):
        a = torch.arange(n, dtype=torch.float32)
        b = torch.full_like(a, 10.0)
        raw_out, dsl_out = torch.empty_like(a), torch.empty_like(a)
        raw_ir = raw_add(a, b, raw_out)
        if first_ir is None:
            first_ir = raw_ir
        check_inputs(a, b, dsl_out)
        add(n, a, b, dsl_out)
        expected = a + b
        torch.testing.assert_close(raw_out, expected, rtol=0, atol=0)
        torch.testing.assert_close(dsl_out, expected, rtol=0, atol=0)
        torch.testing.assert_close(raw_out, dsl_out, rtol=0, atol=0)
        results.append({"n": n, "raw": raw_out.tolist(), "dsl": dsl_out.tolist()})
    print(json.dumps({
        "python": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
        "source_identity": {
            name: source_identity(name) for name in (
                "mlir.mlir_dsl.mlir_dsl", "mlir.dsl.core.dsl", "mlir.dsl.core.plugin",
                "mlir.dsl.plugins.adapters.pytorch.plugin",
            )
        },
        "results": results,
        "status": "PASS: raw Python bindings and mlir dsl agree with PyTorch",
    }, indent=2))
    print("\nRaw bindings captured IR:\n" + first_ir)


if __name__ == "__main__":
    main()
