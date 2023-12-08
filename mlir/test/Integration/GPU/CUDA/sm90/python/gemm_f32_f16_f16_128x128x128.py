# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

import errno
import os
import sys

import numpy as np
import pathlib
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from mlir.dialects import scf
from mlir.dialects import vector
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import nvgpucompiler
print(memref.__file__)

DYNAMIC = -9223372036854775808

def strides(xs):
  strides = []
  stride = 1
  for x in xs[::-1]:
    strides.append(stride)
    stride *= x
  return strides[::-1]


with ir.Context() as ctx, ir.Location.unknown():
  barrier_group_ty = ir.Type.parse(
      "!nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>"
  )
  acc_ty = ir.Type.parse(
      "!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>"
  )
  token_ty = ir.Type.parse("!gpu.async.token")
  lhs_tile_shape = lhs_tma_shape = (128, 64)
  lhs_tensor_map_ty = ir.Type.parse(
      "!nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>"
  )
  rhs_tile_shape = (64, 128)
  rhs_tma_shape = (64, 64)
  rhs_tensor_map_ty = ir.Type.parse(
      "!nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>"
  )

  f16 = ir.F16Type.get()
  f32 = ir.F32Type.get()
  index = ir.IndexType.get()
  i8 = ir.IntegerType.get_signless(8)
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):

    @func.FuncOp.from_py_func()
    def main():
      # matrix A [128][64] * matrix B[64][128] * stages(2)
      smem_size = 65536
      def c(value, ty=index):
        return arith.ConstantOp(ty, value)

      a_ty = b_ty = ir.MemRefType.get((128, 128), f16)
      c_ty = ir.MemRefType.get((128, 128), f32)
      a_host = memref.AllocOp(a_ty, [], [])
      b_host = memref.AllocOp(b_ty, [], [])
      c_host = memref.AllocOp(c_ty, [], [])

      token = gpu.WaitOp(token_ty, [])
      a_device, _ = gpu.AllocOp(a_ty, token_ty, [token], [], []).results
      b_device, _ = gpu.AllocOp(b_ty, token_ty, [token], [], []).results
      c_device, _ = gpu.AllocOp(c_ty, token_ty, [token], [], []).results
      tma_specs = [
          (a_device, a_host, lhs_tensor_map_ty, lhs_tma_shape),
          (b_device, b_host, rhs_tensor_map_ty, rhs_tma_shape)
      ]
      tma_descs = []
      for x_device, x_host, tensor_map_ty, tile_shape in tma_specs:
        gpu.MemcpyOp(token_ty, [token], x_device, x_host)
        x_unranked = memref.CastOp(ir.UnrankedMemRefType.get(f16, a_ty.memory_space), x_device)
        tma_descs.append(nvgpu.TmaCreateDescriptorOp(tensor_map_ty, x_unranked, map(c, tile_shape)).result)
      a_tma_desc, b_tma_desc = tma_descs

      grid = (1, 1, 1)
      block = (128, 1, 1)
      launch_op = gpu.LaunchOp(
          token_ty, [token], *map(c, grid), *map(c, block),
          dynamicSharedMemorySize=c(smem_size, ty=i32))
      launch_op.body.blocks.append(*([index] * 12))  # Append an empty block
      with ir.InsertionPoint(launch_op.body.blocks[0]):
        memref.AssumeAlignmentOp(c_device, 16)
        dynamic_smem = gpu.DynamicSharedMemoryOp(
            ir.MemRefType.get((DYNAMIC,), i8, memory_space=smem))
        a_smem = memref.ViewOp(
            ir.MemRefType.get((2, *lhs_tile_shape), f16, memory_space=smem),
            dynamic_smem, c(0), [])
        a_smem_size = int(2 * 2 * np.prod(lhs_tile_shape))  # * 2 for f16
        b_smem = memref.ViewOp(
            ir.MemRefType.get((2, *rhs_tile_shape), f16, memory_space=smem),
            dynamic_smem, c(a_smem_size), [])

        tidx = gpu.ThreadIdOp(gpu.Dimension.x)
        is_leader = arith.CmpIOp(arith.CmpIPredicate.eq, tidx, c(0))

        barrier_group = nvgpu.MBarrierCreateOp(barrier_group_ty)
        for i in range(2):
          nvgpu.MBarrierInitOp(barrier_group, c(1), c(i))
        for desc in tma_descs:
          nvgpu.TmaPrefetchOp(desc)

        a_layout = ir.Attribute.parse("strided<[64, 1], offset: ?>")
        b_layout = ir.Attribute.parse("strided<[128, 1], offset: ?>")

        def fetch(step: int | ir.Value):
          if isinstance(step, int):
            step = c(step)
          txcount = c(32768)
          if_op = scf.IfOp(is_leader)
          with ir.InsertionPoint(if_op.then_block):
            nvgpu.MBarrierArriveExpectTxOp(barrier_group, txcount, step)
            a_tma_slice = memref.SubViewOp(
                ir.MemRefType.get(lhs_tma_shape, f16, a_layout, smem),
                a_smem,
                [step], [], [],
                [DYNAMIC, 0, 0], [1, *lhs_tile_shape], [1, 1, 1])
            # NOTE: REVERSED INDICES!!!
            nvgpu.TmaAsyncLoadOp(
                a_tma_slice, barrier_group, a_tma_desc,
                coordinates=[arith.MulIOp(c(64), step), c(0)],
                mbarId=step)
            for b_start in (0, 32):
                b_tma_slice = memref.SubViewOp(
                    ir.MemRefType.get(rhs_tma_shape, f16, b_layout, smem),
                    b_smem,
                    [step], [], [],
                    [DYNAMIC, b_start, 0], [1, *rhs_tma_shape], [1, 1, 1])
                nvgpu.TmaAsyncLoadOp(
                    b_tma_slice, barrier_group, b_tma_desc,
                    coordinates=[c(b_start), arith.MulIOp(c(64), step)],
                    mbarId=step)
            scf.YieldOp([])

        fetch(0)
        fetch(1)
        acc = nvgpu.WarpgroupMmaInitAccumulatorOp(acc_ty).result

        for_op = scf.ForOp(c(0), c(2), c(1), [acc])
        with ir.InsertionPoint(for_op.body):
          i = for_op.induction_variable
          (carry_acc,) = for_op.inner_iter_args
          ticks = c(10000000)
          nvgpu.MBarrierTryWaitParityOp(barrier_group, c(0), ticks, mbarId=i)
          a_slice = memref.SubViewOp(
              ir.MemRefType.get(lhs_tile_shape, f16, a_layout, smem),
              a_smem, [i], [], [], [DYNAMIC, 0, 0], [1, 128, 64], [1, 1, 1])
          b_slice = memref.SubViewOp(
              ir.MemRefType.get(rhs_tile_shape, f16, b_layout, smem),
              b_smem, [i], [], [], [DYNAMIC, 0, 0], [1, 64, 128], [1, 1, 1])
          da = nvgpu.WarpgroupGenerateDescriptorOp(
              ir.Type.parse("!nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>"),
              a_slice, a_tma_desc)
          db = nvgpu.WarpgroupGenerateDescriptorOp(
              ir.Type.parse("!nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>"),
              b_slice, b_tma_desc)
          new_acc = nvgpu.WarpgroupMmaOp(acc.type, da, db, carry_acc, transposeB=True)
          scf.YieldOp(new_acc)
        acc = for_op.result
        # Wait until everyone is done with their WMMA
        nvvm.WgmmaWaitGroupSyncOp(0)
        # We can repurpose the tile SMEM for the epilogue now
        acc_smem = memref.ViewOp(
            ir.MemRefType.get((128, 128), f32, memory_space=smem),
            dynamic_smem, c(0), [])
        nvgpu.WarpgroupMmaStoreOp(acc, acc_smem)

        warp = arith.DivUIOp(tidx, c(32))
        within_warp = arith.RemUIOp(tidx, c(32))
        off =  arith.MulIOp(within_warp, c(4))
        for_op = scf.ForOp(warp, c(128), c(4))
        with ir.InsertionPoint(for_op.body):
          acc_part = vector.LoadOp(ir.VectorType.get((4,), f32), acc_smem, [for_op.induction_variable, off])
          vector.StoreOp(acc_part, c_device, [for_op.induction_variable, off])
          scf.YieldOp([])
        
        gpu.printf("hello ", [])
        gpu.TerminatorOp()
      gpu.MemcpyOp(token_ty, [token], c_host, c_device)
      gpu.WaitOp(token_ty, [token])

  main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  m.operation.verify()
  # print(m)  
  options = f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
  support_lib = os.getenv("SUPPORT_LIB")
  #  assert support_lib is not None, "SUPPORT_LIB is undefined"
  if not os.path.exists(support_lib):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)
  compiler = nvgpucompiler.NvgpuCompiler(options, opt_level = 3, shared_libs=[support_lib])

  # Compile.
  engine = compiler.compile_and_jit(m)

  engine.invoke("main")

# CHECK: hello
