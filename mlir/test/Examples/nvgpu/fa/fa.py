import dataclasses
import enum
import itertools

from mlir import ir
from mlir.dialects import gpu, scf, nvgpu, nvvm
from mlir.extras import types as T
from tools.nvdsl import *
import numpy as np
import rich.box
import rich.console
import rich.table

import __init__ as mosaic_gpu
from dsl import *


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  q: int
  kv: int
  stages: int

_utils_c = c


# TODO(apaszke): Implement a Q-scaled, base2 exp implementation.
class ExpImplementation(enum.StrEnum):
  EXACT = enum.auto()
  APPROX = enum.auto()


def build_kernel(
    batch_size: int,
    q_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    blocks: BlockSizes,
    prof_spec: profiler.ProfilerSpec | None = None,
    exp_impl: ExpImplementation = ExpImplementation.EXACT,
):
  q_shape = jax.ShapeDtypeStruct(
      (q_heads, q_seq_len, head_dim), np.float16
  )
  kv_shape = jax.ShapeDtypeStruct(
      (1, kv_seq_len, head_dim), np.float16
  )
  if batch_size != 1:
    raise NotImplementedError
  if blocks.stages < 2:
    raise ValueError("Kernel requires at least 2 stages.")
  if q_seq_len % blocks.q:
    raise ValueError
  if kv_seq_len % blocks.kv:
    raise ValueError
  if blocks.q % 64:
    raise NotImplementedError
  if blocks.kv % 64:
    raise NotImplementedError
  if head_dim % 64:
    raise NotImplementedError
  if blocks.stages * blocks.kv > kv_seq_len:
    raise NotImplementedError

  def exp(x: FragmentedArray) -> FragmentedArray:
    return x.exp(approx=exp_impl == ExpImplementation.APPROX)

  block_partition = Partition(
      elements=(batch_size, q_seq_len, q_heads),
      partition=(0, 1, 2),
      chunk_size=(1, blocks.q, 1),
  )

  index = ir.IndexType.get()
  f16 = ir.F16Type.get()
  f32 = ir.F32Type.get()

  grid = block_partition.num_chunks
  block = (128, 1, 1)
  tiling = (64, 64)
  qo_scratch = jax.ShapeDtypeStruct(
      tile_shape((blocks.q, head_dim), tiling), np.float16
  )
  k_scratch = jax.ShapeDtypeStruct(
      tile_shape((blocks.stages, head_dim, blocks.kv), tiling), np.float16
  )
  v_scratch = jax.ShapeDtypeStruct(
      tile_shape((blocks.stages, blocks.kv, head_dim), tiling), np.float16
  )
  smem_scratch_shape = [
      qo_scratch,
      k_scratch,
      v_scratch,
  ]
  in_shape = (q_shape, kv_shape, kv_shape)
  out_shape = q_shape

  def c(value, ty=index):
    return _utils_c(value, ty)

  def kernel(
      ctx: mosaic_gpu.LaunchContext,
      q_gmem,
      k_gmem,
      v_gmem,
      out_gmem,
      smem_scratch,
  ):
    barriers = BarrierArray(blocks.stages + 1)
    qo_smem, k_smem, v_smem = smem_scratch

    batch_idx, q_seq_base, head_idx = block_partition.get_base(
        gpu.block_id(gpu.Dimension.x),
        gpu.block_id(gpu.Dimension.y),
        gpu.block_id(gpu.Dimension.z),
    )
    del batch_idx

    with ctx.named_region("Q TMA start"):
      ctx.async_copy(
          src_ref=q_gmem,
          gmem_slice=(head_idx, ds(q_seq_base, blocks.q)),
          gmem_transform=mosaic_gpu.TileTransform(tiling),
          dst_ref=qo_smem,
          barrier=barriers[blocks.stages],
          swizzle=128,
      )

    def kv_copy_init(slot, kv_seq_base):
      with once():
        txcount = c(2 * blocks.kv * head_dim * bytewidth(f16))
        nvgpu.mbarrier_arrive_expect_tx(barriers.value, txcount, slot)
        k_tr = (
            mosaic_gpu.TileTransform(tiling),
            mosaic_gpu.TransposeTransform((0, 2, 1, 3, 4)),
        )
        v_tr = mosaic_gpu.TileTransform(tiling)
        for smem, gmem, t in ((k_smem, k_gmem, k_tr), (v_smem, v_gmem, v_tr)):
          ctx.async_copy(
              dst_ref=memref_slice(smem, slot),
              src_ref=gmem,
              gmem_slice=(0, ds(kv_seq_base, blocks.kv)),
              gmem_transform=t,
              barrier=barriers[slot],
              arrive=False,
              uniform=False,
              swizzle=128,
          )

    loop_partition = Partition1D(kv_seq_len, chunk_size=blocks.kv)
    with ctx.named_region("KV TMA warmup"):
      for i in range(blocks.stages - 1):
        kv_copy_init(c(i), loop_partition.get_base(c(i)))

    with ctx.named_region("Q TMA wait"):
      barriers[blocks.stages].wait()

    m_i = FragmentedArray.splat(
        c(-np.inf, f32), shape=(blocks.q,), layout=WGMMA_ROW_LAYOUT
    )
    l_i = FragmentedArray.splat(
        c(0, f32), shape=(blocks.q,), layout=WGMMA_ROW_LAYOUT
    )
    acc = FragmentedArray.splat(
        c(0, f32), shape=(blocks.q, head_dim), layout=WGMMA_LAYOUT
    )

    with ctx.named_region("KV TMA wait"):
      barriers[c(0)].wait()

    @fori(c(loop_partition.num_chunks), (acc, m_i, l_i))
    def kv_loop(kv_step, carry):
      acc, m_i, l_i = carry
      slot = arith.remui(kv_step, c(blocks.stages))

      with ctx.named_region("QK issue"):
        # TODO(apaszke): Support WGMMA without an initial accumulator.
        qk_acc = WGMMAAccumulator.zero(blocks.q, blocks.kv)
        q, k = qo_smem, memref_slice(k_smem, slot)
        qk_acc = wgmma(qk_acc, q, k, b_order=WGMMALayout.COL_MAJOR)
        nvvm.wgmma_commit_group_sync_aligned()

      # We hide the TMA overhead by overlapping it with the QK matmul.
      with ctx.named_region("KV TMA start"):
        tma_step = arith.addi(kv_step, c(blocks.stages - 1))
        tma_slot = arith.remui(tma_step, c(blocks.stages))
        tma_step_in_bounds = arith.cmpi(
            arith.CmpIPredicate.slt, tma_step, c(loop_partition.num_chunks)
        )
        if_op = scf.IfOp(tma_step_in_bounds)
        with ir.InsertionPoint(if_op.then_block):
          kv_copy_init(tma_slot, loop_partition.get_base(tma_step))
          scf.yield_([])

      with ctx.named_region("QK wait"):
        nvvm.wgmma_wait_group_sync_aligned(0)
        qk = qk_acc.value

      with ctx.named_region("Softmax"):
        m_ij = m_i.max(qk.reduce(arith.maximumf, axis=1))
        alpha = exp(m_i - m_ij)
        m_i = m_ij
        p = exp(qk - m_ij.broadcast_minor(blocks.kv))
        acc *= alpha.broadcast_minor(head_dim)
        l_i *= alpha
        l_i += p.reduce(arith.addf, axis=1)

      # For small head_dim we're not really constrained by the register budget.
      # Even though unfusing the adds should have negative performance impact,
      # it ends up emitting slightly better code for unclear reasons.
      duplicate_acc = head_dim == 64  # TODO(apaszke): Investigate why.
      with ctx.named_region("PV issue"):
        if duplicate_acc:
          acc_update = WGMMAAccumulator.zero(*acc.shape)
        else:
          acc_update = WGMMAAccumulator.from_registers(acc)
        v = memref_slice(v_smem, slot)
        acc_update = wgmma(acc_update, p.astype(f16), v)
        nvvm.wgmma_commit_group_sync_aligned()

      # We hide the barrier overhead by overlapping it with the PV matmul.
      with ctx.named_region("KV TMA wait"):
        wait_step = arith.addi(kv_step, c(1))
        wait_slot = arith.remui(wait_step, c(blocks.stages))
        wait_step_in_bounds = arith.cmpi(
            arith.CmpIPredicate.slt, wait_step, c(loop_partition.num_chunks)
        )
        with ir.InsertionPoint(scf.IfOp(wait_step_in_bounds).then_block):
          barriers[wait_slot].wait()
          scf.yield_([])

      with ctx.named_region("PV wait"):
        nvvm.wgmma_wait_group_sync_aligned(0)
        if duplicate_acc:
          acc += acc_update.value  # We can now safely extract the update.
        else:
          acc = acc_update.value

      return acc, m_i, l_i
    acc, m_i, l_i = kv_loop.results
    del m_i
    # TODO(apaszke): Invert and multiply to avoid expensive divisions.
    acc /= l_i.broadcast_minor(head_dim)

    with ctx.named_region("Acc store"):
      acc.astype(f16).store_tiled(qo_smem, swizzle=128)
      gpu.barrier()
      nvvm.fence_proxy(
          nvvm.ProxyKind.async_shared, space=nvvm.SharedSpace.shared_cta
      )  # Make sure the store is visible to the TMA.

    with ctx.named_region("GMEM store"):
      ctx.async_copy(
          src_ref=qo_smem,
          dst_ref=out_gmem,
          gmem_slice=(head_idx, ds(q_seq_base, blocks.q)),
          gmem_transform=mosaic_gpu.TileTransform(tiling),
          swizzle=128,
      )
      ctx.await_async_copy(0)

  return mosaic_gpu.as_gpu_kernel(
      kernel, grid, block, in_shape, out_shape, smem_scratch_shape, prof_spec
  )


def main(*argv):
  del argv
  batch_size = 1
  num_q_heads = 4
  prof_spec = None
  # prof_spec = profiler.ProfilerSpec((4 * 32) * 4096)
  console = rich.console.Console(width=300)
  table = rich.table.Table(title="FlashAttention", box=rich.box.SIMPLE_HEAVY)
  table.add_column("KV seq len", justify="right")
  table.add_column("Q seq len", justify="right")
  table.add_column("Q heads", justify="right")
  table.add_column("Head dim", justify="right")
  table.add_column("Exp implementation", justify="right")
  table.add_column("Time (us)", justify="right")
  table.add_column("TensorCore util (%)", justify="right")
  param_it = itertools.product(
      (4096,), (4096,), (64, 128, 256), ExpImplementation
  )
  for kv_seq_len, q_seq_len, head_dim, exp_impl in param_it:
    time = benchmark_and_verify(
        batch_size,
        q_seq_len,
        kv_seq_len,
        num_q_heads,
        head_dim,
        prof_spec=prof_spec,
        exp_impl=exp_impl,
    )
    matmul_flops = (
        4 * q_seq_len * kv_seq_len * head_dim * num_q_heads * batch_size
    )
    peak_flops = 1e15  # f16 TensorCore peak = 1000TFLOPS
    optimal_time = matmul_flops / peak_flops * 1e6  # us
    achieved_tc_util = optimal_time / time * 100
    params = map(str, (kv_seq_len, q_seq_len, num_q_heads, head_dim, exp_impl))
    results = (f"{v:.1f}" for v in (time, achieved_tc_util))
    table.add_row(*params, *results)
  console.print(table)


def benchmark_and_verify(
    batch_size,
    q_seq_len,
    kv_seq_len,
    num_q_heads,
    head_dim,
    **kwargs,
) -> float:
  with mlir.make_ir_context(), ir.Location.unknown():
    kq, kk, kv = random.split(random.PRNGKey(1234), 3)
    q = random.normal(
        kq, (batch_size, num_q_heads, q_seq_len, head_dim), dtype=np.float16
    )
    k = random.normal(
        kk, (batch_size, 1, kv_seq_len, head_dim), dtype=np.float16
    )
    v = random.normal(
        kv, (batch_size, 1, kv_seq_len, head_dim), dtype=np.float16
    )

    with profiler.measure() as time:
      f = build_kernel(
          batch_size=batch_size,
          q_heads=num_q_heads,
          q_seq_len=q_seq_len,
          kv_seq_len=kv_seq_len,
          head_dim=head_dim,
          blocks=BlockSizes(q=64, kv=64, stages=2),
          **kwargs,
      )
      out = f(q[0], k[0], v[0])[None]
      jax.block_until_ready(out)
    jax.effects_barrier()  # Make sure the profiler has finished running.

    q = q.astype(np.float32)
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    logits = np.einsum("bhqc,bxkc->bhqk", q, k)
    m = logits.max(axis=-1)
    unnormalized = np.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    expected = np.einsum("bhqk,bxkc->bhqc", weights, v)
    np.testing.assert_allclose(out, expected, atol=2e-3, rtol=2e-3)
    return time[0]

