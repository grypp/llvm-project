import contextlib
import json

import jax
import jax.numpy as jnp
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import scf
import numpy as np

from google3.experimental.users.apaszke.mosaic_gpu.utils import *
from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
from google3.perftools.accelerators.xprof.api.python import xprof_session


class ProfilerSpec:
  ENTER = 0
  EXIT = 1 << 31

  def __init__(self, num_entries: int):
    self.num_entries = num_entries
    self.interned_names = {}

  @property
  def mlir_buffer_type(self) -> ir.Type:
    return ir.MemRefType.get(
        (1 + self.num_entries,), ir.IntegerType.get_signless(32)
    )

  @property
  def jax_buffer_type(self) -> ir.Type:
    return jax.ShapeDtypeStruct((1 + self.num_entries,), jnp.uint32)

  def smem_i32_elements(self, grid: tuple[int, ...]):
    return int(self.num_entries // np.prod(grid))

  def smem_bytes(self, grid: tuple[int, ...]):
    bytes_per_entry = 4
    return self.smem_i32_elements(grid) * bytes_per_entry

  def intern_name(self, name: str) -> str:
    if name_id := self.interned_names.get(name, None):
      return name_id
    name_id = self.interned_names[name] = len(self.interned_names)
    if name_id & self.EXIT:
      raise RuntimeError("Allocated too many names")
    return name_id

  def dump(self, buffer, f):
    buffer = np.asarray(buffer)
    num_blocks = buffer[0]
    per_block = self.num_entries // num_blocks
    block_entries = buffer[1 : 1 + num_blocks * per_block].reshape(
        num_blocks, per_block
    )
    start_times = block_entries[:, :2].astype(np.int64)
    start_times = (start_times[:, 0] << 32) + start_times[:, 1]
    start_times -= start_times.min()  # Normalize
    entries_used = block_entries[:, 2]
    if np.any(entries_used > per_block - 2):
      raise RuntimeError("Insufficient space to capture a full trace")
    block_traces = block_entries[:, 3:]
    unintern = {v: k for k, v in self.interned_names.items()}
    events = []
    for block_idx in range(num_blocks):
      valid_entries = entries_used[block_idx] - 3
      local_clock_offset = None
      assert valid_entries % 2 == 0
      start_time = start_times[block_idx]
      block_events = []
      for i in range(0, valid_entries, 2):
        tag = block_traces[block_idx, i]
        time = block_traces[block_idx, i + 1]
        if local_clock_offset is None:
          local_clock_offset = time
        time -= local_clock_offset
        time -= i * 6  # Account for the overhead of profiling.
        if time < 0:
          break  # Detect a timer wraparound
        name_id = tag
        begin = True
        if name_id & ProfilerSpec.EXIT:
          name_id = name_id ^ ProfilerSpec.EXIT
          begin = False
        name = unintern[name_id]
        block_events.append({
            "name": name,
            "ph": "B" if begin else "E",
            "ts": float(start_time + time) / 1e3,
            "pid": 0,
            "tid": block_idx,
        })
      else:  # If we didn't break
        events.extend(block_events)
    return json.dump({"displayTimeUnit": "ns", "traceEvents": events}, f)


class OnDeviceProfiler:

  def __init__(self, spec: ProfilerSpec, smem_buffer: ir.Value, gmem_buffer: ir.Value):
    self.spec = spec
    # self.should_store = gpu.thread_id(gpu.Dimension.x)
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    num_blocks = c(1, index)
    for dim in gpu.Dimension:
      num_blocks = arith.muli(num_blocks, gpu.grid_dim(dim))
    memref.store(arith.index_cast(i32, num_blocks), gmem_buffer, [c(0, index)])
    self.entries_per_block = arith.divui(c(spec.num_entries, index), num_blocks)
    self.smem_buffer = smem_buffer
    self.gmem_buffer = gmem_buffer
    # Hopefully mem2reg will remove the allocation.
    self.offset = memref.alloca(ir.MemRefType.get((), i32), [], [])
    memref.store(c(0, i32), self.offset, [])

  @contextlib.contextmanager
  def record(self, name: str):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    name_id = self.spec.intern_name(name)
    def store(modifier):
      cur = arith.index_cast(index, memref.load(self.offset, []))
      # TODO(apaszke): Clamp indices
      # bound = arith.subi(self.entries_per_block, c(2, index))
      # cur = arith.select(
      #     arith.cmpi(arith.CmpIPredicate.ult, cur, bound), cur, bound
      # )
      memref.store(c(modifier | name_id, i32), self.smem_buffer, [cur])
      memref.store(
          clock(), self.smem_buffer, [arith.addi(cur, c(1, cur.type))]
      )
      memref.store(
          arith.index_cast(i32, arith.addi(cur, c(2, cur.type))),
          self.offset,
          [],
      )
    store(ProfilerSpec.ENTER)
    yield
    store(ProfilerSpec.EXIT)

  def finalize(self, grid):
    index = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)

    block_idx = c(0, index)
    for dim in reversed(gpu.Dimension):
      block_idx = arith.addi(
          arith.muli(block_idx, gpu.grid_dim(dim)), gpu.block_id(dim)
      )
    start_offset = arith.addi(
        arith.muli(block_idx, self.entries_per_block), c(1, index)
    )
    block_gmem_buffer = memref.subview(
        self.gmem_buffer, [start_offset], [self.spec.num_entries], [1],
        result_type=ir.Type.parse(
            f"memref<{self.spec.num_entries}xi32, strided<[1], offset: ?>>"
        ),
    )
    # TODO(apaszke): Either use globaltimer or delete
    # memref.store(globaltimer("high"), block_gmem_buffer, [c(0, index)])
    # memref.store(globaltimer("low"), block_gmem_buffer, [c(1, index)])
    memref.store(c(0, i32), block_gmem_buffer, [c(0, index)])
    memref.store(c(0, i32), block_gmem_buffer, [c(1, index)])
    memref.store(
        arith.addi(memref.load(self.offset, []), c(3, i32)),
        block_gmem_buffer,
        [c(2, index)],
    )

    if_first = scf.IfOp(
        arith.cmpi(
            arith.CmpIPredicate.eq, gpu.thread_id(gpu.Dimension.x), c(0, index)
        )
    )
    with ir.InsertionPoint(if_first.then_block):
      for_op = scf.ForOp(
          c(0, index),
          c(self.spec.smem_i32_elements(grid) - 3, index),
          c(1, index),
      )
      with ir.InsertionPoint(for_op.body):
        x = memref.load(self.smem_buffer, [for_op.induction_variable])
        memref.store(
            x,
            block_gmem_buffer,
            [arith.addi(for_op.induction_variable, c(3, index))],
        )
        scf.yield_([])
      scf.yield_([])


@contextlib.contextmanager
def measure(kernel_name="main_kernel"):
  session = xprof_session.XprofSession()
  session.start_session()
  box = []
  try:
    yield box
  finally:
    session_id = session.end_session_and_get_session_id()
  client = xprof_analysis_client.XprofAnalysisClient()
  _, trace = client.get_profile_data("trace_viewer.json", session_id)
  jtrace = json.loads(trace)

  time = None
  for e in jtrace["traceEvents"]:
    if e["pid"] == 1 and e["name"] == kernel_name:
      if time is not None:
        raise ValueError("Ambiguous events")
      time = e["dur"]
  if time is None:
    raise ValueError("No events found")
  box.append(time)
