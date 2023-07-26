!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// TMA device descriptor
!lhs = memref<4096x4096xf16>
!rhs = memref<4096x4096xf16>
!acc = memref<4096x4096xf16>

!shmemlhs = memref<128x64xf16,3>
!shmemrhs = memref<64x64xf16,3>

!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = !shmemlhs, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = !shmemrhs, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>

!accMatrix = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  
module @mymod{
memref.global "private" @bufferLhsGlobal : !shmemlhs
memref.global "private" @bufferRhsGlobal : !shmemrhs
func.func private @printMemreff16(memref<*xf16>)

func.func @main() {
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %f1 = arith.constant 1.0 : f16
  %f3 = arith.constant 3.0 : f16

  // Step 1. Allocate matrices by managed memory for sake of simplicity.
  %lhs = gpu.alloc_managed() : !lhs
  %rhs = gpu.alloc_managed() : !rhs
  %acc = gpu.alloc_managed() : !acc
  
  // Step 2. Intialize the Input matrix with ones.
  scf.for %k = %c0 to %c4096 step %c1 {
    scf.for %j = %c0 to %c4096 step %c1 {
      %s1 = arith.index_cast %k : index to i64        
      %s2 = arith.uitofp %s1 : i64 to f16
      memref.store %f3, %rhs[%k, %j] : !rhs
      memref.store %s2, %lhs[%k, %j] : !lhs
    }
  }

  // Step 4. Create 2 TMA Descriptors for input matrices
  %lhs_unranked = memref.cast %lhs :!lhs  to memref<*xf16>
  %rhs_unranked = memref.cast %rhs :!rhs  to memref<*xf16>
  %acc_unranked = memref.cast %acc :!acc  to memref<*xf16>

  %lhsTensorMap = nvgpu.tma.create.descriptor %lhs_unranked box[%c128, %c64] : memref<*xf16> -> !lhsTensorMap
  %rhsTensorMap = nvgpu.tma.create.descriptor %rhs_unranked box[%c64, %c64] : memref<*xf16> -> !rhsTensorMap

  // Step 5. Launch the kernel.
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1) {
      %num_threads = gpu.block_dim x
      %ic0 = arith.constant 0 : index
      %ic7 = arith.constant 7 : index
      %ic1 = arith.constant 1 : index
      %ic2 = arith.constant 2 : index
      %ic45 = arith.constant 45 : index
      %tidx = gpu.thread_id x

      // Step 1. Get shared memory pointers. This could be dynamic shared memory, but for now I use static
      %lhsShmem = memref.get_global @bufferLhsGlobal : !shmemlhs
      %rhsShmem = memref.get_global @bufferRhsGlobal : !shmemrhs

      // Step 2. Create a barrier type. This is i64 value in shared memory.
      %barrier = nvgpu.mbarrier.create -> !barrierType

      // Step 3. Initialize the barrier
      nvgpu.mbarrier.init %barrier, %num_threads : !barrierType
      gpu.barrier
      // Step 3.1 threadIdx.x == 0 does TMA request
      %cnd = arith.cmpi eq, %tidx, %ic0 : index    
      scf.if %cnd {
        %x1 = memref.dim %lhsShmem, %c0 : !shmemlhs
        %x2 = memref.dim %lhsShmem, %c1 : !shmemlhs
        %x3 = memref.dim %rhsShmem, %c0 : !shmemrhs
        %x4 = memref.dim %rhsShmem, %c1 : !shmemrhs
        %x5 = arith.muli %x1, %x2 : index
        %x6 = arith.muli %x3, %x4 : index
        %x7 = arith.addi %x5, %x6 : index
        %txcount = arith.muli %x7, %ic2 : index
        gpu.printf "[GPU] TMA SIZE %d\n" %txcount : index

        
        %lhs0 = memref.load %lhsShmem[%ic0, %ic0] : !shmemlhs
        %rhs0 = memref.load %rhsShmem[%ic0, %ic0] : !shmemrhs
        %lhs032 = arith.extf %lhs0: f16 to f32
        %rhs032 = arith.extf %rhs0: f16 to f32
        gpu.printf "[GPU] Before TMA Shmem lhs[0][0] \t %f\n" %lhs032 : f32
        gpu.printf "[GPU] Before TMA Shmem rhs[0][0] \t %f\n" %rhs032 : f32
        nvgpu.tma.async.load %lhsTensorMap[%ic0, %ic0], %barrier to %lhsShmem : !lhsTensorMap, !barrierType -> !shmemlhs
        nvgpu.tma.async.load %rhsTensorMap[%ic0, %ic0], %barrier to %rhsShmem : !rhsTensorMap, !barrierType -> !shmemrhs
        nvgpu.mbarrier.arrive.expect_tx %barrier, %txcount : !barrierType
        scf.yield 
      } else {
        nvgpu.mbarrier.arrive.expect_tx %barrier, %ic0 : !barrierType
        scf.yield 
      }
      
      %phase = arith.constant 0 : index
      %ticks = arith.constant 10000000 : index
      nvgpu.mbarrier.try_wait.parity %barrier, %phase, %ticks : !barrierType

      // Step 4 Sanity check of TMA
      scf.if %cnd {
        %lhs0 = memref.load %lhsShmem[%ic7, %ic7] : !shmemlhs        
        %rhs0 = memref.load %rhsShmem[%ic7, %ic1] : !shmemrhs
        %lhs032 = arith.extf %lhs0: f16 to f32
        %rhs032 = arith.extf %rhs0: f16 to f32
        gpu.printf "[GPU] TMA Loaded shmem lhs[0][0] \t %f\n" %lhs032 : f32
        gpu.printf "[GPU] TMA Loaded shmem rhs[0][0] \t %f\n" %rhs032 : f32
        gpu.printf "WGMMA DescA : 0x%llx\n" %descA : i64
        gpu.printf "WGMMA DescB : 0x%llx\n" %descB : i64
      }

      gpu.terminator
  }

  return 
}
}