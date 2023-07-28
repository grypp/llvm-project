// # GEMM Sequential
// ------------------------------------------
// loop-i(128)
//   loop-j(128)
//     loop-k(64)
//       D = A * B

// # Parallel H100 GEMM (1 CTA, 128 threads)
// ------------------------------------------
// kernel(f16 lhs[128][64], f16 rhs[64][128], f32 acc[128][128]) 
//   mbarriers[1] = mbarriers.init

//   shared_lhs = shmem_alloc(f16[1][128][64])
//   shared_rhs = shmem_alloc(f16[1][64][128])
//   res = shmem_alloc(f16[128][128])

//   shared_lhs[tmaCnt][0][0] = tma_load(lhs_cta[i][tmaCnt*64])
//   shared_rhs[tmaCnt][0][0] = tma_load(rhs_cta[tmaCnt*64][0])
//   mbarrier.expect_tx mbarriers[tmaCnt] 

//   // Step 5. Wait TMA
//   mbarrier.wait mbarriers[i]

//   // Step 6. Get Pointers of the Ready data
//   tiled_lhs = shared_lhs[i%8];
//   tiled_rhs = shared_lhs[i%8];

//   // Step 7. GEMM 128x128x64
//   res = alloc_stack(f32[128][128])
//   wgmma_fence()
//   res[tidx][0:64]  = wgmma_m64n128k16(tiled_lhs[0][0],   tiled_lhs[0][0]) 
//   res[tidx][0:64]  = wgmma_m64n128k16(tiled_lhs[0][16],  tiled_lhs[16][0]) 
//   res[tidx][0:64]  = wgmma_m64n128k16(tiled_lhs[0][32],  tiled_lhs[32][0]) 
//   res[tidx][0:64]  = wgmma_m64n128k16(tiled_lhs[0][48],  tiled_lhs[48][0]) 
//   res[tidx][64:64] = wgmma_m64n128k16(tiled_lhs[64][0],  tiled_lhs[16][0]) 
//   res[tidx][64:64] = wgmma_m64n128k16(tiled_lhs[64][16], tiled_lhs[32][0]) 
//   res[tidx][64:64] = wgmma_m64n128k16(tiled_lhs[64][32], tiled_lhs[48][0]) 
//   res[tidx][64:64] = wgmma_m64n128k16(tiled_lhs[64][48], tiled_lhs[0][0]) 
//   wgmma_commit()
//   wgmma_wait(1)

//   // Step 8. End of mainloop
//   wgmma_wait(0) 

  
!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// TMA device descriptor
!lhs = memref<128x64xf16>
!rhs = memref<64x128xf16>
!acc = memref<128x128xf16>

!shmemlhs = memref<128x64xf16,3>
!shmemrhs = memref<64x128xf16,3>

!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = !shmemlhs, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = !shmemrhs, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>

!accMatrix = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32)>
  
module @mymod{
memref.global "private" @bufferLhsGlobal : !shmemlhs
memref.global "private" @bufferRhsGlobal : !shmemrhs

func.func @main() {
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %f2 = arith.constant 2.0 : f16
  %f3 = arith.constant 3.0 : f16

  // Step 1. Allocate matrices by managed memory for sake of simplicity.
  %lhs = gpu.alloc_managed() : !lhs
  %rhs = gpu.alloc_managed() : !rhs
  %acc = gpu.alloc_managed() : !acc
  
  // Step 2. Intialize the Input matrix with ones.
  scf.for %i = %c0 to %c64 step %c1 {
    scf.for %j = %c0 to %c128 step %c1 {
      memref.store %f3, %rhs[%i, %j] : !rhs
      memref.store %f2, %lhs[%j, %i] : !lhs
    }
  }

  // Step 4. Create 2 TMA Descriptors for input matrices
  %lhs_unranked = memref.cast %lhs :!lhs  to memref<*xf16>
  %rhs_unranked = memref.cast %rhs :!rhs  to memref<*xf16>
  %acc_unranked = memref.cast %acc :!acc  to memref<*xf16>

  %lhsTensorMap = nvgpu.tma.create.descriptor %lhs_unranked box[%c128, %c64] : memref<*xf16> -> !lhsTensorMap
  %rhsTensorMap = nvgpu.tma.create.descriptor %rhs_unranked box[%c128, %c64] : memref<*xf16> -> !rhsTensorMap

  // Step 5. Launch the kernel.
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1) {
      %num_threads = gpu.block_dim x
      %ic0 = arith.constant 0 : index
      %ic7 = arith.constant 7 : index
      %ic1 = arith.constant 1 : index
      %ic2 = arith.constant 2 : index
      %ic45 = arith.constant 45 : index
      %ic63 = arith.constant 63 : index
      %ic15 = arith.constant 15 : index
      %ic127 = arith.constant 127 : index
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
      
      %descA = nvgpu.wgmma.generate.descriptor %lhsShmem, %lhsTensorMap : !shmemlhs, !lhsTensorMap
      %descB = nvgpu.wgmma.generate.descriptor %rhsShmem, %rhsTensorMap : !shmemrhs, !rhsTensorMap

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
      // Step 6. Single GEMM 64x128x16 (TODO: pipeline here)
      %d2 = arith.constant 2 : i64
      %d4 = arith.constant 4 : i64
      %d6 = arith.constant 6 : i64
      %d8 = arith.constant 8 : i64
      %d512 = arith.constant 512 : i64
      %d514 = arith.constant 514 : i64
      %d516 = arith.constant 516 : i64
      %d518 = arith.constant 518 : i64

      %scaleD = arith.constant 1 : i32 // D = A*B (no accumulate itself)
      nvvm.wgmma.fence.aligned
      %wgmma_result = nvvm.wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %descA, %descB, %scaleD -> !accMatrix
      %descA1 = arith.addi %descA, %d2 : i64
      %descB1 = arith.addi %descB, %d2 : i64
      nvvm.wgmma.use.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %wgmma_result, %descA1, %descB1, %scaleD : !accMatrix
      %descA2 = arith.addi %descA, %d4 : i64
      %descB2 = arith.addi %descB, %d4 : i64
      nvvm.wgmma.use.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %wgmma_result, %descA2, %descB2, %scaleD : !accMatrix
      %descA3 = arith.addi %descA, %d6 : i64
      %descB3 = arith.addi %descB, %d6 : i64
      nvvm.wgmma.use.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %wgmma_result, %descA3, %descB3, %scaleD : !accMatrix
      %descA4 = arith.addi %descA, %d512 : i64
      %wgmma_result2 = nvvm.wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %descA4, %descB, %scaleD -> !accMatrix
      %descA5 = arith.addi %descA, %d514 : i64
      %descB5 = arith.addi %descB, %d2 : i64
      nvvm.wgmma.use.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %wgmma_result2, %descA5, %descB5, %scaleD : !accMatrix
      %descA6 = arith.addi %descA, %d516 : i64
      %descB6 = arith.addi %descB, %d4 : i64
      nvvm.wgmma.use.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %wgmma_result2, %descA6, %descB6, %scaleD : !accMatrix
      %descA7 = arith.addi %descA, %d518 : i64
      %descB7 = arith.addi %descB, %d8 : i64
      nvvm.wgmma.use.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %wgmma_result2, %descA7, %descB7, %scaleD : !accMatrix

      nvvm.wgmma.commit.group.sync.aligned
      nvvm.wgmma.wait.group.sync.aligned 1
  
      // Step 7. Sanity check WGMMA (2nd and 3rd Wapr)
      %cnd125 = arith.cmpi uge, %tidx, %ic63 : index          
      %r0 = llvm.extractvalue %wgmma_result[0] : !accMatrix
      %r1 = llvm.extractvalue %wgmma_result[1] : !accMatrix
      %r2 = llvm.extractvalue %wgmma_result[2] : !accMatrix
      %r3 = llvm.extractvalue %wgmma_result[3] : !accMatrix
      %r4 = llvm.extractvalue %wgmma_result[4] : !accMatrix
      %r5 = llvm.extractvalue %wgmma_result[5] : !accMatrix
      gpu.printf "[GPU][thread=%3d] WGMMA [0]=%4.1f [1]=%4.1f [2]=%4.1f [3]=%4.1f [4]=%4.1f [5]=%4.1f\n" %tidx, %r0, %r1, %r2, %r3, %r4, %r5 : index, f32, f32, f32, f32, f32, f32

      gpu.terminator
  }

  return 
}
}

// RESULT

// [GPU] TMA SIZE 32768
// [GPU] Before TMA Shmem lhs[0][0]         0.000000
// [GPU] Before TMA Shmem rhs[0][0]         0.000000
// [GPU] TMA Loaded shmem lhs[0][0]         2.000000
// [GPU] TMA Loaded shmem rhs[0][0]         3.000000
// WGMMA DescA : 0x4000004000010040
// WGMMA DescB : 0x4000004000010440
// [GPU][thread=  0] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  1] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  2] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  3] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  4] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  5] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  6] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  7] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  8] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=  9] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 10] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 11] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 12] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 13] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 14] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 15] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 16] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 17] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 18] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 19] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 20] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 21] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 22] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 23] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 24] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 25] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 26] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 27] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 28] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 29] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 30] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 31] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 32] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 33] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 34] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 35] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 36] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 37] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 38] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 39] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 40] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 41] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 42] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 43] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 44] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 45] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 46] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 47] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 48] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 49] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 50] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 51] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 52] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 53] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 54] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 55] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 56] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 57] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 58] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 59] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 60] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 61] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 62] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 63] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 64] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 65] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 66] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 67] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 68] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 69] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 70] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 71] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 72] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 73] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 74] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 75] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 76] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 77] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 78] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 79] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 80] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 81] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 82] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 83] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 84] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 85] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 86] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 87] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 88] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 89] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 90] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 91] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 92] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 93] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 94] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 95] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 96] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 97] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 98] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread= 99] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=100] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=101] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=102] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=103] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=104] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=105] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=106] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=107] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=108] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=109] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=110] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=111] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=112] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=113] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=114] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=115] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=116] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=117] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=118] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=119] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=120] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=121] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=122] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=123] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=124] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=125] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=126] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0
// [GPU][thread=127] WGMMA [0]=384.0 [1]=384.0 [2]=384.0 [3]=384.0 [4]=384.0 [5]=384.0