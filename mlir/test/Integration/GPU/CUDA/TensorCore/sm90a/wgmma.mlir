!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// TMA device descriptor
!lhs = memref<64x8xf32>
!rhs = memref<8x128xf32>
!acc = memref<64x128xf32>

!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<64x8xf32,3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<8x128xf32,3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>

!shmemlhs = memref<64x8xf32,3>
!shmemrhs = memref<8x128xf32,3>
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
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %f1 = arith.constant 1.0 : f32
  %f3 = arith.constant 3.0 : f32

  // Step 1. Allocate matrices by managed memory for sake of simplicity.
  %lhs = gpu.alloc_managed() : !lhs
  %rhs = gpu.alloc_managed() : !rhs
  %acc = gpu.alloc_managed() : !acc
  
  // Step 2. Intialize the Input matrix with ones.
  scf.for %k = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c128 step %c1 {
      %s1 = arith.index_cast %k : index to i64        
      %s2 = arith.uitofp %s1 : i64 to f32
      memref.store %f3, %rhs[%k, %j] : !rhs
    }
  }
  scf.for %i = %c0 to %c64 step %c1 {
    scf.for %k = %c0 to %c8 step %c1 {             
      %s1 = arith.index_cast %k : index to i64   
      %s2 = arith.uitofp %s1 : i64 to f32
      memref.store %s2, %lhs[%i, %k] : !lhs
    }
  }

  // Step 4. Create 2 TMA Descriptors for input matrices
  %lhs_unranked = memref.cast %lhs :!lhs  to memref<*xf32>
  %rhs_unranked = memref.cast %rhs :!rhs  to memref<*xf32>
  %acc_unranked = memref.cast %acc :!acc  to memref<*xf32>

  func.call @printMemrefF32(%lhs_unranked) : (memref<*xf32>) -> ()  
  func.call @printMemrefF32(%rhs_unranked) : (memref<*xf32>) -> ()  

  %lhsTensorMap = nvgpu.tma.create.descriptor %lhs_unranked box[%c64, %c8] : memref<*xf32> -> !lhsTensorMap
  %rhsTensorMap = nvgpu.tma.create.descriptor %rhs_unranked box[%c8, %c128] : memref<*xf32> -> !rhsTensorMap

  // Step 5. Launch the kernel.
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1) {
      %num_threads = gpu.block_dim x
      %ic0 = arith.constant 0 : index
      %ic7 = arith.constant 7 : index
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
        %totalSizeInBytes = arith.constant 6144 : index
        nvgpu.mbarrier.arrive.expect_tx %barrier, %totalSizeInBytes : !barrierType
        %lhs0 = memref.load %lhsShmem[%ic0, %ic0] : !shmemlhs
        %rhs0 = memref.load %rhsShmem[%ic0, %ic0] : !shmemrhs
        gpu.printf "[GPU] Empty Shmem lhs[0][0] \t %f\n" %lhs0 : f32
        gpu.printf "[GPU] Empty Shmem rhs[0][0] \t %f\n" %rhs0 : f32
        nvgpu.tma.async.load %lhsTensorMap[%ic0, %ic0], %barrier to %lhsShmem : !lhsTensorMap, !barrierType -> !shmemlhs
        nvgpu.tma.async.load %rhsTensorMap[%ic0, %ic0], %barrier to %rhsShmem : !rhsTensorMap, !barrierType -> !shmemrhs
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
        %lhs0 = memref.load %lhsShmem[%ic45, %ic7] : !shmemlhs        
        %rhs0 = memref.load %rhsShmem[%ic7, %ic0] : !shmemrhs
        gpu.printf "[GPU] TMA LOADED lhs[45][7] \t %f\n" %lhs0 : f32
        gpu.printf "[GPU] TMA LOADED rhs[7][0] \t %f\n" %rhs0 : f32
      }

      %descA = nvgpu.wgmma.generate.descriptor %lhsShmem, %lhsTensorMap : !shmemlhs, !lhsTensorMap
      %descB = nvgpu.wgmma.generate.descriptor %rhsShmem, %rhsTensorMap : !shmemrhs, !rhsTensorMap

      // Step 6. Do Matmul Async and Wait. These are not ready yet.
      %scaleD = arith.constant 1 : i32 // D = A*B (no accumulate itself)
      nvvm.wgmma.fence.aligned
      %result = nvvm.wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 %descA, %descB, %scaleD, 1, 1 -> !accMatrix
      nvvm.wgmma.commit.group.sync.aligned
      nvvm.wgmma.wait.group.sync.aligned 0

      // Step 7. Accumulate the result
      %v0 = llvm.extractvalue %result[0] : !accMatrix
      %v1 = llvm.extractvalue %result[1] : !accMatrix
      %v2 = llvm.extractvalue %result[2] : !accMatrix
      %v3 = llvm.extractvalue %result[3] : !accMatrix
      %v4 = llvm.extractvalue %result[4] : !accMatrix
      %v5 = llvm.extractvalue %result[5] : !accMatrix
      %v6 = llvm.extractvalue %result[6] : !accMatrix
      %v7 = llvm.extractvalue %result[7] : !accMatrix
      %v8 = llvm.extractvalue %result[8] : !accMatrix
      %v9 = llvm.extractvalue %result[9] : !accMatrix

      %v10 = llvm.extractvalue %result[10] : !accMatrix
      %v11 = llvm.extractvalue %result[11] : !accMatrix
      %v12 = llvm.extractvalue %result[12] : !accMatrix
      %v13 = llvm.extractvalue %result[13] : !accMatrix
      %v14 = llvm.extractvalue %result[14] : !accMatrix
      %v15 = llvm.extractvalue %result[15] : !accMatrix
      %v16 = llvm.extractvalue %result[16] : !accMatrix
      %v17 = llvm.extractvalue %result[17] : !accMatrix
      %v18 = llvm.extractvalue %result[18] : !accMatrix
      %v19 = llvm.extractvalue %result[19] : !accMatrix
      
      %v20 = llvm.extractvalue %result[20] : !accMatrix
      %v21 = llvm.extractvalue %result[21] : !accMatrix
      %v22 = llvm.extractvalue %result[22] : !accMatrix
      %v23 = llvm.extractvalue %result[23] : !accMatrix
      %v24 = llvm.extractvalue %result[24] : !accMatrix
      %v25 = llvm.extractvalue %result[25] : !accMatrix
      %v26 = llvm.extractvalue %result[26] : !accMatrix
      %v27 = llvm.extractvalue %result[27] : !accMatrix
      %v28 = llvm.extractvalue %result[28] : !accMatrix
      %v29 = llvm.extractvalue %result[29] : !accMatrix
      
      %v30 = llvm.extractvalue %result[30] : !accMatrix
      %v31 = llvm.extractvalue %result[31] : !accMatrix
      %v32 = llvm.extractvalue %result[32] : !accMatrix
      %v33 = llvm.extractvalue %result[33] : !accMatrix
      %v34 = llvm.extractvalue %result[34] : !accMatrix
      %v35 = llvm.extractvalue %result[35] : !accMatrix
      %v36 = llvm.extractvalue %result[36] : !accMatrix
      %v37 = llvm.extractvalue %result[37] : !accMatrix
      %v38 = llvm.extractvalue %result[38] : !accMatrix
      %v39 = llvm.extractvalue %result[39] : !accMatrix
      
      %v40 = llvm.extractvalue %result[40] : !accMatrix
      %v41 = llvm.extractvalue %result[41] : !accMatrix
      %v42 = llvm.extractvalue %result[42] : !accMatrix
      %v43 = llvm.extractvalue %result[43] : !accMatrix
      %v44 = llvm.extractvalue %result[44] : !accMatrix
      %v45 = llvm.extractvalue %result[45] : !accMatrix
      %v46 = llvm.extractvalue %result[46] : !accMatrix
      %v47 = llvm.extractvalue %result[47] : !accMatrix
      %v48 = llvm.extractvalue %result[48] : !accMatrix
      %v49 = llvm.extractvalue %result[49] : !accMatrix
      
      %v50 = llvm.extractvalue %result[50] : !accMatrix
      %v51 = llvm.extractvalue %result[51] : !accMatrix
      %v52 = llvm.extractvalue %result[52] : !accMatrix
      %v53 = llvm.extractvalue %result[53] : !accMatrix
      %v54 = llvm.extractvalue %result[54] : !accMatrix
      %v55 = llvm.extractvalue %result[55] : !accMatrix
      %v56 = llvm.extractvalue %result[56] : !accMatrix
      %v57 = llvm.extractvalue %result[57] : !accMatrix
      %v58 = llvm.extractvalue %result[58] : !accMatrix
      %v59 = llvm.extractvalue %result[59] : !accMatrix
      
      %v60 = llvm.extractvalue %result[60] : !accMatrix
      %v61 = llvm.extractvalue %result[61] : !accMatrix
      %v62 = llvm.extractvalue %result[62] : !accMatrix
      %v63 = llvm.extractvalue %result[63] : !accMatrix
      
      
      %ic1 = arith.constant 1 : index
      %ic2 = arith.constant 2 : index
      %ic3 = arith.constant 3 : index
      %ic4 = arith.constant 4 : index
      %ic5 = arith.constant 5 : index
      %ic6 = arith.constant 6 : index
      %ic8 = arith.constant 8 : index
      %ic9 = arith.constant 9 : index

      %ic10 = arith.constant 10 : index
      %ic11 = arith.constant 11 : index
      %ic12 = arith.constant 12 : index
      %ic13 = arith.constant 13 : index
      %ic14 = arith.constant 14 : index
      %ic15 = arith.constant 15 : index
      %ic16 = arith.constant 16 : index
      %ic17 = arith.constant 17 : index
      %ic18 = arith.constant 18 : index
      %ic19 = arith.constant 19 : index

      %ic20 = arith.constant 20 : index
      %ic21 = arith.constant 21 : index
      %ic22 = arith.constant 22 : index
      %ic23 = arith.constant 23 : index
      %ic24 = arith.constant 24 : index
      %ic25 = arith.constant 25 : index
      %ic26 = arith.constant 26 : index
      %ic27 = arith.constant 27 : index
      %ic28 = arith.constant 28 : index
      %ic29 = arith.constant 29 : index

      %ic30 = arith.constant 30 : index
      %ic31 = arith.constant 31 : index
      %ic32 = arith.constant 32 : index
      %ic33 = arith.constant 33 : index
      %ic34 = arith.constant 34 : index
      %ic35 = arith.constant 35 : index
      %ic36 = arith.constant 36 : index
      %ic37 = arith.constant 37 : index
      %ic38 = arith.constant 38 : index
      %ic39 = arith.constant 39 : index

      %ic40 = arith.constant 40 : index
      %ic41 = arith.constant 41 : index
      %ic42 = arith.constant 42 : index
      %ic43 = arith.constant 43 : index
      %ic44 = arith.constant 44 : index
      %ic46 = arith.constant 46 : index
      %ic47 = arith.constant 47 : index
      %ic48 = arith.constant 48 : index
      %ic49 = arith.constant 49 : index

      %ic50 = arith.constant 50 : index
      %ic51 = arith.constant 51 : index
      %ic52 = arith.constant 52 : index
      %ic53 = arith.constant 53 : index
      %ic54 = arith.constant 54 : index
      %ic55 = arith.constant 55 : index
      %ic56 = arith.constant 56 : index
      %ic57 = arith.constant 57 : index
      %ic58 = arith.constant 58 : index
      %ic59 = arith.constant 59 : index

      %ic60 = arith.constant 60 : index
      %ic61 = arith.constant 61 : index
      %ic62 = arith.constant 62 : index
      %ic63 = arith.constant 63 : index
      
      memref.store %v0, %acc[%ic0, %tidx] : !acc
      memref.store %v1, %acc[%ic1, %tidx] : !acc
      memref.store %v2, %acc[%ic2, %tidx] : !acc
      memref.store %v3, %acc[%ic3, %tidx] : !acc
      memref.store %v4, %acc[%ic4, %tidx] : !acc
      memref.store %v5, %acc[%ic5, %tidx] : !acc
      memref.store %v6, %acc[%ic6, %tidx] : !acc
      memref.store %v7, %acc[%ic7, %tidx] : !acc
      memref.store %v8, %acc[%ic8, %tidx] : !acc
      memref.store %v9, %acc[%ic9, %tidx] : !acc

      memref.store %v10, %acc[%ic10, %tidx] : !acc
      memref.store %v11, %acc[%ic11, %tidx] : !acc
      memref.store %v12, %acc[%ic12, %tidx] : !acc
      memref.store %v13, %acc[%ic13, %tidx] : !acc
      memref.store %v14, %acc[%ic14, %tidx] : !acc
      memref.store %v15, %acc[%ic15, %tidx] : !acc
      memref.store %v16, %acc[%ic16, %tidx] : !acc
      memref.store %v17, %acc[%ic17, %tidx] : !acc
      memref.store %v18, %acc[%ic18, %tidx] : !acc
      memref.store %v19, %acc[%ic19, %tidx] : !acc

      memref.store %v20, %acc[%ic20, %tidx] : !acc
      memref.store %v21, %acc[%ic21, %tidx] : !acc
      memref.store %v22, %acc[%ic22, %tidx] : !acc
      memref.store %v23, %acc[%ic23, %tidx] : !acc
      memref.store %v24, %acc[%ic24, %tidx] : !acc
      memref.store %v25, %acc[%ic25, %tidx] : !acc
      memref.store %v26, %acc[%ic26, %tidx] : !acc
      memref.store %v27, %acc[%ic27, %tidx] : !acc
      memref.store %v28, %acc[%ic28, %tidx] : !acc
      memref.store %v29, %acc[%ic29, %tidx] : !acc

      memref.store %v30, %acc[%ic30, %tidx] : !acc
      memref.store %v31, %acc[%ic31, %tidx] : !acc
      memref.store %v32, %acc[%ic32, %tidx] : !acc
      memref.store %v33, %acc[%ic33, %tidx] : !acc
      memref.store %v34, %acc[%ic34, %tidx] : !acc
      memref.store %v35, %acc[%ic35, %tidx] : !acc
      memref.store %v36, %acc[%ic36, %tidx] : !acc
      memref.store %v37, %acc[%ic37, %tidx] : !acc
      memref.store %v38, %acc[%ic38, %tidx] : !acc
      memref.store %v39, %acc[%ic39, %tidx] : !acc

      memref.store %v40, %acc[%ic40, %tidx] : !acc
      memref.store %v41, %acc[%ic41, %tidx] : !acc
      memref.store %v42, %acc[%ic42, %tidx] : !acc
      memref.store %v43, %acc[%ic43, %tidx] : !acc
      memref.store %v44, %acc[%ic44, %tidx] : !acc
      memref.store %v45, %acc[%ic45, %tidx] : !acc
      memref.store %v46, %acc[%ic46, %tidx] : !acc
      memref.store %v47, %acc[%ic47, %tidx] : !acc
      memref.store %v48, %acc[%ic48, %tidx] : !acc
      memref.store %v49, %acc[%ic49, %tidx] : !acc

      memref.store %v50, %acc[%ic50, %tidx] : !acc
      memref.store %v51, %acc[%ic51, %tidx] : !acc
      memref.store %v52, %acc[%ic52, %tidx] : !acc
      memref.store %v53, %acc[%ic53, %tidx] : !acc
      memref.store %v54, %acc[%ic54, %tidx] : !acc
      memref.store %v55, %acc[%ic55, %tidx] : !acc
      memref.store %v56, %acc[%ic56, %tidx] : !acc
      memref.store %v57, %acc[%ic57, %tidx] : !acc
      memref.store %v58, %acc[%ic58, %tidx] : !acc
      memref.store %v59, %acc[%ic59, %tidx] : !acc

      memref.store %v60, %acc[%ic60, %tidx] : !acc
      memref.store %v61, %acc[%ic61, %tidx] : !acc
      memref.store %v62, %acc[%ic62, %tidx] : !acc
      memref.store %v63, %acc[%ic63, %tidx] : !acc
      
      gpu.terminator
  }

  // Step.5 Print the memref after computation.  
  func.call @printMemrefF32(%acc_unranked) : (memref<*xf32>) -> ()  
  return 
}
}