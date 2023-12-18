// RUN: mlir-opt %s \
// RUN:  -gpu-lower-to-nvvm="cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3" \
// RUN:  | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN:  | FileCheck %s

// This program performs 4096x4096x4096 GEMM (F32 += F16 * F16)

// 6 stages - 6 slots in shared memory. 
// (128*64*2*2*6) = size of shmem in H100

// Execution Timeline (numbers are shared memory slots)
// +------+-----------------+------------------------------------------------------------------------------+
// |      |Prologue ---->   |MainLoop ---->                                                                |
// +------+-----------------+------------------------+------------------------+------------------------+---+
// |tidx=0|[tma-0,1,2,3,4,5]|[wait-0][wgmma-0][tma-0]|[wait-1][wgmma-1][tma-1]|[wait-2][wgmma-2][tma-2]|...|
// |wgroup| ................|[wait-0][wgmma-0]       |[wait-1][wgmma-1]       |[wait-2][wgmma-2]       |...|
// +------+-----------------+------------------------+------------------------+------------------------+---+


!barrierType = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo=l2promo_128b, oob=zero, interleave=none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo=l2promo_128b, oob=zero, interleave=none>

func.func private @printMemrefF32(%ptr : memref<*xf32>)

func.func @main() {
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c65536 = arith.constant 65536 : index
  %c4096 = arith.constant 4096 : index
  %c8 = arith.constant 8 : index
  %f2 = arith.constant 2.0 : f16
  %f0 = arith.constant 0.0 : f32
  %f3 = arith.constant 3.0 : f16

  // Step 1. Allocate and Initilize LHS and RHS Matrices 
  %lhs = memref.alloc() : memref<4096x4096xf16>
  %rhs = memref.alloc() : memref<4096x4096xf16>
  %lhs32 = memref.alloc() : memref<4096x4096xf32>
  %rhs32 = memref.alloc() : memref<4096x4096xf32>
  %acc = memref.alloc() : memref<4096x4096xf32>
  %accGold = memref.alloc() : memref<4096x4096xf32>
  scf.for %i = %c0 to %c4096 step %c1 {
    scf.for %j = %c0 to %c4096 step %c1 {
      memref.store %f0, %acc[%i, %j] : memref<4096x4096xf32>
      memref.store %f0, %accGold[%i, %j] : memref<4096x4096xf32>
      %v0 = arith.muli %i, %c128 : index         // i * 128
      %v00 = arith.addi %v0, %j : index          // i * 128 + j
      %v01 = arith.divui %v00, %c8 : index       // (i * 128 + j) / 8
      %v02 = arith.remui %v01, %c16 : index      // <<<<< mod 128
      %v2 = arith.index_cast %v02 : index to i32
      %vR = arith.sitofp %v2 : i32 to f16
      memref.store %vR, %rhs[%i, %j] : memref<4096x4096xf16>
      %vR32 = arith.extf %vR : f16 to f32
      memref.store %vR32, %rhs32[%i, %j] : memref<4096x4096xf32>
      %b0 = arith.muli %j, %c64 : index
      %b00 = arith.addi %b0, %i : index
      %b01 = arith.divui %b00, %c8 : index
      %b02 = arith.remui %b01, %c16 : index      // <<<<< mod 128
      %v1 = arith.index_cast %b02 : index to i32
      %vL = arith.sitofp %v1 : i32 to f16
      memref.store %vL, %lhs[%j, %i] : memref<4096x4096xf16>
      %vL32 = arith.extf %vL : f16 to f32
      memref.store %vL32, %lhs32[%j, %i] : memref<4096x4096xf32>
    }
  }

  // Step 2. Allocate Device Memory for LHS and RHS Matrices and Copy H2D
  %token = gpu.wait async
  %d_glbmem_lhs, %asyncToken = gpu.alloc async [%token] () : memref<4096x4096xf16>
  %d_glbmem_rhs, %asyncToken_2 = gpu.alloc async [%token] () : memref<4096x4096xf16>
  %d_glbmem_acc, %asyncToken_3 = gpu.alloc async [%token] () : memref<4096x4096xf32>
  %1 = gpu.memcpy async [%token] %d_glbmem_lhs, %lhs : memref<4096x4096xf16>,memref<4096x4096xf16>
  %2 = gpu.memcpy async [%token] %d_glbmem_rhs, %rhs : memref<4096x4096xf16>,memref<4096x4096xf16>
  gpu.wait [%token]

  // Step 3. Create TMA Descriptor
  %d_lhs_unranked = memref.cast %d_glbmem_lhs :memref<4096x4096xf16>  to memref<*xf16>
  %d_rhs_unranked = memref.cast %d_glbmem_rhs :memref<4096x4096xf16>  to memref<*xf16>
  %d_lhsTensorMap = nvgpu.tma.create.descriptor %d_lhs_unranked box[%c128, %c64] : memref<*xf16> -> !lhsTensorMap
  %d_rhsTensorMap = nvgpu.tma.create.descriptor %d_rhs_unranked box[%c64, %c64] : memref<*xf16> -> !rhsTensorMap
  
  %shmemsize = arith.constant 214016 : i32

  // Step 4. Launch GPU Kernel
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c32, %grid_y = %c32, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1)
           dynamic_shared_memory_size %shmemsize 
  { 
      memref.assume_alignment %d_glbmem_acc, 16 : memref<4096x4096xf32>

      %num_threads = gpu.block_dim x
      %bidx = gpu.block_id x
      %bidy = gpu.block_id y
      %num_bars = arith.constant 7 : index

      %ic0 = arith.constant 0 : index
      %ic1 = arith.constant 1 : index
      %ic2 = arith.constant 2 : index
      %ic3 = arith.constant 3 : index
      %ic4 = arith.constant 4 : index
      %ic5 = arith.constant 5 : index
      %ic6 = arith.constant 6 : index
      %ic7 = arith.constant 7 : index
      %ic64 = arith.constant 64 : index
      %ic32 = arith.constant 32 : index
      %ic128 = arith.constant 128 : index
      %txcount = arith.constant 32768 : index      
      %tidx = gpu.thread_id x

      // Step 1. [GPU] Get shared memory pointers. This could be dynamic shared memory, but for now I use static
      %dynamicShmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>

      // Step 2. [GPU] Create Async Transactional Barriers (mbarriers)
      %barrier = nvgpu.mbarrier.create -> !barrierType

      // Step 3. [GPU] Elect fastest thread in CTA
      %mask = arith.constant -1 : i32
      %i0 = arith.constant 0 : i32
      %i32 = arith.constant 32 : i32
      %i4 = arith.constant 4 : i32
      %lanePredicate = nvvm.elect.sync -> i1
      %warpIdx = arith.divui %tidx, %c32 : index
      %warpIdxi32 = index.casts %warpIdx : index to i32    
      %canonical_warp_idx = nvvm.shfl.sync idx %i32, %warpIdxi32, %i0, %mask : i32 -> i32
      %warp_idx_in_group = arith.remui %canonical_warp_idx, %i4 : i32
      %cnd1 = arith.cmpi eq, %warp_idx_in_group, %i0 : i32
      %cnd = arith.andi %cnd1, %lanePredicate : i1

      // Step 3. [GPU] Initialize mbarriers (predicated threadIdx==0)
      nvgpu.mbarrier.init %barrier[%ic0], %ic1, predicate = %cnd : !barrierType
      nvgpu.mbarrier.init %barrier[%ic1], %ic1, predicate = %cnd : !barrierType
      nvgpu.mbarrier.init %barrier[%ic2], %ic1, predicate = %cnd : !barrierType
      nvgpu.mbarrier.init %barrier[%ic3], %ic1, predicate = %cnd : !barrierType
      nvgpu.mbarrier.init %barrier[%ic4], %ic1, predicate = %cnd : !barrierType
      nvgpu.mbarrier.init %barrier[%ic5], %ic1, predicate = %cnd : !barrierType
      nvgpu.mbarrier.init %barrier[%ic6], %ic1, predicate = %cnd : !barrierType
      
      // Step 4. [GPU] Prefetch TMA Descriptors to L1 Cache (predicated)
      nvgpu.tma.prefetch.descriptor %d_lhsTensorMap, predicate = %cnd : !lhsTensorMap
      nvgpu.tma.prefetch.descriptor %d_rhsTensorMap, predicate = %cnd : !rhsTensorMap
      
      // Step 5. [GPU] Prologue, fill the shared memory slots
      %dimX = arith.muli %bidx, %ic128 : index
      %dimY = arith.muli %bidy, %ic128 : index
      %lhsBase = arith.constant 0 : index
      %rhsBase = arith.constant 114688 : index    
      %tileSize = arith.constant 16384 : index    
      %rhsSmallTile = arith.constant 8192 : index 
      scf.for %i = %ic0 to %ic6 step %ic1 {
        %ilhs = arith.muli %tileSize, %i : index
        %irhs_1 = arith.addi %ilhs, %rhsBase : index
        %irhs_2 = arith.addi %irhs_1, %rhsSmallTile : index
        %lhsSlice = memref.view %dynamicShmem[%ilhs][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
        %rhsSlice = memref.view %dynamicShmem[%irhs_1][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
        %rhsSlice2 = memref.view %dynamicShmem[%irhs_2][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
        %dim = arith.muli %i, %ic64 : index
        %dimY64 = arith.addi %ic64, %dimY : index
        nvgpu.mbarrier.arrive.expect_tx %barrier[%i], %txcount, predicate =%cnd : !barrierType
        nvgpu.tma.async.load %d_lhsTensorMap[%dim, %dimX], %barrier[%i] to %lhsSlice, predicate = %cnd : !lhsTensorMap, !barrierType -> memref<128x64xf16, #gpu.address_space<workgroup>>
        nvgpu.tma.async.load %d_rhsTensorMap[%ic0, %dimY], %barrier[%i] to %rhsSlice, predicate = %cnd : !rhsTensorMap, !barrierType -> memref<64x64xf16, #gpu.address_space<workgroup>>
        nvgpu.tma.async.load %d_rhsTensorMap[%ic64, %dimY64], %barrier[%i] to %rhsSlice2, predicate = %cnd : !rhsTensorMap, !barrierType -> memref<64x64xf16, #gpu.address_space<workgroup>>
      }
      
      // Step 6. [GPU] Initiliaze accumulator matrix
      %matrixInit = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
      
      // Step 7. [GPU] Main Loop Starts
      %matrixD = scf.for %i = %ic1 to %ic4 step %ic1 iter_args(%matrixC = %matrixInit) -> !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
      {
        %ticks = arith.constant 10000000 : index
        %waitingBarrierId = arith.remui %i, %num_bars : index
        %shmemStage = arith.remui %i, %ic6 : index
        %phase = arith.divui %i, %num_bars : index

        // Step 7.1. [GPU] TMA wait
        nvgpu.mbarrier.try_wait.parity %barrier[%waitingBarrierId], %phase, %ticks : !barrierType
      
        %ilhs = arith.muli %tileSize, %shmemStage : index
        %irhs_1 = arith.addi %ilhs, %rhsBase : index
        %irhs_2 = arith.addi %irhs_1, %rhsSmallTile : index
        %lhsSlice = memref.view %dynamicShmem[%ilhs][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
        %rhsSlice = memref.view %dynamicShmem[%irhs_1][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
        
        // Step 7.2. [GPU] Descriptor WGMMA
        %dA = nvgpu.warpgroup.generate.descriptor %lhsSlice, %d_lhsTensorMap : memref<128x64xf16, #gpu.address_space<workgroup>>, !lhsTensorMap -> <tensor=memref<128x64xf16, #gpu.address_space<workgroup>>>
        %dB = nvgpu.warpgroup.generate.descriptor %rhsSlice, %d_rhsTensorMap : memref<64x128xf16, #gpu.address_space<workgroup>>, !rhsTensorMap -> <tensor=memref<64x128xf16, #gpu.address_space<workgroup>>>

        // Step 7.3. [GPU] Perform WGMMA 128x128x64
        %matrixD = nvgpu.warpgroup.mma %dA, %dB, %matrixC : 
                    <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>,
                    <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>

        // Step 7.4. [GPU] Fetch TMA for the next stage
        %nextPipe = arith.addi %ic6, %i : index 
        %nextBarrierId = arith.remui %nextPipe, %num_bars : index        
        %cnd3 = arith.cmpi ne, %i, %ic64 : index 
        %cnd2 = arith.andi %cnd3, %cnd : i1
        nvgpu.mbarrier.arrive.expect_tx %barrier[%nextBarrierId], %txcount, predicate = %cnd2 : !barrierType
            
        %rhsSlice1 = memref.view %dynamicShmem[%irhs_1][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
        %rhsSlice2 = memref.view %dynamicShmem[%irhs_2][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
        %dim = arith.muli %nextPipe, %ic64 : index
        nvgpu.tma.async.load %d_lhsTensorMap[%dim, %dimX], %barrier[%nextBarrierId] to %lhsSlice, predicate = %cnd2 : !lhsTensorMap, !barrierType -> memref<128x64xf16, #gpu.address_space<workgroup>>
        nvgpu.tma.async.load %d_rhsTensorMap[%dimY, %dim], %barrier[%nextBarrierId] to %rhsSlice1, predicate = %cnd2 : !rhsTensorMap, !barrierType -> memref<64x64xf16, #gpu.address_space<workgroup>>
        %dimY64 = arith.addi %dimY, %ic64 : index
        nvgpu.tma.async.load %d_rhsTensorMap[%dimY64, %dim], %barrier[%nextBarrierId] to %rhsSlice2, predicate = %cnd2 : !rhsTensorMap, !barrierType -> memref<64x64xf16, #gpu.address_space<workgroup>>
        
        scf.yield %matrixD : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
      }

      // Step 8. [GPU] Wait all to finish mma
      nvvm.wgmma.wait.group.sync.aligned 0

      // Step 9. [GPU] Epilogue, store fragmented register to shared memory
      %accShmemPtr = memref.view %dynamicShmem[%c0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x128xf32, #gpu.address_space<workgroup>>
      nvgpu.warpgroup.mma.store %matrixD, %accShmemPtr : <fragmented = vector<128x128xf32>> to memref<128x128xf32, #gpu.address_space<workgroup>>

      // Step 10. [GPU] Epilogue, shared memory to global memory
      %d_glbmem_acc_part = memref.subview %d_glbmem_acc[%dimX, %dimY][128, 128][1, 1] : memref<4096x4096xf32> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
      memref.assume_alignment %d_glbmem_acc_part, 16 : memref<128x128xf32, strided<[4096, 1], offset: ?>>
      %warpId = arith.divui %tidx, %ic32 : index
      %laneId = arith.remui %tidx, %ic32 : index
      scf.for %i = %warpId to %ic128 step %ic4 {          
        %idx = arith.muli %laneId, %ic4 : index
        %readme = vector.load %accShmemPtr[%i, %idx] : memref<128x128xf32, #gpu.address_space<workgroup>>, vector<4xf32>
        vector.store %readme, %d_glbmem_acc_part[%i, %idx] : memref<128x128xf32, strided<[4096, 1], offset: ?>>, vector<4xf32>
      }
      gpu.terminator
  }
  
  // Step 5. Copy D2H
  %3 = gpu.memcpy async [%token] %acc, %d_glbmem_acc : memref<4096x4096xf32>, memref<4096x4096xf32>
  gpu.wait [%token]

  // Step 6. Compute on host
  // linalg.matmul ins(%lhs, %rhs : memref<4096x4096xf16>, memref<4096x4096xf16>) outs(%accGold : memref<4096x4096xf32>)
  
  // Step 7. Verify
  // %ic1 = arith.constant 1 : i32
  // %ic0 = arith.constant 0 : i32
  // %tolerance = arith.constant 0.00000001 : f32
  // %errorCount, %correctCount = 
  // scf.for %i = %c0 to %c256 step %c1 iter_args(%ec1 = %ic0, %cc1 = %ic0) -> (i32,i32) {
  //   %ec2, %cc2 = 
  //   scf.for %j = %c0 to %c256 step %c1  iter_args(%ec2 = %ec1, %cc2 = %cc1) -> (i32,i32){
  //     %v1 = memref.load %accGold[%i,%j] : memref<4096x4096xf32>
  //     %v2 = memref.load %acc[%i,%j] : memref<4096x4096xf32>
  //     %g1 = arith.subf %v1,%v2 : f32
  //     %g2 = math.absf %g1: f32
  //     %g3 = arith.cmpf ult, %tolerance, %g2 : f32        
  //     %ec3, %cc3 = scf.if %g3 -> (i32, i32) {
  //       %coor = arith.constant dense<-1> : vector<2xi32>
  //       %i32 = arith.index_cast %i : index to i32
  //       %j32 = arith.index_cast %j : index to i32
  //       %coord1 = vector.insert %i32, %coor[0] : i32 into vector<2xi32>
  //       %coord2 = vector.insert %j32, %coord1[1] : i32 into vector<2xi32>
  //       // vector.print %coord2 : vector<2xi32>
  //       %ec3 = arith.addi %ec2, %ic1 : i32
  //       scf.yield %ec3, %cc2 : i32, i32
  //     } else {
  //       %cc3 = arith.addi %cc2, %ic1 : i32
  //       scf.yield %ec2, %cc3 : i32, i32
  //     }
  //     scf.yield %ec3, %cc3 : i32,i32
  //   }
  //   scf.yield %ec2,%cc2 : i32,i32
  // }
  // vector.print str "Correct Results :"
  // vector.print %correctCount : i32
  // vector.print str "Incorrect Results :"
  // vector.print %errorCount : i32
  return 
}

