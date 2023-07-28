!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// TMA device descriptor
!lhs = memref<4096x4096xf16>
!rhs = memref<4096x4096xf16>
!acc = memref<4096x4096xf16>

!shmemlhs = memref<128x64xf16,3>
!shmemrhs = memref<128x64xf16,3>

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
      %scaleD = arith.constant 1 : i32 // D = A*B (no accumulate itself)
      nvvm.wgmma.fence.aligned
      %wgmma_result = nvvm.wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %descA, %descB, %scaleD -> !accMatrix
      nvvm.wgmma.commit.group.sync.aligned
      nvvm.wgmma.wait.group.sync.aligned 1
  
      // Step 7. Sanity check WGMMA (2nd and 3rd Wapr)
      %cnd125 = arith.cmpi uge, %tidx, %ic63 : index          
      scf.if %cnd125 {
        %r0 = llvm.extractvalue %wgmma_result[0] : !accMatrix
        %r1 = llvm.extractvalue %wgmma_result[1] : !accMatrix
        %r2 = llvm.extractvalue %wgmma_result[2] : !accMatrix
        %r3 = llvm.extractvalue %wgmma_result[3] : !accMatrix
        %r4 = llvm.extractvalue %wgmma_result[4] : !accMatrix
        %r5 = llvm.extractvalue %wgmma_result[5] : !accMatrix
        gpu.printf "[GPU][thread=%3d] WGMMA [0]=%4.1f [1]=%4.1f [2]=%4.1f [3]=%4.1f [4]=%4.1f [5]=%4.1f\n" %tidx, %r0, %r1, %r2, %r3, %r4, %r5 : index, f32, f32, f32, f32, f32, f32
      }

      gpu.terminator
  }

  return 
}
}
// Runtime 

// CudaRuntimeWrappers.cpp:321:mgpuTensorMapEncodeTiled(): Created TMA descriptor
//  TMA Desc Addr: 0x7ffcd26b4000
// data type : 6
// rank : 2
// globalDim[5]: (4096, 4096, 0, 0, 0)
// globalStrides[5]: (8192, 0, 0, 0, 0)
// boxDim[5]: (64, 128, 0, 0, 0)
// elementStrides[5]: (1, 1, 0, 0, 0)
// interleave: 0 
// swizzle: 3 
// l2Promotion: 0 
// oobFill: 0 
// CudaRuntimeWrappers.cpp:321:mgpuTensorMapEncodeTiled(): Created TMA descriptor
//  TMA Desc Addr: 0x7ffcd26b3fc0
// data type : 6
// rank : 2
// globalDim[5]: (4096, 4096, 0, 0, 0)
// globalStrides[5]: (8192, 0, 0, 0, 0)
// boxDim[5]: (64, 128, 0, 0, 0)
// elementStrides[5]: (1, 1, 0, 0, 0)
// interleave: 0 
// swizzle: 3 
// l2Promotion: 0 
// oobFill: 0 
// [GPU] TMA SIZE 32768
// [GPU] Before TMA Shmem lhs[0][0]         0.000000
// [GPU] Before TMA Shmem rhs[0][0]         0.000000
// [GPU] TMA Loaded shmem lhs[0][0]         7.000000
// [GPU] TMA Loaded shmem rhs[0][0]         3.000000
// WGMMA DescA : 0x4000004000010040
// WGMMA DescB : 0x4000004000010440
// [GPU][thread= 63] WGMMA [0]=1104.0 [1]=1104.0 [2]=1488.0 [3]=1488.0 [4]=1104.0 [5]=1104.0
// [GPU][thread= 64] WGMMA [0]=1536.0 [1]=1536.0 [2]=1920.0 [3]=1920.0 [4]=1536.0 [5]=1536.0
// [GPU][thread= 65] WGMMA [0]=1536.0 [1]=1536.0 [2]=1920.0 [3]=1920.0 [4]=1536.0 [5]=1536.0
// [GPU][thread= 66] WGMMA [0]=1536.0 [1]=1536.0 [2]=1920.0 [3]=1920.0 [4]=1536.0 [5]=1536.0
// [GPU][thread= 67] WGMMA [0]=1536.0 [1]=1536.0 [2]=1920.0 [3]=1920.0 [4]=1536.0 [5]=1536.0
// [GPU][thread= 68] WGMMA [0]=1584.0 [1]=1584.0 [2]=1968.0 [3]=1968.0 [4]=1584.0 [5]=1584.0
// [GPU][thread= 69] WGMMA [0]=1584.0 [1]=1584.0 [2]=1968.0 [3]=1968.0 [4]=1584.0 [5]=1584.0
// [GPU][thread= 70] WGMMA [0]=1584.0 [1]=1584.0 [2]=1968.0 [3]=1968.0 [4]=1584.0 [5]=1584.0
// [GPU][thread= 71] WGMMA [0]=1584.0 [1]=1584.0 [2]=1968.0 [3]=1968.0 [4]=1584.0 [5]=1584.0
// [GPU][thread= 72] WGMMA [0]=1632.0 [1]=1632.0 [2]=2016.0 [3]=2016.0 [4]=1632.0 [5]=1632.0
// [GPU][thread= 73] WGMMA [0]=1632.0 [1]=1632.0 [2]=2016.0 [3]=2016.0 [4]=1632.0 [5]=1632.0
// [GPU][thread= 74] WGMMA [0]=1632.0 [1]=1632.0 [2]=2016.0 [3]=2016.0 [4]=1632.0 [5]=1632.0
// [GPU][thread= 75] WGMMA [0]=1632.0 [1]=1632.0 [2]=2016.0 [3]=2016.0 [4]=1632.0 [5]=1632.0
// [GPU][thread= 76] WGMMA [0]=1680.0 [1]=1680.0 [2]=2064.0 [3]=2064.0 [4]=1680.0 [5]=1680.0
// [GPU][thread= 77] WGMMA [0]=1680.0 [1]=1680.0 [2]=2064.0 [3]=2064.0 [4]=1680.0 [5]=1680.0
// [GPU][thread= 78] WGMMA [0]=1680.0 [1]=1680.0 [2]=2064.0 [3]=2064.0 [4]=1680.0 [5]=1680.0
// [GPU][thread= 79] WGMMA [0]=1680.0 [1]=1680.0 [2]=2064.0 [3]=2064.0 [4]=1680.0 [5]=1680.0
// [GPU][thread= 80] WGMMA [0]=1728.0 [1]=1728.0 [2]=2112.0 [3]=2112.0 [4]=1728.0 [5]=1728.0
// [GPU][thread= 81] WGMMA [0]=1728.0 [1]=1728.0 [2]=2112.0 [3]=2112.0 [4]=1728.0 [5]=1728.0
// [GPU][thread= 82] WGMMA [0]=1728.0 [1]=1728.0 [2]=2112.0 [3]=2112.0 [4]=1728.0 [5]=1728.0
// [GPU][thread= 83] WGMMA [0]=1728.0 [1]=1728.0 [2]=2112.0 [3]=2112.0 [4]=1728.0 [5]=1728.0
// [GPU][thread= 84] WGMMA [0]=1776.0 [1]=1776.0 [2]=2160.0 [3]=2160.0 [4]=1776.0 [5]=1776.0
// [GPU][thread= 85] WGMMA [0]=1776.0 [1]=1776.0 [2]=2160.0 [3]=2160.0 [4]=1776.0 [5]=1776.0
// [GPU][thread= 86] WGMMA [0]=1776.0 [1]=1776.0 [2]=2160.0 [3]=2160.0 [4]=1776.0 [5]=1776.0
// [GPU][thread= 87] WGMMA [0]=1776.0 [1]=1776.0 [2]=2160.0 [3]=2160.0 [4]=1776.0 [5]=1776.0
// [GPU][thread= 88] WGMMA [0]=1824.0 [1]=1824.0 [2]=2208.0 [3]=2208.0 [4]=1824.0 [5]=1824.0
// [GPU][thread= 89] WGMMA [0]=1824.0 [1]=1824.0 [2]=2208.0 [3]=2208.0 [4]=1824.0 [5]=1824.0
// [GPU][thread= 90] WGMMA [0]=1824.0 [1]=1824.0 [2]=2208.0 [3]=2208.0 [4]=1824.0 [5]=1824.0
// [GPU][thread= 91] WGMMA [0]=1824.0 [1]=1824.0 [2]=2208.0 [3]=2208.0 [4]=1824.0 [5]=1824.0
// [GPU][thread= 92] WGMMA [0]=1872.0 [1]=1872.0 [2]=2256.0 [3]=2256.0 [4]=1872.0 [5]=1872.0
// [GPU][thread= 93] WGMMA [0]=1872.0 [1]=1872.0 [2]=2256.0 [3]=2256.0 [4]=1872.0 [5]=1872.0
// [GPU][thread= 94] WGMMA [0]=1872.0 [1]=1872.0 [2]=2256.0 [3]=2256.0 [4]=1872.0 [5]=1872.0
// [GPU][thread= 95] WGMMA [0]=1872.0 [1]=1872.0 [2]=2256.0 [3]=2256.0 [4]=1872.0 [5]=1872.0
// [GPU][thread= 96] WGMMA [0]=2304.0 [1]=2304.0 [2]=2688.0 [3]=2688.0 [4]=2304.0 [5]=2304.0
// [GPU][thread= 97] WGMMA [0]=2304.0 [1]=2304.0 [2]=2688.0 [3]=2688.0 [4]=2304.0 [5]=2304.0
// [GPU][thread= 98] WGMMA [0]=2304.0 [1]=2304.0 [2]=2688.0 [3]=2688.0 [4]=2304.0 [5]=2304.0
// [GPU][thread= 99] WGMMA [0]=2304.0 [1]=2304.0 [2]=2688.0 [3]=2688.0 [4]=2304.0 [5]=2304.0
// [GPU][thread=100] WGMMA [0]=2352.0 [1]=2352.0 [2]=2736.0 [3]=2736.0 [4]=2352.0 [5]=2352.0
// [GPU][thread=101] WGMMA [0]=2352.0 [1]=2352.0 [2]=2736.0 [3]=2736.0 [4]=2352.0 [5]=2352.0
// [GPU][thread=102] WGMMA [0]=2352.0 [1]=2352.0 [2]=2736.0 [3]=2736.0 [4]=2352.0 [5]=2352.0
// [GPU][thread=103] WGMMA [0]=2352.0 [1]=2352.0 [2]=2736.0 [3]=2736.0 [4]=2352.0 [5]=2352.0
// [GPU][thread=104] WGMMA [0]=2400.0 [1]=2400.0 [2]=2784.0 [3]=2784.0 [4]=2400.0 [5]=2400.0
// [GPU][thread=105] WGMMA [0]=2400.0 [1]=2400.0 [2]=2784.0 [3]=2784.0 [4]=2400.0 [5]=2400.0
// [GPU][thread=106] WGMMA [0]=2400.0 [1]=2400.0 [2]=2784.0 [3]=2784.0 [4]=2400.0 [5]=2400.0
// [GPU][thread=107] WGMMA [0]=2400.0 [1]=2400.0 [2]=2784.0 [3]=2784.0 [4]=2400.0 [5]=2400.0
// [GPU][thread=108] WGMMA [0]=2448.0 [1]=2448.0 [2]=2832.0 [3]=2832.0 [4]=2448.0 [5]=2448.0
// [GPU][thread=109] WGMMA [0]=2448.0 [1]=2448.0 [2]=2832.0 [3]=2832.0 [4]=2448.0 [5]=2448.0
// [GPU][thread=110] WGMMA [0]=2448.0 [1]=2448.0 [2]=2832.0 [3]=2832.0 [4]=2448.0 [5]=2448.0
// [GPU][thread=111] WGMMA [0]=2448.0 [1]=2448.0 [2]=2832.0 [3]=2832.0 [4]=2448.0 [5]=2448.0
// [GPU][thread=112] WGMMA [0]=2496.0 [1]=2496.0 [2]=2880.0 [3]=2880.0 [4]=2496.0 [5]=2496.0
// [GPU][thread=113] WGMMA [0]=2496.0 [1]=2496.0 [2]=2880.0 [3]=2880.0 [4]=2496.0 [5]=2496.0
// [GPU][thread=114] WGMMA [0]=2496.0 [1]=2496.0 [2]=2880.0 [3]=2880.0 [4]=2496.0 [5]=2496.0
// [GPU][thread=115] WGMMA [0]=2496.0 [1]=2496.0 [2]=2880.0 [3]=2880.0 [4]=2496.0 [5]=2496.0
// [GPU][thread=116] WGMMA [0]=2544.0 [1]=2544.0 [2]=2928.0 [3]=2928.0 [4]=2544.0 [5]=2544.0
// [GPU][thread=117] WGMMA [0]=2544.0 [1]=2544.0 [2]=2928.0 [3]=2928.0 [4]=2544.0 [5]=2544.0
// [GPU][thread=118] WGMMA [0]=2544.0 [1]=2544.0 [2]=2928.0 [3]=2928.0 [4]=2544.0 [5]=2544.0
// [GPU][thread=119] WGMMA [0]=2544.0 [1]=2544.0 [2]=2928.0 [3]=2928.0 [4]=2544.0 [5]=2544.0
// [GPU][thread=120] WGMMA [0]=2592.0 [1]=2592.0 [2]=2976.0 [3]=2976.0 [4]=2592.0 [5]=2592.0
// [GPU][thread=121] WGMMA [0]=2592.0 [1]=2592.0 [2]=2976.0 [3]=2976.0 [4]=2592.0 [5]=2592.0
// [GPU][thread=122] WGMMA [0]=2592.0 [1]=2592.0 [2]=2976.0 [3]=2976.0 [4]=2592.0 [5]=2592.0
// [GPU][thread=123] WGMMA [0]=2592.0 [1]=2592.0 [2]=2976.0 [3]=2976.0 [4]=2592.0 [5]=2592.0
// [GPU][thread=124] WGMMA [0]=2640.0 [1]=2640.0 [2]=3024.0 [3]=3024.0 [4]=2640.0 [5]=2640.0
// [GPU][thread=125] WGMMA [0]=2640.0 [1]=2640.0 [2]=3024.0 [3]=3024.0 [4]=2640.0 [5]=2640.0
// [GPU][thread=126] WGMMA [0]=2640.0 [1]=2640.0 [2]=3024.0 [3]=3024.0 [4]=2640.0 [5]=2640.0
// [GPU][thread=127] WGMMA [0]=2640.0 [1]=2640.0 [2]=3024.0 [3]=3024.0 [4]=2640.0 [5]=2640.0