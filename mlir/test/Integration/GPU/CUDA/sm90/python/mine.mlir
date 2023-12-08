module {
  func.func @main() attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<128x128xf16>
    %alloc_0 = memref.alloc() : memref<128x128xf16>
    %alloc_1 = memref.alloc() : memref<128x128xf32>
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<128x128xf16>
    %memref_2, %asyncToken_3 = gpu.alloc async [%0] () : memref<128x128xf16>
    %memref_4, %asyncToken_5 = gpu.alloc async [%0] () : memref<128x128xf32>
    %1 = gpu.memcpy async [%0] %memref, %alloc : memref<128x128xf16>, memref<128x128xf16>
    %cast = memref.cast %memref : memref<128x128xf16> to memref<*xf16>
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %2 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16> -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %3 = gpu.memcpy async [%0] %memref_2, %alloc_0 : memref<128x128xf16>, memref<128x128xf16>
    %cast_6 = memref.cast %memref_2 : memref<128x128xf16> to memref<*xf16>
    %c64_7 = arith.constant 64 : index
    %c64_8 = arith.constant 64 : index
    %4 = nvgpu.tma.create.descriptor %cast_6 box[%c64_7, %c64_8] : memref<*xf16> -> <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %c1 = arith.constant 1 : index
    %c1_9 = arith.constant 1 : index
    %c1_10 = arith.constant 1 : index
    %c128_11 = arith.constant 128 : index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %c65536_i32 = arith.constant 65536 : i32
    %5 = gpu.launch async [%0] blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1_9, %arg8 = %c1_10) threads(%arg3, %arg4, %arg5) in (%arg9 = %c128_11, %arg10 = %c1_12, %arg11 = %c1_13) dynamic_shared_memory_size %c65536_i32 {
      memref.assume_alignment %memref_4, 16 : memref<128x128xf32>
      %8 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c0 = arith.constant 0 : index
      %view = memref.view %8[%c0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x128x64xf16, #gpu.address_space<workgroup>>
      %c32768 = arith.constant 32768 : index
      %view_14 = memref.view %8[%c32768][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x64x128xf16, #gpu.address_space<workgroup>>
      %9 = gpu.thread_id  x
      %c0_15 = arith.constant 0 : index
      %10 = arith.cmpi eq, %9, %c0_15 : index
      %11 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
      %c1_16 = arith.constant 1 : index
      %c0_17 = arith.constant 0 : index
      nvgpu.mbarrier.init %11[%c0_17], %c1_16 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      nvgpu.mbarrier.init %11[%c1_19], %c1_18 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
      nvgpu.tma.prefetch.descriptor %2 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
      nvgpu.tma.prefetch.descriptor %4 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
      %c0_20 = arith.constant 0 : index
      %c32768_21 = arith.constant 32768 : index
      scf.if %10 {
        nvgpu.mbarrier.arrive.expect_tx %11[%c0_20], %c32768_21 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
        %subview = memref.subview %view[%c0_20, 0, 0] [1, 128, 64] [1, 1, 1] : memref<2x128x64xf16, #gpu.address_space<workgroup>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
        %c64_31 = arith.constant 64 : index
        %17 = arith.muli %c64_31, %c0_20 : index
        %c0_32 = arith.constant 0 : index
        nvgpu.tma.async.load %2[%17, %c0_32], %11[%c0_20] to %subview : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2> -> memref<128x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
        %subview_33 = memref.subview %view_14[%c0_20, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x128xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %c0_34 = arith.constant 0 : index
        %c64_35 = arith.constant 64 : index
        %18 = arith.muli %c64_35, %c0_20 : index
        nvgpu.tma.async.load %4[%c0_34, %18], %11[%c0_20] to %subview_33 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2> -> memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %subview_36 = memref.subview %view_14[%c0_20, 32, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x128xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %c32_37 = arith.constant 32 : index
        %c64_38 = arith.constant 64 : index
        %19 = arith.muli %c64_38, %c0_20 : index
        nvgpu.tma.async.load %4[%c32_37, %19], %11[%c0_20] to %subview_36 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2> -> memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      }
      %c1_22 = arith.constant 1 : index
      %c32768_23 = arith.constant 32768 : index
      scf.if %10 {
        nvgpu.mbarrier.arrive.expect_tx %11[%c1_22], %c32768_23 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
        %subview = memref.subview %view[%c1_22, 0, 0] [1, 128, 64] [1, 1, 1] : memref<2x128x64xf16, #gpu.address_space<workgroup>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
        %c64_31 = arith.constant 64 : index
        %17 = arith.muli %c64_31, %c1_22 : index
        %c0_32 = arith.constant 0 : index
        nvgpu.tma.async.load %2[%17, %c0_32], %11[%c1_22] to %subview : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2> -> memref<128x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
        %subview_33 = memref.subview %view_14[%c1_22, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x128xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %c0_34 = arith.constant 0 : index
        %c64_35 = arith.constant 64 : index
        %18 = arith.muli %c64_35, %c1_22 : index
        nvgpu.tma.async.load %4[%c0_34, %18], %11[%c1_22] to %subview_33 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2> -> memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %subview_36 = memref.subview %view_14[%c1_22, 32, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x128xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %c32_37 = arith.constant 32 : index
        %c64_38 = arith.constant 64 : index
        %19 = arith.muli %c64_38, %c1_22 : index
        nvgpu.tma.async.load %4[%c32_37, %19], %11[%c1_22] to %subview_36 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2> -> memref<64x64xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      }
      %12 = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
      %c0_24 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_25 = arith.constant 1 : index
      %13 = scf.for %arg12 = %c0_24 to %c2 step %c1_25 iter_args(%arg13 = %12) -> (!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>) {
        %c10000000 = arith.constant 10000000 : index
        %c0_31 = arith.constant 0 : index
        nvgpu.mbarrier.try_wait.parity %11[%arg12], %c0_31, %c10000000 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
        %subview = memref.subview %view[%arg12, 0, 0] [1, 128, 64] [1, 1, 1] : memref<2x128x64xf16, #gpu.address_space<workgroup>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
        %subview_32 = memref.subview %view_14[%arg12, 0, 0] [1, 64, 128] [1, 1, 1] : memref<2x64x128xf16, #gpu.address_space<workgroup>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
        %17 = nvgpu.warpgroup.generate.descriptor %subview, %2 : memref<128x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, 3>>
        %18 = nvgpu.warpgroup.generate.descriptor %subview_32, %4 : memref<64x128xf16, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, 3>>
        %19 = nvgpu.warpgroup.mma %17, %18, %arg13 {transposeB} : <tensor = memref<128x64xf16, 3>>, <tensor = memref<64x128xf16, 3>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
        scf.yield %19 : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
      }
      nvvm.wgmma.wait.group.sync.aligned 0
      %c0_26 = arith.constant 0 : index
      %view_27 = memref.view %8[%c0_26][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x128xf32, #gpu.address_space<workgroup>>
      nvgpu.warpgroup.mma.store %13, %view_27 : <fragmented = vector<128x128xf32>> to memref<128x128xf32, #gpu.address_space<workgroup>>
      %c32 = arith.constant 32 : index
      %14 = arith.divui %9, %c32 : index
      %c32_28 = arith.constant 32 : index
      %15 = arith.remui %9, %c32_28 : index
      %c4 = arith.constant 4 : index
      %16 = arith.muli %15, %c4 : index
      %c128_29 = arith.constant 128 : index
      %c4_30 = arith.constant 4 : index
      scf.for %arg12 = %14 to %c128_29 step %c4_30 {
        %17 = vector.load %view_27[%arg12, %16] : memref<128x128xf32, #gpu.address_space<workgroup>>, vector<4xf32>
        vector.store %17, %memref_4[%arg12, %16] : memref<128x128xf32>, vector<4xf32>
      }
      gpu.terminator
    }
    %6 = gpu.memcpy async [%0] %alloc_1, %memref_4 : memref<128x128xf32>, memref<128x128xf32>
    %7 = gpu.wait async [%0]
    return
  }
}

