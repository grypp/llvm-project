module @mymod {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    gpu.launch 
    blocks(%arg0, %arg1, %arg2) in (%arg6 = %c4, %arg7 = %c1, %arg8 = %c1) 
    threads(%arg3, %arg4, %arg5) in (%arg9 = %c4, %arg10 = %c1, %arg11 = %c1) {
      %bdimx = nvvm.read.ptx.sreg.ntid.x : i32
      %tidx = nvvm.read.ptx.sreg.tid.x : i32
      %cidx = nvvm.read.ptx.sreg.clusterid.x : i32
      %cnumx = nvvm.read.ptx.sreg.nclusterid.x : i32
      gpu.printf "===----------------=== blockDim: %d threadIdx: %d clusterIdx %d clusterDimX %d \n" %bdimx, %tidx, %cidx,  %cnumx: i32, i32, i32, i32
      gpu.terminator
    }
    return
  }
}

