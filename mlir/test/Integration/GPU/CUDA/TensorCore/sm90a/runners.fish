function gry_compile
    mlir-opt $argv[1] --convert-nvgpu-to-nvvm -gpu-kernel-outlining -convert-scf-to-cf -convert-nvvm-to-llvm \
        -convert-vector-to-llvm \
        -convert-math-to-llvm \
        -expand-strided-metadata \
        -lower-affine \
        -convert-index-to-llvm=index-bitwidth=32 \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-func-to-llvm \
        -canonicalize &>generated_1.mlir
    mlir-opt generated_1.mlir -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-nvgpu-to-nvvm{use-opaque-pointers=1},lower-affine,convert-scf-to-cf,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,convert-index-to-llvm{index-bitwidth=32},convert-arith-to-llvm,reconcile-unrealized-casts,gpu-to-cubin{chip=sm_90a features=+ptx80 dump-ptx}))' 1>generated_cubin.mlir 2>generated_cubin.ptx

    mlir-opt generated_cubin.mlir -convert-index-to-llvm=index-bitwidth=32 \
        -gpu-to-llvm \
        -convert-func-to-llvm \
        -cse -canonicalize \
        -reconcile-unrealized-casts 1>$argv[2]
end



function gry-run --description 'Compile and run'
    gry_compile $argv[1] exe.mlir
    mlir-cpu-runner exe.mlir \
        --shared-libs=../../../../../../../build/lib/libmlir_cuda_runtime.so \
        --shared-libs=../../../../../../..//build/lib/libmlir_runner_utils.so \
        --entry-point-result=void
end

function gry-profile --description 'Debug the given MLIR cuda-gdb (compile with gry_compile first)'
    nsys nvprof --print-gpu-trace mlir-cpu-runner $argv[1] \
        --shared-libs=../../../../../../..//build/lib/libmlir_cuda_runtime.so \
        --shared-libs=../../../../../../..//build/lib/libmlir_runner_utils.so \
        --entry-point-result=void
end


function gry-debug --description 'Debug the given MLIR cuda-gdb (compile with gry_compile first)'
    cuda-gdb --args mlir-cpu-runner $argv[1] \
        --shared-libs=../../../../../../..//build/lib/libmlir_cuda_runtime.so \
        --shared-libs=../../../../../../..//build/lib/libmlir_runner_utils.so \
        --entry-point-result=void
end