# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

import errno
import numpy as np
import subprocess
import ctypes
from tools import nvgpucompiler
from build_matmul import *


def generate_matmul(input_type=np.float16,
                    output_type=np.float32,
                    M=4096,
                    N=4096,
                    K=4096,
                    BLOCK_M=128,
                    BLOCK_N=128,
                    BLOCK_K=64,
                    use_warp_specilization=True,
                    dryrun=False,
                    max_num_stages=3):
    with ir.Context() as ctx, ir.Location.unknown():
        if use_warp_specilization:
            mlir_nvgpu_module = generate_matmul_ws(input_type, output_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                   max_num_stages)
        else:
            mlir_nvgpu_module = generate_matmul_multistage(input_type, output_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                           max_num_stages)

        mlir_nvgpu_module.operation.verify()

        if dryrun:
            print(mlir_nvgpu_module)
            return

        original_stdout = sys.stdout
        subprocess.call("rm -rf gry*", shell=True)
        with open('gry.mlir', 'w') as f:
            sys.stdout = f
            print(mlir_nvgpu_module)
            sys.stdout = original_stdout

        # Get compiler
        options = f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
        support_lib = os.getenv("SUPPORT_LIB")
        if not os.path.exists(support_lib):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)
        compiler = nvgpucompiler.NvgpuCompiler(options, opt_level=3, shared_libs=[support_lib])

        # Compile
        engine = compiler.compile_and_jit(mlir_nvgpu_module)
        return engine


def matmul(input_type=np.float16,
           output_type=np.float32,
           M=128,
           N=128,
           K=128,
           BLOCK_M=128,
           BLOCK_N=128,
           BLOCK_K=64,
           use_warp_specilization=True,
           dryrun=False,
           max_num_stages=3,
           print_results=False,
           no_verify=False):
    # Print the configuration
    ity = "f16" if input_type == np.float16 else "f32"
    oty = "f16" if output_type == np.float16 else "f32"
    gemmty = "Warp Specilization" if use_warp_specilization else "Multistage"
    print("===-- Running GEMM " + gemmty + " " + oty + " += " + ity + " * " + ity + ", Size " + str(M) + "x" + str(N) +
          "x" + str(K) + ", Tile " + str(BLOCK_M) + "x" + str(BLOCK_N) + "x" + str(BLOCK_K) + ", stages " +
          str(max_num_stages) + " --===")

    # Build and compile matmul
    engine = generate_matmul(input_type, output_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, use_warp_specilization,
                             dryrun, max_num_stages)

    # Allocate matrices and invoke the matmul
    c = np.zeros((M, N), output_type)
    a = np.random.randn(M, K).astype(input_type)
    b = np.random.randn(K, N).astype(input_type)
    mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
    mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
    mem_c = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(c)))                              
    kernelName = "mlir_matmul_warpspecialized" if use_warp_specilization else "mlir_matmul_multistage"
    for i in range(10):
        engine.invoke(kernelName, mem_a, mem_b, mem_c)

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    if print_results:
        print(c)
    if not no_verify:
        ref = a.astype(input_type) @ b.astype(input_type)
        if print_results:
            print(ref)
        np.testing.assert_allclose(c, ref, rtol=5e-03, atol=1e-01)

    print("PASS ")


# GEMM Multistage f32 += f16 * f16
matmul(np.float16, np.float32, 4096, 4096, 4096, max_num_stages=7, use_warp_specilization=False, no_verify=True)
matmul(np.float16, np.float32, 4096, 4096, 4096, max_num_stages=7, use_warp_specilization=True, no_verify=True)
