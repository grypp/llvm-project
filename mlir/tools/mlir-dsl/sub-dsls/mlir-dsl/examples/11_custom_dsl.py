# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Building a sub-DSL: subclass the base, pick plugins, name it, extend its boundary.

`mlir.dsl` is a base layer and a sub-DSL is a class. `MlirDSL` with another
`plugins` list keeps everything else (a CPU-only DSL is `plugins = []`), a
`BaseDSL` subclass picks its own `name`, which is the prefix of its environment
variables, and assembles itself from plugins (the `ScfASTPreprocessorPlugin` brings
native control flow), and `register_jit_arg_adapter` teaches the call boundary
a host type the DSL has never seen. `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` prints the
traced IR of the `MlirDSL`-based DSL; the renamed one listens to `MY_DSL_DRYRUN`.
"""

import numpy as np

import mlir.mlir_dsl as m
from mlir import execution_engine, passmanager


# (a) The plugin list is the whole surface a DSL adds to the base: an empty
# list is a CPU-only DSL (no `@kernel`, no gpu passes), the rest is inherited.
class CpuDSL(m.MlirDSL):
    plugins = []


# (b) A DSL that is not `MlirDSL`: `name` is its environment prefix, so it
# reads `MY_DSL_DRYRUN`, `MY_DSL_CACHE_DIR`, ... and ignores `MLIR_DSL_*`. It is
# assembled from plugins alone: the `ScfASTPreprocessorPlugin` is the AST preprocessor plugin
# (the preprocessor and the executors that stage native `for`/`if`/`while`),
# here with `closure_check=False` so nested functions may capture variables
# inside staged regions. Without a AST preprocessor plugin a DSL cannot preprocess and
# its bodies use the explicit `m.for_`/`m.if_` builders.
class MyDSL(m.BaseDSL):
    plugins = [m.ScfASTPreprocessorPlugin(closure_check=False)]
    _jit_arg_adapter_scope = "mlir"

    def __init__(self):
        super().__init__(
            name="MY_DSL",
            dsl_package_name=["mlir", "dsl"],
            compiler_provider=m.Compiler(passmanager, execution_engine),
            pass_sm_arch_name="cubin-chip",
            preprocess=True,
        )


# (c) An extension point: a host class the base knows nothing about ...
class Image:
    def __init__(self, height, width):
        self.pixels = np.zeros((height, width), dtype=np.float32)


# ... gets an adapter. It runs once per call, at the boundary, and hands the
# compiled function a `Pointer[Float32]` over the pixels (`keepalive` pins the
# owner for the call). Registered once, it serves every DSL in the process.
@m.register_jit_arg_adapter(Image)
def adapt_image(img):
    pixels = img.pixels
    return m.Pointer(pixels.ctypes.data, dtype=m.Float32, kind="host", keepalive=pixels)


@CpuDSL.jit
def fill(img: m.Pointer[m.Float32], n: m.Int32, value: m.Float32):
    # The body only ever sees the pointer: the annotation is checked against
    # what the adapter returned, the size travels as its own argument.
    for i in range(n):
        img[i] = value


@MyDSL.jit
def scale(a: m.Int32, n: m.Int32) -> m.Int32:
    def bump(x):
        return x + a  # captures `a`: allowed by this DSL's preprocessor plugin

    acc = m.Int32(1)
    for i in range(n):  # the AST preprocessor plugin stages this as scf.for
        acc = bump(acc)
    return acc * 2


def check(label, dsl, actual, expected):
    # A DSL under its own `<PREFIX>_DRYRUN` only traces: nothing to compare.
    if dsl.envar.dryrun:
        print(f"{label}: traced only ({dsl.name}_DRYRUN is set)")
        return
    assert actual == expected, (actual, expected)
    print(f"{label}: {actual}")


def main():
    cpu, mine = CpuDSL(), MyDSL()
    print("CpuDSL plugins:", [p.name for p in cpu.plugins])
    print("MlirDSL plugins:", [p.name for p in m.MlirDSL().plugins])
    print("MLIR_DSL_DRYRUN:", cpu.envar.dryrun, "MY_DSL_DRYRUN:", mine.envar.dryrun)

    img = Image(2, 3)
    fill(img, img.pixels.size, 2.5)  # an `Image` where a `Pointer` is expected
    check("fill through the Image adapter", cpu, img.pixels.tolist(), [[2.5] * 3] * 2)
    print("MyDSL plugins:", [p.name for p in mine.plugins])
    check("scale under MY_DSL", mine, scale(5, 3), 32)  # (1 + 5 + 5 + 5) * 2

    @CpuDSL.kernel
    def device_only(n: m.Int32):
        pass

    try:  # without the gpu plugin `@kernel` is a diagnostic, not a crash
        device_only(1)
    except m.DSLUserCodeError as e:
        print("CpuDSL has no @kernel:", e.diag_id.name)
        assert e.diag_id is m.DiagId.CALL_PLUGIN_REQUIRED
    print("Building a sub-DSL: passed")


if __name__ == "__main__":
    main()
