# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Building a sub-DSL: subclass the base, name its plugins, name it, extend its boundary.

`mlir.dsl` is a base layer and a sub-DSL is a class that assembles itself
through one `Plugins` record: a plugin per core role (`type_ops`, `func_entry`,
`ast_preprocessor`, `compiler`) and any number per family (`decorators`, which
add decorators such as `@kernel` and their launchers; `adapters`, the host
boundary: objects becoming arguments, the entry exposed as another ABI). The
core knows one decorator, `@jit`; `@kernel` is the gpu kernels plugin's.
`MlirTestDSL` with a changed record keeps everything else (a CPU-only DSL drops
its families); a `BaseDSL` subclass picks its own `name`,
which is the prefix of its environment variables, and names its plugins from
scratch; `register_jit_arg_adapter` teaches the call boundary a host type the
DSL has never seen. A dialect the DSL only emits ops from (`math`, your own)
is a module, not a plugin. `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` prints the
traced IR of the `MlirTestDSL`-based DSL; the renamed one listens to
`MY_DSL_DRYRUN`.
"""

from dataclasses import replace

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.plugins.ast_preprocessor import scf
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.func_entry import func
from mlir.dsl.plugins.type_ops import arith, llvm, vector
from mlir.dsl.plugins.type_ops import TypeOps


# (a) A variant of an existing DSL is its record with a change: dropping the
# two families gives a CPU-only DSL (no gpu kernels plugin, so `@kernel` is
# a diagnostic; no tensor adapters, no TVM-FFI export); the types, the entry,
# the compiler and the syntax are inherited.
class CpuDSL(m.MlirTestDSL):
    plugins = replace(m.MlirTestDSL.plugins, decorators=(), adapters=())


# (b) A DSL that is not `MlirTestDSL`: `name` is its environment prefix, so it
# reads `MY_DSL_DRYRUN`, `MY_DSL_CACHE_DIR`, ... and ignores `MLIR_DSL_*`. It
# names every plugin itself: `TypeOps(scalars=arith, vectors=vector, memory=llvm)` makes the types plain MLIR scalars
# `i32`/`f32` with `arith`/`math`/`vector`/`llvm` ops, `func.Entry` builds the
# `func.func` host entry, `execution_engine.Compiler` lowers and runs, and
# `scf.ASTPreprocessor` is its `ast_preprocessor` (the preprocessor and the
# executors that stage native `for`/`if`/`while`), here with
# `closure_check=False` so nested functions may capture variables inside staged
# regions. Without an `ast_preprocessor` a DSL cannot preprocess and its bodies
# use the explicit `m.for_`/`m.if_` builders. The pass list is the DSL's own:
# here the test DSL's `LOWER_TO_LLVM`; no plugin publishes passes.
# `dsl_package_name` is the package the rewrite imports for `and_`/`or_`/
# `not_`/`as_ir_value`, so a sub-DSL of its own names its own namespace, which
# re-exports them from `scf`.
class MyDSL(m.BaseDSL):
    plugins = m.Plugins(
        type_ops=TypeOps(scalars=arith, vectors=vector, memory=llvm),
        func_entry=func.Entry(),
        ast_preprocessor=scf.ASTPreprocessor(closure_check=False),
        compiler=execution_engine.Compiler(),
    )

    def pipeline(self):
        return list(m.LOWER_TO_LLVM)

    def __init__(self):
        super().__init__(
            name="MY_DSL", dsl_package_name=["mlir", "mlir_dsl"], preprocess=True
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


def roles(dsl):
    # The roles a DSL fills are the record fields that hold a plugin.
    return [f for f in m.Plugins.ROLES if getattr(dsl.plugins, f) is not None]


def families(dsl):
    # The families are the record fields that hold any number of plugins.
    return {f: [p.name for p in getattr(dsl.plugins, f)] for f in m.Plugins.FAMILIES}


def check(label, dsl, actual, expected):
    # A DSL under its own `<PREFIX>_DRYRUN` only traces: nothing to compare.
    if dsl.envar.dryrun:
        print(f"{label}: traced only ({dsl.name}_DRYRUN is set)")
        return
    assert actual == expected, (actual, expected)
    print(f"{label}: {actual}")


def main():
    cpu, mine, base = CpuDSL(), MyDSL(), m.MlirTestDSL()
    print("CpuDSL roles:", roles(cpu), "families:", families(cpu))
    print("MlirTestDSL roles:", roles(base), "families:", families(base))
    print("MLIR_DSL_DRYRUN:", cpu.envar.dryrun, "MY_DSL_DRYRUN:", mine.envar.dryrun)

    img = Image(2, 3)
    fill(img, img.pixels.size, 2.5)  # an `Image` where a `Pointer` is expected
    check("fill through the Image adapter", cpu, img.pixels.tolist(), [[2.5] * 3] * 2)
    print("MyDSL roles:", roles(mine), "families:", families(mine))
    check("scale under MY_DSL", mine, scale(5, 3), 32)  # (1 + 5 + 5 + 5) * 2

    # `CpuDSL` inherits the `kernel` decorator the gpu kernels plugin installed
    # on `MlirTestDSL`, but its own record names no such plugin.
    @CpuDSL.kernel
    def device_only(n: m.Int32):
        pass

    try:  # without the gpu kernels plugin `@kernel` is a diagnostic, not a crash
        device_only(1)
    except m.DSLUserCodeError as e:
        print("CpuDSL has no @kernel:", e.diag_id.name)
        assert e.diag_id is m.DiagId.CALL_PLUGIN_REQUIRED
    print("Building a sub-DSL: passed")


if __name__ == "__main__":
    main()
