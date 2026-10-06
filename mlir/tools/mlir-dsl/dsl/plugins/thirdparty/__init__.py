# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Third-party integrations: ``pytorch`` (tensor arguments), ``dlpack`` (any
DLPack object as a pointer), ``tvm_ffi`` (the TVM-FFI ABI). Each needs its
own optional dependency and nothing imports one unless a DSL lists it."""
