# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``adapters`` family: the host boundary in both directions
(``core.plugin.AdapterPlugin``). Inbound, ``dlpack`` adapts anything speaking
the DLPack protocol, numpy arrays and torch tensors included, through the
``_mlirDslDlpack`` extension built from ``dlpack/csrc``; the core registers no
host buffer type. Outbound, ``tvm_ffi``
exposes a compiled function as a ``tvm_ffi.Function`` when
``<PREFIX>_ENABLE_TVM_FFI`` is set. Each needs its own optional dependency and
nothing imports one unless a DSL lists it.
"""
