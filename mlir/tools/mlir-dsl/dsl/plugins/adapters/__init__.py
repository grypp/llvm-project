# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``adapters`` family: the host boundary in both directions
(``core.plugin.AdapterPlugin``). Inbound, ``pytorch`` adapts ``torch.Tensor``
arguments and ``dlpack`` anything speaking the DLPack protocol (through the
``_mlirDslDlpack`` extension built from ``dlpack/csrc``). Outbound, ``tvm_ffi``
exposes a compiled function as a ``tvm_ffi.Function`` when
``<PREFIX>_ENABLE_TVM_FFI`` is set. Each needs its own optional dependency and
nothing imports one unless a DSL lists it.
"""
