# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``func_entry`` role: plugins building the host entry of a ``@jit``
function, its return and the result slot the host reads
(``core.plugin.FuncEntryPlugin``).

``func.py`` ships ``Entry``: a ``func.func`` with ``llvm.emit_c_interface``
whose results travel back in one ``!llvm.struct`` read through ``ctypes``.
"""
