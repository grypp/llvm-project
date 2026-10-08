# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Jit: decorator plugins adding ``@jit``, the host function, to a DSL.

A ``DecoratorPlugin`` already behaves like ``@jit`` by default (a call from
Python traces, compiles and runs; a call inside a trace inlines the body);
what a jit plugin adds is the entry op and the result ABI. ``func.py`` ships
``Jit``: a ``func.func`` with ``llvm.emit_c_interface`` whose results travel
back in one ``!llvm.struct`` read through ``ctypes``. A DSL over another entry
op (another ABI, another dialect) writes its own ``DecoratorPlugin`` here.
"""
