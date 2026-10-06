# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dialect plugins: the IR worlds a DSL emits into and lowers.

``llvm`` is the core's default for the types (``arith``/``math``/``vector``
ops on builtin types, ``llvm`` memory), ``scf`` the default for control flow,
``gpu`` brings ``gpu``/``nvvm`` kernels and launches. A dialect plugin with an
``emitter`` replaces the defaults (a tile dialect).
"""
