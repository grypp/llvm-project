# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics for function authoring and tracing."""

from enum import Enum


class DiagId(Enum):
    CALL_NOT_CALLABLE = "{decorator} requires a Python function, got {got}"
    CALL_ARGUMENTS = "{function_name}: {detail}"
    CALL_PLUGIN_REQUIRED = "{name} needs {plugin}. {fix}"
    SCOPE_UNBOUND_NAME_IN_TRACE = "{function_name}: unbound name {var}"
    TYPE_RETURN_MISMATCH = "A traced function must return None, got {got}{detail}"
