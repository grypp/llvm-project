# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The test sub-DSL ``MlirTestDSL``: the ``mlir.dsl`` core assembled from the shipped plugins.

``MlirTestDSL`` is one ``BaseDSL`` subclass naming one ``Plugins`` record: the
builtin types over ``arith``/``math``/``vector``/``llvm`` ops, a ``func.func``
host entry, gpu kernels, Python control flow as ``scf``, the ``mlir`` pass
manager and execution engine as its compiler, and the ``dlpack`` adapter
(numpy arrays and torch tensors arrive through DLPack). A plugin whose dependency is absent is dropped from the
record at the first construction. ``MLIR_DSL_*`` is its environment prefix. A
sub-DSL of your own is the same shape with another record
(``examples/11_custom_dsl.py``).
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from .. import ir
from ..dsl.core.common import DSLUserCodeError
from ..dsl.core.diagnostics import DiagId, register_dsl_package
from ..dsl.core.dsl import BaseDSL
from ..dsl.core.plugin import Plugins
from ..dsl.plugins.ast_preprocessor import scf
from ..dsl.plugins.compiler import execution_engine
from ..dsl.plugins.adapters import dlpack
from ..dsl.plugins.decorators.jit import func
from ..dsl.plugins.decorators.kernels import gpu_plugin
from ..dsl.plugins.type_ops import UpstreamDialectTypeOps, arith, llvm, vector
from ..dsl.util.tree_utils import (
    is_leaf,
    is_staged_leaf,
    leaf_entry,
    register_container,
    register_leaf_protocol,
)

__all__ = ["LOWER_TO_LLVM", "MlirTestDSL", "compile", "jit", "kernel"]

#: The passes lowering what the test DSL emits (arith, math, vector and llvm ops
#: on builtin types, the ``scf`` control flow, the ``func.func`` entry) to the
#: LLVM dialect, in order. The sub-DSL owns its pipeline; no plugin publishes
#: passes. A DSL built from the same parts lists these in its ``pipeline()``.
LOWER_TO_LLVM: tuple[str, ...] = (
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-vector-to-llvm",
    "convert-arith-to-llvm",
    "convert-math-to-llvm",
    "convert-func-to-llvm",
    "reconcile-unrealized-casts",
)

# Frames of this package are DSL-internal when a diagnostic locates user code.
# Registered by the name it was imported under, so a copy installed next to
# relocated bindings registers its relocated name.
register_dsl_package(__package__)

# The test DSL's additions to the value model, registered once at import.
# (a) A value protocol: a class whose ``__ir_types__()`` and ``__ir_values__()``
# methods and ``__from_ir_values__(values)`` classmethod define its SSA view is
# a leaf. Its leaf-valued attributes, in definition order, are its host
# payload (marshalled one by one) and tell a staged instance from a host one;
# an optional ``__ir_attributes__()`` gives the argument attributes of its SSA
# values.
_VALUE_PROTOCOL = ("__ir_types__", "__ir_values__", "__from_ir_values__")


def _speaks_value_protocol(cls: type) -> bool:
    return all(hasattr(cls, name) for name in _VALUE_PROTOCOL)


def _leaf_attributes_of(value: Any) -> list[Any]:
    return [v for v in vars(value).values() if is_leaf(v)]


def _marshal_fields(value: Any) -> list[Any]:
    return [
        slot
        for field in _leaf_attributes_of(value)
        for slot in leaf_entry(type(field)).marshal(field)
    ]


def _value_attributes(value: Any) -> list[ir.Attribute]:
    attributes = getattr(value, "__ir_attributes__", None)
    if attributes is not None:
        return list(attributes())
    return [ir.DictAttr.get({}) for _ in value.__ir_types__()]


register_leaf_protocol(
    _speaks_value_protocol,
    ir_types=lambda p: list(p.__ir_types__()),
    ir_values=lambda v: list(v.__ir_values__()),
    from_ir_values=lambda p, vs: type(p).__from_ir_values__(list(vs)),
    marshal=_marshal_fields,
    staged=lambda v: any(is_staged_leaf(f) for f in _leaf_attributes_of(v)),
    attributes=_value_attributes,
)

# (b) ``typing.NamedTuple`` as a container: its fields are the children, named
# in paths and diagnostics; two namedtuple classes with the same fields are
# still different containers at a join.
register_container(
    lambda cls: issubclass(cls, tuple) and hasattr(cls, "_fields"),
    name="namedtuple",
    to_iterable=lambda x, path: (
        SimpleNamespace(kind="namedtuple", cls=type(x), fields=list(x._fields)),
        list(x),
    ),
    from_iterable=lambda metadata, children: metadata.cls._make(children),
)


class MlirTestDSL(BaseDSL):
    """The test sub-DSL: the shipped plugins in one record.

    The gpu kernels plugin, the compiler and the adapters are dropped when
    their dependency is absent (``available()``), so the same class traces on
    a machine without the gpu bindings or the execution engine.
    """

    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
        ast_preprocessor=scf.ASTPreprocessor(),
        compiler=execution_engine.Compiler(),
        # @jit over a func.func host entry, then @kernel over gpu.func.
        decorators=[func.Jit(), gpu_plugin.Kernels(chip_option="cubin-chip")],
        # Inbound: anything speaking DLPack (numpy arrays, torch tensors on the
        # host or a device) through the nanobind extension.
        adapters=[dlpack.DlpackPlugin()],
    )

    def pipeline(self) -> list[str]:
        """The gpu lowering first, when an architecture is set (a launch without
        one that is compiled is a ``gpu:CONFIG_MISSING_ARCH`` diagnostic; a dry run
        needs none), then the lowering
        of the upstream dialects the builtin types and the ``scf`` control flow
        emit."""
        passes: list[str] = []
        kernels = self.plugins.named("gpu")
        if kernels is not None and self.envar.arch:
            # MLIR's upstream gpu lowering (outlining, NVVM, gpu-module-to-binary)
            # for the architecture; the plugin only names the chip option.
            passes.append(
                f"gpu-lower-to-nvvm-pipeline{{{kernels.chip_option}={self.envar.arch}}}"
            )
        passes.extend(LOWER_TO_LLVM)
        return passes

    def __init__(self) -> None:
        super().__init__(name="MLIR_DSL")


jit = MlirTestDSL.jit
kernel = MlirTestDSL.kernel


def compile(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Compile a ``@jit`` function for ``args`` without running it.

    :param func: A function decorated with ``jit``
    :param args: Representative arguments; runtime values fix the argument
        types, Python values specialise the compiled function
    :return: The compiled function, callable with matching runtime arguments
    """
    if not hasattr(func, "_dsl_cls"):
        raise DSLUserCodeError(DiagId.CALL_MISSING_JIT_DECORATOR)
    return func(*args, compile_only=True, **kwargs)
