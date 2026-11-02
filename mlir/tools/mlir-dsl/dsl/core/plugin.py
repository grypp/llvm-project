# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plugin lifecycle and the decorator entry protocol.

A DSL names the decorators it provides in a Plugins record. The entry plugin
creates a function op and its terminator; BaseDSL traces the Python body
inside that function. This layer has no dependency on a particular dialect.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ... import ir
from .common import DSLRuntimeError, DSLUserCodeError, in_trace
from .diagnostics import DiagId

if TYPE_CHECKING:
    from .dsl import BaseDSL


# ===----------------------------------------------------------------------===
# Plugin Lifecycle
# ===----------------------------------------------------------------------===


class Plugin:
    """Lifecycle shared by plugins installed into a concrete DSL instance."""

    role: ClassVar[str] = ""
    name: str = ""
    dsl: BaseDSL | None = None

    @classmethod
    def available(cls) -> bool:
        return True

    def install(self, dsl: BaseDSL) -> None:
        self.dsl = dsl

    def register_dialects(self, context: ir.Context) -> None:
        """Register any out-of-tree dialects the plugin's builders need."""

    def _unsupported(self, what: str) -> Any:
        raise DSLRuntimeError(
            f"`{type(self).__name__}` ({self.role or 'plugin'}) does not implement `{what}`"
        )


class DecoratorPlugin(Plugin):
    """A decorator and the function-entry operations its Python body uses."""

    role: ClassVar[str] = "decorators"
    decorator_name: ClassVar[str] = "jit"

    def decorators(self, dsl_cls: type) -> dict[str, Callable[..., Any]]:
        """Map the plugin's decorator name to a class-bound decorator."""
        return {self.decorator_name: dsl_cls.make_decorator(self.decorator_name)}

    def _dispatch(self, dsl: BaseDSL, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to ``launch`` inside a trace, or ``call`` outside one."""
        if in_trace():
            return self.launch(dsl, func, *args, **kwargs)
        return self.call(dsl, func, *args, **kwargs)

    def call(self, dsl: BaseDSL, func: Any, *args: Any, **kwargs: Any) -> Any:
        return dsl.run(self, func, *args, **kwargs)

    def launch(self, dsl: BaseDSL, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Inline a decorated helper into the active trace."""
        return func(*args, **kwargs)

    def generate_func_op(
        self, name: str, arg_types: list[Any], arg_attrs: list[Any], loc: Any = None
    ) -> tuple[Any, ir.Block]:
        return self._unsupported("generate_func_op")

    def generate_return(self, op: Any, values: list[Any], loc: Any = None) -> None:
        self._unsupported("generate_return")

    def decorate(self, func: Any, **options: Any) -> Any:
        """Process decorator options and return the function to wrap.

        Runs on the class record's plugin, before the DSL is instantiated.
        The function already carries its DSL class, decorator kind and source
        location. The default implementation accepts no options.
        """
        if options:
            unknown = sorted(options)
            raise DSLUserCodeError(
                DiagId.CALL_ARGUMENTS,
                function_name=f"@{func._decorator_kind}",
                detail=f"no option is named `{unknown[0]}`",
            )
        return func

    def wrap(self, func: Any, wrapper: Any) -> Any:
        """Customize the decorated wrapper before it is returned to the user."""
        return wrapper


# ===----------------------------------------------------------------------===
# Plugin Configuration
# ===----------------------------------------------------------------------===


@dataclass(frozen=True)
class Plugins:
    """A DSL's decorator plugins, copied when an instance installs them."""

    decorators: Sequence[DecoratorPlugin] = ()

    def __post_init__(self) -> None:
        if isinstance(self.decorators, (str, bytes)) or not isinstance(
            self.decorators, Sequence
        ):
            raise DSLRuntimeError("Plugins.decorators takes a sequence of plugins")
        plugins = tuple(self.decorators)
        seen: set[str] = set()
        for plugin in plugins:
            if not isinstance(plugin, DecoratorPlugin):
                raise DSLRuntimeError(
                    "Plugins.decorators takes DecoratorPlugin instances"
                )
            kind = plugin.decorator_name
            if not kind:
                raise DSLRuntimeError("a decorator plugin needs a decorator_name")
            if kind in seen:
                raise DSLRuntimeError(f"a record has more than one @{kind} decorator")
            seen.add(kind)
        object.__setattr__(self, "decorators", plugins)

    def decorator(self, decorator_name: str) -> DecoratorPlugin | None:
        return next(
            (p for p in self.decorators if p.decorator_name == decorator_name), None
        )

    def __iter__(self):
        return iter(self.decorators)

    def resolve(self) -> tuple[Plugins, dict[str, str]]:
        kept = []
        unavailable = {}
        for plugin in self.decorators:
            if type(plugin).available():
                kept.append(copy.copy(plugin))
            else:
                unavailable[f"decorators[{plugin.name}]"] = type(plugin).__name__
        return Plugins(decorators=kept), unavailable
