# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The AST preprocessor: rewrites native Python control flow into region functions.

``DSLPreprocessor`` parses the source of a decorated function and rewrites
``for`` loops into ``@loop_selector`` bodies, ``if``/``elif``/``else`` into
``@if_selector`` regions and ``while`` loops into ``@while_selector`` regions
. Each region decides at trace time whether it is staged or
runs as native Python; a ``for`` over a bare ``range`` is likewise dispatched
at trace time between a staged loop and a native Python loop. Ternaries, ``and``/``or``/``not``, comparison chains, ``assert``
and ``bool()`` are routed through the helpers of ``helpers``.

A ``ScopeManager`` tracks the names bound so far; ``analyze_region_variables``
classifies the names a region stores, mutates or invokes and the rewrite
threads the stored ones through the region (the write_args protocol). The
preprocessor is generic: the DSL supplies the executors behind the selectors
(``core.staging.Executor.set_functions``).
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
import types
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from itertools import chain
from types import ModuleType
from typing import Any, TypeVar

from .... import ir
from ...core.common import DSLRuntimeError, DSLUserCodeError
from ...core.diagnostics import DiagId
from ...util import profiler
from ...util.logger import log
from .helpers import register_deferred_for_error

__all__ = [
    "ControlFlowPolicy",
    "DSLPreprocessor",
    "OrderedSet",
    "Region",
    "ScopeManager",
    "SessionData",
]

_AstNode = TypeVar("_AstNode", bound=ast.AST)

# The module the rewritten code imports as ``__base_dsl__``: every helper the
# rewrite references (selectors, executors, ``materialize_for_iter``, ...)
# lives in the sibling ``helpers`` module of this package.
_AST_HELPERS_MODULE = f"{__package__}.helpers"

# A call ``<module>.<op>(...)`` whose ``<module>`` is one of the MLIR dialect
# modules receives its `Numeric` arguments downcast to raw ``ir.Value``.
_MLIR_DIALECTS_PACKAGE = f"{ir.__name__.rpartition('.')[0]}.dialects"


class ControlFlowPolicy(Enum):
    """Select whether control-flow nodes are staged or kept as native Python."""

    TRACE = auto()
    NATIVE = auto()


def _deepcopy_ast_root(node: _AstNode) -> _AstNode:
    """Copy an AST subtree for a speculative dispatch arm."""
    return deepcopy(node)


class _NativeForLocalCollector(ast.NodeVisitor):
    """Collect plain names bound by a native candidate loop.

    Runtime-dispatched loops place a speculative native arm in the same
    generated Python function as the staged arm. CPython determines locals
    for the whole function, so a store that exists only in that native arm
    must not make the corresponding user name local on the staged path.
    """

    def __init__(self) -> None:
        self.bound_names: set[str] = set()
        self.declared_names: set[str] = set()
        self.protected_names: set[str] = set()
        self.all_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.all_names.add(node.id)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound_names.add(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.declared_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.declared_names.update(node.names)

    def _protect_nested_scope(self, node: ast.AST) -> None:
        self.protected_names.update(
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        )

    def visit_For(self, node: ast.For) -> None:
        """Keep loop-target bindings visible after a selected native arm.

        Python publishes a ``for`` target in the enclosing function scope.
        This is also how generator-based DSL APIs such as ``for_`` return
        their generated operation results. Body-only definitions remain
        private so a speculative native arm cannot shadow staged-path names.
        """
        self._protect_nested_scope(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._protect_nested_scope(node.target)
        self.generic_visit(node)

    # Nested scopes own their names; keep every name they mention as it is.
    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _protect_nested_scope
    visit_ClassDef = visit_ListComp = visit_SetComp = _protect_nested_scope
    visit_DictComp = visit_GeneratorExp = _protect_nested_scope


class _NativeForLocalRenamer(ast.NodeTransformer):
    """Give isolated native-loop locals branch-private Python names."""

    def __init__(self, renamed_names: dict[str, str]) -> None:
        self.renamed_names = renamed_names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        replacement = self.renamed_names.get(node.id)
        if replacement is None:
            return node
        node.id = replacement
        return node


# Loop options transform_for_loop understands; any other range keyword keeps the
# trace-time dispatch (see DSLPreprocessor._literal_for_iter_kind), where the
# DSL ``range`` collects it into its ``options`` and the staged arm forwards it.
_STAGED_RANGE_KEYWORDS = frozenset({"unroll", "unroll_full"})


def _native_for_local_renames(
    node: ast.For, active_symbols: list[set[str]], suffix: int
) -> dict[str, str]:
    """Plan private names for locals of a speculative native loop arm.

    Only names absent from the enclosing DSL scope are private. Existing
    variables retain their spelling so the native arm can update them using
    the normal control-flow writeback protocol.
    """
    collector = _NativeForLocalCollector()
    collector.visit(node)
    active_names = set().union(*active_symbols) if active_symbols else set()
    local_names = (
        collector.bound_names
        - collector.declared_names
        - collector.protected_names
        - active_names
        - {"_"}
    )
    renamed_names: dict[str, str] = {}
    reserved_names = collector.all_names | active_names
    for name in sorted(local_names):
        replacement = f"_dsl_native_for_{suffix}_{name}"
        collision_suffix = 0
        while replacement in reserved_names:
            replacement = f"_dsl_native_for_{suffix}_{collision_suffix}_{name}"
            collision_suffix += 1
        renamed_names[name] = replacement
        reserved_names.add(replacement)
    return renamed_names


class _RegionLocalCollector(ast.NodeVisitor):
    """Collect the names a control-flow region binds at its own scope level.

    Records the first binding site of every name so a diagnostic can point at
    the birth of a region-local value. Nested functions, classes, lambdas and
    comprehensions own their bindings and are not entered; their own names are
    bindings of the region.
    """

    def __init__(self) -> None:
        self.bound: dict[str, ast.AST] = {}

    def _bind(self, name: str, node: ast.AST) -> None:
        if name != "_" and name not in self.bound:
            self.bound[name] = node

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._bind(node.id, node)

    def _bind_definition(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        self._bind(node.name, node)

    def _skip_nested_scope(self, node: ast.AST) -> None:
        return

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _bind_definition
    visit_Lambda = visit_ListComp = visit_SetComp = _skip_nested_scope
    visit_DictComp = visit_GeneratorExp = _skip_nested_scope

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._bind(node.name, node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.partition(".")[0], node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name, node)


class _RegionReadFinder(ast.NodeVisitor):
    """Find the first read of each given name outside one control-flow node."""

    def __init__(self, names: Iterable[str], skip: ast.AST) -> None:
        self.names = set(names)
        self.skip = skip
        self.reads: dict[str, ast.Name] = {}

    def visit(self, node: ast.AST) -> Any:
        if node is self.skip:
            return None
        return super().visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in self.names
            and node.id not in self.reads
        ):
            self.reads[node.id] = node


class OrderedSet:
    """
    A deterministic set implementation for ordered operations.
    """

    def __init__(self, iterable: Iterable[str] | None = None) -> None:
        self._dict: dict[str, None] = dict.fromkeys(iterable or [])

    def add(self, item: str) -> None:
        self._dict[item] = None

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __contains__(self, item: object) -> bool:
        return item in self._dict

    def __and__(self, other: OrderedSet) -> OrderedSet:
        return OrderedSet(key for key in self._dict if key in other)

    def __or__(self, other: OrderedSet) -> OrderedSet:
        new_dict = self._dict.copy()
        new_dict.update(dict.fromkeys(other))
        return OrderedSet(new_dict)

    def __sub__(self, other: OrderedSet) -> OrderedSet:
        return OrderedSet(key for key in self._dict if key not in other)

    def __bool__(self) -> bool:
        return bool(self._dict)

    def intersections(self, others: list[set[str]]) -> OrderedSet:
        """The elements of this set that appear in at least one of *others*, in order."""
        result = OrderedSet()
        for key in self._dict:
            for other in reversed(others):
                if key in other:
                    result.add(key)
                    break
        return result


@dataclass
class ScopeManager:
    """The names and callables bound so far, per scope, during the traversal.

    ``scopes`` and ``callables`` are stacks of name sets: a function pushes a
    local scope on both (``enter_local_scope``), a control-flow region pushes a
    scope whose bindings are discarded on exit (``enter_control_flow_scope``).
    ``local_scope_indices`` remembers which entries of ``scopes`` are function
    scopes, for the closure write-back checks.
    """

    scopes: list[set[str]]
    callables: list[set[str]]
    local_scope_indices: list[int] = field(default_factory=list, init=False)

    @classmethod
    def create(cls) -> ScopeManager:
        return cls([], [])

    def restore_from(self, snapshot: ScopeManager) -> None:
        """Restore a snapshot without replacing this manager object."""
        if type(self) is not type(snapshot):
            raise DSLRuntimeError("scope manager snapshots must have the same type")
        for state_field in fields(self):
            setattr(self, state_field.name, getattr(snapshot, state_field.name))

    def add_to_scope(self, name: str) -> None:
        if name == "_":
            return
        self.scopes[-1].add(name)

    def add_to_callables(self, name: str) -> None:
        if not self.callables:
            return
        self.callables[-1].add(name)

    def get_active_symbols(self) -> list[set[str]]:
        return self.scopes.copy()

    def get_active_callables(self) -> list[set[str]]:
        return self.callables.copy()

    def current_function_owns(self, name: str) -> bool:
        """Return whether *name* belongs to the innermost function scope."""
        return (
            bool(self.local_scope_indices)
            and name in self.scopes[self.local_scope_indices[-1]]
        )

    def enclosing_function_owns(self, name: str) -> bool:
        """Return whether *name* belongs to an enclosing function scope."""
        return any(
            name in self.scopes[index] for index in self.local_scope_indices[:-1]
        )

    @contextlib.contextmanager
    def enter_local_scope(self) -> Generator[None, None, None]:
        """Enter a new local variable and callable scope.

        This is conceptually Python's local scope, such as within a function
        or class definition: a new, empty set is pushed onto both the variable
        and the callable stack and popped again on exit.
        """
        self.scopes.append(set())
        self.callables.append(set())
        self.local_scope_indices.append(len(self.scopes) - 1)
        try:
            yield
        finally:
            self.local_scope_indices.pop()
            self.scopes.pop()
            self.callables.pop()

    @contextlib.contextmanager
    def enter_control_flow_scope(
        self, *, isolate_callables: bool = False
    ) -> Generator[None, None, None]:
        """Enter a dynamic control-flow symbol scope.

        Variables introduced in the region are discarded on exit. Callable
        definitions normally remain function-scoped; speculative dispatch
        arms pass ``isolate_callables=True`` so their definitions stay visible
        within that arm but are discarded before tracing another candidate.
        """
        self.scopes.append(set())
        if isolate_callables:
            self.callables.append(set())
        try:
            yield
        finally:
            if isolate_callables:
                self.callables.pop()
            self.scopes.pop()


class Region:
    """
    Context manager for handling regions during AST transformations.

    A region collects the statements generated while a body (a loop, an arm of
    a conditional) is visited. A region that holds statements is pushed onto
    the session's ``region_stack`` on entry, so an expression visit can hoist
    statements (the arm blocks of a ternary) in front of the statement being
    visited. New statements go to ``owning_node._new_value`` when an owning
    node is given, else to the ``new_value`` list.

    :param owning_node: The node whose list field is being rebuilt
    :param new_value: The list collecting a body that is built from scratch
    :param holds_statements: Whether the owning node's list holds statements;
        ``generic_visit`` passes it per field (``With.items`` or
        ``Assign.targets`` are expression lists and must not receive hoisted
        statements). ``None`` falls back to "the owner is a statement".
    """

    def __init__(
        self,
        session_data: SessionData,
        *,
        owning_node: ast.AST | None = None,
        new_value: list[ast.stmt] | None = None,
        holds_statements: bool | None = None,
    ) -> None:
        self.session_data = session_data
        self.owning_node = owning_node
        self.new_value = new_value
        self.holds_statements = holds_statements

    @property
    def _collects_statements(self) -> bool:
        """Whether this region is a statement list (and so joins the stack)."""
        if self.new_value is not None:
            return True
        if self.holds_statements is not None:
            return self.holds_statements
        return isinstance(self.owning_node, ast.stmt)

    def __enter__(self) -> Region:
        if self._collects_statements:
            self.session_data.region_stack.append(self)
        if self.owning_node is not None:
            self.owning_node._new_value = []  # type: ignore[attr-defined]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self._collects_statements:
            self.session_data.region_stack.pop()
        if self.owning_node is not None:
            delattr(self.owning_node, "_new_value")

    def append_new_stmts(self, stmts: list[ast.stmt]) -> None:
        """Append statements to the region's collection."""
        if self.owning_node is not None:
            self.owning_node._new_value.extend(stmts)  # type: ignore[attr-defined]
        elif self.new_value is not None:
            self.new_value.extend(stmts)
        else:
            raise DSLRuntimeError("Region has neither an owning node nor a target list")


@dataclass
class SessionData:
    """The per-function state of one preprocessing session.

    Created by ``DSLPreprocessor.get_session``; ``counter`` numbers the
    generated functions and temporaries, the rest tracks the traversal.
    """

    counter: int = 0  # Unique function names for multiple loops
    scope_manager: ScopeManager = field(default_factory=ScopeManager.create)
    function_name: str = "<unknown function>"
    class_name: str | None = None
    file_name: str = "<unknown filename>"
    function_globals: dict[str, Any] | None = None
    import_top_module: bool = False
    region_stack: list[Region] = field(default_factory=list)
    generator_targets: list[str] = field(default_factory=list)
    lambda_args: list[str] = field(default_factory=list)
    captured_control_flow_writebacks: list[set[str]] = field(default_factory=list)
    control_flow_policies: list[ControlFlowPolicy] = field(
        default_factory=lambda: [ControlFlowPolicy.TRACE]
    )
    # The function, or speculative loop arm, whose statements a staged region
    # may publish names into; the innermost one bounds the escape analysis of
    # region-local names (``DSLPreprocessor._if_born_locals``).
    enclosing_bodies: list[ast.AST] = field(default_factory=list)

    @contextlib.contextmanager
    def _set_attr(self, attr: str, value: str) -> Generator[None, None, None]:
        """Temporarily set ``self.<attr>`` to ``value``, restoring it on exit."""
        old_value = getattr(self, attr)
        setattr(self, attr, value)
        try:
            yield
        finally:
            setattr(self, attr, old_value)

    def set_current_class_name(
        self, class_name: str
    ) -> contextlib.AbstractContextManager[None]:
        return self._set_attr("class_name", class_name)

    def set_current_function_name(
        self, function_name: str
    ) -> contextlib.AbstractContextManager[None]:
        return self._set_attr("function_name", function_name)

    @contextlib.contextmanager
    def control_flow_policy(
        self, policy: ControlFlowPolicy
    ) -> Generator[None, None, None]:
        """Temporarily select how nested control-flow nodes are transformed."""
        self.control_flow_policies.append(policy)
        try:
            yield
        finally:
            popped = self.control_flow_policies.pop()
            if popped is not policy or not self.control_flow_policies:
                raise DSLRuntimeError("control-flow policy stack out of balance")

    @property
    def current_control_flow_policy(self) -> ControlFlowPolicy:
        return self.control_flow_policies[-1]

    @contextlib.contextmanager
    def enclosing_body(self, node: ast.AST) -> Generator[None, None, None]:
        """Make *node* the body that bounds region-local escape analysis."""
        self.enclosing_bodies.append(node)
        try:
            yield
        finally:
            self.enclosing_bodies.pop()

    @contextlib.contextmanager
    def collect_captured_control_flow_writebacks(
        self,
    ) -> Generator[set[str], None, None]:
        """Collect outlined writebacks that target an enclosing closure."""
        writebacks: set[str] = set()
        self.captured_control_flow_writebacks.append(writebacks)
        try:
            yield writebacks
        finally:
            self.captured_control_flow_writebacks.pop()

    def record_captured_control_flow_writeback(self, name: str) -> None:
        """Record a generated writeback only when it changes closure scope."""
        if self.captured_control_flow_writebacks:
            self.captured_control_flow_writebacks[-1].add(name)


def _create_module_attribute(
    func_name: str,
    *,
    use_base_dsl: bool = True,
    submodule_name: str | None = None,
    lineno: int | None = None,
    col_offset: int | None = None,
) -> ast.Attribute:
    """Create the AST of a qualified attribute access through a runtime alias.

    The rewritten function imports two aliases: ``__base_dsl__`` is this
    package's ``helpers`` module (selectors, executors, loop helpers);
    ``__module_dsl__`` is the client DSL package named by
    ``DSLPreprocessor.client_module_name`` (``and_``, ``or_``, ``not_``,
    ``as_ir_value``), used when ``use_base_dsl`` is False.

    :param func_name: The attribute or function name to access
    :param use_base_dsl: Resolve against ``__base_dsl__`` (default) or ``__module_dsl__``
    :param submodule_name: An optional submodule of the aliased module
    :param lineno: The line of the source construct the node stands for
    :param col_offset: The column of that construct
    :return: The ``ast.Attribute`` node, located at the given point
    """

    # The location is one point, not the source node's range: copying the
    # range would make every traceback line of the helper span the construct.
    def set_location(
        node: ast.expr, lineno: int | None, col_offset: int | None
    ) -> None:
        if lineno is None or col_offset is None:
            return
        node.lineno = lineno
        node.end_lineno = lineno
        node.col_offset = col_offset
        node.end_col_offset = col_offset

    base: ast.expr = ast.Name(
        id="__base_dsl__" if use_base_dsl else "__module_dsl__", ctx=ast.Load()
    )
    set_location(base, lineno, col_offset)
    if submodule_name:
        base = ast.Attribute(value=base, attr=submodule_name, ctx=ast.Load())
        set_location(base, lineno, col_offset)
    result = ast.Attribute(value=base, attr=func_name, ctx=ast.Load())
    set_location(result, lineno, col_offset)
    return result


_ComprehensionT = TypeVar(
    "_ComprehensionT", ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp
)


class DSLPreprocessor(ast.NodeTransformer):
    """The AST transformer behind the DSL decorators (``@jit``, ``@kernel``, ...).

    - ``for`` loops become ``@loop_selector`` body functions, ``if``/``elif``/
      ``else`` become ``@if_selector`` regions and ``while`` loops
      ``@while_selector`` regions; each returns the names it stores (the
      write_args protocol).
    - Ternaries, ``and``/``or``/``not``, comparison chains, ``assert``,
      ``bool()``, zero-argument ``super()`` and calls into MLIR dialect modules
      are routed through ``helpers`` and the client DSL package.
    - ``global`` is rejected, ``nonlocal`` must name a bound variable, ``_``
      cannot be read, and the DSL decorator must be the innermost one.

    :param client_module_name: The package the rewrite imports as
        ``__module_dsl__`` (the DSL's ``dsl_package_name`` when given, else the
        ``ast_preprocessor`` plugin's own package, where ``and_``/``or_``/
        ``not_``/``as_ir_value`` live), as path parts
    :param if_born_locals_escape: Whether a name first bound in an arm of a
        staged ``if`` and read after it is threaded through the region (seeded
        ``None``); when False such a read is ``SCOPE_REGION_LOCAL_ESCAPES``
    :param closure_check: Whether a region calling nested functions gets a
        ``closure_check`` call
    :param decorator_names: The decorators marking a function for the rewrite:
        the ``decorator_name`` of every decorator plugin of the DSL
        (``BaseDSL.__init__`` passes them); ``("jit",)`` by default
    """

    DECORATOR_FOR_STATEMENT = "loop_selector"
    DECORATOR_IF_STATEMENT = "if_selector"
    DECORATOR_WHILE_STATEMENT = "while_selector"
    IF_EXECUTOR = "if_executor"
    IFEXP_EXECUTOR = "ifexp_executor"
    WHILE_EXECUTOR = "while_executor"
    ASSERT_EXECUTOR = "assert_executor"
    IMPLICIT_DOWNCAST_NUMERIC_TYPE = "as_ir_value"
    COMPARE_EXECUTOR = "compare_executor"
    BUILTIN_REDIRECTOR = "redirect_builtin_function"
    BOOL_SHORT_CIRCUITS = "bool_short_circuits"

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """
        Copy of :meth:`ast.NodeTransformer.generic_visit` with support for inserting statements during expression visits.

        Every list field is visited inside a ``Region`` owned by *node*; a
        list of statements joins the region stack, so an expression visit may
        hoist statements in front of the statement being visited, while an
        expression list (``With.items``, ``Assign.targets``, a decorator list)
        leaves the hoisted statements to the enclosing statement list.
        """
        for field_name, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                holds_statements = bool(old_value) and isinstance(
                    old_value[0], ast.stmt
                )
                with Region(
                    self.session_data,
                    owning_node=node,
                    holds_statements=holds_statements,
                ):
                    for value in old_value:
                        if isinstance(value, ast.AST):
                            value = self.visit(value)
                            if value is None:
                                continue
                            elif not isinstance(value, ast.AST):
                                node._new_value.extend(value)  # type: ignore[attr-defined]
                                continue
                        node._new_value.append(value)  # type: ignore[attr-defined]
                    old_value[:] = node._new_value  # type: ignore[attr-defined]
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field_name)
                else:
                    setattr(node, field_name, new_node)
        return node

    def __init__(
        self,
        client_module_name: list[str],
        *,
        if_born_locals_escape: bool = True,
        closure_check: bool = True,
        decorator_names: Iterable[str] = ("jit",),
    ) -> None:
        super().__init__()
        # Persistent state
        self.processed_functions: set[Callable[..., Any]] = set()
        self.client_module_name = client_module_name
        # The decorators marking a function for the rewrite (``@m.jit``,
        # ``@m.kernel``, a sub-DSL's own); a function carrying none is left as
        # it is.
        self.decorator_names: frozenset[str] = frozenset(decorator_names)
        # False: a region calling nested functions gets no ``closure_check`` call.
        self.closure_check: bool = closure_check
        # A name first bound inside an arm of a staged ``if`` and read after
        # the ``if`` is threaded through the region (seeded ``None``) when
        # True; when False such a read is a preprocess-time error.
        self.if_born_locals_escape: bool = if_born_locals_escape
        self._session_data: SessionData | None = None

    def _create_session_data(self) -> SessionData:
        return SessionData()

    def _start_session(self) -> None:
        """Start a new preprocessing session by initializing session data."""
        self._session_data = self._create_session_data()
        # Track processed functions per preprocessing run, not for the entire
        # lifetime of this preprocessor instance, so a later session never
        # skips transforming a function entirely.
        self.processed_functions = set()

    def _end_session(self) -> None:
        """End the current preprocessing session and clear session data."""
        self._session_data = None

    @contextlib.contextmanager
    def get_session(self) -> Generator[DSLPreprocessor, None, None]:
        try:
            self._start_session()
            yield self
        finally:
            self._end_session()

    @property
    def session_data(self) -> SessionData:
        if self._session_data is None:
            raise DSLRuntimeError(
                "no preprocessing session: use `with preprocessor.get_session()`"
            )
        return self._session_data

    def exec(
        self,
        function_name: str,
        original_function: Callable[..., Any],
        code_object: types.CodeType,
        exec_globals: dict[str, Any],
    ) -> Callable[..., Any] | None:
        """Execute the compiled rewrite and return the function it defines.

        :param function_name: The name the rewritten module binds
        :param original_function: The user's function (unused here; kept so a
            sub-DSL's override can consult it)
        :param code_object: The compiled transformed module
        :param exec_globals: The namespace to execute in (the function's
            globals plus closure values)
        :return: ``exec_globals[function_name]``, or ``None`` if the rewrite
            did not bind it
        """
        log().info(
            "ASTPreprocessor Executing transformed code for function [%s]",
            function_name,
        )
        exec(code_object, exec_globals)
        return exec_globals.get(function_name)

    @staticmethod
    def print_ast(transformed_tree: ast.AST | None = None) -> None:
        """Print the rewritten source (``MLIR_DSL_DEBUG=1`` calls this)."""
        print("#", "-" * 40, "Transformed AST", "-" * 40)
        unparsed_code = ast.unparse(transformed_tree)  # type: ignore[arg-type]
        print(unparsed_code)
        print("#", "-" * 40, "End Transformed AST", "-" * 40)

    def make_func_param_name(self, base_name: str, used_names: Iterable[str]) -> str:
        """Generate a unique parameter name that doesn't collide with existing names."""
        if base_name not in used_names:
            return base_name

        i = 0
        while f"{base_name}_{i}" in used_names:
            i += 1
        return f"{base_name}_{i}"

    @staticmethod
    def _default_arg_source_names(func_ast: ast.FunctionDef) -> dict[str, str]:
        """Map parameters to source-level names used as their defaults."""
        ast_defaults: dict[str, ast.expr] = {}
        all_args = func_ast.args.posonlyargs + func_ast.args.args
        offset = len(all_args) - len(func_ast.args.defaults)
        for i, default_node in enumerate(func_ast.args.defaults):
            ast_defaults[all_args[offset + i].arg] = default_node
        for kwarg, kw_default in zip(
            func_ast.args.kwonlyargs, func_ast.args.kw_defaults
        ):
            if kw_default is not None:
                ast_defaults[kwarg.arg] = kw_default
        return {
            param_name: default_node.id
            for param_name, default_node in ast_defaults.items()
            if isinstance(default_node, ast.Name)
        }

    def _inject_default_arg_values(
        self,
        function_pointer: Callable[..., Any],
        source_names: dict[str, str],
    ) -> None:
        """Inject default-argument values whose source-level names are unresolvable.

        When a decorated function uses ``_param=name`` where ``name`` is a local
        in an enclosing factory, ``exec()`` needs ``name`` in its namespace.
        We use ``inspect.signature`` for runtime default values and the
        already-parsed source-name map for the name each default references.
        """
        exec_globals = self.session_data.function_globals
        if exec_globals is None:
            return
        sig = inspect.signature(function_pointer)
        params_with_defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        if not params_with_defaults:
            return
        for param_name, default_val in params_with_defaults.items():
            source_name = source_names.get(param_name)
            if source_name is not None and source_name not in exec_globals:
                exec_globals[source_name] = default_val

    @profiler.timed("ast-build")
    def transform_function(
        self, func_name: str, function_pointer: Callable[..., Any]
    ) -> list[ast.stmt]:
        """Rewrite one decorated function into the statements of a module.

        Parses the source, re-bases line and column offsets onto the original
        file, checks the decorator, rewrites the body, prepends the runtime
        alias imports, strips the decorators and, when the function has free
        variables, wraps it in a factory that binds them.

        :param func_name: The function's name
        :param function_pointer: The function whose source is rewritten
        :return: The module-level statements, empty when the function was
            already processed in this session or carries no DSL decorator
        """
        if function_pointer in self.processed_functions:
            log().info(
                "ASTPreprocessor Skipping already processed function [%s]", func_name
            )
            return []

        # Step 1. Parse the given function
        try:
            parse_start = profiler.start("ast-build/parse")
            file_name = inspect.getsourcefile(function_pointer) or "<unknown>"
            lines, start_line = inspect.getsourcelines(function_pointer)
            raw_source = "".join(lines)
            dedented_source = textwrap.dedent(raw_source)
            tree = ast.parse(dedented_source, filename=file_name)
            # Bump the line numbers so they match the real source file
            ast.increment_lineno(tree, start_line - 1)
            # ``textwrap.dedent`` stripped a constant leading-whitespace prefix,
            # so ast column offsets are relative to the dedented source.  Shift
            # them back to the original file's columns so diagnostics underline
            # the right place.  The stripped width is identical on every non-blank
            # line; derive it from the first line that still has content.
            col_shift = 0
            for raw_line, dedented_line in zip(
                raw_source.split("\n"), dedented_source.split("\n")
            ):
                if dedented_line.strip():
                    col_shift = (len(raw_line) - len(raw_line.lstrip())) - (
                        len(dedented_line) - len(dedented_line.lstrip())
                    )
                    break
            if col_shift:
                for walked in ast.walk(tree):
                    if getattr(walked, "col_offset", None) is not None:
                        walked.col_offset += col_shift  # type: ignore[attr-defined]
                    if getattr(walked, "end_col_offset", None) is not None:
                        walked.end_col_offset += col_shift  # type: ignore[attr-defined]
            profiler.stop("ast-build/parse", parse_start)
        except (OSError, TypeError) as e:
            # No retrievable source (REPL / exec())
            raise DSLUserCodeError(DiagId.UNSUP_NO_SOURCE, func=func_name, cause=e)
        except Exception as e:
            raise DSLRuntimeError(f"Failed to parse function {func_name}", cause=e)

        # Step 1.2 Check the decorator
        if not self.check_decorator(tree.body[0]):
            log().info(
                "[%s] - Skipping function due to missing decorator",
                func_name,
            )
            return []

        self.processed_functions.add(function_pointer)
        log().info("ASTPreprocessor Transforming function [%s]", func_name)

        # Step 1.3 Inject default-argument values from enclosing scopes.
        # When a decorated function uses `_param=name` where `name` is a
        # local in the enclosing factory, exec() needs `name` in its
        # namespace.  We use the already-parsed AST to find source-level
        # names and inspect.signature to get runtime values.
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        self._inject_default_arg_values(
            function_pointer, self._default_arg_source_names(func_def)
        )

        # Step 2. Transform the function
        visit_start = profiler.start("ast-build/visit")
        transformed_tree = self.visit(tree)
        profiler.stop("ast-build/visit", visit_start)

        # Step 3. Import the runtime aliases: the client DSL package (only when
        # a rewrite needs it) and this package's helpers module.
        top_module_name = ".".join(self.client_module_name)
        import_stmts: list[ast.stmt] = []
        if self.session_data.import_top_module:
            import_stmts.append(
                ast.Import(
                    names=[ast.alias(name=top_module_name, asname="__module_dsl__")]
                )
            )
        import_stmts.append(
            ast.Import(
                names=[ast.alias(name=_AST_HELPERS_MODULE, asname="__base_dsl__")]
            )
        )

        assert len(transformed_tree.body) == 1
        assert isinstance(transformed_tree.body[0], ast.FunctionDef)
        transformed_tree.body[0].body = import_stmts + transformed_tree.body[0].body
        # Remove all decorators from top level function
        transformed_tree.body[0].decorator_list = []

        # Step 4. A function with free variables is wrapped in a factory that
        # declares them, so the rewritten body keeps them as closure cells
        # (their values come from exec_globals, see
        # ``BaseDSL._inject_closure_cells``):
        # def foo():
        #      free_var_0 = None
        #      free_var_1 = None
        #      def foo(args):
        #          ...
        #      return foo
        # foo = foo()
        free_vars = function_pointer.__code__.co_freevars

        if free_vars:
            assignments: list[ast.stmt] = [
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Constant(value=None),
                )
                for name in free_vars
            ]

            return_expr = [ast.Return(value=ast.Name(id=func_name, ctx=ast.Load()))]

            wrapper_fcn = ast.FunctionDef(
                name=func_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=assignments + transformed_tree.body + return_expr,
                decorator_list=[],
            )
            invoke = ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()), args=[], keywords=[]
            )
            assign = ast.Assign(
                targets=[ast.Name(id=func_name, ctx=ast.Store())], value=invoke
            )
            transformed_tree.body = [wrapper_fcn, assign]

        # Step 5. Fix up locations and return the transformed tree
        ast.fix_missing_locations(transformed_tree)
        return transformed_tree.body

    def _find_early_exit(self, tree: ast.AST, kind: str) -> tuple[ast.AST, str] | None:
        """Find an early exit owned by the given control-flow region.

        ``return`` and ``raise`` anywhere in the region (outside nested
        functions) are early exits; ``break``/``continue`` only when they
        belong to this loop (``kind`` is ``"for"``/``"while"``) and not to a
        loop nested inside it. An ``if`` region owns no ``break``/``continue``.

        :return: ``(node, "return" | "raise" | "break" | "continue")`` of the
            last early exit found, or ``None``
        """

        class EarlyExitChecker(ast.NodeVisitor):
            def __init__(self, kind: str) -> None:
                self.early_exit: tuple[ast.AST, str] | None = None
                self.kind = kind
                self.loop_nest_level = 0

            def _record(self, node: ast.AST, exit_type: str) -> None:
                self.early_exit = (node, exit_type)

            def visit_Return(self, node: ast.Return) -> None:
                self._record(node, "return")

            def visit_Raise(self, node: ast.Raise) -> None:
                self._record(node, "raise")

            def visit_Break(self, node: ast.Break) -> None:
                if self.loop_nest_level == 0 and self.kind != "if":
                    self._record(node, "break")

            def visit_Continue(self, node: ast.Continue) -> None:
                if self.loop_nest_level == 0 and self.kind != "if":
                    self._record(node, "continue")

            def visit_For(self, node: ast.For) -> None:
                self.loop_nest_level += 1
                self.generic_visit(node)
                self.loop_nest_level -= 1

            def visit_While(self, node: ast.While) -> None:
                self.loop_nest_level += 1
                self.generic_visit(node)
                self.loop_nest_level -= 1

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

        checker = EarlyExitChecker(kind)
        checker.generic_visit(tree)
        return checker.early_exit

    def check_early_exit(self, tree: ast.AST, kind: str) -> None:
        """Reject an early exit owned by the given control-flow region."""
        early_exit = self._find_early_exit(tree, kind)
        if early_exit is None:
            return
        offender, exit_type = early_exit
        where = f"`{self.session_data.function_name}`" + (
            f" in `{self.session_data.class_name}`"
            if self.session_data.class_name
            else ""
        )
        raise DSLUserCodeError(
            DiagId.UNSUP_EARLY_EXIT,
            filename=self.session_data.file_name,
            lineno=getattr(offender, "lineno", None),
            col_offset=getattr(offender, "col_offset", None),
            end_col_offset=getattr(offender, "end_col_offset", None),
            kind=exit_type,
            where=where,
        )

    def _handle_native_early_exit(
        self, node: ast.If | ast.While, early_exit: tuple[ast.AST, str]
    ) -> list[ast.stmt]:
        """Keep an ``if``/``while`` that owns an early exit as native Python.

        Outlining the statement into a region function would detach its
        ``return``/``raise`` (or the ``break``/``continue`` of the enclosing
        native loop) from what they exit, so the statement stays a Python one
        and its test is wrapped in ``early_exit_predicate(...)``: at trace time
        a Meta test is evaluated by Python, a staged one raises
        ``UNSUP_EARLY_EXIT`` at the exit's location. The arms are visited under
        the NATIVE policy, so nested regions without an early exit are staged
        as usual. Statements hoisted out of the test (ternary blocks) precede an
        ``if``; a ``while`` re-evaluates them per iteration through a
        ``while True:`` form.
        """
        offender, exit_type = early_exit
        hoisted: list[ast.stmt] = []
        with Region(self.session_data, new_value=hoisted):
            test = self.visit(node.test)
        assert isinstance(test, ast.expr)
        where = f"`{self.session_data.function_name}`" + (
            f" in `{self.session_data.class_name}`"
            if self.session_data.class_name
            else ""
        )
        guard = ast.copy_location(
            ast.Call(
                func=_create_module_attribute(
                    "early_exit_predicate",
                    lineno=node.test.lineno,
                    col_offset=node.test.col_offset,
                ),
                args=[test],
                keywords=[
                    ast.keyword(arg="kind", value=ast.Constant(value=exit_type)),
                    ast.keyword(arg="where", value=ast.Constant(value=where)),
                    ast.keyword(
                        arg="filename",
                        value=ast.Constant(value=self.session_data.file_name),
                    ),
                    *(
                        ast.keyword(
                            arg=key,
                            value=ast.Constant(value=getattr(offender, key, None)),
                        )
                        for key in ("lineno", "col_offset", "end_col_offset")
                    ),
                ],
            ),
            node.test,
        )
        with self.session_data.control_flow_policy(ControlFlowPolicy.NATIVE):
            body: list[ast.stmt] = []
            with Region(self.session_data, new_value=body):
                self._visit_stmts_into(node.body, body)
            orelse: list[ast.stmt] = []
            with Region(self.session_data, new_value=orelse):
                self._visit_stmts_into(node.orelse, orelse)
        if isinstance(node, ast.While) and hoisted:
            # ``while True: <hoisted>; if not guard: <else>; break; <body>``
            stop = ast.copy_location(
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=guard),
                    body=orelse + [ast.copy_location(ast.Break(), node)],
                    orelse=[],
                ),
                node,
            )
            node.test = ast.copy_location(ast.Constant(value=True), node.test)
            node.body = hoisted + [stop] + body
            node.orelse = []
            return [node]
        node.test = guard
        node.body = body
        node.orelse = orelse
        return hoisted + [node]

    def transform(
        self,
        original_function: Callable[..., Any],
        exec_globals: dict[str, Any],
    ) -> ast.Module:
        """
        Transforms the provided function using the preprocessor.
        Requires an active DSL preprocessor session.

        :param original_function: The function to transform.
        :param exec_globals: The globals dict for the function's module.
        """
        self.session_data.file_name = (
            inspect.getsourcefile(original_function) or "<unknown>"
        )
        self.session_data.function_globals = exec_globals
        transformed_tree = self.transform_function(
            original_function.__name__, original_function
        )
        self.session_data.function_globals = None
        unified_tree = ast.Module(body=transformed_tree, type_ignores=[])
        return ast.fix_missing_locations(unified_tree)

    def analyze_region_variables(
        self,
        node: ast.For | ast.If | ast.While,
        active_symbols: list[set[str]],
        active_callables: list[set[str]],
    ) -> tuple[list[str], int, list[str], list[str]]:
        """
        Analyze loop-carried and closure variables in a control-flow region.

        :return: ``(write_args, full_write_args_count, called_functions,
            mutated_names)``: the names stored in the region (in first-store
            order) followed by the receivers of method calls that are not also
            stored, both restricted to *active_symbols*; the length of the
            stored group; the called functions defined in an enclosing scope;
            and the write_args that are only ever the base of an attribute or
            subscript store (mutated in place, never rebound).
        """
        # we need orderedset to keep the insertion order the same. otherwise generated IR is different each time
        write_args = OrderedSet()
        invoked_args = OrderedSet()
        called_functions = OrderedSet()
        rebound_names: set[str] = set()
        store_base_names: set[str] = set()

        class RegionAnalyzer(ast.NodeVisitor):
            force_store = False

            def visit_Name(self, node: ast.Name) -> None:
                """
                Mark every store as write; a store through an attribute or
                subscript marks its base as mutated.
                """
                if isinstance(node.ctx, ast.Store):
                    write_args.add(node.id)
                    rebound_names.add(node.id)
                elif self.force_store:
                    write_args.add(node.id)
                    store_base_names.add(node.id)

            def visit_Subscript(self, node: ast.Subscript) -> None:
                saved_force_store = self.force_store
                if isinstance(node.ctx, ast.Store):
                    self.force_store = True
                self.visit(node.value)
                self.force_store = False
                self.visit(node.slice)
                self.force_store = saved_force_store

            def visit_Assign(self, node: ast.Assign) -> None:
                self.force_store = True
                for target in node.targets:
                    self.visit(target)
                self.force_store = False
                self.visit(node.value)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                self.force_store = True
                self.visit(node.target)
                self.force_store = False
                self.visit(node.value)

            @staticmethod
            def get_call_base(func_node: ast.expr) -> str | None:
                """The receiver name of a method call ``name.attr(...).attr(...)``, if any."""
                while isinstance(func_node, ast.Attribute):
                    if isinstance(func_node.value, ast.Name):
                        return func_node.value.id
                    func_node = func_node.value
                return None

            def visit_Call(self, node: ast.Call) -> None:
                base_name = RegionAnalyzer.get_call_base(node.func)

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    called_functions.add(func_name)

                # Classes are mutable by default. Mark them as write. If they are
                # dataclass(frozen=True), treat them as read in runtime.
                if base_name is not None and base_name not in ("self",):
                    invoked_args.add(base_name)

                self.generic_visit(node)

        analyzer = RegionAnalyzer()
        analyzer.visit(ast.Module(body=node.body, type_ignores=[]))
        if node.orelse:
            analyzer.visit(ast.Module(body=node.orelse, type_ignores=[]))

        # While's loop condition is executed n times, as loop body
        # So collect the variables used in the loop condition
        if isinstance(node, ast.While):
            analyzer.visit(ast.Module(body=node.test, type_ignores=[]))  # type: ignore[arg-type]

        # If arg is both write and invoke, remove from invoked_args
        invoked_args = invoked_args - write_args

        write_args_list: list[str] = list(write_args.intersections(active_symbols))
        invoked_args_list: list[str] = list(invoked_args.intersections(active_symbols))
        mutated_names: list[str] = [
            name
            for name in write_args_list
            if name in store_base_names and name not in rebound_names
        ]
        # The current function is tracked as callable so recursive references
        # resolve inside its body, but it is not an enclosing closure. Checking
        # the decorated runtime wrapper would inspect the decorator's captures
        # instead of the user's function.
        current_function_name = (
            self._session_data.function_name if self._session_data is not None else None
        )
        called_functions_list: list[str] = [
            name
            for name in called_functions.intersections(active_callables)
            if name != current_function_name
        ]
        return (
            write_args_list + invoked_args_list,
            len(write_args_list),
            called_functions_list,
            mutated_names,
        )

    # =============================================================================
    # Region-local names
    # =============================================================================

    @staticmethod
    def _region_bound_names(
        stmts: Sequence[ast.AST],
        active_symbols: list[set[str]],
        *,
        exclude: Iterable[str] = (),
    ) -> dict[str, ast.AST]:
        """Names first bound inside a region, with the node that binds each.

        A name already bound before the region (an *active symbol*) is not
        region-born: the region rebinds it and the write_args protocol carries
        it. The result keeps the region's first-store order.
        """
        collector = _RegionLocalCollector()
        for stmt in stmts:
            collector.visit(stmt)
        active_names = set().union(*active_symbols) if active_symbols else set()
        excluded = active_names | set(exclude)
        return {
            name: birth
            for name, birth in collector.bound.items()
            if name not in excluded
        }

    def _reads_outside_region(
        self, node: ast.AST, names: Iterable[str]
    ) -> dict[str, ast.Name]:
        """First read of each region-born name outside *node*.

        The search is bounded by the innermost enclosing body (the user's
        function, or the speculative loop arm being transformed); anything
        beyond it is the enclosing region's own concern.
        """
        if not self.session_data.enclosing_bodies:
            return {}
        finder = _RegionReadFinder(names, skip=node)
        finder.visit(self.session_data.enclosing_bodies[-1])
        return finder.reads

    def _raise_region_local_escapes(
        self, var: str, region: str, birth: ast.AST, read: ast.Name
    ) -> None:
        raise DSLUserCodeError(
            DiagId.SCOPE_REGION_LOCAL_ESCAPES,
            filename=self.session_data.file_name,
            lineno=getattr(read, "lineno", None),
            col_offset=getattr(read, "col_offset", None),
            end_col_offset=getattr(read, "end_col_offset", None),
            var=var,
            region=region,
            birth_file=self.session_data.file_name,
            birth_line=getattr(birth, "lineno", None),
        )

    def _check_region_locals_escape(
        self, node: ast.For | ast.While, region: str
    ) -> None:
        """Reject a read, after a loop, of a name first bound in its body.

        The body of a staged loop is compiled once and the native arm of a
        dispatched loop privatizes its body-born locals, so under both a
        body-born name never escapes the loop.
        """
        target_names: set[str] = set()
        if isinstance(node, ast.For):
            target_names = {
                child.id
                for child in ast.walk(node.target)
                if isinstance(child, ast.Name)
            }
        born = self._region_bound_names(
            [*node.body, *node.orelse],
            self.session_data.scope_manager.get_active_symbols(),
            exclude=target_names,
        )
        if not born:
            return
        reads = self._reads_outside_region(node, born)
        for name, read in reads.items():
            self._raise_region_local_escapes(name, region, born[name], read)

    def _if_born_locals(
        self, node: ast.If, active_symbols: list[set[str]]
    ) -> list[str]:
        """Names first bound in an arm of a staged ``if`` and read after it.

        With ``if_born_locals_escape`` these names join the region's write_args
        (seeded ``None`` by ``get_locals_or_none``) so the executor joins the
        arms' values; otherwise such a read is an error, the names being local
        to the arm that binds them.
        """
        born = self._region_bound_names([*node.body, *node.orelse], active_symbols)
        if not born:
            return []
        reads = self._reads_outside_region(node, born)
        if not reads:
            return []
        if not self.if_born_locals_escape:
            name, read = next(iter(reads.items()))
            self._raise_region_local_escapes(name, "if", born[name], read)
        return [name for name in born if name in reads]

    # =============================================================================
    # For loops
    # =============================================================================

    def extract_range_args(
        self, iter_node: ast.Call
    ) -> tuple[ast.expr, ast.expr, ast.expr]:
        """The visited ``(start, stop, step)`` of a range call, defaults filled in."""
        args = iter_node.args
        if not 1 <= len(args) <= 3:
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                what="`range(...)` with this number of positional arguments",
                detail=": call it as `range(stop)`, `range(start, stop)` or `range(start, stop, step)`, with loop options such as `unroll=` by keyword",
                filename=self.session_data.file_name,
                lineno=getattr(iter_node, "lineno", None),
                col_offset=getattr(iter_node, "col_offset", None),
                end_col_offset=getattr(iter_node, "end_col_offset", None),
            )
        start = args[0] if len(args) >= 2 else ast.Constant(value=0)
        stop = args[1] if len(args) >= 2 else args[0]
        step = args[2] if len(args) == 3 else ast.Constant(value=1)
        return self.visit(start), self.visit(stop), self.visit(step)

    @staticmethod
    def _keyword_map(iter_node: ast.Call) -> dict[str | None, ast.expr]:
        """Map a call's keyword arguments by name to their value AST nodes."""
        return {kw.arg: kw.value for kw in iter_node.keywords}

    def extract_unroll_args(self, iter_node: ast.Call) -> tuple[ast.expr, ast.expr]:
        keywords = self._keyword_map(iter_node)
        return (
            keywords.get("unroll", ast.Constant(value=-1)),
            keywords.get("unroll_full", ast.Constant(value=False)),
        )

    def extract_loop_options(self, iter_node: ast.Call) -> list[ast.keyword]:
        """Keywords of a range call forwarded to ``loop_selector`` unchanged.

        These are the ``**options`` of the DSL ``range``: every keyword that is
        not a positional bound and not one of ``_STAGED_RANGE_KEYWORDS``,
        including a ``**mapping`` splat.
        """
        return [
            keyword
            for keyword in iter_node.keywords
            if keyword.arg not in _STAGED_RANGE_KEYWORDS
        ]

    def _visit_stmts_into(self, stmts: list[ast.stmt], out: list[ast.stmt]) -> None:
        """Visit each statement and append its result to ``out``, flattening lists.

        Shared inner loop for the control-flow body builders (loop / if-then /
        if-else / while-after); call inside the appropriate Region/scope context.
        """
        for stmt in stmts:
            transformed_stmt = self.visit(stmt)  # Recursively visit inner statements
            if isinstance(transformed_stmt, list):
                out.extend(transformed_stmt)
            else:
                out.append(transformed_stmt)

    @staticmethod
    def _names_constant_list(names: Sequence[str]) -> ast.List:
        return ast.List(
            elts=[ast.Constant(value=name) for name in names], ctx=ast.Load()
        )

    @staticmethod
    def _names_constant_tuple(names: Sequence[str]) -> ast.Tuple:
        return ast.Tuple(
            elts=[ast.Constant(value=name) for name in names], ctx=ast.Load()
        )

    def create_loop_function(
        self,
        func_name: str,
        node: ast.For,
        start: ast.expr,
        stop: ast.expr,
        step: ast.expr,
        unroll: ast.expr,
        unroll_full: ast.expr,
        write_args: list[str],
        full_write_args_count: int,
        mutated_names: Sequence[str] = (),
        options: Sequence[ast.keyword] = (),
    ) -> ast.FunctionDef:
        """
        Creates a loop body function with the `loop_selector` decorator.
        """

        assert isinstance(node.target, ast.Name)
        func_args = [ast.arg(arg=node.target.id, annotation=None)]
        func_args += [ast.arg(arg=var, annotation=None) for var in write_args]

        # Create the loop body
        transformed_body: list[ast.stmt] = []
        with Region(self.session_data, new_value=transformed_body):
            self._visit_stmts_into(node.body, transformed_body)

        # Handle the return for a single iterated argument correctly
        if len(write_args) == 0:
            transformed_body.append(ast.Return())
        else:
            transformed_body.append(
                ast.Return(
                    value=ast.List(
                        elts=[ast.Name(id=var, ctx=ast.Load()) for var in write_args],
                        ctx=ast.Load(),
                    )
                )
            )

        # Define the decorator with parameters
        decorator = ast.copy_location(
            ast.Call(
                func=_create_module_attribute(
                    self.DECORATOR_FOR_STATEMENT,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ),
                args=[start, stop, step],
                keywords=[
                    ast.keyword(
                        arg="write_args",
                        value=self.generate_get_locals_or_none_call(write_args),
                    ),
                    ast.keyword(
                        arg="full_write_args_count",
                        value=ast.Constant(value=full_write_args_count),
                    ),
                    ast.keyword(
                        arg="write_args_names",
                        value=self._names_constant_list(write_args),
                    ),
                    ast.keyword(
                        arg="mutated_names",
                        value=self._names_constant_tuple(mutated_names),
                    ),
                    ast.keyword(arg="unroll", value=unroll),
                    ast.keyword(arg="unroll_full", value=unroll_full),
                    *options,
                ],
            ),
            node,
        )

        return ast.copy_location(
            ast.FunctionDef(
                name=func_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=func_args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=transformed_body,
                decorator_list=[decorator],
            ),
            node,
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        """Expand ``a and b``/``a or b`` so Python short-circuits and staged values use ``and_``/``or_``."""
        self.generic_visit(node)

        # The ternary emitted here is evaluated by Python itself (it tests a
        # Python bool), so it never reaches the staged ternary lowering.
        if isinstance(node.op, ast.And):
            # Emitted form (one lhs evaluation, Python short-circuit):
            # tmp if bool_short_circuits((tmp := lhs), False) else and_(tmp, rhs)
            short_circuit_value = False
            helper_name = "and_"
        elif isinstance(node.op, ast.Or):
            # Emitted form (one lhs evaluation, Python short-circuit):
            # tmp if bool_short_circuits((tmp := lhs), True) else or_(tmp, rhs)
            short_circuit_value = True
            helper_name = "or_"
        else:
            # BoolOp should be either And or Or -- reaching here is a compiler
            # bug (the AST grammar only produces And/Or), not an author mistake.
            raise DSLRuntimeError(
                f"Unsupported boolean operation: {node.op}",
                filename=self.session_data.file_name,
                snippet=ast.unparse(node),
            )
        self.session_data.import_top_module = True

        def located(expr: _AstNode) -> _AstNode:
            return ast.copy_location(expr, node)

        # Evaluate-once lowering: the lhs binds to a synthesized temp INSIDE
        # the test (a named expression), so every synthesized position reads
        # that one evaluation; a side-effecting or staged lhs runs once. The
        # test asks ``bool_short_circuits`` whether the temp is a Python bool
        # equal to the short-circuit value; only the other arm evaluates the
        # rhs, exactly like Python.
        lhs = node.values[0]
        for i in range(1, len(node.values)):
            tmp_name = f"_dsl_bool_{self.session_data.counter}"
            self.session_data.counter += 1
            walrus = located(
                ast.NamedExpr(
                    target=located(ast.Name(id=tmp_name, ctx=ast.Store())),
                    value=lhs,
                )
            )
            test = located(
                ast.Call(
                    func=_create_module_attribute(
                        self.BOOL_SHORT_CIRCUITS,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    ),
                    args=[walrus, located(ast.Constant(value=short_circuit_value))],
                    keywords=[],
                )
            )
            lhs = located(
                ast.IfExp(
                    test=test,
                    body=located(ast.Name(id=tmp_name, ctx=ast.Load())),
                    orelse=located(
                        ast.Call(
                            func=_create_module_attribute(
                                helper_name,
                                use_base_dsl=False,
                                lineno=node.lineno,
                                col_offset=node.col_offset,
                            ),
                            args=[
                                located(ast.Name(id=tmp_name, ctx=ast.Load())),
                                node.values[i],
                            ],
                            keywords=[],
                        )
                    ),
                )
            )

        return lhs

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        """Rewrite ``not x`` to the DSL's ``not_(x)`` (``__dsl_not__`` on a staged value)."""
        self.generic_visit(node)

        if isinstance(node.op, ast.Not):
            func_name = _create_module_attribute(
                "not_",
                use_base_dsl=False,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            self.session_data.import_top_module = True
            return ast.copy_location(
                ast.Call(func=func_name, args=[node.operand], keywords=[]), node
            )

        return node

    def _handle_native_for(self, node: ast.For) -> ast.For | list[ast.stmt]:
        """Visit the body of a loop that runs in Python (the native arm of a
        dispatched loop, a list/tuple display).

        Under the NATIVE policy a nested ``if``/``for``/``while`` that owns an
        early exit of this loop stays native Python, so the ``break`` still
        reaches the loop; nested regions without one are staged as usual. A
        sub-DSL may override this for its own Python-loop handling.
        """
        with self.session_data.control_flow_policy(ControlFlowPolicy.NATIVE):
            self.generic_visit(node)
        return node

    def _transform_runtime_dispatched_for(self, node: ast.For) -> list[ast.stmt]:
        """Select staged or native loop handling from the trace-time iterable."""
        suffix = self.session_data.counter
        self.session_data.counter += 1
        iter_name = f"_dsl_for_iter_{suffix}"

        visited_iter = self.visit(node.iter)
        assert isinstance(visited_iter, ast.expr)
        if isinstance(visited_iter, ast.Call):
            factory = visited_iter.func
            if isinstance(factory, ast.Attribute) and factory.attr == "range":
                factory = _create_module_attribute(
                    "range",
                    lineno=factory.lineno,
                    col_offset=factory.col_offset,
                )
            visited_iter = ast.copy_location(
                ast.Call(
                    func=_create_module_attribute(
                        "materialize_for_iter",
                        lineno=visited_iter.lineno,
                        col_offset=visited_iter.col_offset,
                    ),
                    args=[factory, *visited_iter.args],
                    keywords=visited_iter.keywords,
                ),
                visited_iter,
            )
        iter_assign = ast.copy_location(
            ast.Assign(
                targets=[ast.Name(id=iter_name, ctx=ast.Store())],
                value=visited_iter,
            ),
            node.iter,
        )

        def iter_attr(name: str) -> ast.Attribute:
            return ast.Attribute(
                value=ast.Name(id=iter_name, ctx=ast.Load()),
                attr=name,
                ctx=ast.Load(),
            )

        def iter_option(name: str, default: ast.expr) -> ast.Call:
            return ast.Call(
                func=ast.Name(id="getattr", ctx=ast.Load()),
                args=[
                    ast.Name(id=iter_name, ctx=ast.Load()),
                    ast.Constant(value=name),
                    default,
                ],
                keywords=[],
            )

        dynamic_for = _deepcopy_ast_root(node)
        dynamic_for.iter = ast.copy_location(
            ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[iter_attr("start"), iter_attr("stop"), iter_attr("step")],
                keywords=[
                    ast.keyword(
                        arg="unroll",
                        value=iter_option("unroll", ast.Constant(value=-1)),
                    ),
                    ast.keyword(
                        arg="unroll_full",
                        value=iter_option("unroll_full", ast.Constant(value=False)),
                    ),
                    ast.keyword(
                        arg=None,
                        value=iter_option("options", ast.Dict(keys=[], values=[])),
                    ),
                ],
            ),
            node.iter,
        )

        python_for = _deepcopy_ast_root(node)
        python_for.iter = ast.copy_location(
            ast.Name(id=iter_name, ctx=ast.Load()), node.iter
        )
        active_symbols = self.session_data.scope_manager.get_active_symbols()
        native_local_renames = _native_for_local_renames(
            python_for, active_symbols, suffix
        )
        if native_local_renames:
            # Rename the source arm before transforming nested control flow.
            # Delaying this until the whole tree is built changes ast.Name
            # nodes but leaves generated function parameters and writeback
            # name lists spelled with the old name.
            _NativeForLocalRenamer(native_local_renames).visit(python_for)
        with self.session_data.scope_manager.enter_control_flow_scope(
            isolate_callables=True
        ):
            # Keep the selected arm as a native ``for``. Nested regions decide
            # locally whether an early exit requires native ownership or they
            # can return to TRACE, avoiding a pre-scan of the whole loop body.
            with self.session_data.control_flow_policy(ControlFlowPolicy.NATIVE):
                with self.session_data.enclosing_body(python_for):
                    visited_python_for = self._handle_native_for(python_for)
        assert isinstance(visited_python_for, ast.For)
        python_for = visited_python_for

        active_callables = self.session_data.scope_manager.get_active_callables()
        counter_snapshot = self.session_data.counter
        scope_manager_snapshot = deepcopy(self.session_data.scope_manager)
        python_preamble: list[ast.stmt] = []
        try:
            with self.session_data.scope_manager.enter_control_flow_scope():
                if isinstance(dynamic_for.target, ast.Name):
                    self.session_data.scope_manager.add_to_scope(dynamic_for.target.id)
                with self.session_data.enclosing_body(dynamic_for):
                    dynamic_body = self.transform_for_loop(
                        dynamic_for, active_symbols, active_callables
                    )
        except Exception as error:
            # The dynamic arm is speculative until trace-time dispatch. Keep an
            # error from that dead arm from rejecting a native Python loop, but
            # preserve the exact exception if the iterable is actually dynamic.
            self.session_data.counter = counter_snapshot
            self.session_data.scope_manager.restore_from(scope_manager_snapshot)
            error_id = register_deferred_for_error(error)

            def deferred_error_call(helper_name: str) -> ast.Expr:
                return ast.copy_location(
                    ast.Expr(
                        ast.Call(
                            func=_create_module_attribute(
                                helper_name,
                                lineno=node.lineno,
                                col_offset=node.col_offset,
                            ),
                            args=[ast.Constant(value=error_id)],
                            keywords=[],
                        )
                    ),
                    node,
                )

            dynamic_body = [deferred_error_call("raise_deferred_for_error")]
            python_preamble = [deferred_error_call("discard_deferred_for_error")]

        dispatch = ast.copy_location(
            ast.If(
                test=ast.Call(
                    func=_create_module_attribute(
                        "is_dynamic_range",
                        lineno=node.iter.lineno,
                        col_offset=node.iter.col_offset,
                    ),
                    args=[ast.Name(id=iter_name, ctx=ast.Load())],
                    keywords=[],
                ),
                body=dynamic_body,
                orelse=[*python_preamble, python_for],
            ),
            node,
        )
        self._visit_runtime_dispatch(dispatch)
        # Leaving the loop statement closes a generator iterable at once (the
        # builders hold an insertion point across their ``yield``), so an
        # exception raised in the body unwinds in order.
        close_iter = ast.copy_location(
            ast.Try(
                body=[dispatch],
                handlers=[],
                orelse=[],
                finalbody=[
                    ast.Expr(
                        ast.Call(
                            func=_create_module_attribute(
                                "close_for_iter",
                                lineno=node.iter.lineno,
                                col_offset=node.iter.col_offset,
                            ),
                            args=[ast.Name(id=iter_name, ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
            ),
            node,
        )
        return [iter_assign, close_iter]

    @staticmethod
    def _literal_for_iter_kind(iter_node: ast.expr) -> str | None:
        """Classify an iterable whose loop kind is already known before tracing.

        "range" is a ``<module>.range(...)`` call, the DSL range that stages
        unconditionally, with plain positional arguments and only the loop
        options the staged lowering knows; "python" is a list or tuple display;
        None is anything else (name, attribute, subscript, other call) and the
        loop is dispatched at trace time. A bare ``range(...)`` is None on
        purpose: it is ``builtins.range`` unless the user rebound the name, and
        ``materialize_for_iter`` stages it only when one of its bounds is a
        staged value, so a loop over a Meta bound runs as a Python loop.
        Star-unpacking and any other keyword (``**kw``, a sub-DSL option) keep
        the dispatch, so the range object built at trace time still does their
        argument check and unexpected-keyword error.
        """
        if isinstance(iter_node, (ast.List, ast.Tuple)):
            return "python"
        if not isinstance(iter_node, ast.Call):
            return None
        func = iter_node.func
        is_range = isinstance(func, ast.Attribute) and func.attr == "range"
        if not is_range:
            return None
        if any(isinstance(arg, ast.Starred) for arg in iter_node.args) or any(
            keyword.arg not in _STAGED_RANGE_KEYWORDS for keyword in iter_node.keywords
        ):
            return None
        return "range"

    def _transform_staged_for(self, node: ast.For) -> list[ast.stmt]:
        """Lower a ``for`` over a literal range call straight to a staged loop."""
        assert isinstance(node.iter, ast.Call)
        # The positional bounds are visited by transform_for_loop; the loop
        # options are copied into the loop_selector call as they are, so visit
        # them here like the dispatch visits the whole iterable expression.
        for keyword in node.iter.keywords:
            visited_value = self.visit(keyword.value)
            assert isinstance(visited_value, ast.expr)
            keyword.value = visited_value
        active_symbols = self.session_data.scope_manager.get_active_symbols()
        active_callables = self.session_data.scope_manager.get_active_callables()
        with self.session_data.scope_manager.enter_control_flow_scope():
            if isinstance(node.target, ast.Name):
                self.session_data.scope_manager.add_to_scope(node.target.id)
            with self.session_data.enclosing_body(node):
                statements = self.transform_for_loop(
                    node, active_symbols, active_callables
                )
        return statements

    def _transform_native_for(self, node: ast.For) -> list[ast.stmt]:
        """Keep a ``for`` over a list or tuple display as the Python loop.

        Same shape as the Python candidate of a dispatched loop, minus the
        dispatch: the iterable is bound once, body-only names get private
        spellings and stay out of the enclosing scope.
        """
        suffix = self.session_data.counter
        self.session_data.counter += 1
        iter_name = f"_dsl_for_iter_{suffix}"
        visited_iter = self.visit(node.iter)
        assert isinstance(visited_iter, ast.expr)
        iter_assign = ast.copy_location(
            ast.Assign(
                targets=[ast.Name(id=iter_name, ctx=ast.Store())],
                value=visited_iter,
            ),
            node.iter,
        )
        node.iter = ast.copy_location(ast.Name(id=iter_name, ctx=ast.Load()), node.iter)
        active_symbols = self.session_data.scope_manager.get_active_symbols()
        renames = _native_for_local_renames(node, active_symbols, suffix)
        if renames:
            _NativeForLocalRenamer(renames).visit(node)
        with self.session_data.scope_manager.enter_control_flow_scope(
            isolate_callables=True
        ):
            with self.session_data.control_flow_policy(ControlFlowPolicy.NATIVE):
                with self.session_data.enclosing_body(node):
                    visited = self._handle_native_for(node)
        assert isinstance(visited, ast.For)
        return [iter_assign, visited]

    def visit_For(self, node: ast.For) -> ast.For | list[ast.stmt]:
        """Dispatch a ``for`` to its loop form.

        Under the NATIVE policy a loop owning an early exit stays native; a ``<module>.range(...)`` call
        is staged; a list/tuple display is a native loop; anything else (a
        bare ``range(...)``, a name, ...) is dispatched at trace time.
        """
        if self.session_data.current_control_flow_policy is ControlFlowPolicy.NATIVE:
            if self._find_early_exit(node, "for") is not None:
                return self._handle_native_for(node)
            with self.session_data.control_flow_policy(ControlFlowPolicy.TRACE):
                return self.visit_For(node)

        # Under every loop form below a body-born name stays inside the body.
        self._check_region_locals_escape(node, "for")

        # A literal iterable settles the loop kind here, so its body is
        # transformed once. An iterable the rewrite cannot classify, including
        # a bare ``range(...)`` whose bounds may be Meta or staged, pays for
        # the trace-time dispatch and its two candidate bodies.
        literal_kind = self._literal_for_iter_kind(node.iter)
        if literal_kind == "range":
            return self._transform_staged_for(node)
        if literal_kind == "python":
            return self._transform_native_for(node)
        return self._transform_runtime_dispatched_for(node)

    def _create_closure_check_call(
        self, called_closures: list[str], node: ast.stmt
    ) -> ast.stmt:
        """``closure_check([f, g])`` for the nested functions a region calls; a
        ``pass`` when the DSL disabled the check (``closure_check=False``)."""
        if not self.closure_check:
            return ast.copy_location(ast.Pass(), node)
        return ast.Expr(
            ast.Call(
                func=_create_module_attribute(
                    "closure_check",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ),
                args=[
                    ast.List(
                        elts=[ast.Name(id=c, ctx=ast.Load()) for c in called_closures],
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            )
        )

    def transform_for_loop(
        self,
        node: ast.For,
        active_symbols: list[set[str]],
        active_callables: list[set[str]],
    ) -> list[ast.stmt]:
        """Rewrite a ``for`` over a range call into a ``@loop_selector`` body.

        Rejects early exits and a loop ``else``; a loop target that is live
        before the loop is carried through ``loop_carried_var_N`` and restored
        after it.

        :return: the statements replacing the loop: optional carry setup and
            ``closure_check``, the decorated body and the write-back
        """
        self.check_early_exit(node, "for")
        if node.orelse:
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                what="A `for`/`while` loop with an `else:` clause",
                detail=": put the `else:` code after the loop",
                filename=self.session_data.file_name,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_col_offset=getattr(node.iter, "end_col_offset", None),
            )

        # Get loop target variable name
        target_var_name: str | None = None
        target_var_is_active_before_loop = False
        if isinstance(node.target, ast.Name):
            target_var_name = node.target.id
            for idx, active_symbol in enumerate(active_symbols):
                if target_var_name in active_symbol:
                    target_var_is_active_before_loop = True
                    active_symbols[idx] = active_symbol - {target_var_name}
                    break

        # Add necessary exprs to handle this
        exprs: list[ast.stmt] = []
        if target_var_is_active_before_loop:
            assert target_var_name is not None
            loop_carried_var_name = f"loop_carried_var_{self.session_data.counter}"
            exprs.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=loop_carried_var_name, ctx=ast.Store())],
                        value=ast.Name(id=target_var_name, ctx=ast.Load()),
                    ),
                    node,
                )
            )
            # append an extra assignment to the loop carried variable
            node.body.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=loop_carried_var_name, ctx=ast.Store())],
                        value=ast.Name(id=target_var_name, ctx=ast.Load()),
                    ),
                    node,
                )
            )
            active_symbols.append({loop_carried_var_name})

        assert isinstance(node.iter, ast.Call)
        start, stop, step = self.extract_range_args(node.iter)
        unroll, unroll_full = self.extract_unroll_args(node.iter)
        options = self.extract_loop_options(node.iter)
        (
            write_args,
            full_write_args_count,
            called_closures,
            mutated_names,
        ) = self.analyze_region_variables(node, active_symbols, active_callables)

        if called_closures:
            exprs.append(self._create_closure_check_call(called_closures, node))

        func_name = f"loop_body_{self.session_data.counter}"
        self.session_data.counter += 1

        func_def = self.create_loop_function(
            func_name,
            node,
            start,
            stop,
            step,
            unroll,
            unroll_full,
            write_args,
            full_write_args_count,
            mutated_names,
            options,
        )

        assign = self.create_cf_call(func_name, write_args, node)

        # This should work fine as it modifies the AST structure
        exprs = exprs + [func_def] + assign

        if target_var_is_active_before_loop:
            assert target_var_name is not None
            exprs.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=target_var_name, ctx=ast.Store())],
                        value=ast.Name(id=loop_carried_var_name, ctx=ast.Load()),
                    ),
                    node,
                )
            )

        return exprs

    # =============================================================================
    # Statements and expressions
    # =============================================================================

    def visit_Assert(self, node: ast.Assert) -> ast.Expr:
        """Rewrite ``assert test, msg`` to ``assert_executor(test=..., msg=...)``."""
        test = self.visit(node.test)

        args = [ast.keyword(arg="test", value=test)]
        if node.msg:
            msg = self.visit(node.msg)
            args.append(ast.keyword(arg="msg", value=msg))

        # Rewrite to assert_executor(test, msg)
        new_node = ast.Expr(
            ast.Call(
                func=_create_module_attribute(
                    self.ASSERT_EXECUTOR, lineno=node.lineno, col_offset=node.col_offset
                ),
                args=[],
                keywords=args,
            )
        )

        # Propagate line number from original node to new node
        ast.copy_location(new_node, node)
        return new_node

    @staticmethod
    def _is_mlir_dialect_module(module: object) -> bool:
        """Whether *module* is one of the MLIR dialect modules (``mlir.dialects.*``)."""
        if not isinstance(module, ModuleType):
            return False
        package = module.__package__ or ""
        return package == _MLIR_DIALECTS_PACKAGE or package.startswith(
            f"{_MLIR_DIALECTS_PACKAGE}."
        )

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Rewrite ``bool(x)``, zero-argument ``super()`` and calls into MLIR dialect modules."""
        # The callee expression is visited ONCE: visiting an already-transformed
        # callee re-runs the non-idempotent lowerings inside it (ternary blocks,
        # and/or expansion) over a chained call's receiver arguments.
        func = self.visit(node.func)
        node.func = func
        # Visit args and kwargs
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(kwarg) for kwarg in node.keywords]

        # Rewrite call to some built-in functions
        if isinstance(func, ast.Name):
            # AST rewrite only redirect call to bool to bool_cast
            # If `bool` escapes as a symbol, usually it means type check, do not rewrite it
            # Any other call shape has no `bool_cast` spelling, so it is left as a
            # plain `bool` call for Python itself to accept or reject
            if (
                func.id == "bool"
                and len(node.args) == 1
                and node.keywords == []
                and not isinstance(node.args[0], ast.Starred)
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Call(
                            func=_create_module_attribute(
                                self.BUILTIN_REDIRECTOR,
                                lineno=node.lineno,
                                col_offset=node.col_offset,
                            ),
                            args=[func],
                            keywords=[],
                        ),
                        args=[node.args[0]],
                        keywords=[],
                    ),
                    node,
                )
            elif func.id == "super" and node.args == [] and node.keywords == []:
                # If it's a Python3 argument free super(), rewrite to old style super with args
                # So if this call is under dynamic control flow, it still works.
                return ast.copy_location(
                    ast.Call(
                        func=func,
                        args=node.args
                        + [
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="__class__",
                                ctx=ast.Load(),
                            ),
                            ast.Name(id="self", ctx=ast.Load()),
                        ],
                        keywords=node.keywords,
                    ),
                    node,
                )
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):

            def create_downcast_call(arg: ast.expr) -> ast.Call:
                return ast.copy_location(
                    ast.Call(
                        func=_create_module_attribute(
                            self.IMPLICIT_DOWNCAST_NUMERIC_TYPE,
                            use_base_dsl=False,
                            lineno=node.lineno,
                            col_offset=node.col_offset,
                        ),
                        args=[arg],
                        keywords=[],
                    ),
                    arg,
                )

            fn_globals = self.session_data.function_globals
            module = fn_globals.get(func.value.id) if fn_globals else None
            if self._is_mlir_dialect_module(module):
                # A direct dialect builder call takes raw ir.Values: downcast
                # every Numeric argument through `as_ir_value`.
                self.session_data.import_top_module = True
                args: list[ast.expr] = []
                for arg in node.args:
                    args.append(create_downcast_call(arg))
                kwargs: list[ast.keyword] = []
                for kwarg in node.keywords:
                    kwargs.append(
                        ast.copy_location(
                            ast.keyword(
                                arg=kwarg.arg,
                                value=create_downcast_call(kwarg.value),
                            ),
                            kwarg,
                        )
                    )
                return ast.copy_location(
                    ast.Call(func=func, args=args, keywords=kwargs), node
                )
        # No `else` arm: a callee that is neither a Name nor an Attribute over a
        # Name needs no rewrite here, and it is already visited and stored above.

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        with self.session_data.set_current_class_name(node.name):
            return self.generic_visit(node)

    def _visit_target(self, target: ast.expr) -> None:
        """Publish every plain name an assignment target binds, nested or not."""
        if isinstance(target, ast.Name):
            self.session_data.scope_manager.add_to_scope(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self._visit_target(t)
        elif isinstance(target, ast.Starred):
            self._visit_target(target.value)

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        for target in node.targets:
            self._visit_target(target)
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign | list[ast.stmt]:
        self._visit_target(node.target)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | list[ast.stmt]:
        self._visit_target(node.target)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name | ast.Call:
        """Route builtin reads through ``redirect_builtin_function``; reject a read of ``_``."""
        is_load = isinstance(node.ctx, ast.Load)
        if node.id in ["max", "min", "any", "all", "exec", "eval"] and is_load:
            return ast.copy_location(
                ast.Call(
                    func=_create_module_attribute(
                        self.BUILTIN_REDIRECTOR,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    ),
                    args=[node],
                    keywords=[],
                ),
                node,
            )
        elif node.id == "_" and is_load:
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                what="Reading the throwaway name `_`",
                detail=": give the value a real name",
                filename=self.session_data.file_name,
                lineno=getattr(node, "lineno", None),
                col_offset=getattr(node, "col_offset", None),
                end_col_offset=getattr(node, "end_col_offset", None),
            )
        else:
            self.generic_visit(node)
        return node

    # =============================================================================
    # Decorators and function definitions
    # =============================================================================

    def _dsl_decorator_name(self, decorator: ast.expr) -> str | None:
        """The DSL decorator *decorator* spells (``@m.jit``, ``@jit``, or a call of either)."""
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Attribute):
            name = decorator.attr
        elif isinstance(decorator, ast.Name):
            name = decorator.id
        else:
            return None
        return name if name in self.decorator_names else None

    def get_dsl_decorator_index(self, decorator_list: list[ast.expr]) -> int | None:
        """The index of the DSL decorator in ``decorator_list``, or ``None``;
        ``@jit`` and ``@jit()`` both qualify (the decorators take no options)."""
        for i, d in enumerate(decorator_list):
            if self._dsl_decorator_name(d) is not None:
                return i
        return None

    def check_decorator(self, node: ast.AST) -> bool:
        """Whether ``node`` is a function carrying a DSL decorator.

        The DSL decorator must be the innermost one (written directly above
        ``def``); otherwise ``UNSUP_SYNTAX`` is raised.
        """
        if not isinstance(node, ast.FunctionDef):
            return False
        decorator_list = node.decorator_list
        if len(decorator_list) == 0:
            return False

        dsl_decorator_index = self.get_dsl_decorator_index(decorator_list)

        if (
            dsl_decorator_index is not None
            and dsl_decorator_index < len(decorator_list) - 1
        ):
            decorator_node = decorator_list[dsl_decorator_index]
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                filename=self.session_data.file_name,
                lineno=getattr(decorator_node, "lineno", None),
                col_offset=getattr(decorator_node, "col_offset", None),
                end_col_offset=getattr(decorator_node, "end_col_offset", None),
                what=f"`{ast.unparse(decorator_node)}` above another decorator",
                detail=": it must be the innermost decorator, written directly above `def`",
            )

        return dsl_decorator_index is not None

    def remove_dsl_decorator(self, decorator_list: list[ast.expr]) -> list[ast.expr]:
        """Drop the DSL decorators (``@jit``/``@jit(...)``, ``@kernel``, ...) from a list."""
        return [d for d in decorator_list if self._dsl_decorator_name(d) is None]

    def visit_Global(self, node: ast.Global) -> None:
        """``global`` is not compiled (``UNSUP_SYNTAX``)."""
        raise DSLUserCodeError(
            DiagId.UNSUP_SYNTAX,
            what="`global`",
            detail=": compiled code cannot assign to a module-level variable; pass the value in as an argument and return the updated value",
            filename=self.session_data.file_name,
            lineno=getattr(node, "lineno", None),
            col_offset=getattr(node, "col_offset", None),
            end_col_offset=getattr(node, "end_col_offset", None),
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        """``nonlocal`` may only name a variable already bound in an enclosing scope."""
        active_symbols = self.session_data.scope_manager.get_active_symbols()
        nonlocal_names = OrderedSet(node.names)
        intersect = nonlocal_names.intersections(active_symbols)
        for name in node.names:
            if name not in intersect:
                raise DSLUserCodeError(
                    DiagId.UNSUP_SYNTAX,
                    filename=self.session_data.file_name,
                    lineno=getattr(node, "lineno", None),
                    col_offset=getattr(node, "col_offset", None),
                    end_col_offset=getattr(node, "end_col_offset", None),
                    what=f"`{ast.unparse(node)}`",
                    detail=f": `{name}` belongs to an enclosing function, which compiled code cannot assign to; pass it in as an argument and return the updated value",
                )
        self.generic_visit(node)
        return node

    @staticmethod
    def _insert_captured_writeback_nonlocals(
        node: ast.FunctionDef,
        captured_names: set[str],
    ) -> None:
        """Declare captured names written back by outlined control flow.

        Without this declaration, a generated ``captured = region(...)``
        classifies ``captured`` as a local in the nested function and makes a
        preceding closure read fail. ``nonlocal`` keeps the writeback on the
        original closure cell while still carrying the SSA value across the
        outlined region.
        """
        if not captured_names:
            return

        existing_names = {
            name
            for statement in node.body
            if isinstance(statement, ast.Nonlocal)
            for name in statement.names
        }
        names_to_declare = sorted(captured_names - existing_names)
        if not names_to_declare:
            return

        declaration = ast.copy_location(ast.Nonlocal(names=names_to_declare), node)
        insert_at = 1 if ast.get_docstring(node, clean=False) is not None else 0
        node.body.insert(insert_at, declaration)

    def visit_FunctionDef(
        self, node: ast.FunctionDef
    ) -> ast.FunctionDef | list[ast.stmt]:
        """Visit a function body in its own scope; strip annotations and DSL decorators."""
        # The function is a callable of the enclosing scope.
        self.session_data.scope_manager.add_to_callables(node.name)

        with self.session_data.scope_manager.enter_local_scope():
            with self.session_data.set_current_function_name(node.name):
                with (
                    self.session_data.collect_captured_control_flow_writebacks()
                ) as writebacks:
                    with self.session_data.enclosing_body(node):
                        # Keep recursive calls visible as callables without
                        # pretending the function name is an initialized body
                        # local. Python does not bind that local, so seeding
                        # the variable scope would turn a first same-spelling
                        # assignment into a read-before-write reassignment.
                        self.session_data.scope_manager.add_to_callables(node.name)

                        for arg in node.args.args:
                            self.session_data.scope_manager.add_to_scope(arg.arg)
                            arg.annotation = None

                        for arg in node.args.kwonlyargs:
                            self.session_data.scope_manager.add_to_scope(arg.arg)
                            arg.annotation = None

                        for arg in node.args.posonlyargs:
                            self.session_data.scope_manager.add_to_scope(arg.arg)
                            arg.annotation = None

                        # Strip return annotation
                        node.returns = None

                        self.generic_visit(node)

        # Remove the DSL decorators
        node.decorator_list = self.remove_dsl_decorator(node.decorator_list)
        self._insert_captured_writeback_nonlocals(node, writebacks)

        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name):
                self.session_data.scope_manager.add_to_scope(item.optional_vars.id)
        return self.generic_visit(node)

    # =============================================================================
    # While loops
    # =============================================================================

    def visit_While(self, node: ast.While) -> ast.While | list[ast.stmt]:
        """Rewrite a ``while`` into a ``@while_selector`` region."""
        early_exit = self._find_early_exit(node, "while")
        if self.session_data.current_control_flow_policy is ControlFlowPolicy.NATIVE:
            if early_exit is not None:
                return self._handle_native_early_exit(node, early_exit)
            with self.session_data.control_flow_policy(ControlFlowPolicy.TRACE):
                return self.visit_While(node)

        # A ``while`` owning a ``break``/``continue``/``return``/``raise`` cannot
        # be outlined: it stays native and its test must be Meta at trace time.
        if early_exit is not None:
            return self._handle_native_early_exit(node, early_exit)

        # A staged ``while ... else:`` cannot capture its else-clause (parity with
        # the staged ``for ... else`` rejection).
        if node.orelse:
            raise DSLUserCodeError(
                DiagId.UNSUP_SYNTAX,
                what="A `for`/`while` loop with an `else:` clause",
                detail=": put the `else:` code after the loop",
                filename=self.session_data.file_name,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_col_offset=getattr(node.test, "end_col_offset", None),
            )

        self._check_region_locals_escape(node, "while")

        active_symbols = self.session_data.scope_manager.get_active_symbols()
        active_callables = self.session_data.scope_manager.get_active_callables()

        with self.session_data.scope_manager.enter_control_flow_scope():
            (
                write_args,
                full_write_args_count,
                called_closures,
                mutated_names,
            ) = self.analyze_region_variables(node, active_symbols, active_callables)
            exprs: list[ast.stmt] = []
            if called_closures:
                exprs.append(self._create_closure_check_call(called_closures, node))

            func_name = f"while_region_{self.session_data.counter}"
            self.session_data.counter += 1

            with self.session_data.enclosing_body(node):
                func_def = self.create_while_function(
                    func_name, node, write_args, full_write_args_count, mutated_names
                )
            assign = self.create_cf_call(func_name, write_args, node)

        return exprs + [func_def] + assign

    def create_cf_call(
        self, func_name: str, yield_args: list[str], node: ast.stmt
    ) -> list[ast.stmt]:
        """Create result writeback for a rewritten control-flow region."""
        if not yield_args:
            return [
                ast.copy_location(
                    ast.Expr(value=ast.Name(id=func_name, ctx=ast.Load())), node
                )
            ]
        for name in yield_args:
            scopes = self.session_data.scope_manager
            if not scopes.current_function_owns(
                name
            ) and scopes.enclosing_function_owns(name):
                self.session_data.record_captured_control_flow_writeback(name)
        if len(yield_args) == 1:
            assign = ast.Assign(
                targets=[ast.Name(id=yield_args[0], ctx=ast.Store())],
                value=ast.Name(id=func_name, ctx=ast.Load()),
            )
        else:
            assign = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=var, ctx=ast.Store()) for var in yield_args],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Name(id=func_name, ctx=ast.Load()),
            )
        return [ast.copy_location(assign, node)]

    # =============================================================================
    # Comprehensions and lambdas
    # =============================================================================

    def _visit_Comprehension(
        self, node: _ComprehensionT, ele_visitor: Callable[..., Any]
    ) -> _ComprehensionT:
        node.generators = [self.visit(generator) for generator in node.generators]

        targets: list[str] = []

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Store):
                    targets.append(node.id)

        # Collect generator targets
        collector = NameCollector()
        for generator in node.generators:
            collector.visit(generator)

        # Stack the targets: a comprehension nested in another still sees the
        # outer generators' names (they are read by its element expression).
        current_generator_targets = len(self.session_data.generator_targets)
        self.session_data.generator_targets.extend(targets)

        ele_visitor(node)

        del self.session_data.generator_targets[current_generator_targets:]
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        def key_value_visitor(n: ast.DictComp) -> None:
            n.key = self.visit(n.key)
            n.value = self.visit(n.value)

        return self._visit_Comprehension(node, key_value_visitor)

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        current_lambda_args = len(self.session_data.lambda_args)
        for arg in node.args.args:
            self.session_data.lambda_args.append(arg.arg)

        node.body = self.visit(node.body)

        self.session_data.lambda_args = self.session_data.lambda_args[
            :current_lambda_args
        ]

        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        return self._visit_Comprehension(
            node, lambda n: setattr(n, "elt", self.visit(n.elt))
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        return self._visit_Comprehension(
            node, lambda n: setattr(n, "elt", self.visit(n.elt))
        )

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        return self._visit_Comprehension(
            node, lambda n: setattr(n, "elt", self.visit(n.elt))
        )

    # =============================================================================
    # Conditional expressions and comparisons
    # =============================================================================

    def visit_IfExp(self, node: ast.IfExp) -> ast.Call:
        """
        Transforms an inline if-else (ternary) expression into runtime-dispatched
        control flow using synthesized function definitions for each branch.

        ``x if cond else y`` becomes two local function blocks inserted just
        before the current statement and a call to the conditional executor
        that references them and the predicate.
        """
        # Create unique names for the then and else branch function blocks
        then_block_name = f"ifexp_then_block_{self.session_data.counter}"
        else_block_name = f"ifexp_else_block_{self.session_data.counter}"
        self.session_data.counter += 1

        def block_args() -> ast.arguments:
            return ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=target, annotation=None)
                    for target in chain(
                        self.session_data.generator_targets,
                        self.session_data.lambda_args,
                    )
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            )

        # Define the then-block function, returning the visited body
        then_block_def = ast.FunctionDef(
            name=then_block_name,
            args=block_args(),
            body=[ast.Return(value=self.visit(node.body))],
            decorator_list=[],
        )
        # Define the else-block function, returning the visited orelse
        else_block_def = ast.FunctionDef(
            name=else_block_name,
            args=block_args(),
            body=[ast.Return(value=self.visit(node.orelse))],
            decorator_list=[],
        )

        # Insert the block definitions into the most recent (innermost) region before the statement
        self.session_data.region_stack[-1].append_new_stmts(
            [
                ast.copy_location(then_block_def, node),
                ast.copy_location(else_block_def, node),
            ]
        )

        # Create the executor call node, wiring up the predicate and newly synthesized blocks
        executor_call = ast.Call(
            func=_create_module_attribute(self.IFEXP_EXECUTOR),
            args=[],
            keywords=[
                ast.keyword(arg="pred", value=self.visit(node.test)),
                ast.keyword(
                    arg="block_args",
                    value=ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Load())
                            for name in chain(
                                self.session_data.generator_targets,
                                self.session_data.lambda_args,
                            )
                        ],
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="then_block", value=ast.Name(id=then_block_name, ctx=ast.Load())
                ),
                ast.keyword(
                    arg="else_block", value=ast.Name(id=else_block_name, ctx=ast.Load())
                ),
            ],
        )

        # Return the transformed executor call node at the original location in the AST
        return ast.copy_location(executor_call, node)

    cmpops = {
        "Eq": "==",
        "NotEq": "!=",
        "Lt": "<",
        "LtE": "<=",
        "Gt": ">",
        "GtE": ">=",
        "Is": "is",
        "IsNot": "is not",
        "In": "in",
        "NotIn": "not in",
    }

    def compare_ops_to_str(self, node: ast.Compare) -> ast.List:
        names: list[ast.expr] = [
            ast.Constant(value=self.cmpops[op.__class__.__name__]) for op in node.ops
        ]
        return ast.List(elts=names, ctx=ast.Load())

    def visit_Compare(self, node: ast.Compare) -> ast.Call:
        """Rewrite ``a < b <= c`` to ``compare_executor(left=a, comparators=[b, c], ops=['<', '<='])``."""
        self.generic_visit(node)

        comparator_strs = self.compare_ops_to_str(node)

        keywords = [
            ast.keyword(arg="left", value=node.left),
            ast.keyword(
                arg="comparators", value=ast.List(elts=node.comparators, ctx=ast.Load())
            ),
            ast.keyword(arg="ops", value=comparator_strs),
        ]

        call = ast.copy_location(
            ast.Call(
                func=_create_module_attribute(self.COMPARE_EXECUTOR),
                args=[],
                keywords=keywords,
            ),
            node,
        )

        return call

    # =============================================================================
    # If statements
    # =============================================================================

    def _visit_runtime_dispatch(
        self,
        node: ast.If,
    ) -> None:
        """Visit the dispatch test without leaking for-local definitions.

        A staged for only carries names that were bound before the loop. Names
        first assigned in either dispatch arm remain internal to that arm and
        are intentionally unavailable to later staged control flow.
        """
        visited_test = self.visit(node.test)
        assert isinstance(visited_test, ast.expr)
        node.test = visited_test

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        """Rewrite an ``if`` into an ``@if_selector`` region."""
        if self.session_data.current_control_flow_policy is ControlFlowPolicy.NATIVE:
            # The native policy is entered only while visiting the Python arm
            # of a runtime-dispatched ``for``.  Keep an ``if`` native when
            # outlining it would detach an early exit from that loop/function;
            # ordinary nested conditions must still be staged.
            early_exit = self._find_early_exit(node, "for")
            if early_exit is not None:
                return self._handle_native_early_exit(node, early_exit)
            with self.session_data.control_flow_policy(ControlFlowPolicy.TRACE):
                return self.visit_If(node)

        # An ``if`` owning a ``return``/``raise`` cannot be outlined: it stays
        # native and its test must be a Meta value at trace time.
        early_exit = self._find_early_exit(node, "if")
        if early_exit is not None:
            return self._handle_native_early_exit(node, early_exit)

        active_symbols = self.session_data.scope_manager.get_active_symbols()
        active_callables = self.session_data.scope_manager.get_active_callables()

        with self.session_data.scope_manager.enter_control_flow_scope():
            (
                yield_args,
                full_write_args_count,
                called_closures,
                mutated_names,
            ) = self.analyze_region_variables(node, active_symbols, active_callables)
            # A name born in an arm and read after the ``if`` joins the stored
            # group of the write_args, seeded ``None`` in the enclosing frame.
            born_locals = self._if_born_locals(node, active_symbols)
            if born_locals:
                yield_args = (
                    yield_args[:full_write_args_count]
                    + born_locals
                    + yield_args[full_write_args_count:]
                )
                full_write_args_count += len(born_locals)
            exprs: list[ast.stmt] = []
            if called_closures:
                exprs.append(self._create_closure_check_call(called_closures, node))

            func_name = f"if_region_{self.session_data.counter}"
            self.session_data.counter += 1

            func_def = self.create_if_function(
                func_name, node, yield_args, full_write_args_count, mutated_names
            )
            assign = self.create_cf_call(func_name, yield_args, node)

        # The writeback binds the if-born names in the enclosing scope.
        for name in born_locals:
            self.session_data.scope_manager.add_to_scope(name)

        return exprs + [func_def] + assign

    def generate_get_locals_or_none_call(self, write_args: list[str]) -> ast.Call:
        """``get_locals_or_none(locals(), [...names])``: the region's seed values."""
        return ast.Call(
            func=_create_module_attribute("get_locals_or_none"),
            args=[
                ast.Call(
                    func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]
                ),
                self._names_constant_list(write_args),
            ],
            keywords=[],
        )

    def create_if_function(
        self,
        func_name: str,
        node: ast.If,
        write_args: list[str],
        full_write_args_count: int,
        mutated_names: Sequence[str] = (),
    ) -> ast.FunctionDef:
        """Create the ``@if_selector`` region of an ``if``.

        The region defines ``then_block_N``/``else_block_N`` over the
        write_args and returns ``if_executor(...)``; an ``elif`` becomes a
        nested region inside the else block.
        """
        test_expr = self.visit(node.test)
        pred_name = self.make_func_param_name("pred", write_args)
        func_args = [ast.arg(arg=pred_name, annotation=None)]
        func_args += [ast.arg(arg=var, annotation=None) for var in write_args]
        func_args_then_else = [ast.arg(arg=var, annotation=None) for var in write_args]

        then_body: list[ast.stmt] = []
        with Region(self.session_data, new_value=then_body):
            with self.session_data.scope_manager.enter_control_flow_scope():
                self._visit_stmts_into(node.body, then_body)

        # Create common return list for all blocks
        return_list = ast.List(
            elts=[ast.Name(id=var, ctx=ast.Load()) for var in write_args],
            ctx=ast.Load(),
        )

        # Create common function arguments
        func_decorator_arguments = ast.arguments(
            posonlyargs=[], args=func_args, kwonlyargs=[], kw_defaults=[], defaults=[]
        )
        func_then_else_arguments = ast.arguments(
            posonlyargs=[],
            args=func_args_then_else,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )

        then_block_name = f"then_block_{self.session_data.counter}"
        else_block_name = f"else_block_{self.session_data.counter}"
        elif_region_name = f"elif_region_{self.session_data.counter}"
        self.session_data.counter += 1

        # Create then block
        then_block = ast.copy_location(
            ast.FunctionDef(
                name=then_block_name,
                args=func_then_else_arguments,
                body=then_body + [ast.Return(value=return_list)],
                decorator_list=[],
            ),
            node,
        )

        # Decorator keywords
        decorator_keywords = [
            ast.keyword(arg="pred", value=test_expr),
            ast.keyword(
                arg="write_args",
                value=self.generate_get_locals_or_none_call(write_args),
            ),
        ]

        # Create decorator
        decorator = ast.copy_location(
            ast.Call(
                func=_create_module_attribute(
                    self.DECORATOR_IF_STATEMENT,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ),
                args=[],
                keywords=decorator_keywords,
            ),
            node,
        )

        # Executor keywords
        execute_keywords = [
            ast.keyword(arg="pred", value=ast.Name(id=pred_name, ctx=ast.Load())),
            ast.keyword(
                arg="write_args",
                value=ast.List(
                    elts=[ast.Name(id=arg, ctx=ast.Load()) for arg in write_args],
                    ctx=ast.Load(),
                ),
            ),
            ast.keyword(
                arg="full_write_args_count",
                value=ast.Constant(value=full_write_args_count),
            ),
            ast.keyword(
                arg="write_args_names",
                value=self._names_constant_list(write_args),
            ),
            ast.keyword(
                arg="mutated_names",
                value=self._names_constant_tuple(mutated_names),
            ),
            ast.keyword(
                arg="then_block", value=ast.Name(id=then_block_name, ctx=ast.Load())
            ),
        ]
        # Handle different cases
        if not write_args and node.orelse == []:
            # No write_args case - only then_block needed
            execute_call = ast.copy_location(
                ast.Call(
                    func=_create_module_attribute(
                        self.IF_EXECUTOR, lineno=node.lineno, col_offset=node.col_offset
                    ),
                    args=[],
                    keywords=execute_keywords,
                ),
                node,
            )
            func_body = [then_block, ast.Return(value=execute_call)]
        else:
            # Create else block based on node.orelse
            if node.orelse:
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # Handle elif case
                    elif_node = node.orelse[0]
                    nested_if_name = elif_region_name
                    # `elif pred:` and `else: if pred:` share one AST. Statements
                    # hoisted while visiting the elif test (ternary blocks) must
                    # execute only when every earlier arm's condition is false:
                    # collect them into the synthesized else block.
                    elif_pre: list[ast.stmt] = []
                    # Recursion for nested elif
                    with Region(self.session_data, new_value=elif_pre):
                        nested_if = self.create_if_function(
                            nested_if_name,
                            elif_node,
                            write_args,
                            full_write_args_count,
                            mutated_names,
                        )
                    else_block = ast.FunctionDef(
                        name=else_block_name,
                        args=func_then_else_arguments,
                        body=elif_pre
                        + [
                            nested_if,
                            ast.Return(
                                value=ast.Name(id=nested_if_name, ctx=ast.Load())
                            ),
                        ],
                        decorator_list=[],
                    )
                else:
                    else_body: list[ast.stmt] = []
                    with Region(self.session_data, new_value=else_body):
                        with self.session_data.scope_manager.enter_control_flow_scope():
                            self._visit_stmts_into(node.orelse, else_body)

                    # Regular else block
                    else_block = ast.FunctionDef(
                        name=else_block_name,
                        args=func_then_else_arguments,
                        body=else_body + [ast.Return(value=return_list)],
                        decorator_list=[],
                    )
            else:
                # Default else block
                else_block = ast.FunctionDef(
                    name=else_block_name,
                    args=func_then_else_arguments,
                    body=[ast.Return(value=return_list)],
                    decorator_list=[],
                )

            ast.copy_location(else_block, node)
            # Add else_block to execute keywords
            execute_keywords.append(
                ast.keyword(
                    arg="else_block", value=ast.Name(id=else_block_name, ctx=ast.Load())
                )
            )

            execute_call = ast.copy_location(
                ast.Call(
                    func=_create_module_attribute(
                        self.IF_EXECUTOR, lineno=node.lineno, col_offset=node.col_offset
                    ),
                    args=[],
                    keywords=execute_keywords,
                ),
                node,
            )
            func_body = [
                then_block,
                else_block,
                ast.Return(value=execute_call),
            ]

        return ast.copy_location(
            ast.FunctionDef(
                name=func_name,
                args=func_decorator_arguments,
                body=func_body,
                decorator_list=[decorator],
            ),
            node,
        )

    def create_while_function(
        self,
        func_name: str,
        node: ast.While,
        write_args: list[str],
        full_write_args_count: int,
        mutated_names: Sequence[str] = (),
    ) -> ast.FunctionDef:
        """Create a while function that looks like:

        @while_selector(write_args=[])
        def while_region(write_args):
            def while_before_block(*write_args):
                # Note that during eval of pred can possibly alter yield_args
                return [pred, write_args]
            def while_after_block(*write_args):
                ...loop_body_transformed...
                return write_args
            return while_executor(write_args, full_write_args_count,
                while_before_block, while_after_block, write_args_names, mutated_names)
        write_args = while_region

        The executor builds ``scf.while`` from the two blocks (the before
        block yields the condition, the after block the carried values), or
        runs them as a Python loop when the condition is a Python bool.
        """

        # Section: decorator construction
        decorator_keywords = [
            ast.keyword(
                arg="write_args",
                value=self.generate_get_locals_or_none_call(write_args),
            ),
        ]
        decorator = ast.copy_location(
            ast.Call(
                func=_create_module_attribute(
                    self.DECORATOR_WHILE_STATEMENT,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ),
                args=[],
                keywords=decorator_keywords,
            ),
            node,
        )

        # Section: Shared initialization for before and after blocks
        while_before_block_name = f"while_before_block_{self.session_data.counter}"
        while_after_block_name = f"while_after_block_{self.session_data.counter}"
        self.session_data.counter += 1
        block_args_args = [ast.arg(arg=var, annotation=None) for var in write_args]
        block_args = ast.arguments(
            posonlyargs=[],
            args=block_args_args,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )

        yield_args_ast_name_list = ast.List(
            elts=[ast.Name(id=var, ctx=ast.Load()) for var in write_args],
            ctx=ast.Load(),
        )

        # Section: while_before_block FunctionDef, which contains condition
        while_before_stmts: list[ast.stmt] = []
        with Region(self.session_data, new_value=while_before_stmts):
            test_expr = ast.copy_location(self.visit(node.test), node.test)

        while_before_return_list = ast.List(
            elts=[test_expr, yield_args_ast_name_list],
            ctx=ast.Load(),
        )
        while_before_stmts.append(ast.Return(value=while_before_return_list))
        while_before_block = ast.copy_location(
            ast.FunctionDef(
                name=while_before_block_name,
                args=block_args,
                body=while_before_stmts,
                decorator_list=[],
            ),
            test_expr,
        )

        # Section: while_after_block FunctionDef, which contains loop body
        while_after_stmts: list[ast.stmt] = []
        with Region(self.session_data, new_value=while_after_stmts):
            self._visit_stmts_into(node.body, while_after_stmts)

        while_after_stmts.append(ast.Return(value=yield_args_ast_name_list))

        while_after_block = ast.copy_location(
            ast.FunctionDef(
                name=while_after_block_name,
                args=block_args,
                body=while_after_stmts,
                decorator_list=[],
            ),
            node,
        )

        # Section: Execute via executor
        execute_keywords = [
            ast.keyword(
                arg="write_args",
                value=ast.List(
                    elts=[ast.Name(id=arg, ctx=ast.Load()) for arg in write_args],
                    ctx=ast.Load(),
                ),
            ),
            ast.keyword(
                arg="full_write_args_count",
                value=ast.Constant(value=full_write_args_count),
            ),
            ast.keyword(
                arg="while_before_block",
                value=ast.Name(id=while_before_block_name, ctx=ast.Load()),
            ),
            ast.keyword(
                arg="while_after_block",
                value=ast.Name(id=while_after_block_name, ctx=ast.Load()),
            ),
            ast.keyword(
                arg="write_args_names",
                value=self._names_constant_list(write_args),
            ),
            ast.keyword(
                arg="mutated_names",
                value=self._names_constant_tuple(mutated_names),
            ),
        ]
        execute_call = ast.Call(
            func=_create_module_attribute(
                self.WHILE_EXECUTOR, lineno=node.lineno, col_offset=node.col_offset
            ),
            args=[],
            keywords=execute_keywords,
        )

        # Putting everything together, FunctionDef for while_region
        return ast.copy_location(
            ast.FunctionDef(
                name=func_name,
                args=block_args,
                body=[
                    while_before_block,
                    while_after_block,
                    ast.Return(value=execute_call),
                ],
                decorator_list=[decorator],
            ),
            node,
        )
