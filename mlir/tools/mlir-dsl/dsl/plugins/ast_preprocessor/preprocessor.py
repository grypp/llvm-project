# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The AST preprocessor: rewrites the source of a decorated function.

``DSLPreprocessor`` parses the source of a decorated function and rewrites
it.

A ``ScopeManager`` tracks the names bound so far. The
preprocessor is generic: the DSL supplies the executors
(``core.staging.Executor.set_functions``).
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
import types
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, ClassVar

from .... import ir
from ...core.common import DSLRuntimeError, DSLUserCodeError, _user_error as _dsl_user_error
from ...core.diagnostics import DiagId
from ...util.logger import log

__all__ = [
    "create_module_attribute",
    "DSLPreprocessor",
    "OrderedSet",
    "Region",
    "ScopeManager",
    "SessionData",
]

# The module the rewritten code imports as ``__base_dsl__``: every helper the
# rewrite references lives in the sibling ``helpers`` module of this package.
_AST_HELPERS_MODULE = f"{__package__}.helpers"

# A call ``<module>.<op>(...)`` whose ``<module>`` is one of the MLIR dialect
# modules receives its `Numeric` arguments downcast to raw ``ir.Value``.
_MLIR_DIALECTS_PACKAGE = f"{ir.__name__.rpartition('.')[0]}.dialects"


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
    """The names bound so far, per scope, during the traversal.

    ``scopes`` is a stack of name sets: a function pushes a
    local scope on it (``enter_local_scope``).
    """

    scopes: list[set[str]]

    @classmethod
    def create(cls) -> ScopeManager:
        return cls([])

    def add_to_scope(self, name: str) -> None:
        if name == "_":
            return
        self.scopes[-1].add(name)

    def get_active_symbols(self) -> list[set[str]]:
        return self.scopes.copy()

    @contextlib.contextmanager
    def enter_local_scope(self) -> Generator[None, None, None]:
        """Enter a new local variable scope.

        This is conceptually Python's local scope, such as within a function
        or class definition: a new, empty set is pushed onto the variable
        stack and popped again on exit.
        """
        self.scopes.append(set())
        try:
            yield
        finally:
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


@dataclass
class SessionData:
    """The per-function state of one preprocessing session.

    Created by ``DSLPreprocessor.get_session``; it tracks the traversal.
    """

    scope_manager: ScopeManager = field(default_factory=ScopeManager.create)
    file_name: str = "<unknown filename>"
    function_globals: dict[str, Any] | None = None
    import_top_module: bool = False
    region_stack: list[Region] = field(default_factory=list)




class DSLPreprocessor(ast.NodeTransformer):
    """The AST transformer behind the DSL decorators (``@jit``, ``@kernel``, ...).

    - Zero-argument ``super()`` and calls into MLIR dialect modules
      are routed through ``helpers`` and the client DSL package.
    - ``global`` is rejected, ``nonlocal`` must name a bound variable, ``_``
      cannot be read, and the DSL decorator must be the innermost one.

    :param client_module_name: The package the rewrite imports as
        ``__module_dsl__`` (the DSL's ``dsl_package_name`` when given, else the
        ``ast_preprocessor`` plugin's own package, where ``and_``/``or_``/
        ``not_``/``as_ir_value`` live), as path parts
    :param decorator_names: The decorators marking a function for the rewrite:
        the ``decorator_name`` of every decorator plugin of the DSL
        (``BaseDSL.__init__`` passes them); ``("jit",)`` by default
    """

    IMPLICIT_DOWNCAST_NUMERIC_TYPE = "as_ir_value"
    BUILTIN_REDIRECTOR = "redirect_builtin_function"

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
        decorator_names: Iterable[str] = ("jit",),
        helpers_module: str | None = None,
    ) -> None:
        super().__init__()
        # Persistent state
        self.processed_functions: set[Callable[..., Any]] = set()
        self.client_module_name = client_module_name
        # The module the rewrite imports as ``__base_dsl__``.
        self.helpers_module: str = (
            _AST_HELPERS_MODULE if helpers_module is None else helpers_module
        )
        # The decorators marking a function for the rewrite (``@m.jit``,
        # ``@m.kernel``, a sub-DSL's own); a function carrying none is left as
        # it is.
        self.decorator_names: frozenset[str] = frozenset(decorator_names)
        self._session_data: SessionData | None = None



    def _end_session(self) -> None:
        """End the current preprocessing session and clear session data."""
        self._session_data = None

    @contextlib.contextmanager
    def get_session(self) -> Generator[DSLPreprocessor, None, None]:
        try:
            self.start_session()
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

    def transform_function(
        self, func_name: str, function_pointer: Callable[..., Any]
    ) -> list[ast.stmt]:
        """Rewrite one decorated function into the statements of a module.

        Parses the source (:meth:`parse_function_source`), checks the
        decorator, injects the default values the source names, rewrites the
        definition with its decorators cleared, runs
        :meth:`post_visit_function_body`, prepends the runtime alias imports
        and, when the function has free variables, wraps it in a factory that
        binds them.

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
            tree, _ = self.parse_function_source(function_pointer)
        except (OSError, TypeError) as e:
            # No retrievable source (REPL / exec())
            raise self._user_error("no_source", func=func_name, cause=e)
        except DSLUserCodeError:
            raise
        except Exception as e:
            raise DSLRuntimeError(f"Failed to parse function {func_name}", cause=e)

        # Step 1.2 Check the decorator
        func_def = tree.body[0]
        if not self.check_decorator(func_def):
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
        assert isinstance(func_def, ast.FunctionDef)
        self._inject_default_arg_values(
            function_pointer, self._default_arg_source_names(func_def)
        )

        # Step 2. Transform the function. The definition is visited on its
        # own, decorators cleared: Python applied them before the rewrite ran
        # and the rewritten definition must not apply them again, so an
        # expression in an outer decorator is never rewritten either.
        # Statements hoisted out of the definition itself (a ternary in a
        # default value) precede it in the module.
        func_def.decorator_list = []
        hoisted: list[ast.stmt] = []
        with Region(self.session_data, new_value=hoisted):
            visited = self.visit(func_def)
        assert isinstance(visited, ast.FunctionDef)
        func_def = visited

        # Step 2.1 The pass over the rewritten definition.
        preamble = self.post_visit_function_body(func_def)

        # Step 3. Import the runtime aliases: the client DSL package (only when
        # a rewrite needs it) and the helpers module.
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
                names=[ast.alias(name=self.helpers_module, asname="__base_dsl__")]
            )
        )
        func_def.body = import_stmts + preamble + func_def.body
        transformed_tree = tree
        transformed_tree.body = [*hoisted, func_def]

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
        """Rewrite zero-argument ``super()`` and calls into MLIR dialect modules."""
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
            if func.id == "super" and node.args == [] and node.keywords == []:
                # A zero-argument ``super()`` is spelled out as
                # ``super(__class__, self)``: inside a generated region
                # function the implicit form has no class cell to read.
                # ``__class__`` is the free variable Python gives a method
                # that uses ``super()``, which the factory wrapper and the
                # closure cells of the original function carry over; the
                # defining class, not ``self.__class__``, so a call through
                # a subclass instance does not recurse into itself.
                return ast.copy_location(
                    ast.Call(
                        func=func,
                        args=node.args
                        + [
                            ast.Name(id="__class__", ctx=ast.Load()),
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
                        func=create_module_attribute(
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
                    func=create_module_attribute(
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
            raise self._user_error("underscore_read", node)
        else:
            self.generic_visit(node)
        return node

    # =============================================================================
    # Decorators and function definitions
    # =============================================================================

    def _dsl_decorator_name(self, decorator: ast.expr) -> str | None:
        """The DSL decorator *decorator* spells (``@m.jit``, ``@jit``, a call
        of either, or an alias such as ``my_jit = m.jit``).

        A plain name that is not a decorator name is looked up in the
        function's globals: the decorator objects carry the decorator's name
        as their ``__name__``, so an alias is recognised by what it is bound
        to, not by its spelling.
        """
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Attribute):
            name = decorator.attr
        elif isinstance(decorator, ast.Name):
            name = decorator.id
            if name not in self.decorator_names:
                bound = (self.session_data.function_globals or {}).get(name)
                if bound is not None:
                    name = getattr(bound, "__name__", name)
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
            raise self._user_error(
                "decorator_order",
                decorator_node,
                what=f"`{ast.unparse(decorator_node)}` above another decorator",
            )

        return dsl_decorator_index is not None

    def remove_dsl_decorator(self, decorator_list: list[ast.expr]) -> list[ast.expr]:
        """Drop the DSL decorators (``@jit``/``@jit(...)``, ``@kernel``, ...) from a list."""
        return [d for d in decorator_list if self._dsl_decorator_name(d) is None]

    def visit_Global(self, node: ast.Global) -> None:
        """``global`` is not compiled (the ``global`` entry of ``DIAGNOSTICS``)."""
        raise self._user_error("global", node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        """``nonlocal`` may only name a variable already bound in an enclosing scope."""
        active_symbols = self.session_data.scope_manager.get_active_symbols()
        nonlocal_names = OrderedSet(node.names)
        intersect = nonlocal_names.intersections(active_symbols)
        for name in node.names:
            if name not in intersect:
                raise self._user_error(
                    "nonlocal",
                    node,
                    what=f"`{ast.unparse(node)}`",
                    detail=f": `{name}` belongs to an enclosing function, which compiled code cannot assign to; pass it in as an argument and return the updated value",
                )
        self.generic_visit(node)
        return node

    def visit_FunctionDef(
        self, node: ast.FunctionDef
    ) -> ast.FunctionDef | list[ast.stmt]:
        """Visit a function body in its own scope; strip annotations and DSL decorators."""
        with self.session_data.scope_manager.enter_local_scope():
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

        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name):
                self.session_data.scope_manager.add_to_scope(item.optional_vars.id)
        return self.generic_visit(node)


    DIAGNOSTICS: ClassVar[dict[str, tuple[Any, dict[str, Any]]]] = {
        "no_source": (DiagId.UNSUP_NO_SOURCE, {}),
        "decorator_order": (
            DiagId.UNSUP_SYNTAX,
            {
                "detail": ": it must be the innermost decorator, written directly above `def`"
            },
        ),
        "underscore_read": (
            DiagId.UNSUP_SYNTAX,
            {
                "what": "Reading the throwaway name `_`",
                "detail": ": give the value a real name",
            },
        ),
        "global": (
            DiagId.UNSUP_SYNTAX,
            {
                "what": "`global`",
                "detail": ": compiled code cannot assign to a module-level variable; pass the value in as an argument and return the updated value",
            },
        ),
        "nonlocal": (DiagId.UNSUP_SYNTAX, {}),
    }


    def _user_error(
        self, reason: str, node: ast.AST | None = None, **fields: Any
    ) -> DSLUserCodeError:
        """The error for ``reason`` (a key of :attr:`DIAGNOSTICS`) at ``node``.

        The entry's id with its fixed fields and the ``fields`` of this
        occurrence, located at ``node`` in the file being rewritten (an
        explicit ``end_col_offset`` or other location field in ``fields``
        wins; without a node the error locates itself), built by the DSL
        being preprocessed: its ``user_error_class`` and ``diag_id_aliases``
        apply, as at every other raise site of the core.
        """
        diag_id, fixed = self.DIAGNOSTICS[reason]
        location: dict[str, Any] = {}
        if node is not None:
            location = dict(
                filename=self.session_data.file_name,
                lineno=getattr(node, "lineno", None),
                col_offset=getattr(node, "col_offset", None),
                end_col_offset=getattr(node, "end_col_offset", None),
            )
        return _dsl_user_error(diag_id, **{**location, **fixed, **fields})


    def create_session_data(self) -> SessionData:
        """The state of one preprocessing session; a subclass with more
        per-function state returns its own ``SessionData`` subclass."""
        return SessionData()


    def start_session(self) -> None:
        """Start a new preprocessing session by initializing session data."""
        self._session_data = self.create_session_data()
        # Track processed functions per preprocessing run, not for the entire
        # lifetime of this preprocessor instance, so a later session never
        # skips transforming a function entirely.
        self.processed_functions = set()


    def parse_function_source(
        self, function_pointer: Callable[..., Any]
    ) -> tuple[ast.Module, str]:
        """Parse the source of ``function_pointer`` into a module holding its
        definition, with lines and columns re-based on the source file.

        A sub-DSL that already holds the tree (a parse cache, a tree shared
        with another pass) overrides this and serves it; the rewrite mutates
        the tree it is given, so a cached tree is served as a copy.

        :return: ``(tree, file_name)``: the module whose single statement is
            the function definition, and the file it was read from
            (``"<unknown>"`` when the file is not known)
        :raises OSError: When the source cannot be read
        :raises TypeError: When the function has no retrievable source
        """
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
        return tree, file_name


    def post_visit_function_body(self, func_def: ast.FunctionDef) -> list[ast.stmt]:
        """A pass over the fully rewritten function definition.

        Called once the visitors are done and before the runtime alias
        imports are prepended; the statements returned open the body, right
        after those imports (none by default). A sub-DSL adds a preamble or
        inspects the rewritten body here without touching the visitors.
        """
        return []



def create_module_attribute(
    func_name: str,
    *,
    use_base_dsl: bool = True,
    submodule_name: str | None = None,
    lineno: int | None = None,
    col_offset: int | None = None,
) -> ast.Attribute:
    """Create the AST of a qualified attribute access through a runtime alias.

    The rewritten function imports two aliases: ``__base_dsl__`` is the
    helpers module (this package's ``helpers``, or the module a DSL names as
    ``DSLPreprocessor.helpers_module``); ``__module_dsl__`` is the client DSL
    package named by ``DSLPreprocessor.client_module_name`` (``and_``,
    ``or_``, ``not_``, ``as_ir_value``), used when ``use_base_dsl`` is False.
    A preprocessor subclass emitting a helper call of its own builds the
    callee with this function.

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


_create_module_attribute = create_module_attribute
