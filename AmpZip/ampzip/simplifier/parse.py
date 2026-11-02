"""The one way user text becomes a sympy expression.

Every entry point that accepts an expression string, the CLI, the web service, the input
check and the atomizer, parses it here. `sympify` evaluates
its argument with Python's `eval`, so a crafted string would run before any whitelist is
consulted. This module parses the string into a Python AST, admits only the node types an
arithmetic expression over bracket calls needs, and evaluates that AST in a namespace that
holds nothing but the expression's own callables and constants.

Two policies share the machinery. Under the strict policy of the atomizer only the listed
call names and bare names are admitted, string literals are allowed as call arguments for
the chain type, and anything else is a `ParseError`. Under the lenient policy of the input
check and its callers an unknown call name becomes an undefined `sympy.Function` and an
unknown bare name a `sympy.Symbol`, so the check can report them under its own `LIMIT_*`
tags exactly as it did when `sympify` produced the same objects. A few sympy names, `sqrt`,
`pi`, `E` and `Rational`, are resolved to their sympy meaning for the same reason, since
the check's coefficient-class rule expects to see them.

Nothing here can call anything outside the namespace it is given.

The admission check, the integer rewrite and the name collection are one traversal, and
integer literals become references to pre-built `Integer` objects, so the parse costs less
than sympy's own parser on the same strings.
"""
from __future__ import annotations

import ast
import sys
from typing import Dict, Iterable, Optional, Set

import sympy as sp


class ParseError(ValueError):
    """The text is not an admissible expression. `kind` names the failed rule."""

    def __init__(self, message: str, kind: str = "syntax"):
        super().__init__(message)
        self.kind = kind


_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Name, ast.Load, ast.List, ast.Tuple,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)

# Names the lenient policy resolves to sympy objects so that downstream checks see the same
# objects `sympify` used to build, a transcendental or an algebraic root.
_SYMPY_NAMES: Dict[str, object] = {
    "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "Rational": sp.Rational, "Integer": sp.Integer,
}

_INT_PREFIX = "_int_"

# A long sum parses to a left-nested BinOp chain one level deep per term, and both the
# admission visitor and `eval` would recurse once per term, hitting Python's recursion limit
# near 600 terms. Spines of at least _FLATTEN_MIN additive terms are therefore rewritten
# iteratively into one call `_amp_sum(t1, -t2, t3, ...)` that `eval_tree` binds to
# `sympy.Add`, so the parser puts no limit on the number of summands. Short sums are left as
# written.
_SUM_NAME = "_amp_sum"
_FLATTEN_MIN = 32


def _flatten_spine(node):
    """Rewrite one left-nested +/- spine into an `_amp_sum(...)` call when it is long."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))):
        return node
    terms = []
    cur = node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, (ast.Add, ast.Sub)):
        terms.append((isinstance(cur.op, ast.Sub), cur.right))
        cur = cur.left
    terms.append((False, cur))
    if len(terms) < _FLATTEN_MIN:
        return node
    terms.reverse()
    args = [ast.UnaryOp(op=ast.USub(), operand=t) if neg else t for neg, t in terms]
    call = ast.Call(func=ast.Name(id=_SUM_NAME, ctx=ast.Load()), args=args, keywords=[])
    ast.copy_location(call, node)
    return ast.fix_missing_locations(call)


def _flatten_sums(root):
    """Apply `_flatten_spine` everywhere, iteratively, with no recursion over the sum."""
    stack = [root]
    while stack:
        node = stack.pop()
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        value[i] = _flatten_spine(item)
                        stack.append(value[i])
            elif isinstance(value, ast.AST):
                new = _flatten_spine(value)
                setattr(node, field, new)
                stack.append(new)
    return root


class _Admit(ast.NodeTransformer):
    """One pass that refuses anything outside the grammar, rewrites integer literals to names
    bound to pre-built Integers, and records what the tree refers to."""

    def __init__(self, calls: Set[str], names: Set[str], lenient: bool, allow_str: bool):
        self.calls, self.names, self.lenient, self.allow_str = calls, names, lenient, allow_str
        self.ints: Set[int] = set()
        self.unknown_calls: Set[str] = set()
        self.unknown_names: Set[str] = set()
        self.call_ints: Set[int] = set()     # the integer literals in argument position of a call
        self._in_call_arg = 0

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            raise ParseError(
                f"disallowed syntax: {type(node).__name__} (only calls, integer literals and "
                f"+ - * / ** are permitted)", "syntax")
        return super().generic_visit(node)

    def visit_Call(self, node):  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise ParseError("only plain function calls are allowed", "syntax")
        if node.keywords:
            raise ParseError("keyword arguments are not allowed in the expression", "syntax")
        fid = node.func.id
        if fid == _SUM_NAME:
            # the parser's own long-sum call, whose arguments are ordinary terms
            node.args = [self.visit(a) for a in node.args]
            return node
        if fid not in self.calls:
            if not self.lenient:
                raise ParseError(f"only {', '.join(sorted(self.calls))} calls are allowed", "call")
            self.unknown_calls.add(fid)
        self._in_call_arg += 1
        node.args = [self.visit(a) for a in node.args]
        self._in_call_arg -= 1
        return node

    def visit_Name(self, node):  # noqa: N802
        if node.id not in self.names and node.id not in self.calls:
            if not self.lenient:
                raise ParseError(f"unknown name '{node.id}'", "name")
            self.unknown_names.add(node.id)
        return node

    def visit_Constant(self, node):  # noqa: N802
        v = node.value
        if isinstance(v, int) and not isinstance(v, bool):
            self.ints.add(v)
            if self._in_call_arg:
                self.call_ints.add(v)
            return ast.Name(id=f"{_INT_PREFIX}{v}", ctx=ast.Load(), lineno=node.lineno,
                            col_offset=node.col_offset, end_lineno=node.end_lineno,
                            end_col_offset=node.end_col_offset)
        if (self.allow_str and isinstance(v, str)) or (self.lenient and isinstance(v, float)):
            return node
        raise ParseError(f"unsupported literal {v!r}", "literal")

    def visit_BinOp(self, node):  # noqa: N802
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.BinOp) \
                and isinstance(node.right.op, ast.Pow):
            # `a ** b ** c` with literal exponents is a magnitude bomb that evaluates before
            # any size check could see it, and the admitted expressions do not need one.
            raise ParseError("chained exponentiation is not allowed", "tower")
        return self.generic_visit(node)


def parse_tree(raw: str, *, calls: Iterable[str], names: Iterable[str] = (),
               lenient: bool = False, allow_str: bool = False) -> ast.Expression:
    """Parse `raw` and admit only expression syntax over the given call and bare names.

    `calls` are the callable names allowed in call position and `names` the bare names
    allowed as values. Under `lenient`, other names are admitted too and become undefined
    functions or symbols at evaluation. String literals are admitted only with `allow_str`.
    Integer literals are always admitted. Float literals are admitted under `lenient`, where
    the input check rejects them with its own tag, and refused otherwise. The returned tree
    carries the integer literals that appear as call arguments in `tree.call_ints`.
    """
    if not raw or not raw.strip():
        raise ParseError("empty expression", "empty")
    # A newline is a token separator for Python's tokenizer, so a sum wrapped across lines at
    # top level is a SyntaxError. A space replaces it, since "" would fuse tokens.
    raw = raw.replace("\r", " ").replace("\n", " ")
    try:
        tree = ast.parse(raw, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise ParseError(f"could not parse expression: {e}", "syntax") from e
    tree = _flatten_sums(tree)
    admit = _Admit(set(calls), set(names), lenient, allow_str)
    try:
        tree = admit.visit(tree)
    except RecursionError as e:
        raise ParseError(_DEPTH_MESSAGE, "depth") from e
    tree.ints, tree.unknown_calls, tree.unknown_names = admit.ints, admit.unknown_calls, admit.unknown_names
    tree.call_ints, tree.lenient = admit.call_ints, lenient
    return tree


_DEPTH_MESSAGE = ("expression is too deeply nested to parse, deeper than Python's recursion limit of "
                  "%d levels. Long sums are flattened and do not count. Split it into smaller pieces "
                  "and simplify them separately." % sys.getrecursionlimit())


def eval_tree(tree: ast.Expression, names: Dict[str, object]) -> sp.Expr:
    """Evaluate a tree admitted by `parse_tree` in a namespace holding only `names`, the
    pre-built Integers for its literals and, under the lenient policy, an undefined Function
    or Symbol for each unknown name the tree refers to."""
    lenient = getattr(tree, "lenient", False)
    ns: Dict[str, object] = dict(_SYMPY_NAMES) if lenient else {}
    ns.update(names)
    for v in tree.ints:
        ns[f"{_INT_PREFIX}{v}"] = sp.Integer(v)
    for f in tree.unknown_calls:
        ns.setdefault(f, sp.Function(f))
    for s in tree.unknown_names:
        ns.setdefault(s, sp.Symbol(s))
    ns[_SUM_NAME] = lambda *xs: sp.Add(*xs)      # the flattened long-sum call of _flatten_sums
    try:
        # `parse_tree` flattened the long sums, so what remains is deep nesting.
        value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
    except ParseError:
        raise
    except RecursionError as e:
        raise ParseError(_DEPTH_MESSAGE, "depth") from e
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"could not evaluate expression: {type(e).__name__}: {e}", "eval") from e
    if not isinstance(value, sp.Basic):
        value = sp.sympify(value)      # a number or a sympy-compatible object
    return value


def parse_expr(raw: str, names: Dict[str, object], *, lenient: bool = False,
               allow_str: bool = False, calls: Optional[Iterable[str]] = None) -> sp.Expr:
    """Parse and evaluate `raw` over the callables and constants in `names`.

    `calls` defaults to the callable entries of `names`, and the remaining entries are the
    bare names such as `I`. `parse_tree` describes the policies.
    """
    if calls is None:
        calls = [k for k, v in names.items() if callable(v)]
    bare = [k for k in names if k not in set(calls)]
    tree = parse_tree(raw, calls=calls, names=bare, lenient=lenient, allow_str=allow_str)
    return eval_tree(tree, names)
