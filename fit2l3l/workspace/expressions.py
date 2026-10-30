"""Vectorised evaluation of selection / weight / variable expressions.

HistFitter (via ROOT ``TTree::Draw``) accepts C++-style expressions such as::

    (nSigLep==3 && NlepBL==3) && !(mSFOS>81000 && mSFOS<101000) && met/meff<0.2

This module evaluates such strings on numpy arrays read from ntuples, so the
cut strings of the old ``fitConfigSS3L.py`` can be used verbatim.  Python-style
expressions (``&``, ``|``, ``~``, ``and``, ``or``, ``not``) work as well.

Only a safe subset of Python is allowed: names (branches), numbers,
arithmetic / comparison / boolean operators, ternary ``a if c else b`` and a
whitelist of numpy functions.  No attribute access, no builtins.
"""

from __future__ import annotations

import ast
import functools
import re
from typing import Callable, Dict, Iterable, Mapping, Set

import numpy as np

__all__ = [
    "translate",
    "free_symbols",
    "evaluate",
    "compile_expression",
    "product",
    "and_",
    "ExpressionError",
]


class ExpressionError(ValueError):
    """Raised for malformed or unsupported expressions."""


# functions usable inside expressions (ROOT / C++ names included)
_FUNCTIONS: Dict[str, Callable] = {
    "abs": np.abs,
    "fabs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "pow": np.power,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "acos": np.arccos,
    "asin": np.arcsin,
    "cosh": np.cosh,
    "sinh": np.sinh,
    "tanh": np.tanh,
    "hypot": np.hypot,
    "min": np.minimum,
    "max": np.maximum,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "fmin": np.fmin,
    "fmax": np.fmax,
    "where": np.where,
    "clip": np.clip,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "sign": np.sign,
    "isnan": np.isnan,
    "isfinite": np.isfinite,
}

_CONSTANTS = {"true": True, "false": False, "True": True, "False": False, "pi": np.pi}

# C++ -> python token translation
_CPP_TOKEN = re.compile(r"&&|\|\||!(?!=)|TMath::|std::")


def translate(expression: str) -> str:
    """Translates a C++ / ROOT expression into python syntax.

    ``&&`` -> ``and``, ``||`` -> ``or``, ``!`` -> ``not`` (``!=`` untouched),
    ``TMath::Abs`` -> ``Abs`` (and ROOT capitalised function names are lowered).
    """

    def _repl(match: re.Match) -> str:
        token = match.group(0)
        return {"&&": " and ", "||": " or ", "!": " not "}.get(token, "")

    expr = _CPP_TOKEN.sub(_repl, expression.strip())
    # ROOT style capitalised function names, e.g. Abs(x), Sqrt(x)
    expr = re.sub(
        r"\b(Abs|Sqrt|Exp|Log|Log10|Power|Min|Max|Cos|Sin|Tan|ATan2|Hypot)\s*\(",
        lambda m: {"Power": "pow", "ATan2": "atan2"}.get(m.group(1), m.group(1).lower())
        + "(",
        expr,
    )
    return expr.strip()


class _Rewriter(ast.NodeTransformer):
    """Rewrites boolean logic into element-wise numpy calls."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:  # a and b and c
        self.generic_visit(node)
        func = "__and__" if isinstance(node.op, ast.And) else "__or__"
        return ast.copy_location(
            ast.Call(func=ast.Name(id=func, ctx=ast.Load()), args=node.values, keywords=[]),
            node,
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="__not__", ctx=ast.Load()),
                    args=[node.operand],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:  # a < b < c
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        parts = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            parts.append(ast.Compare(left=left, ops=[op], comparators=[right]))
            left = right
        return ast.copy_location(
            ast.Call(func=ast.Name(id="__and__", ctx=ast.Load()), args=parts, keywords=[]),
            node,
        )

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:  # b if c else d -> where(c,b,d)
        self.generic_visit(node)
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="__where__", ctx=ast.Load()),
                args=[node.test, node.body, node.orelse],
                keywords=[],
            ),
            node,
        )


_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.IfExp,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.keyword,
)


def _parse(expression: str) -> ast.Expression:
    text = translate(expression)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse expression '{expression}': {exc}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"unsupported syntax '{type(node).__name__}' in expression '{expression}'"
            )
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
        ):
            raise ExpressionError(f"unsupported function call in expression '{expression}'")
    return tree


@functools.lru_cache(maxsize=None)
def free_symbols(expression: str) -> frozenset:
    """Returns the set of branch names an expression depends on."""
    tree = _parse(expression)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return frozenset(names - called - set(_CONSTANTS))


def _and(*args):
    return functools.reduce(np.logical_and, args)


def _or(*args):
    return functools.reduce(np.logical_or, args)


def _not(arg):
    arg = np.asarray(arg)
    if arg.dtype == bool:
        return np.logical_not(arg)
    if np.issubdtype(arg.dtype, np.integer):
        # C++ "!x" on an integer means x == 0, python "~x" is a bitwise not; treat
        # both as logical negation which is what is meant in selection strings
        return arg == 0
    return np.logical_not(arg)


@functools.lru_cache(maxsize=None)
def compile_expression(expression: str):
    """Compiles an expression into a function ``f(arrays: Mapping) -> np.ndarray``."""
    tree = _Rewriter().visit(_parse(expression))
    ast.fix_missing_locations(tree)
    code = compile(tree, filename=f"<expr: {expression}>", mode="eval")
    symbols = free_symbols(expression)

    def _func(arrays: Mapping[str, np.ndarray], n_entries: int | None = None):
        namespace = dict(_FUNCTIONS)
        namespace.update(_CONSTANTS)
        namespace.update(
            {"__and__": _and, "__or__": _or, "__not__": _not, "__where__": np.where}
        )
        for sym in symbols:
            try:
                namespace[sym] = arrays[sym]
            except KeyError:
                raise ExpressionError(
                    f"branch '{sym}' needed by '{expression}' is not available"
                ) from None
        with np.errstate(divide="ignore", invalid="ignore"):
            result = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307
        result = np.asarray(result)
        if result.ndim == 0 and n_entries is not None:
            result = np.full(n_entries, result.item())
        return result

    return _func


def evaluate(expression: str, arrays: Mapping[str, np.ndarray], n_entries: int | None = None):
    """Evaluates ``expression`` on the arrays (dict branch -> numpy array)."""
    return compile_expression(expression)(arrays, n_entries)


def _wrap(expr: str) -> str:
    expr = str(expr).strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9.eE+-]+", expr):
        return expr
    return f"({expr})"


def product(factors: Iterable[str]) -> str:
    """Builds the product of weight factors, e.g. ``["w1","w2"] -> "w1*w2"``."""
    factors = [str(f) for f in factors if str(f).strip() not in ("", "1", "1.", "1.0")]
    if not factors:
        return "1.0"
    return "*".join(_wrap(f) for f in factors)


def and_(*cuts: str) -> str:
    """Combines selections with a logical AND (empty/None entries are skipped)."""
    cuts = [c for c in cuts if c and str(c).strip() not in ("1", "true", "True")]
    if not cuts:
        return ""
    if len(cuts) == 1:
        return str(cuts[0])
    return " && ".join(_wrap(c) for c in cuts)


def symbols_of(expressions: Iterable[str]) -> Set[str]:
    """Union of the free symbols of several expressions."""
    out: Set[str] = set()
    for expr in expressions:
        if expr:
            out |= set(free_symbols(expr))
    return out
