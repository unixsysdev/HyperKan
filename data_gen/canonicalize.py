from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import sympy
from sympy import Expr


@dataclass(frozen=True)
class CanonicalForm:
    expr: Expr
    structural: str
    semantic: str
    digest: str


def parse_expression(expression: str | Expr) -> Expr:
    if isinstance(expression, Expr):
        return expression
    return _parse_expression_cached(expression)


@lru_cache(maxsize=200_000)
def _parse_expression_cached(expression: str) -> Expr:
    return sympy.sympify(expression, evaluate=False)


def structural_string(expr: str | Expr) -> str:
    if isinstance(expr, Expr):
        return _structural_string_from_expr(expr)
    return _structural_string_cached(expr)


@lru_cache(maxsize=200_000)
def _structural_string_cached(expr: str) -> str:
    return _structural_string_from_expr(_parse_expression_cached(expr))


def _structural_string_from_expr(expr: Expr) -> str:
    # Reparse through regular SymPy evaluation to normalize literal-only noise
    # such as 1*1 or nested negative-one multipliers while preserving the
    # higher-level rewrite form differences we care about.
    normalized = sympy.sympify(str(expr), evaluate=True)
    return sympy.srepr(normalized)


def semantic_string(expr: str | Expr) -> str:
    if isinstance(expr, Expr):
        parsed = expr
        normalized = sympy.cancel(sympy.together(sympy.expand_power_base(parsed, force=True)))
        return sympy.srepr(sympy.simplify(normalized))
    return _semantic_string_cached(expr)


@lru_cache(maxsize=200_000)
def _semantic_string_cached(expr: str) -> str:
    parsed = _parse_expression_cached(expr)
    normalized = sympy.cancel(sympy.together(sympy.expand_power_base(parsed, force=True)))
    return sympy.srepr(sympy.simplify(normalized))


def canonicalize(expr: str | Expr) -> CanonicalForm:
    parsed = parse_expression(expr)
    structural = structural_string(parsed)
    semantic = semantic_string(parsed)
    digest = sha256(semantic.encode("utf-8")).hexdigest()
    return CanonicalForm(expr=parsed, structural=structural, semantic=semantic, digest=digest)


def equivalent(lhs: str | Expr, rhs: str | Expr) -> bool:
    if not isinstance(lhs, Expr) and not isinstance(rhs, Expr):
        return _equivalent_cached(lhs, rhs)
    left = parse_expression(lhs)
    right = parse_expression(rhs)
    try:
        return bool(sympy.simplify(left - right) == 0)
    except TypeError:
        return bool(left.equals(right))


@lru_cache(maxsize=200_000)
def _equivalent_cached(lhs: str, rhs: str) -> bool:
    left = _parse_expression_cached(lhs)
    right = _parse_expression_cached(rhs)
    try:
        return bool(sympy.simplify(left - right) == 0)
    except TypeError:
        return bool(left.equals(right))


def safe_distance_proxy(current: str | Expr, goal: str | Expr) -> int:
    current_struct = structural_string(current)
    goal_struct = structural_string(goal)
    if current_struct == goal_struct:
        return 0
    current_tokens = _tokenize_structural(current_struct)
    goal_tokens = _tokenize_structural(goal_struct)
    current_set = set(current_tokens)
    goal_set = set(goal_tokens)
    overlap = len(current_set & goal_set)
    return max(1, len(goal_set) + len(current_set) - (2 * overlap))


@lru_cache(maxsize=200_000)
def _tokenize_structural(structural: str) -> tuple[str, ...]:
    return tuple(structural.replace("(", " ").replace(")", " ").replace(",", " ").split())
