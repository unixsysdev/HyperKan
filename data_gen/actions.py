from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import sympy
from sympy import Expr

from data_gen.canonicalize import canonicalize, equivalent, parse_expression


ActionFn = Callable[[Expr], Expr]

ACTION_NAMES = (
    "expand",
    "factor",
    "cancel",
    "apart",
    "together",
    "trigsimp",
)


def _expand(expr: Expr) -> Expr:
    return sympy.expand(expr, trig=True)


def _factor(expr: Expr) -> Expr:
    return sympy.factor(expr)


def _cancel(expr: Expr) -> Expr:
    return sympy.cancel(expr)


def _apart(expr: Expr) -> Expr:
    symbols = sorted(expr.free_symbols, key=lambda sym: sym.name)
    if not symbols:
        return expr
    return sympy.apart(expr, symbols[0], full=False)


def _together(expr: Expr) -> Expr:
    return sympy.together(expr)


def _trigsimp(expr: Expr) -> Expr:
    return sympy.trigsimp(expr)


ACTION_FNS: dict[str, ActionFn] = {
    "expand": _expand,
    "factor": _factor,
    "cancel": _cancel,
    "apart": _apart,
    "together": _together,
    "trigsimp": _trigsimp,
}

ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}
ID_TO_ACTION = {idx: name for name, idx in ACTION_TO_ID.items()}


@dataclass(frozen=True)
class RewriteResult:
    action: str
    input_expr: Expr
    output_expr: Expr | None
    valid: bool
    changed: bool
    reason: str


def apply_action(expr: str | Expr, action: str) -> RewriteResult:
    if action not in ACTION_FNS:
        raise KeyError(f"Unknown action: {action}")

    parsed = parse_expression(expr)
    output_str, valid, changed, reason = _apply_action_cached(str(parsed), action)
    output_expr = parse_expression(output_str) if output_str is not None else None
    return RewriteResult(
        action=action,
        input_expr=parsed,
        output_expr=output_expr,
        valid=valid,
        changed=changed,
        reason=reason,
    )


@lru_cache(maxsize=200_000)
def _apply_action_cached(expr_str: str, action: str) -> tuple[str | None, bool, bool, str]:
    parsed = parse_expression(expr_str)
    before = canonicalize(parsed)
    try:
        candidate = ACTION_FNS[action](parsed)
    except Exception as exc:  # pragma: no cover - defensive against SymPy edge cases
        return None, False, False, f"exception:{type(exc).__name__}"

    after = canonicalize(candidate)
    changed = before.structural != after.structural
    if not changed:
        return str(candidate), False, False, "no_op"

    if not equivalent(parsed, candidate):
        return str(candidate), False, False, "not_equivalent"

    return str(candidate), True, True, "ok"


def action_mask(expr: str | Expr) -> list[bool]:
    expr_str = str(parse_expression(expr))
    return list(_action_mask_cached(expr_str))


def valid_actions(expr: str | Expr) -> list[str]:
    expr_str = str(parse_expression(expr))
    mask = _action_mask_cached(expr_str)
    return [action for action, is_valid in zip(ACTION_NAMES, mask, strict=False) if is_valid]


@lru_cache(maxsize=200_000)
def _action_mask_cached(expr_str: str) -> tuple[bool, ...]:
    return tuple(_apply_action_cached(expr_str, action)[1] for action in ACTION_NAMES)
