from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import sympy
from sympy import Expr

from data_gen.actions import ACTION_FNS, ACTION_NAMES
from data_gen.canonicalize import canonicalize, equivalent, parse_expression


MAX_SITE_DEPTH = 2
ADD_SLICE_LENGTHS = (2, 3)
PATH_SITE_TYPES = ("expr", "numerator", "denominator", "add_slice")
SITE_TYPE_PRIORITY = {
    "expr": 0,
    "numerator": 1,
    "denominator": 2,
    "add_slice": 3,
}

SITE_TYPE_TO_OPS = {
    "expr": ACTION_NAMES,
    "numerator": ("expand", "factor", "cancel", "trigsimp"),
    "denominator": ("expand", "factor", "cancel", "trigsimp"),
    "add_slice": ACTION_NAMES,
}

OP_PRIORITY = {
    "trigsimp": 0,
    "together": 1,
    "factor": 2,
    "cancel": 3,
    "expand": 4,
    "apart": 5,
}


@dataclass(frozen=True)
class ScopedSite:
    site_id: str
    site_type: str
    path: tuple[int, ...]
    expr_str: str
    root_type: str
    slice_bounds: tuple[int, int] | None = None


@dataclass(frozen=True)
class ScopedRewriteResult:
    action_id: str
    site_id: str
    op_name: str
    input_expr: Expr
    output_expr: Expr | None
    valid: bool
    changed: bool
    reason: str


@dataclass(frozen=True)
class ScopedSearchSummary:
    distance: int | None
    optimal_actions: tuple[str, ...]
    reason: str
    nodes_expanded: int


def site_action_id(site_id: str, op_name: str) -> str:
    return f"{site_id}::{op_name}"


def parse_scoped_action_id(action_id: str) -> tuple[str, str]:
    site_id, sep, op_name = action_id.partition("::")
    if not sep:
        raise ValueError(f"Invalid scoped action id: {action_id}")
    return site_id, op_name


def enumerate_sites(expr: str | Expr) -> list[ScopedSite]:
    expr_str = str(parse_expression(expr))
    return list(_enumerate_sites_cached(expr_str))


def enumerate_scoped_action_ids(expr: str | Expr) -> list[str]:
    action_ids: list[str] = []
    for site in enumerate_sites(expr):
        for op_name in candidate_ops_for_site(site):
            action_ids.append(site_action_id(site.site_id, op_name))
    return action_ids


def valid_scoped_actions(expr: str | Expr) -> list[str]:
    expr_str = str(parse_expression(expr))
    return [item[0] for item in _materialize_scoped_successors_cached(expr_str)]


def apply_scoped_action(expr: str | Expr, site_id: str, op_name: str) -> ScopedRewriteResult:
    parsed = parse_expression(expr)
    action_id = site_action_id(site_id, op_name)
    output_str, valid, changed, reason = _apply_scoped_action_cached(str(parsed), site_id, op_name)
    output_expr = parse_expression(output_str) if output_str is not None else None
    return ScopedRewriteResult(
        action_id=action_id,
        site_id=site_id,
        op_name=op_name,
        input_expr=parsed,
        output_expr=output_expr,
        valid=valid,
        changed=changed,
        reason=reason,
    )


def apply_scoped_action_unchecked(expr: str | Expr, site_id: str, op_name: str) -> Expr | None:
    if op_name not in ACTION_FNS:
        raise KeyError(f"Unknown scoped operator: {op_name}")
    expr_str = str(parse_expression(expr))
    parsed = parse_expression(expr_str)
    sites = {site.site_id: site for site in _enumerate_sites_cached(expr_str)}
    site = sites.get(site_id)
    if site is None or op_name not in SITE_TYPE_TO_OPS[site.site_type]:
        return None
    try:
        rewritten_site = ACTION_FNS[op_name](_extract_site_expr(parsed, site))
    except Exception:  # pragma: no cover
        return None
    return _replace_site(parsed, site, rewritten_site)


def search_scoped_actions_to_goal(
    state_str: str,
    goal_str: str,
    max_steps: int,
    max_nodes: int = 4_000,
    stop_at_first_solution: bool = False,
) -> ScopedSearchSummary:
    goal_struct = canonicalize(goal_str).structural
    state_struct = canonicalize(state_str).structural
    if state_struct == goal_struct:
        return ScopedSearchSummary(distance=0, optimal_actions=(), reason="goal_already_reached", nodes_expanded=0)

    shortest_depth: int | None = None
    shortest_first_actions: set[str] = set()
    nodes_expanded = 1
    remaining_node_budget = max_nodes - nodes_expanded
    goal_addend_structs = _top_level_addend_structs(goal_str)

    for action_id, next_expr, next_struct in _iter_scoped_successors(state_str, goal_addend_structs):
        if remaining_node_budget < 0:
            return ScopedSearchSummary(
                distance=None,
                optimal_actions=(),
                reason="node_cap",
                nodes_expanded=max_nodes,
            )

        if next_struct == goal_struct:
            shortest_depth = 1
            shortest_first_actions.add(action_id)
            if stop_at_first_solution:
                return ScopedSearchSummary(
                    distance=1,
                    optimal_actions=(action_id,),
                    reason="ok",
                    nodes_expanded=nodes_expanded,
                )
            continue

        if max_steps <= 1:
            continue
        if shortest_depth == 1:
            continue

        child_max_steps = max_steps - 1
        if shortest_depth is not None:
            child_max_steps = min(child_max_steps, shortest_depth - 2)
        if child_max_steps <= 0:
            continue

        child_summary = _search_scoped_distance_to_goal(
            next_expr,
            goal_struct,
            max_steps=child_max_steps,
            max_nodes=remaining_node_budget,
            goal_addend_structs=goal_addend_structs,
            stop_at_first_solution=stop_at_first_solution,
        )
        nodes_expanded += child_summary.nodes_expanded
        remaining_node_budget = max_nodes - nodes_expanded

        if child_summary.reason == "node_cap":
            return ScopedSearchSummary(
                distance=None,
                optimal_actions=(),
                reason="node_cap",
                nodes_expanded=nodes_expanded,
            )
        if child_summary.distance is None:
            continue

        candidate_depth = 1 + child_summary.distance
        if shortest_depth is None or candidate_depth < shortest_depth:
            shortest_depth = candidate_depth
            shortest_first_actions = {action_id}
            if stop_at_first_solution:
                return ScopedSearchSummary(
                    distance=candidate_depth,
                    optimal_actions=(action_id,),
                    reason="ok",
                    nodes_expanded=nodes_expanded,
                )
        elif candidate_depth == shortest_depth:
            shortest_first_actions.add(action_id)

    if shortest_depth is None:
        return ScopedSearchSummary(distance=None, optimal_actions=(), reason="no_path", nodes_expanded=nodes_expanded)
    return ScopedSearchSummary(
        distance=shortest_depth,
        optimal_actions=tuple(sorted(shortest_first_actions)),
        reason="ok",
        nodes_expanded=nodes_expanded,
    )


@lru_cache(maxsize=200_000)
def shortest_scoped_actions_to_goal(state_str: str, goal_str: str, max_steps: int) -> tuple[int | None, tuple[str, ...]]:
    summary = search_scoped_actions_to_goal(state_str, goal_str, max_steps=max_steps)
    return summary.distance, summary.optimal_actions


@lru_cache(maxsize=200_000)
def _enumerate_sites_cached(expr_str: str) -> tuple[ScopedSite, ...]:
    parsed = parse_expression(expr_str)
    sites: list[ScopedSite] = []
    seen_site_ids: set[str] = set()

    def add_site(site_type: str, path: tuple[int, ...], node: Expr, slice_bounds: tuple[int, int] | None = None) -> None:
        site_id = _make_site_id(site_type, path, slice_bounds)
        if site_id in seen_site_ids:
            return

        if site_type == "expr":
            expr_repr = str(node)
        elif site_type in {"numerator", "denominator"}:
            numerator, denominator = sympy.fraction(node)
            if denominator == 1:
                return
            expr_repr = str(numerator if site_type == "numerator" else denominator)
        elif site_type == "add_slice":
            if not isinstance(node, sympy.Add) or slice_bounds is None:
                return
            start, end = slice_bounds
            expr_repr = str(sympy.Add(*node.args[start:end], evaluate=False))
        else:
            return

        sites.append(
            ScopedSite(
                site_id=site_id,
                site_type=site_type,
                path=path,
                expr_str=expr_repr,
                root_type=type(node).__name__,
                slice_bounds=slice_bounds,
            )
        )
        seen_site_ids.add(site_id)

    def walk(node: Expr, path: tuple[int, ...], depth: int) -> None:
        if _whitelisted_subtree(node, depth):
            add_site("expr", path, node)
            numerator, denominator = sympy.fraction(node)
            if denominator != 1:
                add_site("numerator", path, node)
                add_site("denominator", path, node)
            if isinstance(node, sympy.Add):
                for length in ADD_SLICE_LENGTHS:
                    if len(node.args) < length:
                        continue
                    for start in range(0, len(node.args) - length + 1):
                        add_site("add_slice", path, node, (start, start + length))

        if depth >= MAX_SITE_DEPTH:
            return
        for idx, child in enumerate(node.args):
            if not isinstance(child, Expr):
                continue
            walk(child, (*path, idx), depth + 1)

    walk(parsed, (), 0)
    return tuple(sites)


@lru_cache(maxsize=200_000)
def _materialize_scoped_successors_cached(expr_str: str) -> tuple[tuple[str, str, str], ...]:
    return tuple(_iter_scoped_successors(expr_str))


def _iter_scoped_successors(
    expr_str: str,
    goal_addend_structs: frozenset[str] | None = None,
):
    seen_structures: set[str] = set()
    ordered_sites = sorted(
        _enumerate_sites_cached(expr_str),
        key=lambda site: _site_search_priority(expr_str, site, goal_addend_structs),
    )
    for site in ordered_sites:
        for op_name in candidate_ops_for_site(site):
            output_str, ok, _, _ = _apply_scoped_action_cached(expr_str, site.site_id, op_name)
            if not ok or output_str is None:
                continue
            next_struct = canonicalize(output_str).structural
            if next_struct in seen_structures:
                continue
            seen_structures.add(next_struct)
            yield site_action_id(site.site_id, op_name), output_str, next_struct


def _site_search_priority(
    expr_str: str,
    site: ScopedSite,
    goal_addend_structs: frozenset[str] | None,
) -> tuple[int, int, int, int, int, int]:
    solved_addend_penalty = 1 if _inside_solved_goal_addend(expr_str, site.path, goal_addend_structs) else 0
    root_global_penalty = 1 if not site.path and site.site_type == "expr" and site.root_type == "Add" else 0
    whole_root_slice_penalty = 1 if _is_whole_root_add_slice(expr_str, site) else 0
    return (
        solved_addend_penalty,
        whole_root_slice_penalty,
        root_global_penalty,
        len(site.path),
        SITE_TYPE_PRIORITY[site.site_type],
        len(site.expr_str),
    )


def _search_scoped_distance_to_goal(
    state_str: str,
    goal_struct: str,
    max_steps: int,
    max_nodes: int,
    goal_addend_structs: frozenset[str] | None = None,
    stop_at_first_solution: bool = False,
) -> ScopedSearchSummary:
    state_struct = canonicalize(state_str).structural
    if state_struct == goal_struct:
        return ScopedSearchSummary(distance=0, optimal_actions=(), reason="goal_already_reached", nodes_expanded=0)
    if stop_at_first_solution:
        return _search_first_scoped_path_to_goal(
            state_str,
            goal_struct,
            max_steps=max_steps,
            max_nodes=max_nodes,
            goal_addend_structs=goal_addend_structs,
        )

    queue: deque[tuple[str, int]] = deque([(state_str, 0)])
    visited: dict[str, int] = {state_struct: 0}
    nodes_expanded = 0

    while queue:
        current, depth = queue.popleft()
        if nodes_expanded >= max_nodes:
            return ScopedSearchSummary(
                distance=None,
                optimal_actions=(),
                reason="node_cap",
                nodes_expanded=nodes_expanded,
            )
        nodes_expanded += 1

        if depth >= max_steps:
            continue

        for _, next_expr, next_struct in _iter_scoped_successors(current, goal_addend_structs):
            candidate_depth = depth + 1
            if next_struct == goal_struct:
                return ScopedSearchSummary(
                    distance=candidate_depth,
                    optimal_actions=(),
                    reason="ok",
                    nodes_expanded=nodes_expanded,
                )

            prior_depth = visited.get(next_struct)
            if prior_depth is not None and prior_depth <= candidate_depth:
                continue
            visited[next_struct] = candidate_depth
            queue.append((next_expr, candidate_depth))

    return ScopedSearchSummary(distance=None, optimal_actions=(), reason="no_path", nodes_expanded=nodes_expanded)


def _search_first_scoped_path_to_goal(
    state_str: str,
    goal_struct: str,
    max_steps: int,
    max_nodes: int,
    goal_addend_structs: frozenset[str] | None,
) -> ScopedSearchSummary:
    visited: set[str] = set()
    nodes_expanded = 0

    def dfs(current: str, depth: int) -> int | None:
        nonlocal nodes_expanded
        current_struct = canonicalize(current).structural
        if current_struct == goal_struct:
            return 0
        if depth >= max_steps:
            return None
        if nodes_expanded >= max_nodes:
            return None

        nodes_expanded += 1
        visited.add(current_struct)

        for _, next_expr, next_struct in _iter_scoped_successors(current, goal_addend_structs):
            if next_struct in visited:
                continue
            if next_struct == goal_struct:
                return 1
            child_distance = dfs(next_expr, depth + 1)
            if child_distance is not None:
                return child_distance + 1
        return None

    distance = dfs(state_str, 0)
    if distance is not None:
        return ScopedSearchSummary(distance=distance, optimal_actions=(), reason="ok", nodes_expanded=nodes_expanded)
    reason = "node_cap" if nodes_expanded >= max_nodes else "no_path"
    return ScopedSearchSummary(distance=None, optimal_actions=(), reason=reason, nodes_expanded=nodes_expanded)


@lru_cache(maxsize=200_000)
def _top_level_addend_structs(expr_str: str) -> frozenset[str]:
    parsed = parse_expression(expr_str)
    if not isinstance(parsed, sympy.Add):
        return frozenset()
    return frozenset(canonicalize(str(arg)).structural for arg in parsed.args if isinstance(arg, Expr))


@lru_cache(maxsize=200_000)
def _inside_solved_goal_addend(
    expr_str: str,
    path: tuple[int, ...],
    goal_addend_structs: frozenset[str] | None,
) -> bool:
    if not path or not goal_addend_structs:
        return False
    parsed = parse_expression(expr_str)
    if not isinstance(parsed, sympy.Add):
        return False
    first_index = path[0]
    if first_index >= len(parsed.args):
        return False
    return canonicalize(str(parsed.args[first_index])).structural in goal_addend_structs


@lru_cache(maxsize=200_000)
def _is_whole_root_add_slice(expr_str: str, site: ScopedSite) -> bool:
    if site.site_type != "add_slice" or site.path or site.slice_bounds is None:
        return False
    parsed = parse_expression(expr_str)
    if not isinstance(parsed, sympy.Add):
        return False
    start, end = site.slice_bounds
    return start == 0 and end == len(parsed.args)


def candidate_ops_for_site(site: ScopedSite) -> tuple[str, ...]:
    return _candidate_ops_for_site_cached(
        site.site_type,
        site.root_type,
        site.expr_str,
        site.slice_bounds,
    )


@lru_cache(maxsize=200_000)
def _candidate_ops_for_site_cached(
    site_type: str,
    root_type: str,
    expr_str: str,
    slice_bounds: tuple[int, int] | None,
) -> tuple[str, ...]:
    del slice_bounds  # included for cache key stability if site behavior widens later
    allowed = SITE_TYPE_TO_OPS[site_type]
    parsed = parse_expression(expr_str)
    has_trig = _contains_trig(parsed)
    is_rational = _is_rational_expr(parsed)
    is_add = isinstance(parsed, sympy.Add)
    is_mul = isinstance(parsed, sympy.Mul)
    is_pow = isinstance(parsed, sympy.Pow)
    has_expandable_form = is_add or is_mul or is_pow
    has_factorable_form = is_add or is_mul or is_pow or has_trig or is_rational
    can_together = is_add and _contains_rational_term(parsed)
    can_apart = is_rational and not has_trig
    can_cancel = is_rational

    feasible_ops: list[str] = []
    for op_name in allowed:
        if op_name == "trigsimp" and not has_trig:
            continue
        if op_name == "apart" and not can_apart:
            continue
        if op_name == "together" and not can_together:
            continue
        if op_name == "cancel" and not can_cancel:
            continue
        if op_name == "expand" and not has_expandable_form:
            continue
        if op_name == "factor" and not has_factorable_form:
            continue
        feasible_ops.append(op_name)
    return tuple(sorted(feasible_ops, key=lambda op_name: OP_PRIORITY[op_name]))


@lru_cache(maxsize=200_000)
def _apply_scoped_action_cached(expr_str: str, site_id: str, op_name: str) -> tuple[str | None, bool, bool, str]:
    if op_name not in ACTION_FNS:
        raise KeyError(f"Unknown scoped operator: {op_name}")

    parsed = parse_expression(expr_str)
    sites = {site.site_id: site for site in _enumerate_sites_cached(expr_str)}
    site = sites.get(site_id)
    if site is None:
        return None, False, False, "unknown_site"
    if op_name not in SITE_TYPE_TO_OPS[site.site_type]:
        return None, False, False, "op_not_allowed_for_site"

    target_expr = _extract_site_expr(parsed, site)
    try:
        rewritten_site = ACTION_FNS[op_name](target_expr)
    except Exception as exc:  # pragma: no cover
        return None, False, False, f"exception:{type(exc).__name__}"

    full_candidate = _replace_site(parsed, site, rewritten_site)
    before = canonicalize(parsed)
    after = canonicalize(full_candidate)
    changed = before.structural != after.structural
    if not changed:
        return str(full_candidate), False, False, "no_op"
    if not equivalent(parsed, full_candidate):
        return str(full_candidate), False, False, "not_equivalent"
    return str(full_candidate), True, True, "ok"


def _make_site_id(site_type: str, path: tuple[int, ...], slice_bounds: tuple[int, int] | None = None) -> str:
    path_text = "root" if not path else ".".join(str(item) for item in path)
    if site_type == "add_slice" and slice_bounds is not None:
        start, end = slice_bounds
        return f"{site_type}@{path_text}[{start}:{end}]"
    return f"{site_type}@{path_text}"


def _extract_path(expr: Expr, path: tuple[int, ...]) -> Expr:
    current = expr
    for index in path:
        current = current.args[index]
    return current


def _replace_at_path(expr: Expr, path: tuple[int, ...], new_subexpr: Expr) -> Expr:
    if not path:
        return new_subexpr
    index = path[0]
    args = list(expr.args)
    args[index] = _replace_at_path(args[index], path[1:], new_subexpr)
    return _rebuild_expr(expr, args)


def _extract_site_expr(expr: Expr, site: ScopedSite) -> Expr:
    node = _extract_path(expr, site.path)
    if site.site_type == "expr":
        return node
    if site.site_type == "add_slice":
        if not isinstance(node, sympy.Add) or site.slice_bounds is None:
            return node
        start, end = site.slice_bounds
        return sympy.Add(*node.args[start:end], evaluate=False)
    numerator, denominator = sympy.fraction(node)
    return numerator if site.site_type == "numerator" else denominator


def _replace_site(expr: Expr, site: ScopedSite, rewritten_site: Expr) -> Expr:
    node = _extract_path(expr, site.path)
    if site.site_type == "expr":
        return _replace_at_path(expr, site.path, rewritten_site)

    if site.site_type == "add_slice":
        if not isinstance(node, sympy.Add) or site.slice_bounds is None:
            return expr
        start, end = site.slice_bounds
        new_args = [*node.args[:start], rewritten_site, *node.args[end:]]
        rebuilt = sympy.Add(*new_args, evaluate=False)
        return _replace_at_path(expr, site.path, rebuilt)

    numerator, denominator = sympy.fraction(node)
    if denominator == 1:
        return expr
    if site.site_type == "numerator":
        rebuilt = _rebuild_fraction(rewritten_site, denominator)
    else:
        rebuilt = _rebuild_fraction(numerator, rewritten_site)
    return _replace_at_path(expr, site.path, rebuilt)


def _rebuild_expr(expr: Expr, args: list[Expr]) -> Expr:
    if isinstance(expr, sympy.Add):
        return sympy.Add(*args, evaluate=False)
    if isinstance(expr, sympy.Mul):
        return sympy.Mul(*args, evaluate=False)
    if isinstance(expr, sympy.Pow):
        return sympy.Pow(args[0], args[1], evaluate=False)
    return expr.func(*args)


def _rebuild_fraction(numerator: Expr, denominator: Expr) -> Expr:
    if denominator == 1:
        return numerator
    return sympy.Mul(numerator, sympy.Pow(denominator, -1, evaluate=False), evaluate=False)


def _whitelisted_subtree(node: Expr, depth: int) -> bool:
    if depth == 0:
        return True
    if isinstance(node, (sympy.Add, sympy.Mul)):
        return True
    if _is_rational_expr(node):
        return True
    if _contains_trig(node):
        return True
    return False


def _is_rational_expr(node: Expr) -> bool:
    _, denominator = sympy.fraction(node)
    return denominator != 1


def _contains_rational_term(node: Expr) -> bool:
    if not isinstance(node, sympy.Add):
        return _is_rational_expr(node)
    return any(_is_rational_expr(term) for term in node.args if isinstance(term, Expr))


def _contains_trig(node: Expr) -> bool:
    return bool(node.has(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc))
