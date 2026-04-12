from __future__ import annotations

import argparse
import json
import math
import signal
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import sympy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.actions import ACTION_NAMES, apply_action
from data_gen.canonicalize import canonicalize
from data_gen.generate_backward import shortest_actions_to_goal


x, y, z = sympy.symbols("x y z")


@dataclass(frozen=True)
class FamilyNode:
    name: str
    expr_str: str
    structural: str


@dataclass(frozen=True)
class FamilyEdge:
    source: str
    action: str
    target: str


@dataclass(frozen=True)
class FamilyInstance:
    family: str
    params: dict[str, int]
    goal_name: str
    nodes: tuple[FamilyNode, ...]
    intended_spine: tuple[str, ...]


class VerificationTimeout(RuntimeError):
    pass


def trig_identity() -> sympy.Expr:
    return sympy.Add(
        sympy.Pow(sympy.sin(y), 2, evaluate=False),
        sympy.Pow(sympy.cos(y), 2, evaluate=False),
        evaluate=False,
    )


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
    raise VerificationTimeout("shortest-path verification timed out")


def verify_shortest_distance(state_str: str, goal_str: str, max_steps: int, timeout_seconds: int) -> tuple[int | None, tuple[str, ...]]:
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return shortest_actions_to_goal(state_str, goal_str, max_steps=max_steps)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def linear_term(symbol: sympy.Symbol, shift: int) -> sympy.Expr:
    return sympy.Add(symbol, shift, evaluate=False)


def linear_product(symbol: sympy.Symbol, shifts: tuple[int, ...]) -> sympy.Expr:
    return sympy.Mul(*(linear_term(symbol, shift) for shift in shifts), evaluate=False)


def rational_sum(symbol: sympy.Symbol, terms: list[tuple[int, int]]) -> sympy.Expr:
    return sympy.Add(
        *[
            sympy.Mul(sympy.Integer(coeff), sympy.Pow(linear_term(symbol, shift), -1, evaluate=False), evaluate=False)
            for coeff, shift in terms
        ],
        evaluate=False,
    )


def family_node(name: str, expr: sympy.Expr) -> FamilyNode:
    expr_str = str(expr)
    return FamilyNode(name=name, expr_str=expr_str, structural=canonicalize(expr_str).structural)


def build_block_a_nodes(coeff_a: int, coeff_b: int, shift_a: int, shift_b: int) -> dict[str, FamilyNode]:
    base = rational_sum(x, [(coeff_a, shift_a), (coeff_b, shift_b)])
    together = sympy.together(base)
    goal = sympy.expand(together)
    start = sympy.Mul(trig_identity(), base, evaluate=False)
    return {
        "goal": family_node("goal", goal),
        "together": family_node("together", together),
        "base": family_node("base", base),
        "start": family_node("start", start),
    }


def build_block_b_nodes(p: int, q: int, r: int, d: int) -> dict[str, FamilyNode]:
    reduced = sympy.Mul(
        linear_term(z, d),
        sympy.Pow(linear_product(z, (p, q)), -1, evaluate=False),
        evaluate=False,
    )
    state = sympy.Mul(
        sympy.expand(sympy.Mul(linear_term(z, r), linear_term(z, d), evaluate=False)),
        sympy.Pow(linear_product(z, (p, q, r)), -1, evaluate=False),
        evaluate=False,
    )
    goal = sympy.expand(reduced)
    return {
        "goal": family_node("goal", goal),
        "reduced": family_node("reduced", reduced),
        "state": family_node("state", state),
    }


def compose_sum(*exprs: str) -> sympy.Expr:
    return sympy.Add(*(sympy.sympify(expr, evaluate=False) for expr in exprs), evaluate=False)


def build_family_block_a(params: dict[str, int]) -> FamilyInstance:
    nodes = build_block_a_nodes(
        coeff_a=params["coeff_a"],
        coeff_b=params["coeff_b"],
        shift_a=params["shift_a"],
        shift_b=params["shift_b"],
    )
    return FamilyInstance(
        family="block_a_trig_merge_expand",
        params=params,
        goal_name="goal",
        nodes=tuple(nodes.values()),
        intended_spine=("start", "base", "together", "goal"),
    )


def build_family_block_b(params: dict[str, int]) -> FamilyInstance:
    nodes = build_block_b_nodes(params["p"], params["q"], params["r"], params["d"])
    return FamilyInstance(
        family="block_b_cancel_expand",
        params=params,
        goal_name="goal",
        nodes=tuple(nodes.values()),
        intended_spine=("state", "reduced", "goal"),
    )


def build_family_sum_a3_b1(params: dict[str, int]) -> FamilyInstance:
    block_a = build_block_a_nodes(2, 3, 1, 2)
    block_b = build_block_b_nodes(params["p"], params["q"], params["r"], params["d"])
    nodes = {
        "goal": family_node("goal", compose_sum(block_a["goal"].expr_str, block_b["reduced"].expr_str)),
        "a_goal_b_state": family_node("a_goal_b_state", compose_sum(block_a["goal"].expr_str, block_b["state"].expr_str)),
        "a_together_b_state": family_node("a_together_b_state", compose_sum(block_a["together"].expr_str, block_b["state"].expr_str)),
        "a_base_b_state": family_node("a_base_b_state", compose_sum(block_a["base"].expr_str, block_b["state"].expr_str)),
        "a_start_b_state": family_node("a_start_b_state", compose_sum(block_a["start"].expr_str, block_b["state"].expr_str)),
    }
    return FamilyInstance(
        family="sum_a3_b1_cancel",
        params=params,
        goal_name="goal",
        nodes=tuple(nodes.values()),
        intended_spine=("a_start_b_state", "a_base_b_state", "a_together_b_state", "a_goal_b_state", "goal"),
    )


def build_family_sum_a3_b2(params: dict[str, int]) -> FamilyInstance:
    block_a = build_block_a_nodes(2, 3, 1, 2)
    block_b = build_block_b_nodes(params["p"], params["q"], params["r"], params["d"])
    nodes = {
        "goal": family_node("goal", compose_sum(block_a["goal"].expr_str, block_b["goal"].expr_str)),
        "a_goal_b_reduced": family_node("a_goal_b_reduced", compose_sum(block_a["goal"].expr_str, block_b["reduced"].expr_str)),
        "a_goal_b_state": family_node("a_goal_b_state", compose_sum(block_a["goal"].expr_str, block_b["state"].expr_str)),
        "a_together_b_state": family_node("a_together_b_state", compose_sum(block_a["together"].expr_str, block_b["state"].expr_str)),
        "a_base_b_state": family_node("a_base_b_state", compose_sum(block_a["base"].expr_str, block_b["state"].expr_str)),
        "a_start_b_state": family_node("a_start_b_state", compose_sum(block_a["start"].expr_str, block_b["state"].expr_str)),
    }
    return FamilyInstance(
        family="sum_a3_b2_cancel_expand",
        params=params,
        goal_name="goal",
        nodes=tuple(nodes.values()),
        intended_spine=("a_start_b_state", "a_base_b_state", "a_together_b_state", "a_goal_b_state", "a_goal_b_reduced", "goal"),
    )


FAMILY_BUILDERS = {
    "block_a_trig_merge_expand": build_family_block_a,
    "block_b_cancel_expand": build_family_block_b,
    "sum_a3_b1_cancel": build_family_sum_a3_b1,
    "sum_a3_b2_cancel_expand": build_family_sum_a3_b2,
}


def iter_block_a_params(limit: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    coeff_pairs = [(1, 1), (1, 2), (2, 3)]
    shift_pairs = [(1, 2), (1, 3), (2, 4), (-1, 2)]
    for coeff_a, coeff_b in coeff_pairs:
        for shift_a, shift_b in shift_pairs:
            out.append(
                {
                    "coeff_a": coeff_a,
                    "coeff_b": coeff_b,
                    "shift_a": shift_a,
                    "shift_b": shift_b,
                }
            )
            if len(out) >= limit:
                return out
    return out


def iter_block_b_params(limit: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for p in [1, 2, 3, 4]:
        for q in [2, 3, 4, 5]:
            for r in [3, 4, 5, 6]:
                for d in [1, 2, 6, 7]:
                    if len({p, q, r, d}) < 4:
                        continue
                    key = (p, q, r, d)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"p": p, "q": q, "r": r, "d": d})
                    if len(out) >= limit:
                        return out
    return out


def family_param_grid(family: str, per_family_limit: int) -> list[dict[str, int]]:
    if family == "block_a_trig_merge_expand":
        return iter_block_a_params(per_family_limit)
    if family in {"block_b_cancel_expand", "sum_a3_b1_cancel", "sum_a3_b2_cancel_expand"}:
        return iter_block_b_params(per_family_limit)
    raise KeyError(f"Unsupported family: {family}")


def build_induced_edges(instance: FamilyInstance) -> tuple[list[FamilyEdge], dict[str, list[FamilyEdge]]]:
    struct_to_name = {node.structural: node.name for node in instance.nodes}
    edges: list[FamilyEdge] = []
    outgoing: dict[str, list[FamilyEdge]] = {node.name: [] for node in instance.nodes}
    for node in instance.nodes:
        for action in ACTION_NAMES:
            result = apply_action(node.expr_str, action)
            if not result.valid or result.output_expr is None:
                continue
            next_struct = canonicalize(str(result.output_expr)).structural
            target_name = struct_to_name.get(next_struct)
            if target_name is None or target_name == node.name:
                continue
            edge = FamilyEdge(source=node.name, action=action, target=target_name)
            edges.append(edge)
            outgoing[node.name].append(edge)
    deduped = {(edge.source, edge.action, edge.target): edge for edge in edges}
    edges = list(deduped.values())
    outgoing = {name: [] for name in outgoing}
    for edge in edges:
        outgoing[edge.source].append(edge)
    return edges, outgoing


def shortest_induced_distance(goal_name: str, outgoing: dict[str, list[FamilyEdge]]) -> dict[str, int]:
    reverse: dict[str, list[str]] = {name: [] for name in outgoing}
    for source, edges in outgoing.items():
        for edge in edges:
            reverse[edge.target].append(source)
    distances = {goal_name: 0}
    queue: deque[str] = deque([goal_name])
    while queue:
        current = queue.popleft()
        for previous in reverse[current]:
            if previous in distances:
                continue
            distances[previous] = distances[current] + 1
            queue.append(previous)
    return distances


def entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return entropy


def analyze_instance(instance: FamilyInstance, max_steps: int, timeout_seconds: int) -> dict[str, object]:
    goal = next(node for node in instance.nodes if node.name == instance.goal_name)
    edges, outgoing = build_induced_edges(instance)
    induced_distances = shortest_induced_distance(instance.goal_name, outgoing)

    global_depths: dict[str, int | None] = {}
    optimal_actions: dict[str, tuple[str, ...]] = {}
    timeout_nodes: list[str] = []
    for node in instance.nodes:
        try:
            distance, actions = verify_shortest_distance(
                node.expr_str,
                goal.expr_str,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        except VerificationTimeout:
            distance, actions = None, ()
            timeout_nodes.append(node.name)
        global_depths[node.name] = distance
        optimal_actions[node.name] = actions

    depth_histogram = Counter(depth for depth in global_depths.values() if depth is not None)
    geodesic_action_counts: Counter[str] = Counter()
    shortcut_edges = 0
    geodesic_edges = 0
    decreasing_edges = 0
    leaking_edges = 0

    for edge in edges:
        source_depth = global_depths.get(edge.source)
        target_depth = global_depths.get(edge.target)
        if source_depth is None or target_depth is None:
            continue
        if target_depth < source_depth:
            decreasing_edges += 1
            geodesic_action_counts[edge.action] += 1
            if target_depth == source_depth - 1:
                geodesic_edges += 1
            if target_depth <= source_depth - 2:
                shortcut_edges += 1
        elif target_depth >= source_depth:
            leaking_edges += 1

    spine_depths = [global_depths.get(name) for name in instance.intended_spine]
    intended_spine_geodesic = all(
        current is not None and nxt is not None and current == nxt + 1
        for current, nxt in zip(spine_depths, spine_depths[1:], strict=False)
    )

    optimal_action_counts: Counter[str] = Counter()
    for node_name, actions in optimal_actions.items():
        if global_depths.get(node_name) in (None, 0):
            continue
        optimal_action_counts.update(actions)

    return {
        "family": instance.family,
        "params": instance.params,
        "goal_name": instance.goal_name,
        "goal": goal.expr_str,
        "timeout_nodes": timeout_nodes,
        "node_count": len(instance.nodes),
        "edge_count": len(edges),
        "nodes": [asdict(node) for node in instance.nodes],
        "edges": [asdict(edge) for edge in sorted(edges, key=lambda item: (item.source, item.action, item.target))],
        "global_depths": global_depths,
        "induced_distances": induced_distances,
        "depth_histogram": dict(sorted(depth_histogram.items())),
        "max_global_depth": max((depth for depth in global_depths.values() if depth is not None), default=None),
        "depth_ge_4_nodes": sorted([name for name, depth in global_depths.items() if depth is not None and depth >= 4]),
        "optimal_actions": {name: list(actions) for name, actions in optimal_actions.items()},
        "optimal_action_entropy_bits": entropy_from_counter(optimal_action_counts),
        "geodesic_action_entropy_bits": entropy_from_counter(geodesic_action_counts),
        "geodesic_edges": geodesic_edges,
        "shortcut_edges": shortcut_edges,
        "shortcut_density": (shortcut_edges / decreasing_edges) if decreasing_edges else 0.0,
        "leaking_edges": leaking_edges,
        "intended_spine": list(instance.intended_spine),
        "intended_spine_depths": spine_depths,
        "intended_spine_geodesic": intended_spine_geodesic,
    }


def summarize_family(results: list[dict[str, object]]) -> dict[str, object]:
    max_depths = [result["max_global_depth"] for result in results if result["max_global_depth"] is not None]
    depth_ge_4 = sum(1 for result in results if result["depth_ge_4_nodes"])
    shortcut_densities = [float(result["shortcut_density"]) for result in results]
    action_entropies = [float(result["optimal_action_entropy_bits"]) for result in results]
    geodesic_entropies = [float(result["geodesic_action_entropy_bits"]) for result in results]
    spine_geodesics = sum(1 for result in results if result["intended_spine_geodesic"])
    return {
        "instances": len(results),
        "max_depth_observed": max(max_depths) if max_depths else None,
        "instances_with_depth_ge_4": depth_ge_4,
        "depth_ge_4_rate": (depth_ge_4 / len(results)) if results else 0.0,
        "mean_shortcut_density": (sum(shortcut_densities) / len(shortcut_densities)) if shortcut_densities else 0.0,
        "mean_optimal_action_entropy_bits": (sum(action_entropies) / len(action_entropies)) if action_entropies else 0.0,
        "mean_geodesic_action_entropy_bits": (sum(geodesic_entropies) / len(geodesic_entropies)) if geodesic_entropies else 0.0,
        "intended_spine_geodesic_rate": (spine_geodesics / len(results)) if results else 0.0,
    }


def run_miner(
    families: tuple[str, ...],
    per_family_limit: int,
    max_steps: int,
    timeout_seconds: int,
    examples_per_family: int,
) -> dict[str, object]:
    output: dict[str, object] = {
        "families": {},
        "settings": {
            "families": list(families),
            "per_family_limit": per_family_limit,
            "max_steps": max_steps,
            "timeout_seconds": timeout_seconds,
        },
    }
    for family in families:
        params_grid = family_param_grid(family, per_family_limit)
        results = [
            analyze_instance(FAMILY_BUILDERS[family](params), max_steps=max_steps, timeout_seconds=timeout_seconds)
            for params in params_grid
        ]
        ranked = sorted(
            results,
            key=lambda item: (
                int(item["max_global_depth"] or -1),
                len(item["depth_ge_4_nodes"]),
                -float(item["shortcut_density"]),
            ),
            reverse=True,
        )
        output["families"][family] = {
            "summary": summarize_family(results),
            "examples": ranked[:examples_per_family],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine local rewrite-graph structure over typed parametric families")
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILY_BUILDERS.keys()),
        choices=sorted(FAMILY_BUILDERS.keys()),
    )
    parser.add_argument("--per-family-limit", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--examples-per-family", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_miner(
        families=tuple(args.families),
        per_family_limit=args.per_family_limit,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
        examples_per_family=args.examples_per_family,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
