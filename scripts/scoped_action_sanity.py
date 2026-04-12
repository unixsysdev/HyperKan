from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import sympy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.canonicalize import canonicalize
from data_gen.scoped_actions import enumerate_sites, search_scoped_actions_to_goal


x, y, z = sympy.symbols("x y z")


class VerificationTimeout(RuntimeError):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
    raise VerificationTimeout("scoped shortest-path verification timed out")


def verify_shortest_scoped(
    state_str: str,
    goal_str: str,
    max_steps: int,
    timeout_seconds: int,
    max_nodes: int = 4_000,
    stop_at_first_solution: bool = False,
) -> dict[str, object]:
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        summary = search_scoped_actions_to_goal(
            state_str,
            goal_str,
            max_steps=max_steps,
            max_nodes=max_nodes,
            stop_at_first_solution=stop_at_first_solution,
        )
        return {
            "distance": summary.distance,
            "optimal_actions": list(summary.optimal_actions),
            "reason": summary.reason,
            "nodes_expanded": summary.nodes_expanded,
            "mode": "guided_first_path" if stop_at_first_solution else "strict_bounded",
        }
    except VerificationTimeout:
        return {
            "distance": None,
            "optimal_actions": [],
            "reason": "timeout",
            "nodes_expanded": None,
            "mode": "guided_first_path" if stop_at_first_solution else "strict_bounded",
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def trig_identity() -> sympy.Expr:
    return sympy.Add(
        sympy.Pow(sympy.sin(y), 2, evaluate=False),
        sympy.Pow(sympy.cos(y), 2, evaluate=False),
        evaluate=False,
    )


def linear_term(symbol: sympy.Symbol, shift: int) -> sympy.Expr:
    return sympy.Add(symbol, shift, evaluate=False)


def build_block_a() -> dict[str, sympy.Expr]:
    base = sympy.Add(
        sympy.Mul(sympy.Integer(2), sympy.Pow(linear_term(x, 1), -1, evaluate=False), evaluate=False),
        sympy.Mul(sympy.Integer(3), sympy.Pow(linear_term(x, 2), -1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    return {
        "start": sympy.Mul(trig_identity(), base, evaluate=False),
        "base": base,
        "together": sympy.together(base),
        "goal": sympy.expand(sympy.together(base)),
    }


def build_block_b() -> dict[str, sympy.Expr]:
    reduced = sympy.Mul(
        linear_term(z, 6),
        sympy.Pow(sympy.Mul(linear_term(z, 1), linear_term(z, 2), evaluate=False), -1, evaluate=False),
        evaluate=False,
    )
    state = sympy.Mul(
        sympy.expand(sympy.Mul(linear_term(z, 3), linear_term(z, 6), evaluate=False)),
        sympy.Pow(sympy.Mul(linear_term(z, 1), linear_term(z, 2), linear_term(z, 3), evaluate=False), -1, evaluate=False),
        evaluate=False,
    )
    return {
        "state": state,
        "reduced": reduced,
        "goal": sympy.expand(reduced),
    }


def compose_sum(*terms: sympy.Expr) -> sympy.Expr:
    return sympy.Add(*terms, evaluate=False)


def block_is_addressable_subtree(expr: sympy.Expr, block: sympy.Expr) -> bool:
    target = str(block)
    for site in enumerate_sites(str(expr)):
        if site.site_type == "expr" and site.expr_str == target:
            return True
    return False


def block_is_addressable_structurally(expr: sympy.Expr, block: sympy.Expr) -> bool:
    target = canonicalize(str(block)).structural
    for site in enumerate_sites(str(expr)):
        if site.site_type == "expr" and canonicalize(site.expr_str).structural == target:
            return True
    return False


def site_snapshot(expr: sympy.Expr) -> list[dict[str, str]]:
    return [
        {
            "site_id": site.site_id,
            "site_type": site.site_type,
            "root_type": site.root_type,
            "expr": site.expr_str,
        }
        for site in enumerate_sites(str(expr))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoped-action sanity report for the minimal site vocabulary")
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--composition-timeout-seconds", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=300)
    args = parser.parse_args()

    block_a = build_block_a()
    block_b = build_block_b()

    block_a_depth = verify_shortest_scoped(
        str(block_a["start"]),
        str(block_a["goal"]),
        max_steps=4,
        timeout_seconds=args.timeout_seconds,
        max_nodes=args.max_nodes,
    )
    block_b_depth = verify_shortest_scoped(
        str(block_b["state"]),
        str(block_b["goal"]),
        max_steps=3,
        timeout_seconds=args.timeout_seconds,
        max_nodes=args.max_nodes,
    )

    composed_states = {
        "start": compose_sum(block_a["start"], block_b["state"]),
        "after_trigsimp": compose_sum(block_a["base"], block_b["state"]),
        "after_together": compose_sum(block_a["together"], block_b["state"]),
        "after_a_goal": compose_sum(block_a["goal"], block_b["state"]),
    }
    composed_goal_reduced = compose_sum(block_a["goal"], block_b["reduced"])
    composed_goal_expanded = compose_sum(block_a["goal"], block_b["goal"])

    composed_reduced_depth = verify_shortest_scoped(
        str(composed_states["start"]),
        str(composed_goal_reduced),
        max_steps=4,
        timeout_seconds=args.composition_timeout_seconds,
        max_nodes=args.max_nodes,
    )
    composed_reduced_guided_depth = verify_shortest_scoped(
        str(composed_states["start"]),
        str(composed_goal_reduced),
        max_steps=4,
        timeout_seconds=args.composition_timeout_seconds,
        max_nodes=args.max_nodes,
        stop_at_first_solution=True,
    )
    composed_expanded_depth = verify_shortest_scoped(
        str(composed_states["start"]),
        str(composed_goal_expanded),
        max_steps=6,
        timeout_seconds=args.composition_timeout_seconds,
        max_nodes=args.max_nodes,
    )
    block_a_components = {
        "start": block_a["start"],
        "after_trigsimp": block_a["base"],
        "after_together": block_a["together"],
        "after_a_goal": block_a["goal"],
    }

    report = {
        "site_model": "path_based_scoped_sites",
        "site_rules": {
            "max_path_depth": 2,
            "site_types": ["expr", "numerator", "denominator", "add_slice"],
            "whitelist": ["root", "Add", "Mul", "rational subtree", "trig-containing subtree"],
            "add_slice_lengths": [2, 3],
            "max_nodes_per_query": args.max_nodes,
            "timeout_seconds": args.timeout_seconds,
            "composition_timeout_seconds": args.composition_timeout_seconds,
        },
        "block_a_scoped_shortest": block_a_depth,
        "block_b_scoped_shortest": block_b_depth,
        "composed_reduced_goal_shortest": composed_reduced_depth,
        "composed_reduced_goal_guided": composed_reduced_guided_depth,
        "composed_expanded_goal_shortest": composed_expanded_depth,
        "addressability": {
            name: {
                "a_block_is_addressable_exactly": block_is_addressable_subtree(expr, block_a_components[name]),
                "a_block_is_addressable_structurally": block_is_addressable_structurally(expr, block_a_components[name]),
                "site_count": len(enumerate_sites(str(expr))),
                "site_snapshot": site_snapshot(expr),
            }
            for name, expr in composed_states.items()
        },
        "notes": [
            "This is the path-based replacement for the shallow named-site prototype.",
            "The key question is whether path sites preserve access to the relevant subtree after partial rewriting and normalization.",
            "Strict bounded search still represents the verifier standard for final labels.",
            "The guided first-path field is a smoke-test check that uses the same scoped action semantics but returns after the first bounded valid path.",
        ],
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
