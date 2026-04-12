from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sympy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.generate_backward import shortest_actions_to_goal


x, y = sympy.symbols("x y")


class TimeoutError(RuntimeError):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
    raise TimeoutError("verification timed out")


def trig_identity() -> sympy.Expr:
    return sympy.Add(
        sympy.Pow(sympy.sin(y), 2, evaluate=False),
        sympy.Pow(sympy.cos(y), 2, evaluate=False),
        evaluate=False,
    )


def inverse_trigsimp(expr: sympy.Expr, step_idx: int) -> sympy.Expr:
    return sympy.Mul(expr, trig_identity(), evaluate=False)


def inverse_factor(expr: sympy.Expr, step_idx: int) -> sympy.Expr:
    return sympy.expand(expr)


def inverse_cancel(expr: sympy.Expr, step_idx: int) -> sympy.Expr:
    shift = step_idx + 3
    factor = sympy.Add(x, shift, evaluate=False)
    return sympy.Mul(expr, factor, sympy.Pow(factor, -1, evaluate=False), evaluate=False)


def inverse_together(expr: sympy.Expr, step_idx: int) -> sympy.Expr:
    symbols = sorted(expr.free_symbols, key=lambda sym: sym.name)
    if not symbols:
        return expr
    return sympy.apart(expr, symbols[0], full=False)


def inverse_expand(expr: sympy.Expr, step_idx: int) -> sympy.Expr:
    return sympy.factor(expr)


INVERSE_BUILDERS: dict[str, Callable[[sympy.Expr, int], sympy.Expr]] = {
    "trigsimp": inverse_trigsimp,
    "factor": inverse_factor,
    "cancel": inverse_cancel,
    "together": inverse_together,
    "expand": inverse_expand,
}


GOAL_TEMPLATES: dict[str, Callable[[], sympy.Expr]] = {
    "rat_partial_expanded": lambda: sympy.expand(
        sympy.together(
            sympy.Add(
                sympy.Pow(sympy.Add(x, 1, evaluate=False), -1, evaluate=False),
                sympy.Pow(sympy.Add(x, 2, evaluate=False), -1, evaluate=False),
                evaluate=False,
            )
        )
    ),
    "rat_partial_together": lambda: sympy.together(
        sympy.Add(
            sympy.Pow(sympy.Add(x, 1, evaluate=False), -1, evaluate=False),
            sympy.Pow(sympy.Add(x, 2, evaluate=False), -1, evaluate=False),
            evaluate=False,
        )
    ),
    "poly_factored": lambda: sympy.Mul(
        sympy.Add(x, 1, evaluate=False),
        sympy.Add(x, 2, evaluate=False),
        evaluate=False,
    ),
}


@dataclass
class PrefixRecord:
    step: int
    intended_action: str
    state: str
    distance: int | None
    optimal_actions: list[str]
    status: str


def verify_exact_distance(state_str: str, goal_str: str, max_steps: int, timeout_seconds: int) -> tuple[int | None, tuple[str, ...]]:
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return shortest_actions_to_goal(state_str, goal_str, max_steps=max_steps)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def synthesize(goal_name: str, actions: list[str], max_steps: int, timeout_seconds: int) -> dict[str, object]:
    goal_expr = GOAL_TEMPLATES[goal_name]()
    current_expr = goal_expr
    goal_str = str(goal_expr)
    prefixes: list[PrefixRecord] = []
    previous_distance = 0
    accepted = True
    stop_reason = "accepted"

    for idx, action in enumerate(reversed(actions), start=1):
        current_expr = INVERSE_BUILDERS[action](current_expr, idx)
        state_str = str(current_expr)
        try:
            distance, optimal_actions = verify_exact_distance(
                state_str=state_str,
                goal_str=goal_str,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            prefixes.append(
                PrefixRecord(
                    step=idx,
                    intended_action=action,
                    state=state_str,
                    distance=None,
                    optimal_actions=[],
                    status="timeout",
                )
            )
            accepted = False
            stop_reason = "timeout"
            break

        if distance is None:
            prefixes.append(
                PrefixRecord(
                    step=idx,
                    intended_action=action,
                    state=state_str,
                    distance=None,
                    optimal_actions=[],
                    status="no_path",
                )
            )
            accepted = False
            stop_reason = "no_path"
            break

        if distance != previous_distance + 1:
            prefixes.append(
                PrefixRecord(
                    step=idx,
                    intended_action=action,
                    state=state_str,
                    distance=distance,
                    optimal_actions=list(optimal_actions),
                    status="distance_jump",
                )
            )
            accepted = False
            stop_reason = "distance_jump"
            break

        if action not in optimal_actions:
            prefixes.append(
                PrefixRecord(
                    step=idx,
                    intended_action=action,
                    state=state_str,
                    distance=distance,
                    optimal_actions=list(optimal_actions),
                    status="intended_action_missing",
                )
            )
            accepted = False
            stop_reason = "intended_action_missing"
            break

        prefixes.append(
            PrefixRecord(
                step=idx,
                intended_action=action,
                state=state_str,
                distance=distance,
                optimal_actions=list(optimal_actions),
                status="accepted",
            )
        )
        previous_distance = distance

    return {
        "target_family": goal_name,
        "goal_template": goal_name,
        "goal": goal_str,
        "desired_action_chain": actions,
        "accept_criteria": {
            "exact_distance_increases_by_one_per_prefix": True,
            "intended_action_present_in_optimal_action_set": True,
            "no_shorter_path_than_prefix_length": True,
        },
        "accepted": accepted,
        "stop_reason": stop_reason,
        "final_depth": previous_distance if accepted else prefixes[-1].distance if prefixes else None,
        "prefixes": [record.__dict__ for record in prefixes],
        "final_state": prefixes[-1].state if prefixes else goal_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backward verified deep-chain synthesizer")
    parser.add_argument("--goal-template", choices=sorted(GOAL_TEMPLATES.keys()), required=True)
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    args = parser.parse_args()

    for action in args.actions:
        if action not in INVERSE_BUILDERS:
            raise ValueError(f"Unsupported action in synthesizer: {action}")

    result = synthesize(
        goal_name=args.goal_template,
        actions=args.actions,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
