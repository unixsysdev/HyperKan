from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.canonicalize import canonicalize
from data_gen.generate_backward import MOTIF_BUILDERS, SampleRow, dedupe_rows, multi_hot, shortest_actions_to_goal


def build_mixed_action_chain_v0(rng: random.Random) -> tuple[str, str, str]:
    trig_coeff = rng.choice([1, 2, 3])
    trig_term = f"({trig_coeff}*u + 1)*(sin(y)**2 + cos(y)**2)"

    poly_shift_a = rng.choice([1, 2, 3, 4])
    poly_shift_b = rng.choice([5, 6, 7, 8])
    poly_term = f"z**2 + {poly_shift_a + poly_shift_b}*z + {poly_shift_a * poly_shift_b}"

    rat_num = rng.choice([1, 2, 3])
    rat_shift = rng.choice([3, 4, 5])
    cancel_term = f"(w**2 + {(rat_shift + rat_num)}*w + {rat_shift * rat_num})/(w + {rat_shift})"

    partial_a = rng.choice([1, 2, 3])
    partial_b = rng.choice([4, 5, 6])
    partial_term = f"1/(x + {partial_a}) + 1/(x + {partial_b})"

    state = f"{trig_term} + {poly_term} + {cancel_term} + {partial_term}"
    goal = (
        f"{trig_coeff}*u + 1 + (z + {poly_shift_a})*(z + {poly_shift_b}) + "
        f"w + {rat_num} + (2*x + {partial_a + partial_b})/((x + {partial_a})*(x + {partial_b}))"
    )
    return state, goal, "mixed_action_chain_v0"


EXPERIMENTAL_BUILDERS = {
    "mixed_action_chain_v0": build_mixed_action_chain_v0,
}


def resolve_builder(family: str):
    if family in EXPERIMENTAL_BUILDERS:
        return EXPERIMENTAL_BUILDERS[family]
    if family in MOTIF_BUILDERS:
        return MOTIF_BUILDERS[family]
    choices = ", ".join(sorted((*EXPERIMENTAL_BUILDERS.keys(), *MOTIF_BUILDERS.keys())))
    raise KeyError(f"Unknown family: {family}. Choices: {choices}")


def classify_candidate(
    state_str: str,
    goal_str: str,
    min_depth: int,
    max_depth: int,
) -> tuple[str, int | None, tuple[str, ...]]:
    distance, optimal_actions = shortest_actions_to_goal(state_str, goal_str, max_steps=max_depth)
    if distance is None:
        return "no_path", None, ()
    if not optimal_actions:
        return "no_actions", distance, optimal_actions
    if distance < min_depth:
        return "too_shallow", distance, optimal_actions
    if distance > max_depth:
        return "too_deep", distance, optimal_actions
    return "accepted", distance, optimal_actions


def build_row(
    state_str: str,
    goal_str: str,
    family: str,
    distance: int,
    optimal_actions: tuple[str, ...],
    attempt_idx: int,
    seed: int,
) -> SampleRow:
    return SampleRow(
        state_str=state_str,
        goal_str=goal_str,
        valid_shortest_actions=multi_hot(list(optimal_actions)),
        distance_to_goal=distance,
        trajectory_id=f"deep_{family}_{seed}_{attempt_idx}",
        expr_family=family,
        canonical_hash=canonicalize(state_str).digest,
        depth=distance,
    )


def run_harness(
    family: str,
    attempts: int,
    min_depth: int,
    max_depth: int,
    seed: int,
    example_limit: int,
) -> tuple[list[SampleRow], dict[str, object]]:
    builder = resolve_builder(family)
    rng = random.Random(seed)

    accepted: list[SampleRow] = []
    reject_reasons: Counter[str] = Counter()
    accepted_depths: Counter[int] = Counter()
    first_action_sizes: Counter[int] = Counter()
    observed_depths: Counter[int] = Counter()
    examples: list[dict[str, object]] = []
    pathful = 0

    for attempt_idx in range(attempts):
        state_expr, goal_expr, built_family = builder(rng)
        state_str = str(state_expr)
        goal_str = str(goal_expr)
        reason, distance, optimal_actions = classify_candidate(state_str, goal_str, min_depth=min_depth, max_depth=max_depth)
        reject_reasons[reason] += 1

        if distance is not None:
            pathful += 1
            observed_depths[distance] += 1

        if reason != "accepted":
            if len(examples) < example_limit:
                examples.append(
                    {
                        "kind": "reject",
                        "reason": reason,
                        "distance": distance,
                        "optimal_actions": list(optimal_actions),
                        "state": state_str,
                        "goal": goal_str,
                    }
                )
            continue

        row = build_row(
            state_str=state_str,
            goal_str=goal_str,
            family=built_family,
            distance=distance,
            optimal_actions=optimal_actions,
            attempt_idx=attempt_idx,
            seed=seed,
        )
        accepted.append(row)
        accepted_depths[distance] += 1
        first_action_sizes[len(optimal_actions)] += 1
        if len(examples) < example_limit:
            examples.append(
                {
                    "kind": "accept",
                    "distance": distance,
                    "optimal_actions": list(optimal_actions),
                    "state": state_str,
                    "goal": goal_str,
                }
            )

    accepted = dedupe_rows(accepted)
    metrics = {
        "family": family,
        "attempts": attempts,
        "accepted_rows": len(accepted),
        "accept_rate": (len(accepted) / attempts) if attempts else 0.0,
        "shortcut_rate": (reject_reasons["too_shallow"] / pathful) if pathful else 0.0,
        "pathful_candidates": pathful,
        "reject_reasons": dict(reject_reasons),
        "accepted_depth_histogram": dict(sorted(accepted_depths.items())),
        "observed_depth_histogram": dict(sorted(observed_depths.items())),
        "optimal_first_action_set_sizes": dict(sorted(first_action_sizes.items())),
        "examples": examples,
    }
    return accepted, metrics


def write_output(rows: list[SampleRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tight local harness for deep-motif candidate families")
    parser.add_argument("--family", required=True)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--example-limit", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows, metrics = run_harness(
        family=args.family,
        attempts=args.attempts,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        seed=args.seed,
        example_limit=args.example_limit,
    )
    if args.output is not None:
        write_output(rows, args.output)
        metrics["output"] = str(args.output)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
