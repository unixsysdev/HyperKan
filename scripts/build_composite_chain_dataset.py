from __future__ import annotations

import argparse
import json
import random
import signal
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import sympy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.actions import ACTION_NAMES
from data_gen.canonicalize import canonicalize
from data_gen.generate_backward import SampleRow, dedupe_rows, multi_hot, shortest_actions_to_goal, split_rows


x, y, z = sympy.symbols("x y z")

BLOCK_A_ACTIONS = ("trigsimp", "together", "expand")
FAMILY_DEPTH4 = "composite_a3_b1_cancel"
FAMILY_DEPTH5 = "composite_a3_b2_cancel_expand"


class VerificationTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefixState:
    distance: int
    intended_action: str
    state_str: str
    optimal_actions: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class CandidateResult:
    family: str
    goal_str: str
    desired_action_chain: tuple[str, ...]
    accepted: bool
    stop_reason: str
    prefixes: tuple[PrefixState, ...]
    params: dict[str, int]


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
    raise VerificationTimeout("shortest-path verification timed out")


def trig_identity() -> sympy.Expr:
    return sympy.Add(
        sympy.Pow(sympy.sin(y), 2, evaluate=False),
        sympy.Pow(sympy.cos(y), 2, evaluate=False),
        evaluate=False,
    )


def linear_term(symbol: sympy.Symbol, shift: int) -> sympy.Expr:
    return sympy.Add(symbol, shift, evaluate=False)


def linear_product(symbol: sympy.Symbol, shifts: tuple[int, ...]) -> sympy.Expr:
    return sympy.Mul(*(linear_term(symbol, shift) for shift in shifts), evaluate=False)


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


def build_block_b_cancel_only(params: dict[str, int]) -> dict[str, sympy.Expr]:
    p = params["p"]
    q = params["q"]
    r = params["r"]
    d = params["d"]
    numerator = sympy.expand(sympy.Mul(linear_term(z, r), linear_term(z, d), evaluate=False))
    denominator = linear_product(z, (p, q, r))
    reduced = sympy.Mul(
        linear_term(z, d),
        sympy.Pow(linear_product(z, (p, q)), -1, evaluate=False),
        evaluate=False,
    )
    return {
        "state": sympy.Mul(numerator, sympy.Pow(denominator, -1, evaluate=False), evaluate=False),
        "goal": reduced,
    }


def build_block_b_cancel_expand(params: dict[str, int]) -> dict[str, sympy.Expr]:
    cancel_only = build_block_b_cancel_only(params)
    reduced = cancel_only["goal"]
    return {
        "state": cancel_only["state"],
        "reduced": reduced,
        "goal": sympy.expand(reduced),
    }


def compose_sum(*terms: sympy.Expr) -> sympy.Expr:
    return sympy.Add(*terms, evaluate=False)


def choose_block_b_params(rng: random.Random) -> dict[str, int]:
    p, q, r, d = rng.sample([1, 2, 3, 4, 5, 6, 7, 8], 4)
    return {"p": p, "q": q, "r": r, "d": d}


def verify_exact_distance(state_str: str, goal_str: str, max_steps: int, timeout_seconds: int) -> tuple[int | None, tuple[str, ...]]:
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return shortest_actions_to_goal(state_str, goal_str, max_steps=max_steps)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def verify_prefix_chain(
    family: str,
    goal_expr: sympy.Expr,
    prefix_specs: list[tuple[str, sympy.Expr]],
    desired_action_chain: tuple[str, ...],
    params: dict[str, int],
    max_steps: int,
    timeout_seconds: int,
) -> CandidateResult:
    goal_str = str(goal_expr)
    accepted_prefixes: list[PrefixState] = []

    for distance, (intended_action, state_expr) in enumerate(prefix_specs, start=1):
        state_str = str(state_expr)
        try:
            actual_distance, optimal_actions = verify_exact_distance(
                state_str=state_str,
                goal_str=goal_str,
                max_steps=min(max_steps, distance),
                timeout_seconds=timeout_seconds,
            )
        except VerificationTimeout:
            accepted_prefixes.append(
                PrefixState(
                    distance=distance,
                    intended_action=intended_action,
                    state_str=state_str,
                    optimal_actions=(),
                    status="timeout",
                )
            )
            return CandidateResult(
                family=family,
                goal_str=goal_str,
                desired_action_chain=desired_action_chain,
                accepted=False,
                stop_reason="timeout",
                prefixes=tuple(accepted_prefixes),
                params=params,
            )

        if actual_distance != distance:
            accepted_prefixes.append(
                PrefixState(
                    distance=distance,
                    intended_action=intended_action,
                    state_str=state_str,
                    optimal_actions=optimal_actions,
                    status="distance_jump",
                )
            )
            return CandidateResult(
                family=family,
                goal_str=goal_str,
                desired_action_chain=desired_action_chain,
                accepted=False,
                stop_reason="distance_jump",
                prefixes=tuple(accepted_prefixes),
                params=params,
            )

        if intended_action not in optimal_actions:
            accepted_prefixes.append(
                PrefixState(
                    distance=distance,
                    intended_action=intended_action,
                    state_str=state_str,
                    optimal_actions=optimal_actions,
                    status="intended_action_missing",
                )
            )
            return CandidateResult(
                family=family,
                goal_str=goal_str,
                desired_action_chain=desired_action_chain,
                accepted=False,
                stop_reason="intended_action_missing",
                prefixes=tuple(accepted_prefixes),
                params=params,
            )

        accepted_prefixes.append(
            PrefixState(
                distance=distance,
                intended_action=intended_action,
                state_str=state_str,
                optimal_actions=optimal_actions,
                status="accepted",
            )
        )

    return CandidateResult(
        family=family,
        goal_str=goal_str,
        desired_action_chain=desired_action_chain,
        accepted=True,
        stop_reason="accepted",
        prefixes=tuple(accepted_prefixes),
        params=params,
    )


def synthesize_depth4_candidate(rng: random.Random, max_steps: int, timeout_seconds: int) -> CandidateResult:
    block_a = build_block_a()
    params = choose_block_b_params(rng)
    block_b = build_block_b_cancel_only(params)
    goal_expr = compose_sum(block_a["goal"], block_b["goal"])
    desired_action_chain = (*BLOCK_A_ACTIONS, "cancel")
    prefix_specs = [
        ("cancel", compose_sum(block_a["goal"], block_b["state"])),
        ("expand", compose_sum(block_a["together"], block_b["state"])),
        ("together", compose_sum(block_a["base"], block_b["state"])),
        ("trigsimp", compose_sum(block_a["start"], block_b["state"])),
    ]
    return verify_prefix_chain(
        family=FAMILY_DEPTH4,
        goal_expr=goal_expr,
        prefix_specs=prefix_specs,
        desired_action_chain=desired_action_chain,
        params=params,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
    )


def synthesize_depth5_candidate(rng: random.Random, max_steps: int, timeout_seconds: int) -> CandidateResult:
    block_a = build_block_a()
    params = choose_block_b_params(rng)
    block_b = build_block_b_cancel_expand(params)
    goal_expr = compose_sum(block_a["goal"], block_b["goal"])
    desired_action_chain = (*BLOCK_A_ACTIONS, "cancel", "expand")
    prefix_specs = [
        ("expand", compose_sum(block_a["goal"], block_b["reduced"])),
        ("cancel", compose_sum(block_a["goal"], block_b["state"])),
        ("expand", compose_sum(block_a["together"], block_b["state"])),
        ("together", compose_sum(block_a["base"], block_b["state"])),
        ("trigsimp", compose_sum(block_a["start"], block_b["state"])),
    ]
    return verify_prefix_chain(
        family=FAMILY_DEPTH5,
        goal_expr=goal_expr,
        prefix_specs=prefix_specs,
        desired_action_chain=desired_action_chain,
        params=params,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
    )


def synthesize_candidate(
    family: str,
    rng: random.Random,
    max_steps: int,
    timeout_seconds: int,
) -> CandidateResult:
    if family == FAMILY_DEPTH4:
        return synthesize_depth4_candidate(rng=rng, max_steps=max_steps, timeout_seconds=timeout_seconds)
    if family == FAMILY_DEPTH5:
        return synthesize_depth5_candidate(rng=rng, max_steps=max_steps, timeout_seconds=timeout_seconds)
    raise KeyError(f"Unsupported composite family: {family}")


def rows_from_candidate(candidate: CandidateResult, attempt_idx: int, include_terminal: bool) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for prefix in candidate.prefixes:
        rows.append(
            SampleRow(
                state_str=prefix.state_str,
                goal_str=candidate.goal_str,
                valid_shortest_actions=multi_hot(list(prefix.optimal_actions)),
                distance_to_goal=prefix.distance,
                trajectory_id=f"{candidate.family}_{attempt_idx}_{prefix.distance}",
                expr_family=candidate.family,
                canonical_hash=canonicalize(prefix.state_str).digest,
                depth=prefix.distance,
            )
        )
    if include_terminal:
        rows.append(
            SampleRow(
                state_str=candidate.goal_str,
                goal_str=candidate.goal_str,
                valid_shortest_actions=[0] * len(ACTION_NAMES),
                distance_to_goal=0,
                trajectory_id=f"{candidate.family}_{attempt_idx}_goal",
                expr_family=candidate.family,
                canonical_hash=canonicalize(candidate.goal_str).digest,
                depth=0,
            )
        )
    return rows


def collect_candidates(
    families: tuple[str, ...],
    accepted_target: int,
    max_attempts: int,
    seed: int,
    max_steps: int,
    timeout_seconds: int,
    include_terminal: bool,
) -> tuple[list[SampleRow], dict[str, object]]:
    rng = random.Random(seed)
    rows: list[SampleRow] = []
    metrics: dict[str, object] = {"seed": seed, "families": {}}

    for family in families:
        accepted = 0
        outcomes: Counter[str] = Counter()
        accepted_depths: Counter[int] = Counter()
        examples: list[dict[str, object]] = []

        for attempt_idx in range(max_attempts):
            if accepted >= accepted_target:
                break
            candidate = synthesize_candidate(
                family=family,
                rng=rng,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
            outcomes[candidate.stop_reason] += 1
            if len(examples) < 4:
                examples.append(
                    {
                        "accepted": candidate.accepted,
                        "stop_reason": candidate.stop_reason,
                        "params": candidate.params,
                        "goal": candidate.goal_str,
                        "prefix_statuses": [asdict(prefix) for prefix in candidate.prefixes],
                    }
                )
            if not candidate.accepted:
                continue

            accepted += 1
            final_depth = candidate.prefixes[-1].distance
            accepted_depths[final_depth] += 1
            rows.extend(rows_from_candidate(candidate, attempt_idx=attempt_idx, include_terminal=include_terminal))

        metrics["families"][family] = {
            "accepted_candidates": accepted,
            "attempts_used": sum(outcomes.values()),
            "accept_rate": (accepted / sum(outcomes.values())) if outcomes else 0.0,
            "outcomes": dict(sorted(outcomes.items())),
            "reject_reasons": dict(sorted((reason, count) for reason, count in outcomes.items() if reason != "accepted")),
            "accepted_depth_histogram": dict(sorted(accepted_depths.items())),
            "examples": examples,
        }

    rows = dedupe_rows(rows)
    metrics["rows"] = len(rows)
    metrics["depth_histogram"] = dict(sorted(Counter(row.distance_to_goal for row in rows).items()))
    return rows, metrics


def write_dataset(rows: list[SampleRow], output_dir: Path, split_mode: str) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_parquet(output_dir / "all.parquet", index=False)
    train, val, test = split_rows(rows, mode=split_mode)
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    return {"all": len(frame), "train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verified composite-chain dataset with additive gated depth")
    parser.add_argument(
        "--families",
        nargs="+",
        default=[FAMILY_DEPTH4, FAMILY_DEPTH5],
        choices=[FAMILY_DEPTH4, FAMILY_DEPTH5],
    )
    parser.add_argument("--accepted-target", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--split-mode", choices=("family", "random"), default="random")
    parser.add_argument("--include-terminal", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    rows, metrics = collect_candidates(
        families=tuple(args.families),
        accepted_target=args.accepted_target,
        max_attempts=args.max_attempts,
        seed=args.seed,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
        include_terminal=args.include_terminal,
    )

    if args.output_dir is not None:
        split_sizes = write_dataset(rows, output_dir=args.output_dir, split_mode=args.split_mode)
        metrics["output_dir"] = str(args.output_dir)
        metrics["split_sizes"] = split_sizes

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
