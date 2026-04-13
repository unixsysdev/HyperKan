from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import sympy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.scoped_actions import apply_scoped_action_unchecked, parse_scoped_action_id

x, y, z = sympy.symbols("x y z")

FAMILY_ACTIONS = {
    "trig_merge": (
        "expr@root::trigsimp",
        "expr@root::together",
        "expr@root::expand",
    ),
    "hidden_cancel": (
        "numerator@root::factor",
        "expr@root::cancel",
        "expr@root::expand",
    ),
    "apart_normalize": (
        "denominator@root::factor",
        "expr@root::apart",
    ),
    "mixed_trig_hidden": (
        "expr@0::trigsimp",
        "add_slice@root[0:2]::together",
        "expr@1::expand",
        "numerator@2::factor",
        "expr@2::cancel",
    ),
    "mixed_trig_hidden_apart": (
        "expr@1::trigsimp",
        "add_slice@root[1:3]::together",
        "expr@2::expand",
        "numerator@3::factor",
        "expr@3::cancel",
        "denominator@1::factor",
        "expr@1::apart",
    ),
}
DEFAULT_FAMILIES = (
    "trig_merge",
    "hidden_cancel",
    "apart_normalize",
    "mixed_trig_hidden",
    "mixed_trig_hidden_apart",
)


@dataclass(frozen=True)
class ScopedStructuralRow:
    state_str: str
    goal_str: str
    valid_shortest_actions: list[int]
    distance_to_goal: int
    trajectory_id: str
    expr_family: str
    canonical_hash: str
    depth: int
    benchmark_mode: str
    label_mode: str
    guided_action_id: str
    action_order: str
    parameter_key: str


def linear_term(symbol: sympy.Symbol, shift: int) -> sympy.Expr:
    return sympy.Add(symbol, shift, evaluate=False)


def reciprocal_linear(symbol: sympy.Symbol, shift: int) -> sympy.Expr:
    return sympy.Pow(linear_term(symbol, shift), -1, evaluate=False)


def trig_identity() -> sympy.Expr:
    return sympy.Add(
        sympy.Pow(sympy.sin(y), 2, evaluate=False),
        sympy.Pow(sympy.cos(y), 2, evaluate=False),
        evaluate=False,
    )


def cheap_hash(expr: sympy.Expr) -> str:
    return hashlib.sha1(str(expr).encode("utf-8")).hexdigest()


def action_vector(action_id: str, vocab: dict[str, int]) -> list[int]:
    vector = [0] * len(vocab)
    vector[vocab[action_id]] = 1
    return vector


def build_action_vocab(family_names: tuple[str, ...]) -> dict[str, int]:
    action_ids = sorted({action_id for family in family_names for action_id in FAMILY_ACTIONS[family]})
    return {action_id: idx for idx, action_id in enumerate(action_ids)}


def compose_sum(*terms: sympy.Expr) -> sympy.Expr:
    return sympy.Add(*terms, evaluate=False)


def replay_guided_path(start_expr: sympy.Expr, action_ids: tuple[str, ...]) -> tuple[list[sympy.Expr], sympy.Expr]:
    states: list[sympy.Expr] = []
    current = start_expr
    for action_id in action_ids:
        states.append(current)
        site_id, op_name = parse_scoped_action_id(action_id)
        next_expr = apply_scoped_action_unchecked(str(current), site_id, op_name)
        if next_expr is None:
            raise RuntimeError(f"Scoped action failed during guided replay: {action_id} on {current}")
        current = next_expr
    return states, current


def build_trig_merge_start(params: tuple[int, int, int, int]) -> tuple[sympy.Expr, str]:
    a_coeff_0, a_coeff_1, a_shift_0, a_shift_1 = params
    base = sympy.Add(
        sympy.Mul(sympy.Integer(a_coeff_0), reciprocal_linear(x, a_shift_0), evaluate=False),
        sympy.Mul(sympy.Integer(a_coeff_1), reciprocal_linear(x, a_shift_1), evaluate=False),
        evaluate=False,
    )
    start_expr = sympy.Mul(base, trig_identity(), evaluate=False)
    return start_expr, "_".join(str(item) for item in params)


def build_hidden_cancel_start(params: tuple[int, int, int]) -> tuple[sympy.Expr, str]:
    p_shift, shared_shift, surviving_shift = params
    numerator = sympy.expand(
        sympy.Mul(linear_term(z, shared_shift), linear_term(z, surviving_shift), evaluate=False)
    )
    denominator = sympy.Mul(
        linear_term(z, p_shift),
        sympy.Pow(linear_term(z, shared_shift), 2, evaluate=False),
        evaluate=False,
    )
    start_expr = sympy.Mul(numerator, sympy.Pow(denominator, -1, evaluate=False), evaluate=False)
    return start_expr, "_".join(str(item) for item in params)


def build_apart_normalize_start(params: tuple[int, int, int, int]) -> tuple[sympy.Expr, str]:
    coeff_0, coeff_1, shift_0, shift_1 = params
    goal_expr = sympy.Add(
        sympy.Mul(sympy.Integer(coeff_0), reciprocal_linear(z, shift_0), evaluate=False),
        sympy.Mul(sympy.Integer(coeff_1), reciprocal_linear(z, shift_1), evaluate=False),
        evaluate=False,
    )
    combined = sympy.together(goal_expr)
    numerator, denominator = sympy.fraction(combined)
    expanded_denominator = sympy.expand(denominator)
    start_expr = sympy.Mul(numerator, sympy.Pow(expanded_denominator, -1, evaluate=False), evaluate=False)
    return start_expr, "_".join(str(item) for item in params)


def build_mixed_trig_hidden_start(
    params: tuple[tuple[int, int, int, int], tuple[int, int, int]]
) -> tuple[sympy.Expr, str]:
    trig_params, hidden_params = params
    trig_start, trig_key = build_trig_merge_start(trig_params)
    hidden_start, hidden_key = build_hidden_cancel_start(hidden_params)
    return compose_sum(trig_start, hidden_start), f"{trig_key}__{hidden_key}"


def build_mixed_trig_hidden_apart_start(
    params: tuple[tuple[int, int, int, int], tuple[int, int, int], tuple[int, int, int, int]]
) -> tuple[sympy.Expr, str]:
    trig_params, hidden_params, apart_params = params
    trig_start, trig_key = build_trig_merge_start(trig_params)
    hidden_start, hidden_key = build_hidden_cancel_start(hidden_params)
    apart_start, apart_key = build_apart_normalize_start(apart_params)
    return compose_sum(apart_start, trig_start, hidden_start), f"{trig_key}__{hidden_key}__{apart_key}"


def sample_trig_merge_params(rng: random.Random) -> tuple[int, int, int, int]:
    shift_0, shift_1 = sorted(rng.sample([1, 2, 3, 4, 5], k=2))
    coeff_0 = rng.choice([1, 2, 3, 4])
    coeff_1 = rng.choice([1, 2, 3, 4])
    return coeff_0, coeff_1, shift_0, shift_1


def sample_hidden_cancel_params(rng: random.Random) -> tuple[int, int, int]:
    p_shift, shared_shift = sorted(rng.sample([1, 2, 3, 4, 5], k=2))
    surviving_shift = rng.choice([6, 7, 8, 9])
    return p_shift, shared_shift, surviving_shift


def sample_apart_normalize_params(rng: random.Random) -> tuple[int, int, int, int]:
    shift_0, shift_1 = sorted(rng.sample([1, 2, 3, 4, 5], k=2))
    coeff_0 = rng.choice([1, 2, 3, 4])
    coeff_1 = rng.choice([1, 2, 3, 4])
    return coeff_0, coeff_1, shift_0, shift_1


def sample_mixed_trig_hidden_params(rng: random.Random) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    return sample_trig_merge_params(rng), sample_hidden_cancel_params(rng)


def sample_mixed_trig_hidden_apart_params(
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int], tuple[int, int, int, int]]:
    return (
        sample_trig_merge_params(rng),
        sample_hidden_cancel_params(rng),
        sample_apart_normalize_params(rng),
    )


FAMILY_BUILDERS = {
    "trig_merge": (sample_trig_merge_params, build_trig_merge_start),
    "hidden_cancel": (sample_hidden_cancel_params, build_hidden_cancel_start),
    "apart_normalize": (sample_apart_normalize_params, build_apart_normalize_start),
    "mixed_trig_hidden": (sample_mixed_trig_hidden_params, build_mixed_trig_hidden_start),
    "mixed_trig_hidden_apart": (sample_mixed_trig_hidden_apart_params, build_mixed_trig_hidden_apart_start),
}


def build_trajectory(
    index: int,
    action_vocab: dict[str, int],
    family_name: str,
    parameter_key: str,
    states: list[sympy.Expr],
    goal: sympy.Expr,
) -> list[ScopedStructuralRow]:
    guided_actions = FAMILY_ACTIONS[family_name]
    rows: list[ScopedStructuralRow] = []
    for step_index, (action_id, current) in enumerate(zip(guided_actions, states, strict=True)):
        distance = len(guided_actions) - step_index
        rows.append(
            ScopedStructuralRow(
                state_str=str(current),
                goal_str=str(goal),
                valid_shortest_actions=action_vector(action_id, action_vocab),
                distance_to_goal=distance,
                trajectory_id=f"{family_name}_{index}",
                expr_family=family_name,
                canonical_hash=cheap_hash(current),
                depth=distance,
                benchmark_mode="scoped_actions",
                label_mode="guided_single_path",
                guided_action_id=action_id,
                action_order=" -> ".join(guided_actions),
                parameter_key=parameter_key,
            )
        )

    rows.append(
        ScopedStructuralRow(
            state_str=str(goal),
            goal_str=str(goal),
            valid_shortest_actions=[0] * len(action_vocab),
            distance_to_goal=0,
            trajectory_id=f"{family_name}_{index}",
            expr_family=family_name,
            canonical_hash=cheap_hash(goal),
            depth=0,
            benchmark_mode="scoped_actions",
            label_mode="terminal",
            guided_action_id="",
            action_order=" -> ".join(guided_actions),
            parameter_key=parameter_key,
        )
    )
    return rows


def collect_rows(samples: int, seed: int, family_names: tuple[str, ...]) -> list[ScopedStructuralRow]:
    rng = random.Random(seed)
    action_vocab = build_action_vocab(family_names)
    rows: list[ScopedStructuralRow] = []
    used: set[tuple[str, str]] = set()
    attempts = 0

    while len(used) < samples:
        attempts += 1
        if attempts > samples * 100:
            raise RuntimeError(f"Could only build {len(used)} structural trajectories after {attempts} attempts")

        family_name = family_names[len(used) % len(family_names)]
        sample_fn, build_fn = FAMILY_BUILDERS[family_name]
        params = sample_fn(rng)
        start_expr, parameter_key = build_fn(params)
        dedupe_key = (family_name, parameter_key)
        if dedupe_key in used:
            continue

        states, goal = replay_guided_path(start_expr, FAMILY_ACTIONS[family_name])
        used.add(dedupe_key)
        rows.extend(build_trajectory(len(used) - 1, action_vocab, family_name, parameter_key, states, goal))

    return rows


def split_rows(
    rows: list[ScopedStructuralRow],
    mode: str,
    family_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame([asdict(row) for row in rows])

    if mode == "heldout_test_family" and len(family_names) >= 2:
        test_family = family_names[-1]
        remaining = frame[frame["expr_family"] != test_family].reset_index(drop=True)
        test = frame[frame["expr_family"] == test_family].reset_index(drop=True)
        rng = random.Random(17)
        train_ids: set[str] = set()
        val_ids: set[str] = set()
        for family_name in family_names[:-1]:
            family_rows = remaining[remaining["expr_family"] == family_name]
            trajectories = sorted(family_rows["trajectory_id"].unique())
            rng.shuffle(trajectories)
            if len(trajectories) == 1:
                train_ids.add(trajectories[0])
                continue
            train_end = max(1, int(0.85 * len(trajectories)))
            if train_end >= len(trajectories):
                train_end = len(trajectories) - 1
            train_ids.update(trajectories[:train_end])
            val_ids.update(trajectories[train_end:])
        train = remaining[remaining["trajectory_id"].isin(train_ids)].reset_index(drop=True)
        val = remaining[remaining["trajectory_id"].isin(val_ids)].reset_index(drop=True)
        return train, val, test

    if mode == "heldout_family" and len(family_names) >= 3:
        test_family = family_names[-1]
        val_family = family_names[-2]
        train_families = set(family_names[:-2])
        return (
            frame[frame["expr_family"].isin(train_families)].reset_index(drop=True),
            frame[frame["expr_family"] == val_family].reset_index(drop=True),
            frame[frame["expr_family"] == test_family].reset_index(drop=True),
        )

    trajectories = sorted(frame["trajectory_id"].unique())
    rng = random.Random(17)
    rng.shuffle(trajectories)
    train_end = max(1, int(0.7 * len(trajectories)))
    val_end = max(train_end + 1, int(0.85 * len(trajectories)))
    train_ids = set(trajectories[:train_end])
    val_ids = set(trajectories[train_end:val_end])
    test_ids = set(trajectories[val_end:])
    return (
        frame[frame["trajectory_id"].isin(train_ids)].reset_index(drop=True),
        frame[frame["trajectory_id"].isin(val_ids)].reset_index(drop=True),
        frame[frame["trajectory_id"].isin(test_ids)].reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guided scoped-action structural family datasets")
    parser.add_argument("--samples", type=int, default=48, help="number of guided trajectories")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scoped_structural_probe"))
    parser.add_argument(
        "--split-mode",
        choices=("random", "heldout_family", "heldout_test_family"),
        default="heldout_test_family",
    )
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILY_ACTIONS), default=list(DEFAULT_FAMILIES))
    args = parser.parse_args()

    family_names = tuple(args.families)
    action_vocab = build_action_vocab(family_names)
    rows = collect_rows(args.samples, seed=args.seed, family_names=family_names)
    train, val, test = split_rows(rows, mode=args.split_mode, family_names=family_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet", index=False)
    val.to_parquet(args.output_dir / "val.parquet", index=False)
    test.to_parquet(args.output_dir / "test.parquet", index=False)
    (args.output_dir / "scoped_action_vocab.json").write_text(json.dumps(action_vocab, indent=2), encoding="utf-8")
    (args.output_dir / "meta.json").write_text(
        json.dumps(
            {
                "benchmark_mode": "scoped_actions",
                "label_mode": "guided_single_path",
                "samples": args.samples,
                "split_mode": args.split_mode,
                "families": list(family_names),
                "rows": len(rows),
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "action_vocab": action_vocab,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "train": len(train), "val": len(val), "test": len(test), "actions": len(action_vocab)}))


if __name__ == "__main__":
    main()
