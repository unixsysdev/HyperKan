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

x, y, z = sympy.symbols("x y z")

GUIDED_ACTIONS = (
    "expr@1::factor",
    "expr@0::trigsimp",
    "add_slice@root[0:2]::together",
    "expr@1::expand",
)


@dataclass(frozen=True)
class ScopedSmokeRow:
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
    a_coeff_0: int
    a_coeff_1: int
    a_shift_0: int
    a_shift_1: int
    b_shift_0: int
    b_shift_1: int
    b_cancel_shift: int
    b_surviving_shift: int
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


def build_a_block(a: int, b: int, p: int, q: int) -> tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
    base = sympy.Add(
        sympy.Mul(sympy.Integer(a), reciprocal_linear(x, p), evaluate=False),
        sympy.Mul(sympy.Integer(b), reciprocal_linear(x, q), evaluate=False),
        evaluate=False,
    )
    together = sympy.together(base)
    return sympy.Mul(base, trig_identity(), evaluate=False), base, together, sympy.expand(together)


def build_b_block(p: int, q: int, r: int, s: int) -> tuple[sympy.Expr, sympy.Expr]:
    numerator = sympy.expand(sympy.Mul(linear_term(z, r), linear_term(z, s), evaluate=False))
    denominator = sympy.Mul(linear_term(z, p), linear_term(z, q), linear_term(z, r), evaluate=False)
    state = sympy.Mul(numerator, sympy.Pow(denominator, -1, evaluate=False), evaluate=False)
    reduced = sympy.Mul(
        linear_term(z, s),
        sympy.Pow(sympy.Mul(linear_term(z, p), linear_term(z, q), evaluate=False), -1, evaluate=False),
        evaluate=False,
    )
    return state, reduced


def compose_sum(*terms: sympy.Expr) -> sympy.Expr:
    return sympy.Add(*terms, evaluate=False)


def action_vector(action_id: str, vocab: dict[str, int]) -> list[int]:
    vector = [0] * len(vocab)
    vector[vocab[action_id]] = 1
    return vector


def cheap_hash(expr: sympy.Expr) -> str:
    return hashlib.sha1(str(expr).encode("utf-8")).hexdigest()


def build_trajectory(
    index: int,
    action_vocab: dict[str, int],
    a_params: tuple[int, int, int, int],
    b_params: tuple[int, int, int, int],
) -> list[ScopedSmokeRow]:
    a_coeff_0, a_coeff_1, a_shift_0, a_shift_1 = a_params
    b_shift_0, b_shift_1, b_cancel_shift, b_surviving_shift = b_params
    a_state, a_base, a_together, a_goal = build_a_block(*a_params)
    b_state, b_reduced = build_b_block(*b_params)
    goal = compose_sum(a_goal, b_reduced)
    parameter_key = "_".join(str(item) for item in (*a_params, *b_params))
    states = (
        compose_sum(a_state, b_state),
        compose_sum(a_state, b_reduced),
        compose_sum(a_base, b_reduced),
        compose_sum(b_reduced, a_together),
    )
    final_state = compose_sum(b_reduced, a_goal)

    rows: list[ScopedSmokeRow] = []
    for step_index, (action_id, current) in enumerate(zip(GUIDED_ACTIONS, states, strict=True)):
        distance = len(GUIDED_ACTIONS) - step_index
        rows.append(
            ScopedSmokeRow(
                state_str=str(current),
                goal_str=str(goal),
                valid_shortest_actions=action_vector(action_id, action_vocab),
                distance_to_goal=distance,
                trajectory_id=f"scoped_smoke_{index}",
                expr_family="scoped_guided_a3_b1",
                canonical_hash=cheap_hash(current),
                depth=distance,
                benchmark_mode="scoped_actions",
                label_mode="guided_single_path",
                guided_action_id=action_id,
                a_coeff_0=a_coeff_0,
                a_coeff_1=a_coeff_1,
                a_shift_0=a_shift_0,
                a_shift_1=a_shift_1,
                b_shift_0=b_shift_0,
                b_shift_1=b_shift_1,
                b_cancel_shift=b_cancel_shift,
                b_surviving_shift=b_surviving_shift,
                parameter_key=parameter_key,
            )
        )

    rows.append(
        ScopedSmokeRow(
            state_str=str(goal),
            goal_str=str(goal),
            valid_shortest_actions=[0] * len(action_vocab),
            distance_to_goal=0,
            trajectory_id=f"scoped_smoke_{index}",
            expr_family="scoped_guided_a3_b1",
            canonical_hash=cheap_hash(goal),
            depth=0,
            benchmark_mode="scoped_actions",
            label_mode="terminal",
            guided_action_id="",
            a_coeff_0=a_coeff_0,
            a_coeff_1=a_coeff_1,
            a_shift_0=a_shift_0,
            a_shift_1=a_shift_1,
            b_shift_0=b_shift_0,
            b_shift_1=b_shift_1,
            b_cancel_shift=b_cancel_shift,
            b_surviving_shift=b_surviving_shift,
            parameter_key=parameter_key,
        )
    )
    return rows


def collect_rows(samples: int, seed: int) -> list[ScopedSmokeRow]:
    rng = random.Random(seed)
    action_vocab = {action_id: idx for idx, action_id in enumerate(GUIDED_ACTIONS)}
    rows: list[ScopedSmokeRow] = []
    used: set[tuple[int, int, int, int, int, int, int, int]] = set()
    attempts = 0

    while len(used) < samples:
        attempts += 1
        if attempts > samples * 50:
            raise RuntimeError(f"Could only build {len(used)} scoped smoke trajectories after {attempts} attempts")
        p, q = sorted(rng.sample([1, 2, 3, 4, 5], k=2))
        a = rng.choice([1, 2, 3, 4])
        b = rng.choice([1, 2, 3, 4])
        bp, bq, br = sorted(rng.sample([1, 2, 3, 4, 5], k=3))
        bs = rng.choice([6, 7, 8, 9])
        params = (a, b, p, q, bp, bq, br, bs)
        if params in used:
            continue
        used.add(params)
        rows.extend(build_trajectory(len(used) - 1, action_vocab, (a, b, p, q), (bp, bq, br, bs)))

    return rows


def split_rows(
    rows: list[ScopedSmokeRow],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if mode == "heldout_coeff":
        test_ids = set(frame.loc[(frame["a_coeff_0"] == 4) & (frame["a_coeff_1"] == 4), "trajectory_id"])
        val_ids = set(
            frame.loc[
                ((frame["a_coeff_0"] == 3) & (frame["a_coeff_1"] == 4))
                | ((frame["a_coeff_0"] == 4) & (frame["a_coeff_1"] == 3)),
                "trajectory_id",
            ]
        )
        train_ids = set(frame["trajectory_id"]) - val_ids - test_ids
        if train_ids and val_ids and test_ids:
            return (
                frame[frame["trajectory_id"].isin(train_ids)].reset_index(drop=True),
                frame[frame["trajectory_id"].isin(val_ids)].reset_index(drop=True),
                frame[frame["trajectory_id"].isin(test_ids)].reset_index(drop=True),
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
    parser = argparse.ArgumentParser(description="Build guided scoped-action A3+B1 smoke/medium datasets")
    parser.add_argument("--samples", type=int, default=24, help="number of guided trajectories")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scoped_smoke"))
    parser.add_argument("--split-mode", choices=("random", "heldout_coeff"), default="random")
    args = parser.parse_args()

    action_vocab = {action_id: idx for idx, action_id in enumerate(GUIDED_ACTIONS)}
    rows = collect_rows(args.samples, seed=args.seed)
    train, val, test = split_rows(rows, mode=args.split_mode)

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
