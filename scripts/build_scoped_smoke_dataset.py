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

FAMILY_ACTIONS = {
    "b_first_a3_b1": (
        "expr@1::factor",
        "expr@0::trigsimp",
        "add_slice@root[0:2]::together",
        "expr@1::expand",
    ),
    "trig_b_a2": (
        "expr@0::trigsimp",
        "expr@2::factor",
        "add_slice@root[0:2]::together",
        "expr@1::expand",
    ),
    "trig_together_b_expand": (
        "expr@0::trigsimp",
        "add_slice@root[0:2]::together",
        "expr@0::factor",
        "expr@1::expand",
    ),
    "a_first_b_last": (
        "expr@0::trigsimp",
        "add_slice@root[0:2]::together",
        "expr@1::expand",
        "expr@2::factor",
    ),
}
DEFAULT_FAMILIES = ("b_first_a3_b1",)


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
    action_order: str
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


def build_action_vocab(family_names: tuple[str, ...]) -> dict[str, int]:
    action_ids = sorted({action_id for family in family_names for action_id in FAMILY_ACTIONS[family]})
    return {action_id: idx for idx, action_id in enumerate(action_ids)}


def family_states(
    family_name: str,
    a_state: sympy.Expr,
    a_base: sympy.Expr,
    a_together: sympy.Expr,
    a_goal: sympy.Expr,
    b_state: sympy.Expr,
    b_reduced: sympy.Expr,
) -> tuple[tuple[sympy.Expr, ...], sympy.Expr]:
    if family_name == "b_first_a3_b1":
        return (
            (
                compose_sum(a_state, b_state),
                compose_sum(a_state, b_reduced),
                compose_sum(a_base, b_reduced),
                compose_sum(b_reduced, a_together),
            ),
            compose_sum(b_reduced, a_goal),
        )
    if family_name == "trig_b_a2":
        return (
            (
                compose_sum(a_state, b_state),
                compose_sum(a_base, b_state),
                compose_sum(a_base, b_reduced),
                compose_sum(b_reduced, a_together),
            ),
            compose_sum(b_reduced, a_goal),
        )
    if family_name == "trig_together_b_expand":
        return (
            (
                compose_sum(a_state, b_state),
                compose_sum(a_base, b_state),
                compose_sum(b_state, a_together),
                compose_sum(b_reduced, a_together),
            ),
            compose_sum(b_reduced, a_goal),
        )
    if family_name == "a_first_b_last":
        return (
            (
                compose_sum(a_state, b_state),
                compose_sum(a_base, b_state),
                compose_sum(b_state, a_together),
                compose_sum(a_goal, b_state),
            ),
            compose_sum(a_goal, b_reduced),
        )
    raise KeyError(f"Unknown scoped family: {family_name}")


def build_trajectory(
    index: int,
    action_vocab: dict[str, int],
    family_name: str,
    a_params: tuple[int, int, int, int],
    b_params: tuple[int, int, int, int],
) -> list[ScopedSmokeRow]:
    a_coeff_0, a_coeff_1, a_shift_0, a_shift_1 = a_params
    b_shift_0, b_shift_1, b_cancel_shift, b_surviving_shift = b_params
    a_state, a_base, a_together, a_goal = build_a_block(*a_params)
    b_state, b_reduced = build_b_block(*b_params)
    parameter_key = "_".join(str(item) for item in (*a_params, *b_params))
    states, final_state = family_states(family_name, a_state, a_base, a_together, a_goal, b_state, b_reduced)
    goal = final_state
    guided_actions = FAMILY_ACTIONS[family_name]

    rows: list[ScopedSmokeRow] = []
    for step_index, (action_id, current) in enumerate(zip(guided_actions, states, strict=True)):
        distance = len(guided_actions) - step_index
        rows.append(
            ScopedSmokeRow(
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
            trajectory_id=f"{family_name}_{index}",
            expr_family=family_name,
            canonical_hash=cheap_hash(goal),
            depth=0,
            benchmark_mode="scoped_actions",
            label_mode="terminal",
            guided_action_id="",
            action_order=" -> ".join(guided_actions),
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


def collect_rows(samples: int, seed: int, family_names: tuple[str, ...]) -> list[ScopedSmokeRow]:
    rng = random.Random(seed)
    action_vocab = build_action_vocab(family_names)
    rows: list[ScopedSmokeRow] = []
    used: set[tuple[str, int, int, int, int, int, int, int, int]] = set()
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
        family_name = family_names[len(used) % len(family_names)]
        params = (family_name, a, b, p, q, bp, bq, br, bs)
        if params in used:
            continue
        used.add(params)
        rows.extend(build_trajectory(len(used) - 1, action_vocab, family_name, (a, b, p, q), (bp, bq, br, bs)))

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
    if mode == "heldout_family":
        families = sorted(frame["expr_family"].unique())
        if len(families) >= 3:
            test_family = families[-1]
            val_family = families[-2]
            train_families = set(families[:-2])
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
    parser = argparse.ArgumentParser(description="Build guided scoped-action A3+B1 smoke/medium datasets")
    parser.add_argument("--samples", type=int, default=24, help="number of guided trajectories")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scoped_smoke"))
    parser.add_argument("--split-mode", choices=("random", "heldout_coeff", "heldout_family"), default="random")
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILY_ACTIONS), default=list(DEFAULT_FAMILIES))
    args = parser.parse_args()

    family_names = tuple(args.families)
    action_vocab = build_action_vocab(family_names)
    rows = collect_rows(args.samples, seed=args.seed, family_names=family_names)
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
