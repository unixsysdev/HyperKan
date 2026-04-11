from __future__ import annotations

import argparse
import os
import random
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import sympy

from data_gen.actions import ACTION_NAMES, ACTION_TO_ID, apply_action
from data_gen.canonicalize import canonicalize


x, y = sympy.symbols("x y")
DEFAULT_TEMPLATE_KEYS = (
    "poly_trig_to_expanded",
    "poly_trig_to_factored",
    "rat_partial_trig_to_together",
    "rat_partial_trig_to_expanded",
    "rat_three_partial_trig_to_together",
    "rat_three_partial_trig_to_expanded",
    "rat_poly_trig_to_one",
)


@dataclass(frozen=True)
class SampleRow:
    state_str: str
    goal_str: str
    valid_shortest_actions: list[int]
    distance_to_goal: int
    trajectory_id: str
    expr_family: str
    canonical_hash: str
    depth: int


def default_worker_count() -> int:
    logical = os.cpu_count() or 1
    physicalish = max(1, logical // 2)
    return min(12, physicalish)


def multi_hot(actions: list[str]) -> list[int]:
    vector = [0] * len(ACTION_NAMES)
    for action in actions:
        vector[ACTION_TO_ID[action]] = 1
    return vector


def _chunk_indices(samples: int, workers: int) -> list[list[int]]:
    workers = max(1, min(workers, samples))
    chunk_size = (samples + workers - 1) // workers
    return [list(range(start, min(samples, start + chunk_size))) for start in range(0, samples, chunk_size)]


def trig_wrap(expr: sympy.Expr, var: sympy.Symbol = y) -> sympy.Expr:
    identity = sympy.Add(
        sympy.Pow(sympy.sin(var), 2, evaluate=False),
        sympy.Pow(sympy.cos(var), 2, evaluate=False),
        evaluate=False,
    )
    return sympy.Mul(expr, identity, evaluate=False)


def choose_distinct_shifts(rng: random.Random, count: int) -> list[int]:
    pool = [1, 2, 3, 4, 5, -1, -2, -3]
    chosen = rng.sample(pool, k=count)
    return sorted(chosen)


def linear_product(shifts: list[int]) -> sympy.Expr:
    factors = [sympy.Add(x, shift, evaluate=False) for shift in shifts]
    return sympy.Mul(*factors, evaluate=False)


def rational_sum(terms: list[tuple[int, int]]) -> sympy.Expr:
    pieces = []
    for coeff, shift in terms:
        numerator = sympy.Integer(coeff)
        denominator = sympy.Add(x, shift, evaluate=False)
        pieces.append(sympy.Mul(numerator, sympy.Pow(denominator, -1, evaluate=False), evaluate=False))
    return sympy.Add(*pieces, evaluate=False)


def build_poly_trig_to_expanded(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 2 if rng.random() < 0.7 else 3)
    base = linear_product(shifts)
    state = trig_wrap(base)
    goal = sympy.expand(base)
    return state, goal, "poly_trig_to_expanded"


def build_poly_trig_to_factored(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 2 if rng.random() < 0.7 else 3)
    goal = linear_product(shifts)
    state = trig_wrap(sympy.expand(goal))
    return state, goal, "poly_trig_to_factored"


def build_rat_partial_trig_to_together(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 2)
    coeffs = [rng.choice([1, 2, 3]), rng.choice([1, 2, 3])]
    base = rational_sum(list(zip(coeffs, shifts, strict=False)))
    state = trig_wrap(base)
    goal = sympy.together(base)
    return state, goal, "rat_partial_trig_to_together"


def build_rat_partial_trig_to_expanded(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 2)
    coeffs = [rng.choice([1, 2, 3]), rng.choice([1, 2, 3])]
    base = rational_sum(list(zip(coeffs, shifts, strict=False)))
    state = trig_wrap(base)
    goal = sympy.expand(sympy.together(base))
    return state, goal, "rat_partial_trig_to_expanded"


def build_rat_three_partial_trig_to_together(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 3)
    coeffs = [rng.choice([1, 2, 3]) for _ in range(3)]
    base = rational_sum(list(zip(coeffs, shifts, strict=False)))
    state = trig_wrap(base)
    goal = sympy.together(base)
    return state, goal, "rat_three_partial_trig_to_together"


def build_rat_three_partial_trig_to_expanded(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 3)
    coeffs = [rng.choice([1, 2, 3]) for _ in range(3)]
    base = rational_sum(list(zip(coeffs, shifts, strict=False)))
    state = trig_wrap(base)
    goal = sympy.expand(sympy.together(base))
    return state, goal, "rat_three_partial_trig_to_expanded"


def build_rat_poly_trig_to_one(rng: random.Random) -> tuple[sympy.Expr, sympy.Expr, str]:
    shifts = choose_distinct_shifts(rng, 2)
    numerator = sympy.expand(linear_product(shifts))
    denominator = linear_product(shifts)
    base = sympy.Mul(numerator, sympy.Pow(denominator, -1, evaluate=False), evaluate=False)
    state = trig_wrap(base)
    goal = sympy.Integer(1)
    return state, goal, "rat_poly_trig_to_one"


MOTIF_BUILDERS = {
    "poly_trig_to_expanded": build_poly_trig_to_expanded,
    "poly_trig_to_factored": build_poly_trig_to_factored,
    "rat_partial_trig_to_together": build_rat_partial_trig_to_together,
    "rat_partial_trig_to_expanded": build_rat_partial_trig_to_expanded,
    "rat_three_partial_trig_to_together": build_rat_three_partial_trig_to_together,
    "rat_three_partial_trig_to_expanded": build_rat_three_partial_trig_to_expanded,
    "rat_poly_trig_to_one": build_rat_poly_trig_to_one,
}


@lru_cache(maxsize=200_000)
def shortest_actions_to_goal(state_str: str, goal_str: str, max_steps: int) -> tuple[int | None, tuple[str, ...]]:
    goal_struct = canonicalize(goal_str).structural
    state_struct = canonicalize(state_str).structural
    if state_struct == goal_struct:
        return 0, ()

    queue: deque[tuple[str, int, str | None]] = deque([(state_str, 0, None)])
    visited = {state_struct}
    shortest_depth: int | None = None
    shortest_first_actions: set[str] = set()

    while queue:
        current, depth, first_action = queue.popleft()
        if shortest_depth is not None and depth >= shortest_depth:
            continue
        if depth >= max_steps:
            continue

        for action in ACTION_NAMES:
            result = apply_action(current, action)
            if not result.valid or result.output_expr is None:
                continue

            next_expr = str(result.output_expr)
            next_struct = canonicalize(next_expr).structural
            root_action = action if first_action is None else first_action

            if next_struct == goal_struct:
                candidate_depth = depth + 1
                if shortest_depth is None or candidate_depth < shortest_depth:
                    shortest_depth = candidate_depth
                    shortest_first_actions = {root_action}
                elif candidate_depth == shortest_depth:
                    shortest_first_actions.add(root_action)
                continue

            if next_struct in visited:
                continue
            visited.add(next_struct)
            queue.append((next_expr, depth + 1, root_action))

    if shortest_depth is None:
        return None, ()
    return shortest_depth, tuple(sorted(shortest_first_actions))


def rows_for_seed(
    index: int,
    min_distance: int,
    max_distance: int,
    seed: int,
    template_keys: tuple[str, ...],
    max_attempts: int,
) -> list[SampleRow]:
    rng = random.Random(seed)
    rows: list[SampleRow] = []

    for attempt in range(max_attempts):
        motif_name = rng.choice(template_keys)
        state_expr, goal_expr, family = MOTIF_BUILDERS[motif_name](rng)
        state_str = str(state_expr)
        goal_str = str(goal_expr)
        distance, actions = shortest_actions_to_goal(state_str, goal_str, max_steps=max_distance)
        if distance is None or distance < min_distance or distance > max_distance or not actions:
            continue
        rows.append(
            SampleRow(
                state_str=state_str,
                goal_str=goal_str,
                valid_shortest_actions=multi_hot(list(actions)),
                distance_to_goal=distance,
                trajectory_id=f"traj_{seed}_{index}_{attempt}",
                expr_family=family,
                canonical_hash=canonicalize(state_str).digest,
                depth=distance,
            )
        )
        break
    return rows


def _worker_collect_rows(
    sample_indices: list[int],
    min_distance: int,
    max_distance: int,
    seed: int,
    template_keys: tuple[str, ...],
    max_attempts: int,
) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for index in sample_indices:
        rows.extend(
            rows_for_seed(
                index=index,
                min_distance=min_distance,
                max_distance=max_distance,
                seed=seed + index,
                template_keys=template_keys,
                max_attempts=max_attempts,
            )
        )
    return rows


def add_terminal_rows(rows: list[SampleRow], terminal_ratio: float, seed: int) -> list[SampleRow]:
    if not rows:
        return rows
    terminals_by_goal: dict[tuple[str, str], SampleRow] = {}
    for row in rows:
        key = (row.goal_str, row.expr_family)
        terminals_by_goal[key] = SampleRow(
            state_str=row.goal_str,
            goal_str=row.goal_str,
            valid_shortest_actions=[0] * len(ACTION_NAMES),
            distance_to_goal=0,
            trajectory_id=f"{row.trajectory_id}_goal",
            expr_family=row.expr_family,
            canonical_hash=canonicalize(row.goal_str).digest,
            depth=0,
        )

    terminals = list(terminals_by_goal.values())
    max_terminal = max(1, int(len(rows) * terminal_ratio))
    rng = random.Random(seed)
    if len(terminals) > max_terminal:
        terminals = rng.sample(terminals, k=max_terminal)
    return rows + terminals


def dedupe_rows(rows: list[SampleRow]) -> list[SampleRow]:
    deduped: dict[tuple[str, str, str, int], SampleRow] = {}
    for row in rows:
        key = (row.canonical_hash, canonicalize(row.goal_str).digest, row.expr_family, row.distance_to_goal)
        deduped[key] = row
    result = list(deduped.values())
    result.sort(key=lambda row: (row.expr_family, row.distance_to_goal, row.state_str))
    return result


def collect_rows(
    samples: int,
    min_distance: int,
    max_distance: int,
    seed: int,
    workers: int = 1,
    terminal_ratio: float = 0.25,
    template_keys: tuple[str, ...] = DEFAULT_TEMPLATE_KEYS,
    max_attempts: int = 12,
) -> list[SampleRow]:
    chunks = _chunk_indices(samples, workers)
    if len(chunks) == 1:
        rows = _worker_collect_rows(
            chunks[0],
            min_distance=min_distance,
            max_distance=max_distance,
            seed=seed,
            template_keys=template_keys,
            max_attempts=max_attempts,
        )
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _worker_collect_rows,
                    chunk,
                    min_distance,
                    max_distance,
                    seed + (chunk[0] * 10_000),
                    template_keys,
                    max_attempts,
                )
                for chunk in chunks
            ]
            rows = []
            for future in futures:
                rows.extend(future.result())

    rows = dedupe_rows(rows)
    rows = add_terminal_rows(rows, terminal_ratio=terminal_ratio, seed=seed)
    return dedupe_rows(rows)


def split_rows(rows: list[SampleRow]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = sorted({row.expr_family for row in rows})
    if len(families) >= 3:
        train_families = set(families[:-2])
        val_family = families[-2]
        test_family = families[-1]
        train = pd.DataFrame([asdict(row) for row in rows if row.expr_family in train_families])
        val = pd.DataFrame([asdict(row) for row in rows if row.expr_family == val_family])
        test = pd.DataFrame([asdict(row) for row in rows if row.expr_family == test_family])
        min_eval_rows = max(4, len(rows) // 20)
        if len(train) >= max(8, min_eval_rows * 2) and len(val) >= min_eval_rows and len(test) >= min_eval_rows:
            return train, val, test

    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()
    frame = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    n = len(frame)
    train_end = max(1, int(0.7 * n))
    val_end = max(train_end + 1, int(0.85 * n))
    train = frame.iloc[:train_end].reset_index(drop=True)
    val = frame.iloc[train_end:val_end].reset_index(drop=True)
    test = frame.iloc[val_end:].reset_index(drop=True)
    if test.empty and not val.empty:
        test = val.tail(1).reset_index(drop=True)
        val = val.iloc[:-1].reset_index(drop=True)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact-distance rewrite data from curated motif families")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--terminal-ratio", type=float, default=0.25)
    parser.add_argument("--template-keys", nargs="+", default=list(DEFAULT_TEMPLATE_KEYS))
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generated"))
    args = parser.parse_args()

    rows = collect_rows(
        samples=args.samples,
        min_distance=args.min_depth,
        max_distance=args.max_depth,
        seed=args.seed,
        workers=args.workers,
        terminal_ratio=args.terminal_ratio,
        template_keys=tuple(args.template_keys),
        max_attempts=args.max_attempts,
    )
    train, val, test = split_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet", index=False)
    val.to_parquet(args.output_dir / "val.parquet", index=False)
    test.to_parquet(args.output_dir / "test.parquet", index=False)

    depth_counts = Counter(row.distance_to_goal for row in rows)
    print(
        {
            "rows": len(rows),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "depth_counts": dict(sorted(depth_counts.items())),
            "workers": args.workers,
        }
    )


if __name__ == "__main__":
    main()
