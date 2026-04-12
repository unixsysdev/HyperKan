from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed beam-eval challenge set from prior row outcomes")
    parser.add_argument("--dataset", type=Path, required=True, help="Parquet with test rows (e.g. artifacts/generated/test.parquet)")
    parser.add_argument(
        "--rows-json",
        type=Path,
        required=True,
        help="Row outcomes JSON from scripts/run_shallow_benchmark_worker.py --write-rows (e.g. hyperkan_beam_rows.json)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path for the challenge set")
    parser.add_argument("--meta", type=Path, default=None, help="Optional JSON metadata output")
    parser.add_argument("--depth", type=int, default=3, help="Only include rows with this shortest distance (default: 3)")
    parser.add_argument("--include-solved", action="store_true", help="Include solved rows (default is failures only)")
    parser.add_argument("--limit", type=int, default=80, help="Max rows in challenge set (default: 80)")
    parser.add_argument("--seed", type=int, default=17, help="Shuffle seed for deterministic sampling")
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset)
    outcomes = json.loads(args.rows_json.read_text(encoding="utf-8"))
    depth = int(args.depth)

    # Outcomes entries contain state_str + goal_str, which are unique enough for this benchmark.
    selected = []
    for row in outcomes:
        if int(row.get("depth", -1)) != depth:
            continue
        if not args.include_solved and bool(row.get("solved", False)):
            continue
        selected.append((row["state_str"], row["goal_str"]))

    # Deduplicate while keeping deterministic order.
    seen: set[tuple[str, str]] = set()
    dedup = []
    for key in selected:
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)

    rnd = random.Random(int(args.seed))
    rnd.shuffle(dedup)
    dedup = dedup[: int(args.limit)]

    # Join back to the canonical dataset rows.
    key_to_idx = {(r.state_str, r.goal_str): i for i, r in enumerate(frame.itertuples(index=False))}
    indices = [key_to_idx[k] for k in dedup if k in key_to_idx]
    out = frame.iloc[indices].copy()
    out.to_parquet(args.output, index=False)

    meta = {
        "source_dataset": str(args.dataset),
        "source_rows_json": str(args.rows_json),
        "depth": depth,
        "failures_only": not args.include_solved,
        "requested_limit": int(args.limit),
        "rows_selected": len(dedup),
        "rows_found_in_dataset": len(indices),
        "seed": int(args.seed),
        "output": str(args.output),
    }
    meta_path = args.meta if args.meta is not None else args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

