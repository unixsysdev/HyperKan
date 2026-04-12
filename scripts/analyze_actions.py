"""Analyze first-action distribution in generated dataset.

Reports per-family and overall action label frequencies to check
whether trigsimp or any single action dominates the dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from data_gen.actions import ACTION_NAMES


def analyze(path: Path) -> None:
    frame = pd.read_parquet(path)
    non_terminal = frame[frame["distance_to_goal"] > 0]

    overall: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = {}

    for row in non_terminal.itertuples(index=False):
        actions = row.valid_shortest_actions
        family = row.expr_family
        if family not in by_family:
            by_family[family] = Counter()
        for idx, is_valid in enumerate(actions):
            if is_valid:
                overall[ACTION_NAMES[idx]] += 1
                by_family[family][ACTION_NAMES[idx]] += 1

    total_labels = sum(overall.values())
    print(f"\n=== {path.name}: {len(non_terminal)} non-terminal rows, {total_labels} action labels ===\n")

    print("Overall first-action distribution:")
    for action in ACTION_NAMES:
        count = overall.get(action, 0)
        pct = 100 * count / total_labels if total_labels else 0
        bar = "#" * int(pct / 2)
        print(f"  {action:10s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    for family in sorted(by_family):
        fam_total = sum(by_family[family].values())
        print(f"\n  [{family}] ({fam_total} labels)")
        for action in ACTION_NAMES:
            count = by_family[family].get(action, 0)
            pct = 100 * count / fam_total if fam_total else 0
            if count:
                print(f"    {action:10s}  {count:4d}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.datasets:
        analyze(path)


if __name__ == "__main__":
    main()
