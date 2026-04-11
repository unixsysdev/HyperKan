from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_gen.actions import ACTION_NAMES
from data_gen.canonicalize import canonicalize
from data_gen.generate_backward import shortest_actions_to_goal


def validate_frame(frame: pd.DataFrame) -> dict[str, int]:
    checked = 0
    valid_rows = 0
    terminal_rows = 0
    exact_match_rows = 0

    for row in frame.itertuples(index=False):
        checked += 1
        state = row.state_str
        goal = row.goal_str
        labels = list(row.valid_shortest_actions)
        goal_struct = canonicalize(goal).structural

        if row.distance_to_goal == 0:
            is_exact_terminal = canonicalize(state).structural == goal_struct and not any(labels)
            if is_exact_terminal:
                terminal_rows += 1
                valid_rows += 1
                exact_match_rows += 1
            continue

        expected_distance, optimal_actions = shortest_actions_to_goal(
            state,
            goal,
            max_steps=max(6, int(row.distance_to_goal)),
        )
        labeled_actions = {ACTION_NAMES[idx] for idx, is_valid in enumerate(labels) if is_valid}
        optimal_action_set = set(optimal_actions)

        if expected_distance == row.distance_to_goal and labeled_actions == optimal_action_set:
            valid_rows += 1
            exact_match_rows += 1
    return {
        "checked": checked,
        "valid_rows": valid_rows,
        "terminal_rows": terminal_rows,
        "exact_match_rows": exact_match_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated dataset files")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset)
    print(validate_frame(frame))


if __name__ == "__main__":
    main()
