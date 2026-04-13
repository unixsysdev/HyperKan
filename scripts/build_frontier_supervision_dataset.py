from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.frontier_supervision import build_frontier_label_frame


def load_action_vocab(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [action_id for action_id, _ in sorted(payload.items(), key=lambda item: int(item[1]))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learned-frontier candidate labels for scoped datasets")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=4_000)
    parser.add_argument("--limit-rows", type=int, default=0, help="optional input row cap for smoke checks")
    parser.add_argument("--skip-distance-label", action="store_true", help="skip the heavier bounded goal-distance label")
    args = parser.parse_args()

    action_vocab = load_action_vocab(args.action_vocab)
    frame = pd.read_parquet(args.dataset)
    if args.limit_rows > 0:
        frame = frame.head(args.limit_rows).copy()
    label_frame = build_frontier_label_frame(
        frame=frame,
        action_vocab=action_vocab,
        horizon=args.horizon,
        max_nodes=args.max_nodes,
        include_distance_label=not args.skip_distance_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_frame.to_parquet(args.output, index=False)
    summary = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "input_rows": int(len(frame)),
        "rows": int(len(label_frame)),
        "horizon": int(args.horizon),
        "include_distance_label": not bool(args.skip_distance_label),
        "target_action_counts": label_frame["target_action_id"].value_counts().to_dict() if len(label_frame) else {},
        "positive_counts": {
            "reaches_target_site_within_horizon": int(label_frame["reaches_target_site_within_horizon"].sum()) if len(label_frame) else 0,
            "reaches_target_action_within_horizon": int(label_frame["reaches_target_action_within_horizon"].sum()) if len(label_frame) else 0,
            "reaches_goal_within_horizon": int(label_frame["reaches_goal_within_horizon"].sum()) if len(label_frame) else 0,
            "reduces_distance_to_goal": int(label_frame["reduces_distance_to_goal"].sum()) if len(label_frame) else 0,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
