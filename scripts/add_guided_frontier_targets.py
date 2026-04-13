from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.frontier_supervision import infer_hidden_cancel_action, parse_action_order


def load_action_vocab(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_frontier_targets(row: object, action_vocab: dict[str, int], horizon: int) -> list[int]:
    targets = [0] * len(action_vocab)
    if int(row.distance_to_goal) <= 0:
        return targets
    try:
        target_action_id = infer_hidden_cancel_action(str(row.action_order))
    except ValueError:
        return targets

    actions = parse_action_order(str(row.action_order))
    guided_action_id = str(row.guided_action_id)
    if guided_action_id not in action_vocab or target_action_id not in action_vocab:
        return targets
    try:
        current_idx = actions.index(guided_action_id)
        target_idx = actions.index(target_action_id)
    except ValueError:
        return targets

    if current_idx <= target_idx <= current_idx + horizon - 1:
        targets[action_vocab[guided_action_id]] = 1
    return targets


def annotate_frame(frame: pd.DataFrame, action_vocab: dict[str, int], horizon: int) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["frontier_targets"] = [
        build_frontier_targets(row, action_vocab=action_vocab, horizon=horizon)
        for row in annotated.itertuples(index=False)
    ]
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Add guided short-horizon frontier targets to scoped parquet splits")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    action_vocab = load_action_vocab(args.action_vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"horizon": int(args.horizon), "splits": {}}
    for split_name in ("train", "val", "test"):
        frame = pd.read_parquet(args.input_dir / f"{split_name}.parquet")
        annotated = annotate_frame(frame, action_vocab=action_vocab, horizon=args.horizon)
        annotated.to_parquet(args.output_dir / f"{split_name}.parquet", index=False)
        positives = int(sum(sum(targets) for targets in annotated["frontier_targets"]))
        summary["splits"][split_name] = {"rows": int(len(annotated)), "frontier_positive_labels": positives}

    for name in ("scoped_action_vocab.json", "meta.json"):
        source = args.input_dir / name
        if source.exists():
            target = args.output_dir / name
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (args.output_dir / "frontier_target_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
