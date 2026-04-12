from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from models.factory import create_model
from search.beam_search import run_beam_search
from tokenizer.srepr_tokenizer import SReprTokenizer


def evaluate_dataset(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    frame: pd.DataFrame,
    beam_width: int,
    max_steps: int,
    max_length: int,
    value_weight: float,
    revisit_penalty: float,
    device: torch.device,
) -> dict[str, float]:
    solved = 0
    solved_steps = 0
    attempts = 0

    for row in frame.itertuples(index=False):
        if row.distance_to_goal == 0:
            continue
        attempts += 1
        outcome = run_beam_search(
            model=model,
            tokenizer=tokenizer,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            beam_width=beam_width,
            max_steps=max_steps,
            max_length=max_length,
            value_weight=value_weight,
            revisit_penalty=revisit_penalty,
            device=device,
        )
        if outcome["success"]:
            solved += 1
            solved_steps += len(outcome["node"].steps) if outcome["node"] is not None else 0

    solve_rate = solved / attempts if attempts else 0.0
    mean_steps = solved_steps / solved if solved else 0.0
    return {"attempts": attempts, "solved": solved, "solve_rate": solve_rate, "mean_solved_steps": mean_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verified end-to-end evaluation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/logs/verified_eval.json"))
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=None)
    parser.add_argument("--revisit-penalty", type=float, default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    search_cfg = config.get("search", {})
    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frame = pd.read_parquet(args.dataset)
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        beam_width=args.beam_width if args.beam_width is not None else search_cfg.get("beam_width", 4),
        max_steps=args.max_steps if args.max_steps is not None else search_cfg.get("max_steps", 8),
        max_length=args.max_length if args.max_length is not None else config["data"].get("max_length", 256),
        value_weight=args.value_weight if args.value_weight is not None else search_cfg.get("value_weight", 0.5),
        revisit_penalty=args.revisit_penalty if args.revisit_penalty is not None else search_cfg.get("revisit_penalty", 1.5),
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
