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
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
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
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        max_length=args.max_length,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
