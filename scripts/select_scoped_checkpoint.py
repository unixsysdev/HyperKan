from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.run_scoped_smoke_eval import evaluate_dataset
from models.factory import create_model
from search.scoped_beam_search import build_scoped_action_factorization, load_scoped_action_vocab
from tokenizer.srepr_tokenizer import SReprTokenizer


def score_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    solve_rate = float(metrics["solve_rate"])
    mean_solved_expansions = float(metrics["mean_solved_expansions"])
    mean_expansions = float(metrics["mean_expansions"])
    return (solve_rate, -mean_solved_expansions, -mean_expansions)


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    action_vocab_path: Path,
    beam_width: int | None,
    max_steps: int | None,
    max_length: int | None,
    value_weight: float | None,
    revisit_penalty: float | None,
    policy_temperature: float | None,
    root_action_penalty: float | None,
    root_action_penalty_mode: str,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    action_vocab = load_scoped_action_vocab(action_vocab_path)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    config["model"]["num_actions"] = len(action_vocab)
    if bool(config["model"].get("site_op_factorized", False)) and "num_sites" not in config["model"]:
        factorization = build_scoped_action_factorization(action_vocab)
        config["model"]["num_sites"] = int(factorization["num_sites"])
        config["model"]["num_ops"] = int(factorization["num_ops"])
        config["model"]["action_to_site_idx"] = list(factorization["action_to_site_idx"])
        config["model"]["action_to_op_idx"] = list(factorization["action_to_op_idx"])
    search_cfg = config.get("search", {})

    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frame = pd.read_parquet(dataset_path)
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        action_vocab=action_vocab,
        frame=frame,
        beam_width=beam_width if beam_width is not None else search_cfg.get("beam_width", 4),
        max_steps=max_steps if max_steps is not None else search_cfg.get("max_steps", 8),
        max_length=max_length if max_length is not None else config["data"].get("max_length", 256),
        value_weight=value_weight if value_weight is not None else search_cfg.get("value_weight", 0.5),
        revisit_penalty=revisit_penalty if revisit_penalty is not None else search_cfg.get("revisit_penalty", 1.5),
        policy_temperature=policy_temperature if policy_temperature is not None else search_cfg.get("policy_temperature", 1.0),
        root_action_penalty=float(root_action_penalty or 0.0),
        root_action_penalty_mode=root_action_penalty_mode,
        device=device,
    )

    return {
        "checkpoint": str(checkpoint_path),
        "epoch": payload.get("epoch"),
        "model_type": payload["model_type"],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best scoped checkpoint by search metrics")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-best-to", type=Path, default=None)
    parser.add_argument("--include-best-pt", action="store_true")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=None)
    parser.add_argument("--revisit-penalty", type=float, default=None)
    parser.add_argument("--policy-temperature", type=float, default=None)
    parser.add_argument("--root-action-penalty", type=float, default=0.0)
    parser.add_argument("--root-action-penalty-mode", choices=("always", "mixed_signatures"), default="always")
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt"))
    if args.include_best_pt:
        best_pt = args.checkpoint_dir / "best.pt"
        if best_pt.exists():
            checkpoints.append(best_pt)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")

    results = [
        evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset_path=args.dataset,
            action_vocab_path=args.action_vocab,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            max_length=args.max_length,
            value_weight=args.value_weight,
            revisit_penalty=args.revisit_penalty,
            policy_temperature=args.policy_temperature,
            root_action_penalty=args.root_action_penalty,
            root_action_penalty_mode=args.root_action_penalty_mode,
        )
        for checkpoint_path in checkpoints
    ]
    ranked = sorted(results, key=lambda item: score_key(item["metrics"]), reverse=True)
    best = ranked[0]

    payload = {
        "selection_metric": "solve_rate_then_low_expansions",
        "dataset": str(args.dataset),
        "beam_width": args.beam_width,
        "max_steps": args.max_steps,
        "root_action_penalty": args.root_action_penalty,
        "root_action_penalty_mode": args.root_action_penalty_mode,
        "best": best,
        "results": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.copy_best_to is not None:
        args.copy_best_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["checkpoint"], args.copy_best_to)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
