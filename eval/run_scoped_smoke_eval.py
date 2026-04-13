from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from models.factory import create_model
from search.scoped_beam_search import build_scoped_action_factorization, load_scoped_action_vocab, run_scoped_beam_search
from tokenizer.srepr_tokenizer import SReprTokenizer


def evaluate_dataset(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    frame: pd.DataFrame,
    beam_width: int,
    max_steps: int,
    max_length: int,
    value_weight: float,
    revisit_penalty: float,
    policy_temperature: float,
    root_action_penalty: float,
    root_action_penalty_mode: str,
    early_hidden_bonus: float,
    early_hidden_bonus_steps: int,
    device: torch.device,
) -> dict[str, object]:
    solved = 0
    solved_steps = 0
    solved_expansions = 0
    attempts = 0
    total_expansions = 0
    per_family: dict[str, dict[str, float]] = defaultdict(
        lambda: {"attempts": 0, "solved": 0, "solved_steps": 0, "total_expansions": 0, "solved_expansions": 0}
    )

    for row in frame.itertuples(index=False):
        if row.distance_to_goal == 0:
            continue
        attempts += 1
        family = getattr(row, "expr_family", "unknown")
        per_family[family]["attempts"] += 1
        outcome = run_scoped_beam_search(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            beam_width=beam_width,
            max_steps=max_steps,
            max_length=max_length,
            value_weight=value_weight,
            revisit_penalty=revisit_penalty,
            policy_temperature=policy_temperature,
            root_action_penalty=root_action_penalty,
            root_action_penalty_mode=root_action_penalty_mode,
            early_hidden_bonus=early_hidden_bonus,
            early_hidden_bonus_steps=early_hidden_bonus_steps,
            device=device,
        )
        expansions = len(outcome.get("explored", ()))
        total_expansions += expansions
        per_family[family]["total_expansions"] += expansions
        if outcome["success"]:
            solved += 1
            steps = len(outcome["node"].steps) if outcome["node"] is not None else 0
            solved_steps += steps
            solved_expansions += expansions
            per_family[family]["solved"] += 1
            per_family[family]["solved_steps"] += steps
            per_family[family]["solved_expansions"] += expansions

    solve_rate = solved / attempts if attempts else 0.0
    mean_steps = solved_steps / solved if solved else 0.0
    mean_expansions = total_expansions / attempts if attempts else 0.0
    mean_solved_expansions = solved_expansions / solved if solved else 0.0

    family_metrics: dict[str, dict[str, float]] = {}
    for family, stats in sorted(per_family.items()):
        family_attempts = int(stats["attempts"])
        family_solved = int(stats["solved"])
        family_metrics[family] = {
            "attempts": family_attempts,
            "solved": family_solved,
            "solve_rate": family_solved / family_attempts if family_attempts else 0.0,
            "mean_solved_steps": stats["solved_steps"] / family_solved if family_solved else 0.0,
            "mean_expansions": stats["total_expansions"] / family_attempts if family_attempts else 0.0,
            "mean_solved_expansions": stats["solved_expansions"] / family_solved if family_solved else 0.0,
        }

    return {
        "attempts": attempts,
        "solved": solved,
        "solve_rate": solve_rate,
        "mean_solved_steps": mean_steps,
        "mean_expansions": mean_expansions,
        "mean_solved_expansions": mean_solved_expansions,
        "per_family": family_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scoped-action smoke beam evaluation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scoped_smoke_eval.json"))
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=None)
    parser.add_argument("--revisit-penalty", type=float, default=None)
    parser.add_argument("--policy-temperature", type=float, default=None)
    parser.add_argument("--root-action-penalty", type=float, default=0.0)
    parser.add_argument("--root-action-penalty-mode", choices=("always", "mixed_signatures"), default="always")
    parser.add_argument("--early-hidden-bonus", type=float, default=0.0)
    parser.add_argument("--early-hidden-bonus-steps", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    action_vocab = load_scoped_action_vocab(args.action_vocab)
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
    frame = pd.read_parquet(args.dataset)
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        action_vocab=action_vocab,
        frame=frame,
        beam_width=args.beam_width if args.beam_width is not None else search_cfg.get("beam_width", 4),
        max_steps=args.max_steps if args.max_steps is not None else search_cfg.get("max_steps", 8),
        max_length=args.max_length if args.max_length is not None else config["data"].get("max_length", 256),
        value_weight=args.value_weight if args.value_weight is not None else search_cfg.get("value_weight", 0.5),
        revisit_penalty=args.revisit_penalty if args.revisit_penalty is not None else search_cfg.get("revisit_penalty", 1.5),
        policy_temperature=args.policy_temperature if args.policy_temperature is not None else search_cfg.get("policy_temperature", 1.0),
        root_action_penalty=float(args.root_action_penalty),
        root_action_penalty_mode=args.root_action_penalty_mode,
        early_hidden_bonus=float(args.early_hidden_bonus),
        early_hidden_bonus_steps=int(args.early_hidden_bonus_steps),
        device=device,
    )
    metrics["root_action_penalty"] = float(args.root_action_penalty)
    metrics["root_action_penalty_mode"] = str(args.root_action_penalty_mode)
    metrics["early_hidden_bonus"] = float(args.early_hidden_bonus)
    metrics["early_hidden_bonus_steps"] = int(args.early_hidden_bonus_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
