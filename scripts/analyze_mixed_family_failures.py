from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.canonicalize import structural_string
from data_gen.scoped_actions import apply_scoped_action_unchecked, parse_scoped_action_id
from models.factory import create_model
from search.scoped_beam_search import load_scoped_action_vocab
from tokenizer.srepr_tokenizer import SReprTokenizer


def is_global_root_action(action_id: str) -> bool:
    site_id, _ = parse_scoped_action_id(action_id)
    return site_id in {"expr@root", "numerator@root", "denominator@root"}


def classify_block(action_id: str) -> str:
    site_id, _ = parse_scoped_action_id(action_id)
    if site_id in {"expr@root", "numerator@root", "denominator@root"}:
        return "global"
    if site_id.startswith("expr@0") or site_id.startswith("expr@1") or site_id.startswith("add_slice@root[0:2]"):
        return "trig"
    if site_id.startswith("expr@2") or site_id.startswith("numerator@2"):
        return "hidden"
    return "other"


def encode_single(tokenizer: SReprTokenizer, expression: str, goal: str, max_length: int, device: torch.device):
    encoded = tokenizer.encode_pair(expression, goal, max_length=max_length)
    state_ids = torch.tensor([encoded.state_ids], dtype=torch.long, device=device)
    goal_ids = torch.tensor([encoded.goal_ids], dtype=torch.long, device=device)
    state_lengths = torch.tensor([encoded.state_length], dtype=torch.long, device=device)
    goal_lengths = torch.tensor([encoded.goal_length], dtype=torch.long, device=device)
    return state_ids, state_lengths, goal_ids, goal_lengths


def ranked_valid_actions(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    expression: str,
    goal: str,
    max_length: int,
    device: torch.device,
    root_penalty: float = 0.0,
    top_k: int | None = None,
) -> list[dict[str, object]]:
    state_ids, state_lengths, goal_ids, goal_lengths = encode_single(tokenizer, expression, goal, max_length, device)
    with torch.no_grad():
        outputs = model(state_ids, state_lengths, goal_ids, goal_lengths)
    logits = outputs["logits"][0].detach().clone()

    if root_penalty:
        for idx, action_id in enumerate(action_vocab):
            if is_global_root_action(action_id):
                logits[idx] -= root_penalty

    probabilities = torch.softmax(logits, dim=-1)
    ranked: list[dict[str, object]] = []
    for idx in torch.argsort(probabilities, descending=True).tolist():
        action_id = action_vocab[idx]
        site_id, op_name = parse_scoped_action_id(action_id)
        next_expr = apply_scoped_action_unchecked(expression, site_id, op_name)
        if next_expr is None:
            continue
        next_expr_str = str(next_expr)
        if next_expr_str == expression:
            continue
        ranked.append(
            {
                "action_id": action_id,
                "probability": float(probabilities[idx].item()),
                "next_expr": next_expr_str,
                "block": classify_block(action_id),
                "is_global_root": is_global_root_action(action_id),
            }
        )
        if top_k is not None and len(ranked) >= top_k:
            break
    return ranked


def run_beam_search_with_penalty(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    start_expr: str,
    goal_expr: str,
    max_length: int,
    device: torch.device,
    beam_width: int,
    max_steps: int,
    root_penalty: float,
) -> dict[str, object]:
    goal_struct = structural_string(goal_expr)
    beam = [{"expr": start_expr, "steps": [], "score": 0.0}]
    explored = 0

    for _ in range(max_steps):
        candidates: list[dict[str, object]] = []
        for node in beam:
            if structural_string(node["expr"]) == goal_struct:
                return {"success": True, "steps": node["steps"], "expansions": explored}

            ranked = ranked_valid_actions(
                model=model,
                tokenizer=tokenizer,
                action_vocab=action_vocab,
                expression=node["expr"],
                goal=goal_expr,
                max_length=max_length,
                device=device,
                root_penalty=root_penalty,
                top_k=None,
            )
            for action in ranked:
                explored += 1
                score = node["score"] + math.log(float(action["probability"]) + 1e-8)
                candidates.append(
                    {
                        "expr": action["next_expr"],
                        "steps": [*node["steps"], action["action_id"]],
                        "score": score,
                    }
                )

        if not candidates:
            break
        candidates.sort(key=lambda item: item["score"], reverse=True)
        beam = candidates[:beam_width]

    for node in beam:
        if structural_string(node["expr"]) == goal_struct:
            return {"success": True, "steps": node["steps"], "expansions": explored}
    return {"success": False, "steps": beam[0]["steps"] if beam else [], "expansions": explored}


def evaluate_with_penalty(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    max_length: int,
    device: torch.device,
    beam_width: int,
    max_steps: int,
    root_penalty: float,
) -> dict[str, object]:
    solved = 0
    attempts = 0
    total_expansions = 0
    for row in frame.itertuples(index=False):
        if row.distance_to_goal == 0:
            continue
        attempts += 1
        outcome = run_beam_search_with_penalty(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            max_length=max_length,
            device=device,
            beam_width=beam_width,
            max_steps=max_steps,
            root_penalty=root_penalty,
        )
        total_expansions += int(outcome["expansions"])
        if outcome["success"]:
            solved += 1
    return {
        "attempts": attempts,
        "solved": solved,
        "solve_rate": solved / attempts if attempts else 0.0,
        "mean_expansions": total_expansions / attempts if attempts else 0.0,
        "beam_width": beam_width,
        "root_penalty": root_penalty,
    }


def classify_first_action(chosen_action: str, intended_action: str) -> str:
    if chosen_action == intended_action:
        return "matches_intended"

    chosen_site, chosen_op = parse_scoped_action_id(chosen_action)
    intended_site, intended_op = parse_scoped_action_id(intended_action)

    if chosen_site == "expr@root":
        return "global_collapse_bias"
    if chosen_site == intended_site and chosen_op != intended_op:
        return "op_selection_failure"
    if chosen_site != intended_site and chosen_op == intended_op:
        return "site_selection_failure"
    return "site_and_op_failure"


def analyze_dataset(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    max_length: int,
    device: torch.device,
    max_steps: int,
) -> dict[str, object]:
    trajectory_intended: dict[str, dict[str, str]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        if row.distance_to_goal == 0:
            continue
        trajectory_intended[row.trajectory_id][structural_string(row.state_str)] = row.guided_action_id

    classification_counts: Counter[str] = Counter()
    first_action_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    step2_action_counts: Counter[str] = Counter()
    rows_out: list[dict[str, object]] = []

    for row in frame.itertuples(index=False):
        if row.distance_to_goal == 0:
            continue

        ranked_step1 = ranked_valid_actions(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            expression=row.state_str,
            goal=row.goal_str,
            max_length=max_length,
            device=device,
            top_k=3,
        )
        chosen_step1 = ranked_step1[0]
        first_action_id = str(chosen_step1["action_id"])
        first_action_counts[first_action_id] += 1
        block_counts[str(chosen_step1["block"])] += 1

        step2_top3: list[dict[str, object]] = []
        step2_intended = None
        if ranked_step1:
            next_expr = str(chosen_step1["next_expr"])
            step2_intended = trajectory_intended[row.trajectory_id].get(structural_string(next_expr))
            step2_top3 = ranked_valid_actions(
                model=model,
                tokenizer=tokenizer,
                action_vocab=action_vocab,
                expression=next_expr,
                goal=row.goal_str,
                max_length=max_length,
                device=device,
                top_k=3,
            )
            if step2_top3:
                step2_action_counts[str(step2_top3[0]["action_id"])] += 1

        greedy_outcome = run_beam_search_with_penalty(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            max_length=max_length,
            device=device,
            beam_width=1,
            max_steps=max_steps,
            root_penalty=0.0,
        )
        greedy_blocks = [classify_block(action_id) for action_id in greedy_outcome["steps"]]
        local_block_sequence = [block for block in greedy_blocks if block in {"trig", "hidden"}]
        ever_switches_blocks = any(
            later != earlier for earlier, later in zip(local_block_sequence, local_block_sequence[1:], strict=False)
        )

        first_action_class = classify_first_action(first_action_id, row.guided_action_id)
        classification_counts[first_action_class] += 1

        rows_out.append(
            {
                "trajectory_id": row.trajectory_id,
                "distance_to_goal": int(row.distance_to_goal),
                "state_str": row.state_str,
                "goal_str": row.goal_str,
                "intended_action_step1": row.guided_action_id,
                "top3_step1": [
                    {
                        "action_id": item["action_id"],
                        "probability": item["probability"],
                        "block": item["block"],
                        "is_global_root": item["is_global_root"],
                    }
                    for item in ranked_step1
                ],
                "chosen_action_step1": first_action_id,
                "first_action_classification": first_action_class,
                "intended_action_step2_if_on_path": step2_intended,
                "top3_step2_after_chosen_step1": [
                    {
                        "action_id": item["action_id"],
                        "probability": item["probability"],
                        "block": item["block"],
                        "is_global_root": item["is_global_root"],
                    }
                    for item in step2_top3
                ],
                "greedy_action_sequence": greedy_outcome["steps"],
                "greedy_block_sequence": greedy_blocks,
                "ever_switches_blocks": ever_switches_blocks,
                "greedy_success": bool(greedy_outcome["success"]),
            }
        )

    return {
        "attempts": len(rows_out),
        "first_action_counts": first_action_counts.most_common(),
        "first_action_classification_counts": classification_counts,
        "first_action_block_counts": block_counts,
        "step2_first_action_counts": step2_action_counts.most_common(),
        "rows": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mixed-family scoped failures and test a root-action penalty")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--penalties", type=float, nargs="*", default=(1.0, 2.0))
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    action_vocab = load_scoped_action_vocab(args.action_vocab)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    config["model"]["num_actions"] = len(action_vocab)

    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frame = pd.read_parquet(args.dataset)
    max_length = args.max_length if args.max_length is not None else config["data"].get("max_length", 256)

    analysis = analyze_dataset(
        frame=frame,
        model=model,
        tokenizer=tokenizer,
        action_vocab=action_vocab,
        max_length=max_length,
        device=device,
        max_steps=args.max_steps,
    )

    penalty_results: list[dict[str, object]] = []
    for penalty in args.penalties:
        for beam_width in (1, 4):
            penalty_results.append(
                evaluate_with_penalty(
                    frame=frame,
                    model=model,
                    tokenizer=tokenizer,
                    action_vocab=action_vocab,
                    max_length=max_length,
                    device=device,
                    beam_width=beam_width,
                    max_steps=args.max_steps,
                    root_penalty=float(penalty),
                )
            )

    output = {
        "model_type": payload["model_type"],
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "analysis": {
            **analysis,
            "first_action_classification_counts": dict(analysis["first_action_classification_counts"]),
            "first_action_block_counts": dict(analysis["first_action_block_counts"]),
        },
        "root_penalty_results": penalty_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
