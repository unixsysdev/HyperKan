from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.canonicalize import structural_string
from data_gen.scoped_actions import apply_scoped_action_unchecked, parse_scoped_action_id
from models.factory import create_model
from search.scoped_beam_search import build_scoped_action_factorization, load_scoped_action_vocab
from tokenizer.srepr_tokenizer import SReprTokenizer


def classify_block(action_id: str) -> str:
    site_id, _ = parse_scoped_action_id(action_id)
    if site_id in {"expr@root", "numerator@root", "denominator@root"}:
        return "global"
    if site_id.startswith("expr@0") or site_id.startswith("expr@1") or site_id.startswith("add_slice@root[0:2]"):
        return "trig"
    if site_id.startswith("expr@2") or site_id.startswith("numerator@2"):
        return "hidden"
    return "other"


def is_global_root_action(action_id: str) -> bool:
    site_id, _ = parse_scoped_action_id(action_id)
    return site_id in {"expr@root", "numerator@root", "denominator@root"}


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
                logits[idx] -= float(root_penalty)

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
                "block": classify_block(action_id),
                "is_global_root": is_global_root_action(action_id),
                "next_expr": next_expr_str,
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


def first_non_root_site(path: list[str]) -> str | None:
    for action_id in path:
        site_id, _ = parse_scoped_action_id(action_id)
        if not is_global_root_action(action_id):
            return site_id
    return None


def first_non_root_block(path: list[str]) -> str | None:
    for action_id in path:
        block = classify_block(action_id)
        if block != "global":
            return block
    return None


def reaches_hidden_site_early(path: list[str], max_actions: int) -> bool:
    return any(classify_block(action_id) == "hidden" for action_id in path[:max_actions])


def reaches_hidden_cancel_early(path: list[str], max_actions: int) -> bool:
    return any(action_id == "expr@2::cancel" for action_id in path[:max_actions])


def compare_first_action(default_action: str, penalized_action: str) -> str:
    default_site, default_op = parse_scoped_action_id(default_action)
    penalized_site, penalized_op = parse_scoped_action_id(penalized_action)
    if default_action == penalized_action:
        return "unchanged"
    if default_site != penalized_site and default_op == penalized_op:
        return "site_only_change"
    if default_site == penalized_site and default_op != penalized_op:
        return "op_only_change"
    return "site_and_op_change"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze unconditional root-penalty rescue paths on mixed-family states")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--root-penalty", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=None)
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

    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frame = pd.read_parquet(args.dataset)
    max_length = args.max_length if args.max_length is not None else config["data"].get("max_length", 256)

    rows: list[dict[str, object]] = []
    solved_counter: Counter[str] = Counter()
    unsolved_counter: Counter[str] = Counter()
    first_non_root_site_counter: Counter[str] = Counter()
    first_non_root_block_counter: Counter[str] = Counter()
    solved_path_prefix_counter: Counter[str] = Counter()
    solved_hidden_access_counter: Counter[str] = Counter()
    unsolved_hidden_access_counter: Counter[str] = Counter()
    solved = 0
    attempts = 0

    for row in frame.itertuples(index=False):
        if row.distance_to_goal == 0:
            continue
        attempts += 1
        default_top3 = ranked_valid_actions(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            expression=row.state_str,
            goal=row.goal_str,
            max_length=max_length,
            device=device,
            root_penalty=0.0,
            top_k=3,
        )
        penalized_top3 = ranked_valid_actions(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            expression=row.state_str,
            goal=row.goal_str,
            max_length=max_length,
            device=device,
            root_penalty=float(args.root_penalty),
            top_k=3,
        )
        outcome = run_beam_search_with_penalty(
            model=model,
            tokenizer=tokenizer,
            action_vocab=action_vocab,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            max_length=max_length,
            device=device,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            root_penalty=float(args.root_penalty),
        )

        default_first = str(default_top3[0]["action_id"])
        penalized_first = str(penalized_top3[0]["action_id"])
        first_action_delta = compare_first_action(default_first, penalized_first)
        path = [str(item) for item in outcome["steps"]]
        first_local_site = first_non_root_site(path)
        first_local_block = first_non_root_block(path)
        path_prefix = " -> ".join(path[:3]) if path else "<empty>"
        reaches_hidden3 = reaches_hidden_site_early(path, 3)
        reaches_cancel3 = reaches_hidden_cancel_early(path, 3)

        if outcome["success"]:
            solved += 1
            solved_counter[first_action_delta] += 1
            solved_hidden_access_counter[f"hidden_site<=3:{reaches_hidden3}"] += 1
            solved_hidden_access_counter[f"hidden_cancel<=3:{reaches_cancel3}"] += 1
            if first_local_site is not None:
                first_non_root_site_counter[first_local_site] += 1
            if first_local_block is not None:
                first_non_root_block_counter[first_local_block] += 1
            solved_path_prefix_counter[path_prefix] += 1
        else:
            unsolved_counter[first_action_delta] += 1
            unsolved_hidden_access_counter[f"hidden_site<=3:{reaches_hidden3}"] += 1
            unsolved_hidden_access_counter[f"hidden_cancel<=3:{reaches_cancel3}"] += 1

        rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "default_top3": [
                    {
                        "action_id": item["action_id"],
                        "probability": item["probability"],
                        "block": item["block"],
                    }
                    for item in default_top3
                ],
                "penalized_top3": [
                    {
                        "action_id": item["action_id"],
                        "probability": item["probability"],
                        "block": item["block"],
                    }
                    for item in penalized_top3
                ],
                "first_action_delta": first_action_delta,
                "penalized_beam_success": bool(outcome["success"]),
                "penalized_beam_path": path,
                "penalized_beam_expansions": int(outcome["expansions"]),
                "first_non_root_site": first_local_site,
                "first_non_root_block": first_local_block,
                "path_prefix_3": path_prefix,
                "reaches_hidden_site_within_3": reaches_hidden3,
                "reaches_hidden_cancel_within_3": reaches_cancel3,
            }
        )

    output = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "beam_width": int(args.beam_width),
        "max_steps": int(args.max_steps),
        "root_penalty": float(args.root_penalty),
        "attempts": attempts,
        "solved": solved,
        "solve_rate": (solved / attempts) if attempts else 0.0,
        "summary": {
            "solved_first_action_delta_counts": dict(solved_counter),
            "unsolved_first_action_delta_counts": dict(unsolved_counter),
            "solved_first_non_root_site_counts": dict(first_non_root_site_counter),
            "solved_first_non_root_block_counts": dict(first_non_root_block_counter),
            "solved_path_prefix_counts": dict(solved_path_prefix_counter.most_common(10)),
            "solved_early_hidden_access_counts": dict(solved_hidden_access_counter),
            "unsolved_early_hidden_access_counts": dict(unsolved_hidden_access_counter),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
