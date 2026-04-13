from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.canonicalize import structural_string
from data_gen.scoped_actions import apply_scoped_action_unchecked
from models.factory import create_model
from search.scoped_beam_search import is_root_action_id, load_scoped_action_vocab
from tokenizer.srepr_tokenizer import SReprTokenizer


def encode_single(tokenizer: SReprTokenizer, expression: str, goal: str, max_length: int, device: torch.device):
    encoded = tokenizer.encode_pair(expression, goal, max_length=max_length)
    return (
        torch.tensor([encoded.state_ids], dtype=torch.long, device=device),
        torch.tensor([encoded.state_length], dtype=torch.long, device=device),
        torch.tensor([encoded.goal_ids], dtype=torch.long, device=device),
        torch.tensor([encoded.goal_length], dtype=torch.long, device=device),
    )


def valid_successors(expression: str, action_vocab: list[str]) -> list[tuple[int, str, str]]:
    successors: list[tuple[int, str, str]] = []
    for action_idx, action_id in enumerate(action_vocab):
        site_id, _, op_name = action_id.partition("::")
        next_expr = apply_scoped_action_unchecked(expression, site_id, op_name)
        if next_expr is None:
            continue
        next_expr_str = str(next_expr)
        if next_expr_str == expression:
            continue
        successors.append((action_idx, action_id, next_expr_str))
    return successors


def freeze_except_frontier_head(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    frontier_head = getattr(model, "frontier_head", None)
    if frontier_head is None:
        raise ValueError("Model has no frontier_head; enable model.use_frontier_head first")
    for parameter in frontier_head.parameters():
        parameter.requires_grad = True


def run_episode(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    state_expr: str,
    goal_expr: str,
    max_length: int,
    max_steps: int,
    device: torch.device,
    frontier_weight: float,
    root_action_penalty: float,
    expansion_penalty: float,
    entropy_weight: float,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    current = state_expr
    goal_struct = structural_string(goal_expr)
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    steps: list[str] = []
    solved = False

    for _ in range(max_steps):
        if structural_string(current) == goal_struct:
            solved = True
            break

        state_ids, state_lengths, goal_ids, goal_lengths = encode_single(tokenizer, current, goal_expr, max_length, device)
        outputs = model(state_ids, state_lengths, goal_ids, goal_lengths)
        if "frontier_logits" not in outputs:
            raise ValueError("Checkpoint model output has no frontier_logits")
        successors = valid_successors(current, action_vocab)
        if not successors:
            break

        action_indices = torch.tensor([item[0] for item in successors], dtype=torch.long, device=device)
        base_logits = outputs["logits"][0, action_indices].detach()
        frontier_logits = outputs["frontier_logits"][0, action_indices]
        logits = base_logits + (float(frontier_weight) * frontier_logits)
        if root_action_penalty:
            penalties = torch.tensor(
                [float(root_action_penalty) if is_root_action_id(item[1]) else 0.0 for item in successors],
                dtype=logits.dtype,
                device=device,
            )
            logits = logits - penalties

        distribution = torch.distributions.Categorical(logits=logits)
        sampled = distribution.sample()
        log_probs.append(distribution.log_prob(sampled))
        entropies.append(distribution.entropy())
        _, action_id, next_expr = successors[int(sampled.item())]
        steps.append(action_id)
        current = next_expr

    if structural_string(current) == goal_struct:
        solved = True

    reward = (1.0 if solved else 0.0) - (float(expansion_penalty) * len(steps))
    if not log_probs:
        return None, {"solved": solved, "reward": reward, "steps": steps}

    log_prob_sum = torch.stack(log_probs).sum()
    entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=device)
    loss = -(float(reward) * log_prob_sum) - (float(entropy_weight) * entropy)
    return loss, {"solved": solved, "reward": reward, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE fine-tune only the frontier head for scoped beam control")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frontier-weight", type=float, default=0.1)
    parser.add_argument("--root-action-penalty", type=float, default=0.0)
    parser.add_argument("--expansion-penalty", type=float, default=0.01)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    action_vocab = load_scoped_action_vocab(args.action_vocab)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    config["model"]["num_actions"] = len(action_vocab)
    config["model"]["use_frontier_head"] = True

    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    freeze_except_frontier_head(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=0.0,
    )

    frame = pd.read_parquet(args.dataset)
    frame = frame[frame["distance_to_goal"] > 0].reset_index(drop=True)
    if args.limit_rows > 0:
        frame = frame.head(args.limit_rows).copy()
    max_length = int(args.max_length if args.max_length is not None else config["data"].get("max_length", 256))

    history: list[dict[str, object]] = []
    indices = list(range(len(frame)))
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(indices)
        total_reward = 0.0
        solved = 0
        updates = 0
        for index in indices:
            row = frame.iloc[index]
            loss, outcome = run_episode(
                model=model,
                tokenizer=tokenizer,
                action_vocab=action_vocab,
                state_expr=str(row.state_str),
                goal_expr=str(row.goal_str),
                max_length=max_length,
                max_steps=int(args.max_steps),
                device=device,
                frontier_weight=float(args.frontier_weight),
                root_action_penalty=float(args.root_action_penalty),
                expansion_penalty=float(args.expansion_penalty),
                entropy_weight=float(args.entropy_weight),
            )
            total_reward += float(outcome["reward"])
            solved += int(bool(outcome["solved"]))
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            updates += 1

        record = {
            "epoch": epoch,
            "episodes": len(indices),
            "updates": updates,
            "solved": solved,
            "solve_rate": solved / len(indices) if indices else 0.0,
            "mean_reward": total_reward / len(indices) if indices else 0.0,
        }
        history.append(record)
        print(json.dumps(record))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = args.output_dir / "rl_frontier.pt"
    torch.save(
        {
            **payload,
            "config": config,
            "state_dict": model.state_dict(),
            "rl_frontier": {
                "source_checkpoint": str(args.checkpoint),
                "dataset": str(args.dataset),
                "epochs": int(args.epochs),
                "limit_rows": int(args.limit_rows),
                "frontier_weight": float(args.frontier_weight),
                "root_action_penalty": float(args.root_action_penalty),
                "expansion_penalty": float(args.expansion_penalty),
                "entropy_weight": float(args.entropy_weight),
                "history": history,
            },
        },
        output_checkpoint,
    )
    (args.output_dir / "rl_frontier_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(output_checkpoint), "history": history}, indent=2))


if __name__ == "__main__":
    main()
