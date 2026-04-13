from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from math import log
from pathlib import Path

import torch

from data_gen.canonicalize import structural_string
from data_gen.scoped_actions import apply_scoped_action_unchecked
from tokenizer.srepr_tokenizer import SReprTokenizer


@dataclass(frozen=True)
class ScopedBeamNode:
    expression: str
    score: float
    steps: tuple[tuple[str, str], ...]
    visited: frozenset[str]


def load_scoped_action_vocab(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [action_id for action_id, _ in sorted(payload.items(), key=lambda item: int(item[1]))]


def build_scoped_action_factorization(action_vocab: list[str]) -> dict[str, object]:
    site_vocab: list[str] = []
    op_vocab: list[str] = []
    site_to_idx: dict[str, int] = {}
    op_to_idx: dict[str, int] = {}
    action_to_site_idx: list[int] = []
    action_to_op_idx: list[int] = []

    for action_id in action_vocab:
        site_id, _, op_name = action_id.partition("::")
        site_idx = site_to_idx.setdefault(site_id, len(site_vocab))
        if site_idx == len(site_vocab):
            site_vocab.append(site_id)
        op_idx = op_to_idx.setdefault(op_name, len(op_vocab))
        if op_idx == len(op_vocab):
            op_vocab.append(op_name)
        action_to_site_idx.append(site_idx)
        action_to_op_idx.append(op_idx)

    return {
        "site_vocab": site_vocab,
        "op_vocab": op_vocab,
        "num_sites": len(site_vocab),
        "num_ops": len(op_vocab),
        "action_to_site_idx": action_to_site_idx,
        "action_to_op_idx": action_to_op_idx,
    }


def _encode_single(tokenizer: SReprTokenizer, expression: str, goal: str, max_length: int, device: torch.device):
    encoded = tokenizer.encode_pair(expression, goal, max_length=max_length)
    state_ids = torch.tensor([encoded.state_ids], dtype=torch.long, device=device)
    goal_ids = torch.tensor([encoded.goal_ids], dtype=torch.long, device=device)
    state_lengths = torch.tensor([encoded.state_length], dtype=torch.long, device=device)
    goal_lengths = torch.tensor([encoded.goal_length], dtype=torch.long, device=device)
    return state_ids, state_lengths, goal_ids, goal_lengths


def _cheap_hash(expression: str) -> str:
    return sha1(expression.encode("utf-8")).hexdigest()


def is_root_action_id(action_id: str) -> bool:
    site_id, _, _ = action_id.partition("::")
    return site_id in {"expr@root", "numerator@root", "denominator@root"}


def run_scoped_beam_search(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    action_vocab: list[str],
    start_expr: str,
    goal_expr: str,
    beam_width: int = 4,
    max_steps: int = 8,
    max_length: int = 256,
    value_weight: float = 0.5,
    revisit_penalty: float = 1.5,
    policy_temperature: float = 1.0,
    root_action_penalty: float = 0.0,
    device: torch.device | None = None,
) -> dict[str, object]:
    if device is None:
        device = next(model.parameters()).device

    start_hash = _cheap_hash(start_expr)
    goal_struct = structural_string(goal_expr)
    beam = [
        ScopedBeamNode(
            expression=start_expr,
            score=0.0,
            steps=(),
            visited=frozenset({start_hash}),
        )
    ]
    explored = []

    for _ in range(max_steps):
        candidates: list[ScopedBeamNode] = []
        for node in beam:
            if structural_string(node.expression) == goal_struct:
                return {"success": True, "node": node, "explored": explored}

            state_ids, state_lengths, goal_ids, goal_lengths = _encode_single(
                tokenizer,
                node.expression,
                goal_expr,
                max_length=max_length,
                device=device,
            )
            with torch.no_grad():
                outputs = model(state_ids, state_lengths, goal_ids, goal_lengths)
            logits = outputs["logits"][0].detach().clone()
            if root_action_penalty:
                for action_idx, action_id in enumerate(action_vocab):
                    if is_root_action_id(action_id):
                        logits[action_idx] -= float(root_action_penalty)
            value = float(outputs["value"][0])
            temperature = max(float(policy_temperature), 1e-6)
            probabilities = torch.softmax(logits / temperature, dim=-1)

            for action_idx, action_id in enumerate(action_vocab):
                site_id, _, op_name = action_id.partition("::")
                output_expr = apply_scoped_action_unchecked(node.expression, site_id, op_name)
                if output_expr is None:
                    continue
                next_expr = str(output_expr)
                if next_expr == node.expression:
                    continue
                next_hash = _cheap_hash(next_expr)
                score = node.score + log(float(probabilities[action_idx]) + 1e-8) - (value_weight * value)
                visited = node.visited
                if next_hash in visited:
                    score -= revisit_penalty
                next_node = ScopedBeamNode(
                    expression=next_expr,
                    score=score,
                    steps=(*node.steps, (action_id, next_expr)),
                    visited=visited | {next_hash},
                )
                candidates.append(next_node)
                explored.append({"from": node.expression, "action": action_id, "to": next_expr, "score": score})
        if not candidates:
            break
        candidates.sort(key=lambda node: node.score, reverse=True)
        beam = candidates[:beam_width]

    solved = next((node for node in beam if structural_string(node.expression) == goal_struct), None)
    return {"success": solved is not None, "node": solved, "explored": explored, "beam": beam}
