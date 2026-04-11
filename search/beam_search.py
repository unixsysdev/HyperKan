from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch

from data_gen.actions import ACTION_NAMES, action_mask, apply_action
from data_gen.canonicalize import canonicalize
from tokenizer.srepr_tokenizer import SReprTokenizer


@dataclass(frozen=True)
class BeamNode:
    expression: str
    score: float
    steps: tuple[tuple[str, str], ...]
    visited: frozenset[str]


def _encode_single(tokenizer: SReprTokenizer, expression: str, goal: str, max_length: int, device: torch.device):
    encoded = tokenizer.encode_pair(expression, goal, max_length=max_length)
    state_ids = torch.tensor([encoded.state_ids], dtype=torch.long, device=device)
    goal_ids = torch.tensor([encoded.goal_ids], dtype=torch.long, device=device)
    state_lengths = torch.tensor([encoded.state_length], dtype=torch.long, device=device)
    goal_lengths = torch.tensor([encoded.goal_length], dtype=torch.long, device=device)
    return state_ids, state_lengths, goal_ids, goal_lengths


def run_beam_search(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    start_expr: str,
    goal_expr: str,
    beam_width: int = 4,
    max_steps: int = 8,
    max_length: int = 256,
    value_weight: float = 0.5,
    revisit_penalty: float = 1.5,
    device: torch.device | None = None,
) -> dict[str, object]:
    if device is None:
        device = next(model.parameters()).device

    start_hash = canonicalize(start_expr).digest
    goal_struct = canonicalize(goal_expr).structural
    beam = [
        BeamNode(
            expression=start_expr,
            score=0.0,
            steps=(),
            visited=frozenset({start_hash}),
        )
    ]
    explored = []

    for _ in range(max_steps):
        candidates: list[BeamNode] = []
        for node in beam:
            if canonicalize(node.expression).structural == goal_struct:
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
            logits = outputs["logits"][0]
            value = float(outputs["value"][0])
            probabilities = torch.softmax(logits, dim=-1)
            mask = action_mask(node.expression)

            for action_idx, applicable in enumerate(mask):
                if not applicable:
                    continue
                action = ACTION_NAMES[action_idx]
                result = apply_action(node.expression, action)
                if not result.valid or result.output_expr is None:
                    continue
                next_expr = str(result.output_expr)
                next_hash = canonicalize(next_expr).digest
                score = node.score + log(float(probabilities[action_idx]) + 1e-8) - (value_weight * value)
                visited = node.visited
                if next_hash in visited:
                    score -= revisit_penalty
                next_node = BeamNode(
                    expression=next_expr,
                    score=score,
                    steps=(*node.steps, (action, next_expr)),
                    visited=visited | {next_hash},
                )
                candidates.append(next_node)
                explored.append({"from": node.expression, "action": action, "to": next_expr, "score": score})
        if not candidates:
            break
        candidates.sort(key=lambda node: node.score, reverse=True)
        beam = candidates[:beam_width]

    solved = next((node for node in beam if canonicalize(node.expression).structural == goal_struct), None)
    return {"success": solved is not None, "node": solved, "explored": explored, "beam": beam}
