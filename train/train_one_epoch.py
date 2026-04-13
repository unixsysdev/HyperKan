from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from tokenizer.srepr_tokenizer import SReprTokenizer
from train.losses import multi_task_loss


@dataclass(frozen=True)
class Batch:
    state_ids: Tensor
    goal_ids: Tensor
    state_lengths: Tensor
    goal_lengths: Tensor
    action_targets: Tensor
    value_targets: Tensor


class RewriteDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.records = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def build_tokenizer(train_path: str | Path, save_path: str | Path) -> SReprTokenizer:
    frame = pd.read_parquet(train_path)
    tokenizer = SReprTokenizer()
    tokenizer.build_vocab(frame["state_str"].tolist())
    tokenizer.build_vocab(frame["goal_str"].tolist())
    tokenizer.save(save_path)
    return tokenizer


def build_collate_fn(tokenizer: SReprTokenizer, max_length: int):
    def collate(rows: list[dict[str, Any]]) -> Batch:
        state_ids = []
        goal_ids = []
        state_lengths = []
        goal_lengths = []
        action_targets = []
        value_targets = []

        for row in rows:
            encoded = tokenizer.encode_pair(row["state_str"], row["goal_str"], max_length=max_length)
            state_ids.append(encoded.state_ids)
            goal_ids.append(encoded.goal_ids)
            state_lengths.append(encoded.state_length)
            goal_lengths.append(encoded.goal_length)
            action_targets.append(row["valid_shortest_actions"])
            value_targets.append(row["distance_to_goal"])

        return Batch(
            state_ids=torch.tensor(state_ids, dtype=torch.long),
            goal_ids=torch.tensor(goal_ids, dtype=torch.long),
            state_lengths=torch.tensor(state_lengths, dtype=torch.long),
            goal_lengths=torch.tensor(goal_lengths, dtype=torch.long),
            action_targets=torch.tensor(np.asarray(action_targets), dtype=torch.float32),
            value_targets=torch.tensor(value_targets, dtype=torch.float32),
        )

    return collate


def build_dataloader(
    parquet_path: str | Path,
    tokenizer: SReprTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[Batch]:
    frame = pd.read_parquet(parquet_path)
    dataset = RewriteDataset(frame)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collate_fn(tokenizer, max_length=max_length),
    )


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        state_ids=batch.state_ids.to(device),
        goal_ids=batch.goal_ids.to(device),
        state_lengths=batch.state_lengths.to(device),
        goal_lengths=batch.goal_lengths.to(device),
        action_targets=batch.action_targets.to(device),
        value_targets=batch.value_targets.to(device),
    )


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    value_weight: float = 0.3,
    entropy_weight: float = 0.05,
    mixture_entropy_weight: float = 0.0,
    grad_clip_norm: float = 1.0,
    expr_root_action_mask: Tensor | None = None,
    expr_root_avoidance_weight: float = 0.0,
    action_site_membership: Tensor | None = None,
    action_op_membership: Tensor | None = None,
    site_loss_weight: float = 0.0,
    op_loss_weight: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    steps = 0

    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch.state_ids, batch.state_lengths, batch.goal_ids, batch.goal_lengths)
        mixture_weights = outputs.get("mixture_weights") if isinstance(outputs, dict) else None
        loss, metrics = multi_task_loss(
            logits=outputs["logits"],
            values=outputs["value"],
            action_targets=batch.action_targets,
            value_targets=batch.value_targets,
            value_weight=value_weight,
            entropy_weight=entropy_weight,
            mixture_weights=mixture_weights if isinstance(mixture_weights, list) else None,
            mixture_entropy_weight=mixture_entropy_weight,
            expr_root_action_mask=expr_root_action_mask,
            expr_root_avoidance_weight=expr_root_avoidance_weight,
            site_logits=outputs.get("site_logits") if isinstance(outputs, dict) else None,
            op_logits=outputs.get("op_logits") if isinstance(outputs, dict) else None,
            action_site_membership=action_site_membership,
            action_op_membership=action_op_membership,
            site_loss_weight=site_loss_weight,
            op_loss_weight=op_loss_weight,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1

    if steps == 0:
        return totals
    return {key: value / steps for key, value in totals.items()}


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
