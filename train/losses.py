from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_bce_with_logits(logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None and mask.any():
        logits = logits[mask]
        targets = targets[mask]
    elif mask is not None:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def value_huber_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    return F.huber_loss(predictions, targets.float(), delta=1.0)


def entropy_regularization(logits: Tensor) -> Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return -entropy.mean()


def mixture_entropy_regularization(weights: Tensor) -> Tensor:
    # weights are already probabilities over templates
    weights = weights.clamp_min(1e-12)
    entropy = -(weights * weights.log()).sum(dim=-1)
    return -entropy.mean()


def multi_task_loss(
    logits: Tensor,
    values: Tensor,
    action_targets: Tensor,
    value_targets: Tensor,
    value_weight: float = 0.3,
    entropy_weight: float = 0.05,
    mixture_weights: list[Tensor] | None = None,
    mixture_entropy_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    non_terminal = value_targets > 0
    action_loss = masked_bce_with_logits(logits, action_targets, mask=non_terminal)
    value_loss = value_huber_loss(values, value_targets)
    entropy_loss = entropy_regularization(logits[non_terminal]) if non_terminal.any() else torch.tensor(0.0, device=logits.device)
    mixture_entropy = torch.tensor(0.0, device=logits.device)
    if mixture_weights is not None and mixture_entropy_weight != 0.0:
        mixture_entropy = sum(mixture_entropy_regularization(item) for item in mixture_weights) / float(len(mixture_weights))
    total = action_loss + (value_weight * value_loss) + (entropy_weight * entropy_loss) + (mixture_entropy_weight * mixture_entropy)
    return total, {
        "action_loss": float(action_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy_loss": float(entropy_loss.detach()),
        "mixture_entropy_loss": float(mixture_entropy.detach()),
        "total_loss": float(total.detach()),
    }
