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
    expr_root_action_mask: Tensor | None = None,
    expr_root_avoidance_weight: float = 0.0,
    site_logits: Tensor | None = None,
    op_logits: Tensor | None = None,
    action_site_membership: Tensor | None = None,
    action_op_membership: Tensor | None = None,
    site_loss_weight: float = 0.0,
    op_loss_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    non_terminal = value_targets > 0
    action_loss = masked_bce_with_logits(logits, action_targets, mask=non_terminal)
    value_loss = value_huber_loss(values, value_targets)
    entropy_loss = entropy_regularization(logits[non_terminal]) if non_terminal.any() else torch.tensor(0.0, device=logits.device)
    mixture_entropy = torch.tensor(0.0, device=logits.device)
    expr_root_avoidance = torch.tensor(0.0, device=logits.device)
    site_loss = torch.tensor(0.0, device=logits.device)
    op_loss = torch.tensor(0.0, device=logits.device)
    if mixture_weights is not None and mixture_entropy_weight != 0.0:
        mixture_entropy = sum(mixture_entropy_regularization(item) for item in mixture_weights) / float(len(mixture_weights))
    if expr_root_action_mask is not None and expr_root_avoidance_weight != 0.0:
        root_mass_mask = expr_root_action_mask.to(device=logits.device, dtype=torch.bool)
        if root_mass_mask.any():
            root_allowed = action_targets[:, root_mass_mask].sum(dim=-1) > 0
            avoid_mask = non_terminal & (~root_allowed)
            if avoid_mask.any():
                probs = torch.softmax(logits, dim=-1)
                expr_root_avoidance = probs[:, root_mass_mask].sum(dim=-1)[avoid_mask].mean()
    if site_logits is not None and action_site_membership is not None and site_loss_weight != 0.0:
        site_targets = (action_targets @ action_site_membership.to(device=logits.device, dtype=action_targets.dtype) > 0).float()
        site_loss = masked_bce_with_logits(site_logits, site_targets, mask=non_terminal)
    if op_logits is not None and action_op_membership is not None and op_loss_weight != 0.0:
        op_targets = (action_targets @ action_op_membership.to(device=logits.device, dtype=action_targets.dtype) > 0).float()
        op_loss = masked_bce_with_logits(op_logits, op_targets, mask=non_terminal)
    total = (
        action_loss
        + (value_weight * value_loss)
        + (entropy_weight * entropy_loss)
        + (mixture_entropy_weight * mixture_entropy)
        + (expr_root_avoidance_weight * expr_root_avoidance)
        + (site_loss_weight * site_loss)
        + (op_loss_weight * op_loss)
    )
    return total, {
        "action_loss": float(action_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy_loss": float(entropy_loss.detach()),
        "mixture_entropy_loss": float(mixture_entropy.detach()),
        "expr_root_avoidance_loss": float(expr_root_avoidance.detach()),
        "site_loss": float(site_loss.detach()),
        "op_loss": float(op_loss.detach()),
        "total_loss": float(total.detach()),
    }
