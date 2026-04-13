from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.encoder_bigru import SharedBiGRUEncoder, fuse_state_goal
from models.kan_head import KANSplineState, SimpleKANLayer


class HyperKANPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_actions: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        kan_hidden_dim: int = 128,
        basis_dim: int = 8,
        num_templates: int = 4,
        hyper_hidden_dim: int | None = None,
        condition_kan1: bool = True,
        mixture_temperature: float = 1.0,
        site_op_factorized: bool = False,
        num_sites: int | None = None,
        num_ops: int | None = None,
        action_to_site_idx: list[int] | None = None,
        action_to_op_idx: list[int] | None = None,
        use_frontier_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_templates = num_templates
        self.condition_kan1 = condition_kan1
        self.mixture_temperature = float(mixture_temperature)
        self.site_op_factorized = bool(site_op_factorized)
        self.use_frontier_head = bool(use_frontier_head)
        self.encoder = SharedBiGRUEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            dropout=dropout,
        )
        fused_dim = hidden_dim * 4
        self.pre = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, kan_hidden_dim), nn.GELU())
        hyper_hidden_dim = int(hyper_hidden_dim if hyper_hidden_dim is not None else hidden_dim)
        self.hyper = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hyper_hidden_dim),
            nn.GELU(),
            nn.Linear(hyper_hidden_dim, num_templates * 2),
        )
        self.kan_1 = SimpleKANLayer(kan_hidden_dim, kan_hidden_dim, basis_dim=basis_dim, num_templates=num_templates)
        if self.site_op_factorized:
            if num_sites is None or num_ops is None or action_to_site_idx is None or action_to_op_idx is None:
                raise ValueError("Factorized HyperKAN requires site/op metadata")
            self.site_head = SimpleKANLayer(kan_hidden_dim, num_sites, basis_dim=basis_dim, num_templates=num_templates)
            self.op_head = SimpleKANLayer(kan_hidden_dim, num_ops, basis_dim=basis_dim, num_templates=num_templates)
            self.register_buffer("action_to_site_idx", torch.tensor(action_to_site_idx, dtype=torch.long), persistent=False)
            self.register_buffer("action_to_op_idx", torch.tensor(action_to_op_idx, dtype=torch.long), persistent=False)
            self.kan_2 = None
        else:
            self.kan_2 = SimpleKANLayer(kan_hidden_dim, num_actions, basis_dim=basis_dim, num_templates=num_templates)
            self.site_head = None
            self.op_head = None
        self.value_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.frontier_head = (
            nn.Sequential(
                nn.LayerNorm(kan_hidden_dim),
                nn.Linear(kan_hidden_dim, kan_hidden_dim),
                nn.GELU(),
                nn.Linear(kan_hidden_dim, num_actions),
            )
            if self.use_frontier_head
            else None
        )
        self.dropout = nn.Dropout(dropout)

    def _mixtures(self, goal_embedding: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.hyper(goal_embedding)
        part_a, part_b = raw.chunk(2, dim=-1)
        temperature = max(self.mixture_temperature, 1e-6)
        return (part_a / temperature).softmax(dim=-1), (part_b / temperature).softmax(dim=-1)

    def forward(
        self,
        state_ids: Tensor,
        state_lengths: Tensor,
        goal_ids: Tensor,
        goal_lengths: Tensor,
    ) -> dict[str, Tensor | list[KANSplineState]]:
        state_out = self.encoder(state_ids, state_lengths)
        goal_out = self.encoder(goal_ids, goal_lengths)
        fused = fuse_state_goal(state_out.pooled, goal_out.pooled)
        hidden = self.pre(fused)
        mix_1, mix_2 = self._mixtures(goal_out.pooled)
        if self.condition_kan1:
            hidden, spline_1 = self.kan_1(hidden, mixture_weights=mix_1)
        else:
            # Use uniform mixtures when not conditioning the first layer.
            uniform = torch.full_like(mix_1, 1.0 / float(self.num_templates))
            hidden, spline_1 = self.kan_1(hidden, mixture_weights=uniform)
        hidden = self.dropout(F.gelu(hidden))
        result = {
            "value": self.value_head(fused).squeeze(-1),
            "state_embedding": state_out.pooled,
            "goal_embedding": goal_out.pooled,
            "mixture_weights": [mix_1, mix_2],
        }
        if self.frontier_head is not None:
            result["frontier_logits"] = self.frontier_head(hidden)
        if self.site_op_factorized:
            site_logits, spline_site = self.site_head(hidden, mixture_weights=mix_2)
            op_logits, spline_op = self.op_head(hidden, mixture_weights=mix_2)
            logits = site_logits[:, self.action_to_site_idx] + op_logits[:, self.action_to_op_idx]
            result["logits"] = logits
            result["site_logits"] = site_logits
            result["op_logits"] = op_logits
            result["spline_states"] = [spline_1, spline_site, spline_op]
            return result
        logits, spline_2 = self.kan_2(hidden, mixture_weights=mix_2)
        result["logits"] = logits
        result["spline_states"] = [spline_1, spline_2]
        return result
