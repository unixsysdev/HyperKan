from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.encoder_bigru import create_shared_encoder, fuse_state_goal


@dataclass(frozen=True)
class KANSplineState:
    mixture_weights: Tensor
    basis_activations: Tensor


class SimpleKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        basis_dim: int,
        num_templates: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.basis_dim = basis_dim
        self.num_templates = num_templates

        self.base_linear = nn.Linear(input_dim, output_dim)
        self.basis_linear = nn.Linear(input_dim, output_dim * basis_dim)
        self.template_logits = nn.Parameter(torch.randn(num_templates, output_dim, basis_dim) * 0.02)
        self.output_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs: Tensor, mixture_weights: Tensor | None = None) -> tuple[Tensor, KANSplineState]:
        batch_size = inputs.size(0)
        basis = self.basis_linear(inputs).view(batch_size, self.output_dim, self.basis_dim)
        basis = torch.tanh(basis)

        if mixture_weights is None:
            mixture_weights = torch.full(
                (batch_size, self.num_templates),
                1.0 / self.num_templates,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        templates = torch.einsum("bt,tof->bof", mixture_weights, self.template_logits)
        spline = (basis * templates).sum(dim=-1)
        outputs = self.base_linear(inputs) + spline + self.output_bias
        return outputs, KANSplineState(mixture_weights=mixture_weights, basis_activations=basis)


class StaticKANPolicy(nn.Module):
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
        use_frontier_head: bool = False,
        encoder_type: str = "bigru",
        transformer_heads: int = 4,
        transformer_ff_dim: int | None = None,
        transformer_max_positions: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = create_shared_encoder(
            encoder_type=encoder_type,
            vocab_size=vocab_size,
            pad_id=pad_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            dropout=dropout,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            transformer_max_positions=transformer_max_positions,
        )
        fused_dim = hidden_dim * 4
        self.pre = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, kan_hidden_dim), nn.GELU())
        self.kan_1 = SimpleKANLayer(kan_hidden_dim, kan_hidden_dim, basis_dim=basis_dim, num_templates=num_templates)
        self.kan_2 = SimpleKANLayer(kan_hidden_dim, num_actions, basis_dim=basis_dim, num_templates=num_templates)
        self.frontier_head = (
            nn.Sequential(
                nn.LayerNorm(kan_hidden_dim),
                nn.Linear(kan_hidden_dim, kan_hidden_dim),
                nn.GELU(),
                nn.Linear(kan_hidden_dim, num_actions),
            )
            if use_frontier_head
            else None
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

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
        hidden, spline_1 = self.kan_1(hidden)
        hidden = self.dropout(F.gelu(hidden))
        logits, spline_2 = self.kan_2(hidden)
        result = {
            "logits": logits,
            "value": self.value_head(fused).squeeze(-1),
            "state_embedding": state_out.pooled,
            "goal_embedding": goal_out.pooled,
            "spline_states": [spline_1, spline_2],
        }
        if self.frontier_head is not None:
            result["frontier_logits"] = self.frontier_head(hidden)
        return result
