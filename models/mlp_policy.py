from __future__ import annotations

import torch
from torch import Tensor, nn

from models.encoder_bigru import create_shared_encoder, fuse_state_goal


class MLPPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_actions: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        dropout: float = 0.1,
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
        self.policy_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_ids: Tensor,
        state_lengths: Tensor,
        goal_ids: Tensor,
        goal_lengths: Tensor,
    ) -> dict[str, Tensor]:
        state_out = self.encoder(state_ids, state_lengths)
        goal_out = self.encoder(goal_ids, goal_lengths)
        fused = fuse_state_goal(state_out.pooled, goal_out.pooled)
        return {
            "logits": self.policy_head(fused),
            "value": self.value_head(fused).squeeze(-1),
            "state_embedding": state_out.pooled,
            "goal_embedding": goal_out.pooled,
        }
