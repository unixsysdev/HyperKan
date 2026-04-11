from __future__ import annotations

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
    ) -> None:
        super().__init__()
        self.num_templates = num_templates
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
        self.hyper = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_templates * 2),
        )
        self.kan_1 = SimpleKANLayer(kan_hidden_dim, kan_hidden_dim, basis_dim=basis_dim, num_templates=num_templates)
        self.kan_2 = SimpleKANLayer(kan_hidden_dim, num_actions, basis_dim=basis_dim, num_templates=num_templates)
        self.value_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def _mixtures(self, goal_embedding: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.hyper(goal_embedding)
        part_a, part_b = raw.chunk(2, dim=-1)
        return part_a.softmax(dim=-1), part_b.softmax(dim=-1)

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
        hidden, spline_1 = self.kan_1(hidden, mixture_weights=mix_1)
        hidden = self.dropout(F.gelu(hidden))
        logits, spline_2 = self.kan_2(hidden, mixture_weights=mix_2)
        return {
            "logits": logits,
            "value": self.value_head(fused).squeeze(-1),
            "state_embedding": state_out.pooled,
            "goal_embedding": goal_out.pooled,
            "spline_states": [spline_1, spline_2],
        }
