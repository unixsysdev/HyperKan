from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class EncoderOutput:
    pooled: Tensor
    sequence: Tensor
    mask: Tensor


class SharedBiGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, token_ids: Tensor, lengths: Tensor) -> EncoderOutput:
        embedded = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.gru(packed)
        sequence, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=token_ids.size(1))
        mask = token_ids.ne(self.pad_id)
        hidden = hidden.view(self.gru.num_layers, 2, token_ids.size(0), self.gru.hidden_size)
        last_layer = hidden[-1]
        pooled = torch.cat([last_layer[0], last_layer[1]], dim=-1)
        return EncoderOutput(pooled=pooled, sequence=sequence, mask=mask)


class SharedTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        ff_dim: int | None = None,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.pad_id = pad_id
        self.max_positions = int(max_positions)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.input_projection = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        self.position_embedding = nn.Embedding(self.max_positions, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim if ff_dim is not None else hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> EncoderOutput:
        del lengths
        seq_len = token_ids.size(1)
        if seq_len > self.max_positions:
            raise ValueError(f"sequence length {seq_len} exceeds max_positions {self.max_positions}")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand_as(token_ids)
        embedded = self.input_projection(self.embedding(token_ids))
        embedded = self.dropout(embedded + self.position_embedding(positions))
        mask = token_ids.ne(self.pad_id)
        sequence = self.encoder(embedded, src_key_padding_mask=~mask)
        sequence = self.output_norm(sequence)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        pooled = (sequence * mask.unsqueeze(-1).to(sequence.dtype)).sum(dim=1) / denom
        return EncoderOutput(pooled=pooled, sequence=sequence, mask=mask)


def create_shared_encoder(
    encoder_type: str,
    vocab_size: int,
    pad_id: int,
    embed_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    transformer_heads: int = 4,
    transformer_ff_dim: int | None = None,
    transformer_max_positions: int = 512,
) -> nn.Module:
    if encoder_type == "bigru":
        return SharedBiGRUEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
    if encoder_type == "transformer":
        return SharedTransformerEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            num_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
            max_positions=transformer_max_positions,
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def fuse_state_goal(state_vec: Tensor, goal_vec: Tensor) -> Tensor:
    return torch.cat([state_vec, goal_vec, state_vec - goal_vec, state_vec * goal_vec], dim=-1)
