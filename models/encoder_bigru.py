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


def fuse_state_goal(state_vec: Tensor, goal_vec: Tensor) -> Tensor:
    return torch.cat([state_vec, goal_vec, state_vec - goal_vec, state_vec * goal_vec], dim=-1)

