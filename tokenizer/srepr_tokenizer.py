from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data_gen.canonicalize import structural_string


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|[-+*/^(),]")


@dataclass(frozen=True)
class EncodedPair:
    state_ids: list[int]
    goal_ids: list[int]
    state_length: int
    goal_length: int


class SReprTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is None:
            token_to_id = {
                self.PAD: 0,
                self.UNK: 1,
                self.BOS: 2,
                self.EOS: 3,
            }
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize_expression(self, expression: str) -> list[str]:
        srepr = structural_string(expression)
        tokens = TOKEN_PATTERN.findall(srepr)
        return [self.BOS, *tokens, self.EOS]

    def encode_expression(self, expression: str, max_length: int) -> tuple[list[int], int]:
        tokens = self.tokenize_expression(expression)
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens[:max_length]]
        length = len(ids)
        if length < max_length:
            ids.extend([self.pad_id] * (max_length - length))
        return ids, length

    def encode_pair(self, state: str, goal: str, max_length: int) -> EncodedPair:
        state_ids, state_length = self.encode_expression(state, max_length)
        goal_ids, goal_length = self.encode_expression(goal, max_length)
        return EncodedPair(
            state_ids=state_ids,
            goal_ids=goal_ids,
            state_length=state_length,
            goal_length=goal_length,
        )

    def decode(self, ids: list[int]) -> list[str]:
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.UNK)
            if token == self.PAD:
                continue
            tokens.append(token)
        return tokens

    def build_vocab(self, expressions: Iterable[str], min_freq: int = 1) -> None:
        counts: dict[str, int] = {}
        for expression in expressions:
            for token in self.tokenize_expression(expression):
                counts[token] = counts.get(token, 0) + 1
        for token in sorted(counts):
            if counts[token] < min_freq or token in self.token_to_id:
                continue
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SReprTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=payload)

