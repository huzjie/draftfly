"""文本级确定性 mock：供 TextSpeculator 演示「双端点投机解码」用。"""
from __future__ import annotations

import random

from ..util.rng import make_rng


class MockTextModel:
    """用「下一步词表」确定性生成文本的轻量 mock。

    与字符级 n-gram 不同，本模型面向「按词生成的文本完成」场景，
    用于演示把草稿/目标两个文本完成端点接到一起做投机解码。
    """

    def __init__(self, vocab: list[str], seed: int = 0, name: str = "mock_text") -> None:
        self.vocab = vocab
        self.name = name
        self.rng = make_rng(name, seed)

    def complete(self, prompt: str, max_tokens: int = 16) -> str:
        words = prompt.split()
        out = list(words)
        for _ in range(max_tokens):
            if not out:
                break
            nxt = self.rng.choice(self.vocab)
            out.append(nxt)
            if nxt.endswith(".") and len(out) > 1:
                break
        return " ".join(out[len(words):])
