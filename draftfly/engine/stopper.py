"""停止条件：EOS / 最大长度 / 自定义停止序列。"""
from __future__ import annotations


class StopChecker:
    def __init__(self, tokenizer, stop_sequences=None) -> None:
        self.tokenizer = tokenizer
        self.stop_seqs = [
            tokenizer.encode(s) for s in (stop_sequences or []) if s
        ]

    def reached(self, tokens: list[int]) -> bool:
        if self.tokenizer.eos_id in tokens:
            return True
        for seq in self.stop_seqs:
            if seq and len(tokens) >= len(seq) and tokens[-len(seq):] == seq:
                return True
        return False
