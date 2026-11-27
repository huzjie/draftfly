"""确定性 n-gram 字符级语言模型（backoff + add-k 平滑）。

order 越高越「准」（作目标模型），越低越快（作草稿模型）。
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .base import ModelAdapter


class NGramModel(ModelAdapter):
    def __init__(
        self,
        tokenizer,
        corpus: str,
        order: int,
        alpha: float = 0.01,
        name: str = "ngram",
        cost_ms: float = 0.0,
    ) -> None:
        super().__init__(name=name, cost_ms=cost_ms)
        self.tokenizer = tokenizer
        self.order = max(1, int(order))
        self.alpha = alpha
        self._tables: dict[int, dict] = {}
        self._build(corpus)

    def _build(self, corpus: str) -> None:
        ids = (
            [self.tokenizer.bos_id]
            + self.tokenizer.encode(corpus)
            + [self.tokenizer.eos_id]
        )
        for n in range(1, self.order + 1):
            table = defaultdict(Counter)
            for i in range(len(ids) - n + 1):
                hist = tuple(ids[i : i + n - 1])
                table[hist][ids[i + n - 1]] += 1
            self._tables[n] = table

    def logits(self, context: list[int]) -> list[float]:
        v = self.tokenizer.vocab_size
        ctx = list(context)
        probs: list[float]
        for n in range(self.order, 0, -1):
            hist = tuple(ctx[-(n - 1):]) if n > 1 else ()
            table = self._tables[n].get(hist)
            if table:
                total = sum(table.values())
                denom = total + self.alpha * v
                probs = [
                    (table.get(t, 0) + self.alpha) / denom for t in range(v)
                ]
                return self._mask_controls(probs)
        return self._mask_controls([1.0 / v] * v)

    def _mask_controls(self, probs: list[float]) -> list[float]:
        """屏蔽控制 token（pad/bos/unk 永不作为续写），保留 eos 作停止信号。"""
        for t in (
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            self.tokenizer.unk_id,
        ):
            probs[t] = 0.0
        s = sum(probs)
        if s > 0:
            return [p / s for p in probs]
        return probs
