"""带 skill 的确定性 mock 模型：离线验证投机解码全链路。

skill ∈ [0,1]：把自身分布向「教师模型」分布做凸组合（skill 越大越贴近教师）。
skill→1 时草稿预测与目标高度一致，接受率单调上升——这是评测里「草稿能力越强
接受率越高」的可测量、可复现基线。
"""
from __future__ import annotations

from .base import ModelAdapter
from .ngram import NGramModel


class MockModel(ModelAdapter):
    def __init__(
        self,
        ngram: NGramModel,
        *,
        skill: float = 1.0,
        cost_ms: float = 0.0,
        name: str = "mock",
        teacher: ModelAdapter | None = None,
    ) -> None:
        super().__init__(name=name, cost_ms=cost_ms)
        self.ngram = ngram
        self.skill = skill
        self.teacher = teacher

    def logits(self, context: list[int]) -> list[float]:
        own = self.ngram.logits(context)
        if self.teacher is None:
            return own
        if self.skill >= 1.0:
            return self.teacher.logits(context)
        t = self.teacher.logits(context)
        s = self.skill
        return [s * t[i] + (1 - s) * own[i] for i in range(len(own))]
