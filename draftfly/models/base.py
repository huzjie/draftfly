"""模型适配器抽象：草稿模型与目标模型共用同一接口。

约定：logits() 返回归一化概率分布（长度 = vocab_size，已归一化、可直接采样）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..engine.sampler import filter_probs, sample


class ModelAdapter(ABC):
    """统一模型接口。"""

    name: str = "base"
    cost_ms: float = 0.0  # 单次前向延迟（用于吞吐/加速比建模）

    def __init__(self, name: str = "base", cost_ms: float = 0.0) -> None:
        self.name = name
        self.cost_ms = cost_ms

    @abstractmethod
    def logits(self, context: list[int]) -> list[float]:
        """给定上下文 token id 序列，返回下一 token 的概率分布（已归一化）。"""

    def verify(self, prefix: list[int], draft_tokens: list[int]) -> list[list[float]]:
        """一次前向验证整条草稿，返回 p_0..p_k（共 k+1 个分布）。

        p_i = 在 prefix + draft_tokens[:i] 条件下预测 draft_tokens[i] 的分布。
        真实后端（vLLM/SGLang）在单次 decode 里并行得到这些 logits；
        本实现按语义逐位置计算，但延迟账目上只记一次 target 前向。
        """
        return [
            self.logits(prefix + draft_tokens[:i])
            for i in range(len(draft_tokens) + 1)
        ]

    def sample_with_dist(
        self,
        context: list[int],
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        rng=None,
    ) -> tuple[int, list[float]]:
        """采样一个 token，并返回「实际采样所用的过滤后分布」q。

        q 用于接受采样，必须与采样来源严格一致（含温度/top-k/top-p）。
        """
        probs = self.logits(context)
        filtered = filter_probs(probs, temperature=temperature, top_k=top_k, top_p=top_p)
        tok = sample(filtered, rng=rng)
        return tok, filtered
