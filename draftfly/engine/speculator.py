"""投机解码主循环：草稿连猜 K 个 token，目标一次前向并行验证，接受采样挑选。

参考算法（Leviathan et al. 2022, "Fast Inference from Transformers via
Speculative Decoding"）：
  1. 草稿模型按自回归连猜 K 个 token x_1..x_K（分布 q）
  2. 目标模型在 prefix + x_1..x_K 上单次前向，得到 K+1 个分布 p_0..p_K
  3. 逐位接受：q(x_i) <= p(x_i) 必接受；否则以概率 p(x_i)/q(x_i) 接受
  4. 首个拒绝位 i 处，从残差分布 max(0, p_i - q_i) 归一化采样修正 token
  5. 全部接受则额外从 p_K 采一个 bonus token（不浪费目标最后一次前向）
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..util.timing import Clock
from .acceptance import greedy_accept, reject_sampling, residual_sample
from .kv_cache import KVCache
from .metrics import Metrics
from .sampler import argmax, sample
from .scheduler import AdaptiveK
from .stopper import StopChecker


@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    metrics: Metrics
    iterations: int = 0
    latency_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.metrics.acceptance_rate

    def to_dict(self) -> dict:
        d = self.metrics.to_dict()
        d["text"] = self.text
        d["num_tokens"] = len(self.tokens)
        d["iterations"] = self.iterations
        return d


class Speculator:
    """投机解码引擎。draft 与 target 均为 ModelAdapter。"""

    def __init__(
        self,
        draft,
        target,
        tokenizer,
        *,
        k: int = 4,
        min_k: int = 2,
        max_k: int = 8,
        adaptive: bool = True,
        mode: str = "sampling",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        bonus_token: bool = True,
        seed: int = 0,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.draft = draft
        self.target = target
        self.tokenizer = tokenizer
        self.mode = mode
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.bonus_token = bonus_token
        self.clock = clock
        self.rng = rng if rng is not None else random.Random(seed)
        self.scheduler = AdaptiveK(k, min_k, max_k, adaptive)

    # ------------------------------------------------------------------ #
    def _charge(self, ms: float) -> None:
        if self.clock is not None:
            self.clock.advance_ms(ms)

    def _target_verify(self, prefix, draft_tokens):
        self._charge(self.target.cost_ms)
        return self.target.verify(prefix, draft_tokens)

    def _draft_one(self, ctx):
        self._charge(self.draft.cost_ms)
        return self.draft.sample_with_dist(
            ctx,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            rng=self.rng,
        )

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        k: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        if k is not None:
            self.scheduler.k = k
            self.scheduler.adaptive = False

        stop = StopChecker(self.tokenizer, stop_sequences)
        kv = KVCache()
        metrics = Metrics()
        context = self.tokenizer.encode(prompt)
        tokens_out: list[int] = []

        while len(tokens_out) < max_new_tokens:
            kk = self.scheduler.current()
            prefix = context + tokens_out

            # 1) 草稿连猜 kk 个 token（记录各自分布 q）
            draft_tokens: list[int] = []
            q_list: list[list[float]] = []
            ctx = list(prefix)
            for _ in range(kk):
                if len(tokens_out) + len(draft_tokens) >= max_new_tokens:
                    break
                tok, q = self._draft_one(ctx)
                draft_tokens.append(tok)
                q_list.append(q)
                ctx.append(tok)
                if tok == self.tokenizer.eos_id:
                    break
            if not draft_tokens:
                break
            metrics.draft += len(draft_tokens)
            metrics.draft_calls += len(draft_tokens)

            # 2) 目标一次前向验证（p_0..p_k）
            p_list = self._target_verify(prefix, draft_tokens)
            metrics.target_calls += 1
            kv.grow(prefix, draft_tokens, p_list)

            # 3) 接受采样
            accepted, reject_pos, correction = self._accept(draft_tokens, q_list, p_list)
            tokens_out.extend(accepted)
            metrics.accepted += len(accepted)

            if reject_pos is not None and correction is not None:
                tokens_out.append(correction)
                metrics.corrections += 1
            else:
                # 全部接受：bonus token（目标在位置 kk 的预测）
                if (
                    self.bonus_token
                    and len(draft_tokens) == kk
                    and len(p_list) > len(draft_tokens)
                    and len(tokens_out) < max_new_tokens
                ):
                    p_last = p_list[-1]
                    if self.mode == "greedy":
                        bonus = argmax(p_last)
                    else:
                        bonus = sample(p_last, rng=self.rng)
                    tokens_out.append(bonus)
                    metrics.bonus_tokens += 1

            self.scheduler.update(len(accepted), kk)
            metrics.iterations += 1

            if self.tokenizer.eos_id in tokens_out:
                break
            if stop.reached(tokens_out):
                break

        tokens_out = tokens_out[:max_new_tokens]
        final_tokens = [t for t in tokens_out if t != self.tokenizer.eos_id]
        metrics.latency_ms = self.clock.now() * 1000.0 if self.clock is not None else 0.0
        return GenerationResult(
            text=self.tokenizer.decode(final_tokens),
            tokens=final_tokens,
            metrics=metrics,
            iterations=metrics.iterations,
            latency_ms=metrics.latency_ms,
        )

    def _accept(self, draft_tokens, q_list, p_list):
        """返回 (accepted, reject_pos, correction)。"""
        accepted: list[int] = []
        for i in range(len(draft_tokens)):
            x = draft_tokens[i]
            q = q_list[i]
            p = p_list[i]
            if self.mode == "greedy":
                ok, corr = greedy_accept(x, p)
                if ok:
                    accepted.append(x)
                else:
                    return accepted, i, corr
            else:
                ok, use_residual = reject_sampling(x, q, p, self.rng)
                if ok:
                    accepted.append(x)
                else:
                    if use_residual:
                        corr = residual_sample(p, q, self.rng)
                    else:
                        corr = argmax(p)
                    return accepted, i, corr
        return accepted, None, None
