"""评测：接受率随草稿能力（skill）单调上升 + 贪心输出一致性。

两件可证伪的事：
  1. skill 从 0 扫到 1，接受率单调不减（草稿越贴近目标、猜得越准、接受越多）。
  2. 贪心模式下，投机解码输出与目标模型纯自回归贪心输出逐 token 完全一致
     （这正是投机解码「不损失质量」的硬保证）。
"""
from __future__ import annotations

import random

from .bench import autoregressive
from .engine.speculator import Speculator
from .models.registry import build_mock_with_teacher
from .tokenizer import CharTokenizer


def sweep_acceptance(
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    *,
    skills=(0.0, 0.25, 0.5, 0.75, 1.0),
    order: int = 3,
    seed: int = 0,
    repeats: int = 8,
) -> list[tuple[float, float]]:
    """返回 [(skill, acceptance_rate), ...] 序列（应单调不减）。

    接受率在期望上等于 1 - TV(p, q)，随 skill 单调；但单次短样本有采样噪声，
    故按多个种子取平均，得到稳定、单调的曲线。
    """
    rates: list[tuple[float, float]] = []
    for s in skills:
        draft = build_mock_with_teacher(order, s, tokenizer)
        target = build_mock_with_teacher(order, 1.0, tokenizer)
        acc = 0.0
        for r in range(repeats):
            spec = Speculator(
                draft, target, tokenizer, k=4, mode="sampling",
                temperature=1.0, seed=seed + r, rng=random.Random(seed + r),
            )
            res = spec.generate(prompt, max_new_tokens=max_new_tokens)
            acc += res.acceptance_rate
        rates.append((s, round(acc / repeats, 4)))
    return rates


def assert_greedy_consistency(tokenizer, target, draft, prompt, max_new_tokens=48, seed=0):
    """贪心投机输出 == 贪心自回归输出。返回 bool。"""
    spec = Speculator(
        draft, target, tokenizer, k=4, mode="greedy",
        temperature=1.0, seed=seed, rng=random.Random(seed),
    )
    spec_text = spec.generate(prompt, max_new_tokens=max_new_tokens).text
    base_text, _ = autoregressive(
        target, tokenizer, prompt, max_new_tokens, mode="greedy",
        temperature=1.0, rng=random.Random(seed),
    )
    return spec_text == base_text


def _is_monotonic(vals, tol=0.05) -> bool:
    """容噪单调判定 + 强端点约束（skill=1.0 精确接受 1.0 且远高于 skill=0）。"""
    if abs(vals[-1] - 1.0) > 1e-4:
        return False
    if vals[-1] <= vals[0] + 0.15:
        return False
    return all(vals[i] <= vals[i + 1] + tol for i in range(len(vals) - 1))


def run_eval(tokenizer, prompt, max_new_tokens=256, seed=0) -> dict:
    rates = sweep_acceptance(tokenizer, prompt, max_new_tokens, seed=seed)
    vals = [r for _, r in rates]
    monotonic = _is_monotonic(vals)
    return {
        "rates": [{"skill": s, "acceptance_rate": r} for s, r in rates],
        "monotonic": monotonic,
        "prompt": prompt,
    }
