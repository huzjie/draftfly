"""采样器：温度 / top-k / top-p 过滤 + 多项式采样 + argmax。

输入一律为「归一化概率分布」（模型 logits 的返回值已归一化）。
"""
from __future__ import annotations

import random


def filter_probs(probs, *, temperature=1.0, top_k=0, top_p=1.0):
    """按温度、top-k、top-p 过滤归一化概率分布，返回新的归一化分布。"""
    probs = list(probs)
    t = max(temperature, 1e-9)
    if t != 1.0:
        # temperature：p^(1/t) 再归一化（T<1 锐化、T>1 平坦）
        probs = [p ** (1.0 / t) for p in probs]
        s = sum(probs) or 1.0
        probs = [p / s for p in probs]
    probs = _top_k(probs, top_k)
    probs = _top_p(probs, top_p)
    return probs


def _top_k(probs, k):
    if k <= 0 or k >= len(probs):
        return probs
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    keep = set(order)
    filtered = [p if i in keep else 0.0 for i, p in enumerate(probs)]
    s = sum(filtered)
    return [p / s for p in filtered] if s > 0 else probs


def _top_p(probs, p):
    if p >= 1.0:
        return probs
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cum = 0.0
    keep = set()
    for i in order:
        keep.add(i)
        cum += probs[i]
        if cum >= p:
            break
    filtered = [pp if i in keep else 0.0 for i, pp in enumerate(probs)]
    s = sum(filtered)
    return [pp / s for pp in filtered] if s > 0 else probs


def sample(probs, *, rng=None):
    """从归一化分布做多项式采样，返回索引。"""
    rng = rng if rng is not None else random
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1


def argmax(probs):
    """取概率最大的 token 索引。"""
    return max(range(len(probs)), key=lambda i: probs[i])
