"""接受采样（rejection sampling）与残差采样。

Leviathan et al. 2022 的核心判定：
  - 若草稿分布 q 在某 token 上的概率不高于目标分布 p（q(x) <= p(x)），必接受；
  - 否则以概率 p(x)/q(x) 接受；
  - 拒绝时从残差分布 max(0, p - q) 归一化采样一个修正 token。
"""
from __future__ import annotations

from .sampler import argmax, sample


def greedy_accept(draft_token: int, p) -> tuple[bool, int | None]:
    """贪心模式：草稿 token 等于目标 argmax 才接受。返回 (accepted, correction)。"""
    top = argmax(p)
    if draft_token == top:
        return True, None
    return False, top


def reject_sampling(x: int, q, p, rng) -> tuple[bool, bool]:
    """单步接受判定。返回 (accepted, use_residual)。"""
    qx = q[x] if 0 <= x < len(q) else 0.0
    px = p[x] if 0 <= x < len(p) else 0.0
    if qx <= px:
        return True, False
    if px <= 0.0:
        return False, False
    ratio = (px / qx) if qx > 0 else 0.0
    if rng.random() < ratio:
        return True, False
    return False, True


def residual_sample(p, q, rng) -> int:
    """从残差分布 max(0, p - q) 归一化采样修正 token。"""
    r = [max(0.0, p[i] - q[i]) for i in range(len(p))]
    s = sum(r)
    if s <= 0.0:
        return argmax(p)
    norm = [v / s for v in r]
    return sample(norm, rng=rng)
