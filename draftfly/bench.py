"""基准：投机解码 vs 纯自回归的吞吐/延迟/加速比对比。"""
from __future__ import annotations

import random

from .engine.speculator import Speculator
from .util.timing import Clock


def autoregressive(
    target,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    *,
    mode: str = "sampling",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    clock: Clock | None = None,
    rng: random.Random | None = None,
) -> tuple[str, float]:
    """纯自回归基线：每个 token 都付一次 target 前向。"""
    from .engine.sampler import argmax

    ctx = tokenizer.encode(prompt)
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        if clock is not None:
            clock.advance_ms(target.cost_ms)
        if mode == "greedy":
            tok = argmax(target.logits(ctx))
        else:
            tok, _ = target.sample_with_dist(
                ctx, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng
            )
        tokens.append(tok)
        ctx.append(tok)
        if tok == tokenizer.eos_id:
            break
    out = [t for t in tokens if t != tokenizer.eos_id]
    return tokenizer.decode(out), (clock.now() * 1000.0 if clock else 0.0)


def run_bench(
    draft,
    target,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    *,
    k: int = 4,
    mode: str = "sampling",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    real_clock: bool = False,
) -> dict:
    """跑一组对比，返回可读的指标 dict。"""
    # 基线
    clock_a = Clock(real=real_clock)
    rng_a = random.Random(seed)
    base_text, base_ms = autoregressive(
        target, tokenizer, prompt, max_new_tokens,
        mode=mode, temperature=temperature, top_k=top_k, top_p=top_p,
        clock=clock_a, rng=rng_a,
    )
    base_tokens = len(tokenizer.encode(base_text)) or 1

    # 投机
    clock_s = Clock(real=real_clock)
    rng_s = random.Random(seed)
    spec = Speculator(
        draft, target, tokenizer,
        k=k, mode=mode, temperature=temperature, top_k=top_k, top_p=top_p,
        seed=seed, clock=clock_s, rng=rng_s,
    )
    res = spec.generate(prompt, max_new_tokens=max_new_tokens)
    spec_ms = res.latency_ms
    spec_tokens = len(res.tokens) or 1

    base_tps = base_tokens / (base_ms / 1000.0) if base_ms > 0 else float("inf")
    spec_tps = spec_tokens / (spec_ms / 1000.0) if spec_ms > 0 else float("inf")
    speedup = (base_ms / spec_ms) if spec_ms > 0 else float("inf")

    return {
        "prompt": prompt,
        "k": k,
        "mode": mode,
        "baseline_ms": round(base_ms, 3),
        "speculative_ms": round(spec_ms, 3),
        "baseline_tokens": base_tokens,
        "speculative_tokens": spec_tokens,
        "baseline_tps": round(base_tps, 2),
        "speculative_tps": round(spec_tps, 2),
        "speedup": round(speedup, 2),
        "acceptance_rate": res.acceptance_rate,
        "output": res.text,
    }
