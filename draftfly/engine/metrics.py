"""吞吐/接受率指标记账。"""
from __future__ import annotations


class Metrics:
    def __init__(self) -> None:
        self.iterations = 0
        self.accepted = 0
        self.draft = 0
        self.target_calls = 0
        self.draft_calls = 0
        self.latency_ms = 0.0
        self.bonus_tokens = 0
        self.corrections = 0

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.draft) if self.draft else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return (self.accepted + self.bonus_tokens + self.corrections) / (
            self.latency_ms / 1000.0
        ) if self.latency_ms > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "accepted_draft": self.accepted,
            "total_draft": self.draft,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "target_calls": self.target_calls,
            "draft_calls": self.draft_calls,
            "bonus_tokens": self.bonus_tokens,
            "corrections": self.corrections,
            "latency_ms": round(self.latency_ms, 3),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
        }
