"""自适应草稿长度 K 调度器。"""
from __future__ import annotations


class AdaptiveK:
    """接受率高就加长草稿（多猜多赚），低就缩短（少浪费草稿算力）。"""

    def __init__(
        self,
        k: int = 4,
        min_k: int = 2,
        max_k: int = 8,
        adaptive: bool = True,
        window: int = 8,
        high: float = 0.8,
        low: float = 0.5,
    ) -> None:
        self.k = k
        self.min_k = min_k
        self.max_k = max_k
        self.adaptive = adaptive
        self.window = window
        self.high = high
        self.low = low
        self._history: list[tuple[int, int]] = []

    def current(self) -> int:
        return self.k

    def update(self, accepted: int, total: int) -> None:
        if not self.adaptive or total <= 0:
            return
        self._history.append((accepted, total))
        if len(self._history) > self.window:
            self._history.pop(0)
        acc = sum(a for a, _ in self._history)
        tot = sum(t for _, t in self._history)
        if tot == 0:
            return
        rate = acc / tot
        if rate >= self.high and self.k < self.max_k:
            self.k = min(self.max_k, self.k + 1)
        elif rate < self.low and self.k > self.min_k:
            self.k = max(self.min_k, self.k - 1)
