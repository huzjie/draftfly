"""延迟时钟：real=True 走墙钟，否则纯计数器（确定性、可测试）。"""
from __future__ import annotations

import time


class Clock:
    """统一计账「模型前向耗时」。

    - real=True：真正 sleep，用于 serve/bench 的端到端演示。
    - real=False：累加一个浮点秒数，不 sleep，测试/评测确定性、秒回。
    """

    def __init__(self, real: bool = False) -> None:
        self.real = real
        self._t = 0.0

    def now(self) -> float:
        return time.time() if self.real else self._t

    def advance(self, seconds: float) -> None:
        if self.real:
            time.sleep(seconds)
        self._t += seconds

    def advance_ms(self, ms: float) -> None:
        self.advance(ms / 1000.0)
