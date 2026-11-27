"""极简日志：标准输出前缀，可静默。"""
from __future__ import annotations

import sys


class Logger:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def info(self, msg: str) -> None:
        if not self.quiet:
            print("[draftfly] " + msg)

    def warn(self, msg: str) -> None:
        if not self.quiet:
            print("[draftfly][warn] " + msg, file=sys.stderr)

    def error(self, msg: str) -> None:
        print("[draftfly][error] " + msg, file=sys.stderr)
