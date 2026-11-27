"""确定性随机数：以 md5 派生种子，任意两次运行逐位可复现。"""
from __future__ import annotations

import hashlib
import random


def derive_seed(key: str, salt: int = 0) -> int:
    """由字符串 key 派生一个 32 位种子（md5 前 8 字节）。"""
    digest = hashlib.md5((key + "::" + str(salt)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_rng(key: str, salt: int = 0) -> random.Random:
    """构造一个可复现的 random.Random 实例。"""
    return random.Random(derive_seed(key, salt))
