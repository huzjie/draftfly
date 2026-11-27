"""KV 缓存与 DSA 稀疏注意力索引。

真实 transformer 后端里，目标模型会复用已验证前缀的 KV cache，草稿模型
（DFly2 思路）复用目标的 KV 与稀疏索引，做到「零额外显存」。本模块把这套
「前缀复用 + 回滚 + 稀疏索引」语义做成确定性的数据结构，供上层引擎演示与
记账；接真实模型时由后端在 verify() 内部完成同样的缓存复用。
"""
from __future__ import annotations


class SparseIndex:
    """DSA 稀疏注意力索引：每个 query 只关注少数 key 的掩码。

    真实稀疏注意力只有少量 key 参与打分；这里用确定性规则（局部窗口 +
    稀疏全局采样）生成掩码，供估算注意力计算量与稀疏率。
    """

    def __init__(self, window: int = 8, stride: int = 4) -> None:
        self.window = window
        self.stride = stride
        self.mask: list[list[int]] = []

    def build(self, tokens) -> None:
        n = len(tokens)
        self.mask = []
        for i in range(n):
            keys = set(range(max(0, i - self.window + 1), i + 1))
            keys.update(range(0, i, self.stride))
            self.mask.append(sorted(keys))

    def nnz(self) -> int:
        return sum(len(m) for m in self.mask)

    def density(self, n: int | None = None) -> float:
        n = n if n is not None else len(self.mask)
        if n <= 0:
            return 0.0
        return self.nnz() / (n * n)


class KVCache:
    """已验证前缀与分布的记账缓存，支持前缀复用与回滚。"""

    def __init__(self) -> None:
        self._tokens: list[int] = []
        self._dists: list = []
        self.sparse_index = SparseIndex()

    @property
    def prefix(self) -> list[int]:
        return list(self._tokens)

    @property
    def dists(self) -> list:
        return list(self._dists)

    def grow(self, prefix: list[int], draft_tokens: list[int], p_list: list) -> None:
        """记录一次验证：前缀与草稿 token 序列 + 对应目标分布。"""
        if self._tokens != prefix:
            common = 0
            while (
                common < len(self._tokens)
                and common < len(prefix)
                and self._tokens[common] == prefix[common]
            ):
                common += 1
            del self._tokens[common:]
            del self._dists[common:]
        self._tokens = list(prefix) + list(draft_tokens)
        self._dists = list(p_list)
        self.sparse_index.build(self._tokens)

    def rollback(self, n: int) -> None:
        """回滚到前 n 个 token（丢弃之后的草稿与分布）。"""
        if n < len(self._tokens):
            del self._tokens[n:]
            del self._dists[n:]

    def size(self) -> int:
        return len(self._tokens)
