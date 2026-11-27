"""draftfly — 投机解码（Speculative Decoding）加速引擎。

灵感来自腾讯混元 DFly2「草稿模型」：用一个又小又快的草稿模型（draft）
一口气连猜 K 个 token，再让大模型（target）单次前向并行验证整条草稿，
用接受采样（rejection sampling）挑出可接受的 token，把「串行逐 token 生成」
变成「猜 K 个、验一次」，在输出与纯自回归保持一致的前提下把吞吐拉高数倍。

零依赖纯 Python 内核：字符级 tokenizer + 确定性 n-gram 草稿/目标模型 +
接受采样 + 自适应 K + KV 缓存复用 + DSA 稀疏索引 + 树式多候选验证 +
OpenAI 兼容 HTTP 控制面。

用法：
    python -m draftfly doctor     # 环境自检
    python -m draftfly generate   # 投机解码生成
    python -m draftfly bench      # 投机 vs 自回归吞吐对比
    python -m draftfly eval       # 接受率随草稿能力单调性评测
    python -m draftfly serve      # 起 OpenAI 兼容 HTTP 服务
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
