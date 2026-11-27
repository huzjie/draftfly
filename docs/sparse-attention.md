# DSA 稀疏注意力

DFly2 面向 DSA（稀疏注意力）大模型与长上下文场景：草稿复用目标的 KV cache 与
稀疏索引，做到「计算开销固定、零额外显存」。

draftfly 的 `SparseIndex` 用确定性规则生成稀疏掩码（局部窗口 + 稀疏全局采样），
用于估算注意力计算量与稀疏率（`nnz` / `density`）。真实 DSA 里只有少数 key 参与
打分，稀疏索引决定每个 query 关注哪些 key。
