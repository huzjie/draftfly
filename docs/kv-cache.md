# KV 缓存复用

投机解码里，目标模型验证时前缀（已确认的 token）不需要重新计算——复用 KV cache
即可，只对草稿新增的 K 个位置做增量前向。

draftfly 的 `KVCache` 把这个语义做成确定性数据结构：

- `grow(prefix, draft_tokens, p_list)`：记录一次验证，前缀不一致自动回滚到公共前缀；
- `rollback(n)`：回滚到前 n 个 token；
- `sparse_index`：顺带维护 DSA 稀疏注意力掩码。

真实 transformer 后端在 `verify()` 内部完成同样的缓存复用。
