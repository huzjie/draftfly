# 模型

draftfly 的模型层统一接口见 `draftfly/models/base.py`（`ModelAdapter`）：

- `logits(context) -> list[float]`：返回归一化概率分布；
- `verify(prefix, draft_tokens)`：一次前向验证整条草稿，返回 K+1 个分布；
- `sample_with_dist(context, ...)`：采样并返回「实际采样所用分布」q。

内置模型：

| 模型 | 说明 |
|---|---|
| `NGramModel` | 确定性字符级 n-gram（backoff + add-k），order 越高越准 |
| `MockModel` | 带 skill 的确定性 mock，可把自身分布向教师模型做凸组合 |
| `MockTextModel` | 文本级 mock，供 TextSpeculator 演示 |
| `OpenAICompatModel` | OpenAI 兼容端点适配器，面向文本完成 |

接入真实模型：实现 `ModelAdapter` 或使用 OpenAI 兼容端点。
