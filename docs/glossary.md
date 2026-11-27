# 词汇表

| 术语 | 含义 |
|---|---|
| 投机解码 | Speculative Decoding，草稿猜 K 个、目标验一次 |
| 草稿模型 | Draft model，小而快的猜词模型 |
| 目标模型 | Target model，大而准的验证模型 |
| 接受采样 | Rejection sampling，按 p/q 概率接受草稿 token |
| 残差采样 | 从 max(0, p-q) 采样修正 token |
| 自适应 K | 按接受率动态调整草稿长度 |
| KV 缓存 | Key-Value cache，前缀复用 |
| DSA | 稀疏注意力，只让少数 key 参与打分 |
| 总变差距离 | TV(p,q)，衡量两个分布的距离 |
| bonus token | 全接受时目标在 K+1 位的额外预测 |
