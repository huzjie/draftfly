# draftfly 文档

投机解码（Speculative Decoding）加速引擎的完整文档。

## 目录

| 分类 | 文档 |
|---|---|
| 入门 | [快速开始](quickstart.md) · [安装](installation.md) · [配置](configuration.md) |
| 原理 | [投机解码算法](algorithms.md) · [接受采样](acceptance-sampling.md) · [token 树](token-tree.md) · [自适应 K](adaptive-k.md) · [KV 缓存](kv-cache.md) · [稀疏注意力](sparse-attention.md) |
| 使用 | [CLI](cli.md) · [HTTP API](http-api.md) · [模型](models.md) · [草稿模型选型](draft-model-guide.md) · [OpenAI 后端](openai-backend.md) |
| 评测 | [基准](benchmarking.md) · [评测](evaluation.md) |
| 部署 | [Docker](docker.md) · [Kubernetes](kubernetes.md) · [Helm](helm.md) · [CI/CD](ci-cd.md) |
| 参考 | [架构](architecture.md) · [设计决策](design-decisions.md) · [词汇表](glossary.md) · [FAQ](faq.md) · [排障](troubleshooting.md) · [路线图](roadmap.md) |

## 一句话

投机解码让一个小而快的「草稿模型」连猜 K 个 token，再由大模型单次前向并行验证、按接受采样挑出对的，把串行逐 token 生成变成「猜 K 个、验一次」，在输出与纯自回归完全一致的前提下把吞吐拉高数倍。
