# 更新日志

## [1.0.0] - 2026-11-27

### 新增
- 标准投机解码主循环（草稿链 + 接受采样 + bonus token）
- 自适应草稿长度 K 调度
- KV 缓存复用 + DSA 稀疏注意力索引
- token 树多候选验证（SpecInfer / Medusa 思路）
- 文本级投机解码（双 OpenAI 兼容端点）
- OpenAI 兼容 HTTP 控制面（/v1/completions、/v1/chat/completions）
- 确定性可复现评测（接受率单调性 + 贪心一致性）
- 零依赖纯 Python 内核，47 个单元测试
- Docker / docker-compose / Kubernetes / Helm / CI 部署
