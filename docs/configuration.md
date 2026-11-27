# 配置

配置文件为 YAML（内置极简解析器，无需 PyYAML），示例见 `config.example.yaml`。

```yaml
draft:
  model: ngram      # 草稿模型：ngram
  order: 3          # n-gram 阶数（越低越快、越不准）
  cost_ms: 2.0      # 草稿单次前向延迟（建模用）
target:
  model: ngram      # 目标模型：ngram
  order: 6          # 目标阶数（越高越准、越慢）
  cost_ms: 50.0     # 目标单次前向延迟（建模用）
engine:
  k: 4              # 初始草稿长度
  min_k: 2
  max_k: 8
  adaptive: true    # 自适应 K
  mode: sampling    # sampling | greedy
  temperature: 1.0
  top_k: 0          # 0 = 不启用
  top_p: 1.0        # 1.0 = 不启用
  bonus_token: true
  max_new_tokens: 128
  seed: 42
  real_clock: false # true 用真实 sleep 演示延迟
server:
  host: 127.0.0.1
  port: 8000
```

## 调参口诀

- 草稿越弱（order 越小）→ 接受率越低、加速比越小，但草稿成本也越低。
- `k` 越大 → 单次目标前向摊薄越多 token，但草稿浪费越多；自适应 K 会自动找平衡点。
- 先保质量再抠速度：用 `mode: greedy` 保证输出一致，再逐步加大 `k` 观察加速比。
