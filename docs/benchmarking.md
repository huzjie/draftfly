# 基准

`python -m draftfly bench "prompt" --k 4 --max-tokens 64` 输出：

- `baseline_ms` / `speculative_ms`：自回归 / 投机总延迟；
- `baseline_tps` / `speculative_tps`：吞吐（token/s）；
- `speedup`：加速比；
- `acceptance_rate`：草稿 token 接受率。

延迟模型：`cost_ms` 是每个模型单次前向的「虚拟耗时」，`real_clock: true` 时用
真实 sleep 演示。默认用纯计数器（确定性、秒回），适合进 CI。
