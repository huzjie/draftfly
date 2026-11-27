# 排障

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `配置错误` | 非法 mode / min_k>max_k | 检查 config，看报错提示 |
| 接受率曲线抖动 | 采样噪声（样本太少） | 加大 max_new_tokens / repeats |
| 输出含特殊 token | （已修复）控制 token 屏蔽 | 升级到最新代码 |
| 加速比 = 1 | 草稿与目标耗时接近或接受率极低 | 换更快的草稿 / 提高 order |
| 服务 404 | 路径不对 | 用 /health、/v1/completions |
