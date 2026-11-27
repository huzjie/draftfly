# CLI

```bash
python -m draftfly doctor                 # 环境自检
python -m draftfly generate [prompt]      # 生成文本
python -m draftfly bench [prompt]         # 吞吐对比
python -m draftfly eval [prompt]          # 接受率单调性评测
python -m draftfly serve                  # 起 HTTP 服务
python -m draftfly config                 # 打印解析后的配置
python -m draftfly --version              # 版本
```

常用参数：`--config 文件`、`--max-tokens N`、`--k N`、`--mode sampling|greedy`。
