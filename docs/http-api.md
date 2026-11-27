# HTTP API

`python -m draftfly serve` 启动 OpenAI 兼容控制面（标准库 http.server，零依赖）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活探针 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/completions` | 文本补全（投机解码） |
| POST | `/v1/chat/completions` | 对话补全（OpenAI 兼容） |
| GET | `/metrics` | 最近一次生成指标快照 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Speculative decoding", "max_tokens": 32}'
```
