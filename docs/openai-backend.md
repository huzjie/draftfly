# 接入 OpenAI 兼容端点

把草稿与目标各指向一个 OpenAI 兼容 `/v1/chat/completions` 端点（如 vLLM/SGLang
跑的 DFly2 草稿 + 大模型），用 `TextSpeculator` 做文本级投机解码：

```python
from draftfly.engine.text_speculator import TextSpeculator
from draftfly.models.openai_compat import OpenAICompatModel

draft = OpenAICompatModel(base_url="http://host1:8001", model="draft-model")
target = OpenAICompatModel(base_url="http://host2:8002", model="target-model")
spec = TextSpeculator(draft, target, k=4, max_new_tokens=32)
print(spec.generate("Explain speculative decoding"))
```

文本级接受用「token 前缀匹配」近似：草稿续写 K 个词，目标一次续写验证，逐词
比对目标在该位置的最可能续写是否等于草稿续写。
