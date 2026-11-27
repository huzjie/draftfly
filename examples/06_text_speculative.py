# 示例 6：文本级投机解码（两个「文本完成」端点，按 token 前缀近似接受）
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.engine.text_speculator import TextSpeculator  # noqa: E402
from draftfly.models.mock_text import MockTextModel  # noqa: E402

vocab = ["the", "model", "is", "fast", "and", "accurate", "at", "verification."]
draft = MockTextModel(vocab, seed=1, name="draft")
target = MockTextModel(vocab, seed=2, name="target")

spec = TextSpeculator(draft, target, k=3, max_new_tokens=12)
print("文本级投机解码:", spec.generate("Speculative decoding"))
