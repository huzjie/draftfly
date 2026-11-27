# 示例 7：接入真实 OpenAI 兼容端点（vLLM / SGLang 跑的 DFly2 草稿 + 大模型）
# 说明：草稿与目标各指向一个 OpenAI 兼容 /v1/chat/completions 端点。
import os
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.engine.text_speculator import TextSpeculator  # noqa: E402
from draftfly.models.openai_compat import OpenAICompatModel  # noqa: E402

DRAFT_URL = os.environ.get("DRAFTFLY_DRAFT_URL", "http://localhost:8001")
TARGET_URL = os.environ.get("DRAFTFLY_TARGET_URL", "http://localhost:8002")
API_KEY = os.environ.get("DRAFTFLY_API_KEY", "")

draft = OpenAICompatModel(base_url=DRAFT_URL, model="draft-model", api_key=API_KEY, name="draft")
target = OpenAICompatModel(base_url=TARGET_URL, model="target-model", api_key=API_KEY, name="target")

spec = TextSpeculator(draft, target, k=4, max_new_tokens=32)
print(spec.generate("Explain speculative decoding"))
