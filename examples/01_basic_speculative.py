# 示例 1：贪心投机解码 —— 输出与目标模型纯自回归完全一致
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.bench import autoregressive  # noqa: E402
from draftfly.corpus import CORPUS  # noqa: E402
from draftfly.engine.speculator import Speculator  # noqa: E402
from draftfly.models.ngram import NGramModel  # noqa: E402
from draftfly.tokenizer import CharTokenizer  # noqa: E402

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=3, cost_ms=2.0)
target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)
prompt = "Speculative decoding accelerates large language model"

spec = Speculator(draft, target, tok, k=4, mode="greedy", temperature=1.0, seed=0)
res = spec.generate(prompt, max_new_tokens=48)
base, _ = autoregressive(target, tok, prompt, 48, mode="greedy", temperature=1.0)

print("投机解码输出 :", res.text)
print("自回归基线输出:", base)
print("完全一致      :", res.text == base)
print(f"接受率: {res.acceptance_rate:.2f}")
