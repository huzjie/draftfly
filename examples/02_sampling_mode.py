# 示例 2：采样模式（温度 / top-k / top-p）——接受采样保证分布一致
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.corpus import CORPUS  # noqa: E402
from draftfly.engine.speculator import Speculator  # noqa: E402
from draftfly.models.ngram import NGramModel  # noqa: E402
from draftfly.tokenizer import CharTokenizer  # noqa: E402

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=3, cost_ms=2.0)
target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)

for temp in (1.0, 0.7):
    spec = Speculator(
        draft, target, tok, k=4, mode="sampling",
        temperature=temp, top_p=0.9, seed=1,
    )
    res = spec.generate("The draft model", max_new_tokens=48)
    print(f"[T={temp}] 输出: {res.text}")
    print(f"[T={temp}] 接受率: {res.acceptance_rate:.2f}")
