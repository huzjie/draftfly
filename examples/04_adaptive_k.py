# 示例 4：自适应 K —— 接受率高就多猜、低就少猜
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

fixed = Speculator(draft, target, tok, k=4, adaptive=False, mode="sampling", seed=0)
adapt = Speculator(draft, target, tok, k=2, min_k=2, max_k=8, adaptive=True, mode="sampling", seed=0)

r1 = fixed.generate("The target model", max_new_tokens=64)
r2 = adapt.generate("The target model", max_new_tokens=64)
print(f"固定 K=4 : 接受率={r1.acceptance_rate:.2f} 迭代={r1.iterations}")
print(f"自适应 K : 接受率={r2.acceptance_rate:.2f} 迭代={r2.iterations} 最终K={adapt.scheduler.current()}")
