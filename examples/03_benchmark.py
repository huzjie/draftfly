# 示例 3：吞吐基准 —— 投机解码 vs 纯自回归，看加速比
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.bench import run_bench  # noqa: E402
from draftfly.corpus import CORPUS  # noqa: E402
from draftfly.models.ngram import NGramModel  # noqa: E402
from draftfly.tokenizer import CharTokenizer  # noqa: E402

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=3, cost_ms=2.0)
target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)

for k in (2, 4, 8):
    out = run_bench(
        draft, target, tok,
        "Speculative decoding makes", max_new_tokens=64,
        k=k, mode="greedy", seed=0,
    )
    print(
        f"K={k}: 加速比={out['speedup']}x  "
        f"基线={out['baseline_ms']}ms 投机={out['speculative_ms']}ms  "
        f"接受率={out['acceptance_rate']}"
    )
