# 加速比随 K 的扫描
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftfly.bench import run_bench
from draftfly.corpus import CORPUS
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer

tok = CharTokenizer()
for order in (3, 4, 5):
    draft = NGramModel(tok, CORPUS, order=order, cost_ms=2.0)
    target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)
    row = []
    for k in (2, 4, 6, 8, 12):
        out = run_bench(draft, target, tok, "Speculative decoding", 64, k=k, mode="greedy", seed=0)
        row.append(out["speedup"])
    print(f"draft order={order}: speedup@{'K'.join(str(k) for k in (2,4,6,8,12))} = {row}")
