# 延迟缩放：目标越慢，投机加速上限越高
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftfly.bench import run_bench
from draftfly.corpus import CORPUS
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=4, cost_ms=2.0)
for target_cost in (10.0, 50.0, 100.0, 500.0):
    target = NGramModel(tok, CORPUS, order=6, cost_ms=target_cost)
    out = run_bench(draft, target, tok, "Speculative decoding", 64, k=8, mode="greedy", seed=0)
    print(f"target_cost={target_cost}ms -> speedup={out['speedup']}x")
