# 基准矩阵：扫 K 与 mode，输出加速比表格
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftfly.bench import run_bench
from draftfly.corpus import CORPUS
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=4, cost_ms=2.0)
target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)

print(f"{'mode':<10}{'K':<4}{'speedup':<10}{'acceptance':<12}{'baseline_ms':<12}{'spec_ms'}")
for mode in ("greedy", "sampling"):
    for k in (2, 4, 6, 8):
        out = run_bench(draft, target, tok, "Speculative decoding", 64, k=k, mode=mode, seed=0)
        print(f"{mode:<10}{k:<4}{out['speedup']:<10}{out['acceptance_rate']:<12}{out['baseline_ms']:<12}{out['speculative_ms']}")
