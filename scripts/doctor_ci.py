# CI 快速自检：只跑最关键的贪心一致性 + 单调性，秒回
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftfly.bench import autoregressive
from draftfly.corpus import CORPUS
from draftfly.engine.speculator import Speculator
from draftfly.eval import run_eval
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=3, cost_ms=2.0)
target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)

spec = Speculator(draft, target, tok, k=4, mode="greedy", temperature=1.0, seed=0)
spec_text = spec.generate("Speculative decoding", 48).text
base_text, _ = autoregressive(target, tok, "Speculative decoding", 48, mode="greedy", temperature=1.0)
assert spec_text == base_text, "贪心一致性失败"

ev = run_eval(tok, "the target model", 128)
assert ev["monotonic"], "接受率单调性失败"
print("CI 自检通过: 贪心一致 + 单调性 OK")
