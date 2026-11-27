# 接受率随草稿能力的扫描（多 skill，长样本）
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftfly.eval import sweep_acceptance
from draftfly.tokenizer import CharTokenizer

tok = CharTokenizer()
for s, r in sweep_acceptance(tok, "the target model verifies", 256, repeats=8):
    print(f"skill={s:.2f}  acceptance={r:.4f}")
