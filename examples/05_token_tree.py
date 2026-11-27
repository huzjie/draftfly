# 示例 5：token 树多候选验证（SpecInfer / Medusa 思路）
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.corpus import CORPUS  # noqa: E402
from draftfly.engine.token_tree import (  # noqa: E402
    build_tree_from_branching,
    follow_accepted,
    verify_tree,
)
from draftfly.models.ngram import NGramModel  # noqa: E402
from draftfly.tokenizer import CharTokenizer  # noqa: E402

tok = CharTokenizer()
draft = NGramModel(tok, CORPUS, order=3, cost_ms=2.0)
target = NGramModel(tok, CORPUS, order=6, cost_ms=50.0)
prefix = tok.encode("Speculative")

tree = build_tree_from_branching(prefix, draft, depth=4, branching=2)
p_map = verify_tree(tree, target, prefix)
path = follow_accepted(tree, p_map, mode="greedy")

print("候选树节点数:", tree.size())
print("接受路径 tokens:", path)
print("续写文本:", tok.decode(path))
