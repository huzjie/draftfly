"""kv_cache / scheduler / stopper / token_tree 单元测试。"""
import random
import unittest

from draftfly.engine.kv_cache import KVCache, SparseIndex
from draftfly.engine.scheduler import AdaptiveK
from draftfly.engine.stopper import StopChecker
from draftfly.engine.token_tree import (
    TokenTree,
    follow_accepted,
    verify_tree,
)
from draftfly.corpus import CORPUS
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer


class TestKVCache(unittest.TestCase):
    def test_grow_and_rollback(self):
        kv = KVCache()
        kv.grow([1, 2], [3, 4], [[], [], []])
        self.assertEqual(kv.prefix, [1, 2, 3, 4])
        kv.rollback(2)
        self.assertEqual(kv.prefix, [1, 2])

    def test_sparse_density_lt_one(self):
        idx = SparseIndex(window=8, stride=4)
        idx.build(list(range(32)))
        self.assertLess(idx.density(), 1.0)


class TestScheduler(unittest.TestCase):
    def test_increase_on_high_acceptance(self):
        s = AdaptiveK(k=4, min_k=2, max_k=8, adaptive=True)
        for _ in range(8):
            s.update(4, 4)
        self.assertGreater(s.current(), 4)

    def test_decrease_on_low_acceptance(self):
        s = AdaptiveK(k=4, min_k=2, max_k=8, adaptive=True)
        for _ in range(8):
            s.update(0, 4)
        self.assertLess(s.current(), 4)


class TestStopper(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_eos(self):
        sc = StopChecker(self.tok)
        self.assertTrue(sc.reached([1, self.tok.eos_id]))

    def test_stop_seq(self):
        sc = StopChecker(self.tok, ["end"])
        toks = self.tok.encode("the end")
        self.assertTrue(sc.reached(toks))


class TestTokenTree(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.target = NGramModel(self.tok, CORPUS, order=6)

    def test_tree_flat_size(self):
        tree = TokenTree([1])
        tree.add_path([2, 3])
        tree.add_path([2, 4])
        tree.add_path([5])
        # 非根节点：2, 3, 4, 5（[2,4] 复用节点 2）
        self.assertEqual(tree.size(), 4)

    def test_verify_and_follow(self):
        prefix = self.tok.encode("Speculative decoding")
        top = max(range(self.tok.vocab_size), key=lambda t: self.target.logits(prefix)[t])
        tree = TokenTree(prefix)
        tree.add_path([top])           # 首分支即目标 argmax
        tree.add_path([(top + 1) % self.tok.vocab_size])
        p_map = verify_tree(tree, self.target, prefix)
        path = follow_accepted(tree, p_map, mode="greedy")
        self.assertEqual(path, [top])

    def test_path_tokens(self):
        tree = TokenTree([1])
        leaf = tree.add_path([2, 3])
        self.assertEqual(leaf.path_tokens(), [2, 3])


if __name__ == "__main__":
    unittest.main()
