"""ngram / mock / registry 模型单元测试。"""
import unittest

from draftfly.corpus import CORPUS
from draftfly.models.ngram import NGramModel
from draftfly.models.mock import MockModel
from draftfly.models.registry import (
    build_draft,
    build_mock_with_teacher,
    build_target,
)
from draftfly.config import Config
from draftfly.tokenizer import CharTokenizer


class TestNGram(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_logits_normalized(self):
        m = NGramModel(self.tok, CORPUS, order=4)
        p = m.logits(self.tok.encode("Speculative"))
        self.assertAlmostEqual(sum(p), 1.0, places=5)
        self.assertTrue(all(x >= 0 for x in p))

    def test_higher_order_more_peaked(self):
        lo = NGramModel(self.tok, CORPUS, order=2)
        hi = NGramModel(self.tok, CORPUS, order=6)
        ctx = self.tok.encode("Speculative decoding")
        pl = lo.logits(ctx)
        ph = hi.logits(ctx)
        self.assertGreater(max(ph), max(pl))


class TestMock(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_skill_one_equals_teacher(self):
        teacher = NGramModel(self.tok, CORPUS, order=6)
        own = NGramModel(self.tok, CORPUS, order=3)
        m = MockModel(own, skill=1.0, teacher=teacher)
        ctx = self.tok.encode("the target model")
        self.assertEqual(m.logits(ctx), teacher.logits(ctx))

    def test_skill_zero_equals_own(self):
        teacher = NGramModel(self.tok, CORPUS, order=6)
        own = NGramModel(self.tok, CORPUS, order=3)
        m = MockModel(own, skill=0.0, teacher=teacher)
        ctx = self.tok.encode("the target model")
        self.assertEqual(m.logits(ctx), own.logits(ctx))


class TestRegistry(unittest.TestCase):
    def test_build_draft(self):
        cfg = Config().draft
        tok = CharTokenizer()
        d = build_draft(cfg, tok)
        self.assertEqual(d.name, "draft")

    def test_build_target(self):
        cfg = Config().target
        tok = CharTokenizer()
        t = build_target(cfg, tok)
        self.assertEqual(t.name, "target")

    def test_mock_with_teacher(self):
        tok = CharTokenizer()
        m = build_mock_with_teacher(3, 0.5, tok)
        self.assertEqual(m.skill, 0.5)


if __name__ == "__main__":
    unittest.main()
