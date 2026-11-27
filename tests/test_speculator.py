"""speculator 核心循环单元测试：贪心一致性 / 接受率 / 长度裁剪。"""
import random
import unittest

from draftfly.bench import autoregressive
from draftfly.corpus import CORPUS
from draftfly.engine.speculator import Speculator
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer

PROMPT = "Speculative decoding accelerates large language model"


class TestSpeculator(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.draft = NGramModel(self.tok, CORPUS, order=3, cost_ms=2.0)
        self.target = NGramModel(self.tok, CORPUS, order=6, cost_ms=50.0)

    def test_greedy_consistency(self):
        spec = Speculator(
            self.draft, self.target, self.tok, k=4, mode="greedy",
            temperature=1.0, seed=7, rng=random.Random(7),
        )
        spec_text = spec.generate(PROMPT, max_new_tokens=48).text
        base_text, _ = autoregressive(
            self.target, self.tok, PROMPT, 48, mode="greedy",
            temperature=1.0, rng=random.Random(7),
        )
        self.assertEqual(spec_text, base_text)

    def test_acceptance_rate_in_range(self):
        spec = Speculator(
            self.draft, self.target, self.tok, k=4, mode="sampling",
            temperature=1.0, seed=1, rng=random.Random(1),
        )
        res = spec.generate(PROMPT, max_new_tokens=64)
        self.assertGreaterEqual(res.acceptance_rate, 0.0)
        self.assertLessEqual(res.acceptance_rate, 1.0)

    def test_no_eos_in_output(self):
        spec = Speculator(
            self.draft, self.target, self.tok, k=4, mode="sampling",
            temperature=1.0, seed=2, rng=random.Random(2),
        )
        res = spec.generate(PROMPT, max_new_tokens=64)
        self.assertNotIn(self.tok.eos_id, res.tokens)

    def test_max_tokens_respected(self):
        spec = Speculator(
            self.draft, self.target, self.tok, k=8, mode="sampling",
            temperature=1.0, seed=3, rng=random.Random(3),
        )
        res = spec.generate(PROMPT, max_new_tokens=16)
        self.assertLessEqual(len(res.tokens), 16)

    def test_stop_sequence(self):
        spec = Speculator(
            self.draft, self.target, self.tok, k=4, mode="greedy",
            temperature=1.0, seed=4, rng=random.Random(4),
        )
        res = spec.generate(PROMPT, max_new_tokens=300, stop_sequences=[" "])
        self.assertTrue(res.text.endswith(" "))
        self.assertLess(len(res.tokens), 300)


if __name__ == "__main__":
    unittest.main()
