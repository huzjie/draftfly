"""tokenizer / sampler / acceptance / rng / clock 单元测试。"""
import random
import unittest

from draftfly.tokenizer import CharTokenizer
from draftfly.engine.sampler import argmax, filter_probs, sample
from draftfly.engine.acceptance import (
    greedy_accept,
    reject_sampling,
    residual_sample,
)
from draftfly.util.rng import derive_seed, make_rng
from draftfly.util.timing import Clock


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_vocab_size(self):
        self.assertGreater(self.tok.vocab_size, 90)

    def test_roundtrip(self):
        text = "Hello, speculative decoding!"
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)

    def test_special_ids(self):
        self.assertEqual(self.tok.decode([self.tok.eos_id]), "<eos>")

    def test_unknown_char(self):
        ids = self.tok.encode("\u4e2d")
        self.assertEqual(ids, [self.tok.unk_id])


class TestSampler(unittest.TestCase):
    def test_argmax(self):
        self.assertEqual(argmax([0.1, 0.7, 0.2]), 1)

    def test_filter_temperature_sharpens(self):
        probs = [0.2, 0.8]
        sharp = filter_probs(probs, temperature=0.5)
        self.assertGreater(sharp[1], probs[1])

    def test_top_k(self):
        probs = filter_probs([0.5, 0.3, 0.2], top_k=1)
        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs[1], 0.0)

    def test_sample_deterministic(self):
        r1 = random.Random(1)
        r2 = random.Random(1)
        p = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(sample(p, rng=r1), sample(p, rng=r2))


class TestAcceptance(unittest.TestCase):
    def test_greedy_accept_match(self):
        p = [0.1, 0.9]
        ok, corr = greedy_accept(1, p)
        self.assertTrue(ok)
        self.assertIsNone(corr)

    def test_greedy_reject(self):
        p = [0.1, 0.9]
        ok, corr = greedy_accept(0, p)
        self.assertFalse(ok)
        self.assertEqual(corr, 1)

    def test_reject_sampling_under_confident_accepts(self):
        rng = random.Random(0)
        q = [0.4, 0.6]
        p = [0.3, 0.7]
        ok, _ = reject_sampling(1, q, p, rng)
        self.assertTrue(ok)

    def test_residual_normalizes(self):
        rng = random.Random(0)
        p = [0.1, 0.5, 0.4]
        q = [0.1, 0.7, 0.2]
        tok = residual_sample(p, q, rng)
        self.assertIn(tok, range(3))


class TestRng(unittest.TestCase):
    def test_derive_deterministic(self):
        self.assertEqual(derive_seed("abc"), derive_seed("abc"))

    def test_make_rng_reproducible(self):
        a = make_rng("k", 1)
        b = make_rng("k", 1)
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])


class TestClock(unittest.TestCase):
    def test_simulated(self):
        c = Clock(real=False)
        c.advance_ms(500)
        self.assertEqual(c.now(), 0.5)


if __name__ == "__main__":
    unittest.main()
