"""config / yamlite / bench / eval 单元测试。"""
import unittest

from draftfly.config import Config
from draftfly.util.yamlite import load as yaml_load
from draftfly.corpus import CORPUS
from draftfly.bench import run_bench
from draftfly.eval import assert_greedy_consistency, run_eval, sweep_acceptance
from draftfly.models.ngram import NGramModel
from draftfly.tokenizer import CharTokenizer


class TestYamlite(unittest.TestCase):
    def test_nested(self):
        raw = """
draft:
  model: ngram
  order: 3
engine:
  k: 4
  adaptive: true
"""
        d = yaml_load(raw)
        self.assertEqual(d["draft"]["order"], 3)
        self.assertEqual(d["engine"]["adaptive"], True)

    def test_scalars(self):
        d = yaml_load("a: 1.5\nb: hello\nc: false")
        self.assertEqual(d["a"], 1.5)
        self.assertEqual(d["b"], "hello")
        self.assertFalse(d["c"])


class TestConfig(unittest.TestCase):
    def test_from_dict(self):
        cfg = Config.from_dict({"engine": {"k": 8, "mode": "greedy"}})
        self.assertEqual(cfg.engine.k, 8)
        self.assertEqual(cfg.engine.mode, "greedy")

    def test_validate_bad_mode(self):
        cfg = Config.from_dict({"engine": {"mode": "bad"}})
        self.assertTrue(cfg.validate())


class TestBenchEval(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.draft = NGramModel(self.tok, CORPUS, order=3, cost_ms=2.0)
        self.target = NGramModel(self.tok, CORPUS, order=6, cost_ms=50.0)

    def test_run_bench_speedup(self):
        out = run_bench(
            self.draft, self.target, self.tok,
            "Speculative decoding", max_new_tokens=32, k=4, mode="greedy",
            seed=0,
        )
        self.assertGreater(out["speedup"], 1.0)

    def test_sweep_monotonic(self):
        rates = sweep_acceptance(self.tok, "the target model", 128, repeats=4)
        vals = [r for _, r in rates]
        self.assertAlmostEqual(vals[-1], 1.0, places=4)
        self.assertGreater(vals[-1], vals[0] + 0.15)
        for i in range(len(vals) - 1):
            self.assertLessEqual(vals[i], vals[i + 1] + 0.05)

    def test_run_eval_monotonic(self):
        out = run_eval(self.tok, "the target model", 32)
        self.assertTrue(out["monotonic"])

    def test_greedy_consistency(self):
        ok = assert_greedy_consistency(
            self.tok, self.target, self.draft, "the target model", 32, seed=0
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
