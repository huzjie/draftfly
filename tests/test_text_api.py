"""text_speculator / mock_text / api 单元测试。"""
import unittest

from draftfly.engine.text_speculator import TextSpeculator, tokenize_words
from draftfly.models.mock_text import MockTextModel
from draftfly.config import Config
from draftfly.api.server import DraftFlyServer


class TestTextSpeculator(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize_words("hello world  foo"), ["hello", "world", "foo"])

    def test_generate_returns_text(self):
        draft = MockTextModel(["the", "model", "is", "fast", "and", "good."], seed=1)
        target = MockTextModel(["the", "model", "is", "fast", "and", "good."], seed=2)
        spec = TextSpeculator(draft, target, k=3, max_new_tokens=8)
        out = spec.generate("Speculative decoding")
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)


class TestDraftFlyServer(unittest.TestCase):
    def test_complete(self):
        cfg = Config()
        srv = DraftFlyServer(cfg)
        res = srv.complete("Speculative decoding", 16)
        self.assertIn("text", res)
        self.assertGreater(res["num_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
