# 示例 10：从配置文件加载引擎并生成（填一份 YAML 即可跑）
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.config import Config  # noqa: E402
from draftfly.engine.speculator import Speculator  # noqa: E402
from draftfly.models.registry import build_draft, build_target  # noqa: E402
from draftfly.tokenizer import CharTokenizer  # noqa: E402

cfg = Config.load(join(dirname(dirname(abspath(__file__))), "config.example.yaml"))
tok = CharTokenizer()
draft = build_draft(cfg.draft, tok)
target = build_target(cfg.target, tok)

spec = Speculator(
    draft, target, tok,
    k=cfg.engine.k, mode=cfg.engine.mode, temperature=cfg.engine.temperature,
    seed=cfg.engine.seed,
)
res = spec.generate("Speculative decoding", max_new_tokens=cfg.engine.max_new_tokens)
print("配置驱动输出:", res.text)
print("接受率:", res.acceptance_rate)
