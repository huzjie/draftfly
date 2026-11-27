# 示例 8：接受率随草稿能力（skill）的曲线 —— 草稿越强接受越多
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.eval import sweep_acceptance  # noqa: E402
from draftfly.tokenizer import CharTokenizer  # noqa: E402

tok = CharTokenizer()
for skill, rate in sweep_acceptance(tok, "the target model verifies", 48):
    bar = "#" * int(rate * 40)
    print(f"skill={skill:.2f}  接受率={rate:.2f}  {bar}")
