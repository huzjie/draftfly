# 示例 9：起一个 OpenAI 兼容 HTTP 服务（/v1/completions + /v1/chat/completions）
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from draftfly.api.server import serve  # noqa: E402
from draftfly.config import Config  # noqa: E402

cfg = Config()  # 默认：draft order=3 / target order=6，端口 8000
print("服务启动于 http://127.0.0.1:8000 ，Ctrl+C 退出")
serve(cfg)
