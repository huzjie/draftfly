"""OpenAI 兼容 HTTP 控制面（标准库 http.server，零依赖）。

提供：
  GET  /health            存活探针
  GET  /v1/models         模型列表
  POST /v1/completions    文本补全（投机解码）
  POST /v1/chat/completions 对话补全（OpenAI 兼容）
  GET  /metrics           指标快照
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import Config
from ..engine.speculator import Speculator
from ..models.registry import build_draft, build_target
from ..tokenizer import CharTokenizer
from ..util.logger import Logger


class DraftFlyServer:
    """持有一个投机解码引擎，按配置构建。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tokenizer = CharTokenizer()
        self.draft = build_draft(cfg.draft, self.tokenizer)
        self.target = build_target(cfg.target, self.tokenizer)
        self.log = Logger()
        self._lock = threading.Lock()
        self._last_metrics = {}

    def make_speculator(self) -> Speculator:
        e = self.cfg.engine
        return Speculator(
            self.draft, self.target, self.tokenizer,
            k=e.k, min_k=e.min_k, max_k=e.max_k, adaptive=e.adaptive,
            mode=e.mode, temperature=e.temperature, top_k=e.top_k, top_p=e.top_p,
            bonus_token=e.bonus_token, seed=e.seed,
        )

    def complete(self, prompt: str, max_tokens: int) -> dict:
        spec = self.make_speculator()
        res = spec.generate(prompt, max_new_tokens=max_tokens)
        self._last_metrics = res.to_dict()
        return res.to_dict()


def make_handler(server: DraftFlyServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "draftfly/1.0"

        def log_message(self, *args):  # 静默默认访问日志
            pass

        def _send(self, code: int, obj, content_type="application/json"):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                self._send(200, {"status": "ok", "engine": "draftfly"})
            elif path == "/v1/models":
                self._send(200, {
                    "object": "list",
                    "data": [
                        {"id": server.cfg.draft.name, "object": "model"},
                        {"id": server.cfg.target.name, "object": "model"},
                    ],
                })
            elif path == "/metrics":
                self._send(200, server._last_metrics or {})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/v1/completions":
                self._completions(chat=False)
            elif path == "/v1/chat/completions":
                self._completions(chat=True)
            else:
                self._send(404, {"error": "not found"})

        def _completions(self, chat: bool):
            payload = self._read_json()
            if chat:
                msgs = payload.get("messages", [])
                prompt = ""
                for m in msgs:
                    prompt += (m.get("content", "") or "") + "\n"
            else:
                prompt = payload.get("prompt", "")
            max_tokens = int(payload.get("max_tokens", server.cfg.engine.max_new_tokens))
            result = server.complete(prompt, max_tokens)
            text = result["text"]
            self._send(200, {
                "id": "cmpl-draftfly",
                "object": "text_completion" if not chat else "chat.completion",
                "created": 0,
                "model": server.cfg.target.name,
                "choices": [
                    {
                        "index": 0,
                        "text" if not chat else "message": text if not chat else {
                            "role": "assistant", "content": text
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": result["num_tokens"]},
                "draftfly": result,
            })

    return Handler


def serve(cfg: Config) -> None:
    srv = DraftFlyServer(cfg)
    handler = make_handler(srv)
    httpd = ThreadingHTTPServer((cfg.server.host, cfg.server.port), handler)
    srv.log.info(f"draftfly 服务已启动: http://{cfg.server.host}:{cfg.server.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
