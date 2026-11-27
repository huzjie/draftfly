"""OpenAI 兼容 API 适配器：把草稿/目标模型指向任意兼容端点。

用于「文本级投机解码」：把草稿模型与目标模型都接到真实服务（如 vLLM / SGLang
跑的 DFly2 草稿 + 大模型），用 complete() 拿文本、按 token 前缀做接受校验。

注意：本适配器面向「文本完成」接口，与字符级 logits 路径互补。
"""
from __future__ import annotations

import json
import urllib.request

from .base import ModelAdapter


class OpenAICompatModel(ModelAdapter):
    """通过 OpenAI 兼容 /v1/chat/completions 端点生成文本。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        name: str = "openai",
        cost_ms: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(name=name, cost_ms=cost_ms)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def logits(self, context: list[int]) -> list[float]:
        raise NotImplementedError(
            "OpenAICompatModel 面向文本完成接口，请使用 TextSpeculator（文本级投机解码）"
        )
