"""配置加载与校验（YAML/JSON/dict，零依赖 YAML 子集解析）。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .util.yamlite import load as yaml_load


@dataclass
class ModelConfig:
    model: str = "ngram"
    order: int = 3
    cost_ms: float = 2.0
    skill: float = 1.0
    name: str = "draft"
    seed: int = 0


@dataclass
class TargetConfig:
    model: str = "ngram"
    order: int = 6
    cost_ms: float = 50.0
    name: str = "target"
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""


@dataclass
class EngineConfig:
    k: int = 4
    min_k: int = 2
    max_k: int = 8
    adaptive: bool = True
    mode: str = "sampling"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    bonus_token: bool = True
    max_new_tokens: int = 128
    seed: int = 42
    real_clock: bool = False


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    draft: ModelConfig = field(default_factory=ModelConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        if path and Path(path).exists():
            raw = _read_file(path)
        else:
            raw = {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        cfg = cls()
        for section, typ in (
            ("draft", ModelConfig),
            ("target", TargetConfig),
            ("engine", EngineConfig),
            ("server", ServerConfig),
        ):
            data = raw.get(section) or {}
            if isinstance(data, dict):
                for k, v in data.items():
                    if hasattr(getattr(cfg, section), k):
                        setattr(getattr(cfg, section), k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        errors = []
        if self.engine.mode not in ("sampling", "greedy"):
            errors.append(f"engine.mode 必须是 sampling 或 greedy，收到 {self.engine.mode}")
        if self.draft.order < 1:
            errors.append("draft.order 必须 >= 1")
        if self.target.order < 1:
            errors.append("target.order 必须 >= 1")
        if self.engine.min_k < 1 or self.engine.max_k < self.engine.min_k:
            errors.append("engine.min_k/max_k 非法")
        if not (0.0 <= self.engine.temperature):
            errors.append("engine.temperature 必须 >= 0")
        return errors


def _read_file(path: str) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        return yaml_load(text)
    return json.loads(text)
