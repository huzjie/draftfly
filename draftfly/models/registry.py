"""模型注册表：按名字构造模型。"""
from __future__ import annotations

from ..corpus import CORPUS
from ..tokenizer import CharTokenizer
from .mock import MockModel
from .ngram import NGramModel


def build_ngram(tokenizer: CharTokenizer | None, order: int, name: str, cost_ms: float) -> NGramModel:
    tokenizer = tokenizer or CharTokenizer()
    return NGramModel(tokenizer, CORPUS, order=order, name=name, cost_ms=cost_ms)


def build_draft(cfg, tokenizer: CharTokenizer | None = None) -> ModelAdapter:
    tokenizer = tokenizer or CharTokenizer()
    if cfg.model == "ngram":
        return NGramModel(
            tokenizer, CORPUS, order=cfg.order, name=cfg.name or "draft",
            cost_ms=cfg.cost_ms,
        )
    raise ValueError(f"未知草稿模型: {cfg.model}")


def build_target(cfg, tokenizer: CharTokenizer | None = None) -> ModelAdapter:
    tokenizer = tokenizer or CharTokenizer()
    if cfg.model == "ngram":
        return NGramModel(
            tokenizer, CORPUS, order=cfg.order, name=cfg.name or "target",
            cost_ms=cfg.cost_ms,
        )
    raise ValueError(f"未知目标模型: {cfg.model}")


# 供 MockModel 评测用的教师目标
def build_mock_with_teacher(order: int, skill: float, tokenizer: CharTokenizer | None = None):
    tokenizer = tokenizer or CharTokenizer()
    teacher = NGramModel(tokenizer, CORPUS, order=6, name="teacher", cost_ms=50.0)
    own = NGramModel(tokenizer, CORPUS, order=order, name="draft", cost_ms=2.0)
    return MockModel(own, skill=skill, teacher=teacher, name=f"draft-skill-{skill}")


# 便捷：从模型实例导入
from .base import ModelAdapter  # noqa: E402

__all__ = [
    "ModelAdapter",
    "NGramModel",
    "MockModel",
    "build_ngram",
    "build_draft",
    "build_target",
    "build_mock_with_teacher",
    "CORPUS",
]
