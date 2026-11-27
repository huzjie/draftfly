"""模型层：统一适配器 + 确定性 n-gram/mock + OpenAI 兼容 + 注册表。"""
from __future__ import annotations

from .base import ModelAdapter
from .mock import MockModel
from .mock_text import MockTextModel
from .ngram import NGramModel
from .openai_compat import OpenAICompatModel
from .registry import (
    build_draft,
    build_mock_with_teacher,
    build_ngram,
    build_target,
)

__all__ = [
    "ModelAdapter",
    "NGramModel",
    "MockModel",
    "MockTextModel",
    "OpenAICompatModel",
    "build_draft",
    "build_target",
    "build_ngram",
    "build_mock_with_teacher",
]
