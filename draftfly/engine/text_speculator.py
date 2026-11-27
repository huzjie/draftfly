"""文本级投机解码：把「草稿 + 目标」两个文本完成端点接到一起。

针对只有 complete(prompt)->text 的端点（如 OpenAI 兼容服务），用
「token 前缀匹配」近似接受：草稿先续写 K 个词，目标一次续写验证，逐词
比对目标在该位置的最可能续写是否等于草稿续写，相等即接受。
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\S+")


def tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text)


class TextSpeculator:
    def __init__(self, draft, target, *, k: int = 4, max_new_tokens: int = 64) -> None:
        self.draft = draft
        self.target = target
        self.k = k
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        out = prompt
        for _ in range(self.max_new_tokens):
            # 草稿续写 k 个词
            draft_tail = self.draft.complete(out, max_tokens=self.k)
            draft_words = tokenize_words(draft_tail)[: self.k]
            if not draft_words:
                break
            # 目标一次续写验证（长度 = k+1 个词）
            target_tail = self.target.complete(out, max_tokens=self.k + 1)
            target_words = tokenize_words(target_tail)
            accepted = 0
            for i, w in enumerate(draft_words):
                if i < len(target_words) and target_words[i] == w:
                    accepted += 1
                else:
                    break
            if accepted == 0:
                # 目标自己续写一个词兜底
                if target_words:
                    out += " " + target_words[0]
                continue
            out += " " + " ".join(draft_words[:accepted])
        return out
