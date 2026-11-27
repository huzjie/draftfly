"""字符级确定性 tokenizer。零依赖、逐字节可复现。"""
from __future__ import annotations

_PAD = "<pad>"
_BOS = "<bos>"
_EOS = "<eos>"
_UNK = "<unk>"

_SPECIAL = [_PAD, _BOS, _EOS, _UNK]

# 词表：4 个特殊 token + 可打印 ASCII（32..126）
_CHARS = [chr(c) for c in range(32, 127)]


class CharTokenizer:
    """把文本编码成 token id 序列，字符级、确定性、无未知词表外溢出。"""

    def __init__(self) -> None:
        self.special = list(_SPECIAL)
        self.vocab = self.special + _CHARS
        self._stoi = {s: i for i, s in enumerate(self.vocab)}
        self.pad_id = self._stoi[_PAD]
        self.bos_id = self._stoi[_BOS]
        self.eos_id = self._stoi[_EOS]
        self.unk_id = self._stoi[_UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self._stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.vocab[i] if 0 <= i < self.vocab_size else _UNK for i in ids
        )

    def id_to_token(self, i: int) -> str:
        return self.vocab[i] if 0 <= i < self.vocab_size else _UNK

    def token_to_id(self, tok: str) -> int:
        return self._stoi.get(tok, self.unk_id)
