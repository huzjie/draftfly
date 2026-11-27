# 字符级 tokenizer

`CharTokenizer` 是确定性的字符级 tokenizer：

- 词表 = 4 个特殊 token（`<pad>/<bos>/<eos>/<unk>`）+ 可打印 ASCII（32..126），共 99；
- `encode`/`decode` 逐字节可复现，无词表外溢出；
- 控制 token（pad/bos/unk）在模型层被屏蔽，永不作为续写输出；`<eos>` 保留作停止信号。

优点：零依赖、完全确定性；局限：字符级粒度，语义上比子词 tokenizer 粗。
