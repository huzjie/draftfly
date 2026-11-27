# 架构

```
draftfly/
├── tokenizer.py       字符级确定性 tokenizer
├── corpus.py          内置训练语料
├── config.py          配置加载与校验
├── models/            模型层（base/ngram/mock/openai_compat/mock_text/registry）
├── engine/            推理引擎
│   ├── speculator.py  投机解码主循环
│   ├── acceptance.py  接受采样 + 残差采样
│   ├── sampler.py     温度/top-k/top-p + 多项式采样
│   ├── token_tree.py  token 树多候选验证
│   ├── text_speculator.py 文本级投机解码
│   ├── kv_cache.py    KV 缓存 + 稀疏索引
│   ├── scheduler.py   自适应 K
│   ├── metrics.py     指标记账
│   └── stopper.py     停止条件
├── api/               HTTP 控制面
├── bench.py           基准
├── eval.py            评测
└── cli.py             CLI 入口
```

数据流：`prompt → tokenizer.encode → Speculator.generate → tokenizer.decode → text`。
