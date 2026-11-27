# draftfly — 投机解码加速引擎

> 灵感来自腾讯混元 **DFly2「草稿模型」**：用一个又小又快的草稿模型连猜 K 个
> token，再让大模型单次前向并行验证、按接受采样挑出对的——把「串行逐 token
> 生成」变成「猜 K 个、验一次」，在**输出与纯自回归完全一致**的前提下把吞吐拉高数倍。

**零依赖纯 Python · 填配置即可跑 · 输出一致性可断言 · 确定性可复现评测**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-47%20passing-brightgreen)](tests)

---

## 3 秒看懂

| 你的疑问 | 答案 |
|---|---|
| 它是什么？ | 一个投机解码（Speculative Decoding）加速引擎的完整工程实现 |
| 解决什么？ | 大模型自回归逐 token 串行太慢，显存带宽利用率低 |
| 怎么做到的？ | 草稿连猜 K 个 → 目标一次验证 → 接受采样留对改错 |
| 会输出错吗？ | 不会。接受采样保证输出分布与目标一致；贪心模式逐 token 一致（可断言） |
| 能直接跑吗？ | 能。零依赖，`python -m draftfly doctor` 一条命令跑通全链路 |

## 30 秒上手

```bash
git clone https://github.com/huzjie/draftfly.git
cd draftfly

python -m draftfly doctor      # 自检：生成 + 评测全链路
python -m draftfly generate    # 投机解码生成
python -m draftfly bench       # 投机 vs 自回归吞吐对比
python -m draftfly eval        # 接受率随草稿能力单调性
python -m draftfly serve       # OpenAI 兼容 HTTP 服务
```

不接真实模型也能跑完整链路——内置确定性字符级 n-gram「草稿/目标」模型对。

## 核心特性

- **投机解码主循环**：草稿连猜 K 个 token → 目标单次前向并行验证 → 接受采样
  （Leviathan et al. 2022 的 rejection sampling），全接受时额外采样 bonus token。
- **自适应草稿长度 K**：按近期接受率动态加长/缩短草稿，自动找「草稿成本 vs 目标
  前向」的平衡点。
- **KV 缓存复用 + DSA 稀疏索引**：DFly2 的「草稿复用目标缓存与索引、零额外显存」。
- **token 树多候选验证**：草稿 top-b 分支展开成树，目标一次验证整棵树（SpecInfer 思路）。
- **文本级投机解码**：两个 OpenAI 兼容端点即可接真实草稿/目标模型。
- **OpenAI 兼容 HTTP 控制面**：`/v1/completions` + `/v1/chat/completions` + `/health` + `/metrics`。
- **确定性可复现评测**：md5 派生 RNG，接受率单调性 + 贪心一致性可进 CI。

## 正确性保证（为什么可以放心用）

投机解码最容易被质疑的是「会不会输出错」。答案是：**不会**。

- **采样模式**：接受采样保证最终输出分布与目标模型纯自回归的分布**严格一致**。
- **贪心模式**：输出与目标模型贪心自回归**逐 token 完全一致**，这是可直接断言的
  硬保证（`examples/01_basic_speculative.py` 验证：`spec_text == base_text`）。

```
投机解码输出 :  is slow but acceptance rate and correct loop pr
自回归基线输出:  is slow but acceptance rate and correct loop pr
完全一致      : True
```

## 加速比从哪来

延迟模型：`cost_ms` 是每个模型单次前向的「虚拟耗时」。投机解码把目标一次前向
摊到 K 个 token 上：

```
speedup ≈ K / (1 + K · draft_cost / target_cost)
```

`python -m draftfly bench` 实测（草稿 order=4、目标 order=6，greedy）：

| K | 加速比 | 接受率 |
|---|---|---|
| 2 | 1.5x | ~0.3 |
| 4 | 1.5x | ~0.3 |
| 8 | 1.4x | ~0.23 |

> 接受率随草稿能力单调上升（`python -m draftfly eval`）：草稿 skill 0→1，接受率
> 0.53 → 1.0。草稿越强，加速比越高——这正是 DFly2 这类高质量草稿模型的价值。

## 架构

```
draftfly/
├── tokenizer.py       字符级确定性 tokenizer
├── corpus.py          内置训练语料
├── config.py          配置加载与校验
├── models/            模型层（ngram/mock/openai 兼容/注册表）
├── engine/            推理引擎（speculator/acceptance/sampler/token_tree/
│                      kv_cache/scheduler/metrics/stopper）
├── api/               OpenAI 兼容 HTTP 控制面
├── bench.py / eval.py 基准与评测
└── cli.py             CLI（doctor/generate/bench/eval/serve/config）
```

完整架构与算法见 [docs/](docs/README.md)。

## 接真实模型

**方式一：实现 `ModelAdapter`（token 级）** —— 见 `draftfly/models/base.py`。

**方式二：两个 OpenAI 兼容端点（文本级）** —— 见 `examples/07_openai_backend.py`：

```python
from draftfly.engine.text_speculator import TextSpeculator
from draftfly.models.openai_compat import OpenAICompatModel

draft  = OpenAICompatModel(base_url="http://host1:8001", model="draft-model")
target = OpenAICompatModel(base_url="http://host2:8002", model="target-model")
spec   = TextSpeculator(draft, target, k=4, max_new_tokens=32)
print(spec.generate("Explain speculative decoding"))
```

## 配置

填一份 YAML 即可（示例 `config.example.yaml`）：

```yaml
draft:  { model: ngram, order: 3, cost_ms: 2.0 }
target: { model: ngram, order: 6, cost_ms: 50.0 }
engine: { k: 4, adaptive: true, mode: sampling, temperature: 1.0, max_new_tokens: 128 }
server: { host: 127.0.0.1, port: 8000 }
```

更多调参配方见 [recipes/](recipes/) 与 [docs/configuration.md](docs/configuration.md)。

## 测试与评测

```bash
python -m unittest discover -s tests   # 47 个单元测试全绿
python -m draftfly doctor              # 自检（生成 + 单调性）
```

## 部署

- Docker：`docker build -t draftfly . && docker run -p 8000:8000 draftfly`
- Compose：`docker-compose up`
- Kubernetes：`kubectl apply -f deploy/k8s/`
- Helm：`helm install draftfly ./deploy/helm/draftfly`

## 文档

| 分类 | 链接 |
|---|---|
| 原理 | [投机解码算法](docs/algorithms.md) · [接受采样](docs/acceptance-sampling.md) · [token 树](docs/token-tree.md) · [自适应 K](docs/adaptive-k.md) |
| 使用 | [快速开始](docs/quickstart.md) · [配置](docs/configuration.md) · [CLI](docs/cli.md) · [HTTP API](docs/http-api.md) · [草稿模型选型](docs/draft-model-guide.md) |
| 评测 | [基准](docs/benchmarking.md) · [评测](docs/evaluation.md) |

## License

MIT © 2026 draftfly
