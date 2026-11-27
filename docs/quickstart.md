# 快速开始

## 零依赖，克隆即跑

```bash
git clone https://github.com/huzjie/draftfly.git
cd draftfly
python -m draftfly doctor          # 环境自检（生成 + 评测全链路）
python -m draftfly generate        # 投机解码生成一段文本
python -m draftfly bench           # 投机 vs 自回归吞吐对比
```

不需要安装任何第三方依赖，不接真实模型也能跑完整链路——内置一个确定性的
字符级 n-gram「草稿/目标」模型对。

## 填一份配置直接跑

```bash
python -m draftfly --config config.example.yaml generate "Speculative decoding"
python -m draftfly --config config.example.yaml serve
```

## 关键正确性保证

贪心模式下，投机解码的输出与目标模型纯自回归逐 token 完全一致——投机解码
「不损失质量」不是口号，是可断言的（见 `examples/01_basic_speculative.py`）。
