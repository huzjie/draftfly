# 安装

draftfly 是零依赖纯 Python 包，Python 3.8+ 即可。

## 直接使用

```bash
git clone https://github.com/huzjie/draftfly.git
cd draftfly
python -m draftfly doctor
```

## pip 安装（可选）

```bash
pip install -e .
```

## 验证

```bash
python -m unittest discover -s tests
```

全部 47 个单元测试应通过。
