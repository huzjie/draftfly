# CI/CD

`.github/workflows/ci.yml` 在 push/PR 时自动：

1. 安装 Python 3.11；
2. `python -m compileall -q draftfly`；
3. `python -m unittest discover -s tests`；
4. `python -m draftfly doctor`。

全绿才允许合并，保证投机解码核心正确性可回归。
