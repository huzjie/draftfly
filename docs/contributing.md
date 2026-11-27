# 贡献指南

1. Fork 仓库，建分支；
2. 改动后跑 `python -m unittest discover -s tests` 与 `python -m draftfly doctor`；
3. 提交 PR，附改动说明与验证结果。

代码风格：Python 3.8+，标准库优先，保持零依赖；核心算法改动请补测试。
