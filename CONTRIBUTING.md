# 贡献指南

感谢你考虑为 draftfly 做贡献。

## 开发流程

1. Fork 仓库并克隆；
2. 创建特性分支；
3. 修改后运行验证：
   ```bash
   python -m compileall -q draftfly
   python -m unittest discover -s tests
   python -m draftfly doctor
   ```
4. 提交 Pull Request，说明改动与验证结果。

## 约定

- 保持零依赖（仅标准库），除非有充分理由；
- 核心算法（接受采样、投机循环）改动必须补测试；
- 确定性优先：随机性一律走 md5 派生的 RNG；
- 提交信息用祈使句（如 "Add tree verification"）。
