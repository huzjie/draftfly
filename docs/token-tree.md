# token 树多候选验证

标准投机解码是一条「草稿链」；树式验证（SpecInfer / Medusa 思路）让草稿一次
展开多分支（top-b 候选），目标一次前向并行验证所有节点，再从根往下走
「逐节点都被接受」的最长路径。

优点：多候选提高「至少有一个分支命中目标」的概率，等效提高接受率，适合草稿
能力较弱时。

实现见 `draftfly/engine/token_tree.py`：
- `build_tree_from_branching` 用草稿 top-b 展开；
- `verify_tree` 目标一次前向验证整棵树；
- `follow_accepted` 沿接受路径取最长前缀。
