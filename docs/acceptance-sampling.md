# 接受采样

核心判定（Leviathan et al. 2022）：

- 草稿分布 q 在某 token 上概率不高于目标 p（`q(x) <= p(x)`）→ 必接受；
- 否则以概率 `p(x)/q(x)` 接受；
- 拒绝时从残差分布 `max(0, p - q)` 归一化采样一个修正 token。

接受率的期望值 = `1 - TV(p, q)`（TV = 总变差距离），因此草稿越贴近目标，接受率
越高，且单调。

贪心特例：草稿 token 等于目标 argmax 才接受，否则修正为目标 argmax。这是
「投机解码输出 == 目标贪心输出」的硬保证来源。

实现见 `draftfly/engine/acceptance.py`。
