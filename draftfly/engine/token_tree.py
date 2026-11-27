"""token 树与树式多候选验证（SpecInfer / Medusa 思路）。

标准投机解码是一条「草稿链」；树式验证让草稿一次展开多分支（top-b 候选），
目标模型一次前向并行验证所有节点，再从根往下走「逐节点都被接受」的最长路径。
"""
from __future__ import annotations

from .sampler import argmax


class TreeNode:
    __slots__ = ("token", "parent", "children", "depth", "q")

    def __init__(self, token: int, parent=None, q=None) -> None:
        self.token = token
        self.parent = parent
        self.children: list["TreeNode"] = []
        self.depth = parent.depth + 1 if parent else 0
        self.q = q  # 该节点采样所用分布（可选）

    def path_tokens(self) -> list[int]:
        toks = []
        node = self
        while node is not None and node.token is not None:
            toks.append(node.token)
            node = node.parent
        return toks[::-1]


class TokenTree:
    """以 prefix 为根的候选 token 树。"""

    def __init__(self, prefix: list[int]) -> None:
        self.prefix = list(prefix)
        self.root = TreeNode(None)

    def add_path(self, tokens: list[int]) -> TreeNode:
        node = self.root
        for t in tokens:
            child = next((c for c in node.children if c.token == t), None)
            if child is None:
                child = TreeNode(t, node)
                node.children.append(child)
            node = child
        return node

    def flatten(self) -> list[TreeNode]:
        """BFS 展平所有节点（不含根），顺序稳定。"""
        nodes: list[TreeNode] = []
        queue = list(self.root.children)
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def size(self) -> int:
        return len(self.flatten())


def build_tree_from_branching(
    prefix: list[int],
    draft,
    *,
    depth: int,
    branching: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng=None,
) -> TokenTree:
    """由草稿模型做 top-branching 展开，构建候选树。"""
    tree = TokenTree(prefix)
    frontier = [tree.root]
    for _ in range(depth):
        nxt: list[TreeNode] = []
        for node in frontier:
            ctx = prefix + node.path_tokens()
            probs = draft.logits(ctx)
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            for t in order[:branching]:
                child = TreeNode(t, node)
                node.children.append(child)
                nxt.append(child)
        frontier = nxt
    return tree


def verify_tree(tree: TokenTree, target, prefix: list[int]) -> dict:
    """目标一次前向验证整棵树，返回 {id(node): 目标分布 p}。"""
    nodes = tree.flatten()
    result: dict = {}
    for node in nodes:
        ctx = prefix + node.parent.path_tokens() if node.parent else prefix
        result[id(node)] = target.logits(ctx)
    return result


def follow_accepted(tree: TokenTree, p_map: dict, mode: str = "greedy") -> list[int]:
    """从根出发，沿「逐节点被接受」的最长路径返回 token 序列。"""
    accepted: list[int] = []
    node = tree.root
    while True:
        children = node.children
        if not children:
            break
        if mode == "greedy":
            best = None
            for c in children:
                if c.token == argmax(p_map[id(c)]):
                    best = c
                    break
            if best is None:
                break
            accepted.append(best.token)
            node = best
        else:
            # 采样模式：按目标概率软选择一条路径（此处简化为取概率最高的候选）
            c = max(children, key=lambda c: p_map[id(c)][c.token])
            accepted.append(c.token)
            node = c
    return accepted
