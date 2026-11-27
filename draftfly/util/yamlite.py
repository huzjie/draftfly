"""极简 YAML 子集解析器：支持嵌套 map + 标量，够本项目配置使用。"""
from __future__ import annotations


def load(text: str) -> dict:
    """解析一个缩进式 YAML 子集（map 嵌套 + 标量），返回 dict。"""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        # 去掉行内注释（# 前有空白才视为注释）
        line = _strip_comment(line)
        if ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(val)
    return root


def _strip_comment(line: str) -> str:
    in_quote = False
    quote = ""
    for i, ch in enumerate(line):
        if ch in ('"', "'"):
            if not in_quote:
                in_quote, quote = True, ch
            elif ch == quote:
                in_quote, quote = False, ""
        elif ch == "#" and not in_quote and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _scalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    if s.lower() in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
