# 把任意文本文件转成 corpus.py 里的语料（替换内置 CORPUS）
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_corpus.py input.txt [output.py]")
        sys.exit(2)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "draftfly/corpus.py"
    text = open(src, encoding="utf-8").read().replace('"', "'").replace("\n", " ")
    body = 'CORPUS = (\n    "' + ' "\n    "'.join(_chunks(text)) + '"\n)\n'
    open(dst, "w", encoding="utf-8").write('"""内置训练语料。"""\n\n' + body)
    print("已写入", dst)


def _chunks(s, n=120):
    return [s[i:i + n] for i in range(0, len(s), n)]


if __name__ == "__main__":
    main()
