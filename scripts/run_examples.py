# 运行所有可独立执行的示例（跳过需真实后端/长驻服务的 07/09）
import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
skip = {"07_openai_backend.py", "09_http_server.py"}
examples = sorted(glob.glob(os.path.join(root, "examples", "*.py")))
failed = 0
for ex in examples:
    name = os.path.basename(ex)
    if name in skip:
        print("SKIP", name)
        continue
    print("RUN ", name)
    r = subprocess.run([sys.executable, ex], cwd=root)
    if r.returncode != 0:
        failed += 1
        print("FAIL", name)
sys.exit(1 if failed else 0)
