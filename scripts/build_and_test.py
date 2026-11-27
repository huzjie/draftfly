# 构建 + 测试：compileall + unittest + doctor
import subprocess
import sys

BASE = ".".join([])


def run(cmd):
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd="..")
    if r.returncode != 0:
        sys.exit(r.returncode)


run(["python", "-m", "compileall", "-q", "draftfly"])
run(["python", "-m", "unittest", "discover", "-s", "tests"])
run(["python", "-m", "draftfly", "doctor"])
print("全部通过")
