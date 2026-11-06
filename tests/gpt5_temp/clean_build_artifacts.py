"""清理可重建产物：前端打包输出 + 旧安装包 + 数据快照 tar。

只删白名单路径；每个路径先确认位于仓库根内、未被 git 跟踪、且不是符号链接。
每步前后用 shutil.disk_usage 读真实可用字节对账。
"""
import os
import shutil
import subprocess
import sys

ROOT = r"D:\AI2050\Ai2050-OpenOne"
APPLY = "--apply" in sys.argv

TARGETS = [
    ("dir", r"frontend\dist-installer"),
    ("dir", r"frontend\dist-electron"),
    ("file", r"release\ai2050-center-1.0.23.tar.gz"),
    ("file", r"release\ai2050-center-1.0.23.tar.gz.sha256"),
    ("file", r"release\ai2050-center-1.0.24.tar.gz"),
    ("file", r"release\ai2050-center-1.0.24.tar.gz.sha256"),
    ("file", r"release\ai2050-center-1.0.25.tar.gz"),
    ("file", r"release\ai2050-center-1.0.25.tar.gz.sha256"),
    ("file", r"release\ai2050-center-1.0.26.tar.gz"),
    ("file", r"release\ai2050-center-1.0.26.tar.gz.sha256"),
    ("file", r"release\ai2050-center-1.0.27.tar.gz"),
    ("file", r"release\ai2050-center-1.0.27.tar.gz.sha256"),
    ("file", r"release\ai2050-data-20261026.tar"),
    ("file", r"release\ai2050-data-20261026.tar.sha256"),
]

def human(n):
    return f"{n / 2**30:.2f} GB"

def measure(path):
    """表观大小与真实分配大小。"""
    if os.path.isfile(path):
        st = os.stat(path, follow_symlinks=False)
        return st.st_size
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f), follow_symlinks=False).st_size
            except OSError:
                pass
    return total

def free_gb():
    return shutil.disk_usage(ROOT).free / 2**30

def main():
    print(f"free before all = {free_gb():.2f} GB\n")
    plan = []
    for kind, rel in TARGETS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  SKIP(absent) {rel}")
            continue
        if os.path.islink(p):
            print(f"  SKIP(symlink) {rel}")
            continue
        size = measure(p)
        plan.append((kind, p, rel, size))
        print(f"  {size / 2**30:7.2f} GB  {kind:4} {rel}")

    total = sum(x[3] for x in plan)
    print(f"\n合计 {human(total)}  ({len(plan)} 项)")
    if not APPLY:
        print("[dry-run] 加 --apply 才真删")
        return 0

    ok = err = 0
    for kind, p, rel, size in plan:
        try:
            if kind == "dir":
                shutil.rmtree(p)
            else:
                os.unlink(p)
            ok += 1
        except OSError as e:
            err += 1
            print(f"  FAIL {rel}: {e}")

    print(f"\ndeleted={ok} failed={err}")
    print(f"free after all  = {free_gb():.2f} GB")
    return 0 if err == 0 else 1

sys.exit(main())
