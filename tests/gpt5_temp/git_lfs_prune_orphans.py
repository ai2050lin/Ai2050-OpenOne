"""删除 .git/lfs/objects 中「任何 ref 都不引用」的孤儿 LFS 对象。
安全前提：先用 git lfs ls-files --all --long 取全量引用集合（含 refs/codex/*、refs/stash），
只删不在集合内的对象。执行前打印将删数量与体积，执行后打印回收体积。
用法：python git_lfs_prune_orphans.py [--apply]
"""
import os
import subprocess
import sys

ROOT = r"D:\AI2050\Ai2050-OpenOne"
LFS = os.path.join(ROOT, ".git", "lfs", "objects")
APPLY = "--apply" in sys.argv

out = subprocess.run(
    ["git", "-C", ROOT, "lfs", "ls-files", "--all", "--long"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
)
if out.returncode != 0:
    print("git lfs ls-files failed:", out.stderr[:400])
    sys.exit(1)

ref = set()
for line in out.stdout.splitlines():
    parts = line.split()
    if parts and len(parts[0]) == 64:
        ref.add(parts[0])
print(f"refs 引用 LFS OID: {len(ref)}")

targets = []
all_count = 0
all_bytes = 0
for pre in os.listdir(LFS):
    d = os.path.join(LFS, pre)
    if len(pre) != 2 or not os.path.isdir(d):
        continue
    for pre2 in os.listdir(d):
        d2 = os.path.join(d, pre2)
        if not os.path.isdir(d2):
            continue
        for name in os.listdir(d2):
            p = os.path.join(d2, name)
            if not os.path.isfile(p):
                continue
            sz = os.path.getsize(p)
            all_count += 1
            all_bytes += sz
            if name not in ref:
                targets.append((p, sz, name))

tgt_bytes = sum(t[1] for t in targets)
GB = 1024 ** 3
print(f"本地 LFS 对象: {all_count} ({all_bytes/GB:.2f} GB)")
print(f"待删孤儿    : {len(targets)} ({tgt_bytes/GB:.2f} GB)")
print(f"保留        : {all_count - len(targets)} ({(all_bytes-tgt_bytes)/GB:.2f} GB)")

LIST_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfs_deleted_oids.txt")
with open(LIST_OUT, "w", encoding="utf-8") as f:
    for p, sz, name in targets:
        f.write(f"{name}\t{sz}\n")
print(f"\n待删 OID 清单已写出（供日后 git lfs fetch 恢复）: {LIST_OUT}")

if not APPLY:
    print("\n[dry-run] 加 --apply 才会真删")
    sys.exit(0)

freed = 0
for p, sz, _ in targets:
    try:
        os.remove(p)
        freed += sz
    except OSError as e:
        print("skip", p, e)

# 清空目录
for pre in os.listdir(LFS):
    d = os.path.join(LFS, pre)
    if len(pre) != 2 or not os.path.isdir(d):
        continue
    for pre2 in list(os.listdir(d)):
        d2 = os.path.join(d, pre2)
        if os.path.isdir(d2) and not os.listdir(d2):
            os.rmdir(d2)
    if not os.listdir(d):
        os.rmdir(d)

print(f"\n已删除 {len(targets)} 个对象，回收 {freed/GB:.2f} GB")
