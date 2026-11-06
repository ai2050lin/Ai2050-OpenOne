"""临时审计脚本：统计 .git 松散对象的可达性、体积与路径映射。
只读，不修改任何对象。
用法： python git_object_audit.py
"""
import os
import sys
from collections import defaultdict

ROOT = r"D:\AI2050\Ai2050-OpenOne"
GIT = os.path.join(ROOT, ".git")
OBJ = os.path.join(GIT, "objects")
LIST = os.path.join(os.environ.get("TEMP", "/tmp"), "objlist.txt")

# rev-list --objects --all 输出：<sha> [<path>]
sha2path = {}
with open(LIST, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split(" ", 1)
        if len(parts) == 2:
            sha2path[parts[0]] = parts[1]
        elif parts and parts[0]:
            sha2path.setdefault(parts[0], "")

reach_total = 0
unreach_total = 0
unreach_big = []
reach_big = []
loose_count = 0

for pre in os.listdir(OBJ):
    d = os.path.join(OBJ, pre)
    if len(pre) != 2 or not os.path.isdir(d):
        continue
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        sha = pre + name
        sz = os.path.getsize(p)
        loose_count += 1
        if sha in sha2path:
            reach_total += sz
            reach_big.append((sz, sha, sha2path[sha]))
        else:
            unreach_total += sz
            unreach_big.append((sz, sha, ""))

GB = 1024 ** 3
print(f"loose objects scanned : {loose_count}")
print(f"reachable loose       : {reach_total/GB:.2f} GB")
print(f"UNREACHABLE loose     : {unreach_total/GB:.2f} GB   <-- gc --prune=now 可直接回收")
print(f"reachable listed refs : {len(sha2path)} objects")

unreach_big.sort(reverse=True)
reach_big.sort(reverse=True)

print("\n== 最大 15 个不可达对象（可删）==")
for sz, sha, _ in unreach_big[:15]:
    print(f"  {sz/1024/1024:9.1f} MB  {sha}")

print("\n== 最大 15 个可达对象（仍被历史引用，需重写历史才能删）==")
acc = defaultdict(lambda: [0, 0])
for sz, sha, path in reach_big[:400]:
    print(f"  {sz/1024/1024:9.1f} MB  {sha}  {path}")
    key = path.rsplit("/", 1)[0] if "/" in path else path
    acc[key][0] += sz
    acc[key][1] += 1

print("\n== 可达大对象按目录聚合（top 15 目录）==")
for k, (sz, n) in sorted(acc.items(), key=lambda kv: -kv[1][0])[:15]:
    print(f"  {sz/GB:6.2f} GB  n={n:3d}  {k}")
