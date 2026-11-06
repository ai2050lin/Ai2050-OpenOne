"""临时审计：对比 .git/lfs/objects 本地实际存在的对象 与 所有 ref 引用到的 LFS OID。
只读，输出可删清单（不执行删除）。
用法：python git_lfs_audit.py <refs_oids.txt>
"""
import os
import sys

ROOT = r"D:\AI2050\Ai2050-OpenOne"
LFS = os.path.join(ROOT, ".git", "lfs", "objects")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfs_prunable.txt")

ref = set()
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.split()
        if parts and len(parts[0]) == 64:
            ref.add(parts[0])

present = {}
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
            if os.path.isfile(p):
                present[name] = os.path.getsize(p)

GB = 1024 ** 3
keep = {k: v for k, v in present.items() if k in ref}
drop = {k: v for k, v in present.items() if k not in ref}

print(f"refs 引用的 LFS OID 数 : {len(ref)}")
print(f"本地实际存在对象数     : {len(present)}  ({sum(present.values())/GB:.2f} GB)")
print(f"  其中被任何 ref 引用  : {len(keep)}  ({sum(keep.values())/GB:.2f} GB)  <- 必须保留")
print(f"  任何 ref 都不引用    : {len(drop)}  ({sum(drop.values())/GB:.2f} GB)  <- 可删")
print(f"refs 引用但本地缺失    : {len(ref - set(present))}  （需 git lfs fetch 才能检出）")

with open(OUT, "w", encoding="utf-8") as f:
    for k in sorted(drop, key=lambda x: -drop[x]):
        f.write(f"{k}\n")
print(f"\n可删清单已写入: {OUT}")

print("\n最大的 10 个可删对象:")
for k, v in sorted(drop.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {v/1024/1024:9.1f} MB  {k}")
if keep:
    print("\n保留对象中最大的 5 个:")
    for k, v in sorted(keep.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {v/1024/1024:9.1f} MB  {k}")
