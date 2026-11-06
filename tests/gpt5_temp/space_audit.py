"""工作区体积审计（只读）：按目录聚合、Top 大文件、按扩展名与时间分桶。
处理 Windows 长路径（\\?\ 前缀）。用法：python space_audit.py
"""
import os
import sys
from collections import defaultdict
from datetime import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
PREFIX = "\\\\?\\" + ROOT
SKIP = {".git", "node_modules"}

files = []  # (size, relpath, mtime)
errs = 0

for dirpath, dirnames, filenames in os.walk(PREFIX):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            st = os.stat(p)
        except OSError:
            errs += 1
            continue
        rel = p[len(PREFIX):].lstrip("\\")
        files.append((st.st_size, rel, st.st_mtime))

print(f"扫描文件数: {len(files)}  统计失败: {errs}")
GB = 1024 ** 3
total = sum(f[0] for f in files)
print(f"总计(不含 .git / node_modules): {total/GB:.1f} GB\n")

# 按二级目录聚合
agg = defaultdict(int)
cnt = defaultdict(int)
for sz, rel, _ in files:
    parts = rel.split("\\")
    key = "\\".join(parts[:2]) if len(parts) > 1 else parts[0]
    agg[key] += sz
    cnt[key] += 1

print("== 二级目录 Top 25 ==")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {v/GB:8.2f} GB  {cnt[k]:7d} 文件  {k}")

# 扩展名聚合
ext = defaultdict(int)
extn = defaultdict(int)
for sz, rel, _ in files:
    e = os.path.splitext(rel)[1].lower() or "(no ext)"
    ext[e] += sz
    extn[e] += 1
print("\n== 扩展名 Top 12 ==")
for k, v in sorted(ext.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v/GB:8.2f} GB  {extn[k]:7d} 个  {k}")

# Top 30 大文件
print("\n== 单个文件 Top 30 ==")
for sz, rel, mt in sorted(files, reverse=True)[:30]:
    print(f"  {sz/GB:7.2f} GB  {datetime.fromtimestamp(mt):%Y-%m-%d}  {rel}")

# 时间分桶（只统计 >50MB 文件）
buckets = defaultdict(lambda: [0, 0])
now = datetime.now()
for sz, rel, mt in files:
    if sz < 50 * 1024 * 1024:
        continue
    days = (now - datetime.fromtimestamp(mt)).days
    if days <= 7:
        k = "<=7天"
    elif days <= 30:
        k = "8-30天"
    elif days <= 90:
        k = "31-90天"
    else:
        k = ">90天"
    buckets[k][0] += sz
    buckets[k][1] += 1

print("\n== 大文件(>50MB) 按修改时间分桶 ==")
for k in ["<=7天", "8-30天", "31-90天", ">90天"]:
    sz, n = buckets.get(k, [0, 0])
    print(f"  {k:8s}  {sz/GB:8.2f} GB   {n:6d} 个")
