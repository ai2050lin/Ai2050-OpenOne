# -*- coding: utf-8 -*-
"""Read-only scan of D:/AI2050/Ai2050-OpenOne/tests: aggregate stats by dir/extension/age."""
import os
import time
import heapq
from collections import defaultdict

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\tests_scan_report.txt"

now = time.time()
total_files = 0
total_bytes = 0

dir1 = defaultdict(lambda: [0, 0])     # top-level dir
dir2 = defaultdict(lambda: [0, 0])     # top-level/sub-level
ext_stats = defaultdict(lambda: [0, 0])
age_buckets = defaultdict(lambda: [0, 0])  # age bucket name
all_mtimes = []                        # (mtime, size, relpath)
pyc_count = 0
pyc_bytes = 0
pycache_dirs = set()
tmp_like = []                          # temp/intermediate patterns

for dirpath, dirnames, filenames in os.walk(ROOT):
    base = os.path.basename(dirpath)
    if base == "__pycache__":
        pycache_dirs.add(dirpath)
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        size = st.st_size
        mtime = st.st_mtime
        total_files += 1
        total_bytes += size
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        parts = rel.split("/")
        dir1[parts[0]][0] += 1
        dir1[parts[0]][1] += size
        if len(parts) >= 2:
            dir2["/".join(parts[:2])][0] += 1
            dir2["/".join(parts[:2])][1] += size
        low = fn.lower()
        ext = os.path.splitext(low)[1] or "(noext)"
        ext_stats[ext][0] += 1
        ext_stats[ext][1] += size
        if ext in (".pyc", ".pyo"):
            pyc_count += 1
            pyc_bytes += size
        age_d = (now - mtime) / 86400.0
        if age_d < 30:
            b = "0-30d"
        elif age_d < 90:
            b = "30-90d"
        elif age_d < 180:
            b = "90-180d"
        else:
            b = ">180d"
        age_buckets[b][0] += 1
        age_buckets[b][1] += size
        all_mtimes.append((mtime, size, rel))
        if low.endswith((".tmp", ".bak", ".old", ".orig")) or "debug" in low or "probe_" in low or "scratch" in low:
            tmp_like.append((size, rel))

lines = []
w = lines.append
w("=" * 80)
w("TESTS DIRECTORY SCAN REPORT (read-only)  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
w("root: %s" % ROOT)
w("total: %d files, %.2f GB" % (total_files, total_bytes / 1024**3))
w("")
w("---- by top-level dir (count, GB) ----")
for k, (c, b) in sorted(dir1.items(), key=lambda kv: -kv[1][1]):
    w("%-30s %8d  %10.3f GB" % (k, c, b / 1024**3))
w("")
w("---- by second-level dir (top 40 by size) ----")
for k, (c, b) in sorted(dir2.items(), key=lambda kv: -kv[1][1])[:40]:
    w("%-50s %8d  %10.3f GB" % (k, c, b / 1024**3))
w("")
w("---- by extension (top 30 by size) ----")
for k, (c, b) in sorted(ext_stats.items(), key=lambda kv: -kv[1][1])[:30]:
    w("%-15s %8d  %10.3f GB" % (k, c, b / 1024**3))
w("")
w("---- age buckets (count, GB) ----")
for k in (">180d", "90-180d", "30-90d", "0-30d"):
    c, b = age_buckets.get(k, [0, 0])
    w("%-10s %8d  %10.3f GB" % (k, c, b / 1024**3))
w("")
w("---- pyc cache ----")
w("__pycache__ dirs: %d ; .pyc/.pyo files: %d, %.3f GB" % (len(pycache_dirs), pyc_count, pyc_bytes / 1024**3))
w("")
w("---- tmp/bak/debug-like files (top 50 by size) ----")
w("count=%d, total=%.3f GB" % (len(tmp_like), sum(s for s, _ in tmp_like) / 1024**3))
for s, rel in sorted(tmp_like, key=lambda t: -t[0])[:50]:
    w("%12.1f KB  %s" % (s / 1024.0, rel))
w("")
w("---- largest 40 files ----")
for s, rel in heapq.nlargest(40, ((sz, r) for _, sz, r in all_mtimes)):
    w("%12.1f MB  %s" % (s / 1024.0 / 1024.0, rel))
w("")
w("---- oldest 40 files ----")
for m, s, rel in heapq.nsmallest(40, all_mtimes):
    w("%s  %10.1f MB  %s" % (time.strftime("%Y-%m-%d", time.localtime(m)), s / 1024.0 / 1024.0, rel))
w("")
w("---- newest 20 files ----")
for m, s, rel in heapq.nlargest(20, all_mtimes):
    w("%s  %10.1f MB  %s" % (time.strftime("%Y-%m-%d", time.localtime(m)), s / 1024.0 / 1024.0, rel))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("OK files=%d bytes=%.2fGB pycache=%d" % (total_files, total_bytes / 1024**3, len(pycache_dirs)))
