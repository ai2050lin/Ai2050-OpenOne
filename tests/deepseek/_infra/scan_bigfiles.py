# -*- coding: utf-8 -*-
"""Scan #3: all files >= 100MB under tests root, grouped by parent dir (read-only)."""
import os
import time
from collections import defaultdict

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\bigfiles_scan.txt"

TH = 100 * 1024 * 1024  # 100MB
big = []
tier2 = []  # 30-100MB
total_big = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size >= TH:
            big.append((st.st_size, p, st.st_mtime))
            total_big += st.st_size
        elif st.st_size >= 30 * 1024 * 1024:
            tier2.append((st.st_size, p))

lines = []
w = lines.append
w("files >=100MB: %d, total %.2f GB  (%s)" % (len(big), total_big / 1024**3, time.strftime("%Y-%m-%d %H:%M")))
w("")
# group by campaign dir (3rd level under glm5/result)
grp = defaultdict(lambda: [0, 0.0])
for sz, p, m in big:
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    parts = rel.split("/")
    if parts[0] == "glm5" and len(parts) >= 3 and parts[1] == "result":
        key = parts[2]
    else:
        key = "/".join(parts[:2])
    grp[key][0] += 1
    grp[key][1] += sz
w("---- grouped by campaign/dir ----")
for k, (c, b) in sorted(grp.items(), key=lambda kv: -kv[1][1]):
    w("%-60s %5d files  %10.3f GB" % (k, c, b / 1024**3))
w("")
w("---- full list (desc by size) ----")
for sz, p, m in sorted(big, key=lambda t: -t[0]):
    w("%10.1f MB  %s  [%s]" % (sz / 1024.0 / 1024.0, p, time.strftime("%Y-%m-%d", time.localtime(m))))
w("")
w("---- 30-100MB files: %d, total %.2f GB ----" % (len(tier2), sum(s for s, _ in tier2) / 1024**3))
for sz, p in sorted(tier2, key=lambda t: -t[0])[:60]:
    w("%10.1f MB  %s" % (sz / 1024.0 / 1024.0, p))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("big files: %d, %.2f GB" % (len(big), total_big / 1024**3))
