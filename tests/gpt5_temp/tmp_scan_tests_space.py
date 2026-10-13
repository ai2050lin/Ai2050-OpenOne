import os, json, time, sys
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_tests_scan.txt"

lines = []
def p(s):
    lines.append(str(s))

t0 = time.time()
# 1) 顶层目录体积
lvl1 = {}
ext_total = {}
file_count = 0
total_bytes = 0
biggest = []  # (size, path)

for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = os.path.relpath(dirpath, ROOT)
    key = rel.split(os.sep)[0] if rel != "." else "(root)"
    for fn in filenames:
        fp = os.path.join(dirpath, fn)
        try:
            sz = os.path.getsize(fp)
        except OSError:
            continue
        file_count += 1
        total_bytes += sz
        d = lvl1.setdefault(key, [0, 0])
        d[0] += sz
        d[1] += 1
        e = os.path.splitext(fn)[1].lower() or "(noext)"
        ext_total[e] = ext_total.get(e, 0) + sz
        biggest.append((sz, fp))

p("=== TOTAL ===")
p("files: %d  bytes: %d (%.2f GB)  scan_sec: %.1f" % (file_count, total_bytes, total_bytes/1024**3, time.time()-t0))

p("\n=== LEVEL-1 (size desc) ===")
for k, (sz, cnt) in sorted(lvl1.items(), key=lambda x: -x[1][0]):
    p("%12.2f MB  %8d files  %s" % (sz/1024**2, cnt, k))

p("\n=== BY EXTENSION (size desc, top 25) ===")
for k, sz in sorted(ext_total.items(), key=lambda x: -x[1])[:25]:
    p("%12.2f MB  %s" % (sz/1024**2, k))

biggest.sort(key=lambda x: -x[0])
p("\n=== TOP 40 LARGEST FILES ===")
for sz, fp in biggest[:40]:
    p("%10.2f MB  %s" % (sz/1024**2, os.path.relpath(fp, ROOT)))

with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(lines))
print("done ->", OUT)
