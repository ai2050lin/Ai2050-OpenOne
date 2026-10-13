import os, hashlib, json, time
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_glm5_scan.txt"
lines = []
def p(s): lines.append(str(s))

t0 = time.time()
# level-2 and level-3 aggregation
agg = {}
cache = {}   # dirpath+rel -> (size,count)
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = os.path.relpath(dirpath, ROOT)
    parts = rel.split(os.sep)
    key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
    for fn in filenames:
        fp = os.path.join(dirpath, fn)
        try: sz = os.path.getsize(fp)
        except OSError: continue
        d = agg.setdefault(key, [0,0])
        d[0] += sz; d[1] += 1

p("=== glm5 LEVEL-2 (top 60 by size, MB) ===")
for k,(sz,cnt) in sorted(agg.items(), key=lambda x:-x[1][0])[:60]:
    p("%10.1f MB %7d  %s" % (sz/1024**2, cnt, k))
p("\nscan_sec=%.1f  groups=%d" % (time.time()-t0, len(agg)))

with open(OUT,"w",encoding="utf-8",newline="") as f:
    f.write("\n".join(lines))
print("ok", len(lines))
