import os, hashlib, time, collections
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_dup_scan.txt"
lines = []
def p(s): lines.append(str(s))

MIN = 5 * 1024 * 1024   # only files >= 5MB matter for space reclaim
t0 = time.time()
by_size = collections.defaultdict(list)
total = 0
n = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in filenames:
        fp = os.path.join(dirpath, fn)
        try: sz = os.path.getsize(fp)
        except OSError: continue
        n += 1; total += sz
        if sz >= MIN:
            by_size[sz].append(fp)

p("scanned %d files, %.1f GB; files>=5MB: %d in %d size-groups (%.1fs)" % (
    n, total/1024**3, sum(len(v) for v in by_size.values()), len(by_size), time.time()-t0))

# candidate duplicate groups
cand = {s:v for s,v in by_size.items() if len(v) > 1}
cand_files = sum(len(v) for v in cand.values())
cand_bytes = sum(s*len(v) for s,v in cand.items())
p("same-size candidate groups: %d groups / %d files / %.1f GB (upper bound of dup space)" % (
    len(cand), cand_files, cand_bytes/1024**3))

# hash them
def h32(fp):
    h = hashlib.blake2b(digest_size=16)
    with open(fp, "rb") as f:
        while True:
            b = f.read(8*1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

groups = collections.defaultdict(list)
done = 0
for s, files in cand.items():
    for fp in files:
        try: groups[(s, h32(fp))].append(fp)
        except OSError: pass
    done += len(files)
    if done % 500 < len(files):
        p("  hashed %d/%d (%.0fs)" % (done, cand_files, time.time()-t0))

dup_groups = {k:v for k,v in groups.items() if len(v) > 1}
reclaim = sum(k[0]*(len(v)-1) for k,v in dup_groups.items())
p("\n=== EXACT DUPLICATE SUMMARY ===")
p("dup groups: %d  reclaimable: %.2f GB  (hashing took %.0fs)" % (
    len(dup_groups), reclaim/1024**3, time.time()-t0))

p("\n=== TOP 30 DUP GROUPS (by reclaimable bytes) ===")
ranked = sorted(dup_groups.items(), key=lambda x: -x[0][0]*(len(x[1])-1))
for (sz,h),v in ranked[:30]:
    p("\n[%.1f MB x%d]" % (sz/1024**2, len(v)))
    for fp in v[:8]:
        p("   " + os.path.relpath(fp, ROOT))
    if len(v) > 8: p("   ... +%d more" % (len(v)-8))

with open(OUT,"w",encoding="utf-8",newline="") as f:
    f.write("\n".join(lines))
print("ok")
