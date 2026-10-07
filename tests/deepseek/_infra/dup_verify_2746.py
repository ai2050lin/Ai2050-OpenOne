# -*- coding: utf-8 -*-
"""Full-hash verification: phase2746_fields vs phase2746/field_store (read-only)."""
import os
import hashlib
import time

GR = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
A = os.path.join(GR, "phase2746_fields")
B = os.path.join(GR, "phase2746", "field_store")
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\dup_verify_2746.txt"

def walk(path):
    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, path)] = p
    return out

def md5full(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

t0 = time.time()
da, db = walk(A), walk(B)
only_a = sorted(set(da) - set(db))
only_b = sorted(set(db) - set(da))
common = sorted(set(da) & set(db))
mismatch = []
for i, k in enumerate(common):
    if md5full(da[k]) != md5full(db[k]):
        mismatch.append(k)
    if (i + 1) % 500 == 0:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write("progress %d/%d %.1fs\n" % (i + 1, len(common), time.time() - t0))

lines = []
lines.append("phase2746_fields full-hash verify %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
lines.append("A files=%d B files=%d common=%d" % (len(da), len(db), len(common)))
lines.append("only in A: %d" % len(only_a))
for k in only_a[:20]:
    lines.append("  A-only: %s" % k)
lines.append("only in B: %d" % len(only_b))
for k in only_b[:20]:
    lines.append("  B-only: %s" % k)
lines.append("md5 mismatch: %d" % len(mismatch))
for k in mismatch[:20]:
    lines.append("  DIFF: %s" % k)
lines.append("verdict: %s" % ("IDENTICAL - A is full redundant copy, safe to remove" if not mismatch and not only_a and not only_b else "NOT identical - do not remove blindly"))
lines.append("elapsed %.1fs" % (time.time() - t0))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("verify done")
