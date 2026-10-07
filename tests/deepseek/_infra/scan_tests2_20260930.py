# -*- coding: utf-8 -*-
"""Targeted read-only scan #2: campaign-level breakdown, dupes, temp dirs, junk patterns."""
import os
import time
import hashlib
from collections import defaultdict

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\tests_scan2_report.txt"

lines = []
w = lines.append

def agg(path, depth_from_root):
    """aggregate files under path by next level dir after depth_from_root parts"""
    stats = defaultdict(lambda: [0, 0])
    for dirpath, dirnames, filenames in os.walk(path):
        rel = os.path.relpath(dirpath, path)
        n = 0 if rel == "." else len(rel.replace("\\", "/").split("/"))
        # we want grouping by the dirpath's own level relative to path depth
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            grp = rel.replace("\\", "/") if rel != "." else "(root)"
            # group at first level only
            grp = grp.split("/")[0] if grp != "(root)" else "(root files)"
            stats[grp][0] += 1
            stats[grp][1] += st.st_size
    return stats

# 1. glm5/result campaign-level breakdown
GR = os.path.join(ROOT, "glm5", "result")
w("=" * 90)
w("## glm5/result campaign-level (dir under glm5/result, count, GB) %s" % time.strftime("%Y-%m-%d %H:%M"))
st = agg(GR, 1)
for k, (c, b) in sorted(st.items(), key=lambda kv: -kv[1][1]):
    w("%-60s %8d  %10.3f GB" % (k, c, b / 1024**3))

# 2. junk pattern totals inside glm5/result: client_build, code_checks, forward_pass_demo, vite.svg, node_modules-like
w("")
w("## junk patterns inside glm5/result")
pats = ["client_build", "code_checks", "build_", "node_modules", "smoke"]
pat_stats = defaultdict(lambda: [0, 0])
for dirpath, dirnames, filenames in os.walk(GR):
    rel = os.path.relpath(dirpath, GR).replace("\\", "/")
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            sz = os.stat(p).st_size
        except OSError:
            continue
        for pat in pats:
            if pat in rel or rel.startswith(pat):
                pat_stats[pat][0] += 1
                pat_stats[pat][1] += sz
                break
for k, (c, b) in sorted(pat_stats.items(), key=lambda kv: -kv[1][1]):
    w("%-20s %8d files  %10.3f GB" % (k, c, b / 1024**3))

# forward_pass_demo.json count and total
w("")
c_fp = b_fp = 0
for dirpath, dirnames, filenames in os.walk(GR):
    if "code_checks" in dirpath or "client_build" in dirpath:
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                sz = os.stat(p).st_size
            except OSError:
                continue
            c_fp += 1
            b_fp += sz
w("glm5/result files under client_build|code_checks dirs: %d files, %.3f GB" % (c_fp, b_fp / 1024**3))

# 3. duplicate candidates: phase2746_fields vs phase2746/field_store ; pattern_family_atlas copies
def dir_size_hash(path):
    """return {relpath: (size, hash)} for files under path (hash only if size matches candidate)"""
    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[os.path.relpath(p, path)] = (st.st_size, st.st_mtime)
    return out

w("")
w("## duplicate check: phase2746_fields vs phase2746/field_store")
a = os.path.join(GR, "rdc_query_construction_20260913", "phase2746_fields")
b = os.path.join(GR, "rdc_query_construction_20260913", "phase2746", "field_store")
if os.path.isdir(a) and os.path.isdir(b):
    da = dir_size_hash(a)
    db = dir_size_hash(b)
    common = set(da) & set(db)
    same = sum(1 for k in common if da[k][0] == db[k][0])
    total_b = sum(da[k][0] for k in common if da[k][0] == db[k][0])
    w("files A=%d B=%d common=%d size-identical=%d (%.3f GB)" % (len(da), len(db), len(common), same, total_b / 1024**3))
    # md5 verify up to 8 files
    def md5(p, n=8 * 1024 * 1024):
        h = hashlib.md5()
        with open(p, "rb") as f:
            h.update(f.read(n))
        return h.hexdigest()
    checked = 0
    mismatch = 0
    for k in sorted(common):
        if da[k][0] == db[k][0] and da[k][0] > 1024 * 1024:
            pa, pb = os.path.join(a, k), os.path.join(b, k)
            if md5(pa) != md5(pb):
                mismatch += 1
            checked += 1
            if checked >= 8:
                break
    w("head-md5 spot-check: %d checked, %d mismatch" % (checked, mismatch))
else:
    w("one of dirs missing: A=%s B=%s" % (os.path.isdir(a), os.path.isdir(b)))

w("")
w("## duplicate check: pattern_family_atlas in 3 locations")
loc1 = os.path.join(GR, "client_visualization_assets", "pattern_family_atlas", "v1")
loc2 = os.path.join(ROOT, "result", "pattern_family_atlas", "v1")
for name, lp in (("glm5/client_visualization_assets/pattern_family_atlas/v1", loc1),
                 ("tests/result/pattern_family_atlas/v1", loc2)):
    if os.path.isdir(lp):
        d = dir_size_hash(lp)
        tb = sum(v[0] for v in d.values())
        w("%-55s %5d files  %.3f GB" % (name, len(d), tb / 1024**3))
    else:
        w("%-55s MISSING" % name)
if os.path.isdir(loc1) and os.path.isdir(loc2):
    d1, d2 = dir_size_hash(loc1), dir_size_hash(loc2)
    common = set(d1) & set(d2)
    same = sum(1 for k in common if d1[k][0] == d2[k][0])
    w("common=%d size-identical=%d" % (len(common), same))

# 4. temp dirs breakdown depth1
for td in ("glm5_temp", "codex_temp", "gpt5_temp", "gemini_temp", "result"):
    p = os.path.join(ROOT, td)
    w("")
    w("## tests/%s depth1 breakdown (top 25)" % td)
    if not os.path.isdir(p):
        w("MISSING")
        continue
    st = agg(p, 1)
    for k, (c, b) in sorted(st.items(), key=lambda kv: -kv[1][1])[:25]:
        w("%-60s %8d  %10.3f GB" % (k, c, b / 1024**3))
    if len(st) > 25:
        w("... (%d more entries)" % (len(st) - 25))

# 5. pycache totals under tests root
w("")
w("## __pycache__ dirs under tests root")
tot_c = tot_b = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    if os.path.basename(dirpath) == "__pycache__":
        for fn in filenames:
            try:
                tot_b += os.stat(os.path.join(dirpath, fn)).st_size
                tot_c += 1
            except OSError:
                pass
w("total: %d files, %.3f GB" % (tot_c, tot_b / 1024**3))

# 6. smoke/ckpt/pkl inside glm5/result campaigns
w("")
w("## glm5/result *.pkl checkpoints & smoke dirs")
pk_c = pk_b = 0
smoke_c = smoke_b = 0
for dirpath, dirnames, filenames in os.walk(GR):
    base = os.path.basename(dirpath)
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            sz = os.stat(p).st_size
        except OSError:
            continue
        if fn.endswith(".pkl"):
            pk_c += 1
            pk_b += sz
        if base == "smoke":
            smoke_c += 1
            smoke_b += sz
w(".pkl: %d files %.3f GB | smoke dirs: %d files %.3f GB" % (pk_c, pk_b / 1024**3, smoke_c, smoke_b / 1024**3))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("scan2 OK")
