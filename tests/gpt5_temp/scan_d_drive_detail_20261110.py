# -*- coding: utf-8 -*-
"""Second pass: targeted verification for the new retention policy.

Focus:
  A) ai2050_native (184 GB) - is it intermediate data or client-served results?
  B) distributed_data (103 GB) sub-structure
  C) model weight duplication develop\\model vs models\\hf
  D) client_visualization_assets - what the server actually serves
  E) tests/glm5/result remaining P3 data
  F) release / .git / video
  G) upload / publish scripts and their target directories
"""
import os
import time
from collections import defaultdict
from datetime import datetime

T0 = time.time()
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_detail_20261110.txt"
NATIVE = r"D:\ai2050_native"
PROJ = r"D:\AI2050\Ai2050-OpenOne"
DD = os.path.join(PROJ, "distributed_data")
RES = os.path.join(PROJ, "tests", "glm5", "result")
CLIENT = os.path.join(RES, "client_visualization_assets")

L = []
def w(s=""):
    L.append(str(s))

def gb(n):
    return "%.2f GB" % (n / 1024.0 ** 3)
def mb(n):
    return "%.2f MB" % (n / 1024.0 ** 2)
def ts(t):
    try:
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"

NOW = time.time()
H24 = NOW - 24 * 3600
RATTR = 0x400

DATA_EXT = {".npy", ".npz", ".pkl", ".pickle", ".bin", ".pt", ".pth", ".parquet",
            ".sqlite3", ".sqlite", ".db", ".h5", ".hdf5", ".arrow", ".msgpack",
            ".onnx", ".gz", ".zip", ".tar", ".tgz", ".tmp", ".dat", ".index",
            ".mmap", ".safetensors", ".gguf", ".conllu", ".bz2", ".xz", ".zst"}
WEIGHT_EXT = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}
RESULT_EXT = {".json", ".jsonl", ".md", ".log", ".txt", ".sha256", ".csv", ".tsv",
              ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html", ".yaml", ".yml"}
SCRIPT_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".sh", ".ps1", ".bat", ".css"}

def kind(fn):
    b = fn.lower()
    ext = os.path.splitext(b)[1]
    if ext in WEIGHT_EXT:
        return "WEIGHT"
    if ext in DATA_EXT or b.endswith((".gz.tmp", ".npy.tmp", ".before-import")):
        return "DATA"
    if ext in RESULT_EXT or b.endswith(".sha256"):
        return "RESULT"
    if ext in SCRIPT_EXT:
        return "SCRIPT"
    return "OTHER"

def scan(root, maxdepth=99):
    """return rows of (rel, fn, size, mtime, nlink, kind)"""
    rows = []
    def rec(path, rel, depth):
        try:
            it = os.scandir(path)
        except Exception:
            return
        with it:
            while True:
                try:
                    e = next(it)
                except StopIteration:
                    break
                except Exception:
                    break
                try:
                    st = e.stat(follow_symlinks=False)
                    if getattr(st, "st_file_attributes", 0) & RATTR:
                        continue
                    if e.is_dir(follow_symlinks=False):
                        if depth < maxdepth:
                            rec(e.path, os.path.join(rel, e.name), depth + 1)
                        continue
                    rows.append((rel, e.name, st.st_size, st.st_mtime,
                                 getattr(st, "st_nlink", 1) or 1, kind(e.name)))
                except Exception:
                    continue
    rec(root, "", 0)
    return rows

def rollup(rows, prefix_levels=2):
    d = defaultdict(lambda: {"total": 0, "files": 0, "DATA": 0, "RESULT": 0,
                             "WEIGHT": 0, "SCRIPT": 0, "OTHER": 0,
                             "last": 0, "r24": 0})
    for rel, fn, sz, mt, nl, k in rows:
        parts = rel.split(os.sep) if rel else []
        for n in range(1, prefix_levels + 1):
            if len(parts) >= n:
                key = "\\".join(parts[:n])
                x = d[key]
                x["total"] += sz
                x["files"] += 1
                x[k] += sz
                if mt > x["last"]:
                    x["last"] = mt
                if mt >= H24:
                    x["r24"] += 1
    return d

w("D DRIVE TARGETED VERIFICATION (READ-ONLY)  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# ---------- A) ai2050_native ----------
w("[A] ai2050_native : sub-tree composition (level 1/2, top 30 by total)")
rows = scan(NATIVE)
w("    scanned %s / %d files" % (gb(sum(r[2] for r in rows)), len(rows)))
d = rollup(rows, 2)
w("    %-50s %10s %9s %9s %9s %8s %6s %s" % ("dir", "total", "DATA", "RESULT", "WEIGHT", "files", "24h", "last"))
for k, v in sorted(d.items(), key=lambda kv: -kv[1]["total"])[:30]:
    w("    %-50s %10s %9s %9s %9s %8d %6d %s" % (
        k[:50], gb(v["total"]), gb(v["DATA"]), gb(v["RESULT"]), gb(v["WEIGHT"]),
        v["files"], v["r24"], ts(v["last"])))
w("")
w("    visualization-research sub-experiments (level 2):")
vr = defaultdict(lambda: {"total": 0, "files": 0, "DATA": 0, "RESULT": 0,
                          "WEIGHT": 0, "SCRIPT": 0, "OTHER": 0,
                          "json": 0, "last": 0, "r24": 0})
for rel, fn, sz, mt, nl, k in rows:
    if not rel.lower().startswith("visualization-research\\"):
        continue
    parts = rel.split(os.sep)
    if len(parts) < 2:
        continue
    x = vr[parts[1]]
    x["total"] += sz
    x["files"] += 1
    x[k] += sz
    if fn.lower().endswith(".json"):
        x["json"] += 1
    if mt > x["last"]:
        x["last"] = mt
    if mt >= H24:
        x["r24"] += 1
w("    %-40s %9s %9s %9s %7s %7s %6s %s" % ("experiment", "total", "DATA", "RESULT", "json", "files", "24h", "last"))
for k, v in sorted(vr.items(), key=lambda kv: -kv[1]["total"]):
    w("    %-40s %9s %9s %9s %7d %7d %6d %s" % (
        k[:40], gb(v["total"]), gb(v["DATA"]), gb(v["RESULT"]),
        v["json"], v["files"], v["r24"], ts(v["last"])))
w("")

# ---------- B) distributed_data ----------
w("[B] distributed_data : sub-tree composition (level 1/2, top 25)")
rowsdd = scan(DD)
w("    scanned %s / %d files" % (gb(sum(r[2] for r in rowsdd)), len(rowsdd)))
dd = rollup(rowsdd, 2)
for k, v in sorted(dd.items(), key=lambda kv: -kv[1]["total"])[:25]:
    w("    %-50s %10s %9s %9s %8d %6d %s" % (
        k[:50], gb(v["total"]), gb(v["DATA"]), gb(v["RESULT"]),
        v["files"], v["r24"], ts(v["last"])))
w("")

# ---------- C) model duplication ----------
w("[C] MODEL WEIGHT DUPLICATION: develop\\model  vs  AI2050\\Ai2050-OpenOne\\models")
def model_map(root):
    m = {}
    rowsx = scan(root)
    for rel, fn, sz, mt, nl, k in rowsx:
        if k != "WEIGHT":
            continue
        key = (fn.lower(), sz)
        m[key] = os.path.join(rel, fn)
    return m, rowsx
m1, r1 = model_map(r"D:\develop\model")
m2, r2 = model_map(os.path.join(PROJ, "models"))
common = set(m1) & set(m2)
dup_bytes = sum(k[1] for k in common)
w("    develop\\model   : %s / %d weight files" % (gb(sum(r[2] for r in r1)), len(m1)))
w("    Ai2050\\models   : %s / %d weight files" % (gb(sum(r[2] for r in r2)), len(m2)))
w("    identical (name+size) pairs: %d files / %s  <-- one copy is redundant" % (len(common), gb(dup_bytes)))
w("    pairs sample (top 12 by size):")
for fn, sz in sorted(common, key=lambda x: -x[1])[:12]:
    w("      %9s  %s" % (gb(sz), fn[:70]))
w("")

# ---------- D) client assets ----------
w("[D] CLIENT ASSET ROOT (served by /api/research-assets)")
if os.path.isdir(CLIENT):
    rc = scan(CLIENT)
    dc = rollup(rc, 2)
    w("    total %s / %d files" % (gb(sum(r[2] for r in rc)), len(rc)))
    for k, v in sorted(dc.items(), key=lambda kv: -kv[1]["total"])[:20]:
        w("    %-50s %10s %8d %s" % ((k or "(root)")[:50], gb(v["total"]), v["files"], ts(v["last"])))
    jb = sum(r[2] for r in rc if r[1].lower().endswith(".json"))
    w("    json bytes = %s ; other = %s" % (gb(jb), gb(sum(r[2] for r in rc) - jb)))
else:
    w("    MISSING")
w("")

# ---------- E) tests/glm5/result remaining ----------
w("[E] tests\\glm5\\result remaining (level 1, top 25)")
rr = scan(RES)
dr = rollup(rr, 1)
w("    total %s / %d files" % (gb(sum(r[2] for r in rr)), len(rr)))
for k, v in sorted(dr.items(), key=lambda kv: -kv[1]["total"])[:25]:
    w("    %-50s %10s %9s %9s %8d %6d %s" % (
        k[:50], gb(v["total"]), gb(v["DATA"]), gb(v["RESULT"]),
        v["files"], v["r24"], ts(v["last"])))
w("")

# ---------- F) release / .git / video ----------
w("[F] release (level 1) / .git / video")
for label, path in (("release", os.path.join(PROJ, "release")),
                    ("video", os.path.join(PROJ, "video")),
                    ("frontend", os.path.join(PROJ, "frontend")),
                    (".git", os.path.join(PROJ, ".git"))):
    if not os.path.isdir(path):
        continue
    rx = scan(path)
    dx = rollup(rx, 1)
    w("    --- %s : total %s / %d files ---" % (label, gb(sum(r[2] for r in rx)), len(rx)))
    for k, v in sorted(dx.items(), key=lambda kv: -kv[1]["total"])[:10]:
        w("        %-46s %10s %8d %6d %s" % (
            (k or "(root)")[:46], gb(v["total"]), v["files"], v["r24"], ts(v["last"])))
w("")

# ---------- G) upload / publish chain ----------
w("[G] UPLOAD / PUBLISH SCRIPT CHAIN (scripts whose name suggests delivery)")
hits = []
for base in (os.path.join(PROJ, "tests"), os.path.join(PROJ, "server"),
             os.path.join(PROJ, "deploy"), os.path.join(PROJ, "scripts")):
    if not os.path.isdir(base):
        continue
    for root, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in ("node_modules", "__pycache__", ".git", "result")]
        for fn in files:
            low = fn.lower()
            if not low.endswith((".py", ".sh", ".ps1", ".js", ".ts", ".md")):
                continue
            if any(t in low for t in ("upload", "sync", "publish", "deliver", "deploy", "export_client", "vis_data_export", "asset")):
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                hits.append((st.st_mtime, p, st.st_size))
w("    matched scripts: %d ; most recent 25:" % len(hits))
for mt, p, sz in sorted(hits, reverse=True)[:25]:
    w("      %s %8s %s" % (ts(mt), ("%.0fKB" % (sz / 1024.0)), p.replace(PROJ + "\\", "")[:110]))
w("")

w("=== END (READ-ONLY) ===")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("done", len(L), "lines", round(time.time() - T0, 1), "s")
