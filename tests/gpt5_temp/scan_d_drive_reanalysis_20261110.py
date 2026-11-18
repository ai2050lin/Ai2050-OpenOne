# -*- coding: utf-8 -*-
"""D drive full read-only reanalysis under the new retention policy.

Policy (user, 2026-11-10):
  KEEP   test scripts, test materials, test results (reproducible),
         important results uploaded to server / loadable by client
  DELETE intermediate big-data folders produced during experiments

Special care: hard links (st_nlink) must be checked, otherwise released
space is grossly overestimated (2026-11-09 lesson).
"""
import os
import sys
import time
import csv
from collections import defaultdict
from datetime import datetime, timedelta

T0 = time.time()
ROOT = "D:\\"
BASE = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(BASE, "tests", "gpt5", "result", "d_drive_reanalysis_20261110.txt")
CSV = os.path.join(BASE, "tests", "gpt5", "result", "d_drive_cleanup_candidates_20261110.csv")

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
H72 = NOW - 72 * 3600

# ---------------- classification ----------------
CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".sh",
            ".ps1", ".bat", ".cmd", ".css", ".html", ".htm", ".ipynb", ".sql",
            ".c", ".h", ".cpp", ".rs", ".go", ".java", ".rb", ".php"}
CFG_EXT = {".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env", ".gitignore",
           ".lock", ".editorconfig", ".dockerfile", ".proto", ".gradle", ".cmake"}
DOC_EXT = {".md", ".log", ".txt", ".sha256", ".csv", ".tsv", ".pdf", ".png",
           ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".json", ".jsonl", ".ndjson"}
INTERMEDIATE_EXT = {".npy", ".npz", ".pkl", ".pickle", ".safetensors", ".bin", ".pt",
                    ".pth", ".gguf", ".parquet", ".conllu", ".sqlite3", ".sqlite",
                    ".db", ".h5", ".hdf5", ".arrow", ".msgpack", ".onnx", ".gz",
                    ".zip", ".tar", ".tgz", ".tmp", ".pyc", ".cache", ".dat",
                    ".index", ".mmap", ".nib", ".npy.tmp", ".7z", ".rar", ".whl",
                    ".bz2", ".xz", ".zst", ".blob", ".pack", ".nchw", ".safetensors.tmp"}
RESULT_NAMES = {"result.json", "execution.json", "behavior.json", "summary.json",
                "manifest.json", "report.json", "metrics.json", "ledger.json"}
MATERIAL_HINT = ("corpus", "cases", "material", "prompt", "holdout", "rows",
                 "dataset", "input", "fixture", "sample", "stimuli", "tokenizer",
                 "vocab", "template", "query", "seed")
CLIENT_HINT = ("client_visualization_assets", "research_kernel", "frontend\\public\\data",
               "frontend\\dist", "dist-installer", "uploads", "\\assets\\data")
SERVER_HINT = ("server\\research_kernel", "server\\research_asset", "datapack",
               "research_library")

def norm(p):
    return p.lower().replace("/", "\\")

def classify(rel, fn, ext, size):
    p = norm(rel)
    b = fn.lower()

    # path-first rules
    if "$recycle.bin" in p or "360recyclebin" in p:
        return "RECYCLE-BIN"
    if "\\.git\\" in p or p.startswith(".git\\"):
        return "GIT"
    if "research_native_blobs" in p:
        return "CAS-BLOB"
    if p.startswith("develop\\model\\") or p.startswith("ai2050\\ai2050-openone\\models\\") \
       or ".cache\\huggingface" in p:
        return "MODEL-WEIGHT"
    if p.startswith("ai2050_native\\"):
        return "NATIVE-RUNTIME"
    if p.startswith("ai2050\\ai2050-openone\\release\\") or "node_modules" in p \
       or "dist-installer" in p or "\\frontend\\dist\\" in p:
        return "BUILD-ARTIFACT"
    if p.startswith("ai2050\\ai2050-openone\\video\\") or p.startswith("video\\"):
        return "MEDIA"
    if "pagefile.sys" in b or "hiberfil" in b or "swapfile" in b:
        return "SYSTEM"

    # client / server published assets are KEEP
    if any(h in p for h in CLIENT_HINT):
        if ext in INTERMEDIATE_EXT:
            return "CLIENT-ASSET-BINARY"
        return "KEEP-CLIENT"
    if any(h in p for h in SERVER_HINT) and ext in DOC_EXT:
        return "KEEP-CLIENT"

    # extension rules
    if ext in INTERMEDIATE_EXT or b.endswith((".gz.tmp", ".npy.tmp", ".npz.tmp", ".before-import")):
        return "DELETE-INTERMEDIATE"
    if ext in CODE_EXT or ext in CFG_EXT or fn in ("LICENSE", "VERSION", "Makefile", "Dockerfile"):
        return "KEEP-SCRIPT"
    if ext == ".json" or ext in (".jsonl", ".ndjson"):
        if b in RESULT_NAMES or b.endswith(".sha256"):
            return "KEEP-RESULT"
        if any(h in b for h in MATERIAL_HINT):
            return "KEEP-MATERIAL"
        return "KEEP-RESULT"
    if ext in DOC_EXT or b.endswith(".sha256"):
        return "KEEP-RESULT"
    return "OTHER"

# ---------------- scan ----------------
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

cat = defaultdict(lambda: {"bytes": 0, "files": 0, "solo_bytes": 0, "solo_files": 0,
                           "link_bytes": 0, "link_files": 0, "recent24": 0, "recent72": 0})
dirs = defaultdict(lambda: {"bytes": 0, "files": 0, "solo_bytes": 0, "last": 0, "recent24": 0})
byext = defaultdict(lambda: {"bytes": 0, "files": 0, "solo_bytes": 0})
bigfiles = []
recent_big = []
cat_dir = defaultdict(lambda: {"bytes": 0, "files": 0, "solo_bytes": 0})
errors = 0
pruned = 0
total_files = 0
total_bytes = 0
solo_bytes = 0

def is_reparse(entry):
    try:
        st = entry.stat(follow_symlinks=False)
        return bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)
    except Exception:
        return True

def walk(path, depth, rel):
    global errors, pruned, total_files, total_bytes, solo_bytes
    try:
        it = os.scandir(path)
    except Exception:
        errors += 1
        return
    with it:
        while True:
            try:
                e = next(it)
            except StopIteration:
                break
            except Exception:
                errors += 1
                break
            try:
                if is_reparse(e):
                    pruned += 1
                    continue
                if e.is_dir(follow_symlinks=False):
                    walk(e.path, depth + 1, os.path.join(rel, e.name))
                    continue
                st = e.stat(follow_symlinks=False)
                sz = st.st_size
                mt = st.st_mtime
                nl = getattr(st, "st_nlink", 1) or 1
                fn = e.name
                ext = os.path.splitext(fn)[1].lower()
                c = classify(rel, fn, ext, sz)
            except Exception:
                errors += 1
                continue

            total_files += 1
            total_bytes += sz
            if nl <= 1:
                solo_bytes += sz

            d = cat[c]
            d["bytes"] += sz
            d["files"] += 1
            if nl <= 1:
                d["solo_bytes"] += sz
                d["solo_files"] += 1
            else:
                d["link_bytes"] += sz
                d["link_files"] += 1
            if mt >= H24:
                d["recent24"] += 1
            if mt >= H72:
                d["recent72"] += 1

            x = byext[ext or "(none)"]
            x["bytes"] += sz
            x["files"] += 1
            if nl <= 1:
                x["solo_bytes"] += sz

            # directory rollups at depth 1..3 (relative to D:\)
            parts = rel.split(os.sep) if rel else []
            for k in (1, 2, 3):
                if len(parts) >= k:
                    key = "\\".join(parts[:k])
                    dd = dirs[key]
                    dd["bytes"] += sz
                    dd["files"] += 1
                    if nl <= 1:
                        dd["solo_bytes"] += sz
                    if mt > dd["last"]:
                        dd["last"] = mt
                    if mt >= H24:
                        dd["recent24"] += 1

            # category x top-level-dir rollup
            top = parts[0] if parts else "(root)"
            cd = cat_dir[(c, top)]
            cd["bytes"] += sz
            cd["files"] += 1
            if nl <= 1:
                cd["solo_bytes"] += sz

            if sz >= 200 * 1024 ** 2:
                bigfiles.append((sz, fn, rel, c, nl, mt))
            if mt >= H72 and sz >= 50 * 1024 ** 2:
                recent_big.append((sz, fn, rel, c, nl, mt))

walk(ROOT, 0, "")

# ---------------- disk ----------------
import ctypes
free_bytes = ctypes.c_ulonglong(0)
total_b = ctypes.c_ulonglong(0)
ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p("D:\\"),
                                          ctypes.byref(free_bytes),
                                          ctypes.byref(total_b), None)
cfree = ctypes.c_ulonglong(0)
ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p("C:\\"),
                                          ctypes.byref(cfree), None, None)

w("D DRIVE RE-ANALYSIS (READ-ONLY)  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("POLICY: keep scripts + materials + results (reproducible) + server/client assets;")
w("        delete intermediate big-data produced during experiments.")
w("NOTE: solo = st_nlink==1 (exclusive, deleting really frees space);")
w("      link = st_nlink>=2 (hard link / CAS view, deleting frees nothing).")
w("")
w("[1] DISK")
w("    D: total=%s  free=%s  (%.1f%% used)" % (
    gb(total_b.value), gb(free_bytes.value), 100.0 * (1 - free_bytes.value / max(total_b.value, 1))))
w("    C: free=%s" % gb(cfree.value))
w("    scanned: %s / %d files / %d dirs / pruned_reparse=%d / errors=%d / %.1fs" % (
    gb(total_bytes), total_files, len(dirs), pruned, errors, time.time() - T0))
w("    exclusive(solo) bytes = %s ; hard-link-view bytes = %s" % (
    gb(solo_bytes), gb(total_bytes - solo_bytes)))
w("")

w("[2] LEVEL-1 / LEVEL-2 / LEVEL-3 DIRECTORIES (top 45 by bytes)")
w("    %-62s %12s %12s %9s %8s %s" % ("path", "total", "solo(freeable)", "files", "24h", "last_write"))
dl = sorted(dirs.items(), key=lambda kv: -kv[1]["bytes"])[:45]
for k, v in dl:
    w("    %-62s %12s %12s %9d %8d %s" % (
        k[:62], gb(v["bytes"]), gb(v["solo_bytes"]), v["files"], v["recent24"], ts(v["last"])))
w("")

w("[3] CATEGORY TOTALS (new policy)")
w("    %-24s %12s %12s %12s %10s %8s %8s" % (
    "category", "total", "solo(freeable)", "link(views)", "files", "24h", "72h"))
for c in sorted(cat, key=lambda c: -cat[c]["solo_bytes"]):
    d = cat[c]
    w("    %-24s %12s %12s %12s %10d %8d %8d" % (
        c, gb(d["bytes"]), gb(d["solo_bytes"]), gb(d["link_bytes"]),
        d["files"], d["recent24"], d["recent72"]))
w("")

w("[4] CATEGORY x TOP-LEVEL DIR (top 40 by solo bytes)")
w("    %-24s %-26s %12s %12s %9s" % ("category", "top-dir", "total", "solo", "files"))
cl = sorted(cat_dir.items(), key=lambda kv: -kv[1]["solo_bytes"])[:40]
for (c, top), v in cl:
    w("    %-24s %-26s %12s %12s %9d" % (
        c, top[:26], gb(v["bytes"]), gb(v["solo_bytes"]), v["files"]))
w("")

w("[5] EXTENSION TOTALS (top 30 by bytes)")
w("    %-14s %12s %12s %10s" % ("ext", "total", "solo(freeable)", "files"))
for ext, v in sorted(byext.items(), key=lambda kv: -kv[1]["bytes"])[:30]:
    w("    %-14s %12s %12s %10d" % (ext, gb(v["bytes"]), gb(v["solo_bytes"]), v["files"]))
w("")

w("[6] BIG FILES >=200MB (top 30)")
w("    %10s %-6s %-3s %-22s %s" % ("size", "cat", "nlnk", "mtime", "path"))
for sz, fn, rel, c, nl, mt in sorted(bigfiles, reverse=True)[:30]:
    w("    %10s %-6s %-3d %-22s %s" % (
        gb(sz), c[:6], nl, ts(mt), os.path.join(rel, fn)[-110:]))
w("")

w("[7] RECENT BIG WRITES (72h, >=50MB, top 25)")
for sz, fn, rel, c, nl, mt in sorted(recent_big, key=lambda x: -x[5])[:25]:
    w("    %10s %-22s %-22s %s" % (
        gb(sz), ts(mt), c[:22], os.path.join(rel, fn)[-100:]))
w("")

# ---------------- cleanup candidates ----------------
w("[8] CLEANUP CANDIDATES (directories, by freeable bytes)")
# rollup per category at depth<=3 dirs: recompute using a second lightweight pass
cand = defaultdict(lambda: {"bytes": 0, "solo": 0, "files": 0, "last": 0, "recent24": 0})
def walk2(path, depth, rel):
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
                if is_reparse(e):
                    continue
                if e.is_dir(follow_symlinks=False):
                    walk2(e.path, depth + 1, os.path.join(rel, e.name))
                    continue
                st = e.stat(follow_symlinks=False)
                sz, mt, nl = st.st_size, st.st_mtime, (getattr(st, "st_nlink", 1) or 1)
                ext = os.path.splitext(e.name)[1].lower()
                c = classify(rel, e.name, ext, sz)
            except Exception:
                continue
            if c not in ("DELETE-INTERMEDIATE", "CAS-BLOB", "CLIENT-ASSET-BINARY",
                         "BUILD-ARTIFACT", "RECYCLE-BIN", "MODEL-WEIGHT", "NATIVE-RUNTIME"):
                continue
            parts = rel.split(os.sep) if rel else []
            for k in (2, 3, 4):
                if len(parts) >= k:
                    key = (c, "\\".join(parts[:k]))
                    d = cand[key]
                    d["bytes"] += sz
                    d["files"] += 1
                    if nl <= 1:
                        d["solo"] += sz
                    if mt > d["last"]:
                        d["last"] = mt
                    if mt >= H24:
                        d["recent24"] += 1

walk2(ROOT, 0, "")

# write CSV of candidate dirs (depth 3 rows only, to avoid double counting)
with open(CSV, "w", encoding="utf-8", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["category", "dir_depth3_relative_to_D", "total_bytes", "freeable_bytes",
                 "files", "last_write", "writes_24h"])
    for (c, key), v in sorted(cand.items(), key=lambda kv: -kv[1]["solo"]):
        if key.count("\\") != 2:
            continue
        cw.writerow([c, key, v["bytes"], v["solo"], v["files"], ts(v["last"]), v["recent24"]])

w("    %-22s %-52s %11s %11s %8s %6s %s" % (
    "category", "dir (depth3)", "total", "freeable", "files", "24h", "last_write"))
for (c, key), v in sorted(cand.items(), key=lambda kv: -kv[1]["solo"])[:55]:
    if key.count("\\") != 2:
        continue
    w("    %-22s %-52s %11s %11s %8d %6d %s" % (
        c, key[:52], gb(v["bytes"]), gb(v["solo"]), v["files"], v["recent24"], ts(v["last"])))
w("    full CSV: " + CSV)
w("")

w("[9] 24h ACTIVE DIRECTORIES (any category, top 25 by written-file count)")
active = [(v["recent24"], k, v) for k, v in dirs.items() if v["recent24"] > 0]
for n, k, v in sorted(active, reverse=True)[:25]:
    w("    %8d writes  %-58s %10s  last=%s" % (n, k[:58], gb(v["bytes"]), ts(v["last"])))
w("")

w("=== END (READ-ONLY, nothing deleted/modified) ===")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("done", len(L), "lines", round(time.time() - T0, 1), "s")
