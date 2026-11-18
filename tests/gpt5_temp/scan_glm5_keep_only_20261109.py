# -*- coding: utf-8 -*-
"""
READ-ONLY classifier for tests/glm5/result
Policy given by user: keep ONLY scripts and results; data files are cleanup candidates.
No file is deleted / moved / renamed. Only reports + path lists are written.
"""
import os
import time
import shutil
import datetime
import collections

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
OUT_TXT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_keep_only_20261109.txt"
OUT_PATHS = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_paths_20261109.txt"

REPARSE = 0x400
NOW = time.time()
HOT_WINDOW = NOW - 24 * 3600

# ---- classification tables -------------------------------------------------
# DATA: binary / packed / intermediate numeric payloads -> cleanup candidates
DELETE_EXT = {
    ".npy", ".npz", ".pkl", ".pickle", ".pt", ".pth", ".bin", ".safetensors",
    ".h5", ".hdf5", ".parquet", ".arrow", ".feather", ".db", ".sqlite",
    ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".mdb",
    ".gz", ".zip", ".tar", ".7z", ".rar", ".bz2", ".xz", ".zst", ".tgz",
    ".tmp", ".part", ".partial", ".temp", ".crdownload",
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe",
    ".memmap", ".mmap", ".raw", ".dat", ".nii", ".msgpack", ".onnx", ".pb",
    ".tflite", ".vhdx", ".conllu", ".conll", ".ann", ".vec", ".model",
    ".npy.tmp", ".npz.tmp", ".index", ".faiss", ".binlog",
}

# SCRIPT + RESULT: source code, config, conclusion-level evidence, docs, figures
KEEP_EXT = {
    ".py", ".sh", ".bash", ".ps1", ".bat", ".cmd", ".js", ".jsx", ".mjs",
    ".ts", ".tsx", ".css", ".html", ".htm", ".vue", ".ipynb", ".sql", ".lua",
    ".conf", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".service", ".env",
    ".json", ".jsonl", ".ndjson", ".md", ".markdown", ".txt", ".log", ".out",
    ".err", ".csv", ".tsv", ".pdf", ".svg", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".ico", ".sha256", ".md5", ".fingerprint", ".lease", ".lock",
    ".gitignore", ".manifest", ".ledger", ".patch", ".diff", ".rst", ".tex",
}

# json/jsonl that look like data payloads rather than conclusions -> REVIEW
DATA_JSON_HINT = (
    "documents", "tokenizer", "corpus", "embedding", "activation", "token",
    "vector", "dataset", "sample", "batch", "hidden", "matrix", "weight",
    "vocab", "cache", "pack", "shard", "chunk", "blob", "dump", "snapshot",
    "atlas", "dictionary", "predictor", "features", "latent", "acts", "cache",
    "library", "index", "store", "payload", "records",
)
RESULT_JSON_HINT = (
    "result", "execution", "summary", "report", "analysis", "metrics",
    "ledger", "config", "meta", "stat", "eval", "score", "behavior",
    "answer", "query", "plan", "design", "manifest", "state", "status",
    "acceptance", "fingerprint", "audit", "log", "memo", "note", "index",
)
BIG_JSON = 32 * 1024 * 1024  # 32 MB


def classify(name, ext, size):
    low = name.lower()
    if ext in DELETE_EXT or low.endswith(".before-import-20261108105949"):
        return "DELETE"
    if ext in (".json", ".jsonl", ".ndjson"):
        if any(h in low for h in DATA_JSON_HINT) and not any(h in low for h in RESULT_JSON_HINT):
            return "REVIEW"
        if size >= BIG_JSON and not any(h in low for h in RESULT_JSON_HINT):
            return "REVIEW"
        return "KEEP"
    if ext in KEEP_EXT:
        return "KEEP"
    return "REVIEW"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


# ---- scan -------------------------------------------------------------------
tot = collections.defaultdict(lambda: [0, 0])          # cls -> [bytes, files]
by_ext = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
by_dir = collections.defaultdict(lambda: {
    "delete": [0, 0], "keep": [0, 0], "review": [0, 0],
    "last": 0.0, "hot": 0, "sha": 0, "resjson": 0,
    "ext": collections.defaultdict(lambda: [0, 0]),
})
reviews = []      # (size, relpath, reason)
big_dels = []     # (size, relpath)
del_paths = []    # full paths of DELETE files
pruned = []
t0 = time.time()

for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True, followlinks=False):
    # prune reparse-point dirs (junctions/symlinks) to avoid double counting
    kept = []
    for d in dirnames:
        p = os.path.join(dirpath, d)
        try:
            st = os.lstat(p)
            if getattr(st, "st_file_attributes", 0) & REPARSE:
                pruned.append(p)
                continue
        except OSError:
            continue
        kept.append(d)
    dirnames[:] = kept

    rel_dir = os.path.relpath(dirpath, ROOT)
    top = "(root)" if rel_dir == "." else rel_dir.split(os.sep)[0]
    rec = by_dir[top]

    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if getattr(st, "st_file_attributes", 0) & REPARSE:
            continue
        size = st.st_size
        mt = st.st_mtime
        ext = os.path.splitext(fn)[1].lower()
        cls = classify(fn, ext, size)

        tot[cls][0] += size
        tot[cls][1] += 1
        by_ext[ext][cls][0] += size
        by_ext[ext][cls][1] += 1
        rec[cls.lower()][0] += size
        rec[cls.lower()][1] += 1
        rec["ext"][ext][0] += size
        rec["ext"][ext][1] += 1
        if mt > rec["last"]:
            rec["last"] = mt
        if mt >= HOT_WINDOW:
            rec["hot"] += 1
        if fn.endswith(".sha256"):
            rec["sha"] += 1
        if fn in ("result.json", "execution.json"):
            rec["resjson"] += 1

        rel = os.path.relpath(p, ROOT)
        if cls == "DELETE":
            del_paths.append(p)
            if size >= 200 * 1024 * 1024:
                big_dels.append((size, rel))
        elif cls == "REVIEW":
            reason = "data-like JSON" if ext in (".json", ".jsonl", ".ndjson") else "unknown ext " + (ext or "(none)")
            reviews.append((size, rel, reason))

scan_sec = time.time() - t0
reviews.sort(reverse=True)
big_dels.sort(reverse=True)

# ---- report -----------------------------------------------------------------
L = []
w = L.append
w("GLM5 RESULT / KEEP-ONLY-SCRIPT-AND-RESULT POLICY (READ-ONLY)   " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("TARGET = " + ROOT)
for drv in ("C:\\", "D:\\"):
    try:
        u = shutil.disk_usage(drv)
        w("[DISK] %s total=%s used=%s free=%s" % (drv, human(u.total), human(u.used), human(u.free)))
    except OSError:
        pass
w("[SCAN] %.1f sec, pruned reparse dirs=%d" % (scan_sec, len(pruned)))
w("")
w("POLICY: KEEP = scripts (.py/.sh/.js/.css/.html/...), configs, conclusion files (.json/.jsonl/.md/.txt/.log/.csv/.sha256/...), figures, docs")
w("        DELETE = numeric/binary/packed data (.npy/.npz/.pkl/.safetensors/.gz/.zip/.sqlite3/.pyc/.tmp/...)")
w("        REVIEW = large or data-like JSON, unknown extensions -> needs your decision")
w("")
w("[1] LEDGER")
for cls in ("DELETE", "KEEP", "REVIEW"):
    b, f = tot[cls]
    w("   %-7s %12s  %8d files" % (cls, human(b), f))
total_b = sum(v[0] for v in tot.values())
total_f = sum(v[1] for v in tot.values())
w("   %-7s %12s  %8d files" % ("TOTAL", human(total_b), total_f))
w("   DELETE share = %.2f%% bytes / %.2f%% files" % (100.0 * tot["DELETE"][0] / total_b, 100.0 * tot["DELETE"][1] / total_f))
w("")
w("[2] BY EXTENSION  (cols: total | files | decision | bytes in that decision)")
for ext, d in sorted(by_ext.items(), key=lambda kv: -sum(v[0] for v in kv[1].values())):
    tb = sum(v[0] for v in d.values())
    tf = sum(v[1] for v in d.values())
    parts = []
    for cls in ("DELETE", "KEEP", "REVIEW"):
        if d[cls][1]:
            parts.append("%s=%s/%d" % (cls, human(d[cls][0]), d[cls][1]))
    w("   %-14s %12s %8d  |  %s" % (ext or "(none)", human(tb), tf, "  ".join(parts)))
w("")
w("[3] BY TOP-LEVEL DIR  (size desc by DELETE bytes)")
w("   cols: DELETE | files | KEEP | REVIEW | last_mtime | hot24h | .sha256 | result/exec.json | name")
rows = sorted(by_dir.items(), key=lambda kv: -kv[1]["delete"][0])
for name, r in rows:
    if r["delete"][1] == 0 and r["review"][1] == 0:
        continue
    last = datetime.datetime.fromtimestamp(r["last"]).strftime("%Y-%m-%d %H:%M") if r["last"] else "-"
    w("   %11s | %7d | %9s | %9s | %s | %6d | %4d | %4d | %s" % (
        human(r["delete"][0]), r["delete"][1], human(r["keep"][0]), human(r["review"][0]),
        last, r["hot"], r["sha"], r["resjson"], name))
w("")
w("[4] REVIEW ITEMS  (top 60 by size; decide KEEP or DELETE)")
w("   %11s  %-28s  %s" % ("size", "reason", "relative path"))
for size, rel, reason in reviews[:60]:
    w("   %11s  %-28s  %s" % (human(size), reason[:28], rel))
w("   ... REVIEW total = %d files / %s" % (len(reviews), human(tot["REVIEW"][0])))
w("")
w("[5] EXCEPTIONS / RISK FLAGS  (dirs with 24h writes or .sha256 registration)")
for name, r in rows:
    if r["delete"][0] < 1024 * 1024:
        continue
    if r["hot"] or r["sha"] or r["resjson"]:
        w("   %-52s DEL=%10s  hot24h=%6d  .sha256=%3d  result/exec.json=%4d  last=%s" % (
            name, human(r["delete"][0]), r["hot"], r["sha"], r["resjson"],
            datetime.datetime.fromtimestamp(r["last"]).strftime("%Y-%m-%d %H:%M") if r["last"] else "-"))
w("")
w("[6] LARGEST SINGLE DELETE FILES (top 40, >=200MB)")
for size, rel in big_dels[:40]:
    w("   %11s  %s" % (human(size), rel))
w("")
w("[7] OUTPUT")
w("   Full DELETE path list: " + OUT_PATHS)
w("   lines = %d" % len(del_paths))
w("")
w("=== END (READ-ONLY, nothing deleted) ===")

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
with open(OUT_PATHS, "w", encoding="utf-8") as fh:
    fh.write("\n".join(del_paths) + ("\n" if del_paths else ""))

print("OK report=" + OUT_TXT)
print("OK paths=" + OUT_PATHS + " lines=" + str(len(del_paths)))
