# -*- coding: utf-8 -*-
"""
READ-ONLY companion: split the REVIEW bucket, build batch groups,
and emit a per-file DELETE manifest (csv) with size / mtime / top-dir.
Nothing is deleted, moved or renamed.
"""
import os
import csv
import time
import datetime
import collections

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
OUT_SPLIT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_review_split_20261109.txt"
OUT_CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"

REPARSE = 0x400
NOW = time.time()
HOT = NOW - 24 * 3600
RECENT = NOW - 72 * 3600

DELETE_EXT = {
    ".npy", ".npz", ".pkl", ".pickle", ".pt", ".pth", ".bin", ".safetensors",
    ".h5", ".hdf5", ".parquet", ".arrow", ".feather", ".db", ".sqlite",
    ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".mdb",
    ".gz", ".zip", ".tar", ".7z", ".rar", ".bz2", ".xz", ".zst", ".tgz",
    ".tmp", ".part", ".partial", ".temp", ".crdownload",
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe",
    ".memmap", ".mmap", ".raw", ".dat", ".nii", ".msgpack", ".onnx", ".pb",
    ".tflite", ".vhdx", ".conllu", ".conll", ".ann", ".vec", ".model",
    ".index", ".faiss", ".binlog",
}
KEEP_EXT = {
    ".py", ".sh", ".bash", ".ps1", ".bat", ".cmd", ".js", ".jsx", ".mjs",
    ".ts", ".tsx", ".css", ".html", ".htm", ".vue", ".ipynb", ".sql", ".lua",
    ".conf", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".service", ".env",
    ".json", ".jsonl", ".ndjson", ".md", ".markdown", ".txt", ".log", ".out",
    ".err", ".csv", ".tsv", ".pdf", ".svg", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".ico", ".sha256", ".md5", ".fingerprint", ".lease", ".lock",
    ".gitignore", ".manifest", ".ledger", ".patch", ".diff", ".rst", ".tex",
}
DATA_JSON_HINT = (
    "documents", "tokenizer", "corpus", "embedding", "activation", "token",
    "vector", "dataset", "sample", "batch", "hidden", "matrix", "weight",
    "vocab", "cache", "pack", "shard", "chunk", "blob", "dump", "snapshot",
    "atlas", "dictionary", "predictor", "features", "latent", "acts",
    "library", "index", "store", "payload", "records",
)
RESULT_JSON_HINT = (
    "result", "execution", "summary", "report", "analysis", "metrics",
    "ledger", "config", "meta", "stat", "eval", "score", "behavior",
    "answer", "query", "plan", "design", "manifest", "state", "status",
    "acceptance", "fingerprint", "audit", "log", "memo", "note", "index",
)
BIG_JSON = 32 * 1024 * 1024

# fine-grained REVIEW buckets
CORPUS_HINT = (
    "corpus", "material", "cases", "rows", "inputs", "records", "initial",
    "confirmation", "holdout", "controlled-", "documents", "dataset",
    "samples", "prompts", "passport", "full_field", "fullfield", "trajectory",
)
VIS_MARK = "client_visualization_assets" + os.sep + "research_kernel"


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


split = collections.defaultdict(lambda: [0, 0])   # bucket -> [bytes, files]
split_ext = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
split_examples = collections.defaultdict(list)
dirstat = collections.defaultdict(lambda: {
    "p1": [0, 0], "p2": [0, 0], "p3": [0, 0], "review": [0, 0],
    "last": 0.0, "hot": 0, "sha": 0, "res": 0,
})
batch = collections.defaultdict(lambda: [0, 0])

# ---- pass 1: collect per-top-dir metadata first, so batching is order-independent ----
for dp, dns, fns in os.walk(ROOT, topdown=True, followlinks=False):
    kept = []
    for d in dns:
        q = os.path.join(dp, d)
        try:
            if getattr(os.lstat(q), "st_file_attributes", 0) & REPARSE:
                continue
        except OSError:
            continue
        kept.append(d)
    dns[:] = kept
    rd = os.path.relpath(dp, ROOT)
    top = "(root)" if rd == "." else rd.split(os.sep)[0]
    rec = dirstat[top]
    for fn in fns:
        q = os.path.join(dp, fn)
        try:
            st = os.lstat(q)
        except OSError:
            continue
        if getattr(st, "st_file_attributes", 0) & REPARSE:
            continue
        if st.st_mtime > rec["last"]:
            rec["last"] = st.st_mtime
        if st.st_mtime >= HOT:
            rec["hot"] += 1
        if fn.endswith(".sha256"):
            rec["sha"] += 1
        if fn in ("result.json", "execution.json"):
            rec["res"] += 1

csv_path = OUT_CSV
rows_written = 0
with open(csv_path, "w", encoding="utf-8", newline="") as cf:
    cw = csv.writer(cf)
    cw.writerow(["batch", "top_dir", "ext", "size_bytes", "size_human", "mtime", "path"])

    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True, followlinks=False):
        kept = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            try:
                if getattr(os.lstat(p), "st_file_attributes", 0) & REPARSE:
                    continue
            except OSError:
                continue
            kept.append(d)
        dirnames[:] = kept

        rel_dir = os.path.relpath(dirpath, ROOT)
        top = "(root)" if rel_dir == "." else rel_dir.split(os.sep)[0]
        rec = dirstat[top]

        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if getattr(st, "st_file_attributes", 0) & REPARSE:
                continue
            size, mt = st.st_size, st.st_mtime
            ext = os.path.splitext(fn)[1].lower()
            cls = classify(fn, ext, size)
            rel = os.path.relpath(p, ROOT)

            if cls == "REVIEW":
                low = rel.lower()
                if VIS_MARK.lower() in low:
                    bucket = "VIS-ASSET"
                elif any(h in low for h in CORPUS_HINT):
                    bucket = "CORPUS-DATA"
                elif ext in (".json", ".jsonl", ".ndjson"):
                    bucket = "OTHER-BIG-JSON"
                else:
                    bucket = "UNKNOWN-EXT"
                split[bucket][0] += size
                split[bucket][1] += 1
                split_ext[bucket][ext or "(none)"][0] += size
                split_ext[bucket][ext or "(none)"][1] += 1
                rec["review"][0] += size
                rec["review"][1] += 1
                if len(split_examples[bucket]) < 12:
                    split_examples[bucket].append((size, rel))
                continue

            if cls != "DELETE":
                continue

            # batch assignment
            if ext in (".tmp", ".part", ".partial", ".temp", ".pyc", ".pyo",
                       ".crdownload") or fn.lower().endswith(".before-import-20261108105949"):
                b = "P1-junk"
                rec["p1"][0] += size
                rec["p1"][1] += 1
            elif dirstat[top]["hot"] or dirstat[top]["sha"] or mt >= RECENT:
                b = "P3-active"
                rec["p3"][0] += size
                rec["p3"][1] += 1
            else:
                b = "P2-cold"
                rec["p2"][0] += size
                rec["p2"][1] += 1
            batch[b][0] += size
            batch[b][1] += 1
            mts = datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
            cw.writerow([b, top, ext or "(none)", size, human(size), mts, p])
            rows_written += 1

L = []
w = L.append
w("GLM5 RESULT / REVIEW SPLIT + BATCH PLAN (READ-ONLY)  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("TARGET = " + ROOT)
w("")
w("[1] REVIEW BUCKETS (8.13 GB total -> decide per bucket)")
for k in ("VIS-ASSET", "CORPUS-DATA", "OTHER-BIG-JSON", "UNKNOWN-EXT"):
    b, f = split[k]
    w("   %-16s %10s  %6d files" % (k, human(b), f))
    exts = ", ".join("%s=%s" % (e, human(v[0])) for e, v in
                     sorted(split_ext[k].items(), key=lambda kv: -kv[1][0])[:6])
    w("        ext: " + (exts or "-"))
    for size, rel in sorted(split_examples[k], reverse=True)[:6]:
        w("        eg  %10s  %s" % (human(size), rel))
w("")
w("[2] DELETE BATCHES")
for k in ("P1-junk", "P2-cold", "P3-active"):
    b, f = batch[k]
    w("   %-10s %10s  %7d files" % (k, human(b), f))
w("   %-10s %10s  %7d files" % ("TOTAL", human(sum(v[0] for v in batch.values())),
                                  sum(v[1] for v in batch.values())))
w("")
w("[3] TOP DIRS BY P2-cold (safest large-volume batch)")
w("   cols: P2-cold | files | P3-active | P1-junk | REVIEW | last_mtime | name")
for name, r in sorted(dirstat.items(), key=lambda kv: -kv[1]["p2"][0])[:35]:
    if r["p2"][1] == 0:
        continue
    last = datetime.datetime.fromtimestamp(r["last"]).strftime("%Y-%m-%d %H:%M") if r["last"] else "-"
    w("   %10s | %7d | %10s | %9s | %9s | %s | %s" % (
        human(r["p2"][0]), r["p2"][1], human(r["p3"][0]), human(r["p1"][0]),
        human(r["review"][0]), last, name))
w("")
w("[4] TOP DIRS BY P3-active (needs confirmation before cleanup)")
for name, r in sorted(dirstat.items(), key=lambda kv: -kv[1]["p3"][0])[:25]:
    if r["p3"][1] == 0:
        continue
    last = datetime.datetime.fromtimestamp(r["last"]).strftime("%Y-%m-%d %H:%M") if r["last"] else "-"
    w("   %10s | %7d files | hot24h=%5d sha=%2d result/exec=%4d | %s | %s" % (
        human(r["p3"][0]), r["p3"][1], r["hot"], r["sha"], r["res"], last, name))
w("")
w("[5] CSV MANIFEST")
w("   " + OUT_CSV)
w("   rows = %d (excl. header), cols = batch,top_dir,ext,size_bytes,size_human,mtime,path" % rows_written)
w("")
w("=== END (READ-ONLY, nothing deleted) ===")

with open(OUT_SPLIT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

print("OK " + OUT_SPLIT)
print("OK " + OUT_CSV + " rows=" + str(rows_written))
