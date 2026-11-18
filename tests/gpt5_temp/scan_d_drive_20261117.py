# -*- coding: utf-8 -*-
"""READ-ONLY full D: drive re-analysis, 2026-11-17.

Policy: keep scripts + test inputs + conclusion results + client assets;
clean intermediate big data. Tracks the 131 GB rebound since 11-13.
Outputs:
  d_drive_scan_20261117.txt        decision data
  d_drive_cleanup_candidates_20261117.csv
  d_drive_bigfiles_20261117.csv
"""
import os
import csv
import time
import json
import ctypes
from collections import defaultdict
from datetime import datetime

PROJ = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(PROJ, "tests", "gpt5", "result")
REPORT = os.path.join(OUT, "d_drive_scan_20261117.txt")
CAND = os.path.join(OUT, "d_drive_cleanup_candidates_20261117.csv")
BIG = os.path.join(OUT, "d_drive_bigfiles_20261117.csv")

T0 = time.mktime(datetime(2026, 11, 13, 9, 45).timetuple())  # last cleanup finished
H24 = time.time() - 24 * 3600
H72 = time.time() - 72 * 3600

REPARSE = 0x400
GFA = ctypes.windll.kernel32.GetFileAttributesW
GFA.restype = ctypes.c_uint32
GFA.argtypes = [ctypes.c_wchar_p]
GCFS = ctypes.windll.kernel32.GetCompressedFileSizeW
GCFS.restype = ctypes.c_uint32
GCFS.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]


def is_reparse(p):
    a = GFA(p)
    return a != 0xFFFFFFFF and bool(a & REPARSE)


def alloc(p):
    hi = ctypes.c_uint32(0)
    lo = GCFS(p, ctypes.byref(hi))
    return None if lo == 0xFFFFFFFF else (hi.value << 32) | lo


def free_gb(d):
    fb = ctypes.c_ulonglong(0)
    tb = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(d),
                                               ctypes.byref(fb), ctypes.byref(tb), None)
    return fb.value / 1024 ** 3, tb.value / 1024 ** 3


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def ts(x):
    return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else "-"


KEEP_EXT = {".py", ".sh", ".ps1", ".bat", ".cmd", ".js", ".jsx", ".ts", ".tsx",
            ".css", ".html", ".htm", ".vue", ".json", ".jsonl", ".md", ".txt",
            ".log", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".sha256", ".gitignore", ".mdx", ".rst", ".svg", ".png", ".jpg",
            ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".docx", ".pptx", ".xlsx",
            ".lock", ".map", ".sql", ".proto", ".graphql", ".env", ".example",
            ".pyi", ".h", ".c", ".cc", ".cpp", ".mjs", ".cjs", ".mts", ".cts"}
DATA_EXT = {".npy", ".npz", ".pkl", ".pickle", ".safetensors", ".bin", ".pt", ".pth",
            ".ckpt", ".h5", ".hdf5", ".parquet", ".arrow", ".feather", ".sqlite3",
            ".sqlite", ".db", ".gz", ".tgz", ".tar", ".zip", ".7z", ".rar", ".bz2",
            ".xz", ".zst", ".tmp", ".part", ".conllu", ".onnx", ".gguf", ".model",
            ".pyc", ".pyo", ".so", ".dll", ".exe", ".pdb", ".whl", ".vhdx", ".img",
            ".iso", ".mp4", ".avi", ".mov", ".wav", ".mp3", ".nii", ".mat", ".idx",
            ".pack", ".asar", ".pak", ".node", ".pyd", ".ttf", ".ttc", ".wasm", ".out"}
DATA_SUFFIX = (".gz.tmp", ".npy.tmp", ".npz.tmp", ".before-import", ".partial", ".old")


def cls_of(fn):
    low = fn.lower()
    ext = os.path.splitext(low)[1]
    if ext in KEEP_EXT:
        return "KEEP"
    if ext in DATA_EXT or low.endswith(DATA_SUFFIX):
        return "DATA"
    return "OTHER"


def scan(root, label, level=2):
    """returns dict of relative-directory -> stats, plus big files, junctions, hardlinks."""
    agg = defaultdict(lambda: {"total": 0, "files": 0, "KEEP": 0, "DATA": 0,
                               "OTHER": 0, "r24": 0, "r72": 0, "new": 0,
                               "new_bytes": 0, "last": 0, "nlink2": 0})
    big = []
    junctions = []
    nlink2_total = 0
    nlink2_files = 0
    for dp, dns, fns in os.walk(root, topdown=True):
        keepd = []
        for d in dns:
            q = os.path.join(dp, d)
            if is_reparse(q):
                junctions.append(os.path.relpath(q, root))
            else:
                keepd.append(d)
        dns[:] = keepd
        rel = os.path.relpath(dp, root)
        parts = [] if rel == "." else rel.split(os.sep)
        key = os.sep.join(parts[:level]) if parts else "."
        for fn in fns:
            q = os.path.join(dp, fn)
            try:
                st = os.lstat(q)
            except OSError:
                continue
            if is_reparse(q):
                junctions.append(os.path.relpath(q, root))
                continue
            a = alloc(q) or st.st_size
            c = cls_of(fn)
            for k in {key, rel}:
                d = agg[k]
                d["total"] += a
                d["files"] += 1
                d[c] += a
                if st.st_mtime > d["last"]:
                    d["last"] = st.st_mtime
                if st.st_mtime >= H24:
                    d["r24"] += 1
                if st.st_mtime >= H72:
                    d["r72"] += 1
                if st.st_mtime >= T0:
                    d["new"] += 1
                    d["new_bytes"] += a
            n = getattr(st, "st_nlink", 1)
            if n >= 2:
                nlink2_total += a
                nlink2_files += 1
                agg[key]["nlink2"] += a
            if a >= 100 * 1024 ** 2:
                big.append((a, os.path.relpath(q, root), c, st.st_mtime, n))
    return agg, big, junctions, nlink2_total, nlink2_files


L = []
w = L.append
w("D: DRIVE RE-ANALYSIS / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("baseline: 11-13 09:45 A-tier cleanup finished; D free 151.08 GB")
w("")
cf, ct = free_gb("C:\\")
df, dt = free_gb("D:\\")
w("[0] DISK")
w("    C: free %.2f / %.2f GB" % (cf, ct))
w("    D: free %.2f / %.2f GB   (delta since 11-13 = %.2f GB)" % (df, dt, df - 151.08))
w("")

SCANS = [
    ("D:\\AI2050", r"D:\AI2050", 2),
    ("D:\\ai2050_native", r"D:\ai2050_native", 2),
    ("D:\\develop", r"D:\develop", 2),
    ("PROJECT", PROJ, 2),
]

all_big = []
all_junc = []
results = {}
for name, root, lv in SCANS:
    if not os.path.isdir(root):
        continue
    agg, big, junc, nl2, nlf = scan(root, name, lv)
    results[name] = (root, agg)
    all_big += [(a, name + "\\" + p, c, mt, n) for a, p, c, mt, n in big]
    all_junc += [name + "\\" + j for j in junc]
    w("[%s] %s  (level<=%d, junctions skipped)" % (name, root, lv))
    w("    %-58s %11s %8s %11s %11s %7s %7s %9s %s" %
      ("dir", "total", "files", "DATA", "KEEP", "r24", "r72", "new(>11-13)", "last_write"))
    top = sorted([(k, v) for k, v in agg.items() if k.count(os.sep) < lv],
                 key=lambda kv: -kv[1]["total"])[:32]
    for k, v in top:
        w("    %-58s %11s %8d %11s %11s %7d %7d %9s %s" %
          (k[:58], human(v["total"]), v["files"], human(v["DATA"]),
           human(v["KEEP"]), v["r24"], v["r72"], human(v["new_bytes"]), ts(v["last"])))
    w("    hardlink(nlink>=2) bytes = %s / %d files" % (human(nl2), nlf))
    w("")

# ---- project level-2 DATA hotspots ----
w("[5] PROJECT LEVEL-2 DATA HOTSPOTS (DATA >= 1 GB, sorted)")
root, agg = results["PROJECT"]
hot = sorted([(k, v) for k, v in agg.items() if v["DATA"] >= 1024 ** 3 and k.count(os.sep) == 1],
             key=lambda kv: -kv[1]["DATA"])
w("    %-60s %11s %11s %7s %7s %9s %s" %
  ("dir", "DATA", "KEEP", "r24", "r72", "new", "last"))
for k, v in hot:
    w("    %-60s %11s %11s %7d %7d %9s %s" %
      (k[:60], human(v["DATA"]), human(v["KEEP"]), v["r24"], v["r72"],
       human(v["new_bytes"]), ts(v["last"])))
w("    hotspots = %d ; total DATA = %s" %
  (len(hot), human(sum(v["DATA"] for k, v in hot))))
w("")

# ---- new writes since 11-13 (the 131 GB question) ----
w("[6] NEW WRITES SINCE 11-13 09:45 (by project level-2 dir, top 30)")
new = sorted([(k, v) for k, v in agg.items() if v["new_bytes"] > 0 and k.count(os.sep) == 1],
             key=lambda kv: -kv[1]["new_bytes"])
w("    %-60s %11s %8s %s" % ("dir", "new bytes", "files", "last"))
for k, v in new[:30]:
    w("    %-60s %11s %8d %s" % (k[:60], human(v["new_bytes"]), v["new"], ts(v["last"])))
w("    project total new bytes = %s" % human(sum(v["new_bytes"] for k, v in new)))
w("")

# ---- big files ----
w("[7] LARGEST SINGLE FILES (>=100 MB, top 50)")
all_big.sort(key=lambda x: -x[0])
for a, p, c, mt, n in all_big[:50]:
    w("    %10s  %-6s nlink=%d  %s  %s" % (human(a), c, n, ts(mt), p[-110:]))
w("    big files total = %d / %s" % (len(all_big), human(sum(x[0] for x in all_big))))
w("")

# ---- junctions ----
w("[8] JUNCTIONS")
w("    count = %d" % len(all_junc))
for j in all_junc[:25]:
    w("    " + j[:150])
w("")

# ---- special pools ----
w("[9] SPECIAL POOLS (extension-less / system / evidence)")
special = [
    ("D:\\pagefile.sys", r"D:\pagefile.sys"),
    ("D:\\$RECYCLE.BIN", r"D:\$RECYCLE.BIN"),
    ("distributed_data\\objects (evidence)", os.path.join(PROJ, "distributed_data", "objects")),
    ("distributed_data\\research_native_blobs", os.path.join(PROJ, "distributed_data", "research_native_blobs")),
    ("distributed_data\\mechanism_traces", os.path.join(PROJ, "distributed_data", "mechanism_traces")),
    ("client_visualization_assets", os.path.join(PROJ, "tests", "glm5", "result", "client_visualization_assets")),
    (".git\\lfs", os.path.join(PROJ, ".git", "lfs")),
    (".git\\objects", os.path.join(PROJ, ".git", "objects")),
    ("models\\hf", os.path.join(PROJ, "models", "hf")),
    ("tests\\glm5\\result", os.path.join(PROJ, "tests", "glm5", "result")),
]
for label, p in special:
    if os.path.isfile(p):
        w("    %-52s %10s  %s" % (label, human(os.path.getsize(p)), ts(os.path.getmtime(p))))
    elif os.path.isdir(p):
        b = f = 0
        last = 0
        r24 = 0
        for dp, dns, fns in os.walk(p, topdown=True):
            kd = [d for d in dns if not is_reparse(os.path.join(dp, d))]
            dns[:] = kd
            for fn in fns:
                q = os.path.join(dp, fn)
                try:
                    st = os.lstat(q)
                except OSError:
                    continue
                b += st.st_size
                f += 1
                if st.st_mtime > last:
                    last = st.st_mtime
                if st.st_mtime >= H24:
                    r24 += 1
        w("    %-52s %10s  files=%-8d r24=%-6d last=%s" % (label, human(b), f, r24, ts(last)))
    else:
        w("    %-52s MISSING" % label)
w("")
w("===== END SCAN =====")

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

with open(BIG, "w", encoding="utf-8", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["size_bytes", "path", "class", "mtime", "nlink"])
    for a, p, c, mt, n in all_big:
        cw.writerow([a, p, c, "%.0f" % mt, n])

print("SCAN OK big=%d hotspots=%d new_dirs=%d" % (len(all_big), len(hot), len(new)))
