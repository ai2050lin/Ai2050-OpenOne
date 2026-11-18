# -*- coding: utf-8 -*-
"""READ-ONLY targeted pass: glm5/result level-1, native level-1, C: mirror, refs."""
import os
import csv
import time
import ctypes
import json
from collections import defaultdict
from datetime import datetime

PROJ = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(PROJ, "tests", "gpt5", "result")
RESULT = os.path.join(PROJ, "tests", "glm5", "result")
NATIVE = r"D:\ai2050_native"
CODEX = r"C:\Users\Admin\.codex\visualizations"
REPORT = os.path.join(OUT, "d_drive_detail_20261117.txt")
CSVOUT = os.path.join(OUT, "d_drive_result_dirs_20261117.csv")

T0 = time.mktime(datetime(2026, 11, 13, 9, 45).timetuple())
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


def measure(p, skip_reparse=True):
    d = {"total": 0, "files": 0, "DATA": 0, "KEEP": 0, "OTHER": 0,
         "dfiles": 0, "kfiles": 0, "r24": 0, "r72": 0, "new": 0,
         "new_bytes": 0, "last": 0, "nlink2": 0}
    if not os.path.isdir(p):
        return d
    for dp, dns, fns in os.walk(p, topdown=True):
        if skip_reparse:
            dns[:] = [x for x in dns if not is_reparse(os.path.join(dp, x))]
        for fn in fns:
            q = os.path.join(dp, fn)
            try:
                st = os.lstat(q)
            except OSError:
                continue
            if skip_reparse and is_reparse(q):
                continue
            a = alloc(q) or st.st_size
            c = cls_of(fn)
            d["total"] += a
            d["files"] += 1
            d[c] += a
            if c == "DATA":
                d["dfiles"] += 1
            elif c == "KEEP":
                d["kfiles"] += 1
            if st.st_mtime >= H24:
                d["r24"] += 1
            if st.st_mtime >= H72:
                d["r72"] += 1
            if st.st_mtime >= T0:
                d["new"] += 1
                d["new_bytes"] += a
            if st.st_mtime > d["last"]:
                d["last"] = st.st_mtime
            if getattr(st, "st_nlink", 1) >= 2:
                d["nlink2"] += a
    return d


L = []
w = L.append
w("D: DRIVE TARGETED DETAIL / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# ---- glm5/result level-1 ----
w("[1] tests\\glm5\\result LEVEL-1 DIRECTORIES (junctions flagged & skipped)")
w("    %-56s %10s %10s %7s %7s %8s %7s %9s %s" %
  ("dir", "total", "DATA", "KEEP", "r24", "r72", "new", "nlink2", "last"))
entries = []
for n in sorted(os.listdir(RESULT)):
    p = os.path.join(RESULT, n)
    if os.path.isdir(p):
        j = is_reparse(p)
        d = measure(p, skip_reparse=True)
        entries.append((n, p, j, d))
for n, p, j, d in sorted(entries, key=lambda x: -x[3]["DATA"]):
    if d["total"] < 50 * 1024 ** 2 and not j:
        continue
    w("    %-56s %10s %10s %7s %7d %8d %7d %9s %s%s" %
      (n[:56], human(d["total"]), human(d["DATA"]), human(d["KEEP"]),
       d["r24"], d["r72"], d["new"], human(d["nlink2"]), ts(d["last"]),
       "  [JUNCTION]" if j else ""))
w("    level-1 dirs = %d ; junctions = %d" %
  (len(entries), len([e for e in entries if e[2]])))
w("")

# ---- new-write directories specifically ----
w("[2] NEW DATA SINCE 11-13 (level-1 dirs with new_bytes > 100 MB)")
w("    %-56s %11s %8s %10s %10s %s" %
  ("dir", "new bytes", "files", "DATA", "KEEP", "last"))
for n, p, j, d in sorted([e for e in entries if e[3]["new_bytes"] > 100 * 1024 ** 2],
                         key=lambda x: -x[3]["new_bytes"]):
    w("    %-56s %11s %8d %10s %10s %s" %
      (n[:56], human(d["new_bytes"]), d["new"], human(d["DATA"]),
       human(d["KEEP"]), ts(d["last"])))
w("")

# ---- top new dirs: full file listing (small ones) ----
w("[3] TOP NEW DIRS - FILE INVENTORY (top 6 by new_bytes)")
tops = sorted([e for e in entries if e[3]["new_bytes"] > 100 * 1024 ** 2],
              key=lambda x: -x[3]["new_bytes"])[:6]
for n, p, j, d in tops:
    w("    --- %s (%s new) ---" % (n, human(d["new_bytes"])))
    fs = []
    for dp, dns, fns in os.walk(p, topdown=True):
        dns[:] = [x for x in dns if not is_reparse(os.path.join(dp, x))]
        for fn in fns:
            q = os.path.join(dp, fn)
            try:
                st = os.lstat(q)
            except OSError:
                continue
            fs.append((st.st_size, os.path.relpath(q, p), cls_of(fn), st.st_mtime))
    for sz, rel, c, mt in sorted(fs, key=lambda x: -x[0])[:14]:
        w("        %10s %-6s %s  %s" % (human(sz), c, ts(mt), rel[:92]))
    w("        files = %d ; total = %s" % (len(fs), human(sum(x[0] for x in fs))))
w("")

# ---- D:\ai2050_native level-1 ----
w("[4] D:\\ai2050_native LEVEL-1")
w("    %-40s %11s %11s %9s %7s %7s %9s %s" %
  ("dir", "total", "DATA", "KEEP", "r24", "r72", "nlink2", "last"))
nat = []
for n in sorted(os.listdir(NATIVE)):
    p = os.path.join(NATIVE, n)
    if os.path.isdir(p) and not is_reparse(p):
        d = measure(p)
        nat.append((n, p, d))
for n, p, d in sorted(nat, key=lambda x: -x[2]["DATA"]):
    w("    %-40s %11s %11s %9s %7d %7d %9s %s" %
      (n[:40], human(d["total"]), human(d["DATA"]), human(d["KEEP"]),
       d["r24"], d["r72"], human(d["nlink2"]), ts(d["last"])))
w("    native DATA total = %s ; total = %s" %
  (human(sum(d["DATA"] for n, p, d in nat)),
   human(sum(d["total"] for n, p, d in nat))))
w("")

# ---- C: mirror ----
w("[5] C: CODEX VISUALIZATIONS MIRROR")
if os.path.isdir(CODEX):
    cd = measure(CODEX)
    w("    C: %s  total=%s DATA=%s files=%d last=%s" %
      (CODEX, human(cd["total"]), human(cd["DATA"]), cd["files"], ts(cd["last"])))
    vr = os.path.join(NATIVE, "visualization-research")
    if os.path.isdir(vr):
        vd = measure(vr)
        w("    D: visualization-research total=%s DATA=%s files=%d last=%s" %
          (human(vd["total"]), human(vd["DATA"]), vd["files"], ts(vd["last"])))
        w("    delta total = %s" % human(vd["total"] - cd["total"]))
        w("    D subdirs / C subdirs name comparison:")
        csub = set()
        for dp, dns, fns in os.walk(CODEX):
            if "research-storage" in dp:
                for d in dns:
                    csub.add(d)
        dsub = set(os.listdir(vr))
        w("        D names=%d  C research-storage names=%d  overlap=%d" %
          (len(dsub), len(csub), len(dsub & csub)))
        w("        D-only (first 15) = %s" % sorted(dsub - csub)[:15])
else:
    w("    C: codex visualizations MISSING")
w("")

# ---- junction health ----
w("[6] JUNCTIONS UNDER tests\\glm5\\result")
for n, p, j, d in entries:
    if j:
        try:
            tgt = os.readlink(p).replace("\\\\?\\", "")
        except Exception as e:
            tgt = "readlink_err " + repr(e)
        w("    %-44s -> %s  target_exists=%s" %
          (n[:44], tgt[-110:], os.path.exists(tgt)))
w("")

# ---- special residue ----
w("[7] LEFTOVER STAGING FROM INTERRUPTED 11-13 FINISH")
stage = r"D:\AI2050\_A_stage_finish_20261113"
if os.path.isdir(stage):
    sd = measure(stage)
    w("    %s  total=%s files=%d last=%s" %
      (stage, human(sd["total"]), sd["files"], ts(sd["last"])))
    for dp, dns, fns in os.walk(stage):
        w("    subdir %s  files=%d" % (dp, len(fns)))
else:
    w("    staging MISSING")
w("")

w("===== END DETAIL =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

with open(CSVOUT, "w", encoding="utf-8", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["name", "absolute_path", "is_junction", "total", "DATA", "KEEP",
                 "OTHER", "files", "dfiles", "kfiles", "r24", "r72", "new_bytes",
                 "nlink2", "last_write"])
    for n, p, j, d in entries:
        cw.writerow([n, p, j, d["total"], d["DATA"], d["KEEP"], d["OTHER"],
                     d["files"], d["dfiles"], d["kfiles"], d["r24"], d["r72"],
                     d["new_bytes"], d["nlink2"], "%.0f" % d["last"]])
print("DETAIL OK entries=%d native=%d" % (len(entries), len(nat)))
