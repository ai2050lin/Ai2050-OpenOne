"""READ-ONLY full scan of D:\\AI2050\\Ai2050-OpenOne for the
'keep scripts + test inputs + results (reproducible), clean intermediate big data' policy.

Key accuracy measures:
  - reparse points (junctions) are recorded and NEVER traversed (avoid double counting)
  - both logical size and ALLOCATED (on-disk) size are measured
  - hardlinks (st_nlink >= 2) are counted separately: deleting one link frees nothing
"""
import os
import time
import ctypes
from collections import defaultdict
from datetime import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\project_repro_scan_20261111.txt"

REPARSE = 0x400
GetFileAttributesW = ctypes.windll.kernel32.GetFileAttributesW
GetFileAttributesW.restype = ctypes.c_uint32
GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
GetCompressedFileSizeW = ctypes.windll.kernel32.GetCompressedFileSizeW
GetCompressedFileSizeW.restype = ctypes.c_uint32
GetCompressedFileSizeW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]

def is_reparse(p):
    a = GetFileAttributesW(p)
    return a != 0xFFFFFFFF and bool(a & REPARSE)

def alloc_size(p):
    hi = ctypes.c_uint32(0)
    lo = GetCompressedFileSizeW(p, ctypes.byref(hi))
    if lo == 0xFFFFFFFF:
        return None
    return (hi.value << 32) | lo

def free_gb(d):
    fb = ctypes.c_ulonglong(0); tb = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(d),
        ctypes.byref(fb), ctypes.byref(tb), None)
    return fb.value / 1024 ** 3, tb.value / 1024 ** 3

KEEP_EXT = {".py", ".sh", ".ps1", ".bat", ".cmd", ".js", ".jsx", ".ts", ".tsx",
            ".css", ".html", ".htm", ".vue", ".json", ".jsonl", ".md", ".txt",
            ".log", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".sha256", ".gitignore", ".mdx", ".rst", ".svg", ".png", ".jpg",
            ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".docx", ".pptx", ".xlsx",
            ".lock", ".map", ".sql", ".proto", ".graphql", ".env", ".example"}
DATA_EXT = {".npy", ".npz", ".pkl", ".pickle", ".safetensors", ".bin", ".pt", ".pth",
            ".ckpt", ".h5", ".hdf5", ".parquet", ".arrow", ".feather", ".sqlite3",
            ".sqlite", ".db", ".gz", ".tgz", ".tar", ".zip", ".7z", ".rar", ".bz2",
            ".xz", ".zst", ".tmp", ".part", ".conllu", ".onnx", ".gguf", ".model",
            ".pyc", ".pyo", ".so", ".dll", ".exe", ".pdb", ".whl", ".vhdx", ".img",
            ".iso", ".mp4", ".avi", ".mov", ".wav", ".mp3", ".nii", ".mat", ".idx"}

DATA_EXTRA_SUFFIX = (".gz.tmp", ".npy.tmp", ".npz.tmp", ".before-import", ".partial")

lvl1 = defaultdict(lambda: {"log": 0, "alloc": 0, "n": 0, "keep": 0, "data": 0,
                            "other": 0, "last": 0, "r24": 0, "nlink2": 0})
lvl2 = defaultdict(lambda: {"log": 0, "alloc": 0, "n": 0, "keep": 0, "data": 0,
                            "other": 0, "last": 0, "r24": 0, "nlink2": 0})
byext = defaultdict(lambda: {"log": 0, "alloc": 0, "n": 0, "nlink2": 0})
byday = defaultdict(lambda: {"log": 0, "n": 0})
big = []
junctions = []
H24 = time.time() - 24 * 3600
H72 = time.time() - 72 * 3600
r24_all = r72_all = 0
total_log = total_alloc = total_files = 0
nlink2_bytes = nlink2_files = 0
sparse_saving = 0

t0 = time.time()
for root, dirs, files in os.walk(ROOT, topdown=True):
    kept = []
    for d in dirs:
        dp = os.path.join(root, d)
        if is_reparse(dp):
            try:
                tgt = os.readlink(dp)
            except Exception:
                tgt = "?"
            junctions.append((dp, tgt))
        else:
            kept.append(d)
    dirs[:] = kept
    for fn in files:
        p = os.path.join(root, fn)
        try:
            st = os.lstat(p)
        except Exception:
            continue
        if is_reparse(p):
            try:
                tgt = os.readlink(p)
            except Exception:
                tgt = "?"
            junctions.append((p, tgt))
            continue
        sz = st.st_size
        al = alloc_size(p)
        if al is None:
            al = sz
        if al < sz:
            sparse_saving += sz - al
        mt = st.st_mtime
        nl = getattr(st, "st_nlink", 1)
        low = fn.lower()
        ext = os.path.splitext(low)[1]
        if ext in KEEP_EXT:
            cat = "keep"
        elif ext in DATA_EXT or low.endswith(DATA_EXTRA_SUFFIX):
            cat = "data"
        else:
            cat = "other"
        rel = os.path.relpath(p, ROOT)
        parts = rel.split(os.sep)
        l1 = parts[0] if len(parts) > 1 else "(root files)"
        l2 = os.sep.join(parts[:2]) if len(parts) > 2 else (
             os.sep.join(parts[:2]) if len(parts) == 2 else "(root files)")
        for rec in (lvl1[l1], lvl2[l2]):
            rec["log"] += sz; rec["alloc"] += al; rec["n"] += 1
            rec[cat] += sz
            if mt > rec["last"]:
                rec["last"] = mt
            if mt >= H24:
                rec["r24"] += 1
            if nl >= 2:
                rec["nlink2"] += 1
        e = byext[ext or "(none)"]
        e["log"] += sz; e["alloc"] += al; e["n"] += 1
        if nl >= 2:
            e["nlink2"] += 1
        day = datetime.fromtimestamp(mt).strftime("%Y-%m-%d")
        byday[day]["log"] += sz; byday[day]["n"] += 1
        if mt >= H24: r24_all += 1
        if mt >= H72: r72_all += 1
        if nl >= 2:
            nlink2_bytes += al; nlink2_files += 1
        total_log += sz; total_alloc += al; total_files += 1
        if len(big) < 4000:
            big.append((al, sz, nl, mt, rel))
        else:
            # keep top 400 by allocated size using cheap replacement
            if al > big[-1][0]:
                big[-1] = (al, sz, nl, mt, rel)
                big.sort(reverse=True)

big.sort(reverse=True)
elapsed = time.time() - t0

cf, ct = free_gb("C:\\")
df, dt = free_gb("D:\\")
out = []
def w(s=""):
    out.append(s)

def mb(x):
    return x / 1024 ** 2

def gb(x):
    return x / 1024 ** 3

w("PROJECT REPRO-POLICY SCAN / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("root = " + ROOT)
w("elapsed = %.1f s" % elapsed)
w("")
w("[0] DISK + SCAN TOTALS")
w("    C: free %.2f GB / %.2f GB" % (cf, ct))
w("    D: free %.2f GB / %.2f GB" % (df, dt))
w("    scanned files      = %d" % total_files)
w("    logical total      = %.2f GB" % gb(total_log))
w("    ALLOCATED total    = %.2f GB   <-- real reclaimable basis" % gb(total_alloc))
w("    sparse/prealloc gap= %.2f GB" % gb(sparse_saving))
w("    hardlinked (nlink>=2) = %d files / %.2f GB allocated (deleting frees ~0)" %
  (nlink2_files, gb(nlink2_bytes)))
w("    files written 24h = %d ; 72h = %d" % (r24_all, r72_all))
w("    reparse points skipped = %d" % len(junctions))
w("")
w("[1] LEVEL-1 DIRECTORIES (by allocated size)")
w("    %-44s %10s %10s %8s %10s %10s %8s %7s %s" %
  ("dir", "alloc GB", "logical", "files", "KEEP GB", "DATA GB", "other", "r24", "last_write"))
for k, v in sorted(lvl1.items(), key=lambda x: -x[1]["alloc"]):
    w("    %-44s %10.2f %10.2f %8d %10.2f %10.2f %8.2f %7d %s" %
      (k[:44], gb(v["alloc"]), gb(v["log"]), v["n"], gb(v["keep"]), gb(v["data"]),
       gb(v["other"]), v["r24"],
       datetime.fromtimestamp(v["last"]).strftime("%Y-%m-%d %H:%M") if v["last"] else "-"))
w("")
w("[2] LEVEL-2 DIRECTORIES (top 90 by allocated size)")
w("    %-60s %10s %10s %8s %10s %10s %7s %s" %
  ("dir", "alloc GB", "logical", "files", "KEEP GB", "DATA GB", "r24", "last_write"))
for k, v in sorted(lvl2.items(), key=lambda x: -x[1]["alloc"])[:90]:
    w("    %-60s %10.2f %10.2f %8d %10.2f %10.2f %7d %s" %
      (k[:60], gb(v["alloc"]), gb(v["log"]), v["n"], gb(v["keep"]), gb(v["data"]),
       v["r24"], datetime.fromtimestamp(v["last"]).strftime("%Y-%m-%d %H:%M") if v["last"] else "-"))
w("")
w("[3] EXTENSIONS (top 60 by allocated size)")
w("    %-16s %11s %11s %9s %9s %s" % ("ext", "alloc GB", "logical", "files", "nlink2", "class"))
for k, v in sorted(byext.items(), key=lambda x: -x[1]["alloc"])[:60]:
    cls = "KEEP" if k in KEEP_EXT else ("DATA" if k in DATA_EXT else "other")
    w("    %-16s %11.2f %11.2f %9d %9d %s" %
      (k, gb(v["alloc"]), gb(v["log"]), v["n"], v["nlink2"], cls))
w("")
w("[4] MTIME DAY DISTRIBUTION (logical bytes written)")
for k in sorted(byday.keys(), reverse=True)[:35]:
    w("    %s  %10.2f GB  %8d files" % (k, gb(byday[k]["log"]), byday[k]["n"]))
w("")
w("[5] TOP 120 LARGEST FILES (allocated)")
w("    %10s %10s %5s %-19s %s" % ("alloc MB", "logical", "nlink", "mtime", "relative path"))
for al, sz, nl, mt, rel in big[:120]:
    w("    %10.2f %10.2f %5d %-19s %s" %
      (mb(al), mb(sz), nl, datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S"), rel))
w("")
w("[6] REPARSE POINTS (junctions) - NOT traversed")
for p, t in junctions[:80]:
    w("    %s  ->  %s" % (p, t))
w("    total = %d" % len(junctions))
w("")
w("===== END SCAN =====")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("scan done")
