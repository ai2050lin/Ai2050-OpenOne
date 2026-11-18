"""READ-ONLY detail pass: per-directory classification for the repro-only policy.
Produces a de-duplicated cleanup-candidate CSV plus a decision report.
"""
import os
import csv
import time
import ctypes
from collections import defaultdict
from datetime import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
RESULT = os.path.join(ROOT, "tests", "glm5", "result")
REPORT = os.path.join(ROOT, "tests", "gpt5", "result", "project_repro_detail_20261111.txt")
CSVOUT = os.path.join(ROOT, "tests", "gpt5", "result", "project_repro_candidates_20261111.csv")

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

def alloc(p):
    hi = ctypes.c_uint32(0)
    lo = GetCompressedFileSizeW(p, ctypes.byref(hi))
    if lo == 0xFFFFFFFF:
        return None
    return (hi.value << 32) | lo

# Repro-capable KEEP set: scripts, configs, conclusions, test inputs, client assets
KEEP_EXT = {".py", ".sh", ".ps1", ".bat", ".cmd", ".js", ".jsx", ".ts", ".tsx",
            ".css", ".html", ".htm", ".vue", ".json", ".jsonl", ".md", ".txt",
            ".log", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".sha256", ".gitignore", ".mdx", ".rst", ".svg", ".png", ".jpg",
            ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".docx", ".pptx", ".xlsx",
            ".lock", ".map", ".sql", ".proto", ".graphql", ".env", ".example",
            ".pyi", ".h", ".c", ".cc", ".cpp", ".tsx", ".mjs", ".cjs", ".mts", ".cts"}
# Intermediate / regenerable binary data
DATA_EXT = {".npy", ".npz", ".pkl", ".pickle", ".safetensors", ".bin", ".pt", ".pth",
            ".ckpt", ".h5", ".hdf5", ".parquet", ".arrow", ".feather", ".sqlite3",
            ".sqlite", ".db", ".gz", ".tgz", ".tar", ".zip", ".7z", ".rar", ".bz2",
            ".xz", ".zst", ".tmp", ".part", ".conllu", ".onnx", ".gguf", ".model",
            ".pyc", ".pyo", ".so", ".dll", ".exe", ".pdb", ".whl", ".vhdx", ".img",
            ".iso", ".mp4", ".avi", ".mov", ".wav", ".mp3", ".nii", ".mat", ".idx",
            ".pack", ".asar", ".pak", ".node", ".pyd", ".ttf", ".ttc", ".wasm", ".out"}
DATA_SUFFIX = (".gz.tmp", ".npy.tmp", ".npz.tmp", ".before-import", ".partial", ".old")

H24 = time.time() - 24 * 3600
H72 = time.time() - 72 * 3600

def scan_dir(path, skip_reparse=True):
    """return dict with per-class stats, file list (rel, size, alloc, mtime, nlink, ext, cls)"""
    st = {"log": 0, "alloc": 0, "n": 0, "keep": 0, "data": 0, "other": 0,
          "r24": 0, "r72": 0, "last": 0, "nlink2": 0, "nlink2_bytes": 0,
          "concl": 0, "sha": 0, "bigjson": 0}
    files = []
    if not os.path.isdir(path):
        return st, files
    for root, dirs, fns in os.walk(path, topdown=True):
        if skip_reparse:
            kept = []
            for d in dirs:
                if not is_reparse(os.path.join(root, d)):
                    kept.append(d)
            dirs[:] = kept
        for fn in fns:
            p = os.path.join(root, fn)
            try:
                s = os.lstat(p)
            except Exception:
                continue
            if is_reparse(p):
                continue
            al = alloc(p)
            if al is None:
                al = s.st_size
            low = fn.lower()
            ext = os.path.splitext(low)[1]
            if ext in KEEP_EXT:
                cls = "KEEP"
            elif ext in DATA_EXT or low.endswith(DATA_SUFFIX):
                cls = "DATA"
            else:
                cls = "OTHER"
            nl = getattr(s, "st_nlink", 1)
            st["log"] += s.st_size; st["alloc"] += al; st["n"] += 1
            st[cls.lower()] += s.st_size
            if s.st_mtime >= H24: st["r24"] += 1
            if s.st_mtime >= H72: st["r72"] += 1
            if s.st_mtime > st["last"]: st["last"] = s.st_mtime
            if nl >= 2:
                st["nlink2"] += 1; st["nlink2_bytes"] += al
            if fn in ("result.json", "execution.json", "behavior.json"):
                st["concl"] += 1
            if fn.endswith(".sha256"):
                st["sha"] += 1
            if ext == ".json" and s.st_size >= 10 * 1024 ** 2:
                st["bigjson"] += 1
            files.append((os.path.relpath(p, ROOT), s.st_size, al, s.st_mtime, nl, ext, cls))
    return st, files

def gb(x): return x / 1024 ** 3
def mb(x): return x / 1024 ** 2
def ts(x): return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else "-"

out = []
def w(s=""): out.append(s)

w("PROJECT REPRO DETAIL / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# ---------- [1] tests/glm5/result top-level ----------
w("[1] tests\\glm5\\result  (top-level children, sorted by DATA bytes)")
w("    %-52s %9s %9s %7s %7s %6s %6s %5s %5s %s" %
  ("dir", "DATA GB", "KEEP GB", "files", "r24", "concl", "sha", "nl2", "type", "last_write"))
children = []
for name in os.listdir(RESULT):
    p = os.path.join(RESULT, name)
    if is_reparse(p):
        children.append((name, None, None, "JUNCTION"))
        continue
    if not os.path.isdir(p):
        continue
    st, fs = scan_dir(p)
    children.append((name, st, fs, "DIR"))
children.sort(key=lambda x: -(x[1]["data"] if x[1] else 0))
res_total = {"data": 0, "keep": 0, "alloc": 0, "n": 0}
for name, st, fs, kind in children:
    if kind == "JUNCTION":
        w("    %-52s %9s %9s %7s %7s %6s %6s %5s %5s %s" %
          (name, "-", "-", "-", "-", "-", "-", "-", "JUNC", "(points to C: client data)"))
        continue
    res_total["data"] += st["data"]; res_total["keep"] += st["keep"]
    res_total["alloc"] += st["alloc"]; res_total["n"] += st["n"]
    w("    %-52s %9.2f %9.2f %7d %7d %6d %6d %5d %5s %s" %
      (name[:52], gb(st["data"]), gb(st["keep"]), st["n"], st["r24"],
       st["concl"], st["sha"], st["nlink2"], "DIR", ts(st["last"])))
w("    TOTAL (non-junction) DATA=%.2f GB KEEP=%.2f GB files=%d" %
  (gb(res_total["data"]), gb(res_total["keep"]), res_total["n"]))
w("")

# ---------- [2] other major pools ----------
POOLS = [
    ("distributed_data\\research_native_blobs", os.path.join(ROOT, "distributed_data", "research_native_blobs")),
    ("distributed_data\\objects", os.path.join(ROOT, "distributed_data", "objects")),
    ("distributed_data\\research_library", os.path.join(ROOT, "distributed_data", "research_library")),
    ("distributed_data\\mechanism_traces", os.path.join(ROOT, "distributed_data", "mechanism_traces")),
    ("distributed_data\\datapack", os.path.join(ROOT, "distributed_data", "datapack")),
    ("models\\hf", os.path.join(ROOT, "models", "hf")),
    (".git\\lfs", os.path.join(ROOT, ".git", "lfs")),
    (".git\\objects", os.path.join(ROOT, ".git", "objects")),
    ("release", os.path.join(ROOT, "release")),
    ("video", os.path.join(ROOT, "video")),
    ("frontend\\node_modules", os.path.join(ROOT, "frontend", "node_modules")),
    (".venv\\Lib", os.path.join(ROOT, ".venv", "Lib")),
    ("tests\\glm5_temp", os.path.join(ROOT, "tests", "glm5_temp")),
    ("tests\\gpt5", os.path.join(ROOT, "tests", "gpt5")),
    ("research", os.path.join(ROOT, "research")),
]
w("[2] OTHER MAJOR POOLS")
w("    %-46s %9s %9s %9s %8s %6s %6s %s" %
  ("pool", "DATA GB", "KEEP GB", "OTHER GB", "files", "r24", "nl2", "last_write"))
pool_stats = {}
for label, p in POOLS:
    st, fs = scan_dir(p)
    pool_stats[label] = (st, fs)
    w("    %-46s %9.2f %9.2f %9.2f %8d %6d %6d %s" %
      (label, gb(st["data"]), gb(st["keep"]), gb(st["other"]), st["n"],
       st["r24"], st["nlink2"], ts(st["last"])))
w("")

# ---------- [3] models/hf per-model + hardlink / duplicate check ----------
w("[3] models\\hf PER-MODEL (+ hardlink / D:\\develop\\model duplicate)")
MODELS = os.path.join(ROOT, "models", "hf")
DEV = r"D:\develop\model"
model_rows = []
for name in sorted(os.listdir(MODELS)):
    p = os.path.join(MODELS, name)
    if is_reparse(p):
        w("    %-42s JUNCTION -> %s" % (name, "D:\\develop\\model (no local bytes)"))
        continue
    if not os.path.isdir(p):
        continue
    st, fs = scan_dir(p)
    nl2 = sum(1 for x in fs if x[4] >= 2)
    model_rows.append((name, st, nl2, fs))
    w("    %-42s %8.2f GB  %4d files  nlink2=%3d  %s" %
      (name, gb(st["alloc"]), st["n"], nl2, ts(st["last"])))
w("")
w("    hardlinked safetensors sample (deleting these frees ~0):")
cnt = 0
for name, st, nl2, fs in model_rows:
    for rel, sz, al, mt, nl, ext, cls in fs:
        if nl >= 2 and cnt < 14:
            w("        nlink=%d %8.2f MB  %s" % (nl, mb(al), rel))
            cnt += 1
w("")

# ---------- [4] git lfs / pack ----------
w("[4] .git BREAKDOWN")
for label in (".git\\lfs", ".git\\objects"):
    st, fs = pool_stats[label]
    w("    %-16s alloc=%.2f GB files=%d last=%s" % (label, gb(st["alloc"]), st["n"], ts(st["last"])))
    for rel, sz, al, mt, nl, ext, cls in sorted(fs, key=lambda x: -x[2])[:6]:
        w("        %9.2f MB  %s" % (mb(al), rel))
w("")

# ---------- [5] 24h active under tests/glm5/result ----------
w("[5] 24h-ACTIVE DIRECTORIES (pipeline still writing - DO NOT clean yet)")
active = [(n, s) for n, s, f, k in children if s and s["r24"] > 0]
active.sort(key=lambda x: -x[1]["r24"])
for n, s in active[:30]:
    w("    %-52s r24=%5d r72=%6d DATA=%.2f GB last=%s" %
      (n[:52], s["r24"], s["r72"], gb(s["data"]), ts(s["last"])))
w("    active dirs = %d ; their DATA = %.2f GB" %
  (len(active), gb(sum(s["data"] for n, s in active))))
w("")

# ---------- [6] candidate CSV ----------
w("[6] CANDIDATE CLASSIFICATION (de-duplicated, top-level pool granularity)")
csv_rows = []
def add(tier, label, path, st, basis, precondition):
    csv_rows.append({
        "tier": tier, "label": label, "absolute_path": path,
        "clean_bytes": st["data"], "clean_files": sum(1 for x in []),
        "keep_bytes": st["keep"], "total_bytes": st["log"],
        "r24_writes": st["r24"], "nlink2_files": st["nlink2"],
        "nlink2_bytes": st["nlink2_bytes"], "last_write": ts(st["last"]),
        "basis": basis, "precondition": precondition})

# result children
for name, st, fs, kind in children:
    if kind == "JUNCTION" or st is None:
        continue
    if st["data"] < 50 * 1024 ** 2:
        continue
    p = os.path.join(RESULT, name)
    hot = st["r24"] > 0
    if hot:
        tier = "P3-ACTIVE"
        pre = "stop writing pipeline / confirm conclusions in MEMO first"
    elif st["concl"] > 0 or st["sha"] > 0:
        tier = "P2-COLD-WITH-CONCLUSIONS"
        pre = "keep all json/md/sha256; delete only npy/npz/gz/bin"
    else:
        tier = "P1-COLD"
        pre = "keep all json/md/sha256; delete only npy/npz/gz/bin"
    add(tier, "result\\" + name, p, st,
        "intermediate tensors; conclusions are in json", pre)
for label, p in POOLS:
    st, fs = pool_stats[label]
    if label.startswith("research"):
        continue
    if st["data"] < 20 * 1024 ** 2:
        continue
    if label == "distributed_data\\research_native_blobs":
        tier = "B-CONFIRM"
        pre = "CAS blob library; confirm derived JSON registered at center"
    elif label == "models\\hf":
        tier = "B-CONFIRM"
        pre = "model weights: re-downloadable but 31.29 GB are hardlinks (free ~0)"
    elif label.startswith(".git"):
        tier = "B-CONFIRM"
        pre = "git internal data; DO NOT use `git lfs prune` (drops refs/codex/* objects)"
    elif label == "release":
        tier = "A-BUILD"
        pre = "keep desktop-1.0.13 / deployment-20261108 / sha256 manifests"
    elif label == "video":
        tier = "B-CONFIRM"
        pre = "video source frames; confirm not needed for deliverables"
    else:
        tier = "A-CACHE"
        pre = "regenerable cache / dependency install"
    add(tier, label, p, st, "non-repro intermediate or regenerable data", pre)

with open(CSVOUT, "w", encoding="utf-8", newline="") as f:
    flds = ["tier", "label", "absolute_path", "clean_bytes", "clean_files",
            "keep_bytes", "total_bytes", "r24_writes", "nlink2_files",
            "nlink2_bytes", "last_write", "basis", "precondition"]
    wr = csv.DictWriter(f, fieldnames=flds)
    wr.writeheader()
    for r in csv_rows:
        wr.writerow(r)
for t in ("P1-COLD", "P2-COLD-WITH-CONCLUSIONS", "P3-ACTIVE", "A-BUILD", "A-CACHE", "B-CONFIRM"):
    sub = [r for r in csv_rows if r["tier"] == t]
    w("    %-30s %3d entries  clean=%.2f GB  keep=%.2f GB" %
      (t, len(sub), gb(sum(r["clean_bytes"] for r in sub)),
       gb(sum(r["keep_bytes"] for r in sub))))
w("    CSV = " + CSVOUT)
w("")
w("===== END DETAIL =====")
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("detail done")
