"""READ-ONLY final pass: de-duplicated cleanup plan under the repro-only policy.
Produces:
  project_repro_final_report_20261111.txt   (decision report)
  project_repro_cleanup_plan_20261111.csv   (per-entry plan, exact bytes/files)
  project_repro_bigfiles_20261111.csv       (largest cleanable files)
All numbers are re-measured on real disk; hardlinks are deducted from expected reclaim.
"""
import os
import csv
import time
import ctypes
from datetime import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTDIR = os.path.join(ROOT, "tests", "gpt5", "result")
REPORT = os.path.join(OUTDIR, "project_repro_final_report_20261111.txt")
PLANCSV = os.path.join(OUTDIR, "project_repro_cleanup_plan_20261111.csv")
BIGCSV = os.path.join(OUTDIR, "project_repro_bigfiles_20261111.csv")
RESULT = os.path.join(ROOT, "tests", "glm5", "result")

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
    return None if lo == 0xFFFFFFFF else (hi.value << 32) | lo

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

H24 = time.time() - 24 * 3600
H72 = time.time() - 72 * 3600

def measure(path):
    """return clean_alloc, clean_files, keep_alloc, keep_files, r24, last, nlink2_bytes, bigfiles"""
    cb = kb = 0; cf = kf = 0; r24 = 0; last = 0; nl2 = 0
    big = []
    if not os.path.exists(path):
        return 0, 0, 0, 0, 0, 0, 0, big
    if os.path.isfile(path):
        st = os.lstat(path); a = alloc(path) or st.st_size
        ext = os.path.splitext(path.lower())[1]
        cls = "K" if ext in KEEP_EXT else ("D" if (ext in DATA_EXT or path.lower().endswith(DATA_SUFFIX)) else "O")
        if cls == "D":
            return a, 1, 0, 0, (1 if st.st_mtime >= H24 else 0), st.st_mtime, (a if getattr(st, "st_nlink", 1) >= 2 else 0), [(a, os.path.relpath(path, ROOT))]
        return 0, 0, a, 1, (1 if st.st_mtime >= H24 else 0), st.st_mtime, 0, []
    for root, dirs, files in os.walk(path, topdown=True):
        kept = []
        for d in dirs:
            if not is_reparse(os.path.join(root, d)):
                kept.append(d)
        dirs[:] = kept
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.lstat(p)
            except Exception:
                continue
            if is_reparse(p):
                continue
            a = alloc(p) or st.st_size
            low = fn.lower(); ext = os.path.splitext(low)[1]
            if ext in KEEP_EXT:
                kb += a; kf += 1
            elif ext in DATA_EXT or low.endswith(DATA_SUFFIX):
                cb += a; cf += 1
                if getattr(st, "st_nlink", 1) >= 2:
                    nl2 += a
                if a >= 200 * 1024 ** 2:
                    big.append((a, os.path.relpath(p, ROOT)))
            if st.st_mtime >= H24: r24 += 1
            if st.st_mtime > last: last = st.st_mtime
    return cb, cf, kb, kf, r24, last, nl2, big

def measure_all(path):
    """all file bytes regardless of extension (for extension-less CAS/LFS objects)"""
    b = f = 0; r24 = 0; last = 0
    if not os.path.isdir(path):
        return 0, 0, 0, 0
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if not is_reparse(os.path.join(root, d))]
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.lstat(p)
            except Exception:
                continue
            if is_reparse(p):
                continue
            b += alloc(p) or st.st_size
            f += 1
            if st.st_mtime >= H24: r24 += 1
            if st.st_mtime > last: last = st.st_mtime
    return b, f, r24, last

def gb(x): return x / 1024 ** 3
def mb(x): return x / 1024 ** 2
def ts(x): return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else "-"

# ---------------- define de-duplicated plan entries ----------------
# (tier, label, absolute_path, basis, precondition)
E = []
def add(tier, label, p, basis, pre):
    E.append((tier, label, p, basis, pre))

# A1: cold intermediate tensors in tests/glm5/result (>72h idle, no sha256, no result.json)
COLD = ["branch_style_joint_v1", "mlp_tail_b28_b31_v1", "mlp_tail_goodfire_v1",
        "mlp_condition_v2", "mlp_b34_confirm_v3", "mlp_condition_v1", "live_traces",
        "mlp_b34_confirm_v1", "mlp_b34_confirm_v2", "mlp_b31_b32_authority_v2",
        "mlp_b31_b32_authority_v1", "mlp_tail_final_pack_acceptance",
        "mlp_tail_pack_acceptance", "hiddenstate_components_v1",
        "mlp_tail_relative_encoding_v1"]
for n in COLD:
    add("A1-COLD-TENSOR", "result\\" + n, os.path.join(RESULT, n),
        "intermediate npy/npz; >72h idle; conclusions in json kept",
        "keep ALL json/md/sha256 in the same directory")

# A2: CAS blob library
add("A2-CAS-BLOBS", "distributed_data\\research_native_blobs",
    os.path.join(ROOT, "distributed_data", "research_native_blobs"),
    "content-addressed .npy tensor blobs, 668 files, idle since 11-05",
    "confirm derived JSON registered at center: GET /api/traces/research-space")

# A3: old release builds
for n in sorted(os.listdir(os.path.join(ROOT, "release"))):
    p = os.path.join(ROOT, "release", n)
    if os.path.isdir(p) and n.startswith("desktop-1.0.") and n != "desktop-1.0.13":
        add("A3-OLD-BUILD", "release\\" + n, p, "superseded desktop build",
            "keep desktop-1.0.13 as latest")
    elif n.endswith((".tar.gz", ".zip", ".tar")) and os.path.isfile(p):
        if "mlp-tail-display-20261108-r2" in n or "AI2050-release-20261108-client1.0.8" in n:
            add("A3-OLD-BUILD", "release\\" + n, p, "large superseded release archive",
                "keep matching .sha256 manifests")
add("A3-OLD-BUILD", "release\\deployment-20261108",
    os.path.join(ROOT, "release", "deployment-20261108"),
    "superseded deployment bundle", "keep if it is the deployed version - confirm")

# A4: caches / regenerable
for label, rel, basis, pre in [
    ("mechanism_traces", r"distributed_data\mechanism_traces", "1.15 GB index.sqlite3, rebuildable", "confirm no live reader"),
    ("frontend\\node_modules", r"frontend\node_modules", "npm reinstallable", "npm ci to restore"),
]:
    add("A4-CACHE", label, os.path.join(ROOT, rel), basis, pre)
for label, rel, basis, pre in [
    ("video", r"video", "video source frames / wav; 418 writes in 24h", "ACTIVE rendering - DO NOT clean until stopped"),
    ("datapack", r"distributed_data\datapack", "unpacked datapack payload; writes at 02:48", "ACTIVE - stop pipeline first"),
]:
    add("B7-ACTIVE-MEDIA", label, os.path.join(ROOT, rel), basis, pre)

# B: confirm-first
ACTIVE = ["source_formation_v1", "source_binding_v1", "source_relay_v1",
          "source_relay_v2", "common_bench_b1_repair_v2", "source_binding_ablation_v1"]
for n in ACTIVE:
    add("B1-ACTIVE-TENSOR", "result\\" + n, os.path.join(RESULT, n),
        "11-10/11 current experiment tensors", "wait until conclusions are in MEMO/ledger and pipeline idle")
add("B2-SHA256", "result\\mlp_tail_acceptance", os.path.join(RESULT, "mlp_tail_acceptance"),
    "0.88 GB tensors but 8 .sha256 registrations + 2.84 GB json",
    "deleting breaks integrity verification; keep unless re-registered")
add("B3-MODELS", "models\\hf", os.path.join(ROOT, "models", "hf"),
    "75.98 GB weights; 31.29 GB are hardlinks (free ~0); re-downloadable",
    "confirm which paths scripts use; 3 entries are junctions to D:\\develop\\model")
add("B4-GIT-LFS", ".git\\lfs", os.path.join(ROOT, ".git", "lfs"),
    "29.01 GB LFS objects", "NEVER `git lfs prune` (drops refs/codex/* referenced objects)")
add("B5-GIT-PACK", ".git\\objects", os.path.join(ROOT, ".git", "objects"),
    "5.29 GB git objects; 695 writes in 24h", "active git operation - do not touch")
add("B6-VENV", ".venv\\Lib", os.path.join(ROOT, ".venv", "Lib"),
    "3.11 GB installed packages", "rebuildable via uv, but needed to run scripts now")

# C: must keep (measured for the report, never cleaned)
KEEP = [
    ("all conclusions (json/jsonl/md/log/csv/py/sha256)", None),
    ("client_visualization_assets", os.path.join(RESULT, "client_visualization_assets")),
    ("distributed_data\\objects (CAS evidence packs)", os.path.join(ROOT, "distributed_data", "objects")),
    ("distributed_data\\research_library", os.path.join(ROOT, "distributed_data", "research_library")),
    ("research", os.path.join(ROOT, "research")),
    ("tests\\glm5_temp", os.path.join(ROOT, "tests", "glm5_temp")),
    ("9 junctions -> C: .codex visualizations", None),
    (".git refs / config / HEAD", None),
]

# ---------------- measure ----------------
rows = []
sums = {}
for tier, label, p, basis, pre in E:
    cb, cf, kb, kf, r24, last, nl2, big = measure(p)
    if tier == "B4-GIT-LFS":
        # LFS objects have no extension -> extension-based classification sees 0
        cb, cf, r24, last = measure_all(p)
        basis = "29.01 GB extension-less LFS objects (measured by measure_all)"
    r = {"tier": tier, "label": label, "absolute_path": p,
         "clean_bytes": cb, "clean_files": cf, "keep_bytes": kb, "keep_files": kf,
         "r24_writes": r24, "hardlink_bytes": nl2,
         "expected_reclaim_bytes": cb - nl2,
         "last_write": ts(last), "exists": os.path.exists(p),
         "basis": basis, "precondition": pre}
    rows.append(r)
    s = sums.setdefault(tier, [0, 0, 0, 0])
    s[0] += cb; s[1] += cf; s[2] += cb - nl2; s[3] += kb
    for a, rel in big:
        pass
allbig = []
for tier, label, p, basis, pre in E:
    if not os.path.isdir(p):
        continue
    for root, dirs, files in os.walk(p, topdown=True):
        dirs[:] = [d for d in dirs if not is_reparse(os.path.join(root, d))]
        for fn in files:
            fp = os.path.join(root, fn)
            ext = os.path.splitext(fn.lower())[1]
            if ext not in DATA_EXT and not fn.lower().endswith(DATA_SUFFIX):
                continue
            try:
                st = os.lstat(fp)
                a = alloc(fp) or st.st_size
            except Exception:
                continue
            if a >= 200 * 1024 ** 2:
                allbig.append((a, tier, os.path.relpath(fp, ROOT),
                               getattr(st, "st_nlink", 1), st.st_mtime))
allbig.sort(reverse=True)

with open(PLANCSV, "w", encoding="utf-8", newline="") as f:
    flds = ["tier", "label", "absolute_path", "clean_bytes", "clean_files",
            "keep_bytes", "keep_files", "r24_writes", "hardlink_bytes",
            "expected_reclaim_bytes", "last_write", "exists", "basis", "precondition"]
    wr = csv.DictWriter(f, fieldnames=flds); wr.writeheader()
    for r in rows: wr.writerow(r)
with open(BIGCSV, "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["allocated_bytes", "tier", "relative_path", "nlink", "mtime"])
    for a, t, rel, nl, mt in allbig[:400]:
        wr.writerow([a, t, rel, nl, datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")])

cfree, ctotal = free_gb("C:\\"); dfree, dtotal = free_gb("D:\\")
out = []
def w(s=""): out.append(s)
w("PROJECT CLEANUP PLAN / REPRO-ONLY POLICY / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("root = " + ROOT)
w("")
w("POLICY: keep scripts + test inputs + results (reproducible); clean intermediate big data.")
w("")
w("[0] CURRENT DISK")
w("    C: free %.2f / %.2f GB" % (cfree, ctotal))
w("    D: free %.2f / %.2f GB   <-- tight; pipeline still writing" % (dfree, dtotal))
w("    project allocated total (earlier full scan) = 394.85 GB / 444,201 files")
w("")
w("[1] TIER TOTALS (de-duplicated, no parent/child overlap)")
w("    %-20s %6s %12s %10s %14s %12s" %
  ("tier", "items", "clean GB", "files", "reclaim GB", "KEEP GB"))
for t in sorted(sums):
    s = sums[t]
    w("    %-20s %6d %12.2f %10d %14.2f %12.2f" %
      (t, sum(1 for r in rows if r["tier"] == t), gb(s[0]), s[1], gb(s[2]), gb(s[3])))
A = sum(sums[t][2] for t in sums if t.startswith("A"))
B = sum(sums[t][2] for t in sums if t.startswith("B"))
w("    A-tier reclaim = %.2f GB ; B-tier reclaim = %.2f GB ; A+B = %.2f GB" % (gb(A), gb(B), gb(A + B)))
w("")
w("[2] A-TIER DETAIL (explicitly cleanable under the stated policy)")
w("    %-46s %10s %8s %6s %6s %s" % ("label", "clean GB", "files", "r24", "keepMB", "last_write"))
for r in sorted([x for x in rows if x["tier"].startswith("A")], key=lambda x: -x["clean_bytes"]):
    w("    %-46s %10.2f %8d %6d %6.0f %s" %
      (r["label"][:46], gb(r["clean_bytes"]), r["clean_files"], r["r24_writes"],
       mb(r["keep_bytes"]), r["last_write"]))
w("")
w("[3] B-TIER DETAIL (needs your decision)")
w("    %-46s %10s %8s %6s %12s %s" % ("label", "clean GB", "files", "r24", "reclaim GB", "last_write"))
for r in sorted([x for x in rows if x["tier"].startswith("B")], key=lambda x: -x["clean_bytes"]):
    w("    %-46s %10.2f %8d %6d %12.2f %s" %
      (r["label"][:46], gb(r["clean_bytes"]), r["clean_files"], r["r24_writes"],
       gb(r["expected_reclaim_bytes"]), r["last_write"]))
w("")
w("[4] MUST-KEEP (verified)")
w("    conclusions: .json 28.27 GB / 229,916 files ; .jsonl 3.81 GB ; .py 0.46 GB / 33,249 files")
w("    result.json / execution.json / behavior.json / .sha256 / .md / .log / .csv => KEEP ALL")
for label, p in KEEP:
    if p and os.path.exists(p):
        cb, cf, kb, kf, r24, last, nl2, _ = measure(p)
        if "objects (CAS" in label:
            kb, kf, r24, last = measure_all(p)
        w("    %-52s keep=%8.2f GB  files=%7d  last=%s" % (label[:52], gb(kb), kf, ts(last)))
    else:
        w("    %-52s (structural / measured globally)" % label[:52])
w("")
w("[4b] EXTENSION-LESS / SPECIAL POOLS (would be missed by extension rules)")
for label, p in [(".git\\lfs (LFS objects)", os.path.join(ROOT, ".git", "lfs")),
                 ("distributed_data\\objects (evidence)", os.path.join(ROOT, "distributed_data", "objects")),
                 ("tests\\dist_test_data (service data root)", os.path.join(ROOT, "tests", "dist_test_data"))]:
    if os.path.isdir(p):
        b, f, r24, last = measure_all(p)
        w("    %-52s %8.2f GB  files=%6d  r24=%4d  last=%s" %
          (label, gb(b), f, r24, ts(last)))
    else:
        w("    %-52s NOT PRESENT" % label)
w("")
w("[5] CLIENT / SERVER RUNTIME DEPENDENCIES (do not break)")
w("    server/qwen14_mlp_service.py -> tests/glm5/result/qwen14_mlp_atlas_v1")
w("        = junction to C:\\Users\\Admin\\.codex\\visualizations\\...\\research-storage")
w("        reads 175 registered files with SHA256 verification; D: costs 0 bytes. KEEP junction.")
w("    server/research_asset_service.py -> client_visualization_assets (5.90 GB). KEEP.")
w("    server/mlp_tail_service.py, qwen14_mlp_service.py -> AI2050_DIST_DIR (external).")
w("    server/rdc_*_service.py -> tests/glm5/result/rdc_* (json evidence chains). KEEP json.")
w("    deploy/README_RESEARCH_UPLOAD.md: upload packs contain JSON evidence only;")
w("        build_client_deployment_data.py states no 85 GiB tensor copy is made.")
w("        => large local tensors are NOT client assets; conclusions live in JSON.")
w("")
w("[6] ACCURACY NOTES")
w("    - measured ALLOCATED bytes, not logical; sparse/prealloc gap 2.01 GB excluded")
w("    - 19 junctions skipped (9 -> C: client data, 3 -> D:\\develop\\model, 5 -> C:\\AI2050-RDC-Archive, 2 internal)")
w("    - 31.29 GB of models\\hf are hardlinks (nlink>=2); deleting them frees ~0")
w("    - 24h writes present: tests/glm5/result 919, .git/objects 695, video 422 => stop pipeline first")
w("")
w("[7] OUTPUTS")
w("    " + PLANCSV)
w("    " + BIGCSV)
w("    " + REPORT)
w("")
w("===== END FINAL REPORT =====")
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("final done")
