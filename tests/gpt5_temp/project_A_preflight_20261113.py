# -*- coding: utf-8 -*-
"""READ-ONLY preflight for A-tier cleanup (plan dated 2026-11-11, re-checked 11-13).
Outputs:
  project_A_preflight_20261113.txt        decision data
  project_A_delete_manifest_20261113.csv  fresh exact per-file delete manifest
No deletion / rename / move is performed here.
"""
import os
import csv
import time
import json
import ctypes
import shutil
from datetime import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "gpt5", "result")
PLAN = os.path.join(OUT, "project_repro_cleanup_plan_20261111.csv")
REPORT = os.path.join(OUT, "project_A_preflight_20261113.txt")
MANIFEST = os.path.join(OUT, "project_A_delete_manifest_20261113.csv")

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


def bin_stat(drive):
    size = files = 0
    B = drive + r"$RECYCLE.BIN"
    if os.path.isdir(B):
        for dp, dns, fns in os.walk(B):
            for fn in fns:
                try:
                    size += os.lstat(os.path.join(dp, fn)).st_size
                    files += 1
                except OSError:
                    pass
    return size, files


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


def ts(x):
    return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else "-"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


L = []
w = L.append
w("A-TIER CLEANUP PREFLIGHT / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("root = " + ROOT)
w("")
cf, ct = free_gb("C:\\")
df, dt = free_gb("D:\\")
w("[0] DISK")
w("    C: free %.2f / %.2f GB" % (cf, ct))
w("    D: free %.2f / %.2f GB" % (df, dt))
bD = bin_stat("D:\\")
w("    D recycle bin: %s / %d files" % (human(bD[0]), bD[1]))
w("")

# ---- active processes (cmdline only; open_files enumeration is unstable here) ----
w("[1] ACTIVE PROCESSES touching the project")
procs = []
try:
    import psutil
    for pr in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(pr.info.get("cmdline") or [])
        except Exception:
            continue
        if any(k in cl for k in ("Ai2050-OpenOne", "ai2050_native", "server\\server.py",
                                 "snapshot_worker", "vite", "glm5")):
            procs.append((pr.pid, pr.info.get("name") or "?", cl[:150]))
except Exception as e:
    w("    psutil failed: " + repr(e))
for pid, name, cl in procs[:30]:
    w("    pid=%-7s %-22s %s" % (pid, name[:22], cl))
w("    matches = %d" % len(procs))
w("")

# ---- load the 11-11 A-tier plan ----
plan = []
with open(PLAN, "r", encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        if row["tier"].startswith("A"):
            plan.append(row)
w("[2] A-TIER ENTRIES IN 11-11 PLAN: %d" % len(plan))
w("")

manifest_rows = []
summary = []
w("[3] RE-MEASURED ON REAL DISK (now)")
w("    %-44s %11s %8s %8s %8s %7s %7s %8s %s" %
  ("label", "clean", "files", "planned", "delta", "r24", "nlink2", "keep", "last_write"))
for row in plan:
    p = row["absolute_path"]
    planned = int(row["clean_bytes"])
    exists = os.path.exists(p)
    cb = cf_n = kb = kf_n = r24 = nl2 = 0
    last = 0
    junc = 0
    if exists:
        if os.path.isfile(p) and not is_reparse(p):
            st = os.lstat(p)
            a = alloc(p) or st.st_size
            low = p.lower()
            ext = os.path.splitext(low)[1]
            is_data = ext in DATA_EXT or low.endswith(DATA_SUFFIX)
            if is_data:
                cb, cf_n = a, 1
                last = st.st_mtime
                r24 = 1 if last >= H24 else 0
                nl2 = a if getattr(st, "st_nlink", 1) >= 2 else 0
                manifest_rows.append((row["tier"], row["label"], p, a,
                                      st.st_mtime, getattr(st, "st_nlink", 1)))
            else:
                kb, kf_n = a, 1
        elif os.path.isdir(p):
            for dp, dns, fns in os.walk(p, topdown=True):
                keepd = []
                for d in dns:
                    q = os.path.join(dp, d)
                    if is_reparse(q):
                        junc += 1
                    else:
                        keepd.append(d)
                dns[:] = keepd
                for fn in fns:
                    q = os.path.join(dp, fn)
                    try:
                        st = os.lstat(q)
                    except OSError:
                        continue
                    if is_reparse(q):
                        junc += 1
                        continue
                    a = alloc(q) or st.st_size
                    low = fn.lower()
                    ext = os.path.splitext(low)[1]
                    if st.st_mtime > last:
                        last = st.st_mtime
                    if st.st_mtime >= H24:
                        r24 += 1
                    if ext in KEEP_EXT:
                        kb += a
                        kf_n += 1
                    elif ext in DATA_EXT or low.endswith(DATA_SUFFIX):
                        cb += a
                        cf_n += 1
                        n = getattr(st, "st_nlink", 1)
                        if n >= 2:
                            nl2 += a
                        manifest_rows.append((row["tier"], row["label"], q, a,
                                              st.st_mtime, n))
    delta = cb - planned
    w("    %-44s %11s %8d %8s %+8s %7d %7s %8s %s%s" %
      (row["label"][:44], human(cb), cf_n, human(planned), human(delta),
       r24, human(nl2), human(kb), ts(last),
       "" if exists else "  !! MISSING"))
    summary.append({
        "tier": row["tier"], "label": row["label"], "path": p,
        "exists": exists, "planned": planned, "clean": cb, "files": cf_n,
        "keep": kb, "keep_files": kf_n, "r24": r24, "nlink2": nl2,
        "junctions": junc, "last": last,
        "basis": row["basis"], "precondition": row["precondition"],
    })
w("")

tot_clean = sum(s["clean"] for s in summary)
tot_files = sum(s["files"] for s in summary)
tot_nl2 = sum(s["nlink2"] for s in summary)
w("[4] FRESH TOTALS")
w("    clean bytes = %s  files = %d" % (human(tot_clean), tot_files))
w("    hardlink(nlink>=2) bytes = %s  => expected real reclaim = %s" %
  (human(tot_nl2), human(tot_clean - tot_nl2)))
w("    11-11 planned A-tier = 150.83 GB ; drift = %+.2f GB" %
  ((tot_clean - 150.83 * 1024 ** 3) / 1024 ** 3))
w("    entries with 24h writes = %d" % len([s for s in summary if s["r24"]]))
for s in summary:
    if s["r24"]:
        w("      !! %-52s r24=%d last=%s" % (s["label"][:52], s["r24"], ts(s["last"])))
w("    entries with junctions inside = %d" % len([s for s in summary if s["junctions"]]))
w("")

# ---- release directory listing (to decide deployment bundle) ----
w("[5] release/ TOP LEVEL (for deployed-version check)")
rel = os.path.join(ROOT, "release")
try:
    for n in sorted(os.listdir(rel)):
        q = os.path.join(rel, n)
        try:
            st = os.lstat(q)
            w("    %-64s %10s  %s  %s" %
              (n[:64], human(st.st_size) if os.path.isfile(q) else "<dir>",
               ts(st.st_mtime), "JUNCTION" if is_reparse(q) else ""))
        except OSError:
            pass
except Exception as e:
    w("    " + repr(e))
w("")

# ---- must-keep structural checks ----
w("[6] MUST-KEEP STRUCTURAL CHECKS")
checks = [
    ("client_visualization_assets", os.path.join(ROOT, "tests", "glm5", "result", "client_visualization_assets")),
    ("distributed_data\\objects", os.path.join(ROOT, "distributed_data", "objects")),
    ("distributed_data\\research_library", os.path.join(ROOT, "distributed_data", "research_library")),
    ("research", os.path.join(ROOT, "research")),
    ("deploy\\README_RESEARCH_UPLOAD.md", os.path.join(ROOT, "deploy", "README_RESEARCH_UPLOAD.md")),
    ("result\\qwen14_mlp_atlas_v1 (junction)", os.path.join(ROOT, "tests", "glm5", "result", "qwen14_mlp_atlas_v1")),
]
for label, q in checks:
    w("    %-48s %s%s" % (label, "PRESENT" if os.path.lexists(q) else "MISSING",
                          "  [JUNCTION]" if os.path.lexists(q) and is_reparse(q) else ""))
w("")

# ---- abandoned .workbuddy A-tier staging ----
w("[7] ABORTED .workbuddy A-TIER RESIDUE")
ws = r"C:\Users\Admin\WorkBuddy\2026-11-09-10-35-58"
stage = os.path.join(ws, "_A_staging_v2")
bk = r"D:\AI2050\_cleanup_backup\workbuddy_A_20261110"
w("    staging dir exists = %s" % os.path.isdir(stage))
if os.path.isdir(stage):
    sb = sf = 0
    for dp, dns, fns in os.walk(stage):
        for fn in fns:
            try:
                sb += os.lstat(os.path.join(dp, fn)).st_size
                sf += 1
            except OSError:
                pass
    w("    staging content = %s / %d files" % (human(sb), sf))
w("    backup exists = %s  manifest=%s" %
  (os.path.isdir(bk), os.path.isfile(os.path.join(bk, "_backup_manifest.txt"))))
w("")

w("[8] PRECONDITION NOTES")
w("    A2-CAS-BLOBS: 11-11 plan says confirm derived JSON registered at center")
w("       (GET /api/traces/research-space) before clearing blob store.")
w("    A3 release\\deployment-20261108: confirm it is not the currently deployed bundle.")
w("    A4 frontend\\node_modules: npm ci restores; check vite not running.")
w("")
w("===== END PREFLIGHT =====")

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["tier", "label", "path", "size_bytes", "mtime", "nlink"])
    for t, lab, p, a, mt, n in manifest_rows:
        cw.writerow([t, lab, p, a, "%.0f" % mt, n])

with open(os.path.join(OUT, "project_A_summary_20261113.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

print("PREFLIGHT OK clean=%s files=%d r24entries=%d" %
      (human(tot_clean), tot_files, len([s for s in summary if s["r24"]])))
