# -*- coding: utf-8 -*-
"""READ-ONLY final plan v2 (2026-11-17). Fixes v1 issues:
 - manifest protection now uses ABSOLUTE paths (v1 used wrong relative base)
 - served dirs: only root-level unregistered archives are cleanable
 - 24h-active dirs moved out of A-tier
 - extension-less pools (git LFS, recycle bin) measured by total, not by extension
 - kv_* measured on the 3 real dirs, not the whole result tree
Outputs: d_drive_final_report_v2_20261117.txt / d_drive_cleanup_plan_v2_20261117.csv
         / d_drive_delete_manifest_v2_20261117.csv
"""
import os
import csv
import json
import time
import ctypes
from datetime import datetime

PROJ = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(PROJ, "tests", "gpt5", "result")
RESULT = os.path.join(PROJ, "tests", "glm5", "result")
NATIVE = r"D:\ai2050_native"
REPORT = os.path.join(OUT, "d_drive_final_report_v2_20261117.txt")
PLANCSV = os.path.join(OUT, "d_drive_cleanup_plan_v2_20261117.csv")
MANIFEST = os.path.join(OUT, "d_drive_delete_manifest_v2_20261117.csv")

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


def is_data(fn):
    low = fn.lower()
    return os.path.splitext(low)[1] in DATA_EXT or low.endswith(DATA_SUFFIX)


def is_keep(fn):
    return os.path.splitext(fn.lower())[1] in KEEP_EXT


def measure(path, protect_abs=None, only_root=False, all_files=False):
    """protect_abs: set of ABSOLUTE paths that must never be deleted.
    only_root: only consider files directly inside `path`.
    all_files: count every file as clean (for extension-less pools)."""
    protect_abs = protect_abs or set()
    d = {"clean": 0, "cf": 0, "keep": 0, "kf": 0, "other": 0, "total": 0,
         "r24": 0, "r72": 0, "last": 0, "nlink2": 0, "rows": []}
    if not os.path.isdir(path) or is_reparse(path):
        return d
    for dp, dns, fns in os.walk(path, topdown=True):
        if only_root and dp != path:
            dns[:] = []
            continue
        dns[:] = [x for x in dns if not is_reparse(os.path.join(dp, x))]
        for fn in fns:
            q = os.path.join(dp, fn)
            try:
                st = os.lstat(q)
            except OSError:
                continue
            if is_reparse(q):
                continue
            a = alloc(q) or st.st_size
            d["total"] += a
            if st.st_mtime > d["last"]:
                d["last"] = st.st_mtime
            if st.st_mtime >= H24:
                d["r24"] += 1
            if st.st_mtime >= H72:
                d["r72"] += 1
            if q in protect_abs:
                d["keep"] += a
                d["kf"] += 1
                continue
            if is_keep(fn):
                d["keep"] += a
                d["kf"] += 1
            elif all_files or is_data(fn):
                n = getattr(st, "st_nlink", 1)
                if n >= 2:
                    d["nlink2"] += a
                d["clean"] += a
                d["cf"] += 1
                d["rows"].append((q, a, st.st_mtime, n))
            else:
                d["other"] += a
    return d


def manifest_abs(p):
    """absolute paths of all manifest-registered files in dir p"""
    mf = os.path.join(p, "manifest.json")
    out = set()
    if not os.path.isfile(mf):
        return out
    try:
        d = json.loads(open(mf, encoding="utf-8").read())
        for x in d.get("files", []):
            out.add(os.path.normcase(os.path.abspath(os.path.join(p, x.get("path", "")))))
        out.add(os.path.normcase(os.path.abspath(mf)))
    except Exception:
        pass
    return out


# ---- absolute protection set: every manifest-registered client file ----
PROTECT = set()
SERVED = [
    os.path.join(RESULT, "goodfire_middle_20261113_v2", "qwen4"),
    os.path.join(RESULT, "goodfire_middle_20261113_v2", "qwen14"),
    os.path.join(RESULT, "qwen14_client_v2", "native"),
    os.path.join(RESULT, "qwen14_mlp_atlas_v1"),
    os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1"),
]
for p in SERVED:
    if os.path.isdir(p):
        PROTECT |= manifest_abs(p)

A = []


def adda(tier, label, path, basis, pre, only_root=False, protect=None):
    A.append((tier, label, path, basis, pre, only_root, protect or PROTECT))


# ---- A1: 11-13~11-15 cold experiment tensors (>72h idle) ----
for name in [
    "recursive_branch14_20261113_v6", "recursive_branch14_20261113_v5",
    "recursive_branch14_20261113_v4", "recursive_branch14_20261113_v3",
    "recursive_branch14_20261113_v2", "recursive_branch14_20261113_v1",
    "branch_reuse14_20261113_v3", "branch_reuse14_20261113_v2",
    "branch_reuse14_20261113_v1", "b36_composition14_20261113_v3",
    "b36_composition14_20261113_v2", "b36_composition14_20261113_v1",
    "b36_transition14_20261114_v1", "own_state_explicit_20261115_v2",
    "goodfire_middle_20261113_v1", "source_binding_langparts_v1",
    "source_binding_qkv_followup_v1", "source_binding_enrelay_v1",
    "source_binding_downstream_v1", "mlp_condition_v2",
]:
    p = os.path.join(RESULT, name)
    if os.path.isdir(p) and not is_reparse(p):
        adda("A1-COLD-NEW", name, p,
             "intermediate npy/npz generated 11-13~11-15, >72h idle",
             "keep ALL json/md/log/sha256 in the same directory")

# ---- A2: 11-10/11-11 old source experiments (now cold) ----
for name in ["source_formation_v1", "source_binding_v1", "source_relay_v2",
             "source_relay_v1", "common_bench_b1_repair_v2",
             "qwen14_mlp_b28_b39_v2"]:
    p = os.path.join(RESULT, name)
    if os.path.isdir(p) and not is_reparse(p):
        adda("A2-COLD-SOURCE", name, p,
             "11-10/11-11 experiment tensors, >72h idle; conclusions in json",
             "confirm conclusions written to MEMO/ledger before deleting")

# ---- A3: root-level UNREGISTERED archives in served dirs (only_root) ----
for name, basis in [
    ("goodfire_middle_20261113_v2", "2 root-level native tar.gz 31.17 GB; qwen4/+qwen14/ manifest files PROTECTED"),
    ("qwen14_client_v2", "root-level delta 6.27 GB + research-upload 29 MB; native/ manifest files PROTECTED"),
]:
    p = os.path.join(RESULT, name)
    if os.path.isdir(p):
        adda("A3-PACKS", name, p, basis,
             "ONLY unregistered root-level archives; installed manifest files kept",
             only_root=True)

# ---- A4: D:\ai2050_native pure tensors (no online reader) ----
for name in ["ENT-XF", "_cross_model", "GR-Q", "_transcoder", "reusable",
             "_weights", "GR-NEG", "_mode_atlas", "KN-ISA", "KN-ATTR",
             "_fusion", "_assembly", "_entity_atlas", "GR-ROLE", "_checks",
             "_arbitration", "_weight_space"]:
    p = os.path.join(NATIVE, name)
    if os.path.isdir(p):
        adda("A4-NATIVE", name, p, "pure intermediate tensors, idle >72h, no server reader",
             "keep all json/md/log in the same directory")

# ---- A5: orphaned staging from interrupted 11-13 run ----
p = r"D:\AI2050\_A_stage_finish_20261113"
if os.path.isdir(p):
    adda("A5-ORPHAN", "_A_stage_finish_20261113", p,
         "29 files / 6.66 GB moved out of result on 11-13 but never deleted",
         "already part of the approved 11-13 A-tier deletion list")

# ---- measure A ----
rows = []
per_file = []
for tier, label, path, basis, pre, only_root, protect in A:
    d = measure(path, protect, only_root=only_root)
    rows.append({"tier": tier, "label": label, "absolute_path": path,
                 "clean_bytes": d["clean"], "clean_files": d["cf"],
                 "keep_bytes": d["keep"], "keep_files": d["kf"],
                 "r24_writes": d["r24"], "last_write": "%.0f" % d["last"],
                 "hardlink_bytes": d["nlink2"],
                 "expected_reclaim_bytes": d["clean"] - d["nlink2"],
                 "basis": basis, "precondition": pre})
    per_file += [(tier, label, q, sz, mt, n) for q, sz, mt, n in d["rows"]]

seen = set()
uniq = []
for r in per_file:
    k = os.path.normcase(r[2])
    if k in seen or k in PROTECT:
        continue
    seen.add(k)
    uniq.append(r)

a_clean = sum(r["clean_bytes"] for r in rows)
a_reclaim = sum(r["expected_reclaim_bytes"] for r in rows)

# ---- B tier (measured precisely) ----
B = []
def addb(tier, label, path, basis, measure_mode="data"):
    if measure_mode == "total":
        d = measure(path, all_files=True)
        cb, cf = d["total"], d["cf"] + d["kf"]
    elif measure_mode == "protected":
        d = measure(path, protect_abs=PROTECT)
        cb, cf = d["clean"], d["cf"]
    else:
        d = measure(path)
        cb, cf = d["clean"], d["cf"]
    B.append({"tier": tier, "label": label, "absolute_path": path,
              "clean_bytes": cb, "clean_files": cf,
              "keep_bytes": d["keep"], "r24_writes": d["r24"],
              "last_write": "%.0f" % d["last"], "basis": basis})

addb("B1-MODELS", "models\\hf", os.path.join(PROJ, "models", "hf"),
     "75.98 GB weights; ~31 GB hardlinks free ~0; re-downloadable but needed for repro")
addb("B2-GIT-LFS", ".git\\lfs", os.path.join(PROJ, ".git", "lfs"),
     "29.01 GB extension-less LFS objects; NEVER git lfs prune (11-06 dropped refs/codex/*)",
     "total")
addb("B3-GIT-PACK", ".git\\objects", os.path.join(PROJ, ".git", "objects"),
     "5.41 GB; 1341 writes in 24h - active git operation", "total")
addb("B4-VENV", ".venv\\Lib", os.path.join(PROJ, ".venv", "Lib"),
     "3.11 GB installed packages, rebuildable via uv but needed to run scripts now")
addb("B5-ACTIVE-MEDIA", "video", os.path.join(PROJ, "video"),
     "5.65 GB frames/wav idle since 11-11; confirm whether test material", "total")
addb("B6-NATIVE-AGENT", "_agent_home", os.path.join(NATIVE, "_agent_home"),
     "10.22 GB node_service, 1 write in 24h")
addb("B9-SHA256", "mlp_tail_acceptance", os.path.join(RESULT, "mlp_tail_acceptance"),
     "903 MB tensors but 8 .sha256 registrations + 2.84 GB json - integrity chain")
addb("B10-JUNCTION-UNREG", "visualization-research\\qwen14_mlp_atlas_v1",
     os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1"),
     "UNREGISTERED files only (13.56 GB tar.gz) inside the ONLY junction entity source; "
     "175 manifest files PROTECTED. Publication pack - confirm center deployed before deleting",
     "protected")
addb("B11-ACTIVE-KV", "kv_receipt_20261117_v1", os.path.join(RESULT, "kv_receipt_20261117_v1"),
     "83 MB, 38 writes in 24h")
addb("B11-ACTIVE-KV", "kv_state_20261116_v1", os.path.join(RESULT, "kv_state_20261116_v1"),
     "76 MB, 83 writes in 24h")
addb("B11-ACTIVE-KV", "kv_prefix_20261116_v1", os.path.join(RESULT, "kv_prefix_20261116_v1"),
     "69 MB, 67 writes in 24h")
addb("B12-CACHE", "distributed_data\\datapack", os.path.join(PROJ, "distributed_data", "datapack"),
     "865 MB unpacked datapack, 3 writes in 24h", "total")
addb("B13-CACHE", "frontend\\node_modules", os.path.join(PROJ, "frontend", "node_modules"),
     "616 MB data files; vite/esbuild RUNNING - do not touch")
addb("B14-NONPROJECT", "D:\\Comfy-Desktop", r"D:\Comfy-Desktop",
     "5.48 GB non-project, idle since 10-14", "total")
addb("B15-RECYCLE", "D:\\$RECYCLE.BIN", r"D:\$RECYCLE.BIN",
     "1.33 GB / 132 historical items; clearing frees space but loses old trash", "total")

# pagefile is a system file, reported separately
pf = r"D:\pagefile.sys"
PF_SIZE = os.path.getsize(pf) if os.path.isfile(pf) else 0

C = [
    ("D:\\ai2050_native\\visualization-research", os.path.join(NATIVE, "visualization-research"),
     "73.37 GB - ONLY entity source for 9 result junctions + C: codex junctions"),
    ("client_visualization_assets", os.path.join(RESULT, "client_visualization_assets"),
     "5.90 GB - /api/research-assets direct client payload"),
    ("distributed_data\\objects", os.path.join(PROJ, "distributed_data", "objects"),
     "12.67 GB CAS evidence packs (length-prefixed JSON)"),
    ("distributed_data\\research_library", os.path.join(PROJ, "distributed_data", "research_library"),
     "4.20 GB, 6 writes in 24h - active"),
    ("distributed_data\\mechanism_traces", os.path.join(PROJ, "distributed_data", "mechanism_traces"),
     "1.15 GB index.sqlite3 - research_space_service.py online reader"),
    ("D:\\ai2050_native\\_mainline", os.path.join(NATIVE, "_mainline"),
     "7.18 GB - mainline_sample_service.py"),
    ("D:\\ai2050_native\\_census", os.path.join(NATIVE, "_census"),
     "2.18 GB - neuron_census_service.py / mech_pair_service.py"),
    ("goodfire qwen4+qwen14 manifest files", os.path.join(RESULT, "goodfire_middle_20261113_v2"),
     "30.20 GB / 498 SHA256-verified files - native_mlp_service.py"),
    ("qwen14_client_v2\\native", os.path.join(RESULT, "qwen14_client_v2", "native"),
     "18.07 GB / 206 SHA256-verified files - qwen14_mlp_service.py"),
    ("research", os.path.join(PROJ, "research"), "6.29 GB conclusions"),
    ("9 result junctions + 11 C: codex junctions", RESULT, "structural - never delete junction itself"),
    ("all conclusion files", PROJ, "json/jsonl/md/log/csv/py/sha256 + result/execution/behavior.json"),
]

L = []
w = L.append
cfree, ctot = free_gb("C:\\")
dfree, dtot = free_gb("D:\\")
w("D: DRIVE CLEANUP FINAL PLAN v2 / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("root = " + PROJ)
w("")
w("POLICY: keep scripts + test inputs + conclusion results + client-served data; clean intermediate big data.")
w("")
w("[0] DISK")
w("    C: free %.2f / %.2f GB" % (cfree, ctot))
w("    D: free %.2f / %.2f GB  <-- 11-13 cleanup ended at 151.08 GB; -131.08 GB in 4 days" % (dfree, dtot))
w("    rebound: tests\\glm5 143.76 GB new writes (8002 files), .git\\objects 4.35 GB, research_library 1.83 GB")
w("")
w("[1] STRUCTURE CHANGE SINCE 11-10 (REVERSAL OF PREVIOUS CONCLUSION)")
w("    result\\<9 dirs> -> C:\\.codex\\visualizations\\...\\research-storage\\<dir>")
w("                    -> D:\\ai2050_native\\visualization-research\\<dir>   (two-level junction)")
w("    C: codex WITHOUT following junctions = 0 bytes / 11 junctions; WITH = 73.37 GB / 20552 files")
w("    => D:\\ai2050_native\\visualization-research is now the ONLY entity source. MUST KEEP.")
w("    => 11-10 statement 'D: copy is redundant, safe to clean' is WRONG.")
w("    All 9 junction targets healthy. Center v1.0.29 has NO native-mlp/qwen14-mlp endpoints")
w("    (local release 1.0.36) => those packs are NOT yet deployed to center.")
w("    Local server.py env AI2050_NATIVE_MLP_ROOT / AI2050_QWEN14_MLP_DIR / AI2050_DIST_DIR all EMPTY")
w("       => local service reads manifest-registered tensors from result dirs (LIVE client data).")
w("    Center /api/traces/research-space = HTTP 200 kind=real 11 studies / 990 objects / 693 artifacts")
w("       (11-13: 8/870/673) -> conclusions of new experiments ARE registered at center.")
w("")
w("[2] A-TIER: CLEANABLE (de-duplicated, r24=0, manifest files protected)")
w("    %-50s %11s %7s %9s %9s %s" % ("label", "clean", "files", "keep", "reclaim", "last"))
for r in sorted(rows, key=lambda x: -x["clean_bytes"]):
    if r["clean_bytes"] == 0:
        continue
    w("    %-50s %11s %7d %9s %9s %s" %
      (r["label"][:50], human(r["clean_bytes"]), r["clean_files"],
       human(r["keep_bytes"]), human(r["expected_reclaim_bytes"]),
       ts(float(r["last_write"]))))
w("")
w("    A-tier clean total     = %s / %d files" %
  (human(a_clean), sum(r["clean_files"] for r in rows)))
w("    hardlink(nlink>=2)     = %s  (these free ~0)" % human(sum(r["hardlink_bytes"] for r in rows)))
w("    EXPECTED REAL RECLAIM  = %s" % human(a_reclaim))
w("    D: free after A-tier   ~ %.2f GB" % (dfree + a_reclaim / 1024 ** 3))
w("")
w("[3] B-TIER: NEEDS YOUR DECISION")
w("    %-50s %11s %7s %7s  basis" % ("label", "size", "files", "r24"))
for r in B:
    w("    %-50s %11s %7d %7d  %s" %
      (r["label"][:50], human(r["clean_bytes"]), r["clean_files"],
       r["r24_writes"], r["basis"][:86]))
w("    B-tier total = %s" % human(sum(r["clean_bytes"] for r in B)))
w("    D:\\pagefile.sys = %s (system file; only resize/relocate, never delete)" % human(PF_SIZE))
w("")
w("[4] C-TIER: MUST KEEP")
for label, p, basis in C:
    w("    %-50s %-8s %s" % (label[:50], "PRESENT" if os.path.exists(p) else "MISSING", basis[:96]))
w("")
w("[5] METHOD / RISKS")
w("    - allocated bytes via GetCompressedFileSizeW; junctions skipped; sparse gap excluded")
w("    - manifest protection uses ABSOLUTE paths (v1 bug fixed); served dirs use only_root for archives")
w("    - nlink>=2 bytes deducted from reclaim (11-09 proved hardlinks free nothing)")
w("    - 24h-active dirs excluded from A-tier; pipeline still writing (server.py x2,")
w("      snapshot_worker x2, vite/esbuild, kv_* experiments) -> stop before deleting")
w("    - safe-delete hook redirects to Recycle Bin; must empty bin per batch to free space")
w("    - A-tier keeps every directory structure and ALL json/md/log/sha256/py")
w("")
w("[6] OUTPUTS")
w("    " + REPORT)
w("    " + PLANCSV)
w("    " + MANIFEST)
w("")
w("===== END FINAL PLAN v2 =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

keys = ["tier", "label", "absolute_path", "clean_bytes", "clean_files",
        "keep_bytes", "keep_files", "r24_writes", "last_write",
        "hardlink_bytes", "expected_reclaim_bytes", "basis", "precondition"]
with open(PLANCSV, "w", encoding="utf-8", newline="") as fh:
    cw = csv.DictWriter(fh, fieldnames=keys)
    cw.writeheader()
    for r in rows:
        cw.writerow({k: r.get(k, "") for k in keys})
    for r in B:
        rr = {k: r.get(k, "") for k in keys}
        rr["precondition"] = "B-tier: needs decision"
        cw.writerow(rr)

with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["tier", "label", "path", "size_bytes", "mtime", "nlink"])
    for t, lab, q, sz, mt, n in uniq:
        cw.writerow([t, lab, q, sz, "%.0f" % mt, n])

print("PLAN v2 OK A=%s reclaim=%s files=%d B=%s" %
      (human(a_clean), human(a_reclaim), len(uniq),
       human(sum(r["clean_bytes"] for r in B))))
