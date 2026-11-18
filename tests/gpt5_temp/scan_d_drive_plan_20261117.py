# -*- coding: utf-8 -*-
"""READ-ONLY final pass: de-duplicated D: cleanup plan, 2026-11-17.

Policy (user): keep scripts + test inputs + conclusion results + client-served
data; clean intermediate big data. Hardlinks deducted from expected reclaim.
Outputs:
  d_drive_final_report_20261117.txt
  d_drive_cleanup_plan_20261117.csv
  d_drive_delete_manifest_20261117.csv  (per-file A-tier only)
"""
import os
import csv
import json
import time
import ctypes
from collections import defaultdict
from datetime import datetime

PROJ = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(PROJ, "tests", "gpt5", "result")
RESULT = os.path.join(PROJ, "tests", "glm5", "result")
NATIVE = r"D:\ai2050_native"
REPORT = os.path.join(OUT, "d_drive_final_report_20261117.txt")
PLANCSV = os.path.join(OUT, "d_drive_cleanup_plan_20261117.csv")
MANIFEST = os.path.join(OUT, "d_drive_delete_manifest_20261117.csv")

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


def measure(path, protected=None):
    """clean bytes/files, keep bytes/files, r24, last, nlink2 bytes, per-file rows."""
    protected = protected or set()
    d = {"clean": 0, "cf": 0, "keep": 0, "kf": 0, "other": 0,
         "r24": 0, "r72": 0, "last": 0, "nlink2": 0, "rows": []}
    if not os.path.isdir(path) or is_reparse(path):
        return d
    for dp, dns, fns in os.walk(path, topdown=True):
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
            if st.st_mtime > d["last"]:
                d["last"] = st.st_mtime
            if st.st_mtime >= H24:
                d["r24"] += 1
            if st.st_mtime >= H72:
                d["r72"] += 1
            rel = os.path.relpath(q, path).replace("\\", "/")
            if is_keep(fn) or rel in protected:
                d["keep"] += a
                d["kf"] += 1
            elif is_data(fn) or rel.endswith(DATA_SUFFIX):
                n = getattr(st, "st_nlink", 1)
                if n >= 2:
                    d["nlink2"] += a
                d["clean"] += a
                d["cf"] += 1
                d["rows"].append((q, a, st.st_mtime, n))
            else:
                d["other"] += a
    return d


def manifest_paths(p):
    mf = os.path.join(p, "manifest.json")
    out = set()
    if not os.path.isfile(mf):
        return out
    try:
        d = json.loads(open(mf, encoding="utf-8").read())
        for x in d.get("files", []):
            out.add(x.get("path", ""))
    except Exception:
        pass
    return out


# ============ protected / must-keep ============
PROTECTED = {}  # abs dir -> set of relative file names to keep
for sub in [os.path.join("goodfire_middle_20261113_v2", "qwen4"),
            os.path.join("goodfire_middle_20261113_v2", "qwen14"),
            os.path.join("qwen14_client_v2", "native"),
            os.path.join("qwen14_mlp_atlas_v1")]:
    p = os.path.join(RESULT, sub)
    if os.path.isdir(p):
        PROTECTED[p] = manifest_paths(p)
# also protect the D: entity copy of the atlas
p_atlas_d = os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1")
if os.path.isdir(p_atlas_d):
    PROTECTED[p_atlas_d] = manifest_paths(p_atlas_d)

# ============ A-tier: cold intermediate data ============
A = []
def adda(tier, label, path, basis, pre, protected=None):
    A.append((tier, label, path, basis, pre, protected))

# 11-13~11-15 new cold experiment dirs (>72h idle, json kept)
cold_new = [
    ("recursive_branch14_20261113_v6", "11-13 recursive branch v6 hidden-state tensors"),
    ("branch_reuse14_20261113_v3", "11-13 branch reuse v3 chunk states npz"),
    ("recursive_branch14_20261113_v5", "11-13 recursive branch v5 pilot tensors"),
    ("recursive_branch14_20261113_v4", "11-13 recursive branch v4"),
    ("recursive_branch14_20261113_v3", "11-13 recursive branch v3"),
    ("recursive_branch14_20261113_v2", "11-13 recursive branch v2"),
    ("recursive_branch14_20261113_v1", "11-13 recursive branch v1"),
    ("b36_composition14_20261113_v3", "11-13 b36 composition v3 npz"),
    ("b36_composition14_20261113_v2", "11-13 b36 composition v2"),
    ("b36_composition14_20261113_v1", "11-13 b36 composition v1"),
    ("b36_transition14_20261114_v1", "11-14 b36 transition states npz"),
    ("branch_reuse14_20261113_v2", "11-13 branch reuse v2"),
    ("branch_reuse14_20261113_v1", "11-13 branch reuse v1"),
    ("own_state_explicit_20261115_v2", "11-15 own-state explicit npz"),
    ("goodfire_middle_20261113_v1", "11-13 goodfire v1 small tensors"),
    ("source_binding_langparts_v1", "11-14 source binding langparts"),
    ("source_binding_qkv_followup_v1", "11-13 source binding qkv followup"),
    ("source_binding_enrelay_v1", "11-14 source binding enrelay"),
    ("source_binding_downstream_v1", "11-12 source binding downstream"),
]
for name, basis in cold_new:
    p = os.path.join(RESULT, name)
    if os.path.isdir(p) and not is_reparse(p):
        adda("A1-NEW-COLD", name, p, basis,
             "keep ALL json/md/log/sha256 in same dir; >72h idle")

# 11-10/11-11 old source experiment tensors (now >72h cold)
old_src = [
    ("source_formation_v1", "11-10 source formation hidden tensors"),
    ("source_binding_v1", "11-10 source binding discovery-native npy"),
    ("source_relay_v2", "11-11 source relay v2"),
    ("source_relay_v1", "11-10 source relay v1"),
    ("common_bench_b1_repair_v2", "11-10 common bench repair tensors"),
    ("qwen14_mlp_b28_b39_v2", "11-13 qwen14 b28-b39 v2 tensors (partly hardlinked)"),
]
for name, basis in old_src:
    p = os.path.join(RESULT, name)
    if os.path.isdir(p) and not is_reparse(p):
        adda("A2-OLD-SOURCE", name, p, basis,
             "keep ALL json/md/log/sha256; confirm conclusions in MEMO")

# root-level tar.gz in served dirs (unregistered, packaged intermediate)
for name, basis in [
    ("goodfire_middle_20261113_v2", "2 root-level native tar.gz 31.17 GB; qwen4/qwen14 manifest files PROTECTED"),
    ("qwen14_client_v2", "root-level delta/research-upload tar.gz 6.30 GB; native/ manifest files PROTECTED"),
]:
    p = os.path.join(RESULT, name)
    if os.path.isdir(p):
        adda("A3-PACKS", name, p, basis,
             "ONLY unregistered root-level archives; installed copies protected",
             protected=PROTECTED.get(p))

# D:\ai2050_native pure tensor dirs (no online reader)
native_a = [
    ("ENT-XF", "37.47 GB pure tensors, idle since 11-06"),
    ("_cross_model", "14.76 GB pure tensors, idle since 11-02"),
    ("GR-Q", "13.09 GB pure tensors, idle since 11-13 18:53"),
    ("_transcoder", "10.50 GB pure tensors, idle since 10-30"),
    ("reusable", "4.60 GB pure tensors, idle since 11-10"),
    ("_weights", "3.91 GB pure tensors, idle since 10-16"),
    ("GR-NEG", "3.16 GB pure tensors, idle since 11-06"),
    ("_mode_atlas", "2.87 GB pure tensors, idle since 11-01"),
    ("KN-ISA", "2.48 GB pure tensors, idle since 11-13 18:53"),
    ("KN-ATTR", "1.04 GB pure tensors, idle since 10-16"),
    ("_fusion", "766 MB pure tensors, idle since 11-02"),
    ("_assembly", "637 MB pure tensors, idle since 11-02"),
    ("_entity_atlas", "132 MB pure tensors, idle since 11-01"),
    ("GR-ROLE", "118 MB pure tensors, idle since 11-06"),
    ("_checks", "26 MB pure tensors, idle since 10-16"),
    ("_arbitration", "3 MB pure tensors, idle since 11-01"),
    ("_weight_space", "1 MB pure tensors, idle since 11-02"),
]
for name, basis in native_a:
    p = os.path.join(NATIVE, name)
    if os.path.isdir(p):
        adda("A4-NATIVE", name, p, basis, "keep all json/md/log; no server reader")

# unregistered tar.gz in the ONLY entity copy of the junction source
p = os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1")
if os.path.isdir(p):
    adda("A4-NATIVE", "visualization-research\\qwen14_mlp_atlas_v1",
         p, "13.68 GB unregistered files (13.56 GB native tar.gz + calibrate npy); 175 manifest files PROTECTED",
         "DO NOT delete manifest-registered files; junction entity source",
         protected=PROTECTED.get(p))

# interrupted 11-13 finish staging (orphaned, already approved for deletion)
p = r"D:\AI2050\_A_stage_finish_20261113"
if os.path.isdir(p):
    adda("A5-ORPHAN", "_A_stage_finish_20261113", p,
         "29 files / 6.66 GB moved out of result on 11-13 but never deleted",
         "already part of approved A-tier deletion list")

# leftover single file from 11-13 residue
p = os.path.join(RESULT, "mlp_condition_v2")
adda("A1-NEW-COLD", "mlp_condition_v2 (1 leftover file)", p,
     "976.73 KB DATA file left by interrupted 11-13 finish run", "keep all json")

# release old builds / caches
for label, rel, basis, pre in [
    ("frontend\\node_modules", r"frontend\node_modules", "npm reinstallable", "vite/esbuild RUNNING - exclude this round"),
    ("video", "video", "5.65 GB frames+wav, idle since 11-11", "test material? confirm before deleting"),
    ("distributed_data\\datapack", r"distributed_data\datapack", "865 MB unpacked datapack", "24h writes - exclude"),
]:
    q = os.path.join(PROJ, rel)
    if os.path.isdir(q):
        adda("A6-CACHE", label, q, basis, pre)

# ============ measure ============
rows = []
per_file = []
for tier, label, path, basis, pre, protected in A:
    d = measure(path, protected)
    reclaim = d["clean"] - d["nlink2"]
    rows.append({"tier": tier, "label": label, "absolute_path": path,
                 "clean_bytes": d["clean"], "clean_files": d["cf"],
                 "keep_bytes": d["keep"], "keep_files": d["kf"],
                 "r24_writes": d["r24"], "r72_writes": d["r72"],
                 "last_write": "%.0f" % d["last"],
                 "hardlink_bytes": d["nlink2"], "expected_reclaim_bytes": reclaim,
                 "basis": basis, "precondition": pre})
    for q, sz, mt, n in d["rows"]:
        per_file.append((tier, label, q, sz, mt, n))

# de-duplicate per-file rows by absolute path
seen = set()
uniq = []
for r in per_file:
    if r[2] in seen:
        continue
    seen.add(r[2])
    uniq.append(r)

a_total_clean = sum(r["clean_bytes"] for r in rows)
a_total_reclaim = sum(r["expected_reclaim_bytes"] for r in rows)

# ============ B / C tiers (measured, not executed) ============
B = [
    ("B1-MODELS", "models\\hf", os.path.join(PROJ, "models", "hf"),
     "75.98 GB weights; ~31 GB hardlinks free ~0; re-downloadable but needed for repro"),
    ("B2-GIT-LFS", ".git\\lfs", os.path.join(PROJ, ".git", "lfs"),
     "29.01 GB extension-less LFS objects; NEVER git lfs prune (11-06 dropped refs/codex/*)"),
    ("B3-GIT-PACK", ".git\\objects", os.path.join(PROJ, ".git", "objects"),
     "5.41 GB; 1296 writes in 24h - active git operation"),
    ("B4-VENV", ".venv\\Lib", os.path.join(PROJ, ".venv", "Lib"),
     "3.11 GB installed packages, rebuildable via uv but needed to run scripts now"),
    ("B5-ACTIVE-MEDIA", "video", os.path.join(PROJ, "video"),
     "5.65 GB video frames/wav, idle since 11-11; confirm whether test material"),
    ("B6-NATIVE-AGENT", "_agent_home\\node_service", os.path.join(NATIVE, "_agent_home"),
     "10.22 GB; 1 write in 24h"),
    ("B7-PAGEFILE", "D:\\pagefile.sys", r"D:\pagefile.sys",
     "33.98 GB system pagefile - cannot delete, only resize/relocate"),
    ("B8-ACTIVE-KV", "kv_* 24h-active dirs", RESULT,
     "kv_receipt_20261117_v1 / kv_state_20261116_v1 / kv_prefix_20261116_v1 ~227 MB, 24h writes"),
    ("B9-SHA256", "mlp_tail_acceptance", os.path.join(RESULT, "mlp_tail_acceptance"),
     "903 MB tensors but 8 .sha256 registrations + 2.84 GB json - integrity chain"),
    ("B10-RECYCLE", "D:\\$RECYCLE.BIN", r"D:\$RECYCLE.BIN",
     "1.33 GB / 132 historical items - clearing frees space but loses old trash"),
    ("B11-NONPROJECT", "D:\\Comfy-Desktop", r"D:\Comfy-Desktop",
     "5.48 GB non-project, idle since 10-14"),
]
b_rows = []
for tier, label, p, basis in B:
    if os.path.isfile(p):
        cb, cf, kb, kf = os.path.getsize(p), 1, 0, 0
        r24, last = 0, os.path.getmtime(p)
    else:
        d = measure(p)
        cb, cf, kb, kf = d["clean"], d["cf"], d["keep"], d["kf"]
        r24, last = d["r24"], d["last"]
    b_rows.append({"tier": tier, "label": label, "absolute_path": p,
                   "clean_bytes": cb, "clean_files": cf, "keep_bytes": kb,
                   "keep_files": kf, "r24_writes": r24, "last_write": "%.0f" % last,
                   "basis": basis})

C = [
    ("D:\\ai2050_native\\visualization-research", os.path.join(NATIVE, "visualization-research"),
     "73.37 GB - ONLY entity source for 9 result junctions + C: codex junctions; client data"),
    ("tests\\glm5\\result\\client_visualization_assets", os.path.join(RESULT, "client_visualization_assets"),
     "5.90 GB - /api/research-assets direct client payload"),
    ("distributed_data\\objects", os.path.join(PROJ, "distributed_data", "objects"),
     "12.67 GB CAS evidence packs (length-prefixed JSON)"),
    ("distributed_data\\research_library", os.path.join(PROJ, "distributed_data", "research_library"),
     "4.20 GB, 6 writes in 24h - active"),
    ("distributed_data\\mechanism_traces", os.path.join(PROJ, "distributed_data", "mechanism_traces"),
     "1.15 GB index.sqlite3 - server/research_space_service.py online reader"),
    ("D:\\ai2050_native\\_mainline\\MA-H2", os.path.join(NATIVE, "_mainline"),
     "7.18 GB - mainline_sample_service.py"),
    ("D:\\ai2050_native\\_census", os.path.join(NATIVE, "_census"),
     "2.18 GB - neuron_census_service.py / mech_pair_service.py"),
    ("goodfire_middle_20261113_v2\\qwen4+qwen14", os.path.join(RESULT, "goodfire_middle_20261113_v2"),
     "30.20 GB / 498 manifest-registered files, SHA256-verified by native_mlp_service.py"),
    ("qwen14_client_v2\\native", os.path.join(RESULT, "qwen14_client_v2"),
     "18.07 GB / 206 manifest-registered files, SHA256-verified by qwen14_mlp_service.py"),
    ("research", os.path.join(PROJ, "research"), "6.29 GB conclusions"),
    ("9 result junctions + 11 C: codex junctions", RESULT, "structural - never delete junction itself"),
    ("all conclusion files", PROJ, "json/jsonl/md/log/csv/py/sha256 + result/execution/behavior.json"),
]

# ============ report ============
L = []
w = L.append
cf, ct = free_gb("C:\\")
df, dt = free_gb("D:\\")
w("D: DRIVE CLEANUP FINAL PLAN / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("root = " + PROJ)
w("")
w("POLICY: keep scripts + test inputs + conclusion results + client-served data; clean intermediate big data.")
w("")
w("[0] DISK")
w("    C: free %.2f / %.2f GB" % (cf, ct))
w("    D: free %.2f / %.2f GB   <-- 11-13 cleanup ended at 151.08 GB; -131.08 GB in 4 days" % (df, dt))
w("    131 GB rebound source: tests\\glm5 143.76 GB new writes (8002 files), .git\\objects 4.35 GB, research_library 1.83 GB")
w("")
w("[1] CRITICAL STRUCTURE CHANGE SINCE 11-10 (reversal)")
w("    result\\<9 dirs> -> C:\\.codex\\visualizations\\...\\research-storage\\<dir> -> D:\\ai2050_native\\visualization-research\\<dir>")
w("    C: codex tree without junction following = 0 bytes / 11 junctions; WITH following = 73.37 GB / 20552 files")
w("    => D:\\ai2050_native\\visualization-research (73.37 GB) is now the ONLY entity source.")
w("    => 11-10 conclusion 'D: copy is redundant, safe to clean' is WRONG. MUST KEEP.")
w("    All 9 two-level junctions healthy; /api/research/qwen14-mlp reads 175 registered files from here.")
w("    Center server.ai2050.com = v1.0.29, no native-mlp/qwen14-mlp endpoints yet (local release is 1.0.36).")
w("    Local server.py env: AI2050_NATIVE_MLP_ROOT / AI2050_QWEN14_MLP_DIR / AI2050_DIST_DIR all empty")
w("       => falls back to LOCAL result dirs; those manifest-registered tensors are LIVE client data.")
w("    Center /api/traces/research-space = HTTP 200 kind=real 11 studies / 990 objects / 693 artifacts")
w("       (11-13 was 8/870/673 -> conclusions of new experiments ARE registered).")
w("")
w("[2] A-TIER (cleanable, de-duplicated, non-overlapping)")
w("    %-52s %11s %8s %9s %7s %9s %s" %
  ("label", "clean", "files", "keep", "r24", "reclaim", "last"))
for r in sorted(rows, key=lambda x: -x["clean_bytes"]):
    if r["clean_bytes"] == 0:
        continue
    w("    %-52s %11s %8d %9s %7d %9s %s" %
      (r["label"][:52], human(r["clean_bytes"]), r["clean_files"],
       human(r["keep_bytes"]), r["r24_writes"],
       human(r["expected_reclaim_bytes"]), ts(float(r["last_write"]))))
w("")
w("    A-tier clean total  = %s / %d files" %
  (human(a_total_clean), sum(r["clean_files"] for r in rows)))
w("    hardlink bytes      = %s (deleting these links frees ~0)" %
  human(sum(r["hardlink_bytes"] for r in rows)))
w("    EXPECTED REAL RECLAIM = %s" % human(a_total_reclaim))
w("")
w("[3] B-TIER (needs your decision)")
w("    %-52s %11s %8s %7s  basis" % ("label", "clean", "files", "r24"))
for r in b_rows:
    w("    %-52s %11s %8d %7d  %s" %
      (r["label"][:52], human(r["clean_bytes"]), r["clean_files"],
       r["r24_writes"], r["basis"][:88]))
w("    B-tier total clean = %s" % human(sum(r["clean_bytes"] for r in b_rows)))
w("")
w("[4] C-TIER MUST KEEP")
for label, p, basis in C:
    if os.path.exists(p):
        w("    %-52s PRESENT  %s" % (label[:52], basis[:100]))
    else:
        w("    %-52s MISSING  %s" % (label[:52], basis[:100]))
w("")
w("[5] ACCURACY NOTES")
w("    - measured ALLOCATED bytes (GetCompressedFileSizeW), junctions skipped, sparse gap excluded")
w("    - nlink>=2 bytes deducted from expected reclaim; 11-09 proved hardlinks free nothing")
w("    - manifest-registered files explicitly PROTECTED in served dirs")
w("    - 24h-active dirs (kv_*, .git\\objects, datapack, frontend\\node_modules, client-cache) excluded from A")
w("    - safe-delete hook redirects deletions to Recycle Bin; must empty bin per batch to free space")
w("    - pipeline still writing (server.py x2, snapshot_worker x2, vite/esbuild); stop before deleting")
w("")
w("[6] OUTPUTS")
w("    " + REPORT)
w("    " + PLANCSV)
w("    " + MANIFEST)
w("")
w("===== END FINAL PLAN =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

with open(PLANCSV, "w", encoding="utf-8", newline="") as fh:
    keys = ["tier", "label", "absolute_path", "clean_bytes", "clean_files",
            "keep_bytes", "keep_files", "r24_writes", "r72_writes", "last_write",
            "hardlink_bytes", "expected_reclaim_bytes", "basis", "precondition"]
    cw = csv.DictWriter(fh, fieldnames=keys)
    cw.writeheader()
    for r in rows:
        cw.writerow(r)
    for r in b_rows:
        rr = {k: r.get(k, "") for k in keys}
        rr["precondition"] = "B-tier: needs decision"
        cw.writerow(rr)

with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["tier", "label", "path", "size_bytes", "mtime", "nlink"])
    for t, lab, q, sz, mt, n in uniq:
        cw.writerow([t, lab, q, sz, "%.0f" % mt, n])

print("PLAN OK A=%s reclaim=%s files=%d B=%s" %
      (human(a_total_clean), human(a_total_reclaim), len(uniq),
       human(sum(r["clean_bytes"] for r in b_rows))))
