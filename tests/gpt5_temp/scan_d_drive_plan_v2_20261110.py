# -*- coding: utf-8 -*-
"""Generate a de-duplicated, executable cleanup plan (non-overlapping paths)."""
import os
import csv
from datetime import datetime

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_final_report_20261110.txt"
CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_cleanup_plan_v2_20261110.csv"
PROJ = r"D:\AI2050\Ai2050-OpenOne"
NATIVE = r"D:\ai2050_native"
RES = os.path.join(PROJ, "tests", "glm5", "result")
NOW = __import__("time").time()
H24 = NOW - 24 * 3600

L = []
def w(s=""):
    L.append(str(s))
def gb(n):
    return "%.2f GB" % (n / 1024.0 ** 3)
def ts(t):
    try:
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"

DATA_EXT = {".npy", ".npz", ".pkl", ".pickle", ".bin", ".pt", ".pth", ".parquet",
            ".sqlite3", ".sqlite", ".db", ".h5", ".hdf5", ".arrow", ".msgpack",
            ".onnx", ".gz", ".zip", ".tar", ".tgz", ".tmp", ".dat", ".index",
            ".mmap", ".safetensors", ".gguf", ".conllu", ".bz2", ".xz", ".zst"}
RESULT_EXT = {".json", ".jsonl", ".md", ".log", ".txt", ".sha256", ".csv", ".tsv",
              ".png", ".jpg", ".svg", ".pdf", ".html", ".yaml", ".yml", ".ndjson"}

def develop_weight_index():
    idx = set()
    base = r"D:\develop\model"
    for root, dirs, files in os.walk(base):
        for fn in files:
            if os.path.splitext(fn)[1].lower() != ".safetensors":
                continue
            try:
                idx.add((fn.lower(), os.stat(os.path.join(root, fn)).st_size))
            except Exception:
                pass
    return idx

DEV_IDX = develop_weight_index()

def measure(path, mode, keep_names=()):
    """return clean_bytes, keep_bytes, clean_files, keep_files, last, r24"""
    cb = kb = 0
    cf = kf = 0
    last = 0
    r24 = 0
    for root, dirs, files in os.walk(path):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except Exception:
                continue
            sz = st.st_size
            if st.st_mtime > last:
                last = st.st_mtime
            if st.st_mtime >= H24:
                r24 += 1
            low = fn.lower()
            ext = os.path.splitext(low)[1]
            is_data = (ext in DATA_EXT or low.endswith((".gz.tmp", ".npy.tmp", ".before-import")))
            if mode == "dup_only":
                if (low, sz) in DEV_IDX:
                    cb += sz
                    cf += 1
                else:
                    kb += sz
                    kf += 1
            elif mode == "all":
                cb += sz
                cf += 1
            elif mode == "data_only":
                if is_data:
                    cb += sz
                    cf += 1
                else:
                    kb += sz
                    kf += 1
            elif mode == "old_builds":
                rel = os.path.relpath(p, path).replace("\\", "/")
                if any(rel.startswith(k) for k in keep_names):
                    kb += sz
                    kf += 1
                else:
                    cb += sz
                    cf += 1
    return cb, kb, cf, kf, last, r24

# (priority, category, path, mode, keep_names, basis, precondition)
P = [
    ("A1", "MIRROR-REDUNDANT", os.path.join(NATIVE, "visualization-research"), "all", (),
     "73.37 GB is a 1:1 mirror of C: research-storage (9 experiments, identical file counts); 9 result-side junctions all healthy and point to C:",
     "Keep the C: copy. If C: is ever cleaned, re-point the 9 junctions to D: first"),
    ("A1", "CAS-ORPHAN-BLOBS", os.path.join(PROJ, "distributed_data", "research_native_blobs"), "all", (),
     "668 orphan blobs, nlink=1, last write 11-05, pure .npy/.npz/.gz intermediate tensors; upload protocol explicitly excludes large tensors",
     "Confirm the derived JSON evidence is registered in the center"),
    ("A2", "RESULT-P3-TENSORS", os.path.join(RES, "branch_style_joint_v1"), "data_only", (),
     "24.33 GB npy from the 11-08 batch; only data tensors are removed", "Keep all json / result / execution in place"),
    ("A2", "RESULT-P3-TENSORS", os.path.join(RES, "mlp_tail_b28_b31_v1"), "data_only", (),
     "14.76 GB npy", "Keep json"),
    ("A2", "RESULT-P3-TENSORS", os.path.join(RES, "mlp_tail_goodfire_v1"), "data_only", (),
     "13.93 GB npy", "Keep json"),
    ("A2", "RESULT-P3-TENSORS", os.path.join(RES, "mlp_tail_acceptance"), "data_only", (),
     "0.88 GB tensors (2.83 GB json results kept)", "Keep the 8 .sha256 registrations and all json"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "ENT-XF"), "data_only", (),
     "37.47 GB pure tensors, zero result files, no code reader found", "Verify no planned experiment reuses raw fields"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "GR-Q"), "data_only", (),
     "13.09 GB pure tensors", "Same"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_cross_model"), "data_only", (),
     "14.76 GB pure tensors", "Same"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_transcoder"), "data_only", (),
     "10.50 GB pure tensors", "Same"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_mode_atlas"), "data_only", (),
     "2.87 GB pure tensors", "Same"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_weights"), "all", (),
     "3.91 GB intermediate weight snapshots", "Confirm not referenced by native runtime"),
    ("A3", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "KN-ISA"), "data_only", (),
     "2.48 GB tensors, but 42 writes in last 24h", "WAIT until the pipeline stops"),
    ("A4", "BUILD-OLD", os.path.join(PROJ, "release"), "old_builds",
     ("desktop-1.0.13/", "deployment-20261108/", "center-lang-map-1.0.32/"),
     "old desktop-1.0.8..1.0.12, 7.14 GB mlp-tail tar.gz, 0.99 GB release zip; newest build + sha256 kept",
     "Keep newest build and all .sha256"),
    ("A4", "CACHE-REBUILDABLE", os.path.join(PROJ, "frontend", "node_modules"), "all", (),
     "1.19 GB / 41224 files, regenerable via npm install", "Rebuild cost only; 88 writes in 24h so stop dev server first"),
    ("B1", "MODEL-DUPLICATE", os.path.join(PROJ, "models", "hf"), "dup_only", (),
     "only the 11 safetensors byte-identical by name+size to develop\\model are redundant; the other model files are unique. 3 model dirs are junctions (zero bytes)",
     "Delete ONLY the duplicated shards; confirm which path the run scripts use"),
    ("B2", "NATIVE-RUNTIME", os.path.join(NATIVE, "_agent_home", "node_service"), "data_only", (),
     "9.68 GB tensors + 0.53 GB json", "Confirm node service is stopped"),
    ("B3", "MEDIA", os.path.join(PROJ, "video"), "all", (),
     "5.52 GB / 93173 frame files", "Confirm these are not client media assets"),
    ("B4", "GIT-LFS", os.path.join(PROJ, ".git", "lfs"), "all", (),
     "29.01 GB LFS objects", "DO NOT use git lfs prune (11-06 proved it deletes refs/codex/* objects)"),
    ("C", "CLIENT-ASSET", os.path.join(RES, "client_visualization_assets"), "keep", (),
     "5.90 GB / 2401 files, served by /api/research-assets", "MUST KEEP"),
    ("C", "SERVER-DEP", os.path.join(NATIVE, "_mainline", "MA-H2"), "keep", (),
     "7.18 GB / 9 files, read by mainline_sample_service.py", "MUST KEEP"),
    ("C", "SERVER-DEP", os.path.join(NATIVE, "_census"), "keep", (),
     "2.18 GB / 115 files, read by neuron_census_service.py and 4 other services", "MUST KEEP"),
    ("C", "EVIDENCE-DB", os.path.join(PROJ, "distributed_data", "research_library"), "keep", (),
     "3.35 GB / 6559 files, evidence library for center registration; 819 writes in 24h", "MUST KEEP"),
    ("C", "EVIDENCE-DB", os.path.join(PROJ, "distributed_data", "objects"), "keep", (),
     "12.67 GB / 220 content-addressed objects, header = length-prefix + JSON evidence", "MUST KEEP"),
    ("C", "SYSTEM", r"D:\pagefile.sys", "keep", (),
     "33.98 GB system pagefile", "resize/move only, never delete"),
]

w("D DRIVE FINAL ANALYSIS REPORT (READ-ONLY)  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("Policy: KEEP test scripts + materials + results (reproducible) + client/server")
w("        visible assets; DELETE intermediate big-data folders.")
w("")
w("[0] KEY STRUCTURAL FINDING")
w("    tests\\glm5\\result has 9 NTFS junctions -> C:\\Users\\Admin\\.codex\\visualizations")
w("    \\2026\\11\\<uuid>\\research-storage\\<experiment>.  All 9 are healthy.")
w("    D:\\ai2050_native\\visualization-research is a byte-for-byte 1:1 mirror of")
w("    that C: store (73.36 GB vs 73.36 GB, identical per-experiment file counts).")
w("    => server endpoints resolve through C:, so the D: copy is redundant;")
w("    => but C: must NOT be cleaned, or all 9 client views break.")
w("    Upload protocol (deploy/README_RESEARCH_UPLOAD.md) ships small JSON research")
w("    bundles (e.g. ai2050-qwen14-research-upload.tar.gz, 20 MB) to the center;")
w("    build_client_deployment_data.py states no 85 GiB tensor copy is made.")
w("    => large local tensors are NOT client assets; conclusions live in JSON.")
w("    A1 note: the D: mirror contains 0.93 GB of conclusion json as well; those")
w("    are byte-identical to the C: copy (kept). If you prefer a strictly")
w("    json-preserving delete, clear only the 72.44 GB tensor portion on D:.")
w("")

w("[1] NON-OVERLAPPING CLEANUP PLAN")
w("    %-4s %-22s %10s %10s %8s %6s %s" % (
    "pri", "category", "clean", "keep", "files", "24h", "path"))
rows = []
for pri, cat, path, mode, keep_names, basis, pre in P:
    if not os.path.isdir(path) and not os.path.isfile(path):
        w("    %-4s %-22s %10s %10s %8s %6s %s [ABSENT]" % (pri, cat, "-", "-", "-", "-", path))
        continue
    if mode == "keep":
        cb, kb, cf, kf, last, r24 = measure(path, "all") if os.path.isdir(path) else (0, 0, 0, 0, 0, 0)
        cb, kb = 0, kb + cb
        if os.path.isfile(path):
            kb = os.stat(path).st_size
            kf = 1
    else:
        cb, kb, cf, kf, last, r24 = measure(path, mode, keep_names)
    w("    %-4s %-22s %10s %10s %8d %6d %s" % (
        pri, cat, gb(cb), gb(kb), cf + kf, r24, path.replace("D:\\", "")[:60]))
    rows.append([pri, cat, path, mode, cb, kb, cf, kf, r24, ts(last), basis, pre])

with open(CSV, "w", encoding="utf-8", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["priority", "category", "absolute_path", "mode", "clean_bytes", "keep_bytes",
                 "clean_files", "keep_files", "writes_24h", "last_write", "basis", "precondition"])
    for r in rows:
        cw.writerow(r)
w("")
w("    CSV: " + CSV)
w("")

a1 = sum(r[4] for r in rows if r[0] == "A1")
a2 = sum(r[4] for r in rows if r[0] == "A2")
a3 = sum(r[4] for r in rows if r[0] == "A3")
a4 = sum(r[4] for r in rows if r[0] == "A4")
bt = sum(r[4] for r in rows if r[0].startswith("B"))
keep = sum(r[5] for r in rows if r[0] == "C")
w("[2] TOTALS (real freeable bytes; D: has no hard-link views left, nlink>=2 = 0)")
w("    A1 redundant mirror + orphan CAS blobs : %s" % gb(a1))
w("    A2 remaining P3 result tensors         : %s" % gb(a2))
w("    A3 other native intermediate tensors   : %s" % gb(a3))
w("    A4 old builds + rebuildable cache      : %s" % gb(a4))
w("    ------------------------------------------------")
w("    A total (clear intermediate)           : %s" % gb(a1 + a2 + a3 + a4))
w("    B total (needs your decision)          : %s" % gb(bt))
w("    C total (explicitly keep)              : %s" % gb(keep))
w("")

w("[3] EXECUTION PRECONDITIONS")
w("    1. C: research-storage is the live client data source - do NOT clean C:.")
w("       If C: must be freed, first re-point the 9 junctions to the D: copy.")
w("    2. safe-delete hook redirects deletes to D: recycle bin; the bin quota is")
w("       tiny. Use per-batch delete -> SHEmptyRecycleBin -> verify (proven 11-09).")
w("    3. Directory-level staging (os.rename into a temp dir, then rmtree) is")
w("       300-1000x faster than per-file delete under this hook.")
w("    4. Active writers in 24h: visualization-research 20477, research_library 819,")
w("       node_modules 88, video 497, KN-ISA 42. Stop pipelines before deleting.")
w("    5. Verify center registration of derived JSON evidence before dropping the")
w("       84.71 GB CAS blobs (curl https://server.ai2050.com/api/traces/research-space).")
w("    6. Never run 'git lfs prune' here (11-06 incident).")
w("")
w("=== END (READ-ONLY, no file was deleted, moved or renamed) ===")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("done", len(L), "lines")
