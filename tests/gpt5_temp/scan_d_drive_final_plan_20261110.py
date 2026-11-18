# -*- coding: utf-8 -*-
"""Final pass: junction verification + actionable cleanup plan under new policy."""
import os
import csv
import time
from collections import defaultdict
from datetime import datetime

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_final_plan_20261110.txt"
CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_cleanup_plan_20261110.csv"
PROJ = r"D:\AI2050\Ai2050-OpenOne"
NATIVE = r"D:\ai2050_native"
RES = os.path.join(PROJ, "tests", "glm5", "result")
RATTR = 0x400
NOW = time.time()
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
            ".mmap", ".safetensors", ".gguf", ".conllu", ".bz2", ".xz", ".zst",
            ".npy.tmp", ".gz.tmp"}
RESULT_EXT = {".json", ".jsonl", ".md", ".log", ".txt", ".sha256", ".csv", ".tsv",
              ".png", ".jpg", ".svg", ".pdf", ".html", ".yaml", ".yml", ".ndjson"}

def stat_dir(path):
    """return (total, data, result, files, data_files, result_files, last, r24)"""
    tot = dat = rst = 0
    nf = nfd = nfr = 0
    last = 0
    r24 = 0
    for root, dirs, files in os.walk(path):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except Exception:
                continue
            low = fn.lower()
            ext = os.path.splitext(low)[1]
            sz = st.st_size
            tot += sz
            nf += 1
            if ext in DATA_EXT or low.endswith((".gz.tmp", ".npy.tmp", ".before-import")):
                dat += sz
                nfd += 1
            elif ext in RESULT_EXT or low.endswith(".sha256"):
                rst += sz
                nfr += 1
            if st.st_mtime > last:
                last = st.st_mtime
            if st.st_mtime >= H24:
                r24 += 1
    return tot, dat, rst, nf, nfd, nfr, last, r24

w("D DRIVE FINAL PLAN (READ-ONLY)  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# ---------- 1. reparse points ----------
w("[1] NTFS REPARSE POINTS on D: (junction / symlink) - pruned in size accounting")
reps = []
def find_reps(path, depth=0):
    if depth > 6:
        return
    try:
        it = os.scandir(path)
    except Exception:
        return
    with it:
        while True:
            try:
                e = next(it)
            except StopIteration:
                break
            except Exception:
                break
            try:
                st = e.stat(follow_symlinks=False)
                if getattr(st, "st_file_attributes", 0) & RATTR:
                    try:
                        tgt = os.readlink(e.path)
                    except Exception:
                        tgt = "(readlink failed)"
                    reps.append((e.path, tgt))
                    continue
                if e.is_dir(follow_symlinks=False):
                    find_reps(e.path, depth + 1)
            except Exception:
                continue
find_reps("D:\\")
w("    found %d" % len(reps))
for p, t in reps[:30]:
    w("    %s" % p)
    w("        -> %s" % str(t)[:150])
w("")

q1 = os.path.join(RES, "qwen14_mlp_atlas_v1")
q2 = os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1")
isj1 = False
try:
    isj1 = bool(os.stat(q1, follow_symlinks=False).st_file_attributes & RATTR)
except Exception:
    pass
w("    result-side pack is reparse: %s" % isj1)
w("    => if it is a junction, the 26.95 GB exists ONCE (native side);")
w("       deleting the native source breaks /api/research/qwen14-mlp.")
w("")

# ---------- 2. distributed_data/objects nature ----------
w("[2] distributed_data\\objects (12.67 GB, 220 files) - what is it?")
obj = os.path.join(PROJ, "distributed_data", "objects")
extc = defaultdict(lambda: [0, 0])
samples = []
if os.path.isdir(obj):
    for root, dirs, files in os.walk(obj):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                sz = os.stat(p).st_size
            except Exception:
                continue
            ext = os.path.splitext(fn)[1].lower() or "(none)"
            extc[ext][0] += 1
            extc[ext][1] += sz
            if len(samples) < 3:
                try:
                    head = open(p, "rb").read(60)
                except Exception:
                    head = b""
                samples.append((fn, sz, head[:40]))
w("    extension breakdown: " + ", ".join(
    "%s:%d/%s" % (k, v[0], gb(v[1])) for k, v in sorted(extc.items(), key=lambda x: -x[1][1])[:8]))
for fn, sz, head in samples:
    w("    sample %s (%s) head=%r" % (fn[:45], gb(sz), head))
w("")

# ---------- 3. candidate dirs ----------
# (priority, category, abs path, basis, risk/precondition)
CAND = [
    ("A", "CAS-ORPHAN-BLOBS", os.path.join(PROJ, "distributed_data", "research_native_blobs"),
     "668 orphan blobs, nlink=1, last write 11-05; intermediate native tensors, not results",
     "was earlier marked server-authoritative; confirm center has registered the derived JSON evidence first"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "visualization-research", "state_formation_v1"),
     "25.07 GB tensors + 0.14 GB json; publish script state_formation_publish.py exists",
     "keep the json/manifest; delete only npy/npz/gz"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "visualization-research", "record_selection_v3"),
     "18.17 GB tensors + 0.06 GB json; record_selection_publish.py exists",
     "keep the json; confirm publish finished"),
    ("A", "RESULT-P3-TENSORS", os.path.join(RES, "branch_style_joint_v1"),
     "24.33 GB npy; kept as P3 in 11-09 cleanup", "keep result/execution json; confirm conclusions in MEMO"),
    ("A", "RESULT-P3-TENSORS", os.path.join(RES, "mlp_tail_b28_b31_v1"),
     "14.76 GB npy", "keep json; was active 11-08 23:32"),
    ("A", "RESULT-P3-TENSORS", os.path.join(RES, "mlp_tail_goodfire_v1"),
     "13.93 GB npy", "keep json"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "ENT-XF"),
     "37.47 GB pure tensors, zero result files, last write 11-06",
     "no client/server reader found; verify no future experiment needs the raw fields"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "GR-Q"),
     "13.09 GB pure tensors, zero result files", "same"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_cross_model"),
     "14.76 GB pure tensors, zero result files", "same"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_transcoder"),
     "10.50 GB pure tensors, zero result files", "same"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "_mode_atlas"),
     "2.87 GB pure tensors", "same"),
    ("A", "NATIVE-INTERMEDIATE", os.path.join(NATIVE, "KN-ISA"),
     "2.48 GB pure tensors; 42 writes in 24h", "ACTIVE - wait until pipeline stops"),
    ("A", "BUILD-OLD", os.path.join(PROJ, "release"),
     "10.58 GB; 7.14 GB tar.gz + 0.99 GB zip + old desktop-1.0.8..1.0.12",
     "keep newest build + sha256; old versions are reproducible"),
    ("A", "CACHE-REBUILDABLE", os.path.join(PROJ, "frontend", "node_modules"),
     "1.19 GB / 41224 files; regenerable via npm install", "rebuild cost only"),
    ("A", "UPLOAD-ARTIFACT", os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1"),
     "contains 13.56 GB ai2050-qwen14-native.tar.gz (upload/pack artifact)",
     "KEEP the 175 manifest-registered files; only the tar.gz is deletable, and it is the SAME file via junction"),
    ("B", "MODEL-DUPLICATE", r"D:\develop\model",
     "90.90 GB; 11 weight files (35.00 GB) are byte-identical by name+size to models\\hf",
     "runtime dependency not a test result; keep the copy actually referenced by scripts"),
    ("B", "MODEL-DUPLICATE", os.path.join(PROJ, "models"),
     "76.10 GB; duplicate partner of develop\\model", "same"),
    ("B", "GIT-LFS", os.path.join(PROJ, ".git", "lfs"),
     "29.01 GB LFS objects", "DO NOT git lfs prune (11-06 proven to delete refs/codex/* objects)"),
    ("B", "MEDIA", os.path.join(PROJ, "video"),
     "5.52 GB / 93173 files frame sequences", "client media asset? confirm before deleting"),
    ("B", "NATIVE-RUNTIME", os.path.join(NATIVE, "_agent_home", "node_service"),
     "10.22 GB (9.68 GB tensors + 0.53 GB json)", "node service data; confirm service stopped"),
    ("C", "SYSTEM", r"D:\pagefile.sys", "33.98 GB", "system file - resize/move only, never delete"),
    ("C", "SERVER-DEP", os.path.join(NATIVE, "_mainline", "MA-H2"),
     "7.18 GB / 9 files", "READ by mainline_sample_service.py - KEEP"),
    ("C", "SERVER-DEP", os.path.join(NATIVE, "_census"),
     "2.18 GB / 115 files", "READ by neuron_census_service.py etc - KEEP"),
    ("C", "CLIENT-ASSET", os.path.join(RES, "client_visualization_assets"),
     "5.90 GB / 2401 files (5.81 GB json)", "served by /api/research-assets - KEEP"),
    ("C", "EVIDENCE-DB", os.path.join(PROJ, "distributed_data", "research_library"),
     "3.35 GB / 6559 files", "evidence library for center registration - KEEP"),
    ("C", "EVIDENCE-DB", os.path.join(PROJ, "distributed_data", "objects"),
     "12.67 GB / 220 files", "CAS evidence objects - verify nature before any action"),
]

w("[3] CANDIDATE / KEEP TABLE")
w("    %-3s %-22s %10s %10s %8s %7s %s" % (
    "pri", "category", "total", "data(clean)", "files", "24h", "path"))
rows_out = []
for pri, c, p, basis, risk in CAND:
    if not os.path.isfile(p) and not os.path.isdir(p):
        w("    %-3s %-22s %10s %10s %8s %7s %s  [ABSENT]" % (pri, c, "-", "-", "-", "-", p))
        continue
    if os.path.isfile(p):
        st = os.stat(p)
        tot, dat, nf, nfd, r24, last = st.st_size, 0, 1, 0, 0, st.st_mtime
        rst = 0
    else:
        tot, dat, rst, nf, nfd, nfr, last, r24 = stat_dir(p)
    w("    %-3s %-22s %10s %10s %8d %7d %s" % (
        pri, c, gb(tot), gb(dat), nf, r24, p.replace("D:\\", "")[:66]))
    rows_out.append([pri, c, p, tot, dat, rst, nf, nfd, r24, ts(last), basis, risk])

with open(CSV, "w", encoding="utf-8", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["priority", "category", "absolute_path", "total_bytes", "data_bytes_cleanable",
                 "result_bytes_keep", "files", "data_files", "writes_24h", "last_write",
                 "basis", "risk_or_precondition"])
    for r in rows_out:
        cw.writerow(r)
w("")
w("    CSV: " + CSV)
w("")

w("[4] POLICY SUMMARY (computed)")
a_data = sum(r[4] for r in rows_out if r[0] == "A")
b_data = sum(r[4] for r in rows_out if r[0] == "B")
keep = sum(r[3] for r in rows_out if r[0] == "C")
w("    A (clear intermediate, data bytes)          : %s" % gb(a_data))
w("    B (needs decision, data bytes)              : %s" % gb(b_data))
w("    C (must keep, total bytes incl. pagefile)   : %s" % gb(keep))
w("    NOTE: A includes the qwen14 dir where only the tar.gz (~13.56 GB) is")
w("          actually deletable; adjust downward for the 175 registered files.")
w("")
w("=== END (READ-ONLY, no file touched) ===")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("done", len(L), "lines")
