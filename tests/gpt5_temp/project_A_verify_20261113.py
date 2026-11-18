# -*- coding: utf-8 -*-
"""READ-ONLY verification after A-tier cleanup: find residue + check keepers."""
import os
import csv
import time
import json
import ctypes
import shutil
from datetime import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "gpt5", "result")
MANIFEST = os.path.join(OUT, "project_A_delete_manifest_20261113.csv")
PLAN = os.path.join(OUT, "project_repro_cleanup_plan_20261111.csv")
REPORT = os.path.join(OUT, "project_A_verify_20261113.txt")
RESIDUE_CSV = os.path.join(OUT, "project_A_residue_20261113.csv")
STATE = os.path.join(OUT, "project_A_state_20261113.json")

EXCLUDE_LABELS = {"frontend\\node_modules", "mechanism_traces"}
REPARSE = 0x400
GFA = ctypes.windll.kernel32.GetFileAttributesW
GFA.restype = ctypes.c_uint32
GFA.argtypes = [ctypes.c_wchar_p]


def is_reparse(p):
    a = GFA(p)
    return a != 0xFFFFFFFF and bool(a & REPARSE)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def free_gb(d):
    fb = ctypes.c_ulonglong(0)
    tb = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(d),
                                               ctypes.byref(fb), ctypes.byref(tb), None)
    return fb.value / 1024 ** 3, tb.value / 1024 ** 3


L = []
w = L.append
w("POST-CLEANUP VERIFICATION / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")
cf, ct = free_gb("C:\\")
df, dt = free_gb("D:\\")
w("[0] DISK   C: free %.2f / %.2f GB    D: free %.2f / %.2f GB" % (cf, ct, df, dt))
st = json.load(open(STATE, "r", encoding="utf-8")) if os.path.isfile(STATE) else {}
w("    state: batches=%s files=%s deleted=%s freed=%s done=%s next_index=%s" %
  (st.get("batches"), st.get("deleted_files"), human(st.get("deleted_bytes", 0)),
   human(st.get("freed_bytes", 0)), st.get("done"), st.get("next_index")))
w("")

# ---- residue per manifest row ----
rows = list(csv.DictReader(open(MANIFEST, "r", encoding="utf-8", newline="")))
by_label = {}
residue = []
for r in rows:
    lab = r["label"]
    d = by_label.setdefault(lab, {"total": 0, "left": 0, "left_bytes": 0,
                                  "excluded": lab in EXCLUDE_LABELS})
    d["total"] += 1
    if lab in EXCLUDE_LABELS:
        continue
    p = r["path"]
    if os.path.isfile(p):
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = int(r["size_bytes"])
        d["left"] += 1
        d["left_bytes"] += sz
        residue.append((r["tier"], lab, p, sz, r["mtime"]))

w("[1] RESIDUE BY LABEL (excluded labels are intentionally kept)")
w("    %-52s %8s %8s %12s %s" % ("label", "planned", "left", "left bytes", "status"))
for lab in sorted(by_label):
    d = by_label[lab]
    status = "EXCLUDED" if d["excluded"] else ("CLEAN" if d["left"] == 0 else "!! LEFT")
    w("    %-52s %8d %8d %12s %s" %
      (lab[:52], d["total"], d["left"], human(d["left_bytes"]), status))
w("")
w("    RESIDUE TOTAL = %d files / %s" %
  (len(residue), human(sum(x[3] for x in residue))))
with open(RESIDUE_CSV, "w", encoding="utf-8", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["tier", "label", "path", "size_bytes", "mtime"])
    for t, lab, p, sz, mt in residue:
        cw.writerow([t, lab, p, sz, mt])
w("")

# ---- keepers ----
w("[2] MUST-KEEP CHECKS")
checks = [
    ("client_visualization_assets", os.path.join(ROOT, "tests", "glm5", "result", "client_visualization_assets")),
    ("distributed_data\\objects", os.path.join(ROOT, "distributed_data", "objects")),
    ("distributed_data\\research_library", os.path.join(ROOT, "distributed_data", "research_library")),
    ("distributed_data\\mechanism_traces\\index.sqlite3", os.path.join(ROOT, "distributed_data", "mechanism_traces", "index.sqlite3")),
    ("frontend\\node_modules\\vite\\bin\\vite.js", os.path.join(ROOT, "frontend", "node_modules", "vite", "bin", "vite.js")),
    ("research", os.path.join(ROOT, "research")),
    ("deploy\\README_RESEARCH_UPLOAD.md", os.path.join(ROOT, "deploy", "README_RESEARCH_UPLOAD.md")),
    ("result\\qwen14_mlp_atlas_v1 (junction)", os.path.join(ROOT, "tests", "glm5", "result", "qwen14_mlp_atlas_v1")),
    ("release\\desktop-1.0.13", os.path.join(ROOT, "release", "desktop-1.0.13")),
    ("release\\ai2050-center-1.0.36.tar.gz", os.path.join(ROOT, "release", "ai2050-center-1.0.36.tar.gz")),
]
for label, q in checks:
    w("    %-52s %s%s" % (label[:52], "PRESENT" if os.path.lexists(q) else "!! MISSING",
                          "  [JUNCTION]" if os.path.lexists(q) and is_reparse(q) else ""))
w("")

# ---- count conclusion files in the A-tier result dirs ----
w("[3] CONCLUSION FILES INSIDE A-TIER RESULT DIRS")
plan = list(csv.DictReader(open(PLAN, "r", encoding="utf-8", newline="")))
tot_json = tot_py = tot_sha = 0
tot_json_b = 0
for r in plan:
    if not r["tier"].startswith("A"):
        continue
    p = r["absolute_path"]
    if not os.path.isdir(p):
        continue
    for dp, dns, fns in os.walk(p):
        for fn in fns:
            low = fn.lower()
            q = os.path.join(dp, fn)
            try:
                sz = os.path.getsize(q)
            except OSError:
                continue
            if low.endswith(".json") or low.endswith(".jsonl"):
                tot_json += 1
                tot_json_b += sz
            elif low.endswith(".py"):
                tot_py += 1
            elif low.endswith(".sha256"):
                tot_sha += 1
w("    json/jsonl = %d files / %s ; py = %d ; sha256 = %d" %
  (tot_json, human(tot_json_b), tot_py, tot_sha))
w("")

# ---- staging ----
w("[4] STAGING")
stage = r"D:\AI2050\_A_stage_20261113"
w("    staging root exists = %s" % os.path.isdir(stage))
if os.path.isdir(stage):
    n = 0
    for dp, dns, fns in os.walk(stage):
        n += len(fns)
    w("    staging files = %d" % n)
w("")
w("===== END VERIFY =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("RESIDUE=%d files %s" % (len(residue), human(sum(x[3] for x in residue))))
