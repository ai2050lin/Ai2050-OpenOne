# -*- coding: utf-8 -*-
"""READ-ONLY final verification of the cleanup."""
import os
import csv
import json
import time
import shutil
import datetime
import collections

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
STATE = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_state_20261109.json"
STAGE_ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_cleanup_stage_20261109"
PROBE_DIRS = [
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_hookprobe_20261109",
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_dirprobe_20261109",
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_dirprobe2_20261109",
]
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_final_report_20261109.txt"
REPARSE = 0x400


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


def bin_stat():
    size, files = 0, 0
    B = r"D:\$RECYCLE.BIN"
    if os.path.isdir(B):
        for dp, dns, fns in os.walk(B):
            for fn in fns:
                try:
                    size += os.lstat(os.path.join(dp, fn)).st_size
                    files += 1
                except OSError:
                    pass
    return size, files


L = []
w = L.append
w("GLM5 RESULT CLEANUP - FINAL VERIFICATION  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("SCOPE (user confirmed): P1-junk + P2-cold only. P3-active, REVIEW, KEEP untouched.")
w("")

st = json.load(open(STATE, "r", encoding="utf-8"))
w("[1] EXECUTION STATE")
w("    done=%s  batches=%d  rounds=%d  next_line=%d" % (
    st["done"], st["batches"], st["rounds"], st["next_line"]))
w("    deleted_files=%d  deleted_bytes=%s" % (st["deleted_files"], human(st["deleted_bytes"])))
w("    recorded_freed=%s" % human(st["freed_bytes"]))
w("")

# manifest residue by batch
res = collections.defaultdict(lambda: [0, 0])
manifest_total = collections.defaultdict(lambda: [0, 0])
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for r in csv.DictReader(cf):
        b = r["batch"]
        s = int(r["size_bytes"])
        manifest_total[b][0] += s
        manifest_total[b][1] += 1
        if os.path.isfile(r["path"]):
            res[b][0] += s
            res[b][1] += 1
w("[2] MANIFEST RESIDUE (files listed for deletion that still exist)")
for b in ("P1-junk", "P2-cold", "P3-active"):
    w("    %-10s still_present=%7d / %7d   still_present_bytes=%s / %s" % (
        b, res[b][1], manifest_total[b][1], human(res[b][0]), human(manifest_total[b][0])))
w("")

# current tree
ext = collections.defaultdict(lambda: [0, 0])
total_files = total_bytes = total_dirs = 0
key_names = collections.Counter()
t0 = time.time()
for dp, dns, fns in os.walk(ROOT, topdown=True, followlinks=False):
    kept = []
    for d in dns:
        p = os.path.join(dp, d)
        try:
            if getattr(os.lstat(p), "st_file_attributes", 0) & REPARSE:
                continue
        except OSError:
            continue
        kept.append(d)
    dns[:] = kept
    total_dirs += len(dns)
    for fn in fns:
        p = os.path.join(dp, fn)
        try:
            s = os.lstat(p).st_size
        except OSError:
            continue
        e = os.path.splitext(fn)[1].lower()
        ext[e or "(none)"][0] += s
        ext[e or "(none)"][1] += 1
        total_files += 1
        total_bytes += s
        if fn in ("result.json", "execution.json", "behavior.json"):
            key_names[fn] += 1
        if e in (".sha256", ".py", ".md", ".log", ".txt"):
            key_names[e] += 1
w("[3] CURRENT TREE (scan %.1fs)" % (time.time() - t0))
w("    total=%s  files=%d  dirs=%d" % (human(total_bytes), total_files, total_dirs))
w("    (baseline 23:17 was 314.28 GB / 428,200 files / 11,722 dirs)")
w("    by extension (top 18):")
for e, v in sorted(ext.items(), key=lambda kv: -kv[1][0])[:18]:
    w("      %-14s %10s / %7d files" % (e, human(v[0]), v[1]))
w("")
w("[4] KEPT EVIDENCE COUNTS")
for k, v in sorted(key_names.items()):
    w("    %-18s %7d" % (k, v))
w("")

u = shutil.disk_usage("D:\\")
uc = shutil.disk_usage("C:\\")
bs, bf = bin_stat()
w("[5] DISK")
w("    D: total=%s used=%s free=%s (%.1f%% used)" % (
    human(u.total), human(u.used), human(u.free), 100.0 * u.used / u.total))
w("    C: free=%s" % human(uc.free))
w("    D recycle bin = %s / %d files" % (human(bs), bf))
w("    baseline at 23:17: D free=34.54 GB ; at cleanup start 23:36: D free=34.48 GB")
w("")

w("[6] STAGING / PROBE LEFTOVERS")
for d in [STAGE_ROOT] + PROBE_DIRS:
    if not os.path.exists(d):
        w("    absent  " + d)
        continue
    sz, nf = 0, 0
    for dp, dns, fns in os.walk(d):
        for fn in fns:
            try:
                sz += os.lstat(os.path.join(dp, fn)).st_size
                nf += 1
            except OSError:
                pass
    w("    PRESENT %s / %d files  %s" % (human(sz), nf, d))
w("")

w("[7] HARDLINK / BLOB NOTE")
w("    74.85 GB of the deleted set were hard links into")
w("    distributed_data\\research_native_blobs (CAS store, deliberately NOT touched).")
w("    Deleting those links does not free space; the CAS store still holds the data.")
w("    Blobs whose result-side link was removed are now nlink=1 (orphan candidates).")
w("")
w("=== END FINAL VERIFICATION (READ-ONLY) ===")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
