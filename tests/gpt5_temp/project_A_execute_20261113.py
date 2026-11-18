# -*- coding: utf-8 -*-
"""A-tier cleanup executor (2026-11-13), resumable.

Policy: keep scripts / test inputs / conclusion JSON; remove intermediate big data.
Chain verified on 11-09: os.rename -> isolated staging (instant, same volume),
shutil.rmtree(stage) -> safe-delete hook -> D: recycle bin, then
SHEmptyRecycleBinW -> real space reclaim. Every batch is logged and verified.

Excluded by preflight: frontend\\node_modules (live vite/esbuild),
distributed_data\\mechanism_traces (live server reader).
"""
import os
import sys
import csv
import time
import json
import shutil
import ctypes
import datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "gpt5", "result")
MANIFEST = os.path.join(OUT, "project_A_delete_manifest_20261113.csv")
STATE = os.path.join(OUT, "project_A_state_20261113.json")
LOG = os.path.join(OUT, "project_A_cleanup_log_20261113.txt")
DELETED = os.path.join(OUT, "project_A_deleted_20261113.txt")
ROLLBACK = os.path.join(OUT, "project_A_rollback_20261113.txt")
STAGE_ROOT = r"D:\AI2050\_A_stage_20261113"

EXCLUDE_LABELS = {"frontend\\node_modules", "mechanism_traces"}
BATCH_FILES = 2000
BATCH_BYTES = 6 * 1024 ** 3
MAX_BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
H24 = time.time() - 24 * 3600


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def free_d():
    return shutil.disk_usage("D:\\").free


def bin_stat():
    size = files = 0
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


def empty_bin():
    s = ctypes.windll.shell32
    s.SHEmptyRecycleBinW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    s.SHEmptyRecycleBinW.restype = ctypes.c_long
    return s.SHEmptyRecycleBinW(None, "D:\\", 0x1 | 0x2 | 0x4)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n")


# ---------------- build work list ----------------
items = []
skipped_label = skipped_recent = skipped_missing = 0
with open(MANIFEST, "r", encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        if row["label"] in EXCLUDE_LABELS:
            skipped_label += 1
            continue
        p = row["path"]
        if not os.path.isfile(p):
            skipped_missing += 1
            continue
        mt = float(row["mtime"])
        if mt >= H24:
            skipped_recent += 1
            continue
        items.append({"path": p, "size": int(row["size_bytes"]),
                      "label": row["label"], "tier": row["tier"]})

if os.path.isfile(STATE):
    st = json.load(open(STATE, "r", encoding="utf-8"))
else:
    st = {"next_index": 0, "batches": 0, "deleted_files": 0,
          "deleted_bytes": 0, "freed_bytes": 0, "done": False,
          "started": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
st["rounds"] = st.get("rounds", 0) + 1
st["total_files"] = len(items)
st["total_bytes"] = sum(i["size"] for i in items)
os.makedirs(STAGE_ROOT, exist_ok=True)

log("=== ROUND %d start: total=%d files %s ; excl_label=%d recent=%d missing=%d ===" %
    (st["rounds"], len(items), human(st["total_bytes"]),
     skipped_label, skipped_recent, skipped_missing))

t_start = time.time()
run_batches = run_files = run_bytes = run_freed = 0
stopped = "BATCH_LIMIT_REACHED"
anomaly_global = ""


def process(batch, batch_no):
    global run_batches, run_files, run_bytes, run_freed
    stage = os.path.join(STAGE_ROOT, "B%05d" % batch_no)
    os.makedirs(stage, exist_ok=True)
    pairs = []
    t0 = time.time()
    for i, rec in enumerate(batch):
        src = rec["path"]
        dst = os.path.join(stage, "%06d_%s" % (i, os.path.basename(src)))
        try:
            os.rename(src, dst)
            pairs.append((src, dst, rec["size"]))
        except FileNotFoundError:
            continue
        except OSError as e:
            log("RENAME_FAIL %s : %r" % (src, e))
    if not pairs:
        try:
            os.rmdir(stage)
        except OSError:
            pass
        return "NO_PAIRS", 0
    nbytes = sum(p[2] for p in pairs)
    with open(ROLLBACK, "a", encoding="utf-8") as fh:
        fh.write("\n".join("%s\t%s\t%d" % (s, d, z) for s, d, z in pairs) + "\n")

    b0 = bin_stat()[0]
    f0 = free_d()
    err = ""
    try:
        shutil.rmtree(stage)
    except Exception as e:
        err = repr(e)
    hr = empty_bin()
    time.sleep(0.15)
    b1 = bin_stat()[0]
    f1 = free_d()
    dfree = f1 - f0
    gone = not os.path.exists(stage)

    anom = ""
    if not gone:
        anom = "STAGE_SURVIVED"
    elif err:
        anom = "RMTREE_ERR:" + err[:60]
    elif b1 > b0 + 300 * 1024 ** 2:
        anom = "BIN_ACCUMULATED(+%s)" % human(b1 - b0)

    if gone:
        with open(DELETED, "a", encoding="utf-8") as fh:
            fh.write("\n".join("%d\t%s" % (z, s) for s, d, z in pairs) + "\n")

    st["batches"] += 1
    st["deleted_files"] += len(pairs)
    st["deleted_bytes"] += nbytes
    st["freed_bytes"] += max(dfree, 0)
    run_batches += 1
    run_files += len(pairs)
    run_bytes += nbytes
    run_freed += max(dfree, 0)
    log("BATCH %-4d moved=%-5d bytes=%-10s freed=%-+10s bin=%s->%s hr=%d %.2fs %s..%s %s" % (
        st["batches"], len(pairs), human(nbytes), human(dfree), human(b0), human(b1),
        hr, time.time() - t0, batch[0]["label"][:16], batch[-1]["label"][:16],
        anom or "OK"))
    return anom, dfree


batch = []
batch_bytes = 0
idx = 0
for i, rec in enumerate(items):
    if i < st["next_index"]:
        continue
    batch.append(rec)
    batch_bytes += rec["size"]
    idx = i
    if len(batch) >= BATCH_FILES or batch_bytes >= BATCH_BYTES:
        a, d = process(batch, st["batches"] + 1)
        st["next_index"] = idx + 1
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=2)
        batch, batch_bytes = [], 0
        if a:
            anomaly_global = a
            stopped = "ABORT_" + a
            break
        if run_batches >= MAX_BATCHES:
            break
else:
    stopped = "ALL_A_DONE"

if batch and not anomaly_global:
    a, d = process(batch, st["batches"] + 1)
    st["next_index"] = idx + 1
    if a:
        anomaly_global = a
        stopped = "ABORT_" + a

if stopped == "ALL_A_DONE":
    st["done"] = True
with open(STATE, "w", encoding="utf-8") as fh:
    json.dump(st, fh, indent=2)

# remove empty staging root
if st["done"] and os.path.isdir(STAGE_ROOT):
    try:
        os.rmdir(STAGE_ROOT)
    except OSError:
        pass

L = []
w = L.append
w("A-TIER ROUND %d  " % st["rounds"] + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("  status       = %s" % stopped)
w("  this round   = %d batches / %d files / %s deleted / %s freed" %
  (run_batches, run_files, human(run_bytes), human(run_freed)))
w("  cumulative   = %d batches / %d files / %s deleted / %s freed" %
  (st["batches"], st["deleted_files"], human(st["deleted_bytes"]), human(st["freed_bytes"])))
w("  D free       = %s" % human(free_d()))
w("  bin          = %s / %d files" % (human(bin_stat()[0]), bin_stat()[1]))
w("  next_index   = %d / %d  done=%s" % (st["next_index"], len(items), st["done"]))
w("  elapsed      = %.1f sec" % (time.time() - t_start))
if anomaly_global:
    w("  !! ANOMALY  = " + anomaly_global)
w("=== END ===")
with open(os.path.join(OUT, "project_A_progress_20261113.txt"), "a", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
log("ROUND %d status=%s batches=%d files=%d bytes=%d freed=%d" %
    (st["rounds"], stopped, run_batches, run_files, run_bytes, run_freed))
print("STATUS=%s batches=%d files=%d deleted=%s freed=%s done=%s" %
      (stopped, run_batches, run_files, human(run_bytes), human(run_freed), st["done"]))
if anomaly_global:
    raise SystemExit(2)
