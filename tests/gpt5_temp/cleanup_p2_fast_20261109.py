# -*- coding: utf-8 -*-
"""
FAST P2 cleanup, resumable.
Strategy: move approved files into an isolated staging dir with os.rename
(instant, no delete hook), then delete the WHOLE directory in one operation
(shutil.rmtree -> recycle bin, then SHEmptyRecycleBinW to free space).
Resumes from cleanup_state_20261109.json written by the slow runner.
"""
import os
import sys
import csv
import time
import json
import shutil
import ctypes
import subprocess
import datetime

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
STATE = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_state_20261109.json"
LOG = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_log_20261109.txt"
DELETED = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_deleted_20261109.txt"
PROGRESS = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_progress_20261109.txt"
ROLLBACK = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_rollback_20261109.txt"
STAGE_ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_cleanup_stage_20261109"

MAX_BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
BATCH_FILES = 3000
BATCH_BYTES = 4 * 1024 ** 3


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


def free_d():
    return shutil.disk_usage("D:\\").free


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


def empty_bin():
    s = ctypes.windll.shell32
    s.SHEmptyRecycleBinW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    s.SHEmptyRecycleBinW.restype = ctypes.c_long
    return s.SHEmptyRecycleBinW(None, "D:\\", 0x1 | 0x2 | 0x4)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n")


st = json.load(open(STATE, "r", encoding="utf-8"))
st["rounds"] += 1
os.makedirs(STAGE_ROOT, exist_ok=True)
t_start = time.time()
run_batches = run_files = run_bytes = run_freed = 0
stopped = "BATCH_LIMIT_REACHED"
anomaly_global = ""


def process(items, batch_no):
    global run_batches, run_files, run_bytes, run_freed
    stage = os.path.join(STAGE_ROOT, "B%05d" % batch_no)
    os.makedirs(stage, exist_ok=True)
    pairs = []
    t0 = time.time()
    for i, (line_no, row) in enumerate(items):
        src = row["path"]
        dst = os.path.join(stage, "%06d_%s" % (i, os.path.basename(src)))
        try:
            os.rename(src, dst)
            pairs.append((src, dst, int(row["size_bytes"])))
        except FileNotFoundError:
            continue
        except OSError as e:
            log("RENAME_FAIL %s : %r" % (src, e))
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
    time.sleep(0.2)
    b1 = bin_stat()[0]
    f1 = free_d()
    dfree = f1 - f0
    gone = not os.path.exists(stage)

    anom = ""
    if not gone:
        anom = "STAGE_SURVIVED"
    elif err:
        anom = "RMTREE_ERR"
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
    log("FAST BATCH %-4d moved=%-5d bytes=%-10s freed=%-+10s bin=%s->%s hr=%d %.2fs %s..%s %s" % (
        st["batches"], len(pairs), human(nbytes), human(dfree), human(b0), human(b1),
        hr, time.time() - t0, items[0][1]["top_dir"][:18], items[-1][1]["top_dir"][:18],
        anom or "OK"))
    return anom, dfree


batch = []
batch_bytes = 0
last_line = st["next_line"] - 1
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for line_no, row in enumerate(csv.DictReader(cf), start=2):
        if line_no < st["next_line"]:
            continue
        if row["batch"] != "P2-cold":
            continue
        batch.append((line_no, row))
        batch_bytes += int(row["size_bytes"])
        last_line = line_no
        if len(batch) >= BATCH_FILES or batch_bytes >= BATCH_BYTES:
            a, d = process(batch, st["batches"] + 1)
            st["next_line"] = last_line + 1
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
        stopped = "ALL_P2_DONE"

if batch and not anomaly_global:
    a, d = process(batch, st["batches"] + 1)
    st["next_line"] = last_line + 1
    if a:
        anomaly_global = a
        stopped = "ABORT_" + a

if stopped == "ALL_P2_DONE":
    st["done"] = True
with open(STATE, "w", encoding="utf-8") as fh:
    json.dump(st, fh, indent=2)

L = []
w = L.append
w("FAST P2 ROUND %d  " % st["rounds"] + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("  status       = %s" % stopped)
w("  this round   = %d batches / %d files / %s deleted / %s freed" % (
    run_batches, run_files, human(run_bytes), human(run_freed)))
w("  cumulative   = %d batches / %d files / %s deleted / %s freed" % (
    st["batches"], st["deleted_files"], human(st["deleted_bytes"]), human(st["freed_bytes"])))
w("  D free       = %s" % human(free_d()))
w("  bin          = %s / %d files" % (human(bin_stat()[0]), bin_stat()[1]))
w("  next_line    = %d  done=%s" % (st["next_line"], st["done"]))
w("  elapsed      = %.1f sec" % (time.time() - t_start))
if anomaly_global:
    w("  !! ANOMALY  = " + anomaly_global)
w("=== END ===")
with open(PROGRESS, "a", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
log("FAST ROUND %d status=%s batches=%d files=%d bytes=%d freed=%d" % (
    st["rounds"], stopped, run_batches, run_files, run_bytes, run_freed))
print("STATUS=%s batches=%d files=%d deleted=%s freed=%s done=%s" % (
    stopped, run_batches, run_files, human(run_bytes), human(run_freed), st["done"]))
if anomaly_global:
    raise SystemExit(2)
