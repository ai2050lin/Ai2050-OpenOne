# -*- coding: utf-8 -*-
"""
P2 cold-data cleanup, batched and resumable.
Only rows with batch == "P2-cold" from the frozen CSV manifest are touched.
Each batch: delete files -> empty D recycle bin -> verify the newly created
bin items are gone and space is freed. Aborts on any anomaly.

Usage: python cleanup_p2_run_20261109.py [budget_bytes]
State is persisted after every batch so the run can be resumed safely.
"""
import os
import sys
import csv
import time
import json
import shutil
import ctypes
import datetime

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
STATE = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_state_20261109.json"
LOG = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_log_20261109.txt"
DELETED = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_deleted_20261109.txt"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_progress_20261109.txt"

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 30 * 1024 ** 3
MAX_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BATCH_BYTES = 1500 * 1024 ** 2
BATCH_FILES = 2500
BIN_TOLERANCE = 200 * 1024 ** 2


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


def free_d():
    return shutil.disk_usage("D:\\").free


def bin_stat():
    size, files = 0, 0
    p = r"D:\$RECYCLE.BIN"
    if os.path.isdir(p):
        for dp, dns, fns in os.walk(p):
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


def load_state():
    if os.path.isfile(STATE):
        with open(STATE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"next_line": 2, "done": False, "batches": 0,
            "deleted_files": 0, "deleted_bytes": 0, "freed_bytes": 0, "rounds": 0}


def save_state(st):
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n")


st = load_state()
st["rounds"] += 1
t_start = time.time()
run_files = run_bytes = run_freed = run_batches = 0
stopped = "BUDGET_REACHED"
first_dir = last_dir = ""

f_round_start = free_d()


def process_batch(items):
    global run_files, run_bytes, run_freed, run_batches, first_dir, last_dir
    b0 = bin_stat()[0]
    f0 = free_d()
    ok = fail = skip = 0
    nbytes = 0
    deleted_lines = []
    t0 = time.time()
    for line_no, row in items:
        p = row["path"]
        s = int(row["size_bytes"])
        try:
            os.remove(p)
            ok += 1
            nbytes += s
            deleted_lines.append("%d\t%s" % (s, p))
        except FileNotFoundError:
            skip += 1
        except Exception:
            fail += 1
    hr = empty_bin()
    time.sleep(0.15)
    b1 = bin_stat()[0]
    f1 = free_d()
    dfree = f1 - f0
    with open(DELETED, "a", encoding="utf-8") as fh:
        if deleted_lines:
            fh.write("\n".join(deleted_lines) + "\n")

    st["batches"] += 1
    st["deleted_files"] += ok
    st["deleted_bytes"] += nbytes
    st["freed_bytes"] += max(dfree, 0)
    run_files += ok
    run_bytes += nbytes
    run_freed += max(dfree, 0)
    run_batches += 1
    first_dir = items[0][1]["top_dir"]
    last_dir = items[-1][1]["top_dir"]

    anomaly = ""
    if hr != 0:
        anomaly = "EMPTY_HRESULT=0x%08X" % (hr & 0xFFFFFFFF)
    elif b1 > b0 + BIN_TOLERANCE:
        anomaly = "BIN_ACCUMULATED(+%s)" % human(b1 - b0)
    elif fail > max(10, ok * 0.05):
        anomaly = "TOO_MANY_FAILS(%d/%d)" % (fail, ok)

    log("BATCH %-4d ok=%-5d fail=%-4d skip=%-4d bytes=%-10s freed=%-+10s "
        "bin=%s->%s hr=%d %.1fs dirs=%s..%s %s" % (
            st["batches"], ok, fail, skip, human(nbytes), human(dfree),
            human(b0), human(b1), hr, time.time() - t0,
            first_dir[:22], last_dir[:22], anomaly or "OK"))
    return anomaly, ok, nbytes, dfree


batch = []
batch_bytes = 0
last_line = st["next_line"] - 1
anomaly_global = ""

with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for line_no, row in enumerate(csv.DictReader(cf), start=2):
        if line_no < st["next_line"]:
            continue
        if row["batch"] != "P2-cold":
            continue
        batch.append((line_no, row))
        batch_bytes += int(row["size_bytes"])
        last_line = line_no
        if batch_bytes >= BATCH_BYTES or len(batch) >= BATCH_FILES:
            anom, ok, nb, dfree = process_batch(batch)
            st["next_line"] = last_line + 1
            save_state(st)
            batch, batch_bytes = [], 0
            if anom:
                anomaly_global = anom
                stopped = "ABORT_" + anom
                break
            if run_bytes >= BUDGET:
                stopped = "BUDGET_REACHED"
                break
            if run_batches >= MAX_BATCHES:
                stopped = "BATCH_LIMIT_REACHED"
                break
    else:
        stopped = "ALL_P2_DONE"

if batch and not anomaly_global:
    anom, ok, nb, dfree = process_batch(batch)
    st["next_line"] = last_line + 1
    save_state(st)
    if anom:
        anomaly_global = anom
        stopped = "ABORT_" + anom

if stopped == "ALL_P2_DONE":
    st["done"] = True
save_state(st)

f_round_end = free_d()
elapsed = time.time() - t_start
L = []
w = L.append
w("P2 CLEANUP ROUND %d  " % st["rounds"] + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("  status            = %s" % stopped)
w("  budget            = %s" % human(BUDGET))
w("  this round        = %d batches / %d files / %s deleted / %s freed" % (
    run_batches, run_files, human(run_bytes), human(run_freed)))
w("  cumulative        = %d batches / %d files / %s deleted / %s freed" % (
    st["batches"], st["deleted_files"], human(st["deleted_bytes"]), human(st["freed_bytes"])))
w("  D free            = %s -> %s (%+s)" % (
    human(f_round_start), human(f_round_end), human(f_round_end - f_round_start)))
w("  bin               = %s / %d files" % (human(bin_stat()[0]), bin_stat()[1]))
w("  next_line         = %d" % st["next_line"])
w("  elapsed           = %.1f sec" % elapsed)
w("  dirs touched      = %s .. %s" % (first_dir, last_dir))
if anomaly_global:
    w("  !! ANOMALY       = " + anomaly_global)
w("")
w("=== END ROUND ===")
with open(REPORT, "a", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
log("ROUND %d status=%s batches=%d files=%d bytes=%d freed=%d elapsed=%.1fs" % (
    st["rounds"], stopped, run_batches, run_files, run_bytes, run_freed, elapsed))

print("STATUS=%s batches=%d files=%d deleted=%s freed=%s done=%s" % (
    stopped, run_batches, run_files, human(run_bytes), human(run_freed), st["done"]))
if anomaly_global:
    raise SystemExit(2)
