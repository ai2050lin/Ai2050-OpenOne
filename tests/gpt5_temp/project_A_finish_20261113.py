# -*- coding: utf-8 -*-
"""Idempotent residue cleanup for A-tier (2026-11-13).

Works from the ACTUAL residue list produced by the read-only verifier,
so it cannot suffer from the line-index drift of the first executor.
Re-runnable: only files that still exist are processed.
"""
import os
import csv
import time
import json
import shutil
import ctypes
import datetime

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result"
RESIDUE = os.path.join(OUT, "project_A_residue_20261113.csv")
LOG = os.path.join(OUT, "project_A_cleanup_log_20261113.txt")
DELETED = os.path.join(OUT, "project_A_deleted_20261113.txt")
ROLLBACK = os.path.join(OUT, "project_A_rollback_20261113.txt")
REPORT = os.path.join(OUT, "project_A_finish_report_20261113.txt")
STAGE_ROOT = r"D:\AI2050\_A_stage_finish_20261113"

BATCH_FILES = 2000
BATCH_BYTES = 6 * 1024 ** 3
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


items = []
for r in csv.DictReader(open(RESIDUE, "r", encoding="utf-8", newline="")):
    p = r["path"]
    if not os.path.isfile(p):
        continue
    if float(r["mtime"]) >= H24:
        log("SKIP_RECENT " + p)
        continue
    items.append((p, int(r["size_bytes"]), r["label"]))

os.makedirs(STAGE_ROOT, exist_ok=True)
log("=== FINISH start: residue=%d files %s ===" %
    (len(items), human(sum(x[1] for x in items))))

batches = files_done = bytes_done = freed = 0
anomaly = ""
batch = []
batch_bytes = 0


def process(b, no):
    global batches, files_done, bytes_done, freed
    stage = os.path.join(STAGE_ROOT, "B%05d" % no)
    os.makedirs(stage, exist_ok=True)
    pairs = []
    t0 = time.time()
    for i, (src, sz, lab) in enumerate(b):
        dst = os.path.join(stage, "%06d_%s" % (i, os.path.basename(src)))
        try:
            os.rename(src, dst)
            pairs.append((src, dst, sz))
        except FileNotFoundError:
            continue
        except OSError as e:
            log("RENAME_FAIL %s : %r" % (src, e))
    if not pairs:
        try:
            os.rmdir(stage)
        except OSError:
            pass
        return "NO_PAIRS"
    with open(ROLLBACK, "a", encoding="utf-8") as fh:
        fh.write("\n".join("%s\t%s\t%d" % t for t in pairs) + "\n")
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
    d = f1 - f0
    gone = not os.path.exists(stage)
    a = ""
    if not gone:
        a = "STAGE_SURVIVED"
    elif err:
        a = "RMTREE_ERR:" + err[:60]
    elif b1 > b0 + 300 * 1024 ** 2:
        a = "BIN_ACCUMULATED"
    if gone:
        with open(DELETED, "a", encoding="utf-8") as fh:
            fh.write("\n".join("%d\t%s" % (z, s) for s, dd, z in pairs) + "\n")
    batches += 1
    files_done += len(pairs)
    bytes_done += sum(t[2] for t in pairs)
    freed += max(d, 0)
    log("FINISH BATCH %-3d moved=%-5d bytes=%-10s freed=%-+10s bin=%s->%s hr=%d %.2fs %s" %
        (batches, len(pairs), human(sum(t[2] for t in pairs)), human(d),
         human(b0), human(b1), hr, time.time() - t0, a or "OK"))
    return a


for p, sz, lab in items:
    batch.append((p, sz, lab))
    batch_bytes += sz
    if len(batch) >= BATCH_FILES or batch_bytes >= BATCH_BYTES:
        a = process(batch, batches + 1)
        batch, batch_bytes = [], 0
        if a:
            anomaly = a
            break
if batch and not anomaly:
    anomaly = process(batch, batches + 1)

try:
    os.rmdir(STAGE_ROOT)
except OSError:
    pass

# final residue re-check (disk truth)
left = []
for r in csv.DictReader(open(RESIDUE, "r", encoding="utf-8", newline="")):
    if os.path.isfile(r["path"]):
        left.append(r["path"])

L = []
w = L.append
w("A-TIER FINISH / RESIDUE CLEANUP  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("  batches       = %d" % batches)
w("  deleted files = %d" % files_done)
w("  deleted bytes = %s" % human(bytes_done))
w("  freed bytes   = %s" % human(freed))
w("  D free now    = %s" % human(free_d()))
w("  recycle bin   = %s / %d files" % (human(bin_stat()[0]), bin_stat()[1]))
w("  RESIDUE LEFT  = %d" % len(left))
for p in left[:20]:
    w("     " + p)
w("  anomaly       = %s" % (anomaly or "none"))
w("  stage root    = %s" % os.path.isdir(STAGE_ROOT))
w("=== END FINISH ===")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
log("FINISH status=%s batches=%d files=%d bytes=%d freed=%d residue_left=%d" %
    (anomaly or "OK", batches, files_done, bytes_done, freed, len(left)))
print("FINISH batches=%d files=%d deleted=%s freed=%s residue_left=%d anomaly=%s" %
      (batches, files_done, human(bytes_done), human(freed), len(left), anomaly or "none"))
if anomaly or left:
    raise SystemExit(2)
