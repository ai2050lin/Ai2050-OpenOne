# -*- coding: utf-8 -*-
"""
PHASE 1 verification batch: delete ONLY the 37 P1-junk files (699.14 MB),
measure safe-delete hook behaviour, then decide how P2 should be executed.
User-confirmed scope: P1 + P2 cold data. All paths come from the frozen CSV manifest.
"""
import os
import csv
import time
import shutil
import datetime
import json

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
BIN_D = r"D:\$RECYCLE.BIN"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_p1_verify_20261109.txt"
LOG = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_log_20261109.txt"
DELETED = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_deleted_20261109.txt"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


def free_d():
    return shutil.disk_usage("D:\\").free


def bin_stat():
    size, files = 0, 0
    if os.path.isdir(BIN_D):
        for dp, dns, fns in os.walk(BIN_D):
            for fn in fns:
                try:
                    size += os.lstat(os.path.join(dp, fn)).st_size
                    files += 1
                except OSError:
                    pass
    return size, files


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n")


def mark_deleted(path, size):
    with open(DELETED, "a", encoding="utf-8") as fh:
        fh.write("%d\t%s\n" % (size, path))


L = []
w = L.append
w("P1 VERIFICATION BATCH  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("SCOPE: P1-junk ONLY (37 files / 699.14 MB) from frozen CSV manifest")
w("")

rows = []
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for r in csv.DictReader(cf):
        if r["batch"] == "P1-junk":
            rows.append(r)
w("[0] manifest P1 rows = %d, bytes = %s" % (len(rows), human(sum(int(r["size_bytes"]) for r in rows))))

# preflight: all must exist and match size
bad = []
for r in rows:
    p = r["path"]
    if not os.path.isfile(p):
        bad.append(("missing", p))
    elif os.lstat(p).st_size != int(r["size_bytes"]):
        bad.append(("size-mismatch", p))
w("[0] precheck bad = %d" % len(bad))
for k, p in bad[:5]:
    w("      %s %s" % (k, p))
if bad:
    w("ABORT: manifest changed since scan.")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("P1 ABORT manifest_changed=%d" % len(bad))
    raise SystemExit(2)

# guard: no writes in target within last 5 minutes
REPARSE = 0x400
live = 0
for dp, dns, fns in os.walk(ROOT, topdown=True, followlinks=False):
    kept = []
    for d in dns:
        q = os.path.join(dp, d)
        try:
            if getattr(os.lstat(q), "st_file_attributes", 0) & REPARSE:
                continue
        except OSError:
            continue
        kept.append(d)
    dns[:] = kept
    for fn in fns:
        try:
            if os.lstat(os.path.join(dp, fn)).st_mtime >= time.time() - 300:
                live += 1
        except OSError:
            pass
w("[0] files modified in last 5 min = %d" % live)
if live > 0:
    w("ABORT: target directory is being written to.")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("P1 ABORT live_writes=%d" % live)
    raise SystemExit(3)

small = sorted([r for r in rows if int(r["size_bytes"]) < 16 * 1024 * 1024],
               key=lambda r: int(r["size_bytes"]))
big = [r for r in rows if int(r["size_bytes"]) >= 16 * 1024 * 1024]
w("[0] small files = %d (%s), big files = %d (%s)" % (
    len(small), human(sum(int(r["size_bytes"]) for r in small)),
    len(big), human(sum(int(r["size_bytes"]) for r in big))))
w("")

f0, b0 = free_d(), bin_stat()
w("[1] BEFORE  D_free=%s  bin=%s/%d files" % (human(f0), human(b0[0]), b0[1]))

# ---- probe: one tiny file ----
probe = small[0]
pp, ps = probe["path"], int(probe["size_bytes"])
w("[2] PROBE %s (%s)" % (pp, human(ps)))
try:
    os.remove(pp)
    probe_ok = True
    mark_deleted(pp, ps)
except Exception as e:
    probe_ok = False
    w("    os.remove raised: " + repr(e))
time.sleep(0.4)
f1, b1 = free_d(), bin_stat()
still = os.path.exists(pp)
w("    exists_after=%s  D_free=%s (delta %+d B)  bin=%s/%d files (delta %+d files, %+d B)" % (
    still, human(f1), f1 - f0, human(b1[0]), b1[1], b1[1] - b0[1], b1[0] - b0[0]))

if still:
    verdict = "DELETE_FAILED"
    w("VERDICT=DELETE_FAILED -> STOP, do not proceed to P2")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("P1 VERDICT=DELETE_FAILED")
    raise SystemExit(4)

# ---- rest of small files ----
cnt_small, fail_small = 0, 0
for r in small[1:]:
    p, s = r["path"], int(r["size_bytes"])
    try:
        os.remove(p)
        cnt_small += 1
        mark_deleted(p, s)
    except Exception as e:
        fail_small += 1
        w("    FAIL small %s : %r" % (p, e))
w("[3] rest small deleted=%d failed=%d" % (cnt_small, fail_small))
time.sleep(0.4)
f2, b2 = free_d(), bin_stat()
w("    D_free=%s  bin=%s/%d files" % (human(f2), human(b2[0]), b2[1]))

# ---- big .tmp files (strong free-space signal) ----
cnt_big, fail_big = 0, 0
big_bytes = 0
for r in big:
    p, s = r["path"], int(r["size_bytes"])
    try:
        os.remove(p)
        cnt_big += 1
        big_bytes += s
        mark_deleted(p, s)
    except Exception as e:
        fail_big += 1
        w("    FAIL big %s : %r" % (p, e))
time.sleep(1.0)
f3, b3 = free_d(), bin_stat()
w("[4] big deleted=%d (%s) failed=%d" % (cnt_big, human(big_bytes), fail_big))
w("    AFTER   D_free=%s  bin=%s/%d files" % (human(f3), human(b3[0]), b3[1]))
w("    NET     D_free delta = %+s  bin size delta = %+s  bin files delta = %+d" % (
    human(f3 - f0), human(b3[0] - b0[0]), b3[1] - b0[1]))
w("")

# ---- verdict ----
dfree = f3 - f0
dbin = b3[0] - b0[0]
expected = sum(int(r["size_bytes"]) for r in rows)
if dfree >= big_bytes * 0.5:
    verdict = "DIRECT_FREE"
    note = "os.remove releases space immediately; safe-delete hook did NOT intercept. P2 can use os.remove."
elif dbin >= big_bytes * 0.5:
    verdict = "TO_RECYCLE_BIN"
    note = "deletions were redirected to D recycle bin, space NOT released. P2 must use a hook-bypass path or empty the bin per batch."
else:
    verdict = "AMBIGUOUS"
    note = "neither free space nor recycle bin grew as expected (sparse file / delayed accounting / hook anomaly). Inspect before P2."
w("[5] VERDICT = " + verdict)
w("    " + note)
w("    expected_deleted=%s  actually_deleted=%s" % (
    human(expected), human(sum(int(r["size_bytes"]) for r in rows) - sum(
        int(r["size_bytes"]) for r in rows if os.path.exists(r["path"])))))
w("")
w("=== END P1 VERIFY ===")

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
log("P1 done verdict=%s small=%d/%d big=%d/%d free_delta=%+d bin_delta=%+d" % (
    verdict, cnt_small + 1, len(small), cnt_big, len(big), dfree, dbin))

with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_verdict_20261109.json", "w", encoding="utf-8") as fh:
    json.dump({"verdict": verdict, "note": note, "free_delta": dfree,
               "bin_delta": dbin, "deleted_files": cnt_small + 1 + cnt_big,
               "deleted_bytes": sum(int(r["size_bytes"]) for r in rows) - sum(
                   int(r["size_bytes"]) for r in rows if os.path.exists(r["path"]))}, fh, indent=2)

print("VERDICT=" + verdict)
