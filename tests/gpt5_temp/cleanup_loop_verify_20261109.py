# -*- coding: utf-8 -*-
"""
Closed-loop verification: delete ONE mid-size P2-cold file -> confirm it lands
in the recycle bin -> empty the bin -> confirm the space is actually freed.
Also inspects why 1.33 GB of pre-existing bin items were not emptied.
"""
import os
import csv
import time
import ctypes
import shutil
import datetime
import json

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
BIN_D = r"D:\$RECYCLE.BIN"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_loop_verify_20261109.txt"
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


def empty_bin():
    f = 0x1 | 0x2 | 0x4
    s = ctypes.windll.shell32
    s.SHEmptyRecycleBinW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    s.SHEmptyRecycleBinW.restype = ctypes.c_long
    return s.SHEmptyRecycleBinW(None, "D:\\", f)


L = []
w = L.append
w("DELETE -> EMPTY -> FREE CLOSED LOOP  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# ---- why is there a remainder? ----
w("[1] D RECYCLE BIN subdirs (pre-existing remainder)")
if os.path.isdir(BIN_D):
    for name in os.listdir(BIN_D):
        p = os.path.join(BIN_D, name)
        if not os.path.isdir(p):
            continue
        sz, nf = 0, 0
        for dp, dns, fns in os.walk(p):
            for fn in fns:
                try:
                    sz += os.lstat(os.path.join(dp, fn)).st_size
                    nf += 1
                except OSError:
                    pass
        w("   %-50s %10s / %4d files" % (name, human(sz), nf))
w("")

# ---- pick one mid-size P2 file ----
pick = None
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for r in csv.DictReader(cf):
        if r["batch"] != "P2-cold":
            continue
        s = int(r["size_bytes"])
        if 40 * 1024 * 1024 <= s <= 200 * 1024 * 1024:
            pick = (r["path"], s, r["top_dir"])
            break
if not pick:
    w("ABORT: no mid-size P2 file found")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    raise SystemExit(2)

p, s, d = pick
w("[2] probe file: %s (%s) in %s" % (p, human(s), d))
if not os.path.isfile(p):
    w("ABORT: probe file missing")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    raise SystemExit(3)

f0, b0 = free_d(), bin_stat()
w("    BEFORE  D_free=%s  bin=%s/%d files" % (human(f0), human(b0[0]), b0[1]))

os.remove(p)
time.sleep(0.5)
f1, b1 = free_d(), bin_stat()
w("[3] after os.remove: gone=%s  D_free=%s (%+s)  bin=%s/%d files (%+d files, %+s)" % (
    not os.path.exists(p), human(f1), human(f1 - f0), human(b1[0]), b1[1],
    b1[1] - b0[1], human(b1[0] - b0[0])))
with open(DELETED, "a", encoding="utf-8") as fh:
    fh.write("%d\t%s\n" % (s, p))

hr = empty_bin()
time.sleep(1.0)
f2, b2 = free_d(), bin_stat()
w("[4] after SHEmptyRecycleBinW: HRESULT=0x%08X  D_free=%s (%+s vs start)  bin=%s/%d files" % (
    hr & 0xFFFFFFFF, human(f2), human(f2 - f0), human(b2[0]), b2[1]))
w("")
w("    loop: bin_gain=%+s -> bin_after_empty=%+s ; free_gain=%+s" % (
    human(b1[0] - b0[0]), human(b2[0] - b0[0]), human(f2 - f0)))
w("")

released = f2 - f0
if hr == 0 and released >= s * 0.6 and b2[0] <= b0[0] + 1024 * 1024:
    verdict = "LOOP_OK"
    note = "delete -> empty bin -> space freed, and the newly created bin item was removed. P2 batching is safe."
elif hr == 0 and b2[0] > b0[0] + 1024 * 1024:
    verdict = "BIN_ACCUMULATES"
    note = "newly created bin items were NOT removed by the empty API; batching would fill the bin. STOP."
else:
    verdict = "LOOP_FAILED"
    note = "space did not come back as expected. STOP and report."
w("[5] VERDICT = " + verdict)
w("    " + note)
w("")
w("=== END LOOP VERIFY ===")

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
with open(LOG, "a", encoding="utf-8") as fh:
    fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
             "  loop_verify verdict=%s released=%+d probe=%s\n" % (verdict, released, human(s)))
with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_loop_verdict_20261109.json", "w", encoding="utf-8") as fh:
    json.dump({"verdict": verdict, "freed": released, "probe_size": s,
               "bin_before": b0[0], "bin_after_delete": b1[0], "bin_after_empty": b2[0]}, fh, indent=2)

print("VERDICT=" + verdict + " freed=" + str(released))
