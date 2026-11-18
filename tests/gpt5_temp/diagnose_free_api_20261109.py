# -*- coding: utf-8 -*-
"""READ-ONLY: compare GetDiskFreeSpaceExW's 3 values, hard links, VSS."""
import os
import ctypes
import shutil
import subprocess
import datetime

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_free_api_diagnose_20261109.txt"
BANK = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\mlp_atlas_code_v2\bank"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


L = []
w = L.append
w("FREE-SPACE API / HARD-LINK / VSS DIAGNOSE  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

k32 = ctypes.windll.kernel32
avail = ctypes.c_ulonglong(0)
total = ctypes.c_ulonglong(0)
total_free = ctypes.c_ulonglong(0)
ok = k32.GetDiskFreeSpaceExW(
    ctypes.c_wchar_p("D:\\"),
    ctypes.byref(avail), ctypes.byref(total), ctypes.byref(total_free))
w("[1] GetDiskFreeSpaceExW D: rc=%s" % ok)
w("    free bytes available to caller = %s   <-- what shutil.disk_usage reports" % human(avail.value))
w("    total bytes                    = %s" % human(total.value))
w("    total free bytes (real)        = %s" % human(total_free.value))
w("    DIFF (quota / reserved)        = %s" % human(total_free.value - avail.value))
w("    shutil.disk_usage free         = %s" % human(shutil.disk_usage("D:\\").free))
w("")

try:
    r = subprocess.run(["fsutil", "volume", "diskfree", "D:"], capture_output=True, timeout=30)
    w("[2] fsutil volume diskfree D:")
    for line in (r.stdout + r.stderr).decode("gbk", errors="ignore").splitlines():
        if line.strip():
            w("    " + line.strip())
except Exception as e:
    w("[2] fsutil failed: " + repr(e))
w("")

w("[3] hard-link check on remaining bank files")
nlinks = {}
inode_seen = {}
samples = 0
for dp, dns, fns in os.walk(BANK):
    for fn in fns:
        p = os.path.join(dp, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        n = getattr(st, "st_nlink", 1)
        nlinks[n] = nlinks.get(n, 0) + 1
        key = (st.st_dev, st.st_ino)
        inode_seen[key] = inode_seen.get(key, 0) + 1
        samples += 1
w("    files=%d  st_nlink distribution=%s" % (samples, str(nlinks)))
w("    distinct (dev,ino)=%d  duplicated=%d" % (
    len(inode_seen), sum(1 for v in inode_seen.values() if v > 1)))
w("")

w("[4] VSS shadow copies on D:")
try:
    r = subprocess.run(["vssadmin", "list", "shadows", "/for=D:"], capture_output=True, timeout=40)
    txt = (r.stdout + r.stderr).decode("gbk", errors="ignore")
    for line in txt.splitlines():
        if line.strip():
            w("    " + line.strip()[:160])
except Exception as e:
    w("    vssadmin failed: " + repr(e))
w("")

w("[5] recycle bin size")
B = r"D:\$RECYCLE.BIN"
sz, nf = 0, 0
for dp, dns, fns in os.walk(B):
    for fn in fns:
        try:
            sz += os.lstat(os.path.join(dp, fn)).st_size
            nf += 1
        except OSError:
            pass
w("    %s / %d files" % (human(sz), nf))
w("")
w("=== END (READ-ONLY) ===")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
