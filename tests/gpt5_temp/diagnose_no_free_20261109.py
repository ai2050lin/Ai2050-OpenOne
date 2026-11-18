# -*- coding: utf-8 -*-
"""Diagnose why 1.52 GB batches reported deleted but freed 0 bytes. READ-ONLY."""
import os
import csv
import time
import shutil
import datetime
import ctypes

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
DELETED = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_deleted_20261109.txt"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_diagnose_20261109.txt"
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


L = []
w = L.append
w("DIAGNOSE MISSING FREE SPACE  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

u = shutil.disk_usage("D:\\")
w("[1] D: total=%s used=%s free=%s" % (human(u.total), human(u.used), human(u.free)))
w("")

# 1. Are the reported-deleted files actually gone?
recent = []
with open(DELETED, "r", encoding="utf-8") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) == 2:
            recent.append((int(parts[0]), parts[1]))
w("[2] deleted-log entries = %d, bytes = %s" % (len(recent), human(sum(s for s, _ in recent))))
big = sorted([x for x in recent if x[0] > 100 * 1024 * 1024], reverse=True)
w("    big entries (>100MB) = %d" % len(big))
still_there = 0
for s, p in big[:25]:
    ex = os.path.exists(p)
    if ex:
        still_there += 1
    w("    %10s exists=%-5s %s" % (human(s), ex, p[-100:]))
w("    of sampled big files, still present = %d" % still_there)
w("")

# 2. File attributes (sparse / compressed / reparse) of similar NOT-yet-deleted files
attrs = {}
target_dir = os.path.join(ROOT, "mlp_atlas_code_v2")
sample = []
for dp, dns, fns in os.walk(target_dir):
    for fn in fns:
        p = os.path.join(dp, fn)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        sample.append((st.st_size, p, getattr(st, "st_file_attributes", 0)))
    if len(sample) > 500:
        break
w("[3] mlp_atlas_code_v2 remaining sample = %d files, %s" % (
    len(sample), human(sum(x[0] for x in sample))))
for s, p, a in sorted(sample, reverse=True)[:8]:
    flags = []
    if a & 0x200:
        flags.append("SPARSE")
    if a & 0x800:
        flags.append("COMPRESSED")
    if a & 0x400:
        flags.append("REPARSE")
    w("    %10s attrs=0x%X %-12s %s" % (human(s), a, ",".join(flags) or "normal", p[-90:]))
w("")

# 3. Actual allocated size via GetCompressedFileSizeW for a few large files
w("[4] logical size vs allocated size (GetCompressedFileSizeW)")
k32 = ctypes.windll.kernel32
k32.GetCompressedFileSizeW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
k32.GetCompressedFileSizeW.restype = ctypes.c_uint32
for s, p, a in sorted(sample, reverse=True)[:6]:
    hi = ctypes.c_uint32(0)
    lo = k32.GetCompressedFileSizeW(p, ctypes.byref(hi))
    alloc = (hi.value << 32) | lo
    w("    logical=%10s allocated=%10s ratio=%5.1f%%  %s" % (
        human(s), human(alloc), 100.0 * alloc / s if s else 0, p[-70:]))
w("")

# 4. Recycle bin current state + newest items
w("[5] D recycle bin")
BIN = r"D:\$RECYCLE.BIN"
bsz, bfiles, newest = 0, 0, []
if os.path.isdir(BIN):
    for dp, dns, fns in os.walk(BIN):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                st = os.lstat(p)
                bsz += st.st_size
                bfiles += 1
                newest.append((st.st_mtime, st.st_size, p))
            except OSError:
                pass
w("    size=%s files=%d" % (human(bsz), bfiles))
for mt, s, p in sorted(newest, reverse=True)[:8]:
    w("    %s %10s %s" % (datetime.datetime.fromtimestamp(mt).strftime("%H:%M:%S"), human(s), p[-80:]))
w("")

# 5. Top-level D: dirs modified in last hour, with quick size of first level
w("[6] D: top-level entries modified in last 60 min")
for name in os.listdir("D:\\"):
    p = os.path.join("D:\\", name)
    try:
        st = os.lstat(p)
    except OSError:
        continue
    if st.st_mtime < time.time() - 3600:
        continue
    w("    %s  mtime=%s  %s" % (
        "DIR " if os.path.isdir(p) else "FILE",
        datetime.datetime.fromtimestamp(st.st_mtime).strftime("%H:%M:%S"),
        name[:90]))
w("")

# 6. skipped: psutil.open_files() segfaults on this Windows host
w("[7] skipped (handle enumeration unstable on this host)")
w("")
w("=== END DIAGNOSE (READ-ONLY) ===")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
