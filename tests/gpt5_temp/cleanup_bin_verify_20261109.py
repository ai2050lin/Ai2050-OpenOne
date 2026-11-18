# -*- coding: utf-8 -*-
"""
Verify SHEmptyRecycleBinW actually frees D: space.
Before emptying, export a full inventory of current recycle-bin items
(original path / size / delete time) so nothing is cleared without a trace.
"""
import os
import ctypes
import struct
import shutil
import time
import datetime
import json

BIN_D = r"D:\$RECYCLE.BIN"
INV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\recycle_bin_inventory_20261109.txt"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_bin_verify_20261109.txt"
LOG = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_log_20261109.txt"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


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


def parse_i(p):
    try:
        with open(p, "rb") as fh:
            d = fh.read()
        if len(d) < 28:
            return None
        ver = struct.unpack_from("<Q", d, 0)[0]
        size = struct.unpack_from("<Q", d, 8)[0]
        ft = struct.unpack_from("<Q", d, 16)[0]
        if ver == 2:
            n = struct.unpack_from("<I", d, 24)[0]
            orig = d[28:28 + n * 2].decode("utf-16-le", errors="ignore").rstrip("\x00")
        else:
            orig = d[28:].decode("utf-16-le", errors="ignore").split("\x00")[0]
        try:
            dt = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=ft / 10)
            ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = "-"
        return size, orig, ts
    except Exception:
        return None


def free_d():
    return shutil.disk_usage("D:\\").free


L = []
w = L.append
w("RECYCLE BIN EMPTY VERIFICATION  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# ---- inventory ----
items = []
if os.path.isdir(BIN_D):
    for dp, dns, fns in os.walk(BIN_D):
        for fn in fns:
            p = os.path.join(dp, fn)
            if not fn.startswith("$I"):
                continue
            info = parse_i(p)
            try:
                st = os.lstat(p)
                imtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                imtime = "-"
            if info:
                items.append((info[0], info[2], info[1], imtime, p))
            else:
                items.append((0, "-", "(unparsed)", imtime, p))
items.sort(reverse=True)

with open(INV, "w", encoding="utf-8") as fh:
    fh.write("D RECYCLE BIN INVENTORY (exported before emptying)  " +
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    fh.write("cols: original_size | deleted_at | added_at | original_path | $I file\n")
    for sz, ts, orig, imtime, p in items:
        fh.write("%14s | %s | %s | %s | %s\n" % (human(sz), ts, imtime, orig, p))
    fh.write("total items = %d\n" % len(items))

w("[1] inventory exported: %d items -> %s" % (len(items), INV))
w("    (top 8 by size)")
for sz, ts, orig, imtime, p in items[:8]:
    w("      %10s  %s  %s" % (human(sz), imtime, orig[:110]))
w("")

f0, b0 = free_d(), bin_stat()
w("[2] BEFORE  D_free=%s  bin=%s/%d files" % (human(f0), human(b0[0]), b0[1]))

SHERB_NOCONFIRMATION = 0x00000001
SHERB_NOPROGRESSUI = 0x00000002
SHERB_NOSOUND = 0x00000004
flags = SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND

shell32 = ctypes.windll.shell32
shell32.SHEmptyRecycleBinW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
shell32.SHEmptyRecycleBinW.restype = ctypes.c_long
hr = shell32.SHEmptyRecycleBinW(None, "D:\\", flags)
w("[3] SHEmptyRecycleBinW(D:\\) HRESULT = 0x%08X (%s)" % (hr & 0xFFFFFFFF, "OK" if hr == 0 else "NONZERO"))

time.sleep(1.0)
f1, b1 = free_d(), bin_stat()
w("[4] AFTER   D_free=%s  bin=%s/%d files" % (human(f1), human(b1[0]), b1[1]))
w("    NET free delta = %+s ; bin size delta = %+s" % (human(f1 - f0), human(b1[0] - b0[0])))
w("")

if hr == 0 and f1 - f0 >= b0[0] * 0.5:
    verdict = "EMPTY_WORKS"
    note = "Emptying the D recycle bin releases space. P2 can run in batches: delete -> empty bin -> verify free."
elif hr != 0:
    verdict = "EMPTY_API_FAILED"
    note = "SHEmptyRecycleBinW did not succeed; stop and report."
else:
    verdict = "EMPTY_NO_FREE"
    note = "API succeeded but free space did not grow as expected; stop and report."
w("[5] VERDICT = " + verdict)
w("    " + note)
w("")
w("=== END BIN VERIFY ===")

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
with open(LOG, "a", encoding="utf-8") as fh:
    fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
             "  bin_empty verdict=%s free_delta=%+d bin_items=%d\n" % (verdict, f1 - f0, len(items)))
with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_bin_verdict_20261109.json", "w", encoding="utf-8") as fh:
    json.dump({"verdict": verdict, "hresult": hr, "free_delta": f1 - f0,
               "bin_items_emptied": len(items), "bin_bytes_before": b0[0]}, fh, indent=2)

print("VERDICT=" + verdict + " free_delta=" + str(f1 - f0))
