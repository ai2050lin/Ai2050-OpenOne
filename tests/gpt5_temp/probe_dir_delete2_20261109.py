# -*- coding: utf-8 -*-
"""
Benchmark directory-level deletion by first moving real (approved) P2 files
into isolated temp dirs with os.rename (instant, no delete hook), then deleting
each whole directory. No new user data is created.
"""
import os
import csv
import time
import shutil
import ctypes
import subprocess
import datetime

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
STAGE = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_dirprobe2_20261109"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_dir_probe2_20261109.txt"
DELETED = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_deleted_20261109.txt"
N = 400


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


L = []
w = L.append
w("DIR DELETE BENCHMARK v2  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

# collect N*2 still-existing P2 files, prefer small npz under native_mlp_function_v4
chosen = []
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for r in csv.DictReader(cf):
        if r["batch"] != "P2-cold":
            continue
        if "native_mlp_function_v4" not in r["path"]:
            continue
        if not os.path.isfile(r["path"]):
            continue
        chosen.append((r["path"], int(r["size_bytes"])))
        if len(chosen) >= N * 2:
            break
w("[0] collected %d approved files, %s" % (len(chosen), human(sum(s for _, s in chosen))))

os.makedirs(STAGE, exist_ok=True)
groups = {}
for gi in range(2):
    d = os.path.join(STAGE, "g%d" % gi)
    os.makedirs(d, exist_ok=True)
    moved = []
    for i, (src, size) in enumerate(chosen[gi * N:(gi + 1) * N]):
        dst = os.path.join(d, "%05d_%s" % (i, os.path.basename(src)))
        try:
            os.rename(src, dst)
            moved.append((src, size))
        except OSError:
            pass
    groups[gi] = (d, moved)
    w("[0] group %d: %d files / %s" % (gi, len(moved), human(sum(s for _, s in moved))))
w("")


def bench(label, d, moved, fn):
    b0 = bin_stat()[0]
    f0 = free_d()
    t0 = time.time()
    err = ""
    try:
        fn(d)
    except Exception as e:
        err = repr(e)
    dt = time.time() - t0
    hr = empty_bin()
    time.sleep(0.3)
    b1 = bin_stat()[0]
    f1 = free_d()
    gone = not os.path.exists(d)
    w("[%s] elapsed=%.2fs  ms/file=%.2f  gone=%s" % (label, dt, 1000.0 * dt / max(len(moved), 1), gone))
    w("      bin=%s->%s  free_delta=%+s  empty_hr=%d  err=%s" % (
        human(b0), human(b1), human(f1 - f0), hr, err or "-"))
    if gone:
        with open(DELETED, "a", encoding="utf-8") as fh:
            fh.write("\n".join("%d\t%s" % (s, p) for p, s in moved) + "\n")
    else:
        w("      !! directory survived, files still at " + d)
    return dt, f1 - f0, gone


def via_rmtree(p):
    shutil.rmtree(p)


def via_cmd(p):
    r = subprocess.run(["cmd.exe", "/c", "rmdir", "/S", "/Q", p],
                       capture_output=True, timeout=900)
    txt = (r.stdout + r.stderr).decode("gbk", errors="ignore")
    if r.returncode != 0:
        raise OSError("rmdir rc=%d out=%s" % (r.returncode, txt[:200]))


r1 = bench("A-shutil.rmtree", groups[0][0], groups[0][1], via_rmtree)
r2 = bench("B-cmd-rmdir", groups[1][0], groups[1][1], via_cmd)

w("")
w("[1] RESULT")
w("    rmtree : %.2fs for %d files, freed=%s" % (r1[0], len(groups[0][1]), human(r1[1])))
w("    rmdir  : %.2fs for %d files, freed=%s" % (r2[0], len(groups[1][1]), human(r2[1])))
w("=== END ===")
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("DONE rmtree=%.2fs rmdir=%.2fs" % (r1[0], r2[0]))
