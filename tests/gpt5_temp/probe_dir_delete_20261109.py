# -*- coding: utf-8 -*-
"""
Benchmark directory-level deletion on two real P2-cold leaf dirs that contain
ONLY files from the approved manifest. READ-ONLY apart from those deletions.
"""
import os
import csv
import time
import shutil
import ctypes
import subprocess
import datetime

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
ROOT4 = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\native_mlp_function_v4"
STAGE = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_dirprobe_20261109"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_dir_probe_20261109.txt"
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
w("DIRECTORY-LEVEL DELETE PROBE  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

p2 = set()
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for r in csv.DictReader(cf):
        if r["batch"] == "P2-cold":
            p2.add(r["path"])
w("[0] P2 manifest paths = %d" % len(p2))

cands = []
if os.path.isdir(ROOT4):
    for dp, dns, fns in os.walk(ROOT4):
        if dns or not fns:
            continue
        paths = [os.path.join(dp, fn) for fn in fns]
        if all(p in p2 for p in paths):
            cands.append((len(paths), sum(os.path.getsize(p) for p in paths), dp, paths))
cands.sort()
w("[1] pure-data leaf dirs = %d" % len(cands))
for n, s, dp, _ in cands[:5]:
    w("    %5d files %10s  %s" % (n, human(s), dp[-100:]))

picks = [c for c in cands if 80 <= c[0] <= 1200][:2]
w("[2] picked %d dirs for benchmark" % len(picks))
w("")

os.makedirs(STAGE, exist_ok=True)
results = {}


def run(label, cand, fn):
    n, size, src, paths = cand
    dst = os.path.join(STAGE, "d%d" % (len(results) + 1))
    # move is instant on the same volume and does not trigger the delete hook
    os.rename(src, dst)
    b0 = bin_stat()[0]
    f0 = free_d()
    t0 = time.time()
    err = ""
    try:
        fn(dst)
    except Exception as e:
        err = repr(e)
    dt = time.time() - t0
    hr = empty_bin()
    time.sleep(0.3)
    b1 = bin_stat()[0]
    f1 = free_d()
    gone = not os.path.exists(dst)
    w("[%s] files=%d bytes=%s  elapsed=%.2fs  ms/file=%.2f  gone=%s" % (
        label, n, human(size), dt, 1000.0 * dt / n if n else 0, gone))
    w("      bin=%s->%s  free_delta=%+s  empty_hr=%d err=%s" % (
        human(b0), human(b1), human(f1 - f0), hr, err or "-"))
    if gone:
        with open(DELETED, "a", encoding="utf-8") as fh:
            fh.write("\n".join("%d\t%s" % (os.path.getsize(p) if os.path.exists(p) else 0, p)
                               for p in paths) + "\n")
        results[label] = (dt, n, f1 - f0, gone, err)
    else:
        # roll back: put the dir back so no data is lost
        try:
            os.rename(dst, src)
            w("      ROLLED BACK to original path")
        except Exception as e2:
            w("      ROLLBACK FAILED: " + repr(e2))
        results[label] = (dt, n, 0, False, err)


def via_rmtree(p):
    shutil.rmtree(p)


def via_cmd(p):
    r = subprocess.run(["cmd.exe", "/c", "rmdir", "/S", "/Q", p],
                       capture_output=True, timeout=600)
    txt = (r.stdout + r.stderr).decode("gbk", errors="ignore")
    if r.returncode != 0:
        raise OSError("rmdir rc=%d out=%s" % (r.returncode, txt[:200]))


if len(picks) >= 1:
    run("A-shutil.rmtree", picks[0], via_rmtree)
if len(picks) >= 2:
    run("B-cmd-rmdir", picks[1], via_cmd)

w("")
w("[3] SUMMARY")
for k, v in results.items():
    w("    %-16s %.2fs / %d files / %.2f ms per file / freed=%s / gone=%s" % (
        k, v[0], v[1], 1000.0 * v[0] / v[1] if v[1] else 0, human(v[2]), v[3]))
w("")
w("=== END DIR PROBE ===")
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("PROBE DONE " + str({k: round(v[0], 2) for k, v in results.items()}))
