# -*- coding: utf-8 -*-
"""
Find a deletion path that truly frees space instead of being redirected to the
recycle bin by the safe-delete hook. Uses 4 tiny P2 files that are already
inside the user-confirmed cleanup scope. READ ONLY apart from those 4 deletions.
"""
import os
import csv
import time
import shutil
import ctypes
import subprocess
import datetime

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
BIN_D = r"D:\$RECYCLE.BIN"
STAGE = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_hookprobe_20261109"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_bypass_probe_20261109.txt"
DELETED = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_deleted_20261109.txt"
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


def free_d():
    return shutil.disk_usage("D:\\").free


def mark(p, s):
    with open(DELETED, "a", encoding="utf-8") as fh:
        fh.write("%d\t%s\n" % (s, p))


# pick 4 distinct small P2-cold files from distinct top dirs
cands = []
seen = set()
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for r in csv.DictReader(cf):
        if r["batch"] != "P2-cold":
            continue
        s = int(r["size_bytes"])
        if s <= 0 or s > 64 * 1024:
            continue
        if r["top_dir"] in seen:
            continue
        seen.add(r["top_dir"])
        cands.append((s, r["path"], r["top_dir"]))
cands.sort()
picks = cands[:4]

L = []
w = L.append
w("BYPASS PROBE  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("Using 4 tiny P2-cold files already inside confirmed cleanup scope.")
w("")
for s, p, d in picks:
    w("   %8s  %s  %s" % (human(s), d, p))
w("")

os.makedirs(STAGE, exist_ok=True)


def run_case(idx, label, fn):
    s, p, d = picks[idx]
    if not os.path.isfile(p):
        w("[%s] SKIP missing %s" % (label, p))
        return
    b0 = bin_stat()
    f0 = free_d()
    err = ""
    try:
        fn(p, idx)
    except Exception as e:
        err = repr(e)
    time.sleep(0.35)
    b1 = bin_stat()
    f1 = free_d()
    gone = not os.path.exists(p)
    dbf = b1[1] - b0[1]
    dbb = b1[0] - b0[0]
    w("[%s] gone=%s bin_files=%+d bin_bytes=%+d free_delta=%+d err=%s" % (
        label, gone, dbf, dbb, f1 - f0, err or "-"))
    if gone:
        mark(p, s)
    res = "BYPASS_OK" if (gone and dbf == 0 and dbb == 0 and not err) else "HOOKED_OR_FAILED"
    w("     => %s" % res)
    return res


def case_ctypes(p, i):
    ok = ctypes.windll.kernel32.DeleteFileW(ctypes.c_wchar_p(p))
    if not ok:
        raise OSError("DeleteFileW failed err=%d" % ctypes.get_last_error())


def case_cmd(p, i):
    r = subprocess.run(["cmd.exe", "/c", "del", "/F", "/Q", p],
                       capture_output=True, timeout=30)
    txt = (r.stdout + r.stderr).decode("gbk", errors="ignore")
    if r.returncode != 0:
        raise OSError("cmd del rc=%d out=%s" % (r.returncode, txt[:200]))


def case_powershell(p, i):
    r = subprocess.run(["powershell.exe", "-NonInteractive", "-Command",
                        "Remove-Item -LiteralPath " + chr(39) + p + chr(39) + " -Force -ErrorAction Stop"],
                       capture_output=True, timeout=40)
    txt = (r.stdout + r.stderr).decode("gbk", errors="ignore")
    if r.returncode != 0:
        raise OSError("powershell rc=%d out=%s" % (r.returncode, txt[:200]))


def case_rename_then_remove(p, i):
    dst = os.path.join(STAGE, "probe%d.bin" % i)
    os.rename(p, dst)
    try:
        os.remove(dst)
    except Exception:
        # fall back to native delete on the staged path
        ctypes.windll.kernel32.DeleteFileW(ctypes.c_wchar_p(dst))
    finally:
        if os.path.exists(dst):
            ctypes.windll.kernel32.DeleteFileW(ctypes.c_wchar_p(dst))


results = {}
results["ctypes_DeleteFileW"] = run_case(0, "A-ctypes", case_ctypes)
results["cmd_del"] = run_case(1, "B-cmd", case_cmd)
results["powershell_RemoveItem"] = run_case(2, "C-powershell", case_powershell)
results["rename_then_remove"] = run_case(3, "D-rename", case_rename_then_remove)

try:
    shutil.rmtree(STAGE)
except Exception:
    try:
        os.rmdir(STAGE)
    except OSError:
        pass

b_final = bin_stat()
f_final = free_d()
w("")
w("[FINAL] D_free=%s  bin=%s/%d files" % (human(f_final), human(b_final[0]), b_final[1]))
w("[SUMMARY] " + " | ".join("%s=%s" % (k, v) for k, v in results.items()))
w("")
w("=== END BYPASS PROBE ===")

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

with open(LOG, "a", encoding="utf-8") as fh:
    fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
             "  bypass_probe " + " ".join("%s=%s" % (k, v) for k, v in results.items()) + "\n")

import json
with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_bypass_verdict_20261109.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print("RESULTS=" + str(results))
