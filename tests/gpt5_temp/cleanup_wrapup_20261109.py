# -*- coding: utf-8 -*-
"""READ-ONLY survey of orphan blobs + remove EMPTY temp dirs only."""
import os
import shutil
import datetime

BLOBS = r"D:\AI2050\Ai2050-OpenOne\distributed_data\research_native_blobs"
TMP_ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp"
EMPTY_DIRS = [
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_cleanup_stage_20261109",
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_dirprobe_20261109",
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_dirprobe2_20261109",
    r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_hookprobe_20261109",
]
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_wrapup_20261109.txt"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


L = []
w = L.append
w("CLEANUP WRAP-UP  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")
w("[1] CAS BLOB STORE (NOT touched by this cleanup)")
solo = [0, 0]
linked = [0, 0]
for dp, dns, fns in os.walk(BLOBS):
    for fn in fns:
        p = os.path.join(dp, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if getattr(st, "st_nlink", 1) <= 1:
            solo[0] += st.st_size
            solo[1] += 1
        else:
            linked[0] += st.st_size
            linked[1] += 1
w("    nlink=1  (orphan, no result-side link): %10s / %5d files" % (human(solo[0]), solo[1]))
w("    nlink>=2 (still referenced)           : %10s / %5d files" % (human(linked[0]), linked[1]))
w("    => if you later decide to drop orphan blobs, up to %s more can be freed." % human(solo[0]))
w("    NOTE: this directory was explicitly marked DO-NOT-TOUCH earlier (server")
w("          authoritative data). No blob was deleted in this cleanup.")
w("")
w("[2] EMPTY TEMP DIRECTORIES")
for d in EMPTY_DIRS:
    if not os.path.isdir(d):
        w("    absent   " + d)
        continue
    # only remove if truly empty (walks bottom-up, never deletes files)
    removed = False
    try:
        for dp, dns, fns in os.walk(d, topdown=False):
            if not fns and not dns:
                os.rmdir(dp)
        removed = not os.path.exists(d)
    except OSError as e:
        w("    rmdir error %s : %r" % (d, e))
    w("    %s %s" % ("removed " if removed else "kept   ", d))
w("")
w("=== END WRAP-UP ===")
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
