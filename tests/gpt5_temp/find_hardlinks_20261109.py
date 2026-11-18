# -*- coding: utf-8 -*-
"""READ-ONLY: locate the second hard link of representative .npy files."""
import os
import subprocess
import datetime

BANK = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\mlp_atlas_code_v2\bank"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_hardlink_map_20261109.txt"

targets = [r"hidden.npy", r"logits.npy", r"B35-V.npy", r"B34-A.npy", r"B00-A.npy"]
L = []
w = L.append
w("HARD LINK MAP  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

for t in targets:
    p = os.path.join(BANK, t)
    w("[%s]" % t)
    if not os.path.isfile(p):
        w("    (missing)")
        continue
    try:
        st = os.stat(p)
        w("    size=%d nlink=%d ino=%s" % (st.st_size, getattr(st, "st_nlink", -1), st.st_ino))
    except OSError as e:
        w("    stat failed: " + repr(e))
    try:
        r = subprocess.run(["fsutil", "hardlink", "list", p], capture_output=True, timeout=40)
        txt = (r.stdout + r.stderr).decode("gbk", errors="ignore")
        for line in txt.splitlines():
            if line.strip():
                w("    LINK " + line.strip())
    except Exception as e:
        w("    fsutil failed: " + repr(e))
    w("")

w("=== END (READ-ONLY) ===")
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
