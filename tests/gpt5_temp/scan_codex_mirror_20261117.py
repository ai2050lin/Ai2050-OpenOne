# -*- coding: utf-8 -*-
"""READ-ONLY: resolve the C: codex mirror / D: visualization-research relationship."""
import os
import ctypes
import time
from datetime import datetime

CODEX = r"C:\Users\Admin\.codex\visualizations"
VR = r"D:\ai2050_native\visualization-research"
RESULT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_codex_mirror_20261117.txt"

REPARSE = 0x400
GFA = ctypes.windll.kernel32.GetFileAttributesW
GFA.restype = ctypes.c_uint32
GFA.argtypes = [ctypes.c_wchar_p]


def is_reparse(p):
    a = GFA(p)
    return a != 0xFFFFFFFF and bool(a & REPARSE)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def ts(x):
    return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else "-"


def deep_measure(p, follow_reparse=False, depth_cap=12):
    """count real files; if follow_reparse, descend into junctions but cap depth."""
    total = files = 0
    last = 0
    junc = 0
    stack = [(p, 0)]
    seen = set()
    while stack:
        cur, dep = stack.pop()
        if dep > depth_cap:
            continue
        try:
            rp = os.path.realpath(cur)
        except Exception:
            rp = cur
        if rp in seen:
            continue
        seen.add(rp)
        try:
            with os.scandir(cur) as it:
                for e in it:
                    q = e.path
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if is_reparse(q):
                                junc += 1
                                if follow_reparse:
                                    stack.append((q, dep + 1))
                            else:
                                stack.append((q, dep + 1))
                        else:
                            st = e.stat(follow_symlinks=False)
                            total += st.st_size
                            files += 1
                            if st.st_mtime > last:
                                last = st.st_mtime
                    except OSError:
                        pass
        except OSError:
            pass
    return total, files, last, junc


L = []
w = L.append
w("CODEX MIRROR RESOLUTION / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

w("[1] C: CODEX TREE (no junction following)")
w("    root exists = %s ; root is_reparse = %s" %
  (os.path.exists(CODEX), is_reparse(CODEX) if os.path.exists(CODEX) else "-"))
b, f, last, j = deep_measure(CODEX, follow_reparse=False)
w("    real files (junctions NOT followed) = %s / %d files / junctions=%d / last=%s" %
  (human(b), f, j, ts(last)))
b2, f2, last2, j2 = deep_measure(CODEX, follow_reparse=True)
w("    WITH junction following = %s / %d files / junctions=%d / last=%s" %
  (human(b2), f2, j2, ts(last2)))
w("")

w("[2] C: CODEX\\2026\\11 UUID FOLDERS")
base = os.path.join(CODEX, "2026", "11")
if os.path.isdir(base):
    for d in sorted(os.listdir(base)):
        q = os.path.join(base, d)
        if os.path.isdir(q):
            b3, f3, l3, j3 = deep_measure(q, follow_reparse=True)
            w("    %-42s reparse=%-5s %10s / %6d files / %s" %
              (d[:42], is_reparse(q), human(b3), f3, ts(l3)))
else:
    w("    base MISSING " + base)
w("")

w("[3] 9 JUNCTION TARGETS UNDER result")
names = ["mlp_puzzle_v1", "mlp_trigger_read100_v1", "mlp_trigger_v1",
         "qwen14_mlp_atlas_v1", "record_selection_v1", "record_selection_v2",
         "record_selection_v3", "source_output_v1", "state_formation_v1"]
tot_t = 0
for n in names:
    p = os.path.join(RESULT, n)
    if not os.path.lexists(p) or not is_reparse(p):
        w("    %-38s NOT A JUNCTION" % n)
        continue
    try:
        tgt = os.readlink(p).replace("\\\\?\\", "")
    except Exception as e:
        tgt = "ERR " + repr(e)
    ok = os.path.exists(tgt)
    if ok:
        b4, f4, l4, j4 = deep_measure(tgt, follow_reparse=True)
        tot_t += b4
        w("    %-38s exists=%s %10s / %5d files  last=%s" %
          (n, ok, human(b4), f4, ts(l4)))
    else:
        w("    %-38s TARGET MISSING %s" % (n, tgt[-90:]))
w("    junction target total = %s" % human(tot_t))
w("")

w("[4] D: visualization-research (independent measure)")
b5, f5, l5, j5 = deep_measure(VR, follow_reparse=True)
w("    %s = %s / %d files / junctions=%d / last=%s" %
  (VR, human(b5), f5, j5, ts(l5)))
for n in sorted(os.listdir(VR)):
    q = os.path.join(VR, n)
    if os.path.isdir(q):
        b6, f6, l6, j6 = deep_measure(q, follow_reparse=True)
        w("    %-38s %10s / %6d files / %s%s" %
          (n[:38], human(b6), f6, ts(l6), "  [JUNCTION]" if is_reparse(q) else ""))
w("")

w("[5] HARDLINK SAMPLE (qwen14_mlp_atlas_v1)")
qa = os.path.join(VR, "qwen14_mlp_atlas_v1")
if os.path.isdir(qa):
    cnt = 0
    for dp, dns, fns in os.walk(qa):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if getattr(st, "st_nlink", 1) >= 2:
                w("    nlink=%d %10s %s" % (st.st_nlink, human(st.st_size), fn[:70]))
                cnt += 1
                if cnt >= 8:
                    break
        if cnt >= 8:
            break
    w("    sampled nlink>=2 = %d" % cnt)
w("")

w("===== END CODEX RESOLUTION =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("CODEX OK c_nofollow=%s c_follow=%s vr=%s" % (human(b), human(b2), human(b5)))
