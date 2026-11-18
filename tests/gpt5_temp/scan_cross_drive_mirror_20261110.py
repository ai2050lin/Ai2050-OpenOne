# -*- coding: utf-8 -*-
"""Check the C: research-storage mirror vs D: ai2050_native visualization-research."""
import os
from datetime import datetime

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_crossdrive_mirror_20261110.txt"
CROOT = r"C:\Users\Admin\.codex\visualizations\2026\11"
DROOT = r"D:\ai2050_native\visualization-research"
RES = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"

L = []
def w(s=""):
    L.append(str(s))
def gb(n):
    return "%.2f GB" % (n / 1024.0 ** 3)

def dsize(path):
    tot, nf = 0, 0
    for root, dirs, files in os.walk(path):
        for fn in files:
            try:
                tot += os.stat(os.path.join(root, fn)).st_size
                nf += 1
            except Exception:
                pass
    return tot, nf

w("CROSS-DRIVE MIRROR CHECK (READ-ONLY)  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

w("[1] C: research-storage experiment dirs")
cmap = {}
if os.path.isdir(CROOT):
    for day in sorted(os.listdir(CROOT)):
        dp = os.path.join(CROOT, day)
        if not os.path.isdir(dp):
            continue
        for uuid in os.listdir(dp):
            rs = os.path.join(dp, uuid, "research-storage")
            if not os.path.isdir(rs):
                continue
            for exp in os.listdir(rs):
                p = os.path.join(rs, exp)
                if not os.path.isdir(p):
                    continue
                sz, nf = dsize(p)
                cmap[exp] = (sz, nf, p)
w("    found %d experiment dirs" % len(cmap))
for k in sorted(cmap, key=lambda k: -cmap[k][0]):
    sz, nf, p = cmap[k]
    w("    %-38s %10s %7d files  %s" % (k[:38], gb(sz), nf, p[-70:]))
w("")

w("[2] D: ai2050_native\\visualization-research experiment dirs")
dmap = {}
if os.path.isdir(DROOT):
    for exp in os.listdir(DROOT):
        p = os.path.join(DROOT, exp)
        if os.path.isdir(p):
            sz, nf = dsize(p)
            dmap[exp] = (sz, nf, p)
w("    found %d experiment dirs" % len(dmap))
for k in sorted(dmap, key=lambda k: -dmap[k][0]):
    sz, nf, p = dmap[k]
    w("    %-38s %10s %7d files" % (k[:38], gb(sz), nf))
w("")

w("[3] MIRROR COMPARISON (same name on C: and D:)")
tot_c = tot_d = 0
for k in sorted(set(cmap) | set(dmap), key=lambda k: -(dmap.get(k, (0,))[0])):
    c = cmap.get(k)
    d = dmap.get(k)
    w("    %-38s C=%-10s D=%-10s %s" % (
        k[:38],
        gb(c[0]) if c else "-",
        gb(d[0]) if d else "-",
        "MIRROR-BOTH" if c and d else ("C-ONLY" if c else "D-ONLY")))
    if c and d:
        tot_c += c[0]
        tot_d += d[0]
w("    paired totals: C=%s  D=%s" % (gb(tot_c), gb(tot_d)))
w("")

w("[4] JUNCTION HEALTH (result-side -> C: target)")
try:
    names = [x for x in os.listdir(RES) if os.path.isdir(os.path.join(RES, x))]
except Exception:
    names = []
ok = bad = 0
for n in names:
    p = os.path.join(RES, n)
    try:
        st = os.stat(p, follow_symlinks=False)
        if not (getattr(st, "st_file_attributes", 0) & 0x400):
            continue
        tgt = os.readlink(p)
        t = tgt.replace("\\\\?\\", "")
        exists = os.path.isdir(t)
        if exists:
            sz, nf = dsize(t)
            w("    OK    %-34s -> %s / %d files %s" % (n[:34], gb(sz), nf, t[-58:]))
            ok += 1
        else:
            w("    BROKEN %-34s -> %s" % (n[:34], t[-80:]))
            bad += 1
    except Exception as e:
        w("    ERROR %-34s %s" % (n[:34], repr(e)[:60]))
        bad += 1
w("    healthy=%d broken=%d" % (ok, bad))
w("")

w("[5] CONCLUSION INPUT")
w("    If all C: targets are healthy, the D: ai2050_native\\visualization-research")
w("    copy is a redundant intermediate mirror and can be deleted without")
w("    breaking the client endpoints (they resolve through the C: junctions).")
w("    If any C: target is broken/missing, keep its D: counterpart until restored.")
w("")
w("=== END (READ-ONLY) ===")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("done", len(L), "lines")
