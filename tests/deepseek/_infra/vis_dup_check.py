# -*- coding: utf-8 -*-
"""Compare tests client_visualization_assets vs frontend/public/vis_data (read-only)."""
import os
import time

TESTS_VIS = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\client_visualization_assets"
FE_VIS = r"D:\AI2050\Ai2050-OpenOne\frontend\public\vis_data"
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\vis_dup_check.txt"

def walk(path):
    out = {}
    if not os.path.isdir(path):
        return None
    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[os.path.relpath(p, path)] = (st.st_size, st.st_mtime)
    return out

lines = ["vis_data duplicate check %s" % time.strftime("%Y-%m-%d %H:%M")]
t = walk(TESTS_VIS)
f = walk(FE_VIS)
if t is None:
    lines.append("tests vis dir MISSING")
else:
    lines.append("tests client_visualization_assets: %d files, %.3f GB" % (len(t), sum(v[0] for v in t.values()) / 1024**3))
if f is None:
    lines.append("frontend/public/vis_data MISSING")
else:
    lines.append("frontend/public/vis_data: %d files, %.3f GB" % (len(f), sum(v[0] for v in f.values()) / 1024**3))

if t and f:
    tk = {k.replace("\\", "/") for k in t}
    fk = {k.replace("\\", "/") for k in f}
    common = tk & fk
    same = sum(1 for k in common if t[k][0] == f[k][0])
    tb = sum(t[k][0] for k in common if t[k][0] == f[k][0])
    lines.append("common(rel-path)=%d size-identical=%d (%.3f GB)" % (len(common), same, tb / 1024**3))
    only_t = tk - fk
    only_f = fk - tk
    lines.append("only in tests: %d files (%.3f GB)" % (len(only_t), sum(t[k][0] for k in only_t) / 1024**3))
    lines.append("only in frontend: %d files (%.3f GB)" % (len(only_f), sum(f[k][0] for k in only_f) / 1024**3))
    # sample each side
    for k in sorted(only_t)[:10]:
        lines.append("  T-only: %s (%.1f MB)" % (k, t[k][0] / 1024.0 / 1024.0))
    for k in sorted(only_f)[:10]:
        lines.append("  F-only: %s (%.1f MB)" % (k, f[k][0] / 1024.0 / 1024.0))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("done")
