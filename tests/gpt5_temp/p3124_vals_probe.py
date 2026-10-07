# -*- coding: utf-8 -*-
"""Probe: extract actual values from phase3124 result.json for closeout assertion fix."""
import json, io, sys

RJ = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3124\omega_p122_resid_cm_kalman_l35rel_glm4x\result.json"
out = io.StringIO()

with open(RJ, 'r', encoding='utf-8') as f:
    ds = json.load(f)

w = out.write
w("=== top keys ===\n")
w(", ".join(sorted(ds.keys())) + "\n\n")

w("=== auc_det (full) ===\n")
w(repr(ds.get('auc_det')) + "\n\n")

# dump every numeric field with repr for cross-check
w("=== all fields repr (skip huge arrays) ===\n")
for k in sorted(ds.keys()):
    v = ds[k]
    if isinstance(v, list):
        if len(v) > 64:
            w("%s: list len=%d head=%r\n" % (k, len(v), v[:8]))
        else:
            w("%s: %r\n" % (k, v))
    elif isinstance(v, dict):
        w("%s: dict keys=%s\n" % (k, sorted(v.keys())))
    else:
        w("%s: %r\n" % (k, v))

with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3124_probe_vals.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("WROTE_OK")
