# -*- coding: utf-8 -*-
"""Probe v2: nested fields of phase3124 result.json."""
import json, io

RJ = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3124\omega_p122_resid_cm_kalman_l35rel_glm4x\result.json"
out = io.StringIO()
w = out.write

with open(RJ, 'r', encoding='utf-8') as f:
    ds = json.load(f)

pa = ds['part_a']
det = pa['det_sim']
w("=== part_a.det_sim keys ===\n")
w(", ".join(sorted(det.keys())) + "\n")
w("auc_det = %r\n\n" % (det.get('auc_det'),))

pb = ds['part_b']
own = pb['own_sim']
w("=== part_b.own_sim keys ===\n")
w(", ".join(sorted(own.keys())) + "\n")
w("auc_own = %r\n\n" % (own.get('auc_own'),))

pd = ds['part_d']
w("=== part_d.lstar ===\n")
w(repr(pd.get('lstar')) + "\n\n")
w("=== part_d.curves keys ===\n")
cv = pd.get('curves', {})
w(", ".join(sorted(cv.keys())) + "\n")
for k in sorted(cv.keys()):
    w("%s = %r\n" % (k, cv[k]))
w("\n=== part_d.first_tokens ===\n")
w(repr(pd.get('first_tokens')) + "\n")

with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3124_probe_vals2.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("WROTE_OK")
