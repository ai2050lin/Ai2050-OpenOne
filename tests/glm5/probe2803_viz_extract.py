# -*- coding: utf-8 -*-
# Phase 2803 viz data extraction -> probe2803_viz_data.txt
import numpy as np, json, os

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase2803\qwen4_polysemy_census"
NPZ  = os.path.join(BASE, "census.npz")
RJ   = os.path.join(BASE, "result.json")
OUT  = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2803_viz_data.txt"

lines = []
z = np.load(NPZ, allow_pickle=True)
lines.append("=== NPZ KEYS ===")
for k in z.files:
    a = z[k]
    try:
        lines.append(f"  {k}: shape={a.shape} dtype={a.dtype}")
        if a.size <= 64:
            lines.append(f"      vals={np.round(np.asarray(a, dtype=np.float64).ravel(), 3).tolist()}")
        elif a.dtype.kind in "US" or a.dtype == object:
            lines.append(f"      vals={a.ravel()[:30].tolist()}")
    except Exception as e:
        lines.append(f"  {k}: ERR {e}")

lines.append("=== RESULT.JSON ===")
with open(RJ, "r", encoding="utf-8") as f:
    r = json.load(f)
lines.append("TOP KEYS: " + ", ".join(r.keys()))

# dump everything except bulky pair records
def brief(v, n=400):
    s = json.dumps(v, ensure_ascii=False)
    return s[:n] + ("..." if len(s) > n else "")

for k, v in r.items():
    if isinstance(v, list) and len(v) > 5:
        lines.append(f"LIST {k}: n={len(v)} first={brief(v[0], 600)}")
    elif isinstance(v, dict):
        lines.append(f"DICT {k}: {brief(v, 2000)}")
    else:
        lines.append(f"{k} = {brief(v, 800)}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("WROTE", OUT, len(lines))
