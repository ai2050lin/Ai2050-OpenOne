# -*- coding: utf-8 -*-
# Phase 2803 viz data extraction: layer-diff curve + dose bars + default-sense ranking table
import numpy as np, json, os

BASE = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\tests\glm5\result\rdc_query_construction_20260913\phase2803"
NPZ  = os.path.join(BASE, "qwen4_polysemy_census", "census.npz")
RJ   = os.path.join(BASE, "qwen4_polysemy_census", "result.json")
OUT  = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\tests\glm5\probe2803_viz_data.txt"

lines = []
z = np.load(NPZ, allow_pickle=True)
lines.append("NPZ KEYS: " + ", ".join(z.files))
for k in z.files:
    try:
        lines.append(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")
    except Exception:
        pass

# ---- layer-diff curve (Arm E) ----
def dump_1d(name_hint, maxn=40):
    for k in z.files:
        if name_hint in k.lower():
            a = z[k]
            if hasattr(a, "ravel") and a.size <= 64:
                lines.append(f"CURVE {k} = {np.round(np.asarray(a, dtype=np.float64).ravel(), 3).tolist()}")

for hint in ["diff", "layer", "l_div", "dose", "comp_final", "final"]:
    dump_1d(hint)

# default sense ranking from result.json
if os.path.exists(RJ):
    with open(RJ, "r", encoding="utf-8") as f:
        r = json.load(f)
    lines.append("RJ TOP KEYS: " + ", ".join(list(r.keys())[:40]))
    # search for per-word default sense info
    def walk(obj, path=""):
        if isinstance(obj, dict):
            for kk, vv in obj.items():
                if isinstance(vv, (dict, list)) and kk not in ("record_pairs",):
                    walk(vv, path + "/" + str(kk))
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and len(obj) <= 30:
            lines.append(f"LIST {path}: n={len(obj)} keys={list(obj[0].keys())[:12]}")
    walk(r)
    # words + senses + default margin
    if "words" in r or "targets" in r:
        pass
    # try common placements
    for key in ("summary", "per_word", "census", "word_table", "ranking"):
        if key in r:
            lines.append(f"HASKEY {key}: {json.dumps(r[key], ensure_ascii=False)[:1500]}")

# dose curve known: comp_final = [-7.92,-2.16,-7.86,-2.88]
lines.append("KNOWN dose comp_final = [-7.92,-2.16,-7.86,-2.88]")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("OK", len(lines))
