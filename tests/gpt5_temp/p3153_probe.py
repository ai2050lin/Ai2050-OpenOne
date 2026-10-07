# -*- coding: utf-8 -*-
"""3153 前置探针：检查 3152 collect.npz 与 result.json 数据结构（只读）。"""
import json, io, os
import numpy as np

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3152\g1p2_tri_model_k1"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3153_probe.txt"

out = []
for m in ["qwen3-4b", "qwen3-14b", "glm4k1"]:
    p = os.path.join(BASE, m, "collect.npz")
    z = np.load(p, allow_pickle=True)
    out.append("== %s ==" % m)
    for k in sorted(z.files):
        a = z[k]
        out.append("  %-24s shape=%-18s dtype=%s" % (k, str(a.shape), str(a.dtype)))

r = json.load(io.open(os.path.join(BASE, "qwen3-4b", "result.json"), encoding="utf-8"))
out.append("result keys: " + ",".join(sorted(r.keys())))
w = r["a3_worst20"][0]
out.append("worst20[0] keys: " + ",".join(sorted(w.keys())))
out.append("worst20[0]: " + json.dumps(w, ensure_ascii=False))
out.append("m1_grid keys sample: " + ",".join(sorted(r["m1_grid"].keys())[:12]))
# v5 结构（类子空间）
out.append("v5: " + json.dumps(r["v5"], ensure_ascii=False))
# panel 元信息
for k in ["panel_path", "panel_n", "panel_note", "kstar", "kout", "nlayers", "hidden"]:
    if k in r:
        out.append("%s = %s" % (k, r[k]))

io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("written", len(out), "lines")
