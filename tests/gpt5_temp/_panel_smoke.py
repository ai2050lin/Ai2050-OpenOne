# -*- coding: utf-8 -*-
"""neuron_panel 冒烟：真实 npz + 封存 execution.json，验证片段还原与因子分解。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from server import neuron_panel as NP  # noqa: E402

ix = NP.load_index()
print("design:", ix["design"])
print("axis:", ix["axis_semantics"])
print("source:", ix["source"])
print("snippet(t=1, pair=0):", NP.snippet(1, 0))
print("snippet(t=2, pair=9):", NP.snippet(2, 9))
assert NP.snippet(0, 0).endswith("。")

H = np.load(os.path.join(NP.PANEL_DIR, "collect_qwen3-4b.npz"))["H"]
print("H:", H.shape, H.dtype)
assert H.shape[0] == ix["nt"] and H.shape[1] == ix["pairs"]

for layer, dim in [(9, 222), (18, 219), (27, 91), (0, 0)]:
    p = NP.poles(H, layer, dim, k=3)
    f = NP.factor_of_column(H[:, :, layer, dim])
    print("\nL%d.D%d  factor: T=%.2f C=%.2f E=%.2f  dominant=%s  (te=%.2f tc=%.2f ec=%.2f tec=%.2f)"
          % (layer, dim, f["shares"]["t"], f["shares"]["c"], f["shares"]["e"],
             f["dominant"], f["shares"]["te"], f["shares"]["tc"],
             f["shares"]["ec"], f["shares"]["tec"]))
    s = sum(f["shares"].values())
    print("  shares sum = %.6f (应为 1；恒定坐标除外)  degenerate=%s" % (s, f["degenerate"]))
    assert f["degenerate"] or abs(s - 1.0) < 1e-3, s
    for r in p["positive"]:
        print("  + %.3f  %s" % (r["value"], r["text"]))
    for r in p["negative"]:
        print("  - %.3f  %s" % (r["value"], r["text"]))

print("\n--- 整层因子均值 L18 ---")
m = NP.layer_factor_means(H[:, :, 18, :])
print("mean:", m["mean"])
print("frac_cs>0.5 =", m["frac_cs_gt_0_5"], " frac_ts>0.5 =", m["frac_ts_gt_0_5"])
print("top class dims:", m["top_class_dims"])
print("\nOK")
