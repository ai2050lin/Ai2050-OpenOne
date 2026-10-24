# -*- coding: utf-8 -*-
"""服务端点级冒烟：直接调用 router 里的处理函数（不起 HTTP，不占端口）。"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from server import ai_rnd_service as S  # noqa: E402

MODEL = "qwen3-4b"
LAYER, DIM = 18, 219

sl = asyncio.run(S.atlas_neuron_slice(MODEL, LAYER))
print("slice keys:", sorted(sl.keys()))
print("n_dims =", sl["n_dims"], "pool =", sl["pool"])
print("axis:", sl["axis_semantics"])
fm = sl["factor_mean"]
assert fm is not None, "factor_mean 缺失"
print("factor_mean =", fm["mean"])
print("frac_cs>0.5 =", fm["frac_cs_gt_0_5"], "frac_ts>0.5 =", fm["frac_ts_gt_0_5"])
assert abs(sum(fm["mean"].values()) - 1.0) < 1e-2, fm["mean"]
assert fm["n_dims"] == sl["n_dims"]

pt = asyncio.run(S.atlas_neuron(MODEL, LAYER, DIM))
print("\npoint:", pt["model"], "L%s.D%s" % (pt["layer"], pt["dim"]))
print("stats:", pt["stats"])
assert "n_templates" in pt["stats"], pt["stats"]
assert pt["stats"]["n_templates"] == 3
pos = pt["poles"]["positive"]
neg = pt["poles"]["negative"]
for p in pos[:4]:
    print("  + %.3f  %s  [t%d pair%d]" % (p["value"], p["text"], p["template"], p["pair"]))
for p in neg[:4]:
    print("  - %.3f  %s" % (p["value"], p["text"]))
assert all(isinstance(p.get("text"), str) and p["text"] for p in pos + neg), "片段文本缺失"
assert all("entity" in p and "cls" in p for p in pos + neg)
f = pt["factor"]
print("\nfactor:", f["dominant"], f["shares"])
assert f is not None
assert abs(sum(f["shares"].values()) - 1.0) < 1e-2

# 越界与退化路径
try:
    asyncio.run(S.atlas_neuron_slice(MODEL, 999))
    print("FAIL: 越界层未报错")
except Exception as e:
    print("越界层正确拒绝:", type(e).__name__)
d0 = asyncio.run(S.atlas_neuron(MODEL, 0, 0))
print("L0.D0 degenerate =", d0["factor"]["degenerate"], "|", d0["factor"]["reading"][:28])
assert d0["factor"]["degenerate"] is True

print("\nOK 端点级冒烟通过")
