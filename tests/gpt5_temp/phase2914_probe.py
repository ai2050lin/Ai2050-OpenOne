# -*- coding: utf-8 -*-
# phase2914_probe.py -- anchor-source probe for Phase 2914 (ASCII output only)
# (a) phase2913 tree; (b) 2913 npz keys/shapes/margins; (c) 2913 result.json digest;
# (d) phase2903 tree; (e) 2903 result.json weights_descriptive['26']['attn']
import os, json
import numpy as np

BASE = r"D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913"
OUT = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/phase2914_probe.txt"
out = []
def w(s=""):
    out.append(str(s))

d2913 = os.path.join(BASE, "phase2913")
w("== phase2913 tree ==")
n13 = 0
for root, dirs, files in os.walk(d2913):
    for f in sorted(files):
        p = os.path.join(root, f)
        w(p + "  size=" + str(os.path.getsize(p)))
        n13 += 1
w("files=" + str(n13))

npz_path = os.path.join(d2913, "perhead_wvo_decomposition", "perhead_wvo_decomposition.npz")
res_path = os.path.join(d2913, "perhead_wvo_decomposition", "result.json")

w("")
w("== 2913 npz ==")
z = np.load(npz_path, allow_pickle=True)
for k in z.files:
    a = z[k]
    w(k + " shape=" + str(a.shape) + " dtype=" + str(a.dtype))
mh = np.asarray(z["margins_h"]).astype(np.float64).ravel()
w("margins_h (h0..h31):")
for i, v in enumerate(mh):
    w("  h" + str(i) + ": " + ("%.6f" % v))
order = np.argsort(-np.abs(mh))
w("top5 by |margin|: " + str([(int(i), round(float(mh[i]), 5)) for i in order[:5]]))

w("")
w("== 2913 result.json digest ==")
with open(res_path, "r", encoding="utf-8") as f:
    r13 = json.load(f)
w("top keys: " + json.dumps(list(r13.keys())))
def dig(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = prefix + "/" + str(k)
            dig(v, kk)
    elif isinstance(obj, list):
        if len(obj) > 12:
            w(prefix + " = list(len=" + str(len(obj)) + ") first3=" + json.dumps(obj[:3])[:200])
        else:
            for i, v in enumerate(obj):
                dig(v, prefix + "/" + str(i))
    else:
        s = str(obj)
        if len(s) > 160: s = s[:160] + "..."
        w(prefix + " = " + s)
dig(r13)

w("")
w("== phase2903 tree ==")
d2903 = os.path.join(BASE, "phase2903")
n03 = 0
for root, dirs, files in os.walk(d2903):
    for f in sorted(files):
        p = os.path.join(root, f)
        w(p + "  size=" + str(os.path.getsize(p)))
        n03 += 1
w("files=" + str(n03))

w("")
w("== 2903 result.json weights_descriptive['26'] ==")
cand = []
for root, dirs, files in os.walk(d2903):
    for f in files:
        if f == "result.json":
            cand.append(os.path.join(root, f))
w("result.json candidates: " + str(cand))
r03 = None
for c in cand:
    with open(c, "r", encoding="utf-8") as f:
        rr = json.load(f)
    if "weights_descriptive" in rr:
        r03 = rr
        w("found in: " + c)
        break
if r03 is None:
    w("NO weights_descriptive found in any 2903 result.json")
else:
    wd = r03["weights_descriptive"]
    w("weights_descriptive layers: " + json.dumps(sorted(wd.keys())))
    if "26" in wd:
        w("wd['26'] keys: " + json.dumps(sorted(wd["26"].keys())))
        if "attn" in wd["26"]:
            w("wd['26']['attn'] FULL:")
            w(json.dumps(wd["26"]["attn"], indent=1))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE", OUT, "lines", len(out), flush=True)
