# -*- coding: utf-8 -*-
"""Phase 2917 seal probe: verify v2 artifacts on real disk + SHA256-8 registry.
Writes report file (bash stdout unreliable on this box). Read the report after.
"""
import json, hashlib, os
import numpy as np

BASE = r"D:\AI2050\Ai2050-OpenOne"
ART = os.path.join(BASE, "tests", "glm5", "result", "rdc_query_construction_20260913", "phase2917", "event_atlas")
SCRIPT = os.path.join(BASE, "tests", "glm5", "phase2917_event_atlas.py")
OUT = os.path.join(BASE, "tests", "gpt5_temp", "phase2917_seal_report.txt")

def sha8(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

L = []
for fn in ("execution.json", "result.json", "event_atlas.npz"):
    p = os.path.join(ART, fn)
    if os.path.exists(p):
        L.append("%s | size=%d | sha256-8=%s" % (fn, os.path.getsize(p), sha8(p)))
    else:
        L.append("%s | MISSING" % fn)
L.append("script | sha256-8=%s | %s" % (sha8(SCRIPT), SCRIPT))

with open(os.path.join(ART, "execution.json"), encoding="utf-8") as f:
    ex = json.load(f)
L.append("execution.created=%s" % ex.get("created"))
L.append("execution.phase=%s" % ex.get("phase"))
pr = ex.get("prereg", {})
L.append("prereg keys: %s" % ", ".join(sorted(pr.keys())))
for k in sorted(pr.keys()):
    v = str(pr[k]).replace("\n", " ")
    L.append("prereg.%s = %s" % (k, v[:300]))

with open(os.path.join(ART, "result.json"), encoding="utf-8") as f:
    res = json.load(f)
L.append("verdict=%s" % res.get("verdict"))
L.append("result top-level keys: %s" % ", ".join(sorted(res.keys())))
for k in sorted(res.keys()):
    v = res[k]
    if isinstance(v, (int, float, str, bool)) or v is None:
        L.append("result.%s = %s" % (k, v))
    elif isinstance(v, list) and len(v) <= 30 and all(isinstance(x, (int, float, str)) for x in v):
        L.append("result.%s = %s" % (k, v))

z = np.load(os.path.join(ART, "event_atlas.npz"), allow_pickle=True)
L.append("npz keys: %s" % ", ".join(z.files))
if "sign_M" in z.files and "p_maxT" in z.files:
    sign_M = z["sign_M"]
    p_maxT = z["p_maxT"]
    H, Ln = sign_M.shape
    L.append("sign_M shape=%dx%d" % (H, Ln))
    KNOWN = [(7, 19), (27, 24), (31, 22), (8, 23), (7, 34), (8, 34)]
    sig = []
    for h in range(H):
        for li in range(Ln):
            if float(p_maxT[h, li]) <= 0.05:
                sig.append((float(sign_M[h, li]), h, li, float(p_maxT[h, li])))
    sig.sort(key=lambda t: (-t[0], t[3]))
    L.append("n_sig_recompute=%d" % len(sig))
    L.append("sig events (h,layer) margin p_maxT tag:")
    for m, h, li, p in sig:
        tag = "known" if (h, li) in KNOWN else "novel"
        L.append("  (%d,%d) margin=%.5f p_maxT=%.4f %s" % (h, li, m, p, tag))
    L.append("known 6 detail:")
    for (h, li) in KNOWN:
        L.append("  (%d,%d) margin=%.5f p_maxT=%.4f" % (h, li, sign_M[h, li], p_maxT[h, li]))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("SEAL_REPORT_WRITTEN")
