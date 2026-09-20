# -*- coding: utf-8 -*-
"""Phase 2919 seal probe: verify artifacts + SHA256-8."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
ART = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase2919", "multiaxis_direction_families")
SCRIPT = os.path.join(BASE, "tests", "glm5",
                      "phase2919_multiaxis_direction_families.py")
OUT = os.path.join(BASE, "tests", "gpt5_temp",
                   "phase2919_seal_report.txt")


def sha8(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


L = []
for fn in ("execution.json", "result.json",
           "multiaxis_families.npz"):
    p = os.path.join(ART, fn)
    if os.path.exists(p):
        L.append("%s | size=%d | sha256-8=%s"
                 % (fn, os.path.getsize(p), sha8(p)))
    else:
        L.append("%s | MISSING" % fn)
L.append("script sha256-8=%s" % sha8(SCRIPT))
ex = json.load(open(os.path.join(ART, "execution.json"),
                    encoding="utf-8"))
L.append("execution.created=%s" % ex.get("created"))
res = json.load(open(os.path.join(ART, "result.json"),
                     encoding="utf-8"))
L.append("verdict=%s" % res.get("final_verdict"))
L.append("runtime_s=%s" % res.get("runtime_s"))
L.append("anchors=%s" % res.get("anchors"))
L.append("P1=%s" % (res.get("P1") or {}).get("per_axis_best"))
P2 = res.get("P2") or {}
L.append("P2 global_max=%s argmax=%s@L%s non_collinear=%s"
         % (P2.get("global_max"), P2.get("argmax_pair"),
            P2.get("argmax_layer"), P2.get("non_collinear")))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("SEAL_OK")
