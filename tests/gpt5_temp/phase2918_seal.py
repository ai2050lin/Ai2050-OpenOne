# -*- coding: utf-8 -*-
"""Phase 2918 seal probe: verify artifacts on real disk + SHA256-8."""
import hashlib
import json
import os
import time

BASE = r"D:\AI2050\Ai2050-OpenOne"
ART = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase2918", "event_anatomy")
SCRIPT = os.path.join(BASE, "tests", "glm5",
                      "phase2918_event_anatomy.py")
OUT = os.path.join(BASE, "tests", "gpt5_temp",
                   "phase2918_seal_report.txt")


def sha8(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


L = []
for fn in ("execution.json", "result.json", "event_anatomy.npz"):
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
L.append("machine_now=%s" % time.strftime("%Y-%m-%dT%H:%M:%S"))
res = json.load(open(os.path.join(ART, "result.json"),
                     encoding="utf-8"))
L.append("verdict=%s" % res.get("final_verdict"))
L.append("runtime_s=%s" % res.get("runtime_s"))
P2 = res.get("P2") or {}
L.append("n_linked=%s rho_known_mean=%s rho_early_mean=%s"
         % (P2.get("n_linked"), P2.get("rho_known_mean"),
            P2.get("rho_early_mean")))
L.append("graph_edges=%s" % (P2.get("graph") or {}).get("n_edges"))
L.append("npz_keys=%s"
         % ",".join(sorted(
             __import__("numpy").load(
                 os.path.join(ART, "event_anatomy.npz"),
                 allow_pickle=True).files)))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("SEAL_OK")
