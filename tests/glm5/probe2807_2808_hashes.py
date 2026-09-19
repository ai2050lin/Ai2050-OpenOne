# -*- coding: utf-8 -*-
"""Hash registry for Phase 2807+2808 artifacts."""
import hashlib
import io
import os

GLM5 = r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
BASE = os.path.join(GLM5, "result", "rdc_query_construction_20260913")
OUT = os.path.join(GLM5, "probe2807_2808_hashes.txt")

ITEMS = [
    ("phase2807", os.path.join(GLM5, "phase2807_rdc_heldout.py")),
    ("phase2807", os.path.join(BASE, "phase2807", "qwen4_heldout",
                               "execution.json")),
    ("phase2807", os.path.join(BASE, "phase2807", "qwen4_heldout",
                               "result.json")),
    ("phase2807", os.path.join(BASE, "phase2807", "qwen4_heldout",
                               "heldout.npz")),
    ("phase2807", os.path.join(GLM5, "run2807.log")),
    ("phase2808", os.path.join(GLM5, "phase2808_rdc_crossmodel.py")),
    ("phase2808", os.path.join(BASE, "phase2808", "crossmodel_hierarchy",
                               "execution.json")),
    ("phase2808", os.path.join(BASE, "phase2808", "crossmodel_hierarchy",
                               "result.json")),
    ("phase2808", os.path.join(GLM5, "run2808.log")),
]

lines = ["=== Phase 2807+2808 hash registry "
         "(probe2807_2808_hashes.py) ==="]
for tag, p in ITEMS:
    if not os.path.exists(p):
        lines.append("%s | MISSING: %s" % (tag, p))
        continue
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    lines.append("%s | %s | SHA256=%s | %d bytes"
                 % (tag, p, h.hexdigest(), os.path.getsize(p)))

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("done")
