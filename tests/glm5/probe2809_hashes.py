# -*- coding: utf-8 -*-
"""Hash registry for Phase 2809 artifacts."""
import hashlib
import io
import os

GLM5 = r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
BASE = os.path.join(GLM5, "result", "rdc_query_construction_20260913")
OUT = os.path.join(GLM5, "probe2809_hashes.txt")

ITEMS = [
    ("phase2809", os.path.join(GLM5, "phase2809_rdc_nesting_tautology.py")),
    ("phase2809", os.path.join(BASE, "phase2809", "qwen4_nesting_tautology",
                               "execution.json")),
    ("phase2809", os.path.join(BASE, "phase2809", "qwen4_nesting_tautology",
                               "result.json")),
    ("phase2809", os.path.join(GLM5, "run2809.log")),
]

lines = ["=== Phase 2809 hash registry (probe2809_hashes.py) ==="]
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
