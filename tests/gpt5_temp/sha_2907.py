# -*- coding: utf-8 -*-
"""SHA registration probe for Phase 2907 artifacts."""
import hashlib, os, json

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase2907\shape_correction_law"
SCRIPT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2907_shape_correction_law.py"

def sha8(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

out = {}
for fn in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, fn)
    if os.path.isfile(p):
        out[fn] = {"sha8": sha8(p), "bytes": os.path.getsize(p)}
out["SCRIPT phase2907_shape_correction_law.py"] = {"sha8": sha8(SCRIPT), "bytes": os.path.getsize(SCRIPT)}

rep = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\sha_2907.txt"
with open(rep, "w", encoding="utf-8") as f:
    f.write(json.dumps(out, indent=1, ensure_ascii=False))
print("OK", rep)
