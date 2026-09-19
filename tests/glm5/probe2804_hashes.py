# -*- coding: utf-8 -*-
import hashlib, os
from pathlib import Path
BASE = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase2804\qwen4_polysemy_k3")
files = [
    r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2804_rdc_polysemy_k3.py",
    str(BASE / "execution.json"),
    str(BASE / "result.json"),
    str(BASE / "k3.npz"),
]
out = []
for f in files:
    p = Path(f)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    out.append("%s  %s  (%d B)" % (h[:16], p.name, p.stat().st_size))
report = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2804_hashes.txt")
report.write_text("\n".join(out), encoding="utf-8")
print("WROTE", report)
