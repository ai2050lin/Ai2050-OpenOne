# -*- coding: utf-8 -*-
"""Extract MODELS dict + check loader availability for cross-model phase."""
import io

GLM5 = r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
OUT = GLM5 + r"\probe2807_models.txt"
lines = []

for fn in ("phase2621_native_behavior_run.py", "phase2620_native_coordinate_contract.py"):
    p = GLM5 + "\\" + fn
    try:
        with io.open(p, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception as e:
        lines.append("%s: ERR %s" % (fn, e))
        continue
    i = src.find("MODELS")
    lines.append("=== %s (MODELS at char %d) ===" % (fn, i))
    if i >= 0:
        lines.append(src[max(0, i - 300):i + 700])
    lines.append("")

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("done")
