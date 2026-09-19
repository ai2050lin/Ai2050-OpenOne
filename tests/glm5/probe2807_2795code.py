# -*- coding: utf-8 -*-
"""Find phase2795 script and extract E30 acquisition code."""
import io
import os

GLM5 = r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
OUT = GLM5 + r"\probe2807_2795code.txt"
lines = []

cands = [fn for fn in os.listdir(GLM5) if fn.startswith("phase279") and fn.endswith(".py")]
lines.append("phase279*.py files: %s" % sorted(cands))

target = None
for fn in sorted(cands):
    with io.open(os.path.join(GLM5, fn), "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    if "E30" in src and ("hidden_states" in src or "output_hidden_states" in src):
        target = fn
        lines.append("candidate with E30 + hidden_states: %s (%d chars)" % (fn, len(src)))

if target:
    with io.open(os.path.join(GLM5, target), "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    i = src.find("hidden_states")
    lines.append("=== %s around first hidden_states (char %d) ===" % (target, i))
    lines.append(src[max(0, i - 2500):i + 1500])

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("done")
