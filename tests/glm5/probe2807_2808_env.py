# -*- coding: utf-8 -*-
"""Probe env for Phase 2807/2808: available models, loaders, GPU memory."""
import io
import os
import re

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2807_2808_env.txt"
ROOT = r"D:\AI2050\Ai2050-OpenOne"
GLM5 = os.path.join(ROOT, "tests", "glm5")

lines = []

lines.append("=== models/hf dirs ===")
mh = os.path.join(ROOT, "models", "hf")
if os.path.isdir(mh):
    for d in sorted(os.listdir(mh)):
        p = os.path.join(mh, d)
        if os.path.isdir(p):
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, dn, fn in os.walk(p) for f in fn)
            lines.append("%s  (%.1f GB)" % (d, size / 1e9))
else:
    lines.append("NO models/hf at " + mh)

lines.append("")
lines.append("=== loader defs in phase2662_symmetric_mapping_contract.py ===")
loader = os.path.join(GLM5, "phase2662_symmetric_mapping_contract.py")
if os.path.exists(loader):
    with io.open(loader, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    m = re.search(r"def load_native.*?(?=\ndef |\nclass |\Z)", src, re.S)
    if m:
        body = m.group(0)
        lines.append("load_native found, %d chars" % len(body))
        lines.append(body[:3000])
    else:
        lines.append("load_native NOT found in " + loader)
    ml = re.findall(r"'([a-z0-9_]+)'\s*:", src)
    lines.append("dict-ish keys: %s" % sorted(set(ml))[:60])
else:
    lines.append("loader file missing: " + loader)

lines.append("")
lines.append("=== glm5 scripts mentioning other models ===")
pat = re.compile(r"(glm4|glm-4|gemma|ds7b|ds-r1|dsr1|deepseek|14b|qwen3-14|qwen3_14|7b)", re.I)
hits = []
for dirpath, dirnames, filenames in os.walk(GLM5):
    dirnames[:] = [d for d in dirnames if d not in ("result", "__pycache__")]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            with io.open(p, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except Exception:
            continue
        if pat.search(txt):
            hits.append((os.path.relpath(p, GLM5), len(txt)))
for h in sorted(hits)[:40]:
    lines.append("%s  (%d B)" % h)

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("done")
