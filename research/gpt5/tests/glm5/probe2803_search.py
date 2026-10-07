# -*- coding: utf-8 -*-
import os, json
roots = [
    r"D:\AI2050\Ai2050-OpenOne\research\gpt5",
    r"D:\AI2050\Ai2050-OpenOne\research",
]
seen = set()
hits = []
for root in roots[:1]:
    if root in seen:
        continue
    seen.add(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune heavy dirs
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
        for fn in filenames:
            if ("2803" in fn or "census" in fn or "2802" in fn or "2801" in fn or "polysemy" in fn) and not fn.endswith((".pyc",)):
                p = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = -1
                hits.append(f"{p}  ({sz} B)")
# also check glm5 line locations
for extra in [
    r"D:\AI2050\Ai2050-OpenOne\research\glm5\tests\glm5",
    r"D:\AI2050\Ai2050-OpenOne\tests\glm5",
    r"D:\AI2050\Ai2050-OpenOne\research\gpt5\tests",
]:
    if os.path.isdir(extra):
        hits.append(f"[EXISTS DIR] {extra}")
out = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\tests\glm5\probe2803_search.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(hits) if hits else "NO HITS")
print("WROTE", out, len(hits))
