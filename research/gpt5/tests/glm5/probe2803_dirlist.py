# -*- coding: utf-8 -*-
import os
root = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\tests\glm5"
out = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\tests\glm5\probe2803_dirlist.txt"
lines = []
for dirpath, dirnames, filenames in os.walk(root):
    rel = os.path.relpath(dirpath, root)
    depth = 0 if rel == "." else rel.count(os.sep) + 1
    if depth <= 3:
        lines.append(f"[D] {rel}")
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = -1
            lines.append(f"      {fn}  ({sz} B)")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("WROTE", out, len(lines))
