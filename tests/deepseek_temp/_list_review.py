# -*- coding: utf-8 -*-
import os, hashlib
ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, r"tests\deepseek_temp\_list_review.txt")
r = []
def add(s): r.append(s)
for root in [os.path.join(ROOT, r"tests\deepseek\_review"),
             os.path.join(ROOT, r"tests\deepseek_temp\_review")]:
    add("== %s ==" % root)
    if not os.path.isdir(root):
        add("  (not exist)"); continue
    for fn in sorted(os.listdir(root)):
        fp = os.path.join(root, fn)
        if os.path.isfile(fp):
            add("  %-46s %8d B  %s" % (fn, os.path.getsize(fp), hashlib.sha256(open(fp,'rb').read()).hexdigest()[:8]))
        else:
            add("  DIR %s" % fn)
    add("")
txt = "\n".join(r)
open(OUT, "w", encoding="utf-8").write(txt)
print("WROTE chars=%d" % len(txt))
