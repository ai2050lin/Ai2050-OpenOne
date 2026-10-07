# -*- coding: utf-8 -*-
"""读 ledger 顶层结构 + 最后条目 + MEMORY.md 大小。"""
import json, io, os

L = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json"
d = json.load(io.open(L, encoding="utf-8"))
out = ["top_keys: %s" % list(d.keys()) if isinstance(d, dict) else "list n=%d" % len(d)]
if isinstance(d, dict):
    for k, v in d.items():
        if isinstance(v, list):
            out.append("list key=%r n=%d" % (k, len(v)))
            if v:
                out.append("LAST[%s]: %s" % (k, json.dumps(v[-1], ensure_ascii=False)[:800]))
        elif isinstance(v, dict):
            out.append("dict key=%r n=%d" % (k, len(v)))
            ks = list(v.keys())
            out.append("  last3 keys: %s" % ks[-3:])
            lk = ks[-1]
            out.append("LAST[%s][%s]: %s" % (k, lk, json.dumps(v[lk], ensure_ascii=False)[:800]))
        else:
            out.append("scalar %r = %r" % (k, v))
out.append("MEMORY_MD_SIZE: %d bytes" % os.path.getsize(
    r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md"))
io.open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3153_ledger_tail.txt",
        "w", encoding="utf-8").write("\n".join(out))
print("ok")
