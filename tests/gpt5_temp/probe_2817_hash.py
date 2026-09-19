# -*- coding: utf-8 -*-
"""Probe: register SHA256 for phase2817 artifacts & script."""
import os, hashlib, json, io, datetime

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
D = os.path.join(BASE, "phase2817", "ov_subspace_natural_position")
SCRIPT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2817_ov_subspace_natural_position.py"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_2817_hash.txt"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

buf = io.StringIO()
w = lambda s="": buf.write(s + "\n")

w("now=" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
w("=== phase2817 dir: " + D)
for fn in sorted(os.listdir(D)):
    p = os.path.join(D, fn)
    if os.path.isfile(p):
        w("  %s  size=%d  sha256=%s" % (fn, os.path.getsize(p), sha256(p)))
w()
w("=== script")
w("  phase2817_ov_subspace_natural_position.py  size=%d  sha256=%s"
  % (os.path.getsize(SCRIPT), sha256(SCRIPT)))
w()
w("=== result.json verdict")
r = json.load(open(os.path.join(D, "result.json"), encoding="utf-8"))
for k in sorted(r["verdict"].keys()):
    v = json.dumps(r["verdict"][k], ensure_ascii=False)
    if len(v) > 240:
        v = v[:240] + "...(trunc)"
    w("  %s = %s" % (k, v))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("WROTE", OUT)
