# -*- coding: utf-8 -*-
"""Probe: register SHA256 for phase2815 + phase2816 artifacts & scripts.
Writes report to probe_2816_hash.txt (bash stdout unreliable on this machine).
"""
import os, hashlib, json, io

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
SCRIPTS = [
    r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2815_bias_matrix.py",
    r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2816_layers_semantics_position.py",
]
DIRS = {
    "2815": os.path.join(BASE, "phase2815", "bias_matrix"),
    "2816": os.path.join(BASE, "phase2816", "layers_semantics_position"),
}
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_2816_hash.txt"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

buf = io.StringIO()
def w(s=""):
    buf.write(s + "\n")

for tag, d in DIRS.items():
    w(f"=== phase{tag} dir: {d}")
    if not os.path.isdir(d):
        w("  !! MISSING DIR")
        continue
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            w(f"  {fn}  size={sz}  sha256={sha256(p)}")
    w()

w("=== scripts")
for p in SCRIPTS:
    if os.path.isfile(p):
        w(f"  {os.path.basename(p)}  size={os.path.getsize(p)}  sha256={sha256(p)}")
    else:
        w(f"  {p}  !! MISSING")
w()

# Extract key verdict fields from result.json for MEMO writing
for tag, d in DIRS.items():
    w(f"=== phase{tag} result.json key fields")
    rp = os.path.join(d, "result.json")
    if not os.path.isfile(rp):
        w("  !! MISSING result.json")
        continue
    try:
        with open(rp, "r", encoding="utf-8") as f:
            r = json.load(f)
        v = r.get("verdicts", r)
        w(f"  keys_top={sorted(r.keys())[:20]}")
        if isinstance(v, dict):
            for k in sorted(v.keys()):
                val = v[k]
                if isinstance(val, dict):
                    w(f"  {k}:")
                    for k2 in sorted(val.keys()):
                        val2 = val[k2]
                        s = json.dumps(val2, ensure_ascii=False)
                        if len(s) > 300:
                            s = s[:300] + "...(trunc)"
                        w(f"    {k2} = {s}")
                else:
                    s = json.dumps(val, ensure_ascii=False)
                    if len(s) > 300:
                        s = s[:300] + "...(trunc)"
                    w(f"  {k} = {s}")
    except Exception as e:
        w(f"  !! parse error: {e}")
    w()

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("WROTE", OUT)
