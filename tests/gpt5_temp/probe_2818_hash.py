# -*- coding: utf-8 -*-
"""Probe: register SHA256 for phase2818 artifacts & script."""
import os, hashlib, json, io, datetime

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
D = os.path.join(BASE, "phase2818", "write_causality")
SCRIPT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2818_write_causality.py"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_2818_hash.txt"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

buf = io.StringIO()
w = lambda s="": buf.write(s + "\n")

w("now=" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
w("=== phase2818 dir: " + D)
for fn in sorted(os.listdir(D)):
    p = os.path.join(D, fn)
    if os.path.isfile(p):
        w("  %s  size=%d  sha256=%s" % (fn, os.path.getsize(p), sha256(p)))
w()
w("=== script")
w("  phase2818_write_causality.py  size=%d  sha256=%s"
  % (os.path.getsize(SCRIPT), sha256(SCRIPT)))
w()
w("=== result.json key verdict fields")
r = json.load(open(os.path.join(D, "result.json"), encoding="utf-8"))
v = r["verdict"]
for k in ["median_actual_cos", "rand_head_cos_q95", "ov_write_actual",
          "d_mlp", "rand_neuron_q95", "mlp_write_causal",
          "d_ov", "rand_head_q95", "ov_write_causal",
          "d_mlp_per_word", "d_ov_per_word", "top8_intact",
          "max_logit_shift_mlp", "pass_layers"]:
    w("  %s = %s" % (k, json.dumps(v[k], ensure_ascii=False)))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("WROTE", OUT)
