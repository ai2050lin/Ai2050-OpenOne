# -*- coding: utf-8 -*-
"""Probe: register SHA256 for phase2819 artifacts & script."""
import os, hashlib, json, io, datetime

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
D = os.path.join(BASE, "phase2819", "knowledge_edit_locus")
SCRIPT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2819_knowledge_edit_locus.py"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_2819_hash.txt"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

buf = io.StringIO()
w = lambda s="": buf.write(s + "\n")

w("now=" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
w("=== phase2819 dir: " + D)
for fn in sorted(os.listdir(D)):
    p = os.path.join(D, fn)
    if os.path.isfile(p):
        w("  %s  size=%d  sha256=%s" % (fn, os.path.getsize(p), sha256(p)))
w()
w("=== script")
w("  phase2819_knowledge_edit_locus.py  size=%d  sha256=%s"
  % (os.path.getsize(SCRIPT), sha256(SCRIPT)))
w()
r = json.load(open(os.path.join(D, "result.json"), encoding="utf-8"))
v = r["verdict"]
w("=== key fields")
w("K1 mlp_scan = %s" % json.dumps(v["K1"]["mlp_scan"]))
w("K1 attn_scan = %s" % json.dumps(v["K1"]["attn_scan"]))
w("K1 locus = %s (mlp %s / attn %s)" % (
    v["K1"]["knowledge_layer_locus"],
    v["K1"]["knowledge_layer_locus_mlp"],
    v["K1"]["knowledge_layer_locus_attn"]))
w("K2 mlp_pass_n=%d ov_pass_n=%d present=%s"
  % (v["K2"]["mlp_pass_n"], v["K2"]["ov_pass_n"],
     v["K2"]["color_params_present"]))
w("K2 late MLP rows (L>=24): %s" % json.dumps(
    [r for r in v["K2"]["mlp_rows"] if r["layer"] >= 24], ensure_ascii=False))
w("K2 late OV pass rows (L>=30): %s" % json.dumps(
    [r for r in v["K2"]["ov_rows"] if r["layer"] >= 30
     and r["best_val"] >= 0.30], ensure_ascii=False))
w("K3 sites = %s" % json.dumps(v["K3"]["sites"]))
w("K3 rand_site_q95 = %s best = %s site_causal = %s"
  % (v["K3"]["rand_site_q95"], json.dumps(v["K3"]["best_site"]),
     v["K3"]["site_causal"]))
w("K3 emb_apple_delta = %s rand = %s"
  % (v["K3"]["emb_apple_delta"], v["K3"]["emb_apple_rand_delta"]))
w("R sites = %s lens = %s" % (json.dumps(v["R"]["sites"]),
                              json.dumps(v["R"]["logit_lens"])))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("WROTE", OUT)
