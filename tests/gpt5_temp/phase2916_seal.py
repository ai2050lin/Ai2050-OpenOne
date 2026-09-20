# -*- coding: utf-8 -*-
# phase2916 seal probe: verify artifacts on disk + SHA256-8
import hashlib, json, os

D = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
     r"\rdc_query_construction_20260913\phase2916"
     r"\early_carrier_selection")
SCRIPT = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
          r"\phase2916_early_carrier_selection.py")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2916_seal.txt"
out = []
def w(s=""):
    out.append(str(s))

def sha8(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:8]

w("== dir ==")
for f in sorted(os.listdir(D)):
    w(os.path.join(D, f) + "  size=" + str(os.path.getsize(os.path.join(D, f))))
w("")
w("== SHAs ==")
for f in sorted(os.listdir(D)):
    w(f + " : " + sha8(os.path.join(D, f)))
w("script : " + sha8(SCRIPT))
w("")
with open(os.path.join(D, "execution.json"), "r", encoding="utf-8") as f:
    ex = json.load(f)
w("execution created: " + ex["created"])
with open(os.path.join(D, "result.json"), "r", encoding="utf-8") as f:
    r = json.load(f)
w("final_verdict: " + r["final_verdict"])
w("runtime_s: " + str(r["runtime_s"]))
w("anchors: " + json.dumps(r["anchors"], ensure_ascii=False))
w("P1 V1_replicate: " + json.dumps(r["P1"]["V1_replicate"]))
w("P2 loo highlights:")
for k in ("V1_h7", "V1_h8", "V4_h7", "V4_h8", "V4_h27", "V2_h27", "V2_h31"):
    e = r["P2"][k]
    w("  %s full=%.5f worst_layer=L%d (delta %.5f) sign_peak=L%d"
      % (k, e["full_margin"], e["loo_worst_layer"],
         e["loo_delta"][e["loo_worst_layer"]], e["sign_peak_layer"]))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE", OUT, flush=True)
