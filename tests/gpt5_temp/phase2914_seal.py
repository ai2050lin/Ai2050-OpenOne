# -*- coding: utf-8 -*-
# phase2914 seal probe: verify artifacts on disk + SHA256-8
import hashlib, json, os

D = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
     r"\rdc_query_construction_20260913\phase2914"
     r"\head_identity_replication")
SCRIPT = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5"
          r"\phase2914_head_identity_replication.py")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2914_seal.txt"
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
w("execution script_sha256_8: " + ex["script_sha256_8"])

with open(os.path.join(D, "result.json"), "r", encoding="utf-8") as f:
    r = json.load(f)
w("final_verdict: " + r["final_verdict"])
w("runtime_s: " + str(r["runtime_s"]))
w("anchors.ok: " + str(r["anchors"]["ok"]))
w("P1: " + json.dumps(r["P1"], ensure_ascii=False))
w("P2 nonempty fields: flips_equal=" + str(r["P2"]["flips_equal"])
  + " p_flip_min=" + str(r["P2"]["p_flip_min"]))
w("P3: " + json.dumps(r["P3"], ensure_ascii=False))
w("P4 rankings t12: " + json.dumps(r["P4"]["rankings"]["t12"]))
w("P4 rankings PR: " + json.dumps(r["P4"]["rankings"]["PR"]))
w("P4 rankings zf_gain: " + json.dumps(r["P4"]["rankings"]["zf_gain"]))
w("P4 rankings zf_cos: " + json.dumps(r["P4"]["rankings"]["zf_cos"]))
w("a3_layers worst:")
for l in r["anchors"]["a3_layers"]:
    w("  L%d t12 %.6f (st %.5f) PR %.2f (st %.5f) gain %.4f (st %.5f) cos %.5f (st %.5f) ok=%s"
      % (l["layer"], l["t12"], l["t12_stored"], l["PR"], l["PR_stored"],
         l["zf_gain"], l["zf_gain_stored"], l["zf_cos"], l["zf_cos_stored"], l["ok"]))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE", OUT, flush=True)
