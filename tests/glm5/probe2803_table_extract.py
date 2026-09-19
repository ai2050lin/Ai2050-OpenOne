# -*- coding: utf-8 -*-
import json, os
BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase2803\qwen4_polysemy_census"
RJ   = os.path.join(BASE, "result.json")
OUT  = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2803_table.txt"
with open(RJ, "r", encoding="utf-8") as f:
    r = json.load(f)
lines = []
lines.append("word | senses | allneg | flatness | pr_ratio | default margins")
for rec in r["records"]:
    d = rec["default"]
    dm = ", ".join(f"{k}:{v:+.1f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))
    lines.append(f"{rec['word']} | {'/'.join(rec['senses'])} | {rec['allneg']} | {rec['flatness']:.2f} | {rec['pr_ratio']:.2f} | {dm}")
lines.append("")
lines.append("PR per sense (npz pr matrix row per word):")
import numpy as np
z = np.load(os.path.join(BASE, "census.npz"), allow_pickle=True)
pr = z["pr"]; rho = z["rho"]; sv = z["sv"]
# sense order per record
for rec in r["records"]:
    lines.append(f"{rec['word']}: senses={rec['senses']} pr={[round(x,1) for x in pr[r['records'].index(rec)]]} sv={[round(x,2) for x in sv[r['records'].index(rec)]]}")
# top10 by pr_ratio and bottom5
recs = sorted(r["records"], key=lambda x: -x["pr_ratio"])
lines.append("")
lines.append("TOP pr_ratio: " + ", ".join(f"{x['word']}={x['pr_ratio']:.1f}" for x in recs[:10]))
lines.append("BOTTOM pr_ratio: " + ", ".join(f"{x['word']}={x['pr_ratio']:.1f}" for x in recs[-5:]))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("WROTE", OUT)
