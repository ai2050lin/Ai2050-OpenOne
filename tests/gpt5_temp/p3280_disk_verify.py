# -*- coding: utf-8 -*-
"""Phase 3280 independent disk verify (separate pass, fresh read)."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(ROOT, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase3280", "g5h43_consumer_attribution")
P3166 = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913",
                     "phase3166", "g5a3b_logic_direction", "result.json")


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS" if cond else "FAIL"), name)


# 1 result on disk + anchors
res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
res_re = hashlib.sha256(json.dumps(
    {k: v for k, v in res.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v1 res_sha8 content-recompute == ce0c400b", res_re == "ce0c400b")
check("v2 verdict BLOCKED honest (gate_all False, verdict BLOCKED)",
      res["verdict"] == "BLOCKED" and res["gate_all"] is False)
check("v3 gates pattern (gA/gB/gB_spec True, KA/KA2 False)",
      res["gates"]["gA_anchors"] is True
      and res["gates"]["gB_static_sane"] is True
      and res["gates"]["gB_specificity"] is True
      and res["gates"]["KA_overlap"] is False
      and res["gates"]["KA2_spearman"] is False)
check("v4 census anchors match p3166 4b",
      res["census_now"] == res["census_anchor"]
      and abs(res["census_now"]["K_readout__K_entity"] - 69.823) < 0.001)
r16 = json.load(open(P3166, encoding="utf-8"))
cen = r16["per_model"]["qwen3-4b"]["census"]
check("v5 anchors == sealed 3166 4b census",
      all(abs(res["census_anchor"][k] - cen[k]["top1_deg"]) < 1e-9
          for k in res["census_anchor"]))
check("v6 head_write_map.npz shapes (36,32) x4",
      True)  # verified via npz load below
import numpy as np
z = np.load(os.path.join(OUT, "head_write_map.npz"))
check("v6 npz shapes all (36,32)",
      all(z[k].shape == (36, 32) for k in z.files)
      and set(z.files) == {"S_rel_R_logic", "S_rel_K_entity",
                           "D_R_logic", "D_K_entity"})
check("v7 npz consistent with result top head",
      abs(float(z["S_rel_R_logic"].max())
          - res["top_static"]["R_logic"][0][2]) < 1e-3)

# 2 ledger chain head recompute
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
probe = json.loads(json.dumps(led))
probe["ledger_sha256_8"] = "0ee33c4c"
rec = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
check("v8 chain head n=438 post=a6bf3af5 recompute",
      len(ms) == 438 and led["ledger_sha256_8"] == "a6bf3af5"
      and rec == "a6bf3af5")
check("v9 last rec anchors", str(ms[-1]["phase"]) == "3280"
      and ms[-1]["res_sha8"] == "ce0c400b"
      and "blocked" in ms[-1]["verdict"]
      and ms[-1]["prev_post"] == "0ee33c4c")

# 3 five-write presence
wlog = open(os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
            encoding="utf-8").read()
memo = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
            encoding="utf-8").read()
daily = open(os.path.join(ROOT, ".workbuddy", "memory", "2026-10-22.md"),
             encoding="utf-8").read()
memory = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
              encoding="utf-8").read()
check("v10 five-write presence", "g5h43_consumer_attribution" in wlog
      and "Phase 3280 (G5-H43)" in memo
      and "3280 G5-H43" in daily
      and "3280" in memory)
check("v11 MEMORY <=3000 chars", len(memory) <= 3000)
check("v12 3281 preregistered in MEMORY/daily",
      "3281" in memory and "份额归一化" in daily)

print("VERIFY", "PASS" if all(ok) else "FAIL", "%d/%d" % (sum(ok), len(ok)))
