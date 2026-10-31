# -*- coding: utf-8 -*-
"""Phase 3281 independent disk verify (separate pass, fresh read)."""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(ROOT, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase3281", "g5h44_consumer_attribution_norm")
P3166 = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913",
                     "phase3166", "g5a3b_logic_direction", "result.json")
P3280 = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913",
                     "phase3280", "g5h43_consumer_attribution", "result.json")

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS" if cond else "FAIL"), name)


res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
res_re = hashlib.sha256(json.dumps(
    {k: v for k, v in res.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v1 res_sha8 content-recompute == a8f1e066", res_re == "a8f1e066")
check("v2 verdict registered (gate_all True, 5/5 gates)",
      res["verdict"] == "consumer_attribution_norm_registered"
      and res["gate_all"] is True
      and all(res["gates"][k] for k in
              ("gA_anchors", "gB_static_sane", "gB_specificity",
               "KA_overlap", "KA2_spearman")))
check("v3 K gates values (overlap 15/15, spearman 0.340/0.362)",
      res["gates_k"]["R_logic"]["overlap"] == 15
      and res["gates_k"]["K_entity"]["overlap"] == 15
      and abs(res["gates_k"]["R_logic"]["spearman"] - 0.3398) < 1e-9
      and abs(res["gates_k"]["K_entity"]["spearman"] - 0.3620) < 1e-9)
check("v4 census anchors match p3166 4b",
      res["census_now"] == res["census_anchor"]
      and abs(res["census_now"]["K_readout__K_entity"] - 69.823) < 0.001)
r16 = json.load(open(P3166, encoding="utf-8"))
cen = r16["per_model"]["qwen3-4b"]["census"]
check("v5 anchors == sealed 3166 4b census",
      all(abs(res["census_anchor"][k] - cen[k]["top1_deg"]) < 1e-9
          for k in res["census_anchor"]))
check("v6 static arm identical to 3280 (S_rel top1 & null p95)",
      res["top_static"]["R_logic"][0][:2]
      == json.load(open(P3280, encoding="utf-8"))["top_static"]["R_logic"][0][:2]
      and abs(res["null"]["p95"]["R_logic"]
              - json.load(open(P3280, encoding="utf-8"))["null"]["p95"]["R_logic"]) < 1e-9)
z = np.load(os.path.join(OUT, "head_write_map.npz"))
check("v7 npz shapes all (36,32) x7 keys",
      all(z[k].shape == (36, 32) for k in z.files)
      and set(z.files) == {"S_rel_R_logic", "S_rel_K_entity", "Dnorm_R_logic",
                           "Dnorm_K_entity", "Dabs_R_logic", "Dabs_K_entity",
                           "meanC"})
check("v8 npz Dnorm consistent with result top share head",
      abs(float(z["Dnorm_K_entity"].max())
          - res["KB_report"]["K_entity"]["top8"][0][2]) < 1e-4)
check("v9 confound arm values (D_abs~C 0.97x vs S_rel~C 0.0xx)",
      res["arm_c_confound"]["R_logic"]["spearman_Dabs_meanC"] > 0.95
      and res["arm_c_confound"]["K_entity"]["spearman_Dabs_meanC"] > 0.95
      and res["arm_c_confound"]["R_logic"]["spearman_Srel_meanC"] < 0.15)
check("v10 continuity: 3280-style recompute == #432 verbatim",
      res["arm_c_confound"]["R_logic"]["abs_overlap_3280style"] == 9
      and res["arm_c_confound"]["K_entity"]["abs_overlap_3280style"] == 11
      and abs(res["arm_c_confound"]["R_logic"]["abs_spearman_3280style"] - 0.1372) < 5e-4
      and abs(res["arm_c_confound"]["K_entity"]["abs_spearman_3280style"] - 0.1787) < 5e-4)

# ledger chain head recompute (full measurements, field=prev_post)
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
probe = json.loads(json.dumps(led))
probe["ledger_sha256_8"] = "a6bf3af5"
rec = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
check("v11 chain head n=439 post=8299f955 recompute",
      len(ms) == 439 and led["ledger_sha256_8"] == "8299f955"
      and rec == "8299f955")
check("v12 last rec anchors", str(ms[-1]["phase"]) == "3281"
      and ms[-1]["res_sha8"] == "a8f1e066"
      and ms[-1]["verdict"] == "consumer_attribution_norm_registered"
      and ms[-1]["prev_post"] == "a6bf3af5")

# five-write presence
wlog = open(os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
            encoding="utf-8").read()
wlogm = open(os.path.join(ROOT, "tests", "gpt5_temp", "WORKLOG_gpt5_mirror.md"),
             encoding="utf-8").read()
memo = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
            encoding="utf-8").read()
daily = open(os.path.join(ROOT, ".workbuddy", "memory", "2026-10-31.md"),
             encoding="utf-8").read()
memory = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
              encoding="utf-8").read()
check("v13 five-write presence", "g5h44_consumer_attribution_norm" in wlog
      and "g5h44_consumer_attribution_norm" in wlogm
      and "Phase 3281 (G5-H44)" in memo
      and "3281 G5-H44" in daily
      and "3281" in memory)

# design freeze
design = json.load(open(os.path.join(OUT, "execution.json"), encoding="utf-8"))
CANON = ('phase', 'name', 'prereg', 'arms', 'subspaces', 'gates', 'limits',
         'source_sha8', 'smoke')
dh = hashlib.sha256(json.dumps({k: design[k] for k in CANON}, sort_keys=True,
                               ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v14 design freeze sha == 7807b62c == result.design_sha8",
      dh == "7807b62c" and res["design_sha8"] == dh)

print("VERIFY:", "ALL PASS" if all(ok) else "FAIL", "(%d/%d)" % (sum(ok), len(ok)))
raise SystemExit(0 if all(ok) else 1)
