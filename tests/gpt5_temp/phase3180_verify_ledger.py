# -*- coding: utf-8 -*-
"""Phase 3180 独立复核 + ledger append #332 + 自检（一次完成，独立实现）"""
import json, io, hashlib, re, sys, copy

BASE = r"D:/AI2050/Ai2050-OpenOne"
OUT = BASE + r"/tests/glm5/result/rdc_query_construction_20260913/phase3180/g5b5_ftr15_heldout"
PREV = BASE + r"/tests/glm5/result/rdc_query_construction_20260913/phase3179/g5b4_ftr23_registration"
fails = []

def chk(n, c, d=""):
    print(("PASS" if c else "FAIL"), n, d)
    if not c: fails.append(n)

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

# 1 sha 断言
chk("reg17 sha8 f3ad070f", sha8(open(OUT + "/atlas_registry_v1_7.json", "rb").read()) == "f3ad070f")
chk("html sha8 1dc09043", sha8(open(OUT + "/atlas_v1_8.html", "rb").read()) == "1dc09043")
res = json.load(open(OUT + "/final_result.json", encoding="utf-8"))
chk("final seal ce9f80d4", res["seal_sha8"] == "ce9f80d4")

# 2 registry diff：仅 FTR-15 变更 + upgrade_log +1 + version/supersedes
v16 = json.load(open(PREV + "/atlas_registry_v1_6.json", encoding="utf-8"))
v17 = json.load(open(OUT + "/atlas_registry_v1_7.json", encoding="utf-8"))
same = [f["id"] for f in v16["features"]
        if f["id"] != "FTR-15" and
        json.dumps(f, sort_keys=True, ensure_ascii=False) ==
        json.dumps([x for x in v17["features"] if x["id"] == f["id"]][0], sort_keys=True, ensure_ascii=False)]
chk("22 legacy features byte-preserved", len(same) == 22, "%d/22" % len(same))
f15o = [f for f in v16["features"] if f["id"] == "FTR-15"][0]
f15n = [f for f in v17["features"] if f["id"] == "FTR-15"][0]
chk("FTR-15 E2", f15n["evidence_level"] == "E2_predictive")
chk("FTR-15 anchors 2->3", len(f15o["anchors"]) == 2 and len(f15n["anchors"]) == 3)
chk("FTR-15 held anchor src p3180", f15n["anchors"][-1]["src"] == "p3180")
chk("upgrade_log 5->6", len(v16["upgrade_log"]) == 5 and len(v17["upgrade_log"]) == 6 and
    v17["upgrade_log"][-1]["node"] == "FTR-15")
chk("failures untouched 16", len(v17["failures"]) == 16 and
    json.dumps(v16["failures"], sort_keys=True, ensure_ascii=False) ==
    json.dumps(v17["failures"], sort_keys=True, ensure_ascii=False))
chk("version 1.7 supersedes 3179", v17["version"] == "1.7" and "Phase 3179" in v17["supersedes"])
for k in ("created", "phase", "name", "design_sha8", "principles", "schema",
          "evidence_level_taxonomy", "models_mainline", "models_pending", "provenance"):
    chk("untouched " + k,
        json.dumps(v16.get(k), sort_keys=True, ensure_ascii=False) ==
        json.dumps(v17.get(k), sort_keys=True, ensure_ascii=False))

# 3 数值断言 vs 3158 封存 ratio（gates/arm 读数取自主 result.json）
armres = json.load(open(OUT + "/result.json", encoding="utf-8"))
import numpy as np
for m, exp in [("qwen3-4b", 1.0283477199188433), ("qwen3-14b", 1.3316768638966026),
               ("glm4", 1.2472595295618603)]:
    a = armres["per_model"][m]["ratio_full_recompute"]
    chk("ratio %s drift<=1e-9" % m, abs(a - exp) / exp <= 1e-9, "%.9f" % abs(a - exp))
chk("gates G1 G2 true", armres["gates"]["G1_device"] and armres["gates"]["G2_heldout"])
chk("arm per_model matches final", all(
    abs(res["arm"]["per_model"][m]["ratio_full_recompute"] - armres["per_model"][m]["ratio_full_recompute"]) == 0
    for m in ("qwen3-4b", "qwen3-14b", "glm4")))

# 4 ledger append #332（append-only + 备份）
LP = BASE + r"/research/gpt5/atlas/atlas_ledger.json"
bakp = BASE + r"/tests/gpt5_temp/atlas_ledger_backup_pre_phase3180.json"
import shutil
raw = open(LP, "rb").read()
pre8 = sha8(raw)
d = json.loads(raw.decode("utf-8"))
if d["measurements"][-1]["phase"] == 3180:
    chk("ledger already appended #332", True, "idempotent skip")
else:
    assert d["measurements"][-1]["phase"] == 3179 and len(d["measurements"]) == 331
    shutil.copyfile(LP, bakp)
    entry = {
        "phase": 3180, "name": "g5b5_ftr15_heldout", "line": "G", "date": "2026-10-10",
        "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
        "verdict": "g5b5_ftr15_heldout|ftr15_upgraded_E2_predictive|features_23|upgrades_6|registry_v17|html_fields_704_ok",
        "evidence_level": "E2_predictive", "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
        "prereg_id": "G5-B5", "superseded_by": None,
        "detail": ("G5-B5 FTR-15 held_out entity-LOO arm (zero GPU; sealed 3158 collect BUD "
                   "c4d9b9fb/7f53f3a0/2ea72cfe + results 90e734fd/1f532016/c49614fc + summary 7ed85b36, "
                   "sha-asserted). 16 anchors verified 1-per-entity via 3157 collect ei labels. "
                   "G1 device: full ratio recomputed from BUD drift 0.00e+00 x3 models "
                   "(1.028348/1.331677/1.247260 vs sealed). G2 held_out: all 16 entity-LOO folds "
                   "same quotient class (<3.0) and max |log(ratio_train/ratio_full)| "
                   "0.0300/0.0278/0.0182 < 0.25 bar x3 models (gate+bar frozen pre-observation). "
                   "G3 directional: held-anchor null>rand medians 10/16, 16/16, 15/16 (report only). "
                   "FTR-15 E1_repeatable -> E2_predictive (held_out anchor appended; thin-neighborhood "
                   "local equivalence now cross_model+held_out; scope limit 'not a global quotient "
                   "structure' retained, F1 quotient rejection unchanged). Registry v1.6 -> v1.7 "
                   "append-only (22 legacy features byte-for-byte, failures untouched, upgrade_log #6; "
                   "f3ad070f); atlas_v1_8.html 704 fields 0/0/0/0 (1dc09043). Design fix logged: "
                   "smoke flag moved out of design dict (mode-level not design-level) before FULL "
                   "observation; execution.json 8e38ff2f. res e97867f4 / seal ce9f80d4. "
                   "ledger_sha256_8 NOT recomputed (stale since Phase 8); pre-append file sha8 " + pre8 + "."),
        "created": "2026-10-10T05:20:00"}
    d["measurements"].append(entry)
    open(LP, "wb").write((json.dumps(d, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    d2 = json.loads(open(LP, "rb").read().decode("utf-8"))
    bak = json.load(open(bakp, encoding="utf-8"))
    assert d2["measurements"][:331] == bak["measurements"]
    assert len(d2["measurements"]) == 332 and d2["measurements"][-1]["phase"] == 3180
    chk("ledger #332 appended + prefix equivalent", True,
        "post sha8 " + sha8(open(LP, "rb").read()))

print("=== 3180 INDEPENDENT VERIFY + LEDGER:", "ALL PASS" if not fails else "FAIL " + ",".join(fails), "===")
sys.exit(1 if fails else 0)
