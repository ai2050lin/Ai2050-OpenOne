# -*- coding: utf-8 -*-
"""Independent disk verify for Phase 3274 (verify-only; fix verify on FAIL)."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
RBASE = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913")
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply")
PAYLOAD = {
    "phase": "A-gate",
    "name": "rdc_a_gate_closure_q01_q02_q08_q09_q12",
    "verdict": ("k1_fired_at_readout__kspecific__"
                "k2_k3_not_measurable__A_gate_closed"),
    "evidence_level": "bit_anchored",
    "note": "Q08=甲 seal 后登记；属治理记录非科学测量",
}
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


raw = open(LED, "rb").read()
led = json.loads(raw.decode("utf-8"))
ms = led["measurements"]

# 1 head chain (two levels)
c2 = json.loads(raw.decode("utf-8"))
c2["ledger_sha256_8"] = "498a80f6"
p2 = hashlib.sha256(json.dumps(c2, ensure_ascii=False, indent=1)
                    .encode("utf-8")).hexdigest()[:8]
check("v1_head_chain n=432 field=498a80f6 -> 98c772a8",
      len(ms) == 432 and led["ledger_sha256_8"] == "98c772a8" and p2 == "98c772a8")
# precise prev-level check: rebuild full state minus last entry
led431 = json.loads(raw.decode("utf-8"))
led431["measurements"] = led431["measurements"][:-1]
led431["ledger_sha256_8"] = "3f210467"
p1 = hashlib.sha256(json.dumps(led431, ensure_ascii=False, indent=1)
                    .encode("utf-8")).hexdigest()[:8]
check("v1b_prev_chain_state -> 498a80f6", p1 == "498a80f6")
check("v1_roundtrip", json.dumps(led, ensure_ascii=False,
                                 indent=1).encode("utf-8") == raw)

# 2 op4 entry verbatim at index -2; receipt at -1
op4 = ms[-2]
check("v2_op4_payload_verbatim",
      all(op4[k] == PAYLOAD[k] for k in PAYLOAD))
check("v2_op4_provenance",
      op4.get("lcorr_apply", {}).get("applied_by_phase") == 3274
      and op4["lcorr_apply"]["ledger_post_before"] == "20ecd6c2"
      and "498a80f6" not in op4["lcorr_apply"]["ledger_post_before"])
check("v3_receipt_entry",
      ms[-1]["phase"] == 3274 and ms[-1]["verdict"] == "lcorr_op4_applied"
      and ms[-1]["prev_post"] == "498a80f6")

# 3 landed files
g14 = json.load(open(os.path.join(OUT, "gap_ledger_v1_14.json"),
                     encoding="utf-8"))
r37 = json.load(open(os.path.join(OUT, "atlas_registry_v1_37.json"),
                     encoding="utf-8"))
g13 = json.load(open(os.path.join(RBASE, "phase3265", "g5h28_appx7_landing",
                                  "gap_ledger_v1_13.json"), encoding="utf-8"))
r36 = json.load(open(os.path.join(RBASE, "phase3265", "g5h28_appx7_landing",
                                  "atlas_registry_v1_36.json"),
                     encoding="utf-8"))
check("v4_gap_v114", g14["version"] == "1.14" and len(g14["appendix"]) == 8
      and g14["appendix"][-1]["id"] == "APPX-4/5"
      and g14["gaps"] == g13["gaps"])
check("v5_registry_v137", r37["version"] == "1.37"
      and len(r37["upgrade_log"]) == 16 and "updated" not in r37
      and r37["features"] == r36["features"]
      and r37["failures"] == r36["failures"])
check("v6_file_sha8",
      hashlib.sha256(open(os.path.join(OUT, "gap_ledger_v1_14.json"),
                          "rb").read()).hexdigest()[:8] == "c6d97133"
      and hashlib.sha256(open(os.path.join(OUT, "atlas_registry_v1_37.json"),
                              "rb").read()).hexdigest()[:8] == "bfb7d7e5")

# 4 sealed bytes untouched
def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


check("v7_sealed_untouched",
      s8(os.path.join(BASE, "research", "deepseek", "atlas",
                      "ledger_corrections_v1.json")) == "beb775bd"
      and s8(os.path.join(RBASE, "phase3260", "g5h23_lcorr_rederivation",
                          "lcorr_rederivation_v2.json")) == "ef72c733"
      and s8(os.path.join(BASE, "research", "deepseek", "atlas",
                          "a_gate_closure_v1.json")) == "24c60160"
      and s8(os.path.join(RBASE, "phase3265", "g5h28_appx7_landing",
                          "gap_ledger_v1_13.json")) == "814ce808"
      and s8(os.path.join(RBASE, "phase3265", "g5h28_appx7_landing",
                          "atlas_registry_v1_36.json")) == "78d37211")

# 5 revert bitwise: ledger n=430 state (drop receipt + op4) -> .bak_3274 bytes
bak = open(LED + ".bak_3274", "rb").read()
rev = json.loads(raw.decode("utf-8"))
rev["measurements"] = rev["measurements"][:-2]
rev["ledger_sha256_8"] = "20ecd6c2"
check("v8_revert_ledger_bitwise_to_bak3274",
      json.dumps(rev, ensure_ascii=False, indent=1).encode("utf-8") == bak)
# gap/registry revert rebuilds
rg = json.loads(json.dumps(g14))
rg["version"] = "1.13"
rg["appendix"] = g13["appendix"]
rg["supersedes"] = g13["supersedes"]
rg["updated"] = g13["updated"]
check("v8b_revert_gap_to_814ce808",
      hashlib.sha256(json.dumps(rg, ensure_ascii=False, indent=1,
                                sort_keys=False).encode("utf-8")
                     ).hexdigest()[:8] == "814ce808")
rr = json.loads(json.dumps(r37))
rr["version"] = "1.36"
rr["upgrade_log"] = r36["upgrade_log"]
rr["supersedes"] = r36["supersedes"]
check("v8c_revert_registry_to_78d37211",
      hashlib.sha256(json.dumps(rr, ensure_ascii=False, indent=1,
                                sort_keys=False).encode("utf-8")
                     ).hexdigest()[:8] == "78d37211")

# 6 res_sha8 recompute
res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in res.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v9_res_sha8_recompute", rs == res["res_sha8"] == "fe3ca7a4")

# 7 five writes + receipts presence
def has(p, marker):
    return marker in open(p, encoding="utf-8").read()


check("v10_writes_present",
      has(os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
          "## Phase 3274 (G5-H37):")
      and has(os.path.join(BASE, "research", "deepseek", "docs",
                           "WORKLOG_gpt5_mirror.md"),
              "## Phase 3274 (G5-H37):")
      and has(os.path.join(BASE, "research", "gpt5", "docs",
                           "AGI_GPT5_MEMO.md"),
              "## Phase 3274 (G5-H37):")
      and has(os.path.join(BASE, "research", "deepseek", "docs",
                           "AGI_DEEPSEEK_MEMO.md"), "G 线 3274 回执")
      and has(os.path.join(BASE, "research", "gpt5", "docs",
                           "PENDING_ITEMS_IMPACT_PREVIEW.md"),
              "addendum-3274")
      and has(os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md"),
              "**3274")
      and has(os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md"),
              "3274 G5-H37"))

print("VERIFY", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
