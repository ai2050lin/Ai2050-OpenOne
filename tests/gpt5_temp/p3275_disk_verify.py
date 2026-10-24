# -*- coding: utf-8 -*-
"""Independent disk verify for Phase 3275 (verify-only; fix verify on FAIL)."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
RBASE = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913")
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(BASE, "research", "deepseek", "docs",
                    "AGI_DEEPSEEK_MEMO.md")
OUT = os.path.join(RBASE, "phase3275", "g5h38_fruit_k4_erratum")
L_SCI = [1106, 1160, 1318, 3097, 6678, 7396, 7399]
L_DEF = [311, 386, 432]
L_BKG = [1478, 1486, 1509, 1561, 1766, 1769, 2003, 2005, 2279, 2530, 2834,
         3124, 3406, 3657, 3824, 4077, 4340, 4480, 4667, 4768, 4826,
         7153, 7257, 7360, 7409]
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


# 1 ledger chain (two levels) + receipt
raw = open(LED, "rb").read()
led = json.loads(raw.decode("utf-8"))
ms = led["measurements"]
c2 = json.loads(raw.decode("utf-8"))
c2["ledger_sha256_8"] = "98c772a8"
p2 = hashlib.sha256(json.dumps(c2, ensure_ascii=False, indent=1)
                    .encode("utf-8")).hexdigest()[:8]
check("v1_head_chain n=433 field=98c772a8 -> 923806e0",
      len(ms) == 433 and led["ledger_sha256_8"] == "923806e0" and p2 == "923806e0")
led432 = json.loads(raw.decode("utf-8"))
led432["measurements"] = led432["measurements"][:-1]
led432["ledger_sha256_8"] = "498a80f6"
p1 = hashlib.sha256(json.dumps(led432, ensure_ascii=False, indent=1)
                    .encode("utf-8")).hexdigest()[:8]
check("v1b_prev_chain -> 98c772a8", p1 == "98c772a8")
check("v1c_roundtrip", json.dumps(led, ensure_ascii=False,
                                  indent=1).encode("utf-8") == raw)
check("v2_receipt_entry", ms[-1]["phase"] == 3275
      and ms[-1]["verdict"] == "fruit_k4_erratum_registered"
      and ms[-1]["prev_post"] == "98c772a8")

# 2 MEMO append-only + content
now = open(MEMO, "rb").read()
bak = open(MEMO + ".bak_3275", "rb").read()
nowtxt = now.decode("utf-8")
baktxt = bak.decode("utf-8")
check("v3_memo_sha8", hashlib.sha256(now).hexdigest()[:8] == "fb813f57")
check("v4_prefix_equals_bak", now[:len(bak)] == bak)
check("v5_erratum_present",
      "## 勘误注记：挂账「水果类」「K4」结清" in nowtxt
      and "3210 G5-G9" in nowtxt and "3216 挂账总账 v2" in nowtxt
      and all(str(n) in nowtxt for n in L_SCI + L_DEF + L_BKG))
bak_lines = baktxt.split("\n")
now_lines = nowtxt.split("\n")
check("v6_original_35_lines_unchanged",
      all(now_lines[n - 1] == bak_lines[n - 1] for n in L_SCI + L_DEF + L_BKG))
fruit_after = sum(1 for l in now_lines if "水果类" in l)
err_fruit = sum(1 for l in nowtxt[len(baktxt):].split("\n") if "水果类" in l)
check("v7_census_consistent", fruit_after == 26 + err_fruit)

# 3 res_sha8 recompute
res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in res.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v8_res_sha8_recompute", rs == res["res_sha8"]
      and res["verdict"] == "fruit_k4_erratum_registered"
      and res["gate_all"] is True)

# 4 sealed neighbors untouched (gap v1.14 / registry v1.37 from 3274)
def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


check("v9_3274_landings_untouched",
      s8(os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply",
                      "gap_ledger_v1_14.json")) == "c6d97133"
      and s8(os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply",
                          "atlas_registry_v1_37.json")) == "bfb7d7e5")

# 5 five writes present
def has(p, marker):
    return marker in open(p, encoding="utf-8").read()


check("v10_writes_present",
      has(os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
          "## Phase 3275 (G5-H38):")
      and has(os.path.join(BASE, "research", "deepseek", "docs",
                           "WORKLOG_gpt5_mirror.md"),
              "## Phase 3275 (G5-H38):")
      and has(os.path.join(BASE, "research", "gpt5", "docs",
                           "AGI_GPT5_MEMO.md"), "## Phase 3275 (G5-H38):")
      and has(os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md"),
              "**3275")
      and has(os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md"),
              "3275 G5-H38"))

print("VERIFY", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
