# -*- coding: utf-8 -*-
"""Independent disk verify for Phase 3276 (verify-only; fix verify on FAIL)."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
RBASE = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913")
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3276", "g5h39_appx23_exec")
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


raw = open(LED, "rb").read()
led = json.loads(raw.decode("utf-8"))
ms = led["measurements"]
c2 = json.loads(raw.decode("utf-8"))
c2["ledger_sha256_8"] = "923806e0"
p2 = hashlib.sha256(json.dumps(c2, ensure_ascii=False, indent=1)
                    .encode("utf-8")).hexdigest()[:8]
check("v1_head_chain n=434 field=923806e0 -> ee5b19f3",
      len(ms) == 434 and led["ledger_sha256_8"] == "ee5b19f3"
      and p2 == "ee5b19f3")
led433 = json.loads(raw.decode("utf-8"))
led433["measurements"] = led433["measurements"][:-1]
led433["ledger_sha256_8"] = "98c772a8"
p1 = hashlib.sha256(json.dumps(led433, ensure_ascii=False, indent=1)
                    .encode("utf-8")).hexdigest()[:8]
check("v1b_prev_chain -> 923806e0", p1 == "923806e0")
check("v1c_roundtrip", json.dumps(led, ensure_ascii=False,
                                  indent=1).encode("utf-8") == raw)
check("v2_receipt_entry", ms[-1]["phase"] == 3276
      and ms[-1]["verdict"] == "appx23_exec_registered"
      and ms[-1]["res_sha8"] == "8d5bba36"
      and ms[-1]["prev_post"] == "923806e0")

res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in res.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v3_res_sha8_recompute", rs == res["res_sha8"] == "8d5bba36")
check("v4_xgates", all(res["gates"][k] for k in (
    "X1_attr_bitwise", "X1_syn_bitwise", "X2_kreadout_allclose",
    "X3_sclass_incumbent", "X4_exc_exact", "xgates_all",
    "arm_a_all_registered", "arm_b_all_registered")))
MODELS_A = ['qwen2.5-3b-instruct', 'qwen2-7b', 'deepseek-r1-distill-qwen-7b',
            'glm4-9b-chat-hf', 'gemma-3-4b-it']
art_ok = True
for m in MODELS_A:
    for suf in ('axes.npz', 'kreadout.npz', 'kentity.npz', 'record.json'):
        art_ok = art_ok and os.path.exists(
            os.path.join(OUT, 'appx2_%s_%s' % (m, suf)))
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    art_ok = art_ok and os.path.exists(
        os.path.join(OUT, 'appx3_%s_bases.npz' % m))
    art_ok = art_ok and os.path.exists(
        os.path.join(OUT, 'appx3_%s_record.json' % m))
check("v5_artifacts_on_disk", art_ok)
k3 = all(res["arm_b_appx3"][m]["k3_verdict"]["fires"]
         for m in ("qwen3-4b", "qwen3-14b", "glm4"))
k2 = not any(res["arm_b_appx3"][m]["k2_verdict"]["fires"]
             for m in ("qwen3-4b", "qwen3-14b", "glm4"))
check("v6_deathline_verdicts", k3 and k2)


def has(p, marker):
    return marker in open(p, encoding="utf-8").read()


check("v7_writes_present",
      has(os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
          "## Phase 3276 (G5-H39):")
      and has(os.path.join(BASE, "research", "deepseek", "docs",
                           "WORKLOG_gpt5_mirror.md"),
              "## Phase 3276 (G5-H39):")
      and has(os.path.join(BASE, "research", "gpt5", "docs",
                           "AGI_GPT5_MEMO.md"), "## Phase 3276 (G5-H39):")
      and has(os.path.join(BASE, "research", "gpt5", "docs",
                           "PENDING_ITEMS_IMPACT_PREVIEW.md"),
              "addendum-3276")
      and has(os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md"),
              "**3276")
      and has(os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md"),
              "3276 G5-H39"))

print("VERIFY", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
