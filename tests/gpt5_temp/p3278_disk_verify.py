# -*- coding: utf-8 -*-
"""Phase 3278 independent disk verify (separate from closeout). 10 checks."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
P3278 = os.path.join(RBASE, "phase3278", "g5h41_appx1_exec")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


# 1 chain head recompute: n=436, field=prev(76783ddd) -> c4d25687
raw = open(LED, "rb").read()
led = json.loads(raw.decode("utf-8"))
ms = led["measurements"]
probe = json.loads(raw.decode("utf-8"))
probe["ledger_sha256_8"] = "76783ddd"
rec = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
check("v1 chain head n=436 field=76783ddd -> c4d25687",
      len(ms) == 436 and led["ledger_sha256_8"] == "c4d25687" and rec == "c4d25687")

# 2 prev-level: drop #430 -> ee5b19f3? no: n=435 with field=ee5b19f3 -> 76783ddd
prev = json.loads(raw.decode("utf-8"))
prev["measurements"] = ms[:-1]
prev["ledger_sha256_8"] = "ee5b19f3"
rec2 = hashlib.sha256(json.dumps(prev, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
check("v2 prev-level n=435 field=ee5b19f3 -> 76783ddd", rec2 == "76783ddd")

# 3 #430 record content
r430 = ms[-1]
check("v3 #430 rec phase/verdict/res/prev",
      str(r430["phase"]) == "3278" and r430["verdict"] == "appx1_blocked"
      and r430["res_sha8"] == "588d36c9" and r430["prev_post"] == "76783ddd"
      and "fa1858f9" in r430["note"])

# 4 result.json res_sha8 content recompute
rj = json.load(open(os.path.join(P3278, "result.json"), encoding="utf-8"))
rc = json.loads(json.dumps(rj, ensure_ascii=False))
declared = rc.pop("res_sha8")
rr = hashlib.sha256(json.dumps(rc, sort_keys=True, ensure_ascii=False)
                    .encode("utf-8")).hexdigest()[:8]
check("v4 result res_sha8 content recompute 588d36c9",
      declared == "588d36c9" and rr == "588d36c9")

# 5 gates semantics: gA/gB True, gC False, verdict blocked
check("v5 gates gA/gB=True gC=False verdict appx1_blocked",
      rj["gates"] == {"gA_anchors": True, "gB_fidelity": True,
                      "gC_conservation": False}
      and rj["verdict"] == "appx1_blocked")

# 6 nf4 collects on disk (bf16 active + fp16 archived)
check("v6 nf4_collect.npz + fp16 archive present",
      os.path.exists(os.path.join(P3278, "nf4_collect.npz"))
      and os.path.exists(os.path.join(P3278, "nf4_collect_fp16device.npz"))
      and os.path.exists(os.path.join(P3278, "result_fp16device.json")))

# 7 fp16 archive integrity (its res_sha8 field readable)
rfp = json.load(open(os.path.join(P3278, "result_fp16device.json"),
                     encoding="utf-8"))
check("v7 fp16 archive res fa1858f9 blocked", rfp["res_sha8"] == "fa1858f9"
      and rfp["verdict"] == "appx1_blocked")

# 8 Arm a numbers in result match 3166 anchors
aa = rj["arm_a"]
check("v8 Arm a census anchors",
      aa["census"]["R_logic__K_entity"] == [26.641, 2]
      and aa["census"]["R_logic__K_readout"] == [71.796, 0]
      and aa["census"]["K_readout__K_entity"] == [70.235, 0]
      and round(aa["shared_cos"][0], 6) == 0.893833
      and round(aa["shared_cos"][1], 6) == 0.714872)

# 9 Arm b numbers: G-B 0.0001deg / G-C 30.791 eff2 cos_diff<=0.05
ab = rj["arm_b"]
check("v9 Arm b fidelity+conservation numbers",
      round(ab["fidelity"]["top64_rowspace_deg"], 4) == 0.0001
      and ab["fidelity"]["kent_fp_x_kread_nf4_deg"] == 70.234
      and ab["conservation"]["top1_deg"] == 30.791
      and ab["conservation"]["eff_ge05"] == 2
      and max(ab["conservation"]["cos_diff"]) <= 0.05)

# 10 five-write present + MEMORY<=3000
wl = open(os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
          encoding="utf-8").read()
gm = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
          encoding="utf-8").read()
dl = open(os.path.join(ROOT, ".workbuddy", "memory", "2026-10-21.md"),
          encoding="utf-8").read()
pv = open(os.path.join(ROOT, "research", "gpt5", "docs",
                       "PENDING_ITEMS_IMPACT_PREVIEW.md"), encoding="utf-8").read()
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
check("v10 five-write present + MEMORY<=3000",
      "G5-H41" in wl and "G5-H41" in gm and "3278 G5-H41" in dl
      and "addendum-3278" in pv and "- **3278**" in mem and len(mem) <= 3000)

n_pass = sum(1 for _, v in ok if v)
print("VERIFY %d/%d" % (n_pass, len(ok)))
print("VERIFY PASS" if n_pass == len(ok) else "VERIFY FAIL")
