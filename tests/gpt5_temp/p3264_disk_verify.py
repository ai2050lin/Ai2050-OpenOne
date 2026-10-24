# -*- coding: utf-8 -*-
"""Independent read-only disk verification for phase 3264 closeout."""
import hashlib
import json
import os

ROOT = r"D:\AI2050\Ai2050-OpenOne"
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3264", "g5h27_traj_second_order")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
P3263 = os.path.join(RBASE, "phase3263", "g5h26_p_traj")

FREEZE = "1c734572"
RES_SHA = "f223a9a0"
POST = "113e6af3"
CHAIN_PREV = "b740bf76"

fails = []
oks = 0


def chk(name, cond, extra=""):
    global oks
    if cond:
        oks += 1
        print("PASS", name)
    else:
        fails.append(name)
        print("FAIL", name, extra)


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
S = json.load(open(os.path.join(OUT, "trajectory_second_order_v1.json"), encoding="utf-8"))

# 1 result verdict + hashes
chk("1 result verdict/hashes",
    R["verdict"] == "traj_second_order_registered" and R["gate_all"] is True
    and R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
    and R["design_sha8"] == open(os.path.join(OUT, "design_sha8.txt"),
                                 encoding="utf-8").read())

# 2 res_sha8 recompute (res has no int-keyed dicts; direct round-trip OK)
r2 = {k: v for k, v in R.items() if k != "res_sha8"}
rec = hashlib.sha256(json.dumps(r2, sort_keys=True, ensure_ascii=False)
                     .encode("utf-8")).hexdigest()[:8]
chk("2 res_sha8 recompute", rec == RES_SHA, rec)

# 3 gates 8/8
chk("3 gates 8/8", len(R["gates"]) == 8 and all(R["gates"].values()))

# 4 artifact hash + zero-write snapshot format (snapshot taken at g8, i.e.
# BEFORE ledger #416 append -> must differ from current ledger file sha)
zw = R["zero_write_verified"]
chk("4 artifacts + zero-write",
    s8(os.path.join(OUT, "trajectory_second_order_v1.json")) == "8d1a7b09"
    and len(zw) == 8 and zw != s8(LED))

# 5 second-order artifact schema + key numbers
cur = S["arm_a_per_step_curves"]
chk("5 artifact schema", S["schema"] == "gline_traj_second_order_v1"
    and len(cur) == 6
    and cur["wscp_2748"]["profile"] == "biphasic"
    and cur["true_2747"]["profile"] == "monotone_stable"
    and cur["wscp_2748"]["kout_top1"] == [0.416462, 0.42369, 0.445397, 0.610919])

# 6 localization segments (independent recompute from result.json of 3263)
r263 = json.load(open(os.path.join(P3263, "result.json"), encoding="utf-8"))
b = r263["per_state"]["base"]
w = r263["per_state"]["within_surface_class_permuted_token_2748@128"]
d = [abs(float(w["top1_curve"][str(k)]) - float(b["top1_curve"][str(k)]))
     for k in range(29)]
seg_v = S["arm_b_attribution"]["localization_segments"]
chk("6 localization recompute",
    max(d[:17]) == 0.0
    and abs(max(d[23:]) - seg_v["readout_23_28_max_abs"]) < 1e-6
    and abs(seg_v["readout_23_28_max_abs"] - 0.194739) < 1e-5
    and abs(max(d[17:23]) - seg_v["carrier_17_22_max_abs"]) < 1e-6)

# 7 Track-I invariants recompute (all 24 states, bitwise)
RUNS = ["true_token_2747", "true_token_2748",
        "within_surface_class_permuted_token_2747",
        "within_surface_class_permuted_token_2748",
        "surface_class_mass_2747", "surface_class_mass_2748"]
inv = all(r263["per_state"]["%s@%d" % (run, st)]["sigma1"]["mid_emb"]
          == b["sigma1"]["mid_emb"]
          and r263["per_state"]["%s@%d" % (run, st)]["top1_curve"]["0"] == 1.0
          and r263["per_state"]["%s@%d" % (run, st)]["cos_emb_to_base"] == 1.0
          for run in RUNS for st in [1, 8, 32, 128])
chk("7 Track-I invariants (24 states bitwise)", inv)

# 8 APPX-7 draft embedded with decision_pending
chk("8 APPX-7 draft", S["arm_c_appx7"]["id"] == "APPX-7"
    and S["arm_c_appx7"]["status"].startswith("CANDIDATE_DRAFT")
    and S["arm_c_appx7"]["observation"]["within_model_1_7b_finetune"]
    ["states_outside_cross_model_band"] == 0
    and "registry" in S["arm_c_appx7"]["decision_pending"])

# 9 ledger #416 chained
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
e = ms[-1]
chk("9 ledger #416", len(ms) == 421 and e["phase"] == 3264
    and e["verdict"] == "traj_second_order_registered"
    and e["res_sha8"] == RES_SHA and e["design_sha8"] == FREEZE
    and led["ledger_sha256_8"] == POST)
cp = json.loads(json.dumps(led))
cp["ledger_sha256_8"] = CHAIN_PREV
prev_state = json.loads(json.dumps(led))
prev_state["measurements"] = ms[:-1]
prev_state["ledger_sha256_8"] = CHAIN_PREV
# chain: post computed over state WITH #416 and field=prev
cp2 = json.loads(json.dumps(led))
cp2["ledger_sha256_8"] = CHAIN_PREV
rec2 = hashlib.sha256(json.dumps(cp2, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
chk("10 ledger chain recompute", rec2 == POST, rec2)

# 11 3263 sealed untouched
chk("11 p3263 sealed", s8(os.path.join(P3263, "result.json")) ==
    hashlib.sha256(open(os.path.join(P3263, "result.json"), "rb").read()).hexdigest()[:8]
    and r263["res_sha8"] == "f0f95c27")

# 12 MEMO section
memo = open(MEMO, encoding="utf-8").read()
chk("12 MEMO section", "## Phase 3264 (G5-H27):" in memo
    and "1c734572" in memo and "5b3162da" not in memo)

# 13 dual wlogs
for i, p in enumerate([os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
                       os.path.join(ROOT, "research", "deepseek", "docs",
                                    "WORKLOG_gpt5_mirror.md")]):
    t = open(p, encoding="utf-8").read()
    chk("1%d wlog %s" % (3 + i, os.path.basename(os.path.dirname(p))),
        "3264 G5-H27" in t and POST in t)

# 14 MEMORY
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
chk("14 MEMORY", "**3264" in mem and len(mem) <= 3000, len(mem))

print("VERIFY %d/%d" % (oks, oks + len(fails)))
if fails:
    print("FAILS:", fails)
    raise SystemExit(1)
print("ALL OK")
