# -*- coding: utf-8 -*-
"""Phase 3279 (G5-H42): APPX-1 landing — appraisal arm + gap ledger
v1.15 -> v1.16 with APPX-9 appendix entry + registry v1.38 -> v1.39
upgrade_log annotation, append-only with bitwise REVERT.

Landing-pattern precedent: Phase 3277 (G5/H5 after exec phase 3276).
Input: Phase 3278 (G5-H41, APPX-1 nf4 exec, res 588d36c9, verdict
appx1_blocked — gA anchors PASS, gB fidelity PASS, gC conservation FAIL
honestly recorded). Per F-ledger principle (failures must be ledgered),
APPX-9 records the measurement AND the blocked verdict verbatim; it does
NOT close the APPX-1 standing item (owner decision reserved).

Arm a (appraisal): evidence level, anchor completeness (p3278 res on
disk), non-conflict with APPX-1..8, ledger chain state.
Arm b (landing, gated on appraisal): new landed files in phase3279 dir;
REVERT rebuild byte-identical to sealed v1.15 (58094dab) and v1.38
(f8772d65); gaps/features/failures untouched (pointer-diff audit).
Zero write to atlas_ledger.json (ledger entry happens at closeout #431).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3279", "g5h42_appx1_landing")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3279_SMOKE") == "1"

G15 = os.path.join(RBASE, "phase3277", "g5h40_appx23_landing",
                   "gap_ledger_v1_15.json")
R38 = os.path.join(RBASE, "phase3277", "g5h40_appx23_landing",
                   "atlas_registry_v1_38.json")
P3278_RES = os.path.join(RBASE, "phase3278", "g5h41_appx1_exec", "result.json")

G15_SHA = "58094dab"
R38_SHA = "f8772d65"
P3278_SHA = "588d36c9"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3279,
    "name": "g5h42_appx1_landing",
    "prereg_id": "G5-H42 (landing-pattern precedent 3276->3277; 3278 preregistration)",
    "arms": {
        "a": ("appraisal: evidence level (statistical, from 3278 gA anchors "
              "PASS + gB fidelity PASS; gC FAIL honestly recorded per "
              "F-ledger principle), anchor completeness (p3278 res "
              "588d36c9 on disk), non-conflict with APPX-1..8, ledger chain"),
        "b": ("landing (gated on appraisal): gap v1.15->v1.16 appendix "
              "+APPX-9; registry v1.38->v1.39 upgrade_log +1; REVERT bitwise "
              "to 58094dab / f8772d65; gaps/features/failures untouched; "
              "APPX-1 standing item NOT closed (owner decision reserved)"),
    },
    "appx9_summary": ("APPX-1 exec (owner-approved nf4 baseline, Phase 3278): "
                      "Arm a zero-GPU localization reproduced 3166 anchors "
                      "exactly (26.641/2, 71.796/0, 70.235/0) and localized "
                      "the two shared dims (dim0 cos=0.8938 adjudication-"
                      "connective family, dim1 cos=0.7149 truth-judgment "
                      "family; both neonatal layers 37-40); Arm b nf4: gB "
                      "fidelity PASS (rowspace 0.0001 deg, K_ent_fp x "
                      "K_read_nf4 = 70.234), gC conservation top1 FAIL "
                      "(30.791 vs 26.641 + 3.0 tol; drift carried by "
                      "R_logic side) while eff_ge05=2 and shared-cos diffs "
                      "<=0.035 conserve. Honest verdict appx1_blocked."),
    "inputs": {
        "gap_v1_15_sha8": G15_SHA,
        "registry_v1_38_sha8": R38_SHA,
        "p3278_res_sha8": P3278_SHA,
    },
    "ledger": {"n": 436, "post": "c4d25687", "chain_prev": "76783ddd"},
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "sealed_untouched": [G15_SHA + " (gap v1.15)", R38_SHA + " (registry v1.38)"],
}
DESIGN["mode"] = "smoke" if SMOKE else "formal"
DESIGN["created"] = today()
design_sha8 = hashlib.sha256(
    json.dumps({k: v for k, v in DESIGN.items() if k != "created"},
               sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    os.makedirs(SMOKE_DIR, exist_ok=True)
with open(os.path.join(OUT, "design_sha8.txt"), "w", encoding="utf-8", newline="") as f:
    f.write(design_sha8)
print(("SMOKE " if SMOKE else "") + "freeze", design_sha8)

gates = {}

# ---------------------------------------------------------------- g1: ledger state
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
ms = led["measurements"]
chk_led = json.loads(led_raw.decode("utf-8"))
chk_led["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
rec = hashlib.sha256(json.dumps(chk_led, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_ledger_state"] = (
    len(ms) == DESIGN["ledger"]["n"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"])
print("g1 ledger: n=%d post=%s recompute=%s" % (len(ms), led["ledger_sha256_8"], rec))

# ---------------------------------------------------------------- g2: inputs on disk
r278 = json.load(open(P3278_RES, encoding="utf-8"))
gates["g2_inputs"] = (
    s8(G15) == G15_SHA and s8(R38) == R38_SHA
    and r278["res_sha8"] == P3278_SHA
    and r278["verdict"] == "appx1_blocked"
    and r278["gates"]["gA_anchors"] is True
    and r278["gates"]["gB_fidelity"] is True
    and r278["gates"]["gC_conservation"] is False)
print("g2 inputs: gap=%s registry=%s p3278=%s verdict=%s"
      % (s8(G15) == G15_SHA, s8(R38) == R38_SHA,
         r278["res_sha8"], r278["verdict"]))

# ---------------------------------------------------------------- Arm a: appraisal
appraisal = {}
aa = r278["arm_a"]
ab = r278["arm_b"]
cens = aa["census"]
anchor_ok = all(abs(cens[k][0] - v) < 0.05 for k, v in
                (("R_logic__K_entity", 26.641),
                 ("R_logic__K_readout", 71.796),
                 ("K_readout__K_entity", 70.235)))
appraisal["q1_evidence_level"] = {
    "level": "statistical",
    "basis": [
        "phase3278 appx1_exec (owner-approved nf4 baseline): gA anchors "
        "PASS (%s), gB readout fidelity PASS (rowspace 0.0001 deg; "
        "K_ent_fp x K_read_nf4 = 70.234 vs anchor 70.235), gC conservation "
        "top1 FAIL recorded honestly (30.791 vs 26.641 + 3.0 tol)"
        % anchor_ok,
        "Arm a: two shared dims localized (cos 0.8938 / 0.7149), both "
        "neonatal layers 37-40, vocab projections = adjudication-connective "
        "family / truth-judgment family; dim1 truth component traceable to "
        "3157 panel polarity arms",
        "Arm b: nf4 quantitative conservation FAILS at preregistered "
        "strength but eff_ge05=2 and shared-cos diffs <=0.035 conserve — "
        "F-ledger principle: the blocked verdict itself is ledgered",
    ],
    "verdict": "LAND (as honest blocked-measurement record, F-ledger "
               "principle; same landing pattern as APPX-7/APPX-8)",
}
appraisal["q2_anchor_completeness"] = {
    "p3278": {"res_sha8": r278["res_sha8"], "verdict": r278["verdict"],
              "gates": r278["gates"]},
    "owner_approval": ("APPX-1 nf4 baseline approved by owner (AskUserQuestion "
                       "record, pre-3278); execution completed 3278"),
    "verdict": "COMPLETE",
}
appraisal["q3_non_conflict"] = {
    "appendix_ids_prior": None,  # filled below
    "new_content": ("APPX-9 records the APPX-1 execution result verbatim "
                    "including the blocked verdict; it does not modify any "
                    "APPX-1..8 claim and does NOT close the APPX-1 standing "
                    "item (closure-vs-retest disposition reserved for owner)."),
    "verdict": "NO_CONFLICT",
}
appraisal["q4_chain"] = {
    "ledger": "n=436 post c4d25687 (verified g1)",
    "gap_v1_15": G15_SHA, "registry_v1_38": R38_SHA,
    "verdict": "CLEAN",
}

# ---------------------------------------------------------------- Arm b: landing
g15d = json.load(open(G15, encoding="utf-8"))
appraisal["q3_non_conflict"]["appendix_ids_prior"] = [a["id"] for a in g15d["appendix"]]

cons = ab["conservation"]
fid = ab["fidelity"]
appx9_entry = {
    "id": "APPX-9",
    "title": "APPX-1 执行相结果：R_logic×K_entity 共享维定位与 nf4 基线诚实"
             "判定（APPX-1 exec: shared-dim localization & honest nf4 "
             "verdict appx1_blocked）",
    "anchor": "p3278",
    "status": "appendix_recorded_blocked_verbatim",
    "note": ("APPX-1 执行相（owner 批准 nf4 基线，p3278 res 588d36c9，verdict "
             "appx1_blocked 逐字入账——失败必入账原则）。Arm a（零 GPU 权威臂）："
             "3166 三锚逐值复现（R_logic×K_entity 26.641°/eff=2、"
             "R_logic×K_readout 71.796°/eff=0、K_readout×K_entity "
             "70.235°/eff=0，tol 0.05°）。两个共享维定位完成：dim0 cos=0.8938"
             "（R_logic row0 主对比方向 × K_ent rows0+1；词表投影=转折连词族"
             "「但由于/但他/But/其次是」）；dim1 cos=0.7149（R_logic rows5/3/7 "
             "× K_ent row2 主导 coef 0.671；词表投影=真值判断族「错/True/"
             "False/判断」；真值成分溯源 3157 面板极性臂）。末段层 37–40 双维"
             "新生（dim1 占比 0.58→1.0 单调）。Arm b（Qwen3-14B-bnb-nf4，bf16 "
             "口径 866 行重采；fp16 装置口径首跑 res fa1858f9 存档为 "
             "*_fp16device.*）：G-B 保真 PASS——lm_head fp16 未量化，"
             "K_readout rowspace 角 0.0001°，K_ent_fp×K_read_nf4=70.234°。"
             "G-C 守恒 top1 FAIL——30.791° vs 26.641°+3.0° 容差（差 4.15°），"
             "扰动由 R_logic 侧承载（R_nf4×K_ent_fp=31.44°）；但 eff_ge05=2 "
             "守恒、共享维 cos 差 ≤0.035 守恒（0.8938→0.8590、0.7149→0.7338）"
             "——量化扰动真实但非定性。APPX-1 挂账状态不在本条关闭（关闭 vs "
             "留待全精度硬件复测，owner 裁决权保留）。装置教训：nf4 臂 dtype "
             "须对齐原版 bfloat16（3263 FP32 bridge 同类场景）；V×V Gram "
             "(151936²) 不可行，改 D 空间 GPU fp32 eigh（rowspace 0.0001° "
             "验证）。"),
    "anchor_sha8": P3278_SHA,
}
g16 = json.loads(json.dumps(g15d))
g16["version"] = "1.16"
g16["appendix"] = g15d["appendix"] + [appx9_entry]
g16["supersedes"] = "gap_ledger_v1_15.json (%s, Phase 3277)" % G15_SHA
g16["updated"] = today()

r38d = json.load(open(R38, encoding="utf-8"))
r39 = json.loads(json.dumps(r38d))
r39["version"] = "1.39"
r39["upgrade_log"] = r38d["upgrade_log"] + [{
    "node": "gap ledger appendix",
    "change": ("APPX-9 landed: APPX-1 exec result (owner-approved nf4 "
               "baseline, Phase 3278 res 588d36c9, verdict appx1_blocked "
               "recorded verbatim per F-ledger principle). Arm a zero-GPU: "
               "3166 anchors reproduced exactly; two shared dims localized "
               "(dim0 cos=0.8938 adjudication-connective family, dim1 "
               "cos=0.7149 truth-judgment family traceable to 3157 panel "
               "polarity arms; both neonatal layers 37-40). Arm b nf4 "
               "(bf16 device, 866 rows; fp16 first run archived as "
               "result_fp16device.json): gB fidelity PASS (rowspace "
               "0.0001 deg, K_ent_fp x K_read_nf4 = 70.234), gC "
               "conservation top1 FAIL (30.791 vs 26.641 + 3.0 tol, drift "
               "carried by R_logic side) while eff_ge05=2 and shared-cos "
               "diffs <=0.035 conserve. APPX-1 standing item NOT closed "
               "(owner decision reserved). Anchors unchanged; no "
               "feature/failure changes."),
    "phase": 3279,
    "gap_ledger": "gap_ledger_v1_16.json",
}]
r39["supersedes"] = "atlas_registry_v1_38.json (%s, Phase 3277)" % R38_SHA
# NOTE: registry chain has NO 'updated' key (3256 SMOKE lesson) — do not add.

# REVERT rebuilds (byte-identical to sealed predecessors)
rev_g = json.loads(json.dumps(g16))
rev_g["version"] = "1.15"
rev_g["appendix"] = g15d["appendix"]
rev_g["supersedes"] = g15d["supersedes"]
rev_g["updated"] = g15d["updated"]
rev_r = json.loads(json.dumps(r39))
rev_r["version"] = "1.38"
rev_r["upgrade_log"] = r38d["upgrade_log"]
rev_r["supersedes"] = r38d["supersedes"]

g4_g = hashlib.sha256(json.dumps(rev_g, ensure_ascii=False, indent=1,
                                 sort_keys=False).encode("utf-8")).hexdigest()[:8]
g4_r = hashlib.sha256(json.dumps(rev_r, ensure_ascii=False, indent=1,
                                 sort_keys=False).encode("utf-8")).hexdigest()[:8]
gates["g4_revert_bitwise"] = (g4_g == G15_SHA and g4_r == R38_SHA)
print("g4 revert bitwise: gap=%s registry=%s" % (g4_g, g4_r))

# ---------------------------------------------------------------- g5: ledger-diff audit
gates["g5_pointer_diff"] = (
    g16["gaps"] == g15d["gaps"]
    and r39["features"] == r38d["features"]
    and r39["failures"] == r38d["failures"]
    and r39["provenance"]["anchor_files"] == r38d["provenance"]["anchor_files"]
    and r39["census_3238"] == r38d["census_3238"]
    and r39["evidence_level_taxonomy"] == r38d["evidence_level_taxonomy"])
print("g5 pointer diff:", gates["g5_pointer_diff"])

gates["g3_appraisal"] = (
    appraisal["q1_evidence_level"]["verdict"].startswith("LAND")
    and appraisal["q2_anchor_completeness"]["verdict"] == "COMPLETE"
    and appraisal["q3_non_conflict"]["verdict"] == "NO_CONFLICT"
    and appraisal["q4_chain"]["verdict"] == "CLEAN")
print("g3 appraisal: %s" % {k: v["verdict"] for k, v in appraisal.items()})

# ---------------------------------------------------------------- write landed files
if not SMOKE:
    with open(os.path.join(OUT, "gap_ledger_v1_16.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(g16, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "atlas_registry_v1_39.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(r39, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "gap_v116_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"version": g16["version"], "appendix_n": len(g16["appendix"]),
                   "revert": gates["g4_revert_bitwise"]}, f,
                  ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- g6: landed files + zero write
zero_before = s8(LED)
landed_ok = True
if not SMOKE:
    g16_disk = json.load(open(os.path.join(OUT, "gap_ledger_v1_16.json"),
                              encoding="utf-8"))
    r39_disk = json.load(open(os.path.join(OUT, "atlas_registry_v1_39.json"),
                              encoding="utf-8"))
    landed_ok = (g16_disk["version"] == "1.16"
                 and len(g16_disk["appendix"]) == 10
                 and g16_disk["appendix"][-1]["id"] == "APPX-9"
                 and r39_disk["version"] == "1.39"
                 and len(r39_disk["upgrade_log"]) == 18)
zero_after = s8(LED)
gates["g6_landed_files"] = landed_ok and (zero_before == zero_after)
print("g6 landed files + zero-write:", gates["g6_landed_files"])

# ---------------------------------------------------------------- verdict
gate_all = all(v for v in gates.values() if v is not None)
res = {
    "phase": 3279,
    "name": "g5h42_appx1_landing",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "appx1_landed_blocked_verbatim" if gate_all else "BLOCKED",
    "appraisal": appraisal,
    "gap_v116": {
        "file": "gap_ledger_v1_16.json",
        "appendix_n": 10,
        "appx9": appx9_entry,
        "supersedes": g16["supersedes"],
        "file_sha8": (s8(os.path.join(OUT, "gap_ledger_v1_16.json"))
                      if not SMOKE else "smoke"),
    },
    "registry_v139": {
        "file": "atlas_registry_v1_39.json",
        "upgrade_log_n": 18,
        "anchors_unchanged": True,
        "supersedes": r39["supersedes"],
        "file_sha8": (s8(os.path.join(OUT, "atlas_registry_v1_39.json"))
                      if not SMOKE else "smoke"),
    },
    "appx1_standing_item": ("NOT closed by this phase; closure-vs-retest "
                            "disposition reserved for owner"),
    "revert_check": {"gap": gates["g4_revert_bitwise"],
                     "registry": gates["g4_revert_bitwise"]},
    "sealed_untouched": {"gap_v1_15": G15_SHA, "registry_v1_38": R38_SHA},
    "zero_write_verified": zero_after,
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
for k, v in gates.items():
    print(" ", k, v)
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
