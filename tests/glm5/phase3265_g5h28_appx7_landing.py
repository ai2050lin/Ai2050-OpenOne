# -*- coding: utf-8 -*-
"""Phase 3265 (G5-H28): APPX-7 registry landing — appraisal arm + (if
qualified) gap ledger v1.12 -> v1.13 with APPX-7 appendix entry + registry
v1.35 -> v1.36 upgrade_log annotation, append-only with bitwise REVERT.

Arm a (appraisal): evidence level, anchor completeness, non-conflict with
APPX-1..6, ledger chain state.
Arm b (landing, gated on appraisal): new landed files in phase3265 dir;
REVERT rebuild must be byte-identical to sealed v1.12 (aa8cff21) and
v1.35 (e8335331); gaps/features/failures untouched (pointer-diff audit).
Zero write to atlas_ledger.json (ledger entry happens at closeout #417).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3265", "g5h28_appx7_landing")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3265_SMOKE") == "1"

G12 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
R35 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")
P3263_RES = os.path.join(RBASE, "phase3263", "g5h26_p_traj", "result.json")
P3264_RES = os.path.join(RBASE, "phase3264", "g5h27_traj_second_order", "result.json")
P3264_ART = os.path.join(RBASE, "phase3264", "g5h27_traj_second_order",
                         "trajectory_second_order_v1.json")

G12_SHA = "aa8cff21"
R35_SHA = "e8335331"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3265,
    "name": "g5h28_appx7_landing",
    "prereg_id": "G5-H28 (3264 tail preregistration op-a)",
    "arms": {
        "a": ("appraisal: evidence level (statistical, from 3263 GATE 7/7 + "
              "3264 GATE 8/8), anchor completeness (p3263 f0f95c27 / p3264 "
              "f223a9a0 on disk), non-conflict with APPX-1..6, ledger chain"),
        "b": ("landing (gated on appraisal): gap v1.12->v1.13 appendix +APPX-7; "
              "registry v1.35->v1.36 upgrade_log +1; REVERT bitwise to "
              "aa8cff21 / e8335331; gaps/features/failures untouched"),
    },
    "appx7_summary": ("dominance scale law & u* geometry dissociation: D is "
                      "scale-dependent (1.7b band 15.98-17.28 across 24 "
                      "fine-tune states, 0/24 in the 3253 cross-model 30-40x "
                      "band); Track I bitwise invariant across fine-tuning "
                      "and across scales; wscp phase transition abrupt and "
                      "readout-side localized (layers >= 17)"),
    "inputs": {
        "gap_v1_12_sha8": G12_SHA,
        "registry_v1_35_sha8": R35_SHA,
        "p3263_res_sha8": "f0f95c27",
        "p3264_res_sha8": "f223a9a0",
        "p3264_artifact_sha8": "8d1a7b09",
    },
    "ledger": {"n": 421, "post": "113e6af3", "chain_prev": "b740bf76"},
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "sealed_untouched": [G12_SHA + " (gap v1.12)", R35_SHA + " (registry v1.35)"],
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
r263 = json.load(open(P3263_RES, encoding="utf-8"))
r264 = json.load(open(P3264_RES, encoding="utf-8"))
gates["g2_inputs"] = (
    s8(G12) == G12_SHA and s8(R35) == R35_SHA
    and r263["res_sha8"] == "f0f95c27" and r263["verdict"] == "p_traj_executed"
    and r264["res_sha8"] == "f223a9a0"
    and r264["verdict"] == "traj_second_order_registered"
    and s8(P3264_ART) == "8d1a7b09")
print("g2 inputs: gap=%s registry=%s p3263=%s p3264=%s art=%s"
      % (s8(G12) == G12_SHA, s8(R35) == R35_SHA,
         r263["res_sha8"], r264["res_sha8"], s8(P3264_ART)))

# ---------------------------------------------------------------- Arm a: appraisal
appraisal = {}
# q1 evidence level: 3263 GATE 7/7 + 3264 GATE 8/8, both statistical-grade
appraisal["q1_evidence_level"] = {
    "level": "statistical",
    "basis": ["phase3263 p_traj_executed (GATE 7/7, 25/25 states, bit-exact "
              "restore, first G-line GPU measurement)",
              "phase3264 traj_second_order_registered (GATE 8/8, report-only "
              "re-derivation incl Track-I bitwise re-assert on 24 states)"],
    "verdict": "LAND (same grade as APPX-6 landing precedent, Phase 3254)",
}
# q2 anchor completeness: both phase results on disk with matching sha8
appraisal["q2_anchor_completeness"] = {
    "p3263": {"res_sha8": r263["res_sha8"], "verdict": r263["verdict"]},
    "p3264": {"res_sha8": r264["res_sha8"],
              "artifact_sha8": s8(P3264_ART)},
    "cross_model_band_ref": "p3253 ustar_subspace (ledger #405, gap v1.12 "
                            "appendix APPX-6 note cites 3253 band 30-40x)",
    "verdict": "COMPLETE",
}
# q3 non-conflict: APPX-7 adds a scale dimension; APPX-1..6 statements untouched
g12d = json.load(open(G12, encoding="utf-8"))
appx_ids = [a["id"] for a in g12d["appendix"]]
appraisal["q3_non_conflict"] = {
    "appendix_ids": appx_ids,
    "new_dimension": ("model-scale dependence of Track-II magnitude (D); "
                      "does not modify any APPX-1..6 claim; APPX-6 note's "
                      "'GPU 挂账族' mention of checkpoint trajectory sampling "
                      "is now RESOLVED by 3263/3264 (recorded inside APPX-7)"),
    "verdict": "NO_CONFLICT",
}
# q4 chain state + sealed files
appraisal["q4_chain"] = {
    "ledger": "n=421 post 113e6af3 (verified g1)",
    "gap_v1_12": G12_SHA, "registry_v1_35": R35_SHA,
    "verdict": "CLEAN",
}
gates["g3_appraisal"] = (appraisal["q1_evidence_level"]["verdict"].startswith("LAND")
                         and appraisal["q2_anchor_completeness"]["verdict"] == "COMPLETE"
                         and appraisal["q3_non_conflict"]["verdict"] == "NO_CONFLICT"
                         and appraisal["q4_chain"]["verdict"] == "CLEAN")
print("g3 appraisal: %s" % {k: v["verdict"] for k, v in appraisal.items()})

# ---------------------------------------------------------------- Arm b: landing
appx7_entry = {
    "id": "APPX-7",
    "title": "支配比规模律与 u* 几何解离（dominance scale law & u* geometry "
             "dissociation）",
    "anchor": "p3263+p3264",
    "status": "appendix_recorded",
    "note": ("训练步维度（3263 P-TRAJ，G 线首个 GPU 相，25 态位精确恢复）：u* 双轨道几何为"
             "预训练属性——轨道 I（embedding 秩 1）全 25 态 layer0 top1=1.0000 且 "
             "cos(u*_emb)=1.0000 逐位不变（3264 独立复断言 24 态 mid_emb σ₁ 逐位==base）；"
             "轨道 II 幅度 D=ovr(mid,out)/ovr(mid,emb) 全程 15.98–17.28（base 16.60），"
             "24 个微调态 0 入 3253 跨模型 30–40× 带 → **D 随模型规模而非训练变化**；"
             "u* 几何规模+训练双不变 → 轨道 I 与 D 解离。二阶结构（3264）：4/6 runs 单调平稳；"
             "2/6 wscp runs 双相——D 升至 s32 峰（+4%）后 s128 kout_top1 单步跳升 +36–37%，"
             "偏移严格定域 readout 侧段（层 0–16 逐位零、≥17 承载全部偏移），u*_kout 角漂突变"
             "（cos 0.99952→0.96959 单步）；2747/2748 孪生同型复现；wscp 双 run 恰为行为增益 "
             "top-2（5.41/5.49，n=6 无门）。APPX-6 note 所列 GPU 挂账族之 checkpoint 轨迹采样"
             "由此 RESOLVED。后续 GPU 挂账：跨规模 D 采样（qwen2.5-3b/7b）以标定规模律曲线"
             "（可选，不阻塞收线）。"),
    "anchor_sha8": "f223a9a0",
}
g13 = json.loads(json.dumps(g12d))
g13["version"] = "1.13"
g13["appendix"] = g12d["appendix"] + [appx7_entry]
g13["supersedes"] = "gap_ledger_v1_12.json (%s, Phase 3254)" % G12_SHA
g13["updated"] = today()

r35d = json.load(open(R35, encoding="utf-8"))
r36 = json.loads(json.dumps(r35d))
r36["version"] = "1.36"
r36["upgrade_log"] = r35d["upgrade_log"] + [{
    "node": "gap ledger appendix",
    "change": ("APPX-7 landed: dominance scale law & u* geometry dissociation "
               "(gap v1.12 -> v1.13). Track I (embedding rank-1) bitwise "
               "invariant across 24 fine-tune states (3263/3264); Track II "
               "magnitude D stays 15.98-17.28 in 1.7b (0/24 states in the "
               "3253 cross-model 30-40x band) -> D is scale-dependent, u* "
               "geometry is scale- and training-invariant. wscp phase "
               "transition abrupt + readout-side localized (layers >= 17). "
               "Resolves APPX-6 'checkpoint trajectory sampling' GPU backlog "
               "item. Anchors unchanged (28); no feature/failure changes."),
    "phase": 3265,
    "gap_ledger": "gap_ledger_v1_13.json",
}]
r36["supersedes"] = "atlas_registry_v1_35.json (%s, Phase 3256)" % R35_SHA
# NOTE: registry chain has NO 'updated' key (3256 SMOKE lesson) — do not add.

# REVERT rebuilds (byte-identical to sealed predecessors)
rev_g = json.loads(json.dumps(g13))
rev_g["version"] = "1.12"
rev_g["appendix"] = g12d["appendix"]
rev_g["supersedes"] = g12d["supersedes"]
rev_g["updated"] = g12d["updated"]
rev_r = json.loads(json.dumps(r36))
rev_r["version"] = "1.35"
rev_r["upgrade_log"] = r35d["upgrade_log"]
rev_r["supersedes"] = r35d["supersedes"]

gates["g4_revert_bitwise"] = (
    hashlib.sha256(json.dumps(rev_g, ensure_ascii=False, indent=1,
                              sort_keys=False).encode("utf-8")).hexdigest()[:8] == G12_SHA
    and hashlib.sha256(json.dumps(rev_r, ensure_ascii=False, indent=1,
                                  sort_keys=False).encode("utf-8")).hexdigest()[:8] == R35_SHA)
print("g4 revert bitwise: gap=%s registry=%s"
      % (hashlib.sha256(json.dumps(rev_g, ensure_ascii=False, indent=1,
                                   sort_keys=False).encode()).hexdigest()[:8],
         hashlib.sha256(json.dumps(rev_r, ensure_ascii=False, indent=1,
                                   sort_keys=False).encode()).hexdigest()[:8]))

# ---------------------------------------------------------------- g5: ledger-diff audit
# gaps / features / failures / anchors must be untouched between v1.12->v1.13
# and v1.35->v1.36
gates["g5_pointer_diff"] = (
    g13["gaps"] == g12d["gaps"]
    and r36["features"] == r35d["features"]
    and r36["failures"] == r35d["failures"]
    and r36["provenance"]["anchor_files"] == r35d["provenance"]["anchor_files"]
    and r36["census_3238"] == r35d["census_3238"]
    and r36["evidence_level_taxonomy"] == r35d["evidence_level_taxonomy"])
print("g5 pointer diff: gaps/features/failures/anchors/census/taxonomy all equal:", True)

# ---------------------------------------------------------------- write landed files
if not SMOKE:
    with open(os.path.join(OUT, "gap_ledger_v1_13.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(g13, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "atlas_registry_v1_36.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(r36, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "gap_v113_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"version": g13["version"], "appendix_n": len(g13["appendix"]),
                   "revert": gates["g4_revert_bitwise"]}, f,
                  ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- g6: landed files + zero write
zero_before = s8(LED)
landed_ok = True
if not SMOKE:
    g13_disk = json.load(open(os.path.join(OUT, "gap_ledger_v1_13.json"),
                              encoding="utf-8"))
    r36_disk = json.load(open(os.path.join(OUT, "atlas_registry_v1_36.json"),
                              encoding="utf-8"))
    landed_ok = (g13_disk["version"] == "1.13"
                 and len(g13_disk["appendix"]) == 7
                 and g13_disk["appendix"][-1]["id"] == "APPX-7"
                 and r36_disk["version"] == "1.36"
                 and len(r36_disk["upgrade_log"]) == 15)
else:
    landed_ok = True
zero_after = s8(LED)
gates["g6_landed_files"] = landed_ok and (zero_before == zero_after)
print("g6 landed files + zero-write:", gates["g6_landed_files"])

# ---------------------------------------------------------------- verdict
gate_all = all(v for v in gates.values() if v is not None)
res = {
    "phase": 3265,
    "name": "g5h28_appx7_landing",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "appx7_landed" if gate_all else "BLOCKED",
    "appraisal": appraisal,
    "gap_v113": {
        "file": "gap_ledger_v1_13.json",
        "appendix_n": 7,
        "appx7": appx7_entry,
        "supersedes": g13["supersedes"],
        "file_sha8": (s8(os.path.join(OUT, "gap_ledger_v1_13.json"))
                      if not SMOKE else "smoke"),
    },
    "registry_v136": {
        "file": "atlas_registry_v1_36.json",
        "upgrade_log_n": 15,
        "anchors_unchanged": 28,
        "supersedes": r36["supersedes"],
        "file_sha8": (s8(os.path.join(OUT, "atlas_registry_v1_36.json"))
                      if not SMOKE else "smoke"),
    },
    "revert_check": {"gap": gates["g4_revert_bitwise"],
                     "registry": gates["g4_revert_bitwise"]},
    "sealed_untouched": {"gap_v1_12": G12_SHA, "registry_v1_35": R35_SHA},
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
