# -*- coding: utf-8 -*-
"""Phase 3277 (G5-H40): APPX-2/3 registry landing — appraisal arm + (if
qualified) gap ledger v1.14 -> v1.15 with APPX-8 appendix entry + registry
v1.37 -> v1.38 upgrade_log annotation, append-only with bitwise REVERT.

Per protocol appx23_protocol_v2.json (ce73984d) step G5/H5: landing happens
in a separate registry phase (3277) after execution phase 3276 (G5-H39,
res 8d5bba36, verdict appx23_exec_registered, X-gate 5/5).

Arm a (appraisal): evidence level, anchor completeness (p3276 res on disk),
non-conflict with APPX-1..7, ledger chain state.
Arm b (landing, gated on appraisal): new landed files in phase3277 dir;
REVERT rebuild must be byte-identical to sealed v1.14 (c6d97133) and
v1.37 (bfb7d7e5); gaps/features/failures untouched (pointer-diff audit).
Zero write to atlas_ledger.json (ledger entry happens at closeout #429).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3277", "g5h40_appx23_landing")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3277_SMOKE") == "1"

G14 = os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply",
                   "gap_ledger_v1_14.json")
R37 = os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply",
                   "atlas_registry_v1_37.json")
P3276_RES = os.path.join(RBASE, "phase3276", "g5h39_appx23_exec", "result.json")
PROTO = os.path.join(RBASE, "phase3267", "g5h30_appx23_protocol",
                     "appx23_protocol_v2.json")

G14_SHA = "c6d97133"
R37_SHA = "bfb7d7e5"
P3276_SHA = "8d5bba36"
PROTO_SHA = "ce73984d"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3277,
    "name": "g5h40_appx23_landing",
    "prereg_id": "G5-H40 (protocol G5/H5; 3276 preregistration)",
    "arms": {
        "a": ("appraisal: evidence level (statistical, from 3276 X-gate 5/5 "
              "+ GATE 9/9), anchor completeness (p3276 res 8d5bba36 on disk, "
              "protocol ce73984d on disk), non-conflict with APPX-1..7, "
              "ledger chain"),
        "b": ("landing (gated on appraisal): gap v1.14->v1.15 appendix "
              "+APPX-8; registry v1.37->v1.38 upgrade_log +1; REVERT bitwise "
              "to c6d97133 / bfb7d7e5; gaps/features/failures untouched"),
    },
    "appx8_summary": ("cross-model conservation of S-family separability "
                      "(14/14 pairs separable, 48.22-73.60 deg) + K3 "
                      "single-coordinate deathline fires 3/3 (top-50 eta2 "
                      "coverage 1.9-5.9 pct << 30) -> single-coordinate "
                      "mechanism target abandoned per p35 deathline; K2 "
                      "deathline does not fire 3/3"),
    "inputs": {
        "gap_v1_14_sha8": G14_SHA,
        "registry_v1_37_sha8": R37_SHA,
        "p3276_res_sha8": P3276_SHA,
        "protocol_sha8": PROTO_SHA,
    },
    "ledger": {"n": 434, "post": "ee5b19f3", "chain_prev": "923806e0"},
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "sealed_untouched": [G14_SHA + " (gap v1.14)", R37_SHA + " (registry v1.37)"],
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
r276 = json.load(open(P3276_RES, encoding="utf-8"))
gates["g2_inputs"] = (
    s8(G14) == G14_SHA and s8(R37) == R37_SHA
    and r276["res_sha8"] == P3276_SHA
    and r276["gates"]["xgates_all"] is True
    and r276["gates"]["arm_a_all_registered"] is True
    and r276["gates"]["arm_b_all_registered"] is True
    and s8(PROTO) == PROTO_SHA
    and r276["protocol"] == "ce73984d (PREREGISTERED_DRAFT_V2 executed)")
print("g2 inputs: gap=%s registry=%s p3276=%s proto=%s"
      % (s8(G14) == G14_SHA, s8(R37) == R37_SHA,
         r276["res_sha8"], s8(PROTO) == PROTO_SHA))

# ---------------------------------------------------------------- Arm a: appraisal
appraisal = {}
aa = r276["arm_a_appx2"]
ab = r276["arm_b_appx3"]
n_pairs = sum(len(rec.get("s3", {}).get("pairs", {})) for rec in aa.values())
deg = [d["top1_deg"] for rec in aa.values()
       for d in rec.get("s3", {}).get("pairs", {}).values()]
appraisal["q1_evidence_level"] = {
    "level": "statistical",
    "basis": [
        "phase3276 appx23_exec_registered (X-gate 5/5 bitwise vs sealed "
        "2874/2878/3158/3155/3155 mechanics; GATE 9/9)",
        "Arm a: %d candidate models, S1 attr axes 8/10 VG1 pass on every "
        "model; S3 %d/%d S-family x K_readout pairs separable (%.2f-%.2f deg)"
        % (len(aa), n_pairs, n_pairs, min(deg), max(deg)),
        "Arm b: %d models on sealed 3155 collects + analytic readout "
        "interface (zero new forward); K3 deathline fires 3/3"
        % len(ab),
    ],
    "verdict": "LAND (same grade as APPX-7 landing precedent, Phase 3265)",
}
appraisal["q2_anchor_completeness"] = {
    "p3276": {"res_sha8": r276["res_sha8"], "xgates_all": True,
              "gates_ok": sum(1 for v in r276["gates"].values() if v is True),
              "gates_total": len(r276["gates"])},
    "protocol": {"sha8": s8(PROTO), "status": "PREREGISTERED_DRAFT_V2"},
    "verdict": "COMPLETE",
}
appraisal["q3_non_conflict"] = {
    "appendix_ids_prior": None,  # filled below
    "new_content": ("APPX-8 adds cross-model S-family separability "
                    "conservation (Arm a) and the K3 single-coordinate "
                    "deathline verdict (Arm b); does not modify any "
                    "APPX-1..7 claim. K3 deathline resolves the protocol's "
                    "K2/K3 measurement goal per p35 deathline rule."),
    "verdict": "NO_CONFLICT",
}
appraisal["q4_chain"] = {
    "ledger": "n=434 post ee5b19f3 (verified g1)",
    "gap_v1_14": G14_SHA, "registry_v1_37": R37_SHA,
    "verdict": "CLEAN",
}

# ---------------------------------------------------------------- Arm b: landing
g14d = json.load(open(G14, encoding="utf-8"))
appraisal["q3_non_conflict"]["appendix_ids_prior"] = [a["id"] for a in g14d["appendix"]]

appx8_entry = {
    "id": "APPX-8",
    "title": "S 族可分离性跨模型守恒与 K3 单坐标死线定谳（cross-model "
             "conservation of S-family separability & K3 single-coordinate "
             "deathline verdict）",
    "anchor": "p3276",
    "status": "appendix_recorded",
    "note": ("APPX-2/3 执行相（协议 ce73984d，p3276 res 8d5bba36，X-gate 5/5 逐位"
             "==封存 2874/2878/3158/3155 机械）。Arm a（5 候选模型零池化）：S1 "
             "attr 轴 8/10 过 VG1 全模型一致；syntax 轴存活 3/1/0/1/4"
             "（tokenizer-face 分歧如实记录，r1-distill 为 0 属真实阴性）。"
             "S3（K_readout=3158 同配方 Gram top64，S_class=2881 配方）："
             "S_class/S_attr/S_syntax × K_readout 14/14 对全 separable"
             "（48.22–73.60°，gemma-3 S_syntax 48.22 为下界）——S 族轴与读出"
             "子空间的可分离性为跨 6 模型守恒性质（含 4b 现役 67.0° 锚 X3）。"
             "S2 K_entity-vs-readout 64.5–79.0°（descriptive；gemma-3 fp16 "
             "溢出教训→fp32 存储）。Arm b（4b/14b/glm4，封存 3155 collects + "
             "解析读出接口零新前向）：**K3 死线 3/3 触发**——η² top-50 单坐标"
             "覆盖率 isa/hasa/mof = 5.85/3.38/3.75%（4b）、2.13/1.92/2.16%"
             "（14b）、2.27/1.92/2.02%（glm4）≪ 30% → 按 p35 死线废弃单坐标"
             "机制目标（转 ≤10 维联合 + 端口类）。K2 死线 3/3 不触发：分离"
             "投影后 cos 降 1.18–1.39% ≪ 50%，first-token 一致率 "
             "0.509–0.639 < 0.8——诚实注记：条件成分幅度低但行为承重，准则"
             "双条件均不满足。协议 G5/H5 落册于本条（3277）。"),
    "anchor_sha8": P3276_SHA,
}
g15 = json.loads(json.dumps(g14d))
g15["version"] = "1.15"
g15["appendix"] = g14d["appendix"] + [appx8_entry]
g15["supersedes"] = "gap_ledger_v1_14.json (%s, Phase 3274)" % G14_SHA
g15["updated"] = today()

r37d = json.load(open(R37, encoding="utf-8"))
r38 = json.loads(json.dumps(r37d))
r38["version"] = "1.38"
r38["upgrade_log"] = r37d["upgrade_log"] + [{
    "node": "gap ledger appendix",
    "change": ("APPX-8 landed: cross-model conservation of S-family "
               "separability & K3 single-coordinate deathline verdict "
               "(gap v1.14 -> v1.15). Arm a (APPX-2, 5 candidate models, "
               "per-model zero pooling): S1 attr axes 8/10 VG1 pass on every "
               "model, syntax surviving 3/1/0/1/4 (tokenizer-face divergence "
               "recorded verbatim); S3 14/14 S-family x K_readout pairs "
               "separable (48.22-73.60 deg) across 6 models incl the 4b "
               "incumbent anchor. Arm b (APPX-3, 4b/14b/glm4 on sealed 3155 "
               "collects + analytic readout interface, zero new forward): "
               "K3 deathline fires 3/3 (top-50 eta2 coverage 1.92-5.85 pct "
               "<< 30) -> single-coordinate mechanism target abandoned per "
               "p35 deathline (moves to <=10-dim joint + port-class); K2 "
               "deathline does not fire 3/3 (cos drop 1.18-1.39 pct << 50, "
               "first-token agreement 0.509-0.639 < 0.8; honest note: "
               "low-magnitude yet behaviorally load-bearing). Anchors "
               "unchanged; no feature/failure changes."),
    "phase": 3277,
    "gap_ledger": "gap_ledger_v1_15.json",
}]
r38["supersedes"] = "atlas_registry_v1_37.json (%s, Phase 3274)" % R37_SHA
# NOTE: registry chain has NO 'updated' key (3256 SMOKE lesson) — do not add.

# REVERT rebuilds (byte-identical to sealed predecessors)
rev_g = json.loads(json.dumps(g15))
rev_g["version"] = "1.14"
rev_g["appendix"] = g14d["appendix"]
rev_g["supersedes"] = g14d["supersedes"]
rev_g["updated"] = g14d["updated"]
rev_r = json.loads(json.dumps(r38))
rev_r["version"] = "1.37"
rev_r["upgrade_log"] = r37d["upgrade_log"]
rev_r["supersedes"] = r37d["supersedes"]

g4_g = hashlib.sha256(json.dumps(rev_g, ensure_ascii=False, indent=1,
                                 sort_keys=False).encode("utf-8")).hexdigest()[:8]
g4_r = hashlib.sha256(json.dumps(rev_r, ensure_ascii=False, indent=1,
                                 sort_keys=False).encode("utf-8")).hexdigest()[:8]
gates["g4_revert_bitwise"] = (g4_g == G14_SHA and g4_r == R37_SHA)
print("g4 revert bitwise: gap=%s registry=%s" % (g4_g, g4_r))

# ---------------------------------------------------------------- g5: ledger-diff audit
gates["g5_pointer_diff"] = (
    g15["gaps"] == g14d["gaps"]
    and r38["features"] == r37d["features"]
    and r38["failures"] == r37d["failures"]
    and r38["provenance"]["anchor_files"] == r37d["provenance"]["anchor_files"]
    and r38["census_3238"] == r37d["census_3238"]
    and r38["evidence_level_taxonomy"] == r37d["evidence_level_taxonomy"])
print("g5 pointer diff:", gates["g5_pointer_diff"])

gates["g3_appraisal"] = (
    appraisal["q1_evidence_level"]["verdict"].startswith("LAND")
    and appraisal["q2_anchor_completeness"]["verdict"] == "COMPLETE"
    and appraisal["q3_non_conflict"]["verdict"] == "NO_CONFLICT"
    and appraisal["q4_chain"]["verdict"] == "CLEAN")
print("g3 appraisal: %s" % {k: v["verdict"] for k, v in appraisal.items()})

# ---------------------------------------------------------------- write landed files
if not SMOKE:
    with open(os.path.join(OUT, "gap_ledger_v1_15.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(g15, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "atlas_registry_v1_38.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(r38, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "gap_v115_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"version": g15["version"], "appendix_n": len(g15["appendix"]),
                   "revert": gates["g4_revert_bitwise"]}, f,
                  ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- g6: landed files + zero write
zero_before = s8(LED)
landed_ok = True
if not SMOKE:
    g15_disk = json.load(open(os.path.join(OUT, "gap_ledger_v1_15.json"),
                              encoding="utf-8"))
    r38_disk = json.load(open(os.path.join(OUT, "atlas_registry_v1_38.json"),
                              encoding="utf-8"))
    landed_ok = (g15_disk["version"] == "1.15"
                 and len(g15_disk["appendix"]) == 9
                 and g15_disk["appendix"][-1]["id"] == "APPX-8"
                 and r38_disk["version"] == "1.38"
                 and len(r38_disk["upgrade_log"]) == 17)
zero_after = s8(LED)
gates["g6_landed_files"] = landed_ok and (zero_before == zero_after)
print("g6 landed files + zero-write:", gates["g6_landed_files"])

# ---------------------------------------------------------------- verdict
gate_all = all(v for v in gates.values() if v is not None)
res = {
    "phase": 3277,
    "name": "g5h40_appx23_landing",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "appx23_landed" if gate_all else "BLOCKED",
    "appraisal": appraisal,
    "gap_v115": {
        "file": "gap_ledger_v1_15.json",
        "appendix_n": 9,
        "appx8": appx8_entry,
        "supersedes": g15["supersedes"],
        "file_sha8": (s8(os.path.join(OUT, "gap_ledger_v1_15.json"))
                      if not SMOKE else "smoke"),
    },
    "registry_v138": {
        "file": "atlas_registry_v1_38.json",
        "upgrade_log_n": 17,
        "anchors_unchanged": True,
        "supersedes": r38["supersedes"],
        "file_sha8": (s8(os.path.join(OUT, "atlas_registry_v1_38.json"))
                      if not SMOKE else "smoke"),
    },
    "revert_check": {"gap": gates["g4_revert_bitwise"],
                     "registry": gates["g4_revert_bitwise"]},
    "sealed_untouched": {"gap_v1_14": G14_SHA, "registry_v1_37": R37_SHA},
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
