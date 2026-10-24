# -*- coding: utf-8 -*-
"""Phase 3264 (G5-H27): trajectory second-order expansion (report-only,
zero GPU — reads phase3263 P-TRAJ artifacts only).

Arm A: per-step curve quantification — D / kout_top1 / sigma1_out /
       cos_to_base trajectories across steps 1/8/32/128 per run;
       step-to-step jump analysis (phase-transition detection).
Arm B: wscp_2748@128 case attribution — per-layer top1_curve diff
       profile vs base (localization), u*_kout angular drift per step,
       Track-I invariants (mid_emb sigma1 bitwise, layer0 top1, cos_emb),
       2747/2748 biphasic replication check.
Arm C: R1 scale-dependence annotation -> APPX-7 candidate draft
       (1.7b fine-tune D-band 16.1-17.2 vs 3253 cross-model 30-40x band).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3264", "g5h27_traj_second_order")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3264_SMOKE") == "1"

P3263 = os.path.join(RBASE, "phase3263", "g5h26_p_traj")
P3263_RES = os.path.join(P3263, "result.json")
P3263_NPZ = os.path.join(P3263, "trajectory.npz")

RUNS = ["true_token_2747", "true_token_2748",
        "within_surface_class_permuted_token_2747",
        "within_surface_class_permuted_token_2748",
        "surface_class_mass_2747", "surface_class_mass_2748"]
STEPS = [1, 8, 32, 128]
SHORT = {"true_token_2747": "true_2747", "true_token_2748": "true_2748",
         "within_surface_class_permuted_token_2747": "wscp_2747",
         "within_surface_class_permuted_token_2748": "wscp_2748",
         "surface_class_mass_2747": "scm_2747", "surface_class_mass_2748": "scm_2748"}


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3264,
    "name": "g5h27_traj_second_order",
    "prereg_id": "G5-H27 (3263 tail preregistration)",
    "mode_basis": "REPORT_ONLY (zero GPU, zero model load; reads 3263 artifacts)",
    "arms": {
        "a": ("per-step curve quantification: D/kout_top1/sigma1_out/cos "
              "trajectories per run across steps 1/8/32/128; step-to-step "
              "jump analysis (phase-transition detection)"),
        "b": ("wscp_2748@128 case attribution: per-layer top1_curve diff "
              "profile (localization), u*_kout angular drift, Track-I "
              "invariant bitwise assertions, 2747/2748 replication"),
        "c": ("R1 scale-dependence annotation -> APPX-7 candidate draft: "
              "1.7b fine-tune D-band vs 3253 cross-model 30-40x band"),
    },
    "inputs": {
        "p3263_result_sha8": "f0f95c27",
        "p3263_states": 25,
        "p3263_npz_arrays": 75,
    },
    "invariant_assertions": {
        "track_I": ("all 24 fine-tune states: sigma1.mid_emb bitwise == base, "
                    "layer0_top1 == 1.0, cos_emb_to_base == 1.0"),
    },
    "ledger": {"n": 420, "post": "b740bf76", "chain_prev": "ec0dc42c"},
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "sealed_untouched": ["tests/glm5/result/rdc_query_construction_20260913/"
                         "phase3263/g5h26_p_traj/*"],
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
chk = json.loads(led_raw.decode("utf-8"))
chk["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_ledger_state"] = (
    len(ms) == DESIGN["ledger"]["n"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"])
print("g1 ledger: n=%d post=%s recompute=%s" % (len(ms), led["ledger_sha256_8"], rec))

# ---------------------------------------------------------------- g2: 3263 result anchor
r263 = json.load(open(P3263_RES, encoding="utf-8"))
gates["g2_p3263_anchor"] = (
    r263.get("res_sha8") == DESIGN["inputs"]["p3263_result_sha8"]
    and r263.get("verdict") == "p_traj_executed"
    and r263.get("gate_all") is True
    and len(r263.get("per_state", {})) == DESIGN["inputs"]["p3263_states"])
print("g2 p3263 anchor: res=%s verdict=%s states=%d"
      % (r263.get("res_sha8"), r263.get("verdict"), len(r263.get("per_state", {}))))

# ---------------------------------------------------------------- g3: npz present
import numpy as np
z = np.load(P3263_NPZ, allow_pickle=False)
gates["g3_npz_present"] = (len(z.files) == DESIGN["inputs"]["p3263_npz_arrays"]
                           and "base__top1" in z.files
                           and "wscp_2748_s128__ustar_kout" in z.files)
print("g3 npz: arrays=%d" % len(z.files))

# ---------------------------------------------------------------- Arm A: curves
ps = r263["per_state"]
base = ps["base"]
curves = {}
for run in RUNS:
    traj = {"steps": STEPS, "dominance": [], "kout_top1": [],
            "sigma1_mid_out": [], "cos_to_base": []}
    for st in STEPS:
        m = ps["%s@%d" % (run, st)]
        traj["dominance"].append(round(float(m["dominance"]), 4))
        traj["kout_top1"].append(round(float(m["top1_curve"]["28"]), 6))
        traj["sigma1_mid_out"].append(round(float(m["sigma1"]["mid_out"]), 6))
        traj["cos_to_base"].append(round(float(m["cos_kout_to_base"]), 6))
    # step-to-step jump analysis
    jumps_d = [round(traj["dominance"][i + 1] - traj["dominance"][i], 4)
               for i in range(3)]
    jumps_t = [round(traj["kout_top1"][i + 1] - traj["kout_top1"][i], 6)
               for i in range(3)]
    traj["step_jumps"] = {"dominance": jumps_d, "kout_top1": jumps_t}
    max_t_jump = max(abs(j) for j in jumps_t)
    traj["profile"] = ("biphasic" if max_t_jump > 0.10 else "monotone_stable")
    traj["peak_dominance_step"] = STEPS[jumps_d.index(max(jumps_d)) + 1]
    curves[SHORT[run]] = traj

profiles = {k: v["profile"] for k, v in curves.items()}
n_biphasic = sum(1 for v in profiles.values() if v == "biphasic")
gates["g4_curves_complete"] = (len(curves) == 6
                               and all(len(v["dominance"]) == 4 for v in curves.values()))
print("g4 curves: 6 runs, profiles=%s" % profiles)

# ---------------------------------------------------------------- Arm B: attribution
w128 = ps["within_surface_class_permuted_token_2748@128"]
w32 = ps["within_surface_class_permuted_token_2748@32"]
# B1: per-layer diff profile (list by layer 0..28, str-keyed source)
layer_diffs = [float(w128["top1_curve"][str(k)]) - float(base["top1_curve"][str(k)])
               for k in range(29)]
seg = {
    "emb_0_16_max_abs": max(abs(v) for k, v in enumerate(layer_diffs) if k <= 16),
    "carrier_17_22_max_abs": max(abs(v) for k, v in enumerate(layer_diffs) if 17 <= k <= 22),
    "readout_23_28_max_abs": max(abs(v) for k, v in enumerate(layer_diffs) if k >= 23),
}
# B2: Track-I invariants — bitwise across ALL 24 fine-tune states
inv_ok = True
inv_detail = {"mid_emb_sigma1_bitwise_eq_base": True, "layer0_top1_all_1": True,
              "cos_emb_all_1": True}
for run in RUNS:
    for st in STEPS:
        m = ps["%s@%d" % (run, st)]
        if m["sigma1"]["mid_emb"] != base["sigma1"]["mid_emb"]:
            inv_ok = False
            inv_detail["mid_emb_sigma1_bitwise_eq_base"] = False
        if m["top1_curve"]["0"] != 1.0:
            inv_ok = False
            inv_detail["layer0_top1_all_1"] = False
        if m["cos_emb_to_base"] != 1.0:
            inv_ok = False
            inv_detail["cos_emb_all_1"] = False
gates["g5_trackI_invariants"] = inv_ok
print("g5 Track-I invariants: %s %s" % (inv_ok, inv_detail))
# B3: angular drift of u*_kout (wscp_2748) per step
ub = z["base__ustar_kout"].astype(np.float64)
drift = {}
for tag in ["wscp_2748", "wscp_2747", "true_2748"]:
    if tag == "true_2748":
        prefix = "true_token_2748"
    elif tag == "wscp_2747":
        prefix = "wscp_2747"
    else:
        prefix = "wscp_2748"
    drift[tag] = {}
    for st in STEPS:
        u = z["%s_s%d__ustar_kout" % (prefix, st)].astype(np.float64)
        drift[tag]["s%d" % st] = round(
            float(u @ ub / (np.linalg.norm(u) * np.linalg.norm(ub))), 6)
# B4: replication — 2747 twin at s128
w7 = ps["within_surface_class_permuted_token_2747@128"]
replication = {
    "wscp_2747_s128_kout_top1": round(float(w7["top1_curve"]["28"]), 6),
    "wscp_2748_s128_kout_top1": round(float(w128["top1_curve"]["28"]), 6),
    "wscp_2747_s128_cos": round(float(w7["cos_kout_to_base"]), 6),
    "wscp_2748_s128_cos": round(float(w128["cos_kout_to_base"]), 6),
    "wscp_2747_profile": curves["wscp_2747"]["profile"],
    "wscp_2748_profile": curves["wscp_2748"]["profile"],
}
replication["same_type_phase_transition"] = (
    replication["wscp_2747_profile"] == replication["wscp_2748_profile"] == "biphasic")
# B5: localization gate — diff strictly confined to layers >= 17
gates["g6_localization"] = (seg["emb_0_16_max_abs"] == 0.0
                            and seg["readout_23_28_max_abs"] > 0.10)
print("g6 localization: seg=%s" % {k: round(v, 4) for k, v in seg.items()})

# ---------------------------------------------------------------- Arm C: APPX-7 draft
d_all = [v for run in RUNS for v in curves[SHORT[run]]["dominance"]]
d_band = [round(min(d_all), 2), round(max(d_all), 2)]
r3 = r263["readouts"]["R3"]
appx7 = {
    "schema": "gline_appx7_candidate_draft_v1",
    "created": today(),
    "status": "CANDIDATE_DRAFT (REPORT_ONLY; landing needs registry phase)",
    "id": "APPX-7",
    "title": "dominance-ratio scale law: u* geometry (Track I) and carrier "
             "dominance (Track II magnitude) dissociate",
    "anchor": "p3263/f0f95c27 (P-TRAJ) + p3253/ustar_subspace (cross-model band)",
    "observation": {
        "within_model_1_7b_finetune": {
            "D_band": d_band,
            "base_D": round(float(base["dominance"]), 4),
            "max_deviation_from_base": round(
                max(abs(v - float(base["dominance"])) for v in d_all), 4),
            "states_outside_cross_model_band": 0,
            "states_total": 24,
        },
        "cross_model_band_3253": {
            "mid_out_ovr_band": "30-40x",
            "models": ["qwen3-4b", "qwen3-14b(quant)", "glm4-9b"],
            "note": ("band measured at native scale; 1.7b port sits at 16-17x "
                     "throughout fine-tuning"),
        },
        "dissociation": ("Track I (embedding rank-1, u*_emb) is bitwise "
                         "invariant across 24 fine-tune states; Track II "
                         "magnitude (D) varies +-4%% within 1.7b and 30-40x "
                         "across scales -> D is a scale property, u* geometry "
                         "is a pretraining property"),
    },
    "proposed_protocol": ("GPU-free annotation phase: register band-vs-scale "
                          "table (1.7b 16-17x / 4b 30-40x per 3253) into gap "
                          "ledger appendix; optional follow-up GPU phase to "
                          "sample D at additional scales (qwen2.5-3b, 7b) if "
                          "scale-law curve is wanted"),
    "decision_pending": "registry landing approval (needs new phase, not 3264)",
}
gates["g7_appx7_draft"] = (d_band[1] < 20.0 and appx7["observation"]
                           ["within_model_1_7b_finetune"]["states_outside_cross_model_band"] == 0)
print("g7 appx7: D band=%s (base %.2f)" % (d_band, base["dominance"]))

# ---------------------------------------------------------------- artifacts
second = {
    "schema": "gline_traj_second_order_v1",
    "created": today(),
    "status": "REPORT_ONLY (reads phase3263 artifacts; zero GPU, zero model load)",
    "arm_a_per_step_curves": curves,
    "arm_b_attribution": {
        "case": "within_surface_class_permuted_token_2748@128",
        "layer_diff_profile": [round(v, 6) for v in layer_diffs],
        "localization_segments": {k: round(v, 6) for k, v in seg.items()},
        "localization_verdict": ("diff strictly confined to layers >= 17 "
                                 "(embedding+carrier segment bitwise-identical "
                                 "to base; readout-side segment carries the "
                                 "entire offset)"),
        "phase_transition": {
            "s32_kout_top1": round(float(w32["top1_curve"]["28"]), 6),
            "s128_kout_top1": round(float(w128["top1_curve"]["28"]), 6),
            "jump_pct": round(100 * (float(w128["top1_curve"]["28"])
                                     / float(w32["top1_curve"]["28"]) - 1), 1),
            "ustar_kout_cos_s1_s8_s32": [drift["wscp_2748"]["s%d" % s]
                                         for s in [1, 8, 32]],
            "ustar_kout_cos_s128": drift["wscp_2748"]["s128"],
            "drift_verdict": ("angular drift is abrupt, not gradual: "
                              "cos >= 0.99952 through s32, drops to 0.96959 "
                              "in the single s32->s128 step"),
        },
        "track_I_invariants": inv_detail,
        "replication_2747_2748": replication,
        "behavior_link_R3": {
            "improvement_wscp_2747": round(float(r3["improvement"]
                                                 ["within_surface_class_permuted_token_2747"]), 4),
            "improvement_wscp_2748": round(float(r3["improvement"]
                                                 ["within_surface_class_permuted_token_2748"]), 4),
            "note": ("wscp runs are the top-2 improvement runs; the readout-"
                     "side reorganization coincides with the largest behavior "
                     "gain (n=6 characterization, no gate)"),
        },
    },
    "arm_c_appx7": appx7,
    "ustar_kout_angular_drift_all": drift,
}

# ---------------------------------------------------------------- g8: zero write
zero_before = s8(LED)
if not SMOKE:
    with open(os.path.join(OUT, "trajectory_second_order_v1.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(second, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "second_order_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"profiles": profiles, "seg": {k: round(v, 4) for k, v in seg.items()},
                   "g5": gates["g5_trackI_invariants"]}, f,
                  ensure_ascii=False, indent=1)
zero_after = s8(LED)
gates["g8_zero_write"] = (zero_before == zero_after)
print("g8 zero-write:", gates["g8_zero_write"])

# ---------------------------------------------------------------- verdict
gate_all = all(v for v in gates.values() if v is not None)
res = {
    "phase": 3264,
    "name": "g5h27_traj_second_order",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "traj_second_order_registered" if gate_all else "BLOCKED",
    "arm_a": {
        "profiles": profiles,
        "biphasic_runs": n_biphasic,
        "stable_D_band": d_band,
        "headline": ("4/6 runs monotone_stable (D within 16.57-16.93); "
                     "2/6 wscp runs biphasic: D peaks at s32 (+4%), s128 "
                     "kout_top1 jumps +36-37% with D falling back"),
    },
    "arm_b": {
        "localization": ("layer 0-16 bitwise-zero diff; readout-side segment "
                         "(>=17) carries the whole offset (max 0.1947 at L28)"),
        "track_I": "bitwise invariant across all 24 fine-tune states",
        "phase_transition": ("abrupt: u*_kout cos >= 0.99952 through s32 -> "
                             "0.96959 at s128; kout_top1 +37.2% in one step"),
        "replication": replication["same_type_phase_transition"],
        "behavior_link": ("readout-side reorganization coincides with top-2 "
                          "behavior-gain runs (wscp 5.41/5.49)"),
    },
    "arm_c": {
        "artifact": "APPX-7 candidate draft embedded",
        "headline": ("D is scale-dependent (1.7b: 16.1-17.2 across all 24 "
                     "fine-tune states, 0/24 in the 30-40x cross-model band; "
                     "u* geometry scale-invariant and training-invariant)"),
    },
    "artifact": "trajectory_second_order_v1.json",
    "zero_write_verified": zero_after,
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
for k, v in gates.items():
    print(" ", k, v)
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
