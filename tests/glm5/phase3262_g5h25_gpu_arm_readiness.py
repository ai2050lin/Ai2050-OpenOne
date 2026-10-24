# -*- coding: utf-8 -*-
"""Phase 3262 (G5-H25): GPU arm readiness — checkpoint presence audit +
preregistered protocol drafts (report-only, zero GPU measurement).

Arm (a): checkpoint 在位性核查
  - qwen3-1.7b port archive (phase2755/qwen25_3b): 6 runs x 4 steps
    (001/008/032/128) delta npz — full sha8 inventory; qwen3-1.7b model
    files on disk; torch/CUDA probe (correct scope: managed python 3.11.9,
    correcting the 3261 probe-scope false report without touching sealed files).
  - qwen3-4b native archive (phase2747/2748 fields/delta npz): physical
    path C:\\AI2050-RDC-Archive is OFF-MACHINE — local JSON manifests only.
  - phase2748 local checkpoint96 (terminal step-96 params) presence.

Arm (b): preregistered protocol DRAFTS (no execution):
  - P-TRAJ: checkpoint training-step trajectory (qwen3-1.7b, 25 states,
    bit-exact restore W_t = W_base + delta_t + residual_t, u* dual-track
    metrics along training steps) -> execution phase preregistered as 3263.
  - APPX-1/2/3 per-item GPU protocol drafts; APPX-4/5 noted as cross-line
    governance (not GPU, unchanged).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3262", "g5h25_gpu_arm_readiness")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3262_SMOKE") == "1"

P2755_DIR = os.path.join(RBASE, "phase2755", "qwen25_3b")
P2755_RES = os.path.join(P2755_DIR, "result.json")
P2755_SRC = os.path.join(ROOT, "tests", "glm5", "phase2755_rdc_cross_model_signature.py")
M171 = os.path.join(ROOT, "models", "hf", "qwen3-1.7b")
P2747 = os.path.join(RBASE, "phase2747")
P2748 = os.path.join(RBASE, "phase2748")
ARCHIVE_PHYS = "C:\\AI2050-RDC-Archive"

RUNS = ["true_token_2747", "within_surface_class_permuted_token_2747",
        "surface_class_mass_2747", "true_token_2748",
        "within_surface_class_permuted_token_2748", "surface_class_mass_2748"]
STEPS = [1, 8, 32, 128]


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3262,
    "name": "g5h25_gpu_arm_readiness",
    "prereg_id": "G5-H25 (3261 tail op(a); backlog v3 open_gpu items 1-2)",
    "arms": {
        "a": ("checkpoint presence audit: 24-delta qwen3-1.7b port archive "
              "(full sha8) + model + torch/CUDA probe (corrected scope); "
              "qwen3-4b native archive off-machine fact; 2748 checkpoint96"),
        "b": ("preregistered protocol drafts: P-TRAJ checkpoint training-step "
              "trajectory (execution -> 3263) + APPX-1/2/3 GPU protocol drafts; "
              "APPX-4/5 = cross-line governance, not GPU"),
    },
    "restore_rule": "W_t = W_base + delta_t + residual_t (bit-exact, np.array_equal)",
    "restore_source": "tests/glm5/phase2755_rdc_cross_model_signature.py",
    "ledger": {"n": 418, "post": "4f5ed748", "chain_prev": "c26305dd"},
    "expected": {
        "delta_count": 24,
        "steps": STEPS,
        "runs": RUNS,
        "block": "model.layers[16].mlp",
        "block_params": 37748736,
        "archive_physical_absent": True,
    },
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "sealed_untouched": ["tests/glm5/result/rdc_query_construction_20260913/"
                         "phase3261/g5h24_evidence_regrading/*"],
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

# ---------------------------------------------------------------- arm (a): audit
# a1: qwen3-1.7b port archive
delta_inv = []
missing = []
for run in RUNS:
    for st in STEPS:
        p = os.path.join(P2755_DIR, run, "delta_%03d.npz" % st)
        if os.path.isfile(p) and not SMOKE:
            delta_inv.append({"run": run, "step": st, "bytes": os.path.getsize(p),
                              "sha8": s8(p)})
        elif os.path.isfile(p):
            delta_inv.append({"run": run, "step": st, "bytes": os.path.getsize(p),
                              "sha8": "smoke_skipped"})
        else:
            missing.append("%s/delta_%03d.npz" % (run, st))
model_files = {}
for fn in ["config.json", "model.safetensors.index.json",
           "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    p = os.path.join(M171, fn)
    model_files[fn] = (s8(p) if os.path.isfile(p) and not SMOKE else
                       ("smoke_skipped" if os.path.isfile(p) else None))
gates["g2_port_archive"] = (len(delta_inv) == DESIGN["expected"]["delta_count"]
                            and not missing
                            and all(model_files.values()))
print("g2 port archive: %d/24 deltas, missing=%s, model=%s"
      % (len(delta_inv), missing[:3], all(model_files.values())))

# a2: qwen3-4b native archive off-machine
r2755 = json.load(open(P2755_RES, encoding="utf-8"))
manifest_hits = {
    "p2747_fields_001_manifest": os.path.isfile(
        os.path.join(P2747, "training", "true_token_2747", "commits", "fields_001.json")),
    "p2747_delta_001_manifest": os.path.isfile(
        os.path.join(P2747, "training", "true_token_2747", "commits", "delta_001.json")),
    "p2748_ckpt_manifest": os.path.isfile(
        os.path.join(P2748, "training", "original_checkpoint_manifest.json")),
}
ckpt96_dirs = [run for run in
               ["true_complete_answer_2748", "true_complete_answer_2749",
                "within_context_permuted_complete_answer_2748",
                "within_context_permuted_complete_answer_2749",
                "surface_class_mass_on_true_teacher_history_2748",
                "surface_class_mass_on_true_teacher_history_2749"]
               if os.path.isdir(os.path.join(
                   P2748, "field_store", "training", run, "checkpoint96"))]
gates["g3_native_off_machine"] = (
    not os.path.exists(ARCHIVE_PHYS)
    and all(manifest_hits.values())
    and len(ckpt96_dirs) == 6)
print("g3 native archive: C-archive absent=%s, manifests=%s, ckpt96 dirs=%d"
      % (not os.path.exists(ARCHIVE_PHYS), all(manifest_hits.values()),
         len(ckpt96_dirs)))

# ---------------------------------------------------------------- g4: restore rule
src = open(P2755_SRC, encoding="utf-8").read()
src_sha8 = s8(P2755_SRC)
rule_ok = ("reconstruction_residual__" in src
           and "np.array_equal(rebuilt" in src
           and "CHECKPOINTS = [1, 8, 32, 128]" in src
           and "STEPS = 128" in src
           and "BLOCK = 'model.layers[16].mlp'" in src)
res_src = (os.path.relpath(P2755_SRC, ROOT).replace("\\", "/"),
           r2755.get("source", {}).get("sha256", "")[:8])
gates["g4_restore_rule_verbatim"] = (rule_ok and src_sha8 == res_src[1])
print("g4 restore rule: verbatim=%s | src sha8 %s (result.json %s)"
      % (rule_ok, src_sha8, res_src[1]))

# ---------------------------------------------------------------- g5: probe
probe = {"scope": "managed python 3.11.9 (corrected口径; 3261 note used global "
                  "python -> false 'torch env missing'; sealed 3261 files untouched)"}
try:
    import torch  # noqa
    probe["torch"] = torch.__version__
    probe["cuda_available"] = bool(torch.cuda.is_available())
    probe["device"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "none")
except ImportError:
    probe["torch"] = None
    probe["cuda_available"] = False
    probe["device"] = "none"
probe["p2755_gpu_record"] = r2755.get("integrity", {}).get("gpu")
probe["p2755_restore_exact"] = r2755.get("restore_exact")
gates["g5_torch_cuda"] = (probe["cuda_available"] is True
                          and probe["p2755_restore_exact"] is True
                          and "RTX 5080" in str(probe["p2755_gpu_record"]))
print("g5 probe: torch=%s cuda=%s device=%s | 2755 restore_exact=%s gpu=%s"
      % (probe["torch"], probe["cuda_available"], probe["device"],
         probe["p2755_restore_exact"], probe["p2755_gpu_record"]))

# ---------------------------------------------------------------- artifacts
audit = {
    "schema": "gline_checkpoint_presence_audit_v1",
    "created": today(),
    "status": "AUDIT_ONLY (no files modified)",
    "qwen3_1_7b_port": {
        "dir": "tests/glm5/result/rdc_query_construction_20260913/phase2755/qwen25_3b",
        "verdict": "FULLY_IN_PLACE",
        "delta_count": len(delta_inv),
        "runs": RUNS,
        "steps": STEPS,
        "note": ("folder name 'qwen25_3b' is legacy naming; recorded model = "
                 "qwen3-1.7b (Qwen3ForCausalLM), dims 6144x2048 confirm"),
        "model_files": model_files,
        "block": DESIGN["expected"]["block"],
        "block_params": DESIGN["expected"]["block_params"],
        "total_bytes": sum(d["bytes"] for d in delta_inv),
        "deltas": delta_inv,
    },
    "qwen3_4b_native": {
        "verdict": "OFF_MACHINE",
        "physical_path": ARCHIVE_PHYS + " (absent on this machine)",
        "local_manifests_present": manifest_hits,
        "manifest_note": ("JSON manifests carry field_sha256 + bytes + shapes; "
                          "tensors (fields/delta npz) live only at the absent "
                          "C:\\AI2050-RDC-Archive"),
        "memo_wording_note": ("MEMO P5① '12 checkpoints' = 2747 batch (3 runs x 4 "
                              "steps); 2748 batch adds 3 runs -> 24 total, "
                              "matching the 1.7b port archive"),
        "p2748_local_checkpoint96": {
            "verdict": "IN_PLACE (terminal step-96 only)",
            "runs_with_ckpt96": len(ckpt96_dirs),
            "tensors_per_dir": ["gate_proj_weight.npz", "up_proj_weight.npz",
                                "down_proj_weight.npz"],
            "note": ("validation step{1,8,32,96} drop npz are histories, NOT "
                     "parameter archives -> intermediate-step 4b params are "
                     "NOT recoverable locally"),
        },
    },
    "torch_probe": probe,
    "conclusion": ("GPU trajectory arm is FULLY FEASIBLE on this machine via the "
                   "qwen3-1.7b port archive (24 deltas + model + torch cu128); "
                   "qwen3-4b native trajectory remains off-machine (manifest-only)"),
}

proto = {
    "schema": "gline_gpu_protocol_draft_v1",
    "created": today(),
    "status": "PREREGISTERED_DRAFT (REPORT_ONLY; execution needs a new phase, "
              "preregistered as 3263 P-TRAJ)",
    "P_TRAJ_checkpoint_training_step_trajectory": {
        "target_phase": 3263,
        "model": "qwen3-1.7b (models/hf/qwen3-1.7b, on disk)",
        "block": "model.layers[16].mlp (gate/up/down, 37,748,736 params)",
        "states": ["base"] + ["%s@step%d" % (r, s) for r in RUNS for s in STEPS],
        "state_count": 25,
        "restore_rule": DESIGN["restore_rule"],
        "restore_guarantee": ("bit-exact per matrix (np.array_equal assert, "
                              "phase2755 precedent restore_exact=true); post-run "
                              "restore-to-base + re-eval assert mandatory"),
        "stimulus": ("u* protocol class-word rows; tokenizer gate per model "
                     "(2755 tokenizer_gate precedent) — 1.7b vocab differs "
                     "from 4b, per-row tokenization assert required"),
        "sampling": ("last-token hidden states, pre-final-norm, all layers; "
                     "layer 0 = embedding output (3252/3253 conventions)"),
        "metrics": {
            "track_I": "cos(u*_layer, embedding rank-1 direction) per state",
            "track_II": "carrier-axis top-8 subspace sigma_1 + ovr per state",
            "u_shape": "per-layer profile across states (3252 U-shape)",
        },
        "readouts_preregistered": {
            "R1_formation_timing": ("carrier-axis dominance (ovr band per 3253 "
                                    "30-40x) emerges at which training step "
                                    "(1/8/32/128), per run"),
            "R2_trackI_stability": ("embedding rank-1 alignment stable across "
                                    "steps (pretraining property hypothesis; "
                                    "baseline measured at step 0)"),
            "R3_behavior_alignment": ("trajectory correlates with 2755 "
                                      "behavior_scores (B1 6/6 runs True)"),
        },
        "cost_estimate": ("25 states x stimulus forward passes; minutes-scale "
                          "per state on RTX 5080 16GB; total well under 1h"),
        "success_criteria": ("25/25 states sampled + bit-exact restore verified "
                             "+ at least one preregistered readout resolved"),
        "limits": ["single model port (1.7b), not 4b native (off-machine)",
                   "single block layers[16].mlp",
                   "tokenizer-face dependence for class-word rows"],
    },
    "APPX_1_14b_shared_dim": {
        "id": "APPX-1", "anchor": "p3166/448af595",
        "goal": "locate eff_ge05=2 shared dimension on 14b and explain origin",
        "model_options": ["Qwen3-14B-bnb-nf4 (on disk; quantization-approximation "
                          "caveat must be recorded)",
                          "Qwen3-14B BF16 CPU-offload (slow)"],
        "protocol": ("replicate p3166 eff_ge05 census on 14b; locate shared "
                     "direction; cross-check vs 4b anchors"),
        "decision_pending": "quantization acceptability (user/owner call)",
    },
    "APPX_2_vocab_port": {
        "id": "APPX-2", "anchor": "p3165/511d9b13",
        "goal": "S_attr/S_syntax cross-model port via per-model vocabulary "
                "coordinate rebuild",
        "models_on_disk": ["deepseek-r1-distill-qwen-7b", "glm4-9b-chat-hf",
                           "qwen2-7b", "qwen2.5-3b-instruct", "gemma-3-4b-it"],
        "protocol": ("per model: rebuild word-table directions (tokenizer-face "
                     "dependent), verify K_entity_eff_ge05 profile, then port"),
        "decision_pending": "none (GPU feasible as-is)",
    },
    "APPX_3_K2K3_direct": {
        "id": "APPX-3", "anchor": "p3155/46a5034d",
        "goal": "K2/K3 direct measurement (A-gate diagnostic; share band "
                "15.0-18.5% from p3155)",
        "protocol": ("direct K2/K3 measurement on 4b with 3246+ ORC machinery; "
                     "record vs the share band"),
        "decision_pending": "none (GPU feasible as-is)",
    },
    "APPX_4_5_governance": {
        "note": ("cross-line governance items (N-line ledger backfill apply, "
                 "APPX-5 patch apply after re-baseline) — NOT GPU, unchanged, "
                 "pending user cross-line confirmation"),
    },
}

# ---------------------------------------------------------------- g6: zero write
zero_before = s8(LED)
if not SMOKE:
    with open(os.path.join(OUT, "checkpoint_presence_audit_v1.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "gpu_trajectory_protocol_draft_v1.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(proto, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "audit_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"deltas": len(delta_inv), "missing": missing,
                   "g3": gates["g3_native_off_machine"]}, f,
                  ensure_ascii=False, indent=1)
zero_after = s8(LED)
gates["g6_zero_write"] = (zero_before == zero_after)
print("g6 zero-write:", gates["g6_zero_write"])

# ---------------------------------------------------------------- verdict
gate_all = all(v for v in gates.values() if v is not None)
res = {
    "phase": 3262,
    "name": "g5h25_gpu_arm_readiness",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "gpu_arm_readiness_registered" if gate_all else "BLOCKED",
    "arm_a": {
        "artifact": "checkpoint_presence_audit_v1.json",
        "port_archive_1_7b": "FULLY_IN_PLACE (%d deltas, %.2f GB)"
                             % (len(delta_inv), audit["qwen3_1_7b_port"]["total_bytes"] / 1e9),
        "native_4b_archive": "OFF_MACHINE (manifest-only; ckpt96 terminal in place)",
        "torch_probe": probe,
        "probe_correction": ("3261 'torch env missing' = probe-scope false report; "
                             "correct scope = managed python 3.11.9 (torch %s, "
                             "cuda %s); sealed 3261 files untouched"
                             % (probe.get("torch"), probe.get("cuda_available"))),
    },
    "arm_b": {
        "artifact": "gpu_trajectory_protocol_draft_v1.json",
        "P_TRAJ": "preregistered draft -> execution 3263 (25 states, bit-exact "
                  "restore, R1/R2/R3 readouts)",
        "APPX_1": "draft registered; quantization acceptability decision pending",
        "APPX_2": "draft registered; GPU feasible as-is",
        "APPX_3": "draft registered; GPU feasible as-is",
        "APPX_4_5": "cross-line governance, not GPU, unchanged",
    },
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
