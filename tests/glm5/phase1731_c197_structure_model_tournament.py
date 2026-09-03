#!/usr/bin/env python3
"""C197: tournament of simple signed q24-to-q25 trajectory models."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1731_c197_structure_model_tournament"
C195 = RESULT / "phase1729_c195_signed_role_checkpoint_trajectory"
C196 = RESULT / "phase1730_c196_multidose_orthogonal_identification"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1726_c192_multi_program_response_equivalence as c192

PHASE, CAMPAIGN = 1731, "C197"
DIM, ROLES = 2560, c192.ROLES
MODELS = ("identity", "global_gain", "role_gain", "coordinate_gain", "role_coordinate_gain", "program_role_coordinate_gain", "family_role_coordinate_gain")


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C196 / "audit/independent_final_audit.json"); index = core.rows(C195 / "raw/index.jsonl")
    discovery = [r["anchor_index"] for r in index if r["unit"] == 1]; confirmation = [r["anchor_index"] for r in index if r["unit"] == 4]
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"] == "C197_structure_model_tournament_and_holdout_prediction", "discovery": len(discovery) == 56, "confirmation": len(confirmation) == 56, "disjoint": not set(discovery) & set(confirmation), "models": len(MODELS) == 7}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "structure_model_tournament_frozen",
        "discovery": "C195 unit1 all families/programs/phrases/source coordinates", "confirmation": "C195 untouched unit4",
        "joint_stimulus_panel": "all C196 orthogonal multi-dose responses; q24 response is observed and q25 is predicted",
        "models": list(MODELS), "fit": "zero-intercept least-squares gains, clipped to [-4,4]; no PCA or coordinate selection",
        "selection": "minimum worst-panel NRMSE across C195 confirmation and C196 joint-stimulus panel; ties follow lower complexity order",
        "primary_gate": {"confirmation_identity_nrmse_improvement_min": 0.05, "joint_identity_nrmse_improvement_min": 0.05, "confirmation_nrmse_max": 0.75, "joint_nrmse_max": 0.75},
        "claim_boundary": "predictive local checkpoint transform, not a semantic operator, global dynamical law, or unique parameter circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "intercepts", "post-reveal model additions", "training on unit4"],
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_C197_tournament_then_C198_broad_natural_programs",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "protocol/splits.json", {"discovery": discovery, "confirmation": confirmation})
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks, "models": list(MODELS)}, indent=2))


def safe_gain(num, den):
    return np.clip(np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 1e-12), -4.0, 4.0).astype(np.float32)


def fit_models(raw, index, discovery):
    families = sorted({r["family"] for r in index}); programs = list(c192.PROGRAMS)
    num_rc = np.zeros((6, DIM), np.float64); den_rc = np.zeros((6, DIM), np.float64)
    num_prc = np.zeros((len(programs), 6, DIM), np.float64); den_prc = np.zeros_like(num_prc)
    num_frc = np.zeros((len(families), 6, DIM), np.float64); den_frc = np.zeros_like(num_frc)
    for anchor_i in discovery:
        u = np.asarray(raw[anchor_i, :, 0], dtype=np.float32); v = np.asarray(raw[anchor_i, :, 1], dtype=np.float32)
        uv = (u * v).sum(axis=0, dtype=np.float64); uu = np.square(u, dtype=np.float64).sum(axis=0)
        num_rc += uv; den_rc += uu
        pi = programs.index(index[anchor_i]["program"]); fi = families.index(index[anchor_i]["family"])
        num_prc[pi] += uv; den_prc[pi] += uu; num_frc[fi] += uv; den_frc[fi] += uu
    gains = {
        "identity": np.array(1.0, np.float32),
        "global_gain": safe_gain(np.array(num_rc.sum()), np.array(den_rc.sum())),
        "role_gain": safe_gain(num_rc.sum(axis=1), den_rc.sum(axis=1)),
        "coordinate_gain": safe_gain(num_rc.sum(axis=0), den_rc.sum(axis=0)),
        "role_coordinate_gain": safe_gain(num_rc, den_rc),
        "program_role_coordinate_gain": safe_gain(num_prc, den_prc),
        "family_role_coordinate_gain": safe_gain(num_frc, den_frc),
    }
    return gains, families, programs


def predict(name, gain, u, meta, families, programs):
    if name in ("identity", "global_gain"): return u * gain
    if name == "role_gain": return u * gain[None, :, None]
    if name == "coordinate_gain": return u * gain[None, None, :]
    if name == "role_coordinate_gain": return u * gain[None, :, :]
    if name == "program_role_coordinate_gain": return u * gain[programs.index(meta["program"])][None, :, :]
    if name == "family_role_coordinate_gain": return u * gain[families.index(meta["family"])][None, :, :]
    raise ValueError(name)


def accum(metrics, name, prediction, truth):
    metrics[name]["error2"] += float(np.square(prediction - truth, dtype=np.float64).sum())
    metrics[name]["truth2"] += float(np.square(truth, dtype=np.float64).sum())
    weight = np.minimum(np.abs(prediction), np.abs(truth)).astype(np.float64)
    metrics[name]["sign_num"] += float((weight * (np.signbit(prediction) == np.signbit(truth))).sum()); metrics[name]["sign_den"] += float(weight.sum())


def finalize(metrics):
    return {name: {"nrmse": float(np.sqrt(m["error2"] / max(m["truth2"], 1e-30))), "weighted_sign_agreement": m["sign_num"] / max(m["sign_den"], 1e-30)} for name, m in metrics.items()}


def build():
    raw = np.load(C195 / "raw/signed_q23_q24_q25.float16.npy", mmap_mode="r"); index = core.rows(C195 / "raw/index.jsonl"); splits = core.load(OUT / "protocol/splits.json")
    gains, families, programs = fit_models(raw, index, splits["discovery"])
    (OUT / "analysis/operators").mkdir(parents=True, exist_ok=True)
    for name, gain in gains.items(): np.save(OUT / f"analysis/operators/{name}.float32.npy", gain)
    metrics = {name: {"error2": 0.0, "truth2": 0.0, "sign_num": 0.0, "sign_den": 0.0} for name in MODELS}
    for anchor_i in splits["confirmation"]:
        u = np.asarray(raw[anchor_i, :, 0], dtype=np.float32); v = np.asarray(raw[anchor_i, :, 1], dtype=np.float32)
        for name in MODELS: accum(metrics, name, predict(name, gains[name], u, index[anchor_i], families, programs), v)
    confirmation = finalize(metrics)
    actual = np.load(C196 / "raw/orthogonal_actual.float16.npy", mmap_mode="r"); source_indices = core.load(C196 / "protocol/source_anchor_indices.json")["indices"]
    metrics = {name: {"error2": 0.0, "truth2": 0.0, "sign_num": 0.0, "sign_den": 0.0} for name in MODELS}
    for local_i, source_i in enumerate(source_indices):
        meta = index[source_i]
        for dose_i in range(actual.shape[1]):
            u = np.asarray(actual[local_i, dose_i, :, 0], dtype=np.float32); v = np.asarray(actual[local_i, dose_i, :, 1], dtype=np.float32)
            for name in MODELS: accum(metrics, name, predict(name, gains[name], u, meta, families, programs), v)
    joint = finalize(metrics)
    rank = sorted(MODELS, key=lambda name: (max(confirmation[name]["nrmse"], joint[name]["nrmse"]), MODELS.index(name)))
    winner = rank[0]; identity = "identity"; gate = core.load(OUT / "protocol/preregistration.json")["primary_gate"]
    confirmation_improvement = confirmation[identity]["nrmse"] - confirmation[winner]["nrmse"]; joint_improvement = joint[identity]["nrmse"] - joint[winner]["nrmse"]
    passed = confirmation_improvement >= gate["confirmation_identity_nrmse_improvement_min"] and joint_improvement >= gate["joint_identity_nrmse_improvement_min"] and confirmation[winner]["nrmse"] <= gate["confirmation_nrmse_max"] and joint[winner]["nrmse"] <= gate["joint_nrmse_max"]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "structure_model_tournament_analyzed", "confirmation": confirmation, "joint_stimulus": joint, "ranking": rank, "winner": winner, "winner_confirmation_identity_improvement": confirmation_improvement, "winner_joint_identity_improvement": joint_improvement, "primary_gate_passed": passed, "operator_shapes": {name: list(np.asarray(gain).shape) for name, gain in gains.items()}, "interpretation": "The winner is the simplest registered local q24-to-q25 gain structure with best worst-panel prediction. It is a checkpoint transform, not a language-level relation algebra.", "next_authorization": "C198_broad_natural_program_behavior_and_signed_trajectory"}
    core.save(OUT / "analysis/tournament.json", report)
    checks = {"models": set(confirmation) == set(MODELS) == set(joint), "ranking": set(rank) == set(MODELS), "finite": bool(np.isfinite([[confirmation[m]["nrmse"], joint[m]["nrmse"]] for m in MODELS]).all()), "operators": all((OUT / f"analysis/operators/{m}.float32.npy").exists() for m in MODELS)}
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"winner": winner, "confirmation": confirmation, "joint": joint, "improvements": [confirmation_improvement, joint_improvement], "passed": passed, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/tournament.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "build": core.load(OUT / "audit/internal_build_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "build", "close")); args = parser.parse_args(); {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__": main()
