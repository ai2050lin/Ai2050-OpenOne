#!/usr/bin/env python3
"""C145: matched correct/error and direct/three-hop type-graph trajectories."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
OUT = RESULT / "phase1679_c145_correct_error_depth_trajectory_atlas"
C141 = RESULT / "phase1675_c141_multifamily_full_coordinate_atlas"
C142 = RESULT / "phase1676_c142_mobius_output_code_separation"
C144 = RESULT / "phase1678_c144_dual_graph_composition_reconstruction"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141

PHASE, CAMPAIGN = 1679, "C145"
ROLES, CHECKPOINTS, DIM = c141.ROLES, c127.CHECKPOINTS, 2560


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den <= 1e-12 else float(np.dot(a.ravel(), b.ravel()) / den)


def overlap(a: np.ndarray, b: np.ndarray, k: int = 256) -> float:
    aa = set(np.argpartition(np.abs(a), -k)[-k:].tolist())
    bb = set(np.argpartition(np.abs(b), -k)[-k:].tolist())
    return len(aa & bb) / k


def sources() -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    compiled = core.rows(C141 / "compiled/qwen3.jsonl")
    behavior = core.rows(C141 / "raw/qwen3_behavior_index.jsonl")
    raw = np.load(C141 / "raw/qwen3_six_role_field.bf16.npy", mmap_mode="r")
    logits = np.load(C141 / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    return compiled, behavior, raw, logits


def key(row: dict) -> tuple:
    f = row["factors"]
    return (f["f1"], f["f2"], f["f3"], row["surface_factor"], row["codebook_factor"])


def error_residuals(partition: str, *, cross_partition_fallback: bool = False) -> tuple[np.ndarray, list[dict], list[dict]]:
    _, behavior, raw, _ = sources()
    correct = defaultdict(list)
    all_correct = defaultdict(list)
    selected = []
    for row in behavior:
        if row["arm"] == "type_graph" and row["correct"]:
            all_correct[key(row)].append(row)
        if row["arm"] == "type_graph" and row["partition"] == partition:
            if row["correct"]:
                correct[key(row)].append(row)
            else:
                selected.append(row)
    out, index, missing = [], [], []
    for row in selected:
        refs = correct[key(row)]
        if not refs:
            missing.append({"case_id": row["case_id"], "unit_id": row["unit_id"], "stratum": list(key(row))})
            if not cross_partition_fallback:
                continue
            refs = all_correct[key(row)]
        if not refs:
            continue
        value = c127.decode(raw[row["row_index"]])
        reference = np.mean([c127.decode(raw[ref["row_index"]]) for ref in refs], axis=0, dtype=np.float32)
        out.append(value - reference)
        index.append({"case_id": row["case_id"], "unit_id": row["unit_id"], "matched_correct_count": len(refs), "stratum": list(key(row)), "cross_partition_fallback": row["case_id"] in {x["case_id"] for x in missing}})
    return np.asarray(out, np.float32), index, missing


def behavior_table() -> dict:
    _, behavior, _, logits = sources()
    groups = defaultdict(list)
    for row in behavior:
        if row["arm"] != "type_graph":
            continue
        label = (row["partition"], "direct" if row["factors"]["f2"] == 1 else "three_hop", "valid" if row["factors"]["f3"] == 1 else "invalid")
        gold = row["gold_position"]
        margin = float(logits[row["row_index"], gold] - logits[row["row_index"], 1 - gold])
        groups[label].append((bool(row["correct"]), margin))
    table = {}
    for label, values in groups.items():
        table["|".join(label)] = {
            "n": len(values),
            "accuracy": float(np.mean([v[0] for v in values])),
            "mean_gold_margin": float(np.mean([v[1] for v in values])),
            "median_gold_margin": float(np.median([v[1] for v in values])),
            "errors": sum(not v[0] for v in values),
        }
    return table


def discover() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C144 / "audit/independent_closure_audit.json")
    if not parent["all_checks_passed"] or parent["authorization"] != "start_C145":
        raise RuntimeError(parent)
    OUT.mkdir(parents=True); (OUT / "protocol").mkdir(); (OUT / "analysis").mkdir(); (OUT / "audit").mkdir()
    residuals, index, missing = error_residuals("discovery")
    np.save(OUT / "analysis/discovery_matched_error_residuals.float32.npy", residuals)
    core.write_rows(OUT / "analysis/discovery_error_index.jsonl", index)
    left = residuals[:6].mean(0)
    right = residuals[6:].mean(0)
    candidates = []
    for ri, role in enumerate(ROLES):
        for q in range(29):
            co = cosine(left[ri, q], right[ri, q])
            norm = min(float(np.linalg.norm(left[ri, q])), float(np.linalg.norm(right[ri, q])))
            candidates.append((max(co, 0.0) * norm, co, norm, ri, q))
    score, co, norm, ri, q = max(candidates)
    vector = residuals[:, ri, q].mean(0)
    np.save(OUT / "protocol/discovery_error_nominee.float32.npy", vector)
    freeze = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "matched_error_nominee_frozen",
        "definition": "error state minus mean correct state matched on f1,f2,f3,surface,codebook within partition",
        "discovery_error_count": len(residuals),
        "role": ROLES[ri], "role_index": ri,
        "checkpoint": CHECKPOINTS[q], "checkpoint_index": q,
        "split_error_count": [6, 2], "split_half_cosine": co,
        "split_half_min_norm": norm, "selection_score": score,
        "support": sorted(np.argpartition(np.abs(vector), -256)[-256:].tolist()),
        "confirmation_gate": {"cosine_min": 0.50, "top256_overlap_min": 0.20, "wrong_role_cosine_margin_min": 0.05, "wrong_coordinate_cosine_margin_min": 0.05},
        "source_hashes": {"role": core.sha(C141 / "raw/qwen3_six_role_field.bf16.npy"), "behavior": core.sha(C141 / "raw/qwen3_behavior_index.jsonl")},
        "confirmation_unread": True,
        "claim_boundary": "matched error correlate, not a cause of error or a knowledge-depth mechanism",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "validate_C145",
    }
    core.save(OUT / "protocol/frozen_error_nominee.json", freeze)
    table = behavior_table()
    core.save(OUT / "analysis/behavior_depth_table.json", table)
    checks = {"errors": list(residuals.shape) == [8, 6, 38, DIM], "matched": all(x["matched_correct_count"] >= 1 for x in index), "no_missing": len(missing) == 0, "candidate": q <= 28, "behavior_cells": len(table) == 8, "finite": bool(np.isfinite(residuals).all())}
    core.save(OUT / "audit/internal_discovery_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps({"checks": checks, "nominee": {k: freeze[k] for k in ("role", "checkpoint", "split_half_cosine", "split_half_min_norm")}, "behavior": table}, indent=2))


def validate() -> None:
    freeze = core.load(OUT / "protocol/frozen_error_nominee.json")
    residuals, index, missing = error_residuals("confirmation")
    exploratory, exploratory_index, fallback_missing = error_residuals("confirmation", cross_partition_fallback=True)
    np.save(OUT / "analysis/confirmation_exact_support_error_residuals.float32.npy", residuals)
    np.save(OUT / "analysis/confirmation_exploratory_all_error_residuals.float32.npy", exploratory)
    core.write_rows(OUT / "analysis/confirmation_exact_support_error_index.jsonl", index)
    core.write_rows(OUT / "analysis/confirmation_exploratory_error_index.jsonl", exploratory_index)
    core.save(OUT / "analysis/confirmation_missing_exact_support.json", missing)
    disc = np.load(OUT / "protocol/discovery_error_nominee.float32.npy")
    mean = exploratory.mean(0)
    ri, q = freeze["role_index"], freeze["checkpoint_index"]
    target = mean[ri, q]
    target_cos, target_overlap = cosine(disc, target), overlap(disc, target)
    wrong_role_cos = cosine(disc, np.roll(mean, 1, axis=0)[ri, q])
    wrong_coord_cos = cosine(disc, np.roll(target, 1))
    gate = freeze["confirmation_gate"]
    descriptive_gates = {
        "cosine": target_cos >= gate["cosine_min"],
        "overlap": target_overlap >= gate["top256_overlap_min"],
        "wrong_role": target_cos - wrong_role_cos >= gate["wrong_role_cosine_margin_min"],
        "wrong_coordinate": target_cos - wrong_coord_cos >= gate["wrong_coordinate_cosine_margin_min"],
    }
    eligibility_passed = len(missing) == 0
    gates = {name: (value if eligibility_passed else False) for name, value in descriptive_gates.items()}
    # Preserve the full cross-partition depth-effect trajectory from C142.
    disc_m = np.load(C142 / "analysis/discovery_mobius.float32.npy", mmap_mode="r")[1, :, 1].mean(0)
    conf_m = np.load(C142 / "analysis/confirmation_mobius.float32.npy", mmap_mode="r")[1, :, 1].mean(0)
    depth_rows = []
    for role_i, role in enumerate(ROLES):
        for checkpoint_i, checkpoint in enumerate(CHECKPOINTS):
            depth_rows.append({"role": role, "role_index": role_i, "checkpoint": checkpoint, "checkpoint_index": checkpoint_i, "cosine": cosine(disc_m[role_i, checkpoint_i], conf_m[role_i, checkpoint_i]), "discovery_norm": float(np.linalg.norm(disc_m[role_i, checkpoint_i])), "confirmation_norm": float(np.linalg.norm(conf_m[role_i, checkpoint_i]))})
    report = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "correct_error_depth_atlas_adjudicated",
        "confirmation_error_count": len(exploratory),
        "exact_support_error_count": len(residuals),
        "missing_exact_support_count": len(missing),
        "missing_exact_support": missing,
        "eligibility_passed": eligibility_passed,
        "nominee": {"role": freeze["role"], "checkpoint": freeze["checkpoint"], "cosine": target_cos, "top256_overlap": target_overlap, "wrong_role_cosine": wrong_role_cos, "wrong_coordinate_cosine": wrong_coord_cos},
        "descriptive_fallback_gates": descriptive_gates,
        "gates": gates, "matched_error_replication_passed": eligibility_passed and all(gates.values()),
        "depth_effect_rows": depth_rows,
        "claim_boundary": freeze["claim_boundary"],
        "authorization": "close_C145_continue_C146",
    }
    core.save(OUT / "analysis/confirmation.json", report)
    checks = {"exact_support": list(residuals.shape) == [7, 6, 38, DIM], "all_errors_descriptive": list(exploratory.shape) == [11, 6, 38, DIM], "missing_typed": len(missing) == 4 and len(fallback_missing) == 4, "matched": all(x["matched_correct_count"] >= 1 for x in exploratory_index), "depth_rows": len(depth_rows) == 228, "finite": all(np.isfinite(x) for x in (target_cos, target_overlap, wrong_role_cos, wrong_coord_cos))}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_error_replication_passed": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"checks": checks, "eligibility_passed": eligibility_passed, "missing": missing, "nominee": report["nominee"], "descriptive_fallback_gates": descriptive_gates, "scientific_gates": gates}, indent=2))


def close() -> None:
    report = core.load(OUT / "analysis/confirmation.json")
    checks = {"discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"], "typed": "cause" in report["claim_boundary"]}
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "status": "correct_error_depth_atlas_closed", "headline": {"replicated": report["matched_error_replication_passed"], "errors": [8, 11], "nominee": report["nominee"]}, "theory_update": "separates depth response from a matched behavior-error response without deleting errors", "claim_boundary": report["claim_boundary"], "next_authorization": "C146 sequential cross-model behavior-interface sweep"}
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_final_then_C146"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"discover": discover, "validate": validate, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes: raise SystemExit("discover|validate|close")
    modes[sys.argv[1]]()


if __name__ == "__main__": main()
