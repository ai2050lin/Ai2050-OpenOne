#!/usr/bin/env python3
"""C124 frozen validation of C123 registered-role transition nominations."""
from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1658_c124_role_transition_validation"
C123 = RESULT / "phase1657_c123_role_transition_atlas_discovery"
C115 = RESULT / "phase1625_c115_fifth_lexicon_prospective_replication"
C116 = RESULT / "phase1630_c116_negation_scope_observation_campaign"
C117 = RESULT / "phase1635_c117_whole_part_exception_observation_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1657_c123_role_transition_atlas as c123

FAMILIES = c123.FAMILIES
ROLES = c123.ROLES
STATES = c123.STATES
DIM = c123.DIM
SUPPORT_K = c123.SUPPORT_K


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int = SUPPORT_K) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def validation_cells() -> list[dict]:
    c116 = np.load(C116 / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    c117 = np.load(C117 / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    return [
        {"family": "attribute_binding", "partition": "fifth_lockbox", "fields": c123.c115_family_fields("attribute_binding", "fifth_lockbox")},
        {"family": "agent_patient", "partition": "fifth_lockbox", "fields": c123.c115_family_fields("agent_patient", "fifth_lockbox")},
        {"family": "negation_scope", "partition": "confirmation", "fields": np.asarray(c116[:12], dtype=np.float32)},
        {"family": "negation_scope", "partition": "lockbox", "fields": np.asarray(c116[12:], dtype=np.float32)},
        {"family": "explicit_exception_log", "partition": "confirmation", "fields": np.asarray(c117[:12], dtype=np.float32)},
        {"family": "explicit_exception_log", "partition": "lockbox", "fields": np.asarray(c117[12:], dtype=np.float32)},
    ]


def validation_source_paths() -> dict[str, Path]:
    return {
        "c123_protocol": C123 / "protocol/preregistration.json",
        "c123_nomination": C123 / "protocol/frozen_discovery_nomination.json",
        "c123_discovery_audit": C123 / "audit/independent_discovery_audit.json",
        "c123_trajectories": C123 / "analysis/discovery_selected_role_trajectories.float32.npy",
        "c123_increments": C123 / "analysis/discovery_selected_role_increments.float32.npy",
        "c115_fields": C115 / "analysis/unit_truth_role_state.float32.npy",
        "c116_validation": C116 / "analysis/validation_unit_truth_role_state.float32.npy",
        "c117_validation": C117 / "analysis/validation_unit_truth_role_state.float32.npy",
        "public_atlas": PUBLIC,
    }


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C124 output already exists: {OUT}")
    paths = validation_source_paths()
    audit = core.load(paths["c123_discovery_audit"])
    c123_protocol = core.load(paths["c123_protocol"])
    nomination = core.load(paths["c123_nomination"])
    cells = validation_cells()
    checks = {
        "c123_audit": audit["all_checks_passed"] and audit["authorization"] == "execute_c124_validation",
        "nomination_frozen": nomination["authorization"] == "freeze_c124_validation_without_reselection",
        "family_order": nomination["family_order"] == list(FAMILIES),
        "cell_count": len(cells) == 6,
        "cell_shapes": all(list(cell["fields"].shape) == [12, 7, 37, 2560] for cell in cells),
        "partitions": [(cell["family"], cell["partition"]) for cell in cells] == [
            ("attribute_binding", "fifth_lockbox"),
            ("agent_patient", "fifth_lockbox"),
            ("negation_scope", "confirmation"),
            ("negation_scope", "lockbox"),
            ("explicit_exception_log", "confirmation"),
            ("explicit_exception_log", "lockbox"),
        ],
        "atlas_schema": core.load(PUBLIC)["schema"] == "c109_role_state_field_atlas.v1",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    protocol = {
        "phase": 1658,
        "campaign": "C124",
        "created_at_utc": now(),
        "status": "frozen_role_transition_holdout_validation_contract",
        "object": c123_protocol["object"],
        "frozen_nomination_sha256": core.sha(paths["c123_nomination"]),
        "family_order": list(FAMILIES),
        "roles": list(ROLES),
        "validation_cells": [{"family": cell["family"], "partition": cell["partition"], "independent_units": 12} for cell in cells],
        "gates": c123_protocol["c124_validation_gates"],
        "control_definitions": {
            "wrong_state": "same selected role, every other layer increment",
            "wrong_role": "same selected layer increment, every other registered role",
            "coordinate_clock": "argmax absolute increment state for each discovery top256 coordinate",
            "candidate_rank": "frozen family nomination ranked against all 7x36 candidates in each holdout cell",
        },
        "evidence_policy": "a failed layer retires only that descriptor; vector repeatability, coordinate clock, and local specificity are adjudicated separately",
        "typed_missingness": c123_protocol["typed_missingness"],
        "forbidden_upgrades": c123_protocol["forbidden_upgrades"],
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_hashes": {name: core.sha(path) for name, path in paths.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "validate_without_reselection",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1658, "campaign": "C124", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def candidate_rank(fields: np.ndarray, target_role: str, target_to_state: int) -> tuple[int, int, float]:
    rows, _, _ = c123.candidate_table(fields, "validation")
    rows.sort(key=lambda row: (-row["score"], -row["split_half_cosine"], row["to_state"], row["role_index"]))
    rank = next(index + 1 for index, row in enumerate(rows) if row["role"] == target_role and row["to_state"] == target_to_state)
    target = rows[rank - 1]
    return rank, len(rows), float(target["score"])


def evaluate(
    family: str,
    partition: str,
    fields: np.ndarray,
    nomination: dict,
    discovery_trajectory: np.ndarray,
    discovery_increments: np.ndarray,
    gates: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    role = nomination["role"]
    role_index = ROLES.index(role)
    transition = int(nomination["to_state"]) - 1
    support = np.asarray(nomination["support"], dtype=np.int64)
    trajectory = np.mean(fields[:, role_index], axis=0, dtype=np.float32)
    increments = trajectory[1:] - trajectory[:-1]
    discovery_delta = discovery_increments[transition]
    validation_delta = increments[transition]
    target_cosine = cosine(discovery_delta, validation_delta)
    overlap = len(topk(discovery_delta) & topk(validation_delta)) / SUPPORT_K
    sign_agreement = float(np.mean(np.sign(discovery_delta[support]) == np.sign(validation_delta[support])))
    norm_ratio = min(float(np.linalg.norm(discovery_delta)), float(np.linalg.norm(validation_delta))) / max(float(np.linalg.norm(discovery_delta)), float(np.linalg.norm(validation_delta)), 1e-12)
    discovery_profile = np.linalg.norm(discovery_increments, axis=1)
    validation_profile = np.linalg.norm(increments, axis=1)
    profile_cosine = cosine(discovery_profile, validation_profile)
    discovery_clock = np.argmax(np.abs(discovery_increments[:, support]), axis=0)
    validation_clock = np.argmax(np.abs(increments[:, support]), axis=0)
    clock_exact = float(np.mean(discovery_clock == validation_clock))
    clock_within_one = float(np.mean(np.abs(discovery_clock - validation_clock) <= 1))
    wrong_state = max(cosine(discovery_delta, increments[index]) for index in range(STATES - 1) if index != transition)
    all_role_increments = np.mean(fields[:, :, 1:, :] - fields[:, :, :-1, :], axis=0, dtype=np.float32)
    wrong_role = max(cosine(discovery_delta, all_role_increments[index, transition]) for index in range(len(ROLES)) if index != role_index)
    rank, total_candidates, validation_score = candidate_rank(fields, role, int(nomination["to_state"]))
    vector_gate = target_cosine >= gates["vector_cosine_min"] and overlap >= gates["top256_overlap_min"] and sign_agreement >= gates["support_sign_agreement_min"]
    clock_gate = clock_within_one >= gates["coordinate_peak_clock_within_one_min"]
    state_gate = target_cosine - wrong_state > gates["state_specific_margin_gt"]
    role_gate = target_cosine - wrong_role > gates["role_specific_margin_gt"]
    result = {
        "family": family,
        "partition": partition,
        "independent_units": int(fields.shape[0]),
        "role": role,
        "from_state": int(nomination["from_state"]),
        "to_state": int(nomination["to_state"]),
        "target_increment_cosine": target_cosine,
        "top256_overlap": overlap,
        "support_sign_agreement": sign_agreement,
        "increment_norm_ratio": norm_ratio,
        "layer_norm_profile_cosine": profile_cosine,
        "coordinate_peak_clock_exact": clock_exact,
        "coordinate_peak_clock_within_one": clock_within_one,
        "best_wrong_state_cosine": wrong_state,
        "state_specific_margin": target_cosine - wrong_state,
        "best_wrong_role_cosine": wrong_role,
        "role_specific_margin": target_cosine - wrong_role,
        "frozen_candidate_rank": rank,
        "candidate_count": total_candidates,
        "validation_candidate_score": validation_score,
        "gates": {
            "level_1_full_vector": vector_gate,
            "level_2_coordinate_clock": clock_gate,
            "level_3_state_specific": state_gate,
            "level_3_role_specific": role_gate,
        },
        "delta_norm_profile": validation_profile.astype(np.float32).tolist(),
    }
    return result, trajectory, increments


def evaluate_common(cell: dict, common: dict, gates: dict) -> dict:
    family = cell["family"]
    fields = cell["fields"]
    discovery = c123.discovery_fields()[family]
    role_index = ROLES.index(common["role"])
    transition = int(common["to_state"]) - 1
    ddelta = np.mean(discovery[:, role_index, transition + 1] - discovery[:, role_index, transition], axis=0, dtype=np.float32)
    vdelta = np.mean(fields[:, role_index, transition + 1] - fields[:, role_index, transition], axis=0, dtype=np.float32)
    support = set(common["family_supports"][family])
    similarity = cosine(ddelta, vdelta)
    overlap = len(topk(vdelta) & support) / SUPPORT_K
    return {
        "family": family,
        "partition": cell["partition"],
        "role": common["role"],
        "from_state": int(common["from_state"]),
        "to_state": int(common["to_state"]),
        "target_increment_cosine": similarity,
        "top256_overlap": overlap,
        "level_1_full_vector": similarity >= gates["vector_cosine_min"] and overlap >= gates["top256_overlap_min"],
    }


def quantile_scale(rows: list[dict]) -> float:
    samples = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in rows])
    return float(np.quantile(samples, 0.99))


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/internal_contract_audit.json")
    if protocol["authorization"] != "validate_without_reselection" or not audit["all_checks_passed"]:
        raise RuntimeError("C124 validation authorization missing")
    nomination = core.load(C123 / "protocol/frozen_discovery_nomination.json")
    if core.sha(C123 / "protocol/frozen_discovery_nomination.json") != protocol["frozen_nomination_sha256"]:
        raise RuntimeError("C123 nomination drift")
    for name, path in protocol["source_paths"].items():
        if core.sha(Path(path)) != protocol["source_hashes"][name]:
            raise RuntimeError(f"C124 source drift: {name}")

    amendment = {
        "phase": 1658,
        "campaign": "C124",
        "created_at_utc": now(),
        "reason": "Create the derived analysis directory before the first np.save call after the initial validation execution stopped before writing any result artifact.",
        "original_producer_sha256": protocol["producer_sha256"],
        "repaired_producer_sha256": core.sha(Path(__file__)),
        "unchanged": ["frozen C123 nominations", "validation cells", "gates", "wrong-state controls", "wrong-role controls", "coordinate-clock rule", "claim boundary"],
    }
    core.save(OUT / "protocol/phase1658_execution_amendment.json", amendment)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)

    discovery_trajectories = np.load(C123 / "analysis/discovery_selected_role_trajectories.float32.npy", mmap_mode="r")
    discovery_increments = np.load(C123 / "analysis/discovery_selected_role_increments.float32.npy", mmap_mode="r")
    nominations = {row["family"]: row for row in nomination["family_nominations"]}
    cells = validation_cells()
    results = []
    common_results = []
    validation_trajectories = np.empty((len(cells), STATES, DIM), dtype=np.float32)
    validation_increments = np.empty((len(cells), STATES - 1, DIM), dtype=np.float32)
    cell_manifest = []
    for cell_index, cell in enumerate(cells):
        family = cell["family"]
        family_index = FAMILIES.index(family)
        result, trajectory, increments = evaluate(
            family,
            cell["partition"],
            cell["fields"],
            nominations[family],
            np.asarray(discovery_trajectories[family_index]),
            np.asarray(discovery_increments[family_index]),
            protocol["gates"],
        )
        results.append(result)
        common_results.append(evaluate_common(cell, nomination["common_nomination"], protocol["gates"]))
        validation_trajectories[cell_index] = trajectory
        validation_increments[cell_index] = increments
        cell_manifest.append({"cell_index": cell_index, "family": family, "partition": cell["partition"]})

    np.save(OUT / "analysis/validation_selected_role_trajectories.float32.npy", validation_trajectories)
    np.save(OUT / "analysis/validation_selected_role_increments.float32.npy", validation_increments)
    core.write_rows(OUT / "analysis/validation_results.jsonl", results)
    core.write_rows(OUT / "analysis/common_candidate_validation.jsonl", common_results)
    core.write_rows(OUT / "analysis/validation_cell_manifest.jsonl", cell_manifest)

    level_counts = {name: sum(row["gates"][name] for row in results) for name in ("level_1_full_vector", "level_2_coordinate_clock", "level_3_state_specific", "level_3_role_specific")}
    family_rollup = {}
    for family in FAMILIES:
        selected = [row for row in results if row["family"] == family]
        family_rollup[family] = {
            "cells": len(selected),
            "level_1_passed": sum(row["gates"]["level_1_full_vector"] for row in selected),
            "level_2_passed": sum(row["gates"]["level_2_coordinate_clock"] for row in selected),
            "state_specific_passed": sum(row["gates"]["level_3_state_specific"] for row in selected),
            "role_specific_passed": sum(row["gates"]["level_3_role_specific"] for row in selected),
            "cosine_range": [min(row["target_increment_cosine"] for row in selected), max(row["target_increment_cosine"] for row in selected)],
            "clock_within_one_range": [min(row["coordinate_peak_clock_within_one"] for row in selected), max(row["coordinate_peak_clock_within_one"] for row in selected)],
            "candidate_rank_range": [min(row["frozen_candidate_rank"] for row in selected), max(row["frozen_candidate_rank"] for row in selected)],
        }

    checks = {
        "execution_amendment": amendment["original_producer_sha256"] == protocol["producer_sha256"] and amendment["repaired_producer_sha256"] == core.sha(Path(__file__)),
        "result_count": len(results) == 6 and len(common_results) == 6,
        "trajectory_shape": list(validation_trajectories.shape) == [6, 37, 2560],
        "increment_shape": list(validation_increments.shape) == [6, 36, 2560],
        "finite": bool(np.isfinite(validation_trajectories).all() and np.isfinite(validation_increments).all()),
        "telescoping": all(np.allclose(validation_trajectories[index, -1] - validation_trajectories[index, 0], np.sum(validation_increments[index], axis=0), rtol=1e-5, atol=1e-5) for index in range(6)),
        "no_reselection": all(row["role"] == nominations[row["family"]]["role"] and row["to_state"] == nominations[row["family"]]["to_state"] for row in results),
        "candidate_counts": all(row["candidate_count"] == 252 for row in results),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1658,
        "campaign": "C124",
        "created_at_utc": now(),
        "status": "frozen_role_transition_holdout_validation_complete",
        "checks": checks,
        "level_counts": level_counts,
        "family_rollup": family_rollup,
        "common_candidate": {
            "role": nomination["common_nomination"]["role"],
            "from_state": nomination["common_nomination"]["from_state"],
            "to_state": nomination["common_nomination"]["to_state"],
            "level_1_passed_cells": sum(row["level_1_full_vector"] for row in common_results),
            "cells": len(common_results),
        },
        "claim_boundary": "Frozen registered-role layer response increments were tested on already-existing held-out partitions. This is a procedurally sealed secondary validation, not a new blind model run, an endogenous operator, or a complete-token causal transmission graph.",
        "authorization": "synthesize_transition_heatmap_and_close",
    }
    core.save(OUT / "analysis/validation_summary.json", report)
    core.save(OUT / "audit/internal_validation_audit.json", {"phase": 1658, "campaign": "C124", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/validation_summary.json")
    if summary["authorization"] != "synthesize_transition_heatmap_and_close" or not core.load(OUT / "audit/internal_validation_audit.json")["all_checks_passed"]:
        raise RuntimeError("C124 synthesis authorization missing")
    nomination = core.load(C123 / "protocol/frozen_discovery_nomination.json")
    discovery_trajectories = np.load(C123 / "analysis/discovery_selected_role_trajectories.float32.npy", mmap_mode="r")
    discovery_increments = np.load(C123 / "analysis/discovery_selected_role_increments.float32.npy", mmap_mode="r")
    validation_trajectories = np.load(OUT / "analysis/validation_selected_role_trajectories.float32.npy", mmap_mode="r")
    validation_increments = np.load(OUT / "analysis/validation_selected_role_increments.float32.npy", mmap_mode="r")
    manifest = core.rows(OUT / "analysis/validation_cell_manifest.jsonl")
    results = core.rows(OUT / "analysis/validation_results.jsonl")
    by_family_indices: dict[str, list[int]] = defaultdict(list)
    for row in manifest:
        by_family_indices[row["family"]].append(int(row["cell_index"]))
    transition_rows = []
    profiles = []
    for family_index, family in enumerate(FAMILIES):
        nominee = next(row for row in nomination["family_nominations"] if row["family"] == family)
        validation_trajectory = np.mean(np.asarray(validation_trajectories[by_family_indices[family]], dtype=np.float32), axis=0, dtype=np.float32)
        validation_increment = np.mean(np.asarray(validation_increments[by_family_indices[family]], dtype=np.float32), axis=0, dtype=np.float32)
        for scope, trajectories, increments in (
            ("discovery", np.asarray(discovery_trajectories[family_index]), np.asarray(discovery_increments[family_index])),
            ("heldout_mean", validation_trajectory, validation_increment),
        ):
            for state in range(STATES):
                transition_rows.append({
                    "dataset": "C123-C124",
                    "family": family,
                    "partition": scope,
                    "role": nominee["role"],
                    "state": state,
                    "state_kind": "embedding" if state == 0 else "hidden_state",
                    "effect": "balanced_truth_response",
                    "values": np.asarray(trajectories[state], dtype=np.float32).tolist(),
                })
            norms = np.linalg.norm(increments, axis=1)
            profiles.append({"family": family, "partition": scope, "role": nominee["role"], "values": norms.astype(np.float32).tolist()})
            for transition in range(STATES - 1):
                transition_rows.append({
                    "dataset": "C123-C124",
                    "family": family,
                    "partition": scope,
                    "role": nominee["role"],
                    "from_state": transition,
                    "to_state": transition + 1,
                    "state_kind": "layer_increment",
                    "effect": "balanced_truth_response_increment",
                    "values": np.asarray(increments[transition], dtype=np.float32).tolist(),
                })

    payload = core.load(PUBLIC)
    payload["transition_rows"] = transition_rows
    payload["c123_c124_transition_batch"] = {
        "discovery": core.load(C123 / "analysis/discovery_summary.json"),
        "nomination": {
            "family_nominations": [{key: value for key, value in row.items() if key != "support"} for row in nomination["family_nominations"]],
            "common_nomination": {key: value for key, value in nomination["common_nomination"].items() if key != "family_supports"},
        },
        "validation": summary,
        "results": results,
        "profiles": profiles,
    }
    payload["scale"]["transition_symmetric_abs_q99"] = quantile_scale([row for row in transition_rows if row["effect"] == "balanced_truth_response_increment"])
    payload.update({
        "phase": 1658,
        "campaign": "C109-C117 + C123-C124",
        "title": "C109-C117 Role-State Atlas + C123-C124 Layer Response Transitions",
        "claim_boundary": "C123-C124 add full-2560 registered-role response trajectories and adjacent-layer increments. They are activation-coordinate observations from old Qwen3 archives, not model weights, independent neurons, a complete-token atlas, Jacobians, natural causal flow, attention/MLP mechanism, fiber bundles, curvature, topology, or new mathematics.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c124_role_transition_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    closure = {
        "phase": 1658,
        "campaign": "C124",
        "created_at_utc": now(),
        "status": "registered_role_transition_atlas_large_stage_complete",
        "headline": summary,
        "new_puzzles": {
            "K315": "C123 freezes family-conditioned registered-role adjacent-layer response increments using discovery partitions only.",
            "K316": "C124 separates full-vector repeatability, coordinate peak-clock repeatability, and wrong-state/wrong-role specificity on held-out old partitions; no layer increment is promoted to an endogenous language operator.",
        },
        "theory_update": "RDC gains a descriptive transition object DeltaR between adjacent registered-role response states. The object records where a balanced relation response grows, reverses, or migrates across activation coordinates; operator identity and causal transport remain open.",
        "unified_formula": "R_f(r,s+1)=R_f(r,s)+DeltaR_f(r,s->s+1); DeltaR is descriptive until it predicts fresh trajectories and survives causal controls.",
        "problems": [
            "all source arrays predate C123, so C124 is not a de novo blinded model replication",
            "only seven registered role spans are observed; unregistered token positions remain latent",
            "C115-C117 have typed output-protocol failures and cannot support general successful-language-mechanism claims",
            "one Qwen3 model and controlled synthetic English only",
            "adjacent-layer subtraction does not identify which upstream coordinates naturally caused downstream changes",
        ],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": 2560, "includes_embedding": True, "includes_all_hidden_states": True},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C125 fresh semantic-program family is authorized only as a prospective test of the frozen C123 common/family transition descriptors; no topology or operator claim is authorized.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "validation": core.load(OUT / "audit/internal_validation_audit.json")["all_checks_passed"],
        "transition_rows": len(transition_rows) == len(FAMILIES) * 2 * (STATES + STATES - 1),
        "full_coordinates": all(len(row["values"]) == DIM for row in transition_rows),
        "embedding_rows": sum(row.get("state") == 0 for row in transition_rows) == len(FAMILIES) * 2,
        "hidden_rows": sum(row.get("effect") == "balanced_truth_response" and int(row.get("state", 0)) > 0 for row in transition_rows) == len(FAMILIES) * 2 * 36,
        "increment_rows": sum(row["effect"] == "balanced_truth_response_increment" for row in transition_rows) == len(FAMILIES) * 2 * 36,
        "asset": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": all(term in payload["claim_boundary"] for term in ("not model weights", "not", "attention/MLP")),
    }
    report = {"phase": 1658, "campaign": "C124", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "run_independent_closure_audit"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"audit": report, "headline": summary, "next_authorization": closure["next_authorization"]}, indent=2))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"contract", "validate", "synthesize"}:
        raise SystemExit("usage: phase1658_c124_role_transition_validation.py {contract|validate|synthesize}")
    {"contract": contract, "validate": validate, "synthesize": synthesize}[sys.argv[1]]()


if __name__ == "__main__":
    main()
