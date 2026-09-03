#!/usr/bin/env python3
"""C123 discovery-only atlas for registered-role layer response increments."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1657_c123_role_transition_atlas_discovery"
C115 = RESULT / "phase1625_c115_fifth_lexicon_prospective_replication"
C116 = RESULT / "phase1630_c116_negation_scope_observation_campaign"
C117 = RESULT / "phase1635_c117_whole_part_exception_observation_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "explicit_exception_log")
ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
STATES = 37
DIM = 2560
SUPPORT_K = 256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def sha_map(paths: dict[str, Path]) -> dict[str, str]:
    return {name: core.sha(path) for name, path in paths.items()}


def source_paths() -> dict[str, Path]:
    return {
        "c115_units": C115 / "material/units.jsonl",
        "c115_fields": C115 / "analysis/unit_truth_role_state.float32.npy",
        "c115_closure_audit": C115 / "audit/independent_closure_audit.json",
        "c116_discovery": C116 / "analysis/discovery_unit_truth_role_state.float32.npy",
        "c116_validation": C116 / "analysis/validation_unit_truth_role_state.float32.npy",
        "c116_closure_audit": C116 / "audit/independent_closure_audit.json",
        "c117_discovery": C117 / "analysis/discovery_unit_truth_role_state.float32.npy",
        "c117_validation": C117 / "analysis/validation_unit_truth_role_state.float32.npy",
        "c117_closure_audit": C117 / "audit/independent_closure_audit.json",
    }


def c115_family_fields(family: str, partition: str) -> np.ndarray:
    units = core.rows(C115 / "material/units.jsonl")
    indices = [index for index, row in enumerate(units) if row["family"] == family and row["partition"] == partition]
    fields = np.load(C115 / "analysis/unit_truth_role_state.float32.npy", mmap_mode="r")
    return np.asarray(fields[indices], dtype=np.float32)


def discovery_fields() -> dict[str, np.ndarray]:
    return {
        "attribute_binding": c115_family_fields("attribute_binding", "fifth_confirmation"),
        "agent_patient": c115_family_fields("agent_patient", "fifth_confirmation"),
        "negation_scope": np.asarray(np.load(C116 / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r"), dtype=np.float32),
        "explicit_exception_log": np.asarray(np.load(C117 / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r"), dtype=np.float32),
    }


def topk(values: np.ndarray, k: int = SUPPORT_K) -> list[int]:
    indices = np.argpartition(np.abs(values), -k)[-k:]
    return [int(index) for index in indices[np.argsort(-np.abs(values[indices]))]]


def candidate_table(fields: np.ndarray, family: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    if fields.shape != (12, len(ROLES), STATES, DIM):
        raise RuntimeError((family, fields.shape))
    increments = fields[:, :, 1:, :] - fields[:, :, :-1, :]
    left = np.mean(increments[:6], axis=0, dtype=np.float32)
    right = np.mean(increments[6:], axis=0, dtype=np.float32)
    rows = []
    for role_index, role in enumerate(ROLES):
        for transition_index in range(STATES - 1):
            lvec = left[role_index, transition_index]
            rvec = right[role_index, transition_index]
            similarity = cosine(lvec, rvec)
            left_norm = float(np.linalg.norm(lvec))
            right_norm = float(np.linalg.norm(rvec))
            rows.append({
                "family": family,
                "role": role,
                "role_index": role_index,
                "from_state": transition_index,
                "to_state": transition_index + 1,
                "split_half_cosine": similarity,
                "left_norm": left_norm,
                "right_norm": right_norm,
                "score": max(0.0, similarity) * min(left_norm, right_norm),
            })
    return rows, left, right


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C123 output already exists: {OUT}")
    paths = source_paths()
    missing = [name for name, path in paths.items() if not path.exists()]
    audits = [core.load(paths[name]) for name in ("c115_closure_audit", "c116_closure_audit", "c117_closure_audit")]
    c115 = np.load(paths["c115_fields"], mmap_mode="r")
    c116d = np.load(paths["c116_discovery"], mmap_mode="r")
    c116v = np.load(paths["c116_validation"], mmap_mode="r")
    c117d = np.load(paths["c117_discovery"], mmap_mode="r")
    c117v = np.load(paths["c117_validation"], mmap_mode="r")
    checks = {
        "sources_exist": not missing,
        "source_audits": all(report["all_checks_passed"] for report in audits),
        "c115_shape": list(c115.shape) == [48, 7, 37, 2560],
        "c116_shapes": list(c116d.shape) == [12, 7, 37, 2560] and list(c116v.shape) == [24, 7, 37, 2560],
        "c117_shapes": list(c117d.shape) == [12, 7, 37, 2560] and list(c117v.shape) == [24, 7, 37, 2560],
        "finite_samples": all(np.isfinite(np.asarray(array.reshape(-1)[:4096], dtype=np.float32)).all() for array in (c115, c116d, c116v, c117d, c117v)),
        "roles_aligned": all(core.load(root / "protocol/preregistration.json")["roles"] == list(ROLES) for root in (C115, C116, C117)),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "missing": missing})
    protocol = {
        "phase": 1657,
        "campaign": "C123",
        "created_at_utc": now(),
        "status": "registered_role_transition_atlas_contract_frozen",
        "object": "balanced-truth registered-role layer response increments",
        "definition": "DeltaR[f,u,r,s]=R[f,u,r,s]-R[f,u,r,s-1], s=1..36",
        "families": list(FAMILIES),
        "roles": list(ROLES),
        "states": STATES,
        "activation_coordinates": DIM,
        "discovery_units_per_family": 12,
        "discovery_sources": {
            "attribute_binding": "C115 fifth_confirmation",
            "agent_patient": "C115 fifth_confirmation",
            "negation_scope": "C116 discovery",
            "explicit_exception_log": "C117 discovery",
        },
        "discovery_rule": {
            "unit_split": "first six versus last six within each registered discovery partition",
            "candidate_score": "max(0, split_half_cosine) * min(split_half_L2_norms)",
            "family_nomination": "largest score; then cosine; then smaller to_state; then role order",
            "common_nomination": "largest minimum family-normalized score; then mean normalized score; then smaller to_state; then role order",
            "support_k": SUPPORT_K,
        },
        "c124_validation_gates": {
            "vector_cosine_min": 0.90,
            "top256_overlap_min": 0.50,
            "support_sign_agreement_min": 0.75,
            "coordinate_peak_clock_within_one_min": 0.70,
            "state_specific_margin_gt": 0.0,
            "role_specific_margin_gt": 0.0,
        },
        "layered_evidence": {
            "level_1": "full increment vector repeatability",
            "level_2": "coordinate peak-clock repeatability",
            "level_3": "target transition and role beat registered wrong-state and wrong-role controls",
        },
        "behavior_ledger": {
            "C115": "standard code passes while reversed code fails; truth fields are descriptive upstream responses",
            "C116": "output protocol is strongly code-conditioned; no general negation behavior claim",
            "C117": "output protocol fails and task is explicit exception-log reading; no default inheritance claim",
        },
        "typed_missingness": {
            "full_token_state": "only seven registered role spans are available",
            "natural_causal_flow": "not tested",
            "jacobian": "not computed",
            "fresh_blind_data": "all source archives existed before C123; C124 is a procedurally sealed re-analysis, not de novo blinded replication",
            "cross_model": "Qwen3 only",
            "human_naturalness": "not independently blind-rated",
        },
        "forbidden_upgrades": [
            "activation increment is not an endogenous operator",
            "role-aligned atlas is not a complete-token transmission graph",
            "activation coordinates are not weights or independent semantic neurons",
            "no attention, MLP, PCA, SVD, manifold, fiber-bundle, curvature, topology, or new-mathematics claim",
        ],
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_hashes": sha_map(paths),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_c123_discovery_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {
        "phase": 1657,
        "campaign": "C123",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": protocol["authorization"],
    }
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def discover() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/internal_contract_audit.json")
    if protocol["authorization"] != "run_c123_discovery_only" or not audit["all_checks_passed"]:
        raise RuntimeError("C123 discovery authorization missing")
    if sha_map({name: Path(path) for name, path in protocol["source_paths"].items()}) != protocol["source_hashes"]:
        raise RuntimeError("C123 source hash drift")

    amendment_path = OUT / "protocol/phase1657_execution_amendment.json"
    amendment = {
        "phase": 1657,
        "campaign": "C123",
        "created_at_utc": now(),
        "reason": "Create the derived analysis directory before the first np.save call after the initial execution stopped before any analysis artifact was written.",
        "original_producer_sha256": protocol["producer_sha256"],
        "repaired_producer_sha256": core.sha(Path(__file__)),
        "unchanged": ["research object", "source hashes", "discovery partitions", "candidate score", "tie breaks", "support size", "C124 gates", "claim boundary"],
    }
    core.save(amendment_path, amendment)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)

    fields_by_family = discovery_fields()
    all_rows: list[dict] = []
    family_tables: dict[str, dict[tuple[str, int], dict]] = {}
    family_nominations = []
    selected_trajectories = np.empty((len(FAMILIES), STATES, DIM), dtype=np.float32)
    selected_increments = np.empty((len(FAMILIES), STATES - 1, DIM), dtype=np.float32)
    coordinate_rows = []

    for family_index, family in enumerate(FAMILIES):
        fields = fields_by_family[family]
        rows, left, right = candidate_table(fields, family)
        rows.sort(key=lambda row: (-row["score"], -row["split_half_cosine"], row["to_state"], row["role_index"]))
        all_rows.extend(rows)
        family_tables[family] = {(row["role"], row["to_state"]): row for row in rows}
        winner = dict(rows[0])
        role_index = int(winner["role_index"])
        transition_index = int(winner["to_state"]) - 1
        trajectory = np.mean(fields[:, role_index], axis=0, dtype=np.float32)
        increments = trajectory[1:] - trajectory[:-1]
        delta = increments[transition_index]
        support = topk(delta)
        winner.update({
            "support_k": SUPPORT_K,
            "support": support,
            "mean_increment_norm": float(np.linalg.norm(delta)),
            "discovery_units": 12,
        })
        family_nominations.append(winner)
        selected_trajectories[family_index] = trajectory
        selected_increments[family_index] = increments
        absolute_order = np.argsort(-np.abs(delta))
        ranks = np.empty(DIM, dtype=np.int32)
        ranks[absolute_order] = np.arange(1, DIM + 1, dtype=np.int32)
        for coordinate in range(DIM):
            coordinate_rows.append({
                "family": family,
                "role": winner["role"],
                "from_state": winner["from_state"],
                "to_state": winner["to_state"],
                "coordinate": coordinate,
                "left_increment": float(left[role_index, transition_index, coordinate]),
                "right_increment": float(right[role_index, transition_index, coordinate]),
                "mean_increment": float(delta[coordinate]),
                "absolute_rank": int(ranks[coordinate]),
                "selected_top256": coordinate in set(support),
            })

    maxima = {family: max(row["score"] for row in family_tables[family].values()) for family in FAMILIES}
    common_rows = []
    for role_index, role in enumerate(ROLES):
        for to_state in range(1, STATES):
            fractions = [family_tables[family][(role, to_state)]["score"] / max(maxima[family], 1e-12) for family in FAMILIES]
            common_rows.append({
                "role": role,
                "role_index": role_index,
                "from_state": to_state - 1,
                "to_state": to_state,
                "minimum_family_score_fraction": float(min(fractions)),
                "mean_family_score_fraction": float(np.mean(fractions)),
                "family_score_fractions": {family: float(value) for family, value in zip(FAMILIES, fractions, strict=True)},
            })
    common_rows.sort(key=lambda row: (-row["minimum_family_score_fraction"], -row["mean_family_score_fraction"], row["to_state"], row["role_index"]))
    common_nomination = dict(common_rows[0])
    common_nomination["family_supports"] = {}
    for family, fields in fields_by_family.items():
        role_index = ROLES.index(common_nomination["role"])
        delta = np.mean(fields[:, role_index, common_nomination["to_state"]] - fields[:, role_index, common_nomination["from_state"]], axis=0, dtype=np.float32)
        common_nomination["family_supports"][family] = topk(delta)

    np.save(OUT / "analysis/discovery_selected_role_trajectories.float32.npy", selected_trajectories)
    np.save(OUT / "analysis/discovery_selected_role_increments.float32.npy", selected_increments)
    core.write_rows(OUT / "analysis/discovery_candidate_table.jsonl", all_rows)
    core.write_rows(OUT / "analysis/discovery_common_candidate_table.jsonl", common_rows)
    core.write_rows(OUT / "analysis/discovery_coordinate_events.jsonl", coordinate_rows)
    nomination = {
        "phase": 1657,
        "campaign": "C123",
        "created_at_utc": now(),
        "status": "discovery_role_transition_coordinate_nominations_frozen",
        "family_nominations": family_nominations,
        "common_nomination": common_nomination,
        "family_order": list(FAMILIES),
        "role_order": list(ROLES),
        "trajectory_sha256": core.sha(OUT / "analysis/discovery_selected_role_trajectories.float32.npy"),
        "increment_sha256": core.sha(OUT / "analysis/discovery_selected_role_increments.float32.npy"),
        "authorization": "freeze_c124_validation_without_reselection",
    }
    core.save(OUT / "protocol/frozen_discovery_nomination.json", nomination)
    checks = {
        "execution_amendment": amendment["original_producer_sha256"] == protocol["producer_sha256"] and amendment["repaired_producer_sha256"] == core.sha(Path(__file__)),
        "source_hashes": sha_map({name: Path(path) for name, path in protocol["source_paths"].items()}) == protocol["source_hashes"],
        "candidate_count": len(all_rows) == len(FAMILIES) * len(ROLES) * (STATES - 1),
        "common_count": len(common_rows) == len(ROLES) * (STATES - 1),
        "family_nominations": len(family_nominations) == len(FAMILIES),
        "support_sizes": all(len(row["support"]) == SUPPORT_K and len(set(row["support"])) == SUPPORT_K for row in family_nominations),
        "trajectory_shape": list(selected_trajectories.shape) == [4, 37, 2560],
        "increment_shape": list(selected_increments.shape) == [4, 36, 2560],
        "telescoping": all(np.allclose(selected_trajectories[index, -1] - selected_trajectories[index, 0], np.sum(selected_increments[index], axis=0), rtol=1e-5, atol=1e-5) for index in range(4)),
        "finite": bool(np.isfinite(selected_trajectories).all() and np.isfinite(selected_increments).all()),
    }
    report = {
        "phase": 1657,
        "campaign": "C123",
        "created_at_utc": now(),
        "status": "discovery_role_transition_atlas_complete",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "family_nominations": [{key: value for key, value in row.items() if key != "support"} for row in family_nominations],
        "common_nomination": {key: value for key, value in common_nomination.items() if key != "family_supports"},
        "claim_boundary": "Discovery-only registered-role response increments. No validation archive was used for nomination; increments are descriptive layer-to-layer changes, not endogenous operators or complete-token causal flow.",
        "authorization": nomination["authorization"],
    }
    core.save(OUT / "analysis/discovery_summary.json", report)
    core.save(OUT / "audit/internal_discovery_audit.json", {
        "phase": 1657,
        "campaign": "C123",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": nomination["authorization"],
    })
    print(json.dumps(report, indent=2))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"contract", "discover"}:
        raise SystemExit("usage: phase1657_c123_role_transition_atlas.py {contract|discover}")
    {"contract": contract, "discover": discover}[sys.argv[1]]()


if __name__ == "__main__":
    main()
