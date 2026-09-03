#!/usr/bin/env python3
"""Phase1613 / C111: observe value identity and role geometry in frozen C109-C110 fields."""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C109 = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
C110 = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
OUT = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("fresh_confirmation", "fresh_lockbox")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def build_pairs(rows: list[dict]) -> list[dict]:
    by_unit: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_unit[row["unit_id"]].append((index, row))
    pairs = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        for surface, distractor, code in itertools.product((1, -1), repeat=3):
            def pick(truth: int) -> tuple[int, dict]:
                return next(item for item in unit_rows if (item[1]["truth_factor"], item[1]["surface_factor"], item[1]["distractor_factor"], item[1]["code"]) == (truth, surface, distractor, code))
            recipient_index, recipient = pick(-1)
            donor_index, donor = pick(1)
            pairs.append({
                "pair_id": f"c110-pair-{len(pairs):04d}",
                "unit_id": unit_id,
                "family": recipient["family"],
                "partition": recipient["partition"],
                "surface_factor": surface,
                "distractor_factor": distractor,
                "code": code,
                "recipient_index": recipient_index,
                "donor_index": donor_index,
            })
    return pairs


def main() -> None:
    contract = core.load(OUT / "protocol/preregistration.json")
    contract_audit = core.load(OUT / "audit/independent_contract_audit.json")
    if contract["authorization"] != "run_phase1613_c111_read_only_observation" or not contract_audit["all_checks_passed"]:
        raise RuntimeError("C111 observation authorization missing")
    c110_protocol = core.load(C110 / "protocol/preregistration.json")
    adapter = core.load(C110 / "protocol/transport_adapter.json")
    rows = core.rows(C110 / "compiled/qwen3.jsonl")
    transport_rows = {row["pair_id"]: row for row in core.rows(C110 / "analysis/fresh_transport_results.jsonl")}
    pairs = build_pairs(rows)
    if len(pairs) != 192 or set(transport_rows) != {pair["pair_id"] for pair in pairs}:
        raise RuntimeError("pair reconstruction mismatch")
    manifest = core.rows(C110 / "protocol/role_occurrence_manifest.jsonl")
    occurrence_lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        occurrence_lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    archive = np.load(C110 / c110_protocol["archive"]["path"], mmap_mode="r")
    mean_c110 = np.load(C110 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    mean_c109 = np.load(C109 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    roles = c110_protocol["roles"]
    role_index = {role: index for index, role in enumerate(roles)}

    pair_geometry = []
    for pair in pairs:
        result = transport_rows[pair["pair_id"]]
        family = pair["family"]
        family_index = FAMILIES.index(family)
        partition_index = PARTITIONS.index(pair["partition"])
        support_name = "attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"
        support = np.asarray(c110_protocol["supports"][support_name], dtype=np.int64)
        permutation = np.asarray(adapter["coordinate_permutations"][family], dtype=np.int64)
        recipient_occurrences = occurrence_lookup[(pair["recipient_index"], "query_anchor")]
        donor_occurrences = occurrence_lookup[(pair["donor_index"], "query_anchor")]
        if len(recipient_occurrences) != len(donor_occurrences):
            raise RuntimeError(pair["pair_id"])
        recipient = decode_bf16(archive[19, recipient_occurrences])
        donor = decode_bf16(archive[19, donor_occurrences])
        target_movement = donor[:, support] - recipient[:, support]
        permuted_movement = donor[:, permutation[support]] - recipient[:, support]
        field = np.asarray(mean_c110[family_index, partition_index, role_index["query_anchor"], 19, support], dtype=np.float32)
        repeated_field = np.broadcast_to(field, target_movement.shape)
        focus_recipient = decode_bf16(archive[19, occurrence_lookup[(pair["recipient_index"], "focus_record")]])
        focus_donor = decode_bf16(archive[19, occurrence_lookup[(pair["donor_index"], "focus_record")]])
        target_gain = float(result["modes"]["frozen_support"]["truth_direction_gain"])
        permuted_gain = float(result["modes"]["coordinate_permuted"]["truth_direction_gain"])
        whole_gain = float(result["modes"]["whole_query_anchor"]["truth_direction_gain"])
        multi_gain = float(result["modes"]["whole_query_anchor_plus_focus_record"]["truth_direction_gain"])
        pair_geometry.append({
            **{key: pair[key] for key in ("pair_id", "unit_id", "family", "partition", "surface_factor", "distractor_factor", "code")},
            "k": len(support),
            "query_span": len(recipient_occurrences),
            "target_movement_l2": float(np.linalg.norm(target_movement)),
            "permuted_movement_l2": float(np.linalg.norm(permuted_movement)),
            "target_vs_permuted_movement_cosine": cosine(target_movement, permuted_movement),
            "target_movement_field_cosine": cosine(target_movement, repeated_field),
            "permuted_movement_field_cosine": cosine(permuted_movement, repeated_field),
            "target_output_gain": target_gain,
            "permuted_output_gain": permuted_gain,
            "target_minus_permuted_output_gain": target_gain - permuted_gain,
            "whole_query_l2": float(np.linalg.norm(donor - recipient)),
            "focus_record_l2": float(np.linalg.norm(focus_donor - focus_recipient)),
            "whole_query_output_gain": whole_gain,
            "query_plus_record_output_gain": multi_gain,
            "focus_record_increment": multi_gain - whole_gain,
            "whole_query_truth_flip": bool(result["modes"]["whole_query_anchor"]["truth_flip"]),
            "query_plus_record_truth_flip": bool(result["modes"]["whole_query_anchor_plus_focus_record"]["truth_flip"]),
        })
    core.write_rows(OUT / "analysis/pair_value_role_geometry.jsonl", pair_geometry)

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in pair_geometry:
        grouped[(row["family"], row["partition"], row["code"])].append(row)
    pair_summary = []
    for family, partition, code in sorted(grouped):
        selected = grouped[(family, partition, code)]
        pair_summary.append({
            "family": family,
            "partition": partition,
            "code": code,
            "pairs": len(selected),
            "median_target_vs_permuted_movement_cosine": med([row["target_vs_permuted_movement_cosine"] for row in selected]),
            "median_target_movement_field_cosine": med([row["target_movement_field_cosine"] for row in selected]),
            "median_permuted_movement_field_cosine": med([row["permuted_movement_field_cosine"] for row in selected]),
            "median_permuted_to_target_l2_ratio": med([row["permuted_movement_l2"] / max(row["target_movement_l2"], 1e-12) for row in selected]),
            "target_output_gain_gt_permuted_pairs": int(sum(row["target_output_gain"] > row["permuted_output_gain"] for row in selected)),
            "median_target_minus_permuted_output_gain": med([row["target_minus_permuted_output_gain"] for row in selected]),
            "positive_focus_record_increment_pairs": int(sum(row["focus_record_increment"] > 0 for row in selected)),
            "median_focus_record_increment": med([row["focus_record_increment"] for row in selected]),
            "additional_truth_flips": int(sum(row["query_plus_record_truth_flip"] and not row["whole_query_truth_flip"] for row in selected)),
        })
    core.write_rows(OUT / "analysis/pair_value_role_geometry_summary.jsonl", pair_summary)

    trajectory = []
    for family_index, family in enumerate(FAMILIES):
        for role, role_i in role_index.items():
            for state in range(37):
                old_vector = np.mean(np.asarray(mean_c109[family_index, :, role_i, state], dtype=np.float32), axis=0, dtype=np.float32)
                new_vectors = np.asarray(mean_c110[family_index, :, role_i, state], dtype=np.float32)
                new_vector = np.mean(new_vectors, axis=0, dtype=np.float32)
                trajectory.append({
                    "family": family,
                    "role": role,
                    "state": state,
                    "state_kind": "embedding" if state == 0 else "hidden_state",
                    "c109_c110_mean_cosine": cosine(old_vector, new_vector),
                    "c109_mean_norm": float(np.linalg.norm(old_vector)),
                    "c110_mean_norm": float(np.linalg.norm(new_vector)),
                    "c110_cross_partition_cosine": cosine(new_vectors[0], new_vectors[1]),
                    "c110_confirmation_norm": float(np.linalg.norm(new_vectors[0])),
                    "c110_lockbox_norm": float(np.linalg.norm(new_vectors[1])),
                })
    core.write_rows(OUT / "analysis/cross_archive_role_state_trajectory.jsonl", trajectory)

    role_matrix = []
    for family_index, family in enumerate(FAMILIES):
        vectors = {
            role: np.mean(np.asarray(mean_c110[family_index, :, role_i, 19], dtype=np.float32), axis=0, dtype=np.float32)
            for role, role_i in role_index.items()
        }
        for left in roles:
            for right in roles:
                role_matrix.append({"family": family, "state": 19, "left_role": left, "right_role": right, "cosine": cosine(vectors[left], vectors[right])})
    core.write_rows(OUT / "analysis/state19_role_cosine_matrix.jsonl", role_matrix)

    locators = []
    for family in FAMILIES:
        for role in roles:
            selected = [row for row in trajectory if row["family"] == family and row["role"] == role and row["state"] > 0]
            within = [row["state"] for row in selected if row["c110_cross_partition_cosine"] >= 0.9 and min(row["c110_confirmation_norm"], row["c110_lockbox_norm"]) >= 1.0]
            across = [row["state"] for row in selected if row["c109_c110_mean_cosine"] >= 0.9 and min(row["c109_mean_norm"], row["c110_mean_norm"]) >= 1.0]
            locators.append({
                "family": family,
                "role": role,
                "earliest_c110_partition_stable_high_amplitude_state": min(within) if within else None,
                "earliest_c109_c110_stable_high_amplitude_state": min(across) if across else None,
            })
    core.write_rows(OUT / "analysis/role_state_descriptive_locators.jsonl", locators)

    report = {
        "phase": 1613,
        "campaign": "C111",
        "created_at_utc": now(),
        "status": "read_only_value_identity_role_coalition_observation_complete",
        "pair_summary": pair_summary,
        "locators": locators,
        "planned_missingness": contract["planned_missingness"],
        "interpretation_boundary": "descriptive full-coordinate observation from frozen archives; no new model behavior, intervention, support discovery, minimality, necessity, or universal mechanism claim",
        "authorization": "run_phase1614_c111_synthesis_heatmap_and_closure",
    }
    core.save(OUT / "analysis/observation_report.json", report)
    checks = {
        "pairs": len(pair_geometry) == 192,
        "summaries": len(pair_summary) == 8 and all(row["pairs"] == 24 for row in pair_summary),
        "trajectory": len(trajectory) == 2 * 7 * 37,
        "role_matrix": len(role_matrix) == 2 * 7 * 7,
        "locators": len(locators) == 14,
        "finite_pairs": all(np.isfinite(value) for row in pair_geometry for key, value in row.items() if isinstance(value, float)),
        "finite_trajectory": all(np.isfinite(value) for row in trajectory for key, value in row.items() if isinstance(value, float)),
        "no_new_model": contract["model_run"].startswith("forbidden"),
        "sources_unchanged": all(core.sha(Path(contract["source_paths"][name])) == digest for name, digest in contract["source_hashes"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    audit = {"phase": 1613, "campaign": "C111", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "pair_sha256": core.sha(OUT / "analysis/pair_value_role_geometry.jsonl"), "trajectory_sha256": core.sha(OUT / "analysis/cross_archive_role_state_trajectory.jsonl")}
    core.save(OUT / "audit/internal_observation_audit.json", audit)
    print(json.dumps({"checks": checks, "pair_summary": pair_summary, "locators": locators}, indent=2))


if __name__ == "__main__":
    main()
