#!/usr/bin/env python3
"""Phase1605 / C109: basic all-coordinate observation of the fresh role-state field."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
SOURCE = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1601_c108_fresh_coordinate_interventions as c108

PHASE = 1605
CAMPAIGN = "C109"
PARTITIONS = ("prospective_confirmation", "independent_lockbox")
FAMILIES = ("attribute_binding", "agent_patient")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int) -> set[int]:
    if k >= values.size:
        return set(range(values.size))
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def sign_agreement(left: np.ndarray, right: np.ndarray) -> float:
    visible = (left != 0.0) | (right != 0.0)
    return 1.0 if not np.any(visible) else float(np.mean(np.sign(left[visible]) == np.sign(right[visible])))


def energy_fraction(vector: np.ndarray, support: list[int]) -> float:
    total = float(np.dot(vector, vector))
    return 0.0 if total <= 1e-20 else float(np.dot(vector[support], vector[support]) / total)


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    capture_audit = core.load(OUT / "audit/independent_capture_audit.json")
    if capture["authorization"] != "run_phase1605_c109_basic_coordinate_observation" or not capture_audit["all_checks_passed"]:
        raise RuntimeError("C109 observation authorization missing")
    field_path = OUT / protocol["archive"]["path"]
    field = np.load(field_path, mmap_mode="r")
    rows = core.rows(SOURCE / "compiled/qwen3.jsonl")
    units = core.rows(SOURCE / "material/units.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    roles = protocol["roles"]
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    role_index = {role: index for index, role in enumerate(roles)}
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))

    unit_path = OUT / "analysis/unit_truth_role_state.float32.npy"
    unit_truth = np.lib.format.open_memmap(unit_path, mode="w+", dtype=np.float32, shape=(len(units), len(roles), field.shape[0], field.shape[2]))
    unit_truth[:] = 0.0
    for state in range(field.shape[0]):
        for row_index, row in enumerate(rows):
            coefficient = float(row["truth_factor"]) / 16.0
            u = unit_index[row["unit_id"]]
            for role in roles:
                values = decode_bf16(field[state, lookup[(row_index, role)], :])
                unit_truth[u, role_index[role], state] += coefficient * np.mean(values, axis=0, dtype=np.float32)
        if state % 6 == 0 or state == field.shape[0] - 1:
            unit_truth.flush()
            print(f"[phase1605] derived truth field state {state}/{field.shape[0] - 1}", flush=True)

    mean_path = OUT / "analysis/mean_truth_role_state.float32.npy"
    mean_truth = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=(len(FAMILIES), len(PARTITIONS), len(roles), field.shape[0], field.shape[2]))
    unit_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        unit_groups[(unit["family"], unit["partition"])].append(index)
    for family_index, family in enumerate(FAMILIES):
        for partition_index, partition in enumerate(PARTITIONS):
            mean_truth[family_index, partition_index] = np.mean(unit_truth[unit_groups[(family, partition)]], axis=0, dtype=np.float32)
    mean_truth.flush()

    supports = protocol["supports"]
    family_support = {
        "attribute_binding": supports["attribute_binding_k256"],
        "agent_patient": supports["agent_patient_k128"],
    }
    wrong_support = {
        "attribute_binding": supports["attribute_wrong_agent_k256"],
        "agent_patient": supports["agent_wrong_attribute_k128"],
    }
    trajectory = []
    for family_index, family in enumerate(FAMILIES):
        k = len(family_support[family])
        frozen = set(family_support[family])
        for r, role in enumerate(roles):
            for state in range(field.shape[0]):
                left = np.asarray(mean_truth[family_index, 0, r, state], dtype=np.float32)
                right = np.asarray(mean_truth[family_index, 1, r, state], dtype=np.float32)
                unit_overlaps = {}
                for partition in PARTITIONS:
                    values = [
                        len(topk(np.asarray(unit_truth[index, r, state]), k) & frozen) / k
                        for index in unit_groups[(family, partition)]
                    ]
                    unit_overlaps[partition] = float(np.mean(values))
                left_top = topk(left, k)
                right_top = topk(right, k)
                trajectory.append({
                    "family": family,
                    "role": role,
                    "state": state,
                    "state_kind": "embedding" if state == 0 else "hidden_state",
                    "k": k,
                    "prospective_norm": float(np.linalg.norm(left)),
                    "lockbox_norm": float(np.linalg.norm(right)),
                    "cross_partition_cosine": cosine(left, right),
                    "cross_partition_sign_agreement": sign_agreement(left, right),
                    "cross_partition_topk_overlap": len(left_top & right_top) / k,
                    "prospective_frozen_support_overlap": len(left_top & frozen) / k,
                    "lockbox_frozen_support_overlap": len(right_top & frozen) / k,
                    "prospective_frozen_support_energy_fraction": energy_fraction(left, family_support[family]),
                    "lockbox_frozen_support_energy_fraction": energy_fraction(right, family_support[family]),
                    "prospective_wrong_support_energy_fraction": energy_fraction(left, wrong_support[family]),
                    "lockbox_wrong_support_energy_fraction": energy_fraction(right, wrong_support[family]),
                    "mean_unit_frozen_support_overlap": unit_overlaps,
                })
    core.write_rows(OUT / "analysis/role_state_truth_trajectory.jsonl", trajectory)

    source_protocol = core.load(SOURCE / "protocol/preregistration.json")
    pairs = c108.build_pairs(rows, source_protocol)
    case_index = {row["case_id"]: index for index, row in enumerate(rows)}
    pair_energy = []
    for pair in pairs:
        family = pair["family"]
        state = 19
        role = pair["role"]
        donor_index = case_index[pair["donor"]["case_id"]]
        recipient_index = case_index[pair["recipient"]["case_id"]]
        donor = np.mean(decode_bf16(field[state, lookup[(donor_index, role)], :]), axis=0, dtype=np.float32)
        recipient = np.mean(decode_bf16(field[state, lookup[(recipient_index, role)], :]), axis=0, dtype=np.float32)
        delta = donor - recipient
        target_energy = float(np.dot(delta[family_support[family]], delta[family_support[family]]))
        wrong_energy = float(np.dot(delta[wrong_support[family]], delta[wrong_support[family]]))
        whole_energy = float(np.dot(delta, delta))
        pair_energy.append({
            "pair_id": pair["pair_id"],
            "unit_id": pair["unit_id"],
            "family": family,
            "partition": pair["partition"],
            "code": pair["code"],
            "surface_factor": pair["surface_factor"],
            "distractor_factor": pair["distractor_factor"],
            "state": state,
            "role": role,
            "k": len(family_support[family]),
            "target_support_energy": target_energy,
            "same_k_wrong_support_energy": wrong_energy,
            "whole_state_energy": whole_energy,
            "target_to_wrong_energy_ratio": target_energy / max(wrong_energy, 1e-20),
            "target_energy_fraction": target_energy / max(whole_energy, 1e-20),
            "wrong_energy_fraction": wrong_energy / max(whole_energy, 1e-20),
        })
    core.write_rows(OUT / "analysis/c108_pair_support_energy.jsonl", pair_energy)
    pair_summary = []
    for family in FAMILIES:
        for partition in PARTITIONS:
            for code in (1, -1):
                selected = [row for row in pair_energy if row["family"] == family and row["partition"] == partition and row["code"] == code]
                pair_summary.append({
                    "family": family,
                    "partition": partition,
                    "code": code,
                    "pairs": len(selected),
                    "median_target_to_wrong_energy_ratio": float(np.median([row["target_to_wrong_energy_ratio"] for row in selected])),
                    "median_target_energy_fraction": float(np.median([row["target_energy_fraction"] for row in selected])),
                    "median_wrong_energy_fraction": float(np.median([row["wrong_energy_fraction"] for row in selected])),
                    "target_energy_exceeds_wrong_pairs": int(sum(row["target_support_energy"] > row["same_k_wrong_support_energy"] for row in selected)),
                })
    core.write_rows(OUT / "analysis/c108_pair_support_energy_summary.jsonl", pair_summary)

    candidate_rows = [row for row in trajectory if row["role"] == "query_anchor" and row["state"] == 19]
    best_rows = {}
    for family in FAMILIES:
        eligible = [row for row in trajectory if row["family"] == family and row["state"] > 0]
        best_rows[family] = sorted(eligible, key=lambda row: (row["cross_partition_cosine"], min(row["prospective_norm"], row["lockbox_norm"])), reverse=True)[:10]
    summary = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "basic_full_coordinate_observation_complete",
        "object": protocol["object"],
        "candidate_query_anchor_state19": candidate_rows,
        "descriptive_top_stable_role_states": best_rows,
        "pair_energy_summary": pair_summary,
        "support_overlap_counts": protocol["support_overlap_counts"],
        "behavior_boundary": capture["behavior"],
        "interpretation_rules": {
            "stable_field_not_fixed_support": "high cross-partition field cosine with low frozen-support overlap would favor a distributed/dynamic field over a fixed coordinate identity",
            "energy_control": "same K was frozen before C108; C109 now reports actual donor-recipient energy but does not rerun a post-hoc energy-matched intervention",
            "no_independent_confirmation": "both partitions were already exposed in C108; all C109 locators require future lexical confirmation",
        },
        "claim_boundary": protocol["claim_boundary"],
        "artifacts": {
            "unit_truth": str(unit_path.relative_to(ROOT)),
            "mean_truth": str(mean_path.relative_to(ROOT)),
            "trajectory": str((OUT / "analysis/role_state_truth_trajectory.jsonl").relative_to(ROOT)),
            "pair_energy": str((OUT / "analysis/c108_pair_support_energy.jsonl").relative_to(ROOT)),
        },
        "authorization": "run_phase1606_c109_heatmap_synthesis_and_closure",
    }
    core.save(OUT / "analysis/basic_observation_summary.json", summary)
    checks = {
        "source_archive": core.sha(field_path) == capture["raw_sha256"],
        "unit_shape": list(unit_truth.shape) == [24, 7, 37, 2560],
        "mean_shape": list(mean_truth.shape) == [2, 2, 7, 37, 2560],
        "finite": bool(np.isfinite(unit_truth).all() and np.isfinite(mean_truth).all()),
        "trajectory": len(trajectory) == 2 * 7 * 37,
        "candidate": len(candidate_rows) == 2,
        "pairs": len(pair_energy) == 192 and len(pair_summary) == 8,
        "factorial_units": all(len(unit_groups[(family, partition)]) == 6 for family in FAMILIES for partition in PARTITIONS),
        "artifacts": all(Path(ROOT / value).exists() for value in summary["artifacts"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": summary["authorization"]}
    core.save(OUT / "audit/basic_observation_internal_audit.json", report)
    print(json.dumps({"checks": checks, "candidate_query_anchor_state19": candidate_rows, "pair_energy_summary": pair_summary}, indent=2))


if __name__ == "__main__":
    main()
