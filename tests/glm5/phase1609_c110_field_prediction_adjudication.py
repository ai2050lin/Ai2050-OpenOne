#!/usr/bin/env python3
"""Phase1609 / C110: adjudicate the prospectively frozen fresh-field predictions."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
C109 = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("fresh_confirmation", "fresh_lockbox")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    audit = core.load(OUT / "audit/independent_capture_audit.json")
    if capture["authorization"] != "run_phase1609_c110_field_prediction_and_transport_tests" or not audit["all_checks_passed"]:
        raise RuntimeError("C110 field adjudication authorization missing")
    field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    roles = protocol["roles"]
    role_index = {role: index for index, role in enumerate(roles)}
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    unit_path = OUT / "analysis/unit_truth_role_state.float32.npy"
    unit_truth = np.lib.format.open_memmap(unit_path, mode="w+", dtype=np.float32, shape=(24, 7, 37, 2560))
    unit_truth[:] = 0.0
    for state in range(37):
        for row_index, row in enumerate(rows):
            coefficient = float(row["truth_factor"]) / 16.0
            u = unit_index[row["unit_id"]]
            for role in roles:
                values = decode_bf16(field[state, lookup[(row_index, role)]])
                unit_truth[u, role_index[role], state] += coefficient * np.mean(values, axis=0, dtype=np.float32)
        if state % 6 == 0 or state == 36:
            unit_truth.flush()
            print(f"[phase1609] derived state {state}/36", flush=True)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        groups[(unit["family"], unit["partition"])].append(index)
    mean_path = OUT / "analysis/mean_truth_role_state.float32.npy"
    mean_truth = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=(2, 2, 7, 37, 2560))
    for family_index, family in enumerate(FAMILIES):
        for partition_index, partition in enumerate(PARTITIONS):
            mean_truth[family_index, partition_index] = np.mean(unit_truth[groups[(family, partition)]], axis=0, dtype=np.float32)
    mean_truth.flush()
    old_mean = np.load(C109 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    prediction = protocol["frozen_field_prediction"]
    state = int(prediction["state"])
    r = role_index[prediction["role"]]
    results = []
    for family_index, family in enumerate(FAMILIES):
        vectors = [np.asarray(mean_truth[family_index, partition_index, r, state], dtype=np.float32) for partition_index in range(2)]
        reference = np.mean(np.asarray(old_mean[family_index, :, r, state], dtype=np.float32), axis=0, dtype=np.float32)
        support = protocol["supports"]["attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"]
        k = len(support)
        support_set = set(support)
        cross_cos = cosine(vectors[0], vectors[1])
        reference_cos = [cosine(vector, reference) for vector in vectors]
        overlaps = [len(topk(vector, k) & support_set) / k for vector in vectors]
        gates = {
            "cross_partition": cross_cos >= float(prediction["cross_fresh_partition_cosine_min"]),
            "reference": all(value >= float(prediction["each_fresh_partition_to_c109_reference_cosine_min"]) for value in reference_cos),
            "support_overlap": all(value >= float(prediction["each_partition_frozen_support_topk_overlap_min"]) for value in overlaps),
        }
        results.append({
            "family": family, "role": prediction["role"], "state": state, "k": k,
            "cross_fresh_partition_cosine": cross_cos,
            "fresh_partition_to_c109_reference_cosine": dict(zip(PARTITIONS, reference_cos, strict=True)),
            "frozen_support_topk_overlap": dict(zip(PARTITIONS, overlaps, strict=True)),
            "norms": dict(zip(PARTITIONS, [float(np.linalg.norm(vector)) for vector in vectors], strict=True)),
            "gates": gates, "prediction_passed": all(gates.values()),
        })
    core.write_rows(OUT / "analysis/fresh_field_prediction_results.jsonl", results)

    trajectory = []
    for family_index, family in enumerate(FAMILIES):
        for role, role_i in role_index.items():
            for state_i in range(37):
                left = np.asarray(mean_truth[family_index, 0, role_i, state_i], dtype=np.float32)
                right = np.asarray(mean_truth[family_index, 1, role_i, state_i], dtype=np.float32)
                trajectory.append({"family": family, "role": role, "state": state_i, "state_kind": "embedding" if state_i == 0 else "hidden_state", "fresh_cross_partition_cosine": cosine(left, right), "fresh_confirmation_norm": float(np.linalg.norm(left)), "fresh_lockbox_norm": float(np.linalg.norm(right))})
    core.write_rows(OUT / "analysis/fresh_role_state_trajectory.jsonl", trajectory)
    report = {
        "phase": 1609, "campaign": "C110", "created_at_utc": now(), "status": "fresh_field_prediction_adjudicated",
        "results": results, "passed_families": [row["family"] for row in results if row["prediction_passed"]],
        "interpretation": "field prediction is independent of transport result; a pass establishes fresh readout stability only",
        "producer_sha256": core.sha(Path(__file__)), "unit_sha256": core.sha(unit_path), "mean_sha256": core.sha(mean_path),
        "authorization": "execute_phase1610_c110_frozen_transport_comparison_regardless_of_field_gate",
    }
    core.save(OUT / "analysis/field_prediction_adjudication.json", report)
    checks = {
        "source": core.sha(OUT / protocol["archive"]["path"]) == capture["raw_sha256"],
        "unit_shape": list(unit_truth.shape) == [24, 7, 37, 2560], "mean_shape": list(mean_truth.shape) == [2, 2, 7, 37, 2560],
        "finite": bool(np.isfinite(unit_truth).all() and np.isfinite(mean_truth).all()), "results": len(results) == 2,
        "trajectory": len(trajectory) == 518, "authorization": "regardless_of_field_gate" in report["authorization"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "audit/internal_field_adjudication_audit.json", {"phase": 1609, "campaign": "C110", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": report["producer_sha256"]})
    print(json.dumps({"checks": checks, "results": results}, indent=2))


if __name__ == "__main__":
    main()
