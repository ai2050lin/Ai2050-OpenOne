#!/usr/bin/env python3
"""Phase1620 / C113: adjudicate fourth-lexicon full-field predictions before interventions."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
C110 = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


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
    if capture["authorization"] != "run_phase1620_c113_field_adjudication" or not audit["all_checks_passed"]:
        raise RuntimeError("C113 field adjudication authorization missing")
    field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    roles = protocol["roles"]
    families = protocol["families"]
    partitions = protocol["partitions"]
    role_index = {role: index for index, role in enumerate(roles)}
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    unit_path = OUT / "analysis/unit_truth_role_state.float32.npy"
    mean_path = OUT / "analysis/mean_truth_role_state.float32.npy"
    if unit_path.exists() or mean_path.exists():
        raise RuntimeError("C113 derived field already exists")
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
            print(f"[phase1620] derived state {state}/36", flush=True)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        groups[(unit["family"], unit["partition"])].append(index)
    mean_truth = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=(2, 2, 7, 37, 2560))
    for family_index, family in enumerate(families):
        for partition_index, partition in enumerate(partitions):
            mean_truth[family_index, partition_index] = np.mean(unit_truth[groups[(family, partition)]], axis=0, dtype=np.float32)
    mean_truth.flush()
    old_mean = np.load(C110 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    prediction = protocol["frozen_field_prediction"]
    state = int(prediction["state"])
    role_i = role_index[prediction["role"]]
    results = []
    for family_index, family in enumerate(families):
        vectors = [np.asarray(mean_truth[family_index, partition_index, role_i, state], dtype=np.float32) for partition_index in range(2)]
        reference = np.mean(np.asarray(old_mean[family_index, :, role_i, state], dtype=np.float32), axis=0, dtype=np.float32)
        support = protocol["supports"]["attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"]
        support_set = set(support)
        cross_cos = cosine(vectors[0], vectors[1])
        reference_cos = [cosine(vector, reference) for vector in vectors]
        overlaps = [len(topk(vector, len(support)) & support_set) / len(support) for vector in vectors]
        gates = {
            "cross_partition": cross_cos >= float(prediction["cross_partition_cosine_min"]),
            "reference": all(value >= float(prediction["each_partition_to_c110_reference_cosine_min"]) for value in reference_cos),
            "support_overlap": all(value >= float(prediction["each_partition_frozen_support_topk_overlap_min"]) for value in overlaps),
        }
        results.append({
            "family": family, "role": prediction["role"], "state": state, "k": len(support),
            "cross_partition_cosine": cross_cos,
            "partition_to_c110_reference_cosine": dict(zip(partitions, reference_cos, strict=True)),
            "frozen_support_topk_overlap": dict(zip(partitions, overlaps, strict=True)),
            "norms": dict(zip(partitions, [float(np.linalg.norm(vector)) for vector in vectors], strict=True)),
            "gates": gates, "prediction_passed": all(gates.values()),
        })
    result_path = OUT / "analysis/field_prediction_results.jsonl"
    core.write_rows(result_path, results)
    trajectory = []
    for family_index, family in enumerate(families):
        for role, r in role_index.items():
            for state_i in range(37):
                left = np.asarray(mean_truth[family_index, 0, r, state_i], dtype=np.float32)
                right = np.asarray(mean_truth[family_index, 1, r, state_i], dtype=np.float32)
                trajectory.append({
                    "family": family, "role": role, "state": state_i,
                    "state_kind": "embedding" if state_i == 0 else "hidden_state",
                    "cross_partition_cosine": cosine(left, right),
                    "confirmation_norm": float(np.linalg.norm(left)), "lockbox_norm": float(np.linalg.norm(right)),
                })
    trajectory_path = OUT / "analysis/role_state_trajectory.jsonl"
    core.write_rows(trajectory_path, trajectory)
    checks = {
        "source": core.sha(OUT / protocol["archive"]["path"]) == capture["raw_sha256"],
        "unit_shape": list(unit_truth.shape) == [24, 7, 37, 2560],
        "mean_shape": list(mean_truth.shape) == [2, 2, 7, 37, 2560],
        "finite": bool(np.isfinite(unit_truth).all() and np.isfinite(mean_truth).all()),
        "results": len(results) == 2, "trajectory": len(trajectory) == 518,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1620, "campaign": "C113", "created_at_utc": now(), "status": "fourth_lexicon_field_prediction_adjudicated",
        "results": results, "passed_families": [row["family"] for row in results if row["prediction_passed"]],
        "interpretation": "field stability is an upstream readout result and remains separate from intervention leverage",
        "checks": checks, "producer_sha256": core.sha(Path(__file__)), "unit_sha256": core.sha(unit_path),
        "mean_sha256": core.sha(mean_path), "results_sha256": core.sha(result_path), "trajectory_sha256": core.sha(trajectory_path),
        "authorization": "execute_phase1621_c113_coordinate_and_role_interventions_regardless_of_field_gate",
    }
    core.save(OUT / "analysis/field_adjudication.json", report)
    print(json.dumps({"checks": checks, "results": results}, indent=2))


if __name__ == "__main__":
    main()
