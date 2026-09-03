#!/usr/bin/env python3
"""Phase1467: capture C079 holdouts and validate frozen full-vector candidates."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1465_c079_discovery_full_field_capture as capture_module
import phase1466_c079_discovery_basic_observation_and_freeze as observation

CONTRACT = TESTS / "result/phase1463_c079_aggregate_observation_contract"
BEHAVIOR = TESTS / "result/phase1464_c079_behavior"
DISCOVERY = TESTS / "result/phase1466_c079_discovery_basic_observation_and_freeze"
OUT = TESTS / "result/phase1467_c079_holdout_capture_and_validation"


def scalar_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def ordered_holdout_cases(protocol: dict, eligible: list[dict], compiled: dict[str, dict]) -> list[dict]:
    result = []
    for group in eligible:
        if group["partition"] not in protocol["validation"]["partitions"]:
            continue
        for surface in protocol["surfaces"]:
            for cell in protocol["cells"]:
                result.append(compiled[group[f"{surface}_{cell}"]])
    return result


def validate(field: np.ndarray, index: list[dict], protocol: dict, manifest: dict, vectors: np.lib.npyio.NpzFile) -> list[dict]:
    roles = protocol["role_slots"]
    cells = list(protocol["cells"])
    lookup = {(row["partition"], row["family"], row["index"], row["record_relation_id"], row["surface"], row["cell"]): row["row_index"] for row in index}
    set_keys = sorted({(row["partition"], row["family"], row["index"], row["record_relation_id"]) for row in index})
    thresholds = manifest["validation_thresholds"]
    results = []
    for candidate in manifest["candidates"]:
        relation = candidate["relation"]
        state = candidate["state"]
        role = candidate["role"]
        role_index = roles.index(role)
        for split in protocol["validation"]["partitions"]:
            current_sets = [key for key in set_keys if key[0] == split and key[3] == relation]
            surface_stats = {}
            holdout_means = {}
            for surface in protocol["surfaces"]:
                samples = {factor: [] for factor in observation.FACTORS}
                for _, family, index_value, _ in current_sets:
                    row_ids = [lookup[(split, family, index_value, relation, surface, cell)] for cell in cells]
                    block = np.asarray(field[row_ids, state, role_index], dtype=np.float32)
                    for factor, bit in observation.FACTORS.items():
                        samples[factor].append(observation.factor_effect(block, cells, bit))
                stacks = {factor: np.stack(values, axis=0) for factor, values in samples.items()}
                means = {factor: np.mean(stack, axis=0, dtype=np.float32) for factor, stack in stacks.items()}
                discovery_vector = vectors[f"{candidate['candidate_id']}__{surface}"]
                relation_stack = stacks["relation_label"]
                cosine_to_discovery = scalar_cosine(means["relation_label"], discovery_vector)
                direction_to_discovery = float(np.mean([scalar_cosine(value, discovery_vector) for value in relation_stack]))
                nuisance_norm = max(float(np.linalg.norm(means["entity_nuisance"])), float(np.linalg.norm(means["object_nuisance"])), 1e-12)
                selectivity = float(np.linalg.norm(means["relation_label"]) / nuisance_norm)
                passed = cosine_to_discovery >= thresholds["cosine_to_discovery_each_surface_min"] and direction_to_discovery >= thresholds["direction_to_discovery_min"] and selectivity >= thresholds["selectivity_ratio_min"]
                surface_stats[surface] = {"sample_count": len(current_sets), "cosine_to_discovery": cosine_to_discovery, "direction_to_discovery": direction_to_discovery, "selectivity_ratio": selectivity, "passed": passed}
                holdout_means[surface] = means["relation_label"]
            cross = scalar_cosine(holdout_means[protocol["surfaces"][0]], holdout_means[protocol["surfaces"][1]])
            split_passed = all(value["passed"] for value in surface_stats.values()) and cross >= thresholds["holdout_cross_surface_cosine_min"]
            results.append({"candidate_id": candidate["candidate_id"], "relation": relation, "state": state, "role": role, "split": split, "surface": surface_stats, "holdout_cross_surface_cosine": cross, "split_passed": split_passed})
    return results


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1467 exists")
    discovery = core.load(DISCOVERY / "analysis/final.json")
    discovery_audit = core.load(DISCOVERY / "audit/independent_final_audit.json")
    manifest = core.load(DISCOVERY / "frozen/candidate_manifest.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if discovery["authorization"] != "run_phase1467_c079_holdout_capture_and_validation" or not discovery_audit["all_checks_passed"]:
        raise RuntimeError("Phase1466 did not authorize holdout")
    eligible = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    cases = ordered_holdout_cases(protocol, eligible, compiled)
    expected = sum(row["partition"] in protocol["validation"]["partitions"] for row in eligible) * len(protocol["surfaces"]) * len(protocol["cells"])
    if len(cases) != expected or expected != 2208:
        raise RuntimeError((len(cases), expected))
    capture_module.OUT = OUT
    index, runtime = capture_module.capture(cases, protocol)
    temporary = OUT / "raw/discovery_role_field.float16.npy"
    raw_path = OUT / "raw/holdout_role_field.float16.npy"
    temporary.replace(raw_path)
    core.write_rows(OUT / "raw/holdout_role_field_index.jsonl", index)
    raw_sha = core.sha(raw_path)
    index_sha = core.sha(OUT / "raw/holdout_role_field_index.jsonl")
    behavior = {row["case_id"]: row for row in core.rows(BEHAVIOR / "raw/active_behavior.jsonl")}
    capture_checks = {
        "count": len(index) == expected == runtime["shape"][0],
        "shape": runtime["shape"][1:] == [37, 9, 2560],
        "splits": Counter(row["partition"] for row in index) == {"confirmation": 1088, "lockbox": 1120},
        "behavior": all(behavior[row["case_id"]]["correct"] for row in index),
        "prediction": all(row["capture_prediction"] == row["gold_position"] for row in index),
        "finite": runtime["finite_during_capture"] and all(math.isfinite(value) for row in index for value in row["capture_scores"]),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    if not all(capture_checks.values()):
        raise RuntimeError({key: value for key, value in capture_checks.items() if not value})
    field = np.load(raw_path, mmap_mode="r")
    vectors = np.load(DISCOVERY / "frozen/discovery_candidate_mean_vectors.npz")
    validation = validate(field, index, protocol, manifest, vectors)
    core.write_rows(OUT / "analysis/candidate_holdout_validation.jsonl", validation)
    robust = []
    for candidate in manifest["candidates"]:
        rows = [row for row in validation if row["candidate_id"] == candidate["candidate_id"]]
        if len(rows) == 2 and all(row["split_passed"] for row in rows):
            robust.append(candidate["candidate_id"])
    metadata = {"phase": 1467, "campaign": "C079", "shape": runtime["shape"], "dtype": "float16", "file_size_bytes": raw_path.stat().st_size, "raw_sha256": raw_sha, "index_sha256": index_sha, "capture_checks": capture_checks, "candidate_count": len(manifest["candidates"]), "robust_candidate_count": len(robust), "robust_candidates": robust, "robust_role_counts": dict(Counter(row["role"] for row in manifest["candidates"] if row["candidate_id"] in robust)), "robust_relation_counts": dict(Counter(row["relation"] for row in manifest["candidates"] if row["candidate_id"] in robust)), "thresholds": manifest["validation_thresholds"], "discovery_freeze_sha256": manifest["freeze_sha256"], "runtime": runtime, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/holdout_summary.json", metadata)
    core.save(OUT / "analysis/final.json", {"phase": 1467, "campaign": "C079", "holdout_validation_complete": True, "robust_candidate_count": len(robust), "robust_candidates": robust, "authorization": "run_phase1468_c079_campaign_closure"})
    print(json.dumps({key: value for key, value in metadata.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
