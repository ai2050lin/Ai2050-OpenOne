#!/usr/bin/env python3
"""Independent raw-array audit for Phase1143."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE = 1143
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1143_ground_truth_mechanism_calibration"
PRIMARY = ROOT / "tests/glm5/phase1143_ground_truth_mechanism_calibration.py"
SPLITS = ("discovery", "confirmation")
N_ITEMS = 32
N_RELATIONS = 4
ITEMS_PER_RELATION = N_ITEMS // N_RELATIONS
N_LAYERS = 12
PAYLOAD_LAYER = 7
EPSILON = 1e-7


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else float("nan")


def relation_of(item: int) -> int:
    return item // ITEMS_PER_RELATION


def expected_mapping(metadata: dict[str, Any], config_id: str, context: int, scenario: str) -> list[int] | None:
    if scenario in {"linear_shared_only", "nonlinear_shared_only"}:
        return None
    if scenario != "linear_permuted_payload":
        return list(range(N_ITEMS))
    permutation = metadata["permutations"][f"{config_id}.context{context}"]
    inverse = [0] * N_ITEMS
    for donor, payload_id in enumerate(permutation):
        inverse[int(payload_id)] = donor
    return inverse


def recompute_metrics(
    effect: np.ndarray,
    endpoint: np.ndarray,
    target_positive: np.ndarray,
    expected: list[int] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "finite_fraction": float(np.mean(np.isfinite(effect) & np.isfinite(endpoint))),
        "within_relation_spread_max": 0.0,
        "same_relation_advantage_median": 0.0,
        "same_relation_advantage_positive_fraction": 0.0,
        "expected_top1_within_relation_fraction": 0.0,
        "identity_top1_within_relation_fraction": 0.0,
        "expected_endpoint_flip_fraction": 0.0,
        "expected_reconstruction_max_abs_error": None,
    }
    spreads = []
    identity = []
    for target in range(N_ITEMS):
        relation = relation_of(target)
        donors = list(range(relation * ITEMS_PER_RELATION, (relation + 1) * ITEMS_PER_RELATION))
        values = effect[target, donors]
        spreads.append(float(np.max(values) - np.min(values)))
        own = donors.index(target)
        identity.append(bool(values[own] > np.max(np.delete(values, own)) + EPSILON))
    result["within_relation_spread_max"] = float(np.max(spreads))
    result["identity_top1_within_relation_fraction"] = float(np.mean(identity))
    if expected is None:
        return result
    advantages = []
    top1 = []
    flips = []
    reconstruction = []
    for target, donor in enumerate(expected):
        relation = relation_of(target)
        others = [
            candidate
            for candidate in range(relation * ITEMS_PER_RELATION, (relation + 1) * ITEMS_PER_RELATION)
            if candidate != donor
        ]
        expected_value = float(effect[target, donor])
        other_values = [float(effect[target, candidate]) for candidate in others]
        advantages.append(expected_value - median(other_values))
        top1.append(expected_value > max(other_values) + EPSILON)
        flips.append(float(endpoint[target, donor]) > 0.0)
        reconstruction.append(abs(float(endpoint[target, donor]) - float(target_positive[target])))
    result.update(
        {
            "same_relation_advantage_median": median(advantages),
            "same_relation_advantage_positive_fraction": float(np.mean(np.asarray(advantages) > 0.0)),
            "expected_top1_within_relation_fraction": float(np.mean(top1)),
            "expected_endpoint_flip_fraction": float(np.mean(flips)),
            "expected_reconstruction_max_abs_error": float(np.max(reconstruction)),
        }
    )
    return result


def close(left: Any, right: Any, tolerance: float = 2e-6) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def audit_split(split: str, prereg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = OUT_ROOT / "runs" / split
    metadata = read_json(run_dir / "metadata.json")
    stored = read_jsonl(run_dir / "metrics.jsonl")
    stored_lookup = {
        (
            row["config_id"],
            int(row["context"]),
            row["scenario"],
            row["method"],
            int(row["layer"]),
        ): row
        for row in stored
    }
    raw = np.load(run_dir / "raw_matrices.npz")
    effects = raw["effects"]
    endpoints = raw["endpoints"]
    positive = raw["positive_endpoints"]
    dims = metadata["dimensions"]
    checks: list[dict[str, Any]] = []
    mismatches = []
    for config_index, config in enumerate(dims["configs"]):
        for context_index, context in enumerate(dims["contexts"]):
            for scenario_index, scenario in enumerate(dims["scenarios"]):
                expected = expected_mapping(metadata, config["config_id"], int(context), scenario)
                for method_index, method in enumerate(dims["methods"]):
                    for layer_index, layer in enumerate(dims["layers"]):
                        key = (config["config_id"], int(context), scenario, method, int(layer))
                        recomputed = recompute_metrics(
                            effects[config_index, context_index, scenario_index, method_index, layer_index],
                            endpoints[config_index, context_index, scenario_index, method_index, layer_index],
                            positive[config_index, context_index, scenario_index],
                            expected,
                        )
                        original = stored_lookup[key]
                        metric_names = (
                            "finite_fraction",
                            "within_relation_spread_max",
                            "same_relation_advantage_median",
                            "same_relation_advantage_positive_fraction",
                            "expected_top1_within_relation_fraction",
                            "identity_top1_within_relation_fraction",
                            "expected_endpoint_flip_fraction",
                            "expected_reconstruction_max_abs_error",
                        )
                        for metric in metric_names:
                            if not close(recomputed[metric], original[metric]):
                                mismatches.append({"key": key, "metric": metric, "stored": original[metric], "recomputed": recomputed[metric]})
    summary = read_json(run_dir / "summary.json")
    expected_rows = (
        len(dims["configs"])
        * len(dims["contexts"])
        * len(dims["scenarios"])
        * len(dims["methods"])
        * len(dims["layers"])
    )
    named_checks = {
        "protocol_digest_matches": metadata["protocol_digest"] == prereg["protocol_digest"],
        "raw_hash_matches": sha256_file(run_dir / "raw_matrices.npz") == summary["raw_sha256"],
        "metrics_hash_matches": sha256_file(run_dir / "metrics.jsonl") == summary["metrics_sha256"],
        "metadata_hash_matches": sha256_file(run_dir / "metadata.json") == summary["metadata_sha256"],
        "row_count_matches": len(stored) == expected_rows == summary["metric_row_count"],
        "raw_shape_matches": list(effects.shape) == metadata["raw_shape"],
        "all_metrics_recomputed": len(mismatches) == 0,
        "all_raw_finite": bool(np.isfinite(effects).all() and np.isfinite(endpoints).all()),
        "six_cells": summary["cell_count"] == 6,
        "all_cells_pass": bool(summary["all_cells_pass"]),
    }
    for name, passed in named_checks.items():
        checks.append({"split": split, "name": name, "passed": bool(passed)})
    return checks, {"split": split, "mismatches": mismatches[:20], "named_checks": named_checks}


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(prereg)
    stored_digest = body.pop("protocol_digest")
    checks = [
        {"name": "protocol_digest_recomputed", "passed": digest(body) == stored_digest},
        {"name": "primary_script_hash_matches", "passed": sha256_file(PRIMARY) == prereg["script_sha256"]},
        {"name": "protocol_audit_passed", "passed": bool(read_json(OUT_ROOT / "protocol/audit.json")["all_checks_passed"])},
    ]
    details = []
    for split in SPLITS:
        split_checks, split_details = audit_split(split, prereg)
        checks.extend(split_checks)
        details.append(split_details)
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks.extend(
        [
            {"name": "selection_qualified", "passed": bool(selection["candidate_qualified"])},
            {"name": "confirmation_authorized", "passed": bool(selection["confirmation_authorized"])},
            {"name": "calibration_confirmed", "passed": bool(final["calibration_passed"])},
            {"name": "natural_protocol_only", "passed": bool(final["natural_discovery_protocol_authorized"] and not final["natural_hidden_scan_authorized"])},
            {"name": "component_search_still_denied", "passed": not bool(final["component_search_authorized"])},
        ]
    )
    all_passed = all(bool(row["passed"]) for row in checks)
    audit = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "passed_count": sum(bool(row["passed"]) for row in checks),
        "all_checks_passed": all_passed,
        "checks": checks,
        "details": details,
        "protocol_digest": stored_digest,
        "selection_digest": selection["selection_digest"],
        "final_digest": final["final_digest"],
    }
    body = dict(audit)
    audit["audit_digest"] = digest(body)
    write_json(OUT_ROOT / "audit/independent_result_audit.json", audit)
    print(canonical(audit))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
