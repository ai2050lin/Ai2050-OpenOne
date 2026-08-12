#!/usr/bin/env python3
"""Independent preaudit and result audit for Phase 1219."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1219_target_typed_prerule_prediction as experiment  # noqa: E402
import phase1217_factorial_free_formation_clock_transfer_audit as clock_audit  # noqa: E402


OUT_ROOT = experiment.OUT_ROOT
PROTOCOL_PATH = experiment.PROTOCOL_PATH
PREAUDIT_PATH = experiment.PREAUDIT_PATH
DISCOVERY_MODEL_PATH = experiment.DISCOVERY_MODEL_PATH
FINAL_PATH = experiment.FINAL_PATH
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def valid_embedded_digest(value: dict[str, Any], field: str) -> bool:
    clean = dict(value)
    stored = clean.pop(field, None)
    return isinstance(stored, str) and digest(clean) == stored


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def preaudit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest_valid", valid_embedded_digest(protocol, "protocol_digest"))
    add(checks, "phase_exact", protocol["phase"] == 1219)
    add(checks, "main_hash_current", protocol["script_hashes"]["phase1219_main"] == sha256_file(experiment.SCRIPT))
    add(checks, "audit_hash_current", protocol["script_hashes"]["phase1219_audit"] == sha256_file(Path(__file__)))
    add(checks, "source_hash_current", protocol["script_hashes"]["phase1218_source"] == sha256_file(Path(experiment.source.__file__)))
    add(checks, "source_gate_all_true", all(protocol["source_gate"].values()), protocol["source_gate"])
    add(checks, "run_count_exact", protocol["formal_run_count"] == 64 and protocol["runs_per_split"] == 32)
    add(checks, "replicates_exact", protocol["replicates"] == 4)
    add(
        checks,
        "full_factor_dimensions",
        all(len(protocol[key][split]) == 2 for key in ("tasks", "lexicons", "architectures") for split in ("discovery", "confirmation")),
    )
    task_maps = {
        split: [tuple(task["source_roles"][query] for query in experiment.core.FUNCTION_QUERIES) for task in tasks]
        for split, tasks in protocol["tasks"].items()
    }
    add(checks, "task_maps_split_disjoint", set(task_maps["discovery"]).isdisjoint(task_maps["confirmation"]), task_maps)
    add(checks, "task_maps_unique", len(set(task_maps["discovery"] + task_maps["confirmation"])) == 4)
    lexicon_seeds = [item["seed"] for rows in protocol["lexicons"].values() for item in rows]
    add(checks, "lexicon_seeds_unique", len(lexicon_seeds) == len(set(lexicon_seeds)) == 4)
    add(checks, "lexicons_new", all(seed // 1000 == 1219 for seed in lexicon_seeds))
    widths = [config["width"] for rows in protocol["architectures"].values() for config in rows.values()]
    depths = [config["layers"] for rows in protocol["architectures"].values() for config in rows.values()]
    add(checks, "architecture_width_fixed", set(widths) == {112}, widths)
    add(checks, "architecture_depths_unique", len(set(depths)) == 4, depths)
    observation = protocol["observation_contract"]
    add(checks, "prefix_grid_exact", tuple(observation["prefix_grid"]) == experiment.PREFIX_STEPS)
    add(checks, "anchor_grid_exact", tuple(observation["anchor_grid"]) == experiment.ANCHOR_STEPS)
    add(checks, "observation_grid_exact", tuple(observation["observation_grid"]) == experiment.OBSERVATION_STEPS)
    add(checks, "prefix_count_eleven", len(observation["prefix_grid"]) == 11)
    add(checks, "observation_count_35", len(observation["observation_grid"]) == 35)
    add(checks, "clock_anchor_only", observation["clock_outcomes_use_anchor_grid_only"] is True)
    add(checks, "prediction_unit_system", observation["prediction_unit"] == "system, never checkpoint")
    targets = protocol["target_contract"]
    add(checks, "classification_keeps_censoring", "right-censored" in targets["classification"]["negative"])
    add(checks, "onset_excludes_censoring", "excluded" in targets["onset"]["right_censored_systems"])
    add(checks, "target_gates_distinct", set(targets["gates"]["classification"]) != set(targets["gates"]["onset"]))
    add(checks, "classification_balance_8_8", targets["gates"]["classification"]["positive_min"] == 8 and targets["gates"]["classification"]["negative_min"] == 8)
    baseline = protocol["baseline_contract"]
    add(checks, "factor_schema_exact", tuple(baseline["factor_features"]) == experiment.BASELINE_FACTOR_NAMES)
    add(checks, "eight_scalar_families", tuple(baseline["scalar_families"]) == experiment.BASELINE_FAMILIES)
    add(checks, "scalar_schema_exact", tuple(baseline["scalar_features"]) == experiment.BASELINE_SCALAR_NAMES)
    mechanism = protocol["mechanism_contract"]
    add(checks, "six_mechanism_families", len(mechanism["families"]) == 6)
    add(checks, "mechanism_schema_exact", tuple(mechanism["features"]) == experiment.MECHANISM_FEATURE_NAMES)
    add(checks, "matched_nulls_exact", tuple(mechanism["matched_null_shifts_within_exact_factor_cell"]) == experiment.NULL_SHIFTS)
    predictor = protocol["predictor_contract"]
    add(checks, "ridge_grid_exact", tuple(predictor["ridge_grid"]) == experiment.RIDGE_GRID)
    add(checks, "confirmation_blinding_frozen", predictor["discovery_model_frozen_before_confirmation_training"] is True)
    forbidden = protocol["forbidden"]
    add(checks, "checkpoint_independence_forbidden", "using checkpoints as independent samples" in forbidden)
    add(checks, "censored_drop_forbidden", "dropping right-censored systems from classification" in forbidden)
    add(checks, "mixed_gate_forbidden", "requiring observed onset breadth to authorize classification" in forbidden)
    add(checks, "confirmation_peeking_forbidden", "reading confirmation outcomes before the discovery model is frozen" in forbidden)
    all_passed = all(row["passed"] for row in checks)
    result = {
        "phase": 1219,
        "mode": "preaudit",
        "protocol_digest": protocol["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all_passed,
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    write_json(PREAUDIT_PATH, result)
    if not all_passed:
        raise RuntimeError(f"preaudit failed: {[row for row in checks if not row['passed']]}")
    return result


def independent_onsets(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = [row for row in trajectory if int(row["step"]) in set(experiment.ANCHOR_STEPS)]
    return {
        clock: clock_audit.independent_onset(anchors, "primary", clock) for clock in experiment.CLOCKS
    }


def independent_predict(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    design = np.concatenate((np.ones((len(values), 1)), (values - mean) / scale), axis=1)
    return design @ coefficient


def independent_classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    probability = np.clip(predictions, 0.0, 1.0)
    labels = probability >= 0.5
    truth = targets >= 0.5
    positive, negative = truth, ~truth
    tpr = float(np.mean(labels[positive] == truth[positive])) if np.any(positive) else 0.0
    tnr = float(np.mean(labels[negative] == truth[negative])) if np.any(negative) else 0.0
    pairwise = [float(predictions[i] > predictions[j]) + 0.5 * float(predictions[i] == predictions[j]) for i in np.where(positive)[0] for j in np.where(negative)[0]]
    return {
        "accuracy": float(np.mean(labels == truth)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "positive_recall": tpr,
        "negative_recall": tnr,
        "brier": float(np.mean((probability - targets) ** 2)),
        "auc_pairwise": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def independent_onset_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = np.abs(predictions - targets)
    return {
        "mae_steps": float(np.mean(errors)),
        "median_absolute_error_steps": float(np.median(errors)),
        "within_100_fraction": float(np.mean(errors <= 100.0)),
        "within_200_fraction": float(np.mean(errors <= 200.0)),
    }


def close_dict(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1.0e-10) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if isinstance(left[key], dict) and isinstance(right[key], dict):
            if not close_dict(left[key], right[key], tolerance):
                return False
        elif isinstance(left[key], (int, float)) and isinstance(right[key], (int, float)):
            if abs(float(left[key]) - float(right[key])) > tolerance:
                return False
        elif left[key] != right[key]:
            return False
    return True


def result_audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    preaudit_result = read_json(PREAUDIT_PATH)
    discovery_model = read_json(DISCOVERY_MODEL_PATH)
    final = read_json(FINAL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest_valid", valid_embedded_digest(protocol, "protocol_digest"))
    add(checks, "preaudit_digest_valid", valid_embedded_digest(preaudit_result, "audit_digest"))
    add(checks, "preaudit_passed", preaudit_result["all_checks_passed"] is True)
    add(checks, "discovery_model_digest_valid", valid_embedded_digest(discovery_model, "model_digest"))
    add(checks, "final_digest_valid", valid_embedded_digest(final, "final_digest"))
    add(checks, "protocol_links", discovery_model["protocol_digest"] == final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "manifest_count", len(final["run_manifest"]) == 64)
    add(checks, "manifest_unique", len({row["run_id"] for row in final["run_manifest"]}) == 64)
    add(checks, "discovery_model_link", final["discovery_model_digest"] == discovery_model["model_digest"])

    rows_by_split: dict[str, list[dict[str, Any]]] = {"discovery": [], "confirmation": []}
    for item in final["run_manifest"]:
        path = ROOT / item["path"]
        add(checks, f"run_exists_{item['run_id']}", path.exists())
        add(checks, f"run_hash_{item['run_id']}", path.exists() and sha256_file(path) == item["sha256"])
        row = read_json(path)
        add(checks, f"metrics_digest_{item['run_id']}", valid_embedded_digest(row, "metrics_digest"))
        add(checks, f"metrics_manifest_{item['run_id']}", row["metrics_digest"] == item["metrics_digest"])
        add(checks, f"phase_{item['run_id']}", row["phase"] == 1219)
        add(checks, f"protocol_{item['run_id']}", row["protocol_digest"] == protocol["protocol_digest"])
        add(checks, f"grid_{item['run_id']}", [point["step"] for point in row["trajectory"]] == list(experiment.OBSERVATION_STEPS))
        prefix = [point for point in row["trajectory"] if int(point["step"]) <= experiment.LANDMARK_STEP]
        add(checks, f"prefix_grid_{item['run_id']}", [point["step"] for point in prefix] == list(experiment.PREFIX_STEPS))
        add(checks, f"prefix_camera_count_{item['run_id']}", sum("mechanism_camera" in point for point in prefix) == 11)
        endpoint = prefix[-1]
        add(checks, f"probe_present_{item['run_id']}", "local_gradient_probe" in endpoint)
        add(checks, f"probe_directions_{item['run_id']}", set(endpoint["local_gradient_probe"]["responses"]) == {"correct", "anti", "random"})
        add(checks, f"probe_restore_{item['run_id']}", endpoint["local_gradient_probe"]["restore_drift_max"] <= experiment.TARGET_GATES["zero_drift_max"])
        add(checks, f"finite_{item['run_id']}", experiment.all_finite(row))
        add(checks, f"checkpoint_count_{item['run_id']}", len(row["checkpoint_manifest"]) == 4)
        add(checks, f"checkpoint_steps_{item['run_id']}", [value["step"] for value in row["checkpoint_manifest"]] == list(experiment.SAVED_CHECKPOINT_STEPS))
        for checkpoint in row["checkpoint_manifest"]:
            checkpoint_path = ROOT / checkpoint["path"]
            add(checks, f"checkpoint_exists_{item['run_id']}_{checkpoint['step']}", checkpoint_path.exists())
            add(checks, f"checkpoint_hash_{item['run_id']}_{checkpoint['step']}", checkpoint_path.exists() and sha256_file(checkpoint_path) == checkpoint["sha256"])
        onsets = independent_onsets(row["trajectory"])
        onset_match = all(
            onsets[clock]["status"] == row["formation"]["primary_clocks"][clock]["status"]
            and onsets[clock].get("step") == row["formation"]["primary_clocks"][clock].get("step")
            for clock in experiment.CLOCKS
        )
        add(checks, f"onsets_recompute_{item['run_id']}", onset_match)
        landmark = all(not point["gates"]["primary"]["R"] for point in prefix)
        add(checks, f"landmark_recompute_{item['run_id']}", landmark == row["formation"]["landmark_pre_rule"])
        record = experiment.feature_record(row)
        add(checks, f"feature_schema_{item['run_id']}", tuple(record["factor"]) == experiment.BASELINE_FACTOR_NAMES and tuple(record["scalar"]) == experiment.BASELINE_SCALAR_NAMES and tuple(record["mechanism"]) == experiment.MECHANISM_FEATURE_NAMES)
        add(checks, f"feature_finite_{item['run_id']}", experiment.all_finite(record))
        rows_by_split[row["split"]].append(row)

    records_by_split = {}
    for split, rows in rows_by_split.items():
        add(checks, f"split_count_{split}", len(rows) == 32)
        cells = [tuple(int(row[key]) for key in ("task_index", "lexicon_index", "architecture_index", "replicate")) for row in rows]
        expected_cells = set(itertools_product_cells())
        add(checks, f"full_factorial_{split}", set(cells) == expected_cells and len(cells) == len(set(cells)) == 32)
        records = [experiment.feature_record(row) for row in rows]
        records_by_split[split] = records
        qualification = experiment.target_qualification(records)
        add(checks, f"qualification_recompute_{split}", qualification == final["qualifications"][split])

    add(checks, "discovery_model_before_confirmation", discovery_model["created_at"] < min(row["created_at"] for row in rows_by_split["confirmation"]))
    add(checks, "final_blinding_claim", final["claims"]["confirmation_trained_after_discovery_model_freeze"] is True)
    add(checks, "discovery_record_digest", discovery_model["records_digest"] == digest(records_by_split["discovery"]))
    add(checks, "discovery_run_digest_links", discovery_model["run_metrics_digests"] == {row["run_id"]: row["metrics_digest"] for row in rows_by_split["discovery"]})

    for target, target_result in final["confirmation"].items():
        if target_result.get("tested") is False:
            add(checks, f"untested_target_gate_{target}", final["qualifications"]["confirmation"][target]["authorized"] is False)
            continue
        eligible = records_by_split["confirmation"] if target == "classification" else [row for row in records_by_split["confirmation"] if row["primary_onset"] is not None and row["primary_onset"] > experiment.LANDMARK_STEP]
        models = discovery_model["models"][target]
        for name in ("factor", "scalar", "baseline", "augmented"):
            specification = models[name]
            values = experiment.matrix(eligible, specification["kind"], specification["null_shift"])
            targets = experiment.target_vector(eligible, target)
            predictions = independent_predict(specification["model"], values)
            metrics = independent_classification_metrics(targets, predictions) if target == "classification" else independent_onset_metrics(targets, predictions)
            stored = target_result["evaluations"][name]
            add(checks, f"prediction_recompute_{target}_{name}", np.allclose(predictions, np.asarray(stored["predictions"]), atol=1.0e-12, rtol=0.0))
            add(checks, f"metrics_recompute_{target}_{name}", close_dict(metrics, stored["metrics"]))
        for index, specification in enumerate(models["matched_nulls"]):
            values = experiment.matrix(eligible, specification["kind"], specification["null_shift"], records_by_split["confirmation"])
            targets = experiment.target_vector(eligible, target)
            predictions = independent_predict(specification["model"], values)
            metrics = independent_classification_metrics(targets, predictions) if target == "classification" else independent_onset_metrics(targets, predictions)
            stored = target_result["matched_nulls"][index]
            add(checks, f"null_prediction_recompute_{target}_{index}", np.allclose(predictions, np.asarray(stored["predictions"]), atol=1.0e-12, rtol=0.0))
            add(checks, f"null_metrics_recompute_{target}_{index}", close_dict(metrics, stored["metrics"]))
        recomputed = experiment.target_confirmation(target, models, records_by_split["confirmation"])
        add(checks, f"target_confirmation_recompute_{target}", recomputed == target_result)

    class_pass = bool(final["confirmation"].get("classification", {}).get("passed", False))
    expected_status = (
        "classification_and_onset_incremental_prediction_confirmed"
        if class_pass and final["confirmation"].get("onset", {}).get("passed", False)
        else "classification_incremental_prediction_confirmed_onset_not_confirmed"
        if class_pass
        else "frozen_prerule_mechanism_increment_not_confirmed"
    )
    add(checks, "status_recompute", final["status"] == expected_status)
    add(checks, "authorization_recompute", final["authorized_next"]["automatic_execution"] == class_pass)
    add(checks, "right_censored_claim", final["claims"]["right_censored_retained_in_classification"] is True)
    add(checks, "new_math_false", final["new_mathematics_required"] is False)

    all_passed = all(row["passed"] for row in checks)
    result = {
        "phase": 1219,
        "mode": "result",
        "protocol_digest": protocol["protocol_digest"],
        "discovery_model_digest": discovery_model["model_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all_passed,
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    write_json(RESULT_AUDIT_PATH, result)
    if not all_passed:
        raise RuntimeError(f"result audit failed: {[row for row in checks if not row['passed']][:30]}")
    return result


def itertools_product_cells() -> list[tuple[int, int, int, int]]:
    return [(task, lexicon, architecture, replicate) for task in range(2) for lexicon in range(2) for architecture in range(2) for replicate in range(experiment.REPLICATES)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preaudit", "result"), required=True)
    args = parser.parse_args()
    value = preaudit() if args.mode == "preaudit" else result_audit()
    print(json.dumps({key: value[key] for key in ("mode", "check_count", "passed_count", "all_checks_passed", "audit_digest")}, indent=2))


if __name__ == "__main__":
    main()
