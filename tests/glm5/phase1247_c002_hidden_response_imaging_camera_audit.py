#!/usr/bin/env python3
"""Independent audit for Phase1247 C002 hidden-response imaging camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1247_c002_hidden_response_imaging_camera as main


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def verify_protocol_digest(protocol: dict[str, Any]) -> bool:
    value = dict(protocol)
    stored = value.pop("protocol_digest", None)
    return stored == digest(value)


def preaudit() -> None:
    protocol = load_json(main.PROTOCOL_PATH)
    rows = main.read_jsonl(main.MATERIAL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest", verify_protocol_digest(protocol))
    add(
        checks,
        "source_hashes",
        protocol["source_hashes"]
        == {"main": main.file_sha256(main.SCRIPT), "audit": main.file_sha256(main.AUDIT_SCRIPT)},
    )
    add(checks, "row_count", len(rows) == 112, len(rows))
    add(checks, "unique_examples", len({row["example_id"] for row in rows}) == len(rows))
    row_digest_ok = True
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        row_digest_ok &= stored == digest(value)
    add(checks, "row_digests", row_digest_ok)
    counts = Counter(row["partition"] for row in rows)
    add(checks, "partition_counts", dict(counts) == main.PARTITION_COUNTS, dict(counts))
    triples = {(row["value0"], row["value1"], row["query"]) for row in rows}
    add(checks, "unique_triples", len(triples) == 112)
    balance = {
        partition: Counter(row["query"] for row in rows if row["partition"] == partition)
        for partition in main.PARTITION_COUNTS
    }
    add(
        checks,
        "query_balance",
        all(values[0] == values[1] == main.PARTITION_COUNTS[partition] // 2 for partition, values in balance.items()),
        {key: dict(value) for key, value in balance.items()},
    )
    target_semantics = all(
        row["swapped_target"] != row["target"]
        and row["target_donor_ids"][row["target_position"]] - 4 == row["swapped_target"]
        for row in rows
    )
    add(checks, "target_donor_changes_answer", target_semantics)
    null_semantics = all(
        row["null_donor_ids"][row["target_position"]] == row["receiver_ids"][row["target_position"]]
        and row["null_donor_ids"][1 - row["target_position"]] != row["receiver_ids"][1 - row["target_position"]]
        for row in rows
    )
    add(checks, "matched_null_preserves_answer", null_semantics)
    add(checks, "final_third_excluded", "ceil(2L/3)" in protocol["eligible_events"])
    add(checks, "confirmation_not_selection", protocol["camera"]["selection_intervention"]["partition"] == "selection")
    add(checks, "prediction_primary", protocol["camera"]["readout"] == "centered eight-candidate logit response")
    add(checks, "nulls_registered", len(protocol["camera"]["nulls"]) == 3)
    add(checks, "typed_abstention", set(protocol["typed_abstention"]) == {"in_domain", "out_of_domain", "nonidentifiable"})
    add(checks, "one_shot_budget", protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0)
    add(checks, "no_pretrained_authorization", any("No Qwen3" in value for value in protocol["hard_stops"]))
    add(checks, "no_formal_result_before_run", not main.ARRAY_PATH.exists() and not main.RAW_SUMMARY_PATH.exists())
    payload = {
        "phase": main.PHASE,
        "schema_version": "phase1247.c002.imaging_camera.preaudit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
        "claim_boundary": "Preaudit verifies frozen material and protocol only; it contains no formal response result.",
    }
    payload["audit_digest"] = digest(payload)
    write_json(main.PREAUDIT_PATH, payload)
    print(canonical_json({"status": "phase1247_preaudit", "passed": payload["all_checks_passed"], "checks": len(checks)}))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def final_audit() -> None:
    protocol = load_json(main.PROTOCOL_PATH)
    run = load_json(main.RAW_SUMMARY_PATH)
    analysis = load_json(main.ANALYSIS_PATH)
    final = load_json(main.FINAL_PATH)
    preaudit_payload = load_json(main.PREAUDIT_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "preaudit_passed", preaudit_payload["all_checks_passed"])
    add(checks, "protocol_digest", verify_protocol_digest(protocol))
    run_value = dict(run)
    run_digest = run_value.pop("run_digest")
    add(checks, "run_digest", digest(run_value) == run_digest)
    add(checks, "array_hash", main.file_sha256(main.ARRAY_PATH) == run["array_sha256"])
    add(checks, "model_count", len(run["models"]) == len(main.ARCHITECTURES) * main.REPLICATES)
    add(checks, "no_pretrained_loaded", run["qwen_or_other_pretrained_loaded"] is False)
    add(checks, "gpu_budget", run["gpu_hours"] <= protocol["budgets"]["max_gpu_hours"], run["gpu_hours"])
    recomputed_model_gates: dict[str, bool] = {}
    selected_ok = True
    array_prediction_ok = True
    metric_ok = True
    sentinel_rows = []
    with np.load(main.ARRAY_PATH) as arrays:
        for row in run["models"]:
            selected = row["selected_event"]
            scores = row["selection_scores"]
            selected_ok &= selected == max(scores, key=scores.get)
            weights = arrays[main.array_key(row["model_key"], selected, "weights")]
            actual = arrays[main.array_key(row["model_key"], selected, "confirmation_target_actual")]
            features = arrays[main.array_key(row["model_key"], selected, "confirmation_target_x")]
            stored_prediction = arrays[main.array_key(row["model_key"], selected, "confirmation_target_predicted")]
            recomputed_prediction = main.ridge_predict(features, weights)
            array_prediction_ok &= np.allclose(stored_prediction, recomputed_prediction, rtol=1.0e-10, atol=1.0e-10)
            recomputed = main.response_metrics(recomputed_prediction, actual)
            stored = row["selected_event_summary"]["splits"]["confirmation"]["donors"]["target"]["camera"]
            metric_ok &= all(close(recomputed[key], stored[key]) for key in recomputed)
            confirmation = row["selected_event_summary"]["splits"]["confirmation"]
            gate = bool(
                row["behavior"]["accuracy"] >= main.THRESHOLDS["behavior_accuracy_min"]
                and recomputed["cosine_mean"] >= main.THRESHOLDS["confirmation_cosine_mean_min"]
                and recomputed["cosine_positive_fraction"] >= main.THRESHOLDS["confirmation_positive_fraction_min"]
                and recomputed["relative_error_mean"] <= main.THRESHOLDS["confirmation_relative_error_mean_max"]
                and confirmation["prediction_advantage"] >= main.THRESHOLDS["prediction_advantage_min"]
                and confirmation["target_to_null_effect_ratio"] >= main.THRESHOLDS["target_to_null_effect_ratio_min"]
            )
            recomputed_model_gates[row["model_key"]] = gate
            sentinel_rows.append(
                main.response_metrics(recomputed_prediction, actual)["cosine_mean"]
                - main.response_metrics(-recomputed_prediction, actual)["cosine_mean"]
                >= 0.50
            )
    add(checks, "selection_recomputed", selected_ok)
    add(checks, "array_predictions_recomputed", array_prediction_ok)
    add(checks, "selected_metrics_recomputed", metric_ok)
    add(
        checks,
        "model_gates_recomputed",
        recomputed_model_gates == {row["model_key"]: row["model_gate"] for row in run["models"]},
        recomputed_model_gates,
    )
    passing = [row for row in run["models"] if recomputed_model_gates[row["model_key"]]]
    per_architecture = {
        architecture: sum(
            recomputed_model_gates[row["model_key"]]
            for row in run["models"]
            if row["architecture"] == architecture
        )
        for architecture in main.ARCHITECTURES
    }
    recomputed_gates = {
        "G-BEHAVIOR": all(row["behavior"]["accuracy"] >= main.THRESHOLDS["behavior_accuracy_min"] for row in run["models"]),
        "G-IMAGING": len(passing) >= main.THRESHOLDS["passing_models_min"]
        and all(value >= main.THRESHOLDS["passing_per_architecture_min"] for value in per_architecture.values()),
        "G-SPECIFICITY": bool(passing)
        and all(
            row["selected_event_summary"]["splits"]["confirmation"]["prediction_advantage"]
            >= main.THRESHOLDS["prediction_advantage_min"]
            and row["selected_event_summary"]["splits"]["confirmation"]["target_to_null_effect_ratio"]
            >= main.THRESHOLDS["target_to_null_effect_ratio_min"]
            for row in passing
        ),
        "G-IDENTIFIABILITY": float(np.mean(sentinel_rows))
        >= main.THRESHOLDS["sentinel_corruption_detection_min"],
    }
    add(checks, "typed_gates_recomputed", recomputed_gates == analysis["gates"], recomputed_gates)
    expected_verdict = "known_truth_imaging_camera_confirmed" if all(recomputed_gates.values()) else "known_truth_imaging_camera_not_confirmed"
    add(checks, "verdict_recomputed", expected_verdict == analysis["verdict"] == final["verdict"])
    analysis_value = dict(analysis)
    analysis_digest = analysis_value.pop("adjudication_digest")
    add(checks, "analysis_digest", digest(analysis_value) == analysis_digest)
    final_value = dict(final)
    final_digest = final_value.pop("final_digest")
    add(checks, "final_digest", digest(final_value) == final_digest)
    add(checks, "authorization_typed", final["hidden_language_mechanism_claim_authorized"] is False)
    payload = {
        "phase": main.PHASE,
        "schema_version": "phase1247.c002.imaging_camera.final_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
        "recomputed_gates": recomputed_gates,
        "recomputed_verdict": expected_verdict,
        "claim_boundary": "Audit verifies the TinyTransformer imaging calibration only; no pretrained language mechanism follows.",
    }
    payload["audit_digest"] = digest(payload)
    write_json(main.FINAL_AUDIT_PATH, payload)
    print(canonical_json({"status": "phase1247_final_audit", "passed": payload["all_checks_passed"], "checks": len(checks), "verdict": expected_verdict}))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.mode == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    main_cli()
