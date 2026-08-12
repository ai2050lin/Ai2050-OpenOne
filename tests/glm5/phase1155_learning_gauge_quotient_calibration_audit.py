#!/usr/bin/env python3
"""Independent artifact audit for Phase1155."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1155_learning_gauge_quotient_calibration"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1155_learning_gauge_quotient_calibration.py"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1154_learned_morphology_external_validity"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.where(norm > 1e-12, norm, 1.0)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = normalize_vector(left)
    b = normalize_vector(right)
    if not np.any(a) or not np.any(b):
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b))


def predict(features: np.ndarray, labels: list[str], prototypes: np.ndarray) -> list[str]:
    scores = normalize_rows(features) @ np.asarray(prototypes, dtype=np.float64).T
    return [labels[int(index)] for index in np.argmax(scores, axis=1)]


def metrics(predicted: list[str], truth: list[dict[str, Any]], indices: list[int], label_key: str, gauges: list[str]) -> dict[str, Any]:
    correct = [predicted[offset] == str(truth[index][label_key]) for offset, index in enumerate(indices)]
    labels = sorted({str(truth[index][label_key]) for index in indices})
    per_label = {}
    for label in labels:
        selected = [offset for offset, index in enumerate(indices) if str(truth[index][label_key]) == label]
        per_label[label] = float(np.mean([correct[offset] for offset in selected]))
    per_gauge = {}
    for gauge in gauges:
        selected = [offset for offset, index in enumerate(indices) if truth[index]["gauge"] == gauge]
        if selected:
            per_gauge[gauge] = float(np.mean([correct[offset] for offset in selected]))
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "per_gauge_accuracy": per_gauge,
        "gauge_accuracy_gap": float(max(per_gauge.values()) - min(per_gauge.values())) if per_gauge else 0.0,
        "count": len(indices),
    }


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    required = [
        OUT_ROOT / "protocol/preregistration.json",
        OUT_ROOT / "protocol/audit.json",
        OUT_ROOT / "runs/discovery/summary.json",
        OUT_ROOT / "runs/confirmation/summary.json",
        OUT_ROOT / "analysis/fit.json",
        OUT_ROOT / "analysis/score.json",
        OUT_ROOT / "analysis/final.json",
        OUT_ROOT / "predictions/manifest.json",
        OUT_ROOT / "predictions/confirmation_predictions.jsonl",
    ]
    for path in required:
        checks[f"exists::{path.relative_to(OUT_ROOT)}"] = path.exists()
    if not all(checks.values()):
        raise RuntimeError("Phase1155 artifacts are incomplete")

    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_body = dict(protocol)
    protocol_digest = protocol_body.pop("protocol_digest")
    checks["protocol_digest"] = digest(protocol_body) == protocol_digest
    checks["main_script_hash"] = sha256_file(MAIN_SCRIPT) == protocol["script_sha256"]
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    checks["protocol_audit_passed"] = bool(protocol_audit["all_checks_passed"])
    checks["protocol_audit_digest_link"] = protocol_audit["protocol_digest"] == protocol_digest
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks["source_final_link"] = source_final["final_digest"] == protocol["source_phase1154_digest"]
    checks["source_audit_link"] = source_audit["audit_digest"] == protocol["source_phase1154_audit_digest"]
    checks["source_failure_preserved"] = not bool(source_final["learned_morphology_external_validity_confirmed"])
    checks["candidate_frozen"] = protocol["candidate"] == "dependency_rank_quotient"
    checks["coarse_scope_three"] = len(set(protocol["coarse_labels"].values())) == 3

    gauges = list(protocol["gauges"])
    algorithms = list(protocol["algorithms"])
    split_data: dict[str, dict[str, Any]] = {}
    for split in ("discovery", "confirmation"):
        root = OUT_ROOT / "runs" / split
        paths = {
            "feature_pack": root / "feature_pack.npz",
            "public_manifest": root / "public_manifest.jsonl",
            "sealed_truth": root / "sealed_truth.jsonl",
            "diagnostics": root / "diagnostics.jsonl",
            "training_metrics": root / "training_metrics.jsonl",
            "summary": root / "summary.json",
        }
        for name, path in paths.items():
            checks[f"{split}::exists::{name}"] = path.exists()
        truth = read_jsonl(paths["sealed_truth"])
        public = read_jsonl(paths["public_manifest"])
        diagnostics = read_jsonl(paths["diagnostics"])
        training = read_jsonl(paths["training_metrics"])
        summary = read_json(paths["summary"])
        with np.load(paths["feature_pack"]) as pack:
            arrays = {name: np.asarray(pack[name]) for name in pack.files}
        split_data[split] = {"truth": truth, "public": public, "diagnostics": diagnostics, "training": training, "summary": summary, "arrays": arrays}
        checks[f"{split}::unit_count"] = len(truth) == len(public) == len(diagnostics) == 96
        checks[f"{split}::model_count"] = len(training) == 24
        checks[f"{split}::indices"] = all(row["index"] == index for index, row in enumerate(truth))
        checks[f"{split}::public_truth_ids"] = [row["unit_id"] for row in public] == [row["unit_id"] for row in truth]
        checks[f"{split}::diagnostic_ids"] = [row["unit_id"] for row in diagnostics] == [row["unit_id"] for row in truth]
        for gauge in gauges:
            checks[f"{split}::gauge_count::{gauge}"] = sum(row["gauge"] == gauge for row in truth) == 24
        for group in protocol["groups"]:
            checks[f"{split}::group_count::{group}"] = sum(row["functional_group"] == group for row in truth) == 16
        for algorithm in algorithms:
            checks[f"{split}::feature::{algorithm}"] = algorithm in arrays and arrays[algorithm].shape[0] == 96
            checks[f"{split}::finite::{algorithm}"] = bool(np.isfinite(arrays[algorithm]).all())
        checks[f"{split}::feature_hash"] = sha256_file(paths["feature_pack"]) == summary["feature_pack_sha256"]
        checks[f"{split}::public_hash"] = sha256_file(paths["public_manifest"]) == summary["public_manifest_sha256"]
        checks[f"{split}::truth_hash"] = sha256_file(paths["sealed_truth"]) == summary["sealed_truth_sha256"]
        checks[f"{split}::diagnostic_hash"] = sha256_file(paths["diagnostics"]) == summary["diagnostics_sha256"]
        checks[f"{split}::training_hash"] = sha256_file(paths["training_metrics"]) == summary["training_metrics_sha256"]
        checks[f"{split}::protocol_link"] = summary["protocol_digest"] == protocol_digest
        checks[f"{split}::summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"}) == summary["summary_digest"]
        checks[f"{split}::run_gate"] = bool(summary["run_gate_passed"] and all(summary["checks"].values()))
        checks[f"{split}::accuracy_recompute"] = close(min(row["accuracy"] for row in training), summary["accuracy_min"])
        checks[f"{split}::probability_recompute"] = close(min(row["min_probability"] for row in training), summary["min_probability_min"])
        checks[f"{split}::output_error_recompute"] = close(max(row["output_abs_error_max"] for row in diagnostics), summary["clean_output_abs_error_max"])
        for row in training:
            model_path = root / "models" / f"{row['model_id']}.pt"
            checks[f"{split}::model_hash::{row['model_id']}"] = model_path.exists() and sha256_file(model_path) == row["model_sha256"]

        by_key = {(row["functional_group"], int(row["replicate"]), row["gauge"]): int(row["index"]) for row in truth}
        candidate_matches = []
        site_functional = []
        site_gram = []
        cross_functional = []
        for group in protocol["groups"]:
            for replicate in range(protocol["replicates"]):
                base = by_key[(group, replicate, "identity")]
                for gauge in gauges[1:]:
                    current = by_key[(group, replicate, gauge)]
                    candidate_matches.append(bool(np.array_equal(arrays[protocol["candidate"]][base], arrays[protocol["candidate"]][current])))
                site = by_key[(group, replicate, "site_gl")]
                site_functional.append(cosine(arrays["functional_tomography"][base], arrays["functional_tomography"][site]))
                site_gram.append(cosine(arrays["state_gram"][base], arrays["state_gram"][site]))
                for gauge in ("cross_site_orthogonal", "cross_site_gl"):
                    current = by_key[(group, replicate, gauge)]
                    cross_functional.append(cosine(arrays["functional_tomography"][base], arrays["functional_tomography"][current]))
        break_threshold = protocol["thresholds"]["cross_site_physical_break_cosine_max"]
        break_count = int(sum(value < break_threshold for value in cross_functional))
        checks[f"{split}::candidate_match_recompute"] = close(np.mean(candidate_matches), summary["candidate_gauge_match_fraction"])
        checks[f"{split}::site_functional_min_recompute"] = close(min(site_functional), summary["site_gl_functional_cosine_min"])
        checks[f"{split}::site_gram_median_recompute"] = close(np.median(site_gram), summary["site_gl_state_gram_cosine_median"])
        checks[f"{split}::cross_min_recompute"] = close(min(cross_functional), summary["cross_site_functional_cosine_min"])
        checks[f"{split}::cross_break_recompute"] = break_count == summary["cross_site_physical_break_count"]

    discovery_ids = {row["unit_id"] for row in split_data["discovery"]["public"]}
    confirmation_ids = {row["unit_id"] for row in split_data["confirmation"]["public"]}
    checks["split_ids_disjoint"] = not bool(discovery_ids & confirmation_ids)

    fit = read_json(OUT_ROOT / "analysis/fit.json")
    labels = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    prototype_path = OUT_ROOT / "analysis/frozen_prototypes.npz"
    with np.load(prototype_path) as pack:
        prototypes = {name: np.asarray(pack[name]) for name in pack.files}
    checks["fit_digest"] = digest({key: value for key, value in fit.items() if key != "fit_digest"}) == fit["fit_digest"]
    checks["fit_prototype_hash"] = sha256_file(prototype_path) == fit["prototype_sha256"]
    checks["fit_labels_hash"] = sha256_file(OUT_ROOT / "analysis/prototype_labels.json") == fit["labels_sha256"]
    checks["fit_authorized"] = bool(fit["candidate_qualified"] and fit["confirmation_run_authorized"])
    dtruth = split_data["discovery"]["truth"]
    darrays = split_data["discovery"]["arrays"]
    fit_indices = [index for index, row in enumerate(dtruth) if int(row["replicate"]) in protocol["fit_replicates"] and row["gauge"] == "identity"]
    validation_indices = [index for index, row in enumerate(dtruth) if int(row["replicate"]) in protocol["validation_replicates"]]
    checks["fit_count"] = fit["fit_count"] == len(fit_indices) == 18
    checks["validation_count"] = fit["validation_count"] == len(validation_indices) == 24
    for algorithm in algorithms:
        coarse_prediction = predict(darrays[algorithm][validation_indices], labels[algorithm]["coarse_labels"], prototypes[f"{algorithm}__coarse"])
        exact_prediction = predict(darrays[algorithm][validation_indices], labels[algorithm]["exact_labels"], prototypes[f"{algorithm}__exact"])
        coarse_metrics = metrics(coarse_prediction, dtruth, validation_indices, "coarse_group", gauges)
        exact_metrics = metrics(exact_prediction, dtruth, validation_indices, "functional_group", gauges)
        checks[f"fit_metric::{algorithm}::coarse"] = canonical(coarse_metrics) == canonical(fit["algorithm_metrics"][algorithm]["coarse"])
        checks[f"fit_metric::{algorithm}::exact"] = canonical(exact_metrics) == canonical(fit["algorithm_metrics"][algorithm]["exact"])

    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    prediction_path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    checks["prediction_truth_blind"] = manifest["confirmation_truth_read"] is False
    checks["prediction_hash"] = sha256_file(prediction_path) == manifest["prediction_sha256"]
    checks["prediction_count"] = len(predictions) == manifest["prediction_count"] == 96
    checks["prediction_unit_alignment"] = [row["unit_id"] for row in predictions] == [row["unit_id"] for row in split_data["confirmation"]["public"]]
    carrays = split_data["confirmation"]["arrays"]
    for algorithm in algorithms:
        expected_coarse = predict(carrays[algorithm], labels[algorithm]["coarse_labels"], prototypes[f"{algorithm}__coarse"])
        expected_exact = predict(carrays[algorithm], labels[algorithm]["exact_labels"], prototypes[f"{algorithm}__exact"])
        checks[f"prediction_recompute::{algorithm}::coarse"] = expected_coarse == [row["algorithms"][algorithm]["coarse"] for row in predictions]
        checks[f"prediction_recompute::{algorithm}::exact"] = expected_exact == [row["algorithms"][algorithm]["exact"] for row in predictions]

    score = read_json(OUT_ROOT / "analysis/score.json")
    ctruth = split_data["confirmation"]["truth"]
    indices = list(range(len(ctruth)))
    checks["score_digest"] = digest({key: value for key, value in score.items() if key != "score_digest"}) == score["score_digest"]
    checks["score_confirmed"] = bool(score["candidate_confirmed"] and all(score["candidate_checks"].values()))
    for algorithm in algorithms:
        coarse_metrics = metrics([row["algorithms"][algorithm]["coarse"] for row in predictions], ctruth, indices, "coarse_group", gauges)
        exact_metrics = metrics([row["algorithms"][algorithm]["exact"] for row in predictions], ctruth, indices, "functional_group", gauges)
        checks[f"score_metric::{algorithm}::coarse"] = canonical(coarse_metrics) == canonical(score["algorithm_metrics"][algorithm]["coarse"])
        checks[f"score_metric::{algorithm}::exact"] = canonical(exact_metrics) == canonical(score["algorithm_metrics"][algorithm]["exact"])

    final = read_json(OUT_ROOT / "analysis/final.json")
    checks["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"}) == final["final_digest"]
    checks["final_protocol_link"] = final["protocol_digest"] == protocol_digest
    checks["final_split_overlap"] = final["split_overlap"] == 0
    checks["final_coarse_confirmed"] = bool(final["coarse_dependency_quotient_confirmed"])
    checks["final_scope_guard"] = not bool(final["full_six_way_morphology_identification_claim"])
    checks["final_no_free_scan"] = not bool(final["free_transformer_scan_authorized"] or final["pretrained_model_scan_authorized"])
    checks["final_no_auto_continue"] = final["auto_continue"] is False

    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "phase": 1155,
        "audit_kind": "independent_artifact_recomputation",
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "failed_checks": failed,
        "all_checks_passed": not failed,
        "details": {
            "protocol_digest": protocol_digest,
            "fit_digest": fit["fit_digest"],
            "score_digest": score["score_digest"],
            "final_digest": final["final_digest"],
        },
    }
    result["audit_digest"] = digest(result)
    output = OUT_ROOT / "audit/independent_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical(result))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
