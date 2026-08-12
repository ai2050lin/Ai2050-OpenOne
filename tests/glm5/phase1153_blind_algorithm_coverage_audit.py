#!/usr/bin/env python3
"""Independent recomputation audit for Phase1153."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = 1153
OUT_ROOT = ROOT / "tests/glm5/result/phase1153_blind_algorithm_coverage"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1152_tie_aware_morphology_library"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1153_blind_algorithm_coverage.py"
AUDIT_SCRIPT = Path(__file__).resolve()


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


def build_prototypes(features: np.ndarray, truth: list[dict[str, Any]], indices: list[int], key: str) -> tuple[list[str], np.ndarray]:
    normalized = normalize_rows(features)
    labels = sorted({str(truth[index][key]) for index in indices})
    rows = []
    for label in labels:
        selected = [index for index in indices if str(truth[index][key]) == label]
        rows.append(normalize_vector(np.mean(normalized[selected], axis=0)))
    return labels, np.stack(rows)


def predict(features: np.ndarray, labels: list[str], prototypes: np.ndarray) -> tuple[list[str], np.ndarray]:
    scores = normalize_rows(features) @ np.asarray(prototypes, dtype=np.float64).T
    selected = np.argmax(scores, axis=1)
    return [labels[int(index)] for index in selected], np.max(scores, axis=1)


def metric(predicted: list[str], truth: list[dict[str, Any]], key: str) -> dict[str, Any]:
    labels = sorted({str(row[key]) for row in truth})
    correct = [predicted[index] == str(row[key]) for index, row in enumerate(truth)]
    per_label = {label: float(np.mean([correct[index] for index, row in enumerate(truth) if str(row[key]) == label])) for label in labels}
    chart = {name: float(np.mean([correct[index] for index, row in enumerate(truth) if row["chart"] == name])) for name in ("identity", "rotated")}
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "chart_accuracy": chart,
        "chart_accuracy_gap": abs(chart["identity"] - chart["rotated"]),
        "count": len(truth),
    }


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    predictions = read_jsonl(OUT_ROOT / "predictions/confirmation_predictions.jsonl")
    metadata = read_json(OUT_ROOT / "analysis/prototype_metadata.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    protocol_body = dict(protocol)
    stored_protocol = protocol_body.pop("protocol_digest")
    add("protocol_digest", digest(protocol_body) == stored_protocol)
    add("script_hash", sha256_file(MAIN_SCRIPT) == protocol["script_sha256"])
    add("protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add("candidate_frozen", protocol["candidate"] == "functional_tomography")
    add("prediction_file_hash", sha256_file(OUT_ROOT / "predictions/confirmation_predictions.jsonl") == manifest["prediction_sha256"])
    manifest_body = dict(manifest)
    stored_prediction_digest = manifest_body.pop("prediction_digest")
    add("prediction_manifest_digest", digest(manifest_body) == stored_prediction_digest)
    add("truth_not_read_at_prediction", not bool(manifest["confirmation_truth_read"]))

    discovery_truth = read_jsonl(SOURCE_ROOT / "runs/discovery/sealed_truth.jsonl")
    confirmation_truth = read_jsonl(SOURCE_ROOT / "runs/confirmation/sealed_truth.jsonl")
    with np.load(SOURCE_ROOT / "runs/discovery/feature_pack.npz") as pack:
        discovery = {name: np.asarray(pack[name]) for name in pack.files}
    with np.load(SOURCE_ROOT / "runs/confirmation/feature_pack.npz") as pack:
        confirmation = {name: np.asarray(pack[name]) for name in pack.files}
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as pack:
        stored_prototypes = {name: np.asarray(pack[name]) for name in pack.files}
    add("prediction_count", len(predictions) == len(confirmation_truth) == manifest["prediction_count"])
    add("prediction_alignment", all(predictions[index]["index"] == confirmation_truth[index]["index"] and predictions[index]["unit_id"] == confirmation_truth[index]["unit_id"] for index in range(len(predictions))))

    fit_indices = [index for index, row in enumerate(discovery_truth) if int(row["replicate"]) in protocol["fit_replicates"]]
    validation_indices = [index for index, row in enumerate(discovery_truth) if int(row["replicate"]) in protocol["validation_replicates"]]
    add("fit_count", len(fit_indices) == fit["fit_count"])
    add("validation_count", len(validation_indices) == fit["validation_count"])
    recomputed_metrics = {}
    for algorithm in protocol["algorithms"]:
        group_labels, group_prototype = build_prototypes(discovery[algorithm], discovery_truth, fit_indices, "functional_group")
        exact_labels, exact_prototype = build_prototypes(discovery[algorithm], discovery_truth, fit_indices, "mechanism")
        add(f"prototype.group_labels.{algorithm}", group_labels == metadata[algorithm]["group_labels"])
        add(f"prototype.exact_labels.{algorithm}", exact_labels == metadata[algorithm]["exact_labels"])
        add(f"prototype.group_values.{algorithm}", bool(np.allclose(group_prototype, stored_prototypes[f"{algorithm}__group"], rtol=0.0, atol=2e-7)))
        add(f"prototype.exact_values.{algorithm}", bool(np.allclose(exact_prototype, stored_prototypes[f"{algorithm}__exact"], rtol=0.0, atol=2e-7)))

        validation_group, _ = predict(discovery[algorithm][validation_indices], group_labels, group_prototype)
        validation_exact, _ = predict(discovery[algorithm][validation_indices], exact_labels, exact_prototype)
        validation_truth = [discovery_truth[index] for index in validation_indices]
        expected_validation_group = metric(validation_group, validation_truth, "functional_group")
        expected_validation_exact = metric(validation_exact, validation_truth, "mechanism")
        for key in ("accuracy", "min_label_accuracy", "chart_accuracy_gap", "count"):
            add(f"validation.group.{algorithm}.{key}", math.isclose(float(fit["validation_metrics"][algorithm]["group"][key]), float(expected_validation_group[key]), rel_tol=0.0, abs_tol=1e-12))
            add(f"validation.exact.{algorithm}.{key}", math.isclose(float(fit["validation_metrics"][algorithm]["exact"][key]), float(expected_validation_exact[key]), rel_tol=0.0, abs_tol=1e-12))

        # The prediction command intentionally reloads the persisted float32
        # prototypes.  Replay that exact numerical path here; the float64
        # recomputation above audits construction but can break near ties.
        group_prediction, group_confidence = predict(
            confirmation[algorithm],
            group_labels,
            stored_prototypes[f"{algorithm}__group"],
        )
        exact_prediction, exact_confidence = predict(
            confirmation[algorithm],
            exact_labels,
            stored_prototypes[f"{algorithm}__exact"],
        )
        for index, row in enumerate(predictions):
            stored = row["algorithms"][algorithm]
            add(f"prediction.group.{algorithm}.{index}", stored["group"] == group_prediction[index])
            add(f"prediction.exact.{algorithm}.{index}", stored["exact"] == exact_prediction[index])
            add(f"prediction.group_cosine.{algorithm}.{index}", math.isclose(float(stored["group_cosine"]), float(group_confidence[index]), rel_tol=0.0, abs_tol=3e-7))
            add(f"prediction.exact_cosine.{algorithm}.{index}", math.isclose(float(stored["exact_cosine"]), float(exact_confidence[index]), rel_tol=0.0, abs_tol=3e-7))
        group_metric = metric(group_prediction, confirmation_truth, "functional_group")
        exact_metric = metric(exact_prediction, confirmation_truth, "mechanism")
        recomputed_metrics[algorithm] = {"group": group_metric, "exact": exact_metric}
        for endpoint, values in (("group", group_metric), ("exact", exact_metric)):
            stored_values = score["algorithm_metrics"][algorithm][endpoint]
            for key in ("accuracy", "min_label_accuracy", "chart_accuracy_gap", "count"):
                add(f"score.{algorithm}.{endpoint}.{key}", math.isclose(float(stored_values[key]), float(values[key]), rel_tol=0.0, abs_tol=1e-12))
            for label, value in values["per_label_accuracy"].items():
                add(f"score.{algorithm}.{endpoint}.label.{label}", math.isclose(float(stored_values["per_label_accuracy"][label]), float(value), rel_tol=0.0, abs_tol=1e-12))

    expected_tail = ["degraded" if float(confirmation["tail_ratio"][index]) < protocol["tail_threshold"] else "stable" for index in range(len(confirmation_truth))]
    for index, value in enumerate(expected_tail):
        add(f"tail_prediction.{index}", predictions[index]["tail_prediction"] == value)
    tail_accuracy = float(np.mean([expected_tail[index] == confirmation_truth[index]["tail"] for index in range(len(expected_tail))]))
    add("tail_accuracy", math.isclose(float(score["tail_accuracy"]), tail_accuracy, rel_tol=0.0, abs_tol=1e-12))

    candidate = recomputed_metrics[protocol["candidate"]]["group"]
    t = protocol["thresholds"]
    expected_candidate = {
        "group_accuracy": candidate["accuracy"] >= t["confirmation_group_accuracy_min"],
        "min_group_accuracy": candidate["min_label_accuracy"] >= t["confirmation_min_group_accuracy_min"],
        "chart_accuracy_gap": candidate["chart_accuracy_gap"] <= t["chart_accuracy_gap_max"],
        "matched_chart_cosine": score["matched_chart_cosine_min"] >= t["matched_chart_cosine_min"],
        "tail_accuracy": tail_accuracy >= t["tail_accuracy_min"],
        "equivalence_boundary_respected": True,
    }
    add("candidate_checks", score["candidate_checks"] == expected_candidate, {"stored": score["candidate_checks"], "expected": expected_candidate})
    add("candidate_confirmed", bool(score["candidate_confirmed"]) == all(expected_candidate.values()))
    fit_body = dict(fit)
    stored_fit_digest = fit_body.pop("fit_digest")
    add("fit_digest", digest(fit_body) == stored_fit_digest)
    score_body = dict(score)
    stored_score_digest = score_body.pop("score_digest")
    add("score_digest", digest(score_body) == stored_score_digest)
    expected_final = bool(fit["candidate_qualified"] and score["candidate_confirmed"])
    add("final_controlled_qualification", bool(final["controlled_functional_tomography_qualified"]) == expected_final)
    add("final_exact_denied", not bool(final["exact_implementation_identification_qualified"]))
    add("final_natural_claim_denied", not bool(final["pretrained_model_mechanism_claim_authorized"]))
    final_body = dict(final)
    stored_final_digest = final_body.pop("final_digest")
    add("final_digest", digest(final_body) == stored_final_digest)

    passed_count = sum(row["passed"] for row in checks)
    result = {
        "phase": PHASE,
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": passed_count,
        "all_checks_passed": passed_count == len(checks),
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    path = OUT_ROOT / "audit/independent_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({key: result[key] for key in ("check_count", "passed_count", "all_checks_passed", "audit_digest")}))
    if not result["all_checks_passed"]:
        raise SystemExit([row for row in checks if not row["passed"]][:10])


if __name__ == "__main__":
    main()
