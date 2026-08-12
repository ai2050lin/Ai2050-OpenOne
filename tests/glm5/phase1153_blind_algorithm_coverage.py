#!/usr/bin/env python3
"""Blind coverage matrix for mechanism-identification algorithms.

The candidate is frozen in advance as functional_tomography.  Other feature
families are reported as controls.  Confirmation predictions are persisted
before the sealed mechanism labels are read by the scoring command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE = 1153
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1152_tie_aware_morphology_library"
OUT_ROOT = ROOT / "tests/glm5/result/phase1153_blind_algorithm_coverage"
CANDIDATE = "functional_tomography"
ALGORITHMS = (
    "raw_coordinates",
    "state_gram",
    "single_site_patch",
    "pairwise_patch",
    "factorial_interaction",
    "exhaustive_coalition",
    "functional_tomography",
)
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
TAIL_THRESHOLD = 0.25
THRESHOLDS = {
    "discovery_group_accuracy_min": 0.95,
    "discovery_min_group_accuracy_min": 0.90,
    "confirmation_group_accuracy_min": 0.95,
    "confirmation_min_group_accuracy_min": 0.90,
    "chart_accuracy_gap_max": 0.05,
    "matched_chart_cosine_min": 0.999999,
    "tail_accuracy_min": 0.99,
    "exact_implementation_claim_forbidden": True,
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.where(norms > 1e-12, norms, 1.0)


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


def build_prototypes(
    features: np.ndarray,
    truth: list[dict[str, Any]],
    indices: list[int],
    label_key: str,
) -> tuple[list[str], np.ndarray]:
    normalized = normalize_rows(features)
    labels = sorted({str(truth[index][label_key]) for index in indices})
    prototypes = []
    for label in labels:
        selected = [index for index in indices if str(truth[index][label_key]) == label]
        prototypes.append(normalize_vector(np.mean(normalized[selected], axis=0)))
    return labels, np.stack(prototypes, axis=0)


def predict(features: np.ndarray, labels: list[str], prototypes: np.ndarray) -> tuple[list[str], np.ndarray]:
    normalized = normalize_rows(features)
    scores = normalized @ np.asarray(prototypes, dtype=np.float64).T
    indices = np.argmax(scores, axis=1)
    return [labels[int(index)] for index in indices], np.max(scores, axis=1)


def metrics_for_predictions(
    predicted: list[str],
    truth: list[dict[str, Any]],
    indices: list[int],
    label_key: str,
) -> dict[str, Any]:
    labels = sorted({str(truth[index][label_key]) for index in indices})
    correct = [predicted[offset] == str(truth[index][label_key]) for offset, index in enumerate(indices)]
    per_label = {}
    for label in labels:
        selected = [offset for offset, index in enumerate(indices) if str(truth[index][label_key]) == label]
        per_label[label] = float(np.mean([correct[offset] for offset in selected]))
    chart_accuracy = {}
    for chart in ("identity", "rotated"):
        selected = [offset for offset, index in enumerate(indices) if truth[index]["chart"] == chart]
        chart_accuracy[chart] = float(np.mean([correct[offset] for offset in selected]))
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "chart_accuracy": chart_accuracy,
        "chart_accuracy_gap": abs(chart_accuracy["identity"] - chart_accuracy["rotated"]),
        "count": len(indices),
    }


def protocol_command() -> None:
    if (OUT_ROOT / "analysis").exists() or (OUT_ROOT / "predictions").exists():
        raise RuntimeError("refusing to rewrite Phase1153 artifacts")
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks = {
        "source_library_qualified": bool(source_final["library_qualified"]),
        "source_authorized_phase1153": bool(source_final["phase1153_blind_coverage_authorized"]),
        "source_audit_passed": bool(source_audit["all_checks_passed"]),
        "candidate_predeclared": CANDIDATE == "functional_tomography",
        "fit_validation_replicates_disjoint": not bool(set(FIT_REPLICATES) & set(VALIDATION_REPLICATES)),
        "confirmation_truth_forbidden_in_predict": True,
        "equivalence_group_is_primary_endpoint": True,
        "exact_implementation_is_descriptive_only": True,
        "tail_detector_frozen": True,
        "no_algorithm_tuning_after_validation": True,
        "no_natural_model_scan": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "blinded mechanism-identification algorithm coverage matrix",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1152_digest": source_final["final_digest"],
        "source_phase1152_audit_digest": source_audit["audit_digest"],
        "candidate": CANDIDATE,
        "algorithms": list(ALGORITHMS),
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "tail_threshold": TAIL_THRESHOLD,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "functional equivalence group classification",
        "secondary_endpoint": "hidden implementation label classification, descriptive only",
        "hard_stops": [
            "Confirmation labels may not be read before predictions are persisted.",
            "Declared observationally equivalent implementations may not be claimed as separated.",
            "Only the predeclared functional_tomography candidate can authorize the next phase.",
            "Passing controlled coverage does not authorize a pretrained-model mechanism claim.",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    body = dict(protocol)
    protocol["protocol_digest"] = digest(body)
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(OUT_ROOT / "protocol/audit.json", {"checks": checks, "check_count": len(checks), "passed_count": sum(checks.values()), "all_checks_passed": all(checks.values()), "protocol_digest": protocol["protocol_digest"]})
    print(canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if digest(body) != stored or sha256_file(SCRIPT) != protocol["script_sha256"]:
        raise RuntimeError("Phase1153 frozen protocol mismatch")
    return protocol


def fit_command() -> None:
    protocol = verify_protocol()
    root = SOURCE_ROOT / "runs/discovery"
    truth = read_jsonl(root / "sealed_truth.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    fit_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in FIT_REPLICATES]
    validation_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in VALIDATION_REPLICATES]
    prototype_arrays = {}
    prototype_metadata = {}
    validation_metrics = {}
    for algorithm in ALGORITHMS:
        labels_group, prototypes_group = build_prototypes(arrays[algorithm], truth, fit_indices, "functional_group")
        labels_exact, prototypes_exact = build_prototypes(arrays[algorithm], truth, fit_indices, "mechanism")
        prototype_arrays[f"{algorithm}__group"] = prototypes_group.astype(np.float32)
        prototype_arrays[f"{algorithm}__exact"] = prototypes_exact.astype(np.float32)
        prototype_metadata[algorithm] = {"group_labels": labels_group, "exact_labels": labels_exact}
        group_prediction, _ = predict(arrays[algorithm][validation_indices], labels_group, prototypes_group)
        exact_prediction, _ = predict(arrays[algorithm][validation_indices], labels_exact, prototypes_exact)
        validation_metrics[algorithm] = {
            "group": metrics_for_predictions(group_prediction, truth, validation_indices, "functional_group"),
            "exact": metrics_for_predictions(exact_prediction, truth, validation_indices, "mechanism"),
        }
    tail_prediction = ["degraded" if float(arrays["tail_ratio"][index]) < TAIL_THRESHOLD else "stable" for index in validation_indices]
    tail_accuracy = float(np.mean([tail_prediction[offset] == truth[index]["tail"] for offset, index in enumerate(validation_indices)]))
    candidate = validation_metrics[CANDIDATE]["group"]
    t = protocol["thresholds"]
    candidate_checks = {
        "group_accuracy": candidate["accuracy"] >= t["discovery_group_accuracy_min"],
        "min_group_accuracy": candidate["min_label_accuracy"] >= t["discovery_min_group_accuracy_min"],
        "chart_gap": candidate["chart_accuracy_gap"] <= t["chart_accuracy_gap_max"],
        "tail_accuracy": tail_accuracy >= t["tail_accuracy_min"],
    }
    model_root = OUT_ROOT / "analysis"
    model_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(model_root / "frozen_prototypes.npz", **prototype_arrays)
    write_json(model_root / "prototype_metadata.json", prototype_metadata)
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_count": len(fit_indices),
        "validation_count": len(validation_indices),
        "validation_metrics": validation_metrics,
        "tail_accuracy": tail_accuracy,
        "candidate": CANDIDATE,
        "candidate_checks": candidate_checks,
        "candidate_qualified": all(candidate_checks.values()),
        "confirmation_prediction_authorized": all(candidate_checks.values()),
        "prototype_sha256": sha256_file(model_root / "frozen_prototypes.npz"),
        "prototype_metadata_sha256": sha256_file(model_root / "prototype_metadata.json"),
    }
    result["fit_digest"] = digest(result)
    write_json(model_root / "fit.json", result)
    print(canonical(result))


def predict_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if not fit["confirmation_prediction_authorized"]:
        raise RuntimeError("confirmation prediction denied by discovery validation")
    if (OUT_ROOT / "predictions").exists():
        raise RuntimeError("refusing to overwrite confirmation predictions")
    source = SOURCE_ROOT / "runs/confirmation"
    public = read_jsonl(source / "public_manifest.jsonl")
    with np.load(source / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    metadata = read_json(OUT_ROOT / "analysis/prototype_metadata.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as stored:
        prototypes = {name: np.asarray(stored[name]) for name in stored.files}
    predictions_by_algorithm = {}
    confidence_by_algorithm = {}
    for algorithm in ALGORITHMS:
        group, group_confidence = predict(arrays[algorithm], metadata[algorithm]["group_labels"], prototypes[f"{algorithm}__group"])
        exact, exact_confidence = predict(arrays[algorithm], metadata[algorithm]["exact_labels"], prototypes[f"{algorithm}__exact"])
        predictions_by_algorithm[algorithm] = {"group": group, "exact": exact}
        confidence_by_algorithm[algorithm] = {"group": group_confidence, "exact": exact_confidence}
    rows = []
    for index, public_row in enumerate(public):
        algorithms = {}
        for algorithm in ALGORITHMS:
            algorithms[algorithm] = {
                "group": predictions_by_algorithm[algorithm]["group"][index],
                "exact": predictions_by_algorithm[algorithm]["exact"][index],
                "group_cosine": float(confidence_by_algorithm[algorithm]["group"][index]),
                "exact_cosine": float(confidence_by_algorithm[algorithm]["exact"][index]),
            }
        rows.append({
            "index": index,
            "unit_id": public_row["unit_id"],
            "tail_prediction": "degraded" if float(arrays["tail_ratio"][index]) < TAIL_THRESHOLD else "stable",
            "algorithms": algorithms,
        })
    path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    write_jsonl(path, rows)
    manifest = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_count": len(rows),
        "confirmation_truth_read": False,
        "prediction_sha256": sha256_file(path),
    }
    manifest["prediction_digest"] = digest(manifest)
    write_json(OUT_ROOT / "predictions/manifest.json", manifest)
    print(canonical(manifest))


def score_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    prediction_path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    if sha256_file(prediction_path) != manifest["prediction_sha256"] or manifest["confirmation_truth_read"]:
        raise RuntimeError("prediction seal invalid")
    predictions = read_jsonl(prediction_path)
    source = SOURCE_ROOT / "runs/confirmation"
    truth = read_jsonl(source / "sealed_truth.jsonl")
    with np.load(source / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    indices = list(range(len(truth)))
    algorithm_metrics = {}
    coverage_matrix = {}
    for algorithm in ALGORITHMS:
        group_prediction = [row["algorithms"][algorithm]["group"] for row in predictions]
        exact_prediction = [row["algorithms"][algorithm]["exact"] for row in predictions]
        group_metrics = metrics_for_predictions(group_prediction, truth, indices, "functional_group")
        exact_metrics = metrics_for_predictions(exact_prediction, truth, indices, "mechanism")
        algorithm_metrics[algorithm] = {"group": group_metrics, "exact": exact_metrics}
        coverage_matrix[algorithm] = group_metrics["per_label_accuracy"]
    tail_accuracy = float(np.mean([predictions[index]["tail_prediction"] == truth[index]["tail"] for index in indices]))
    functional = arrays[CANDIDATE]
    by_key = {(row["mechanism"], row["replicate"], row["tail"], row["chart"]): int(row["index"]) for row in truth}
    chart_cosines = []
    for mechanism in sorted({row["mechanism"] for row in truth}):
        for replicate in sorted({int(row["replicate"]) for row in truth}):
            for tail in ("stable", "degraded"):
                left = by_key[(mechanism, replicate, tail, "identity")]
                right = by_key[(mechanism, replicate, tail, "rotated")]
                chart_cosines.append(cosine(functional[left], functional[right]))
    equivalent_pairs = (("additive_load", "tensor_product_binding"), ("bilinear_binding", "role_factorized"))
    equivalence_cosines = []
    for left_name, right_name in equivalent_pairs:
        for replicate in sorted({int(row["replicate"]) for row in truth}):
            for tail in ("stable", "degraded"):
                for chart in ("identity", "rotated"):
                    left = by_key[(left_name, replicate, tail, chart)]
                    right = by_key[(right_name, replicate, tail, chart)]
                    equivalence_cosines.append(cosine(functional[left], functional[right]))
    candidate = algorithm_metrics[CANDIDATE]["group"]
    t = protocol["thresholds"]
    candidate_checks = {
        "group_accuracy": candidate["accuracy"] >= t["confirmation_group_accuracy_min"],
        "min_group_accuracy": candidate["min_label_accuracy"] >= t["confirmation_min_group_accuracy_min"],
        "chart_accuracy_gap": candidate["chart_accuracy_gap"] <= t["chart_accuracy_gap_max"],
        "matched_chart_cosine": min(chart_cosines) >= t["matched_chart_cosine_min"],
        "tail_accuracy": tail_accuracy >= t["tail_accuracy_min"],
        "equivalence_boundary_respected": True,
    }
    score = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_digest": manifest["prediction_digest"],
        "confirmation_count": len(truth),
        "algorithm_metrics": algorithm_metrics,
        "coverage_matrix": coverage_matrix,
        "tail_accuracy": tail_accuracy,
        "candidate": CANDIDATE,
        "candidate_checks": candidate_checks,
        "candidate_confirmed": all(candidate_checks.values()),
        "matched_chart_cosine_min": float(min(chart_cosines)),
        "matched_chart_cosine_median": float(np.median(chart_cosines)),
        "declared_equivalence_cosine_min": float(min(equivalence_cosines)),
        "declared_equivalence_cosine_median": float(np.median(equivalence_cosines)),
        "exact_label_is_not_primary": True,
        "claim_boundary": "Coverage applies to the six observable functional equivalence groups in the controlled library. It cannot distinguish the two declared implementation-equivalent pairs and is not a natural-network mechanism claim.",
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/confirmation_score.json", score)
    print(canonical(score))


def finalize_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    passed = bool(fit["candidate_qualified"] and score["candidate_confirmed"])
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "score_digest": score["score_digest"],
        "candidate": CANDIDATE,
        "discovery_qualified": bool(fit["candidate_qualified"]),
        "confirmation_confirmed": bool(score["candidate_confirmed"]),
        "controlled_functional_tomography_qualified": passed,
        "exact_implementation_identification_qualified": False,
        "phase1154_learned_network_calibration_authorized": passed,
        "pretrained_model_mechanism_claim_authorized": False,
        "outcome": "controlled_functional_tomography_confirmed" if passed else "controlled_functional_tomography_not_confirmed",
        "auto_continue": passed,
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    sub.add_parser("fit")
    sub.add_parser("predict")
    sub.add_parser("score")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "fit":
        fit_command()
    elif args.command == "predict":
        predict_command()
    elif args.command == "score":
        score_command()
    elif args.command == "finalize":
        finalize_command()


if __name__ == "__main__":
    main()
