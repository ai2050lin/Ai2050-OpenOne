#!/usr/bin/env python3
"""Independent discovery-stop audit for Phase1154."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1154_learned_morphology_external_validity"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1154_learned_morphology_external_validity.py"
FINALIZER = ROOT / "tests/glm5/phase1154_discovery_stop_finalize.py"
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


def build_prototypes(features: np.ndarray, truth: list[dict[str, Any]], indices: list[int]) -> tuple[list[str], np.ndarray]:
    normalized = normalize_rows(features)
    labels = sorted({truth[index]["functional_group"] for index in indices})
    rows = []
    for label in labels:
        selected = [index for index in indices if truth[index]["functional_group"] == label]
        rows.append(normalize_vector(np.mean(normalized[selected], axis=0)))
    return labels, np.stack(rows)


def predict(features: np.ndarray, labels: list[str], prototypes: np.ndarray) -> list[str]:
    scores = normalize_rows(features) @ prototypes.T
    return [labels[int(index)] for index in np.argmax(scores, axis=1)]


def metrics(predicted: list[str], truth: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({row["functional_group"] for row in truth})
    correct = [predicted[index] == row["functional_group"] for index, row in enumerate(truth)]
    per_label = {label: float(np.mean([correct[index] for index, row in enumerate(truth) if row["functional_group"] == label])) for label in labels}
    chart = {name: float(np.mean([correct[index] for index, row in enumerate(truth) if row["chart"] == name])) for name in ("identity", "rotated")}
    return {"accuracy": float(np.mean(correct)), "min_label_accuracy": float(min(per_label.values())), "per_label_accuracy": per_label, "chart_accuracy": chart, "chart_accuracy_gap": abs(chart["identity"] - chart["rotated"]), "count": len(truth)}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize_vector(left), normalize_vector(right)))


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    root = OUT_ROOT / "runs/discovery"
    public = read_jsonl(root / "public_manifest.jsonl")
    truth = read_jsonl(root / "sealed_truth.jsonl")
    training = read_jsonl(root / "training_metrics.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    labels_metadata = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as pack:
        stored_prototypes = {name: np.asarray(pack[name]) for name in pack.files}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    body = dict(protocol)
    stored_protocol = body.pop("protocol_digest")
    add("protocol_digest", digest(body) == stored_protocol)
    add("main_script_hash", sha256_file(MAIN_SCRIPT) == protocol["script_sha256"])
    add("protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add("finalizer_present", FINALIZER.exists())
    add("feature_hash", sha256_file(root / "feature_pack.npz") == summary["feature_pack_sha256"])
    add("public_hash", sha256_file(root / "public_manifest.jsonl") == summary["public_manifest_sha256"])
    add("truth_hash", sha256_file(root / "sealed_truth.jsonl") == summary["sealed_truth_sha256"])
    add("training_hash", sha256_file(root / "training_metrics.jsonl") == summary["training_metrics_sha256"])
    add("counts", len(training) == 24 and len(public) == len(truth) == 48)
    add("public_blind", all("functional_group" not in row and "chart" not in row for row in public))
    add("alignment", all(public[index]["index"] == truth[index]["index"] == index and public[index]["unit_id"] == truth[index]["unit_id"] for index in range(len(public))))
    add("unique_units", len({row["unit_id"] for row in public}) == 48)
    add("unique_models", len({row["model_id"] for row in training}) == 24)

    for group in protocol["groups"]:
        add(f"group_count.{group}", sum(row["functional_group"] == group for row in truth) == 8)
        add(f"model_count.{group}", sum(row["group"] == group for row in training) == 4)
    for row in training:
        model_path = root / "models" / f"{row['model_id']}.pt"
        add(f"model_exists.{row['model_id']}", model_path.exists())
        add(f"model_hash.{row['model_id']}", model_path.exists() and sha256_file(model_path) == row["model_sha256"])
        add(f"model_accuracy.{row['model_id']}", float(row["accuracy"]) == 1.0)
        add(f"model_probability.{row['model_id']}", float(row["min_probability"]) >= protocol["thresholds"]["model_min_probability_min"], row["min_probability"])

    for algorithm in protocol["algorithms"]:
        add(f"feature_rows.{algorithm}", int(arrays[algorithm].shape[0]) == 48, list(arrays[algorithm].shape))
        add(f"feature_finite.{algorithm}", bool(np.isfinite(arrays[algorithm]).all()))
        add(f"feature_nonzero.{algorithm}", bool(np.any(np.abs(arrays[algorithm]) > 0)))
    recomputed_summary = {
        "accuracy_min": float(min(row["accuracy"] for row in training)),
        "min_probability_min": float(min(row["min_probability"] for row in training)),
        "steps_min": int(min(row["steps"] for row in training)),
        "steps_max": int(max(row["steps"] for row in training)),
        "finite_fraction": float(np.mean([np.isfinite(value).mean() for value in arrays.values()])),
    }
    functional = arrays[protocol["candidate"]]
    by_key = {(row["functional_group"], row["replicate"], row["chart"]): int(row["index"]) for row in truth}
    chart_values = []
    for group in protocol["groups"]:
        for replicate in range(int(protocol["replicates"])):
            value = cosine(functional[by_key[(group, replicate, "identity")]], functional[by_key[(group, replicate, "rotated")]])
            chart_values.append(value)
            add(f"chart.{group}.{replicate}", math.isfinite(value), value)
    recomputed_summary["chart_cosine_min"] = float(min(chart_values))
    for key, value in recomputed_summary.items():
        add(f"summary.{key}", math.isclose(float(summary[key]), float(value), rel_tol=0.0, abs_tol=1e-7), {"stored": summary[key], "recomputed": value})
    summary_body = dict(summary)
    stored_summary_digest = summary_body.pop("summary_digest")
    add("summary_digest", digest(summary_body) == stored_summary_digest)
    add("behavior_gate", bool(summary["behavior_gate_passed"]) == all(summary["checks"].values()))

    fit_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in protocol["fit_replicates"]]
    validation_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in protocol["validation_replicates"]]
    validation_truth = [truth[index] for index in validation_indices]
    recomputed_metrics = {}
    for algorithm in protocol["algorithms"]:
        labels, prototypes = build_prototypes(arrays[algorithm], truth, fit_indices)
        add(f"labels.{algorithm}", labels == labels_metadata[algorithm])
        add(f"prototypes.{algorithm}", bool(np.allclose(prototypes, stored_prototypes[algorithm], rtol=0.0, atol=2e-7)))
        predicted = predict(arrays[algorithm][validation_indices], labels, prototypes)
        values = metrics(predicted, validation_truth)
        recomputed_metrics[algorithm] = values
        stored = fit["algorithm_metrics"][algorithm]
        for key in ("accuracy", "min_label_accuracy", "chart_accuracy_gap", "count"):
            add(f"fit.{algorithm}.{key}", math.isclose(float(stored[key]), float(values[key]), rel_tol=0.0, abs_tol=1e-12))
        for label, value in values["per_label_accuracy"].items():
            add(f"fit.{algorithm}.label.{label}", math.isclose(float(stored["per_label_accuracy"][label]), value, rel_tol=0.0, abs_tol=1e-12))
    candidate = recomputed_metrics[protocol["candidate"]]
    t = protocol["thresholds"]
    expected_checks = {"group_accuracy": candidate["accuracy"] >= t["discovery_group_accuracy_min"], "min_group_accuracy": candidate["min_label_accuracy"] >= t["discovery_min_group_accuracy_min"], "chart_gap": candidate["chart_accuracy_gap"] <= t["chart_accuracy_gap_max"]}
    add("candidate_checks", fit["candidate_checks"] == expected_checks, {"stored": fit["candidate_checks"], "expected": expected_checks})
    add("candidate_failed", not bool(fit["candidate_qualified"]) and not all(expected_checks.values()))
    fit_body = dict(fit)
    stored_fit_digest = fit_body.pop("fit_digest")
    add("fit_digest", digest(fit_body) == stored_fit_digest)
    add("confirmation_absent", not (OUT_ROOT / "runs/confirmation").exists())
    add("predictions_absent", not (OUT_ROOT / "predictions").exists())
    add("final_stop_reason", final["stop_reason"] == "discovery_identification_gate_failed")
    add("phase1155_denied", not bool(final["phase1155_free_network_tomography_authorized"]))
    add("natural_claim_denied", not bool(final["pretrained_model_mechanism_claim_authorized"]))
    add("auto_continue_false", not bool(final["auto_continue"]))
    final_body = dict(final)
    stored_final_digest = final_body.pop("final_digest")
    add("final_digest", digest(final_body) == stored_final_digest)

    passed_count = sum(row["passed"] for row in checks)
    result = {
        "phase": 1154,
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
        "finalizer_sha256": sha256_file(FINALIZER),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": passed_count,
        "all_checks_passed": passed_count == len(checks),
        "recomputed_candidate_metrics": recomputed_metrics[protocol["candidate"]],
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
