#!/usr/bin/env python3
"""Independent artifact recomputation for Phase1156."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1156_causal_use_quotient_calibration"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1156_causal_use_quotient_calibration.py"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1155_learning_gauge_quotient_calibration"


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


def predict(features: np.ndarray, labels: list[str], prototypes: np.ndarray) -> list[str]:
    scores = normalize_rows(features) @ np.asarray(prototypes, dtype=np.float64).T
    return [labels[int(index)] for index in np.argmax(scores, axis=1)]


def metrics(predicted: list[str], truth: list[dict[str, Any]], indices: list[int], gauges: list[str]) -> dict[str, Any]:
    correct = [predicted[offset] == str(truth[index]["use_label"]) for offset, index in enumerate(indices)]
    labels = sorted({str(truth[index]["use_label"]) for index in indices})
    per_label = {}
    for label in labels:
        selected = [offset for offset, index in enumerate(indices) if str(truth[index]["use_label"]) == label]
        per_label[label] = float(np.mean([correct[offset] for offset in selected]))
    per_gauge = {}
    for gauge in gauges:
        selected = [offset for offset, index in enumerate(indices) if truth[index]["gauge"] == gauge]
        if selected:
            per_gauge[gauge] = float(np.mean([correct[offset] for offset in selected]))
    bit_accuracy = {}
    for bit_index, factor in enumerate(("row", "col", "context")):
        bit_accuracy[factor] = float(
            np.mean(
                [
                    int(predicted[offset][:3][bit_index]) == int(truth[index]["use_mask"][bit_index])
                    for offset, index in enumerate(indices)
                ]
            )
        )
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "per_gauge_accuracy": per_gauge,
        "gauge_accuracy_gap": float(max(per_gauge.values()) - min(per_gauge.values())) if per_gauge else 0.0,
        "factor_bit_accuracy": bit_accuracy,
        "count": len(indices),
    }


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    checks: dict[str, bool] = {}
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
        raise RuntimeError("Phase1156 artifacts are incomplete")

    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_body = dict(protocol)
    protocol_digest = protocol_body.pop("protocol_digest")
    checks["protocol_digest"] = digest(protocol_body) == protocol_digest
    checks["main_script_hash"] = sha256_file(MAIN_SCRIPT) == protocol["script_sha256"]
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    checks["protocol_audit_passed"] = bool(protocol_audit["all_checks_passed"])
    checks["protocol_audit_link"] = protocol_audit["protocol_digest"] == protocol_digest
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks["source_final_link"] = source_final["final_digest"] == protocol["source_phase1155_digest"]
    checks["source_audit_link"] = source_audit["audit_digest"] == protocol["source_phase1155_audit_digest"]
    checks["source_confirmed"] = bool(source_final["coarse_dependency_quotient_confirmed"] and source_audit["all_checks_passed"])
    checks["candidate_frozen"] = protocol["candidate"] == "matched_transport_causal_use"
    checks["eight_use_masks"] = len(protocol["use_masks"]) == 8

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
            "representation_manifest": root / "representation_manifest.jsonl",
            "summary": root / "summary.json",
        }
        for name, path in paths.items():
            checks[f"{split}::exists::{name}"] = path.exists()
        truth = read_jsonl(paths["sealed_truth"])
        public = read_jsonl(paths["public_manifest"])
        diagnostics = read_jsonl(paths["diagnostics"])
        representations = read_jsonl(paths["representation_manifest"])
        summary = read_json(paths["summary"])
        with np.load(paths["feature_pack"]) as pack:
            arrays = {name: np.asarray(pack[name]) for name in pack.files}
        split_data[split] = {
            "truth": truth,
            "public": public,
            "diagnostics": diagnostics,
            "representations": representations,
            "summary": summary,
            "arrays": arrays,
        }
        checks[f"{split}::unit_count"] = len(truth) == len(public) == len(diagnostics) == 128
        checks[f"{split}::representation_count"] = len(representations) == 4
        checks[f"{split}::indices"] = all(row["index"] == index for index, row in enumerate(truth))
        checks[f"{split}::public_truth_ids"] = [row["unit_id"] for row in public] == [row["unit_id"] for row in truth]
        checks[f"{split}::diagnostic_ids"] = [row["unit_id"] for row in diagnostics] == [row["unit_id"] for row in truth]
        for gauge in gauges:
            checks[f"{split}::gauge_count::{gauge}"] = sum(row["gauge"] == gauge for row in truth) == 32
        for label in protocol["use_masks"]:
            checks[f"{split}::label_count::{label}"] = sum(row["use_label"] == label for row in truth) == 16
        for algorithm in algorithms:
            checks[f"{split}::feature::{algorithm}"] = algorithm in arrays and arrays[algorithm].shape[0] == 128
            checks[f"{split}::finite::{algorithm}"] = bool(np.isfinite(arrays[algorithm]).all())
        checks[f"{split}::feature_hash"] = sha256_file(paths["feature_pack"]) == summary["feature_pack_sha256"]
        checks[f"{split}::public_hash"] = sha256_file(paths["public_manifest"]) == summary["public_manifest_sha256"]
        checks[f"{split}::truth_hash"] = sha256_file(paths["sealed_truth"]) == summary["sealed_truth_sha256"]
        checks[f"{split}::diagnostic_hash"] = sha256_file(paths["diagnostics"]) == summary["diagnostics_sha256"]
        checks[f"{split}::representation_manifest_hash"] = (
            sha256_file(paths["representation_manifest"]) == summary["representation_manifest_sha256"]
        )
        checks[f"{split}::protocol_link"] = summary["protocol_digest"] == protocol_digest
        checks[f"{split}::summary_digest"] = (
            digest({key: value for key, value in summary.items() if key != "summary_digest"}) == summary["summary_digest"]
        )
        checks[f"{split}::run_gate"] = bool(summary["run_gate_passed"] and all(summary["checks"].values()))
        for row in representations:
            path = root / "representations" / f"{row['representation_id']}.npz"
            checks[f"{split}::representation_hash::{row['representation_id']}"] = (
                path.exists() and sha256_file(path) == row["representation_sha256"]
            )

        by_key = {(row["use_label"], int(row["replicate"]), row["gauge"]): int(row["index"]) for row in truth}
        representation_matches = []
        dependency_gauge_matches = []
        candidate_gauge_errors = []
        for replicate in range(protocol["replicates"]):
            for gauge in gauges:
                base = by_key[("000_none", replicate, gauge)]
                for label in protocol["use_masks"]:
                    current = by_key[(label, replicate, gauge)]
                    representation_matches.append(
                        bool(
                            np.array_equal(arrays["state_gram"][base], arrays["state_gram"][current])
                            and np.array_equal(
                                arrays["dependency_rank_quotient"][base], arrays["dependency_rank_quotient"][current]
                            )
                        )
                    )
            for label in protocol["use_masks"]:
                identity = by_key[(label, replicate, "identity")]
                for gauge in gauges[1:]:
                    current = by_key[(label, replicate, gauge)]
                    dependency_gauge_matches.append(
                        bool(
                            np.array_equal(
                                arrays["dependency_rank_quotient"][identity], arrays["dependency_rank_quotient"][current]
                            )
                        )
                    )
                    candidate_gauge_errors.append(
                        float(np.max(np.abs(arrays[protocol["candidate"]][identity] - arrays[protocol["candidate"]][current])))
                    )
        checks[f"{split}::representation_match_recompute"] = close(
            np.mean(representation_matches), summary["representation_mask_match_fraction"]
        )
        checks[f"{split}::dependency_match_recompute"] = close(
            np.mean(dependency_gauge_matches), summary["dependency_gauge_match_fraction"]
        )
        checks[f"{split}::candidate_gauge_error_recompute"] = close(
            max(candidate_gauge_errors), summary["candidate_gauge_abs_error_max"]
        )

        clean_accuracy = [row["clean_accuracy"] for row in diagnostics]
        clean_probability = [row["clean_target_probability_min"] for row in diagnostics]
        clean_error = [row["clean_function_abs_error_max"] for row in diagnostics]
        expected = [row["causal_diagnostics"]["expected_hybrid_probability_min"] for row in diagnostics]
        alpha_zero = [row["causal_diagnostics"]["alpha_zero_tv_max"] for row in diagnostics]
        null_tv = [row["causal_diagnostics"]["matched_null_tv_max"] for row in diagnostics]
        used = [
            row["causal_diagnostics"]["used_donor_probability_min"]
            for row in diagnostics
            if any(truth[row["index"]]["use_mask"])
        ]
        unused = [
            row["causal_diagnostics"]["unused_receiver_probability_min"]
            for row in diagnostics
            if not all(truth[row["index"]]["use_mask"])
        ]
        checks[f"{split}::clean_accuracy_recompute"] = close(min(clean_accuracy), summary["clean_accuracy_min"])
        checks[f"{split}::clean_probability_recompute"] = close(min(clean_probability), summary["clean_target_probability_min"])
        checks[f"{split}::clean_error_recompute"] = close(max(clean_error), summary["clean_function_abs_error_max"])
        checks[f"{split}::expected_recompute"] = close(min(expected), summary["expected_hybrid_probability_min"])
        checks[f"{split}::alpha_zero_recompute"] = close(max(alpha_zero), summary["alpha_zero_tv_max"])
        checks[f"{split}::null_recompute"] = close(max(null_tv), summary["matched_null_tv_max"])
        checks[f"{split}::used_recompute"] = close(min(used), summary["used_donor_probability_min"])
        checks[f"{split}::unused_recompute"] = close(min(unused), summary["unused_receiver_probability_min"])

        camera = arrays[protocol["candidate"]]
        for index, row in enumerate(truth):
            for factor_index in range(3):
                donor = float(camera[index, factor_index * 3])
                receiver = float(camera[index, factor_index * 3 + 1])
                tv = float(camera[index, factor_index * 3 + 2])
                bit = int(row["use_mask"][factor_index])
                checks[f"{split}::camera_truth::{index}::{factor_index}"] = (
                    donor > 0.9999 and receiver < 1e-6 and tv > 0.9999
                    if bit
                    else receiver > 0.9999 and donor < 1e-6 and tv < 1e-6
                )

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
    fit_indices = [
        index
        for index, row in enumerate(dtruth)
        if int(row["replicate"]) in protocol["fit_replicates"] and row["gauge"] == "identity"
    ]
    validation_indices = [
        index for index, row in enumerate(dtruth) if int(row["replicate"]) in protocol["validation_replicates"]
    ]
    checks["fit_count"] = fit["fit_count"] == len(fit_indices) == 24
    checks["validation_count"] = fit["validation_count"] == len(validation_indices) == 32
    for algorithm in algorithms:
        predicted = predict(darrays[algorithm][validation_indices], labels[algorithm]["labels"], prototypes[algorithm])
        recomputed = metrics(predicted, dtruth, validation_indices, gauges)
        checks[f"fit_metric::{algorithm}"] = canonical(recomputed) == canonical(fit["algorithm_metrics"][algorithm])

    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    prediction_path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    checks["prediction_truth_blind"] = manifest["confirmation_truth_read"] is False
    checks["prediction_hash"] = sha256_file(prediction_path) == manifest["prediction_sha256"]
    checks["prediction_count"] = len(predictions) == manifest["prediction_count"] == 128
    checks["prediction_unit_alignment"] = [row["unit_id"] for row in predictions] == [
        row["unit_id"] for row in split_data["confirmation"]["public"]
    ]
    carrays = split_data["confirmation"]["arrays"]
    for algorithm in algorithms:
        expected = predict(carrays[algorithm], labels[algorithm]["labels"], prototypes[algorithm])
        actual = [row["algorithms"][algorithm]["prediction"] for row in predictions]
        checks[f"prediction_recompute::{algorithm}"] = expected == actual

    score = read_json(OUT_ROOT / "analysis/score.json")
    ctruth = split_data["confirmation"]["truth"]
    indices = list(range(len(ctruth)))
    checks["score_digest"] = digest({key: value for key, value in score.items() if key != "score_digest"}) == score["score_digest"]
    checks["score_confirmed"] = bool(score["candidate_confirmed"] and all(score["candidate_checks"].values()))
    for algorithm in algorithms:
        actual = [row["algorithms"][algorithm]["prediction"] for row in predictions]
        recomputed = metrics(actual, ctruth, indices, gauges)
        checks[f"score_metric::{algorithm}"] = canonical(recomputed) == canonical(score["algorithm_metrics"][algorithm])

    final = read_json(OUT_ROOT / "analysis/final.json")
    checks["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"}) == final["final_digest"]
    checks["final_protocol_link"] = final["protocol_digest"] == protocol_digest
    checks["final_split_overlap"] = final["split_overlap"] == 0
    checks["final_candidate_confirmed"] = bool(final["matched_transport_causal_use_camera_confirmed"])
    checks["final_scope_guard"] = not bool(
        final["causal_use_on_independently_learned_networks_confirmed"]
        or final["interaction_hyperedge_identification_confirmed"]
        or final["redundancy_or_gate_identification_confirmed"]
    )
    checks["final_no_free_scan"] = not bool(
        final["free_transformer_scan_authorized"] or final["pretrained_model_scan_authorized"]
    )
    checks["final_auto_continue"] = final["auto_continue"] is True

    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "phase": 1156,
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
