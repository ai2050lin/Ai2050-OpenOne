#!/usr/bin/env python3
"""Independent artifact recomputation for Phase1157."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1157_causal_hyperedge_calibration"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1157_causal_hyperedge_calibration.py"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1156_causal_use_quotient_calibration"


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
    correct = [predicted[offset] == truth[index]["morphology"] for offset, index in enumerate(indices)]
    labels = sorted({truth[index]["morphology"] for index in indices})
    per_label = {
        label: float(
            np.mean(
                [correct[offset] for offset, index in enumerate(indices) if truth[index]["morphology"] == label]
            )
        )
        for label in labels
    }
    per_gauge = {
        gauge: float(np.mean([correct[offset] for offset, index in enumerate(indices) if truth[index]["gauge"] == gauge]))
        for gauge in gauges
    }
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "per_gauge_accuracy": per_gauge,
        "gauge_accuracy_gap": float(max(per_gauge.values()) - min(per_gauge.values())),
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
        raise RuntimeError("Phase1157 artifacts are incomplete")

    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    protocol_digest = body.pop("protocol_digest")
    checks["protocol_digest"] = digest(body) == protocol_digest
    checks["main_script_hash"] = sha256_file(MAIN_SCRIPT) == protocol["script_sha256"]
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    checks["protocol_audit"] = bool(protocol_audit["all_checks_passed"] and protocol_audit["protocol_digest"] == protocol_digest)
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks["source_final_link"] = source_final["final_digest"] == protocol["source_phase1156_digest"]
    checks["source_audit_link"] = source_audit["audit_digest"] == protocol["source_phase1156_audit_digest"]
    checks["source_confirmed"] = bool(source_final["matched_transport_causal_use_camera_confirmed"] and source_audit["all_checks_passed"])
    checks["candidate_frozen"] = protocol["candidate"] == "matched_hyperedge_tomography"
    checks["five_morphologies"] = len(protocol["morphologies"]) == 5

    gauges = protocol["gauges"]
    algorithms = protocol["algorithms"]
    split_data: dict[str, dict[str, Any]] = {}
    for split in ("discovery", "confirmation"):
        root = OUT_ROOT / "runs" / split
        summary = read_json(root / "summary.json")
        truth = read_jsonl(root / "sealed_truth.jsonl")
        public = read_jsonl(root / "public_manifest.jsonl")
        diagnostics = read_jsonl(root / "diagnostics.jsonl")
        representations = read_jsonl(root / "representation_manifest.jsonl")
        with np.load(root / "feature_pack.npz") as pack:
            arrays = {name: np.asarray(pack[name]) for name in pack.files}
        split_data[split] = {
            "summary": summary,
            "truth": truth,
            "public": public,
            "diagnostics": diagnostics,
            "representations": representations,
            "arrays": arrays,
        }
        checks[f"{split}::counts"] = len(truth) == len(public) == len(diagnostics) == 80 and len(representations) == 4
        checks[f"{split}::indices"] = all(row["index"] == index for index, row in enumerate(truth))
        checks[f"{split}::id_alignment"] = (
            [row["unit_id"] for row in truth] == [row["unit_id"] for row in public] == [row["unit_id"] for row in diagnostics]
        )
        for gauge in gauges:
            checks[f"{split}::gauge::{gauge}"] = sum(row["gauge"] == gauge for row in truth) == 20
        for morphology in protocol["morphologies"]:
            checks[f"{split}::morphology::{morphology}"] = sum(row["morphology"] == morphology for row in truth) == 16
        for algorithm in algorithms:
            checks[f"{split}::feature::{algorithm}"] = algorithm in arrays and arrays[algorithm].shape[0] == 80
            checks[f"{split}::finite::{algorithm}"] = bool(np.isfinite(arrays[algorithm]).all())
        checks[f"{split}::feature_hash"] = sha256_file(root / "feature_pack.npz") == summary["feature_pack_sha256"]
        checks[f"{split}::public_hash"] = sha256_file(root / "public_manifest.jsonl") == summary["public_manifest_sha256"]
        checks[f"{split}::truth_hash"] = sha256_file(root / "sealed_truth.jsonl") == summary["sealed_truth_sha256"]
        checks[f"{split}::diagnostics_hash"] = sha256_file(root / "diagnostics.jsonl") == summary["diagnostics_sha256"]
        checks[f"{split}::representation_hash"] = (
            sha256_file(root / "representation_manifest.jsonl") == summary["representation_manifest_sha256"]
        )
        checks[f"{split}::summary_digest"] = (
            digest({key: value for key, value in summary.items() if key != "summary_digest"}) == summary["summary_digest"]
        )
        checks[f"{split}::run_gate"] = bool(summary["run_gate_passed"] and all(summary["checks"].values()))
        for row in representations:
            path = root / "representations" / f"{row['representation_id']}.npz"
            checks[f"{split}::representation_file::{row['representation_id']}"] = (
                path.exists() and sha256_file(path) == row["representation_sha256"]
            )

        by_key = {(row["morphology"], int(row["replicate"]), row["gauge"]): int(row["index"]) for row in truth}
        representation_matches = []
        dependency_matches = []
        candidate_errors = []
        for replicate in range(protocol["replicates"]):
            for gauge in gauges:
                base = by_key[("bypass_none", replicate, gauge)]
                for morphology in protocol["morphologies"]:
                    current = by_key[(morphology, replicate, gauge)]
                    representation_matches.append(
                        bool(
                            np.array_equal(arrays["state_gram"][base], arrays["state_gram"][current])
                            and np.array_equal(
                                arrays["dependency_rank_quotient"][base], arrays["dependency_rank_quotient"][current]
                            )
                        )
                    )
            for morphology in protocol["morphologies"]:
                identity = by_key[(morphology, replicate, "identity")]
                for gauge in gauges[1:]:
                    current = by_key[(morphology, replicate, gauge)]
                    dependency_matches.append(
                        bool(np.array_equal(arrays["dependency_rank_quotient"][identity], arrays["dependency_rank_quotient"][current]))
                    )
                    candidate_errors.append(
                        float(np.max(np.abs(arrays[protocol["candidate"]][identity] - arrays[protocol["candidate"]][current])))
                    )
        checks[f"{split}::representation_match"] = close(
            np.mean(representation_matches), summary["representation_morphology_match_fraction"]
        )
        checks[f"{split}::dependency_match"] = close(
            np.mean(dependency_matches), summary["dependency_gauge_match_fraction"]
        )
        checks[f"{split}::candidate_error"] = close(max(candidate_errors), summary["candidate_gauge_abs_error_max"])

        synergy = []
        non_synergy = []
        for index, row in enumerate(truth):
            feature = arrays[protocol["candidate"]][index]
            feature_gap = float(feature[3])
            diagnostic = diagnostics[index]["causal_diagnostics"]
            checks[f"{split}::feature_diag::{index}"] = close(feature_gap, diagnostic["synergy_gap"])
            if row["morphology"] == "synergy_row_col":
                synergy.append(feature_gap)
                checks[f"{split}::synergy_truth::{index}"] = feature_gap > 0.9999 and feature[0] < 1e-6 and feature[1] < 1e-6
            else:
                non_synergy.append(abs(feature_gap))
                checks[f"{split}::nonsynergy_truth::{index}"] = abs(feature_gap) < 1e-6
        checks[f"{split}::synergy_recompute"] = close(min(synergy), summary["synergy_gap_min"])
        checks[f"{split}::nonsynergy_recompute"] = close(max(non_synergy), summary["non_synergy_gap_abs_max"])
        checks[f"{split}::clean_accuracy"] = close(
            min(row["clean_accuracy"] for row in diagnostics), summary["clean_accuracy_min"]
        )
        checks[f"{split}::clean_probability"] = close(
            min(row["clean_target_probability_min"] for row in diagnostics), summary["clean_target_probability_min"]
        )
        checks[f"{split}::clean_error"] = close(
            max(row["clean_function_abs_error_max"] for row in diagnostics), summary["clean_function_abs_error_max"]
        )
        checks[f"{split}::null_tv"] = close(
            max(row["causal_diagnostics"]["matched_null_tv"] for row in diagnostics), summary["matched_null_tv_max"]
        )

    checks["split_ids_disjoint"] = not bool(
        {row["unit_id"] for row in split_data["discovery"]["public"]}
        & {row["unit_id"] for row in split_data["confirmation"]["public"]}
    )
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    labels = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as pack:
        prototypes = {name: np.asarray(pack[name]) for name in pack.files}
    checks["fit_digest"] = digest({key: value for key, value in fit.items() if key != "fit_digest"}) == fit["fit_digest"]
    checks["fit_hash"] = sha256_file(OUT_ROOT / "analysis/frozen_prototypes.npz") == fit["prototype_sha256"]
    checks["label_hash"] = sha256_file(OUT_ROOT / "analysis/prototype_labels.json") == fit["labels_sha256"]
    checks["fit_authorized"] = bool(fit["candidate_qualified"] and fit["confirmation_run_authorized"])
    dtruth = split_data["discovery"]["truth"]
    darrays = split_data["discovery"]["arrays"]
    fit_indices = [
        index for index, row in enumerate(dtruth) if row["replicate"] in protocol["fit_replicates"] and row["gauge"] == "identity"
    ]
    validation_indices = [index for index, row in enumerate(dtruth) if row["replicate"] in protocol["validation_replicates"]]
    checks["fit_count"] = fit["fit_count"] == len(fit_indices) == 15
    checks["validation_count"] = fit["validation_count"] == len(validation_indices) == 20
    for algorithm in algorithms:
        predicted = predict(darrays[algorithm][validation_indices], labels[algorithm]["labels"], prototypes[algorithm])
        checks[f"fit_metric::{algorithm}"] = canonical(metrics(predicted, dtruth, validation_indices, gauges)) == canonical(
            fit["algorithm_metrics"][algorithm]
        )

    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    predictions = read_jsonl(OUT_ROOT / "predictions/confirmation_predictions.jsonl")
    checks["prediction_blind"] = manifest["confirmation_truth_read"] is False
    checks["prediction_hash"] = sha256_file(OUT_ROOT / "predictions/confirmation_predictions.jsonl") == manifest["prediction_sha256"]
    checks["prediction_count"] = len(predictions) == manifest["prediction_count"] == 80
    checks["prediction_alignment"] = [row["unit_id"] for row in predictions] == [
        row["unit_id"] for row in split_data["confirmation"]["public"]
    ]
    carrays = split_data["confirmation"]["arrays"]
    for algorithm in algorithms:
        expected = predict(carrays[algorithm], labels[algorithm]["labels"], prototypes[algorithm])
        actual = [row["algorithms"][algorithm]["prediction"] for row in predictions]
        checks[f"prediction::{algorithm}"] = expected == actual

    score = read_json(OUT_ROOT / "analysis/score.json")
    ctruth = split_data["confirmation"]["truth"]
    indices = list(range(len(ctruth)))
    checks["score_digest"] = digest({key: value for key, value in score.items() if key != "score_digest"}) == score["score_digest"]
    checks["score_confirmed"] = bool(score["candidate_confirmed"] and all(score["candidate_checks"].values()))
    for algorithm in algorithms:
        actual = [row["algorithms"][algorithm]["prediction"] for row in predictions]
        checks[f"score_metric::{algorithm}"] = canonical(metrics(actual, ctruth, indices, gauges)) == canonical(
            score["algorithm_metrics"][algorithm]
        )

    final = read_json(OUT_ROOT / "analysis/final.json")
    checks["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"}) == final["final_digest"]
    checks["final_confirmed"] = bool(final["matched_causal_hyperedge_camera_confirmed"])
    checks["final_scope"] = not bool(
        final["redundancy_identification_confirmed"]
        or final["gate_external_validity_confirmed"]
        or final["learned_network_external_validity_confirmed"]
        or final["free_transformer_scan_authorized"]
        or final["pretrained_model_scan_authorized"]
    )
    checks["final_auto_continue"] = final["auto_continue"] is True

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "phase": 1157,
        "audit_kind": "independent_artifact_recomputation",
        "check_count": len(checks),
        "passed_count": int(sum(checks.values())),
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
