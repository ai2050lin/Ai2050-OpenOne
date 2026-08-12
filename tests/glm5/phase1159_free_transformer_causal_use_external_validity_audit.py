#!/usr/bin/env python3
"""Independent artifact audit for Phase 1159."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1159_free_transformer_causal_use_external_validity"
PRIMARY = ROOT / "tests/glm5/phase1159_free_transformer_causal_use_external_validity.py"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1159_free_transformer_causal_use_external_validity as phase  # noqa: E402


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


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def top_sites(profile: np.ndarray, control: np.ndarray) -> list[int]:
    selective = np.asarray(profile) - np.abs(np.asarray(control))
    return [int(value) for value in np.argsort(-selective, kind="stable")[: phase.TOP_K]]


def append(checks: list[dict[str, Any]], name: str, passed: bool, value: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "value": value})


def verify_digest(value: dict[str, Any], field: str) -> bool:
    body = dict(value)
    stored = body.pop(field)
    return digest(body) == stored


def audit_split(split: str, checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    root = OUT_ROOT / "runs" / split
    summary = read_json(root / "summary.json")
    public = read_jsonl(root / "public_manifest.jsonl")
    truth = read_jsonl(root / "sealed_truth.jsonl")
    training = read_jsonl(root / "training_metrics.jsonl")
    diagnostics = read_jsonl(root / "diagnostics.jsonl")
    with np.load(root / "effect_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    append(checks, f"{split}.summary_digest", verify_digest(summary, "summary_digest"))
    for field, filename in (
        ("effect_pack_sha256", "effect_pack.npz"),
        ("public_manifest_sha256", "public_manifest.jsonl"),
        ("sealed_truth_sha256", "sealed_truth.jsonl"),
        ("training_metrics_sha256", "training_metrics.jsonl"),
        ("diagnostics_sha256", "diagnostics.jsonl"),
    ):
        append(checks, f"{split}.{field}", summary[field] == sha256_file(root / filename))
    expected_count = len(phase.ARCHITECTURES) * phase.REPLICATES
    append(checks, f"{split}.public_count", len(public) == expected_count, len(public))
    append(checks, f"{split}.truth_count", len(truth) == expected_count, len(truth))
    append(checks, f"{split}.training_count", len(training) == expected_count, len(training))
    append(checks, f"{split}.diagnostic_count", len(diagnostics) == expected_count, len(diagnostics))
    append(checks, f"{split}.public_blind", all("architecture" not in row for row in public))
    append(checks, f"{split}.truth_has_architecture", all(row.get("architecture") in phase.ARCHITECTURES for row in truth))
    append(checks, f"{split}.id_alignment", [row["model_id"] for row in public] == [row["model_id"] for row in truth] == [row["model_id"] for row in training])
    append(checks, f"{split}.all_models_qualified", all(row["qualified"] for row in training))
    append(checks, f"{split}.accuracy_recompute", close(summary["behavior_accuracy_min"], min(row["accuracy"] for row in training)))
    append(checks, f"{split}.probability_recompute", close(summary["behavior_min_probability_min"], min(row["minimum_probability"] for row in training)))
    for row in training:
        model_path = root / "models" / f"{row['model_id']}.pt"
        append(checks, f"{split}.model_hash.{row['model_id']}", model_path.exists() and sha256_file(model_path) == row["model_sha256"])
    expected_shape = (expected_count, len(phase.FACTORS), len(phase.common_sites()))
    for name in ("matched_median", "control_median", "matched_mean", "control_mean"):
        append(checks, f"{split}.shape.{name}", arrays[name].shape == expected_shape, list(arrays[name].shape))
        append(checks, f"{split}.finite.{name}", bool(np.isfinite(arrays[name]).all()))
    append(checks, f"{split}.summary_gate", bool(summary["behavior_and_scan_gate_passed"]))
    append(checks, f"{split}.null_recompute", close(summary["null_max_abs"], 0.0))
    denominator_min = min(
        float(row["factor"][factor]["denominator_min"])
        for row in diagnostics
        for factor in phase.FACTORS
    )
    append(checks, f"{split}.denominator_recompute", close(summary["denominator_min"], denominator_min))
    append(checks, f"{split}.positive_denominator", denominator_min > 1e-5, denominator_min)
    return public, truth, arrays


def main() -> None:
    checks: list[dict[str, Any]] = []
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    append(checks, "protocol.digest", verify_digest(protocol, "protocol_digest"))
    append(checks, "protocol.primary_hash", protocol["source_hashes"]["primary_script"] == sha256_file(PRIMARY))
    append(checks, "protocol.audit_hash", protocol["source_hashes"]["audit_script"] == sha256_file(Path(__file__).resolve()))
    append(checks, "protocol.audit_passed", bool(protocol_audit["all_checks_passed"]))
    append(checks, "protocol.no_mechanism_classes", bool(protocol["checks"]["mechanism_class_labels_absent"]))
    append(checks, "protocol.abstention", "abstain" in protocol["allowed_outputs"])
    append(checks, "protocol.pretrained_forbidden", any("pretrained" in value.lower() for value in protocol["hard_stops"]))

    discovery_public, _discovery_truth, discovery_arrays = audit_split("discovery", checks)
    confirmation_public, confirmation_truth, confirmation_arrays = audit_split("confirmation", checks)

    fit = read_json(OUT_ROOT / "analysis/fit.json")
    append(checks, "fit.digest", verify_digest(fit, "fit_digest"))
    fit_indices = [index for index, row in enumerate(discovery_public) if row["analysis_partition"] == "fit"]
    validation_indices = [index for index, row in enumerate(discovery_public) if row["analysis_partition"] == "validation"]
    predicted = np.median(discovery_arrays["matched_median"][fit_indices], axis=0)
    predicted_control = np.median(discovery_arrays["control_median"][fit_indices], axis=0)
    append(checks, "fit.indices", fit_indices == fit["fit_indices"] and validation_indices == fit["validation_indices"])
    append(checks, "fit.predicted_profile", bool(np.allclose(predicted, np.asarray(fit["predicted_profile"]), atol=1e-8)))
    append(checks, "fit.predicted_control", bool(np.allclose(predicted_control, np.asarray(fit["predicted_control"]), atol=1e-8)))
    recomputed_authorized = []
    for factor_index, factor in enumerate(phase.FACTORS):
        result = fit["factor_results"][factor]
        selected = top_sites(predicted[factor_index], predicted_control[factor_index])
        correlations = [
            correlation(predicted[factor_index], discovery_arrays["matched_median"][index, factor_index])
            for index in validation_indices
        ]
        top_effect = float(np.median(discovery_arrays["matched_median"][validation_indices, factor_index][:, selected]))
        top_control = float(np.median(discovery_arrays["control_median"][validation_indices, factor_index][:, selected]))
        fit_top = float(np.median(predicted[factor_index, selected]))
        local_checks = {
            "fit_top_effect": fit_top >= phase.THRESHOLDS["discovery_fit_top_effect_min"],
            "validation_profile_correlation": float(np.median(correlations)) >= phase.THRESHOLDS["discovery_validation_profile_correlation_min"],
            "validation_top_effect": top_effect >= phase.THRESHOLDS["discovery_validation_top_effect_min"],
            "validation_control_gap": top_effect - top_control >= phase.THRESHOLDS["discovery_validation_control_gap_min"],
        }
        decision = "recoverable_causal_use_profile" if all(local_checks.values()) else "abstain"
        if decision != "abstain":
            recomputed_authorized.append(factor)
        append(checks, f"fit.{factor}.sites", selected == result["top_site_indices"])
        append(checks, f"fit.{factor}.decision", decision == result["decision"], decision)
        append(checks, f"fit.{factor}.correlation", close(np.median(correlations), result["validation_profile_correlation_median"]))
        append(checks, f"fit.{factor}.top_effect", close(top_effect, result["validation_top_effect"]))
        append(checks, f"fit.{factor}.control_gap", close(top_effect - top_control, result["validation_control_gap"]))
    append(checks, "fit.authorized", recomputed_authorized == fit["authorized_factors"], recomputed_authorized)
    append(checks, "fit.confirmation_authorized", fit["confirmation_authorized"] == (len(recomputed_authorized) >= phase.THRESHOLDS["authorized_factor_count_min"]))

    predictions_path = OUT_ROOT / "predictions/confirmation_predictions.json"
    predictions = read_json(predictions_path)
    prediction_manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    append(checks, "predictions.digest", verify_digest(predictions, "prediction_digest"))
    append(checks, "predictions.file_hash", prediction_manifest["prediction_sha256"] == sha256_file(predictions_path))
    append(checks, "predictions.sealed_before_confirmation", bool(prediction_manifest["confirmation_run_absent_at_sealing"]))
    append(checks, "predictions.no_architecture_label", "architecture" not in canonical(predictions).lower())
    confirmation_summary = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    append(checks, "predictions.timestamp_order", predictions["created_at_utc"] <= confirmation_summary["created_at_utc"])
    append(checks, "predictions.fit_digest", predictions["fit_digest"] == fit["fit_digest"])
    for factor_index, factor in enumerate(phase.FACTORS):
        prediction = predictions["factors"][factor]
        append(checks, f"predictions.{factor}.decision", prediction["decision"] == fit["factor_results"][factor]["decision"])
        append(checks, f"predictions.{factor}.profile", bool(np.allclose(prediction["predicted_profile"], predicted[factor_index], atol=1e-8)))

    score = read_json(OUT_ROOT / "analysis/score.json")
    append(checks, "score.digest", verify_digest(score, "score_digest"))
    recomputed_confirmed = []
    for factor_index, factor in enumerate(phase.FACTORS):
        prediction = predictions["factors"][factor]
        result = score["factor_results"][factor]
        if prediction["decision"] == "abstain":
            append(checks, f"score.{factor}.abstain", result["decision"] == "abstain" and not result["confirmed"])
            continue
        predicted_profile = np.asarray(prediction["predicted_profile"], dtype=np.float64)
        selected = prediction["top_site_indices"]
        correlations = [
            correlation(predicted_profile, confirmation_arrays["matched_median"][index, factor_index])
            for index in range(len(confirmation_public))
        ]
        per_arch = {}
        for architecture in phase.ARCHITECTURES:
            indices = [index for index, row in enumerate(confirmation_truth) if row["architecture"] == architecture]
            per_arch[architecture] = float(np.median([correlations[index] for index in indices]))
        top_effect = float(np.median(confirmation_arrays["matched_median"][:, factor_index][:, selected]))
        top_control = float(np.median(confirmation_arrays["control_median"][:, factor_index][:, selected]))
        control_abs = float(np.median(np.abs(confirmation_arrays["control_median"][:, factor_index][:, selected])))
        local_checks = {
            "profile_correlation_median": float(np.median(correlations)) >= phase.THRESHOLDS["confirmation_profile_correlation_median_min"],
            "model_pass_count": sum(value >= phase.THRESHOLDS["confirmation_architecture_correlation_min"] for value in correlations) >= phase.THRESHOLDS["confirmation_model_pass_count_min"],
            "both_architectures": min(per_arch.values()) >= phase.THRESHOLDS["confirmation_architecture_correlation_min"],
            "top_effect": top_effect >= phase.THRESHOLDS["confirmation_top_effect_min"],
            "control_gap": top_effect - top_control >= phase.THRESHOLDS["confirmation_control_gap_min"],
            "control_abs": control_abs <= phase.THRESHOLDS["confirmation_control_abs_max"],
        }
        confirmed = all(local_checks.values())
        if confirmed:
            recomputed_confirmed.append(factor)
        append(checks, f"score.{factor}.correlations", bool(np.allclose(correlations, result["profile_correlations"], atol=1e-7)))
        append(checks, f"score.{factor}.per_arch", all(close(per_arch[key], result["per_architecture_correlation_median"][key]) for key in per_arch))
        append(checks, f"score.{factor}.top_effect", close(top_effect, result["top_effect"]))
        append(checks, f"score.{factor}.control_gap", close(top_effect - top_control, result["control_gap"]))
        append(checks, f"score.{factor}.decision", confirmed == result["confirmed"], confirmed)
    external = len(recomputed_confirmed) >= phase.THRESHOLDS["confirmed_factor_count_min"] and confirmation_summary["behavior_and_scan_gate_passed"]
    append(checks, "score.confirmed_factors", recomputed_confirmed == score["confirmed_factors"], recomputed_confirmed)
    append(checks, "score.external_validity", external == score["free_transformer_causal_use_external_validity_passed"], external)
    append(checks, "score.not_full_recovery", not bool(score["full_blind_mechanism_recovery_complete"]))

    final = read_json(OUT_ROOT / "analysis/final.json")
    append(checks, "final.digest", verify_digest(final, "final_digest"))
    append(checks, "final.score_link", final["score_digest"] == score["score_digest"])
    append(checks, "final.scope", final["free_transformer_causal_use_external_validity_passed"] == external)
    append(checks, "final.phase1160_authorization", final["phase1160_graph_recovery_protocol_authorized"] == external)
    append(checks, "final.pretrained_forbidden", not bool(final["pretrained_model_scan_authorized"]))
    append(checks, "final.no_auto_continue", not bool(final["auto_continue"]))
    append(checks, "final.not_full_recovery", not bool(final["full_blind_mechanism_recovery_complete"]))

    passed_count = sum(row["passed"] for row in checks)
    audit = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "passed_count": passed_count,
        "failed_count": len(checks) - passed_count,
        "all_checks_passed": passed_count == len(checks),
        "checks": checks,
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = digest(audit)
    path = OUT_ROOT / "audit/independent_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({key: audit[key] for key in ("check_count", "passed_count", "failed_count", "all_checks_passed", "audit_digest")}))
    if not audit["all_checks_passed"]:
        failed = [row for row in checks if not row["passed"]]
        raise RuntimeError(f"Phase1159 independent audit failed: {failed}")


if __name__ == "__main__":
    main()
