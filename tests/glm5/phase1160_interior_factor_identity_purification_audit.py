#!/usr/bin/env python3
"""Independent audit for Phase1160, including the discovery-abstention branch."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1160_interior_factor_identity_purification"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1160_interior_factor_identity_purification as phase  # noqa: E402


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


def valid_digest(value: dict[str, Any], field: str) -> bool:
    body = dict(value)
    stored = body.pop(field)
    return digest(body) == stored


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def add(checks: list[dict[str, Any]], name: str, passed: bool, value: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "value": value})


def audit_run(split: str, checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    root = OUT_ROOT / "runs" / split
    summary = read_json(root / "summary.json")
    public = read_jsonl(root / "public_manifest.jsonl")
    truth = read_jsonl(root / "sealed_truth.jsonl")
    training = read_jsonl(root / "training_metrics.jsonl")
    with np.load(root / "identity_pack.npz") as pack:
        matched = np.asarray(pack["matched"], dtype=np.float64)
        units = np.asarray(pack["units"], dtype=np.float64)
        norms = np.asarray(pack["norms"], dtype=np.float64)
    recomputed_units, recomputed_norms = phase.normalized_residuals(matched)
    add(checks, f"{split}.summary_digest", valid_digest(summary, "summary_digest"))
    for field, filename in (
        ("identity_pack_sha256", "identity_pack.npz"),
        ("public_manifest_sha256", "public_manifest.jsonl"),
        ("sealed_truth_sha256", "sealed_truth.jsonl"),
        ("training_metrics_sha256", "training_metrics.jsonl"),
    ):
        add(checks, f"{split}.{field}", summary[field] == sha256_file(root / filename))
    expected = len(phase.ARCHITECTURES) * phase.REPLICATES
    add(checks, f"{split}.counts", len(public) == len(truth) == len(training) == expected)
    add(checks, f"{split}.public_blind", all("architecture" not in row for row in public))
    add(checks, f"{split}.truth_architectures", set(row["architecture"] for row in truth) == set(phase.ARCHITECTURES))
    add(checks, f"{split}.ids", [row["model_id"] for row in public] == [row["model_id"] for row in truth] == [row["model_id"] for row in training])
    add(checks, f"{split}.qualified", all(row["qualified"] for row in training))
    add(checks, f"{split}.unit_recompute", bool(np.allclose(units, recomputed_units, atol=1e-8)))
    add(checks, f"{split}.norm_recompute", bool(np.allclose(norms, recomputed_norms, atol=1e-8)))
    add(checks, f"{split}.summary_norm", close(summary["residual_norm_min"], np.min(norms)))
    add(checks, f"{split}.finite", bool(np.isfinite(units).all()))
    for row in training:
        path = root / "models" / f"{row['model_id']}.pt"
        add(checks, f"{split}.model_hash.{row['model_id']}", path.exists() and sha256_file(path) == row["model_sha256"])
    return public, truth, units, norms


def main() -> None:
    checks: list[dict[str, Any]] = []
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    add(checks, "protocol.digest", valid_digest(protocol, "protocol_digest"))
    add(checks, "protocol.primary_hash", protocol["source_hashes"]["primary_script"] == sha256_file(Path(phase.SCRIPT)))
    add(checks, "protocol.audit_hash", protocol["source_hashes"]["audit_script"] == sha256_file(Path(__file__).resolve()))
    add(checks, "protocol.source_hash", protocol["source_hashes"]["phase1159_script"] == sha256_file(phase.SOURCE_SCRIPT))
    add(checks, "protocol.endpoints_excluded", bool(protocol["checks"]["endpoints_excluded"]))
    add(checks, "protocol.query_only", bool(protocol["checks"]["query_only"]))
    add(checks, "protocol.abstention", "abstain" in protocol["allowed_outputs"])

    discovery_public, _discovery_truth, discovery_units, discovery_norms = audit_run("discovery", checks)
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    add(checks, "fit.digest", valid_digest(fit, "fit_digest"))
    fit_indices = [index for index, row in enumerate(discovery_public) if row["analysis_partition"] == "fit"]
    validation_indices = [index for index, row in enumerate(discovery_public) if row["analysis_partition"] == "validation"]
    prototypes = phase.make_prototypes(discovery_units, fit_indices)
    evaluation = phase.evaluate_identity(discovery_units, prototypes, validation_indices)
    add(checks, "fit.indices", fit_indices == fit["fit_indices"] and validation_indices == fit["validation_indices"])
    add(checks, "fit.prototypes", bool(np.allclose(prototypes, fit["prototypes"], atol=1e-8)))
    add(checks, "fit.correct", evaluation["correct_count"] == fit["validation"]["correct_count"])
    add(checks, "fit.assignments", evaluation["identity_assignment_count"] == fit["validation"]["identity_assignment_count"])
    add(checks, "fit.margin", close(evaluation["assignment_margin_median"], fit["validation"]["assignment_margin_median"]))
    recomputed_checks = {
        "measurement_gate": bool(read_json(OUT_ROOT / "runs/discovery/summary.json")["behavior_and_measurement_gate_passed"]),
        "validation_correct": evaluation["correct_count"] >= phase.THRESHOLDS["discovery_validation_correct_min"],
        "identity_assignment_count": evaluation["identity_assignment_count"] >= phase.THRESHOLDS["discovery_identity_assignment_count_min"],
        "assignment_margin": evaluation["assignment_margin_median"] >= phase.THRESHOLDS["discovery_assignment_margin_median_min"],
        "validation_norm": float(np.min(discovery_norms[validation_indices])) >= phase.THRESHOLDS["residual_norm_min"],
    }
    decision = "interior_factor_identity_recoverable" if all(recomputed_checks.values()) else "abstain"
    add(checks, "fit.checks", recomputed_checks == fit["checks"])
    add(checks, "fit.decision", decision == fit["decision"], decision)

    if fit["confirmation_authorized"]:
        predictions = read_json(OUT_ROOT / "predictions/confirmation_predictions.json")
        add(checks, "predictions.digest", valid_digest(predictions, "prediction_digest"))
        add(checks, "predictions.sealed", bool(predictions["confirmation_run_absent_at_sealing"]))
        add(checks, "predictions.no_architecture_labels", predictions["architecture_labels_used"] is False)
        add(checks, "predictions.prototypes", bool(np.allclose(predictions["prototypes"], prototypes, atol=1e-8)))
        confirmation_public, confirmation_truth, confirmation_units, confirmation_norms = audit_run("confirmation", checks)
        score = read_json(OUT_ROOT / "analysis/score.json")
        add(checks, "score.digest", valid_digest(score, "score_digest"))
        confirmation_eval = phase.evaluate_identity(confirmation_units, prototypes, list(range(len(confirmation_units))))
        add(checks, "score.correct", confirmation_eval["correct_count"] == score["evaluation"]["correct_count"])
        add(checks, "score.assignment_count", confirmation_eval["identity_assignment_count"] == score["evaluation"]["identity_assignment_count"])
        add(checks, "score.margin", close(confirmation_eval["assignment_margin_median"], score["evaluation"]["assignment_margin_median"]))
        predicted = np.asarray(confirmation_eval["predicted"])
        per_arch = {}
        for architecture in phase.ARCHITECTURES:
            indices = [index for index, row in enumerate(confirmation_truth) if row["architecture"] == architecture]
            per_arch[architecture] = int(np.sum(predicted[indices] == np.arange(len(phase.FACTORS))[None, :]))
        add(checks, "score.per_architecture", per_arch == score["per_architecture_correct"])
        score_checks = {
            "measurement_gate": bool(read_json(OUT_ROOT / "runs/confirmation/summary.json")["behavior_and_measurement_gate_passed"]),
            "correct": confirmation_eval["correct_count"] >= phase.THRESHOLDS["confirmation_correct_min"],
            "per_architecture_correct": min(per_arch.values()) >= phase.THRESHOLDS["confirmation_per_architecture_correct_min"],
            "identity_assignment_count": confirmation_eval["identity_assignment_count"] >= phase.THRESHOLDS["confirmation_identity_assignment_count_min"],
            "assignment_margin": confirmation_eval["assignment_margin_median"] >= phase.THRESHOLDS["confirmation_assignment_margin_median_min"],
            "residual_norm": float(np.min(confirmation_norms)) >= phase.THRESHOLDS["residual_norm_min"],
        }
        passed = all(score_checks.values())
        add(checks, "score.checks", score_checks == score["checks"])
        add(checks, "score.decision", passed == score["interior_factor_identity_external_validity_passed"])
        add(checks, "branch.confirmation_files", len(confirmation_public) == 8)
    else:
        passed = False
        add(checks, "branch.no_predictions", not (OUT_ROOT / "predictions/confirmation_predictions.json").exists())
        add(checks, "branch.no_confirmation", not (OUT_ROOT / "runs/confirmation").exists())
        add(checks, "branch.no_score", not (OUT_ROOT / "analysis/score.json").exists())

    final = read_json(OUT_ROOT / "analysis/final.json")
    add(checks, "final.digest", valid_digest(final, "final_digest"))
    add(checks, "final.decision", final["interior_factor_identity_external_validity_passed"] == passed)
    add(checks, "final.preserve_phase1159", bool(final["phase1159_narrow_causal_use_result_preserved"]))
    add(checks, "final.no_identity_upgrade", not bool(final["phase1159_interpretation_upgraded_to_factor_identity"]))
    add(checks, "final.no_pretrained", not bool(final["pretrained_model_scan_authorized"]))
    add(checks, "final.no_auto_continue", not bool(final["auto_continue"]))
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
        raise RuntimeError([row for row in checks if not row["passed"]])


if __name__ == "__main__":
    main()
