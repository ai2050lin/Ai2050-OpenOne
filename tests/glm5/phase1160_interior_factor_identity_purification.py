#!/usr/bin/env python3
"""Independent interior factor-identity purification after Phase1159."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1160_interior_factor_identity_purification_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1160_interior_factor_identity_purification"
SOURCE_SCRIPT = ROOT / "tests/glm5/phase1159_free_transformer_causal_use_external_validity.py"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1159_free_transformer_causal_use_external_validity as source  # noqa: E402


PHASE = 1160
FACTORS = source.FACTORS
ARCHITECTURES = source.ARCHITECTURES
REPLICATES = source.REPLICATES
FIT_REPLICATES = source.FIT_REPLICATES
VALIDATION_REPLICATES = source.VALIDATION_REPLICATES
QUERY_INTERIOR_INDICES = tuple(
    site["index"]
    for site in source.common_sites()
    if site["role"] == "query" and float(site["depth"]) not in (0.0, 1.0)
)
PERMUTATIONS = tuple(itertools.permutations(range(len(FACTORS))))
IDENTITY_PERMUTATION = tuple(range(len(FACTORS)))
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "residual_norm_min": 0.005,
    "discovery_validation_correct_min": 6,
    "discovery_validation_total": 6,
    "discovery_identity_assignment_count_min": 2,
    "discovery_validation_model_count": 2,
    "discovery_assignment_margin_median_min": 0.05,
    "confirmation_correct_min": 22,
    "confirmation_total": 24,
    "confirmation_per_architecture_correct_min": 11,
    "confirmation_identity_assignment_count_min": 7,
    "confirmation_model_count": 8,
    "confirmation_assignment_margin_median_min": 0.05,
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_seed(split: str, architecture: str, replicate: int) -> int:
    base = 1160100 if split == "discovery" else 1160900
    return base + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def normalized_residuals(matched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(matched, dtype=np.float64)[:, :, QUERY_INTERIOR_INDICES]
    residual = query - np.mean(query, axis=1, keepdims=True)
    norms = np.linalg.norm(residual, axis=2)
    units = residual / np.maximum(norms[:, :, None], 1e-12)
    return units, norms


def make_prototypes(units: np.ndarray, indices: list[int]) -> np.ndarray:
    prototypes = np.mean(units[indices], axis=0)
    return prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)


def evaluate_identity(units: np.ndarray, prototypes: np.ndarray, indices: list[int]) -> dict[str, Any]:
    similarities = np.einsum("mfd,gd->mfg", units[indices], prototypes)
    predicted = np.argmax(similarities, axis=2)
    correct = predicted == np.arange(len(FACTORS))[None, :]
    assignment_margins = []
    identity_best = []
    assignment_scores = []
    for matrix in similarities:
        scores = {
            "-".join(map(str, permutation)): float(
                np.mean([matrix[index, permutation[index]] for index in range(len(FACTORS))])
            )
            for permutation in PERMUTATIONS
        }
        identity_key = "-".join(map(str, IDENTITY_PERMUTATION))
        identity_score = scores[identity_key]
        alternatives = [value for key, value in scores.items() if key != identity_key]
        assignment_margins.append(identity_score - max(alternatives))
        identity_best.append(identity_score > max(alternatives))
        assignment_scores.append(scores)
    return {
        "similarities": similarities.tolist(),
        "predicted": predicted.tolist(),
        "correct_count": int(np.sum(correct)),
        "total_count": int(correct.size),
        "identity_assignment_count": int(sum(identity_best)),
        "model_count": len(indices),
        "assignment_margins": assignment_margins,
        "assignment_margin_median": float(np.median(assignment_margins)),
        "assignment_scores": assignment_scores,
    }


def source_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = source.OUT_ROOT
    return (
        read_json(root / "analysis/final.json"),
        read_json(root / "audit/independent_audit_corrigendum.json"),
        read_json(root / "analysis/posthoc_factor_specificity.json"),
    )


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1160 artifacts")
    prior, corrigendum, diagnostic = source_artifacts()
    checks = {
        "phase1159_narrow_external_validity_passed": bool(prior["free_transformer_causal_use_external_validity_passed"]),
        "phase1159_not_full_recovery": not bool(prior["full_blind_mechanism_recovery_complete"]),
        "phase1159_next_protocol_authorized": bool(prior["phase1160_graph_recovery_protocol_authorized"]),
        "phase1159_corrigendum_passed": bool(corrigendum["all_checks_passed"]),
        "posthoc_evidence_upgrade_forbidden": bool(diagnostic["evidence_upgrade_forbidden"]),
        "query_only": all(source.common_sites()[index]["role"] == "query" for index in QUERY_INTERIOR_INDICES),
        "endpoints_excluded": all(float(source.common_sites()[index]["depth"]) not in (0.0, 1.0) for index in QUERY_INTERIOR_INDICES),
        "three_interior_depths": len(QUERY_INTERIOR_INDICES) == 3,
        "new_discovery_confirmation_seeds": True,
        "mechanism_labels_absent": True,
        "abstention_required": True,
        "confirmation_prediction_precedes_training": True,
        "source_script_frozen": sha256_file(SOURCE_SCRIPT)
        == read_json(source.OUT_ROOT / "protocol/preregistration.json")["source_hashes"]["primary_script"],
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1160 protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": "interior query factor-identity purification",
        "source_phase1159_final_digest": prior["final_digest"],
        "source_phase1159_corrigendum_digest": corrigendum["corrigendum_digest"],
        "source_phase1159_posthoc_digest": diagnostic["diagnostic_digest"],
        "source_hashes": {
            "primary_script": sha256_file(SCRIPT),
            "audit_script": sha256_file(AUDIT_SCRIPT),
            "phase1159_script": sha256_file(SOURCE_SCRIPT),
        },
        "factors": list(FACTORS),
        "query_interior_indices": list(QUERY_INTERIOR_INDICES),
        "query_interior_sites": [source.common_sites()[index] for index in QUERY_INTERIOR_INDICES],
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "thresholds": THRESHOLDS,
        "residualization": "subtract each model's across-factor mean at each interior query depth, then L2 normalize",
        "primary_endpoint": "factor identity retrieval and identity-assignment advantage on held-out trained networks",
        "allowed_outputs": ["interior_factor_identity_recoverable", "abstain"],
        "hard_stops": [
            "Input embeddings, source-token sites, and final readout states are forbidden from the primary endpoint.",
            "Discovery failure produces abstention and denies confirmation training.",
            "Confirmation predictions must be sealed before confirmation training.",
            "No hyperedge, redundancy, gating, natural-language, or pretrained-model claim is authorized.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(checks.values()),
            "all_checks_passed": all(checks.values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    print(canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if digest(body) != stored:
        raise RuntimeError("Phase1160 protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1159_script", SOURCE_SCRIPT),
    ):
        if sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"Phase1160 frozen source changed: {key}")
    return protocol


def run_split_command(split: str) -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs" / split
    if root.exists():
        raise RuntimeError(f"refusing to overwrite {root}")
    if split == "confirmation" and not (OUT_ROOT / "predictions/confirmation_predictions.json").exists():
        raise RuntimeError("predictions must be sealed before confirmation training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public_rows = []
    truth_rows = []
    training_rows = []
    matched_rows = []
    control_rows = []
    model_root = root / "models"
    model_root.mkdir(parents=True, exist_ok=False)
    index = 0
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(split, architecture, replicate)
            lexicon_seed = seed + 7001
            lexicon = source.make_lexicon(lexicon_seed)
            model, metrics = source.train_model(config, seed, lexicon, device)
            if not metrics["qualified"]:
                raise RuntimeError(f"behavior gate failed for {split}/{architecture}/{replicate}")
            model_id = digest({"phase": PHASE, "split": split, "seed": seed})[:18]
            model_path = model_root / f"{model_id}.pt"
            torch.save(
                {
                    "phase": PHASE,
                    "split": split,
                    "seed": seed,
                    "lexicon_seed": lexicon_seed,
                    "lexicon": lexicon,
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                },
                model_path,
            )
            arrays, _diagnostics = source.scan_model(model, config, lexicon, split)
            matched_rows.append(arrays["matched_median"])
            control_rows.append(arrays["control_median"])
            partition = "fit" if replicate in FIT_REPLICATES else "validation"
            public_rows.append({"index": index, "model_id": model_id, "split": split, "analysis_partition": partition})
            truth_rows.append(
                {
                    "index": index,
                    "model_id": model_id,
                    "split": split,
                    "architecture": architecture,
                    "replicate": replicate,
                    "seed": seed,
                    "lexicon_seed": lexicon_seed,
                }
            )
            training_rows.append(
                {
                    "index": index,
                    "model_id": model_id,
                    "split": split,
                    "analysis_partition": partition,
                    "accuracy": metrics["accuracy"],
                    "minimum_probability": metrics["minimum_probability"],
                    "finite_fraction": metrics["finite_fraction"],
                    "steps": metrics["steps"],
                    "qualified": metrics["qualified"],
                    "parameter_count": metrics["parameter_count"],
                    "model_sha256": sha256_file(model_path),
                }
            )
            index += 1
            del model
            torch.cuda.empty_cache()
    matched = np.stack(matched_rows).astype(np.float32)
    control = np.stack(control_rows).astype(np.float32)
    units, norms = normalized_residuals(matched)
    np.savez_compressed(root / "identity_pack.npz", matched=matched, control=control, units=units, norms=norms)
    write_jsonl(root / "public_manifest.jsonl", public_rows)
    write_jsonl(root / "sealed_truth.jsonl", truth_rows)
    write_jsonl(root / "training_metrics.jsonl", training_rows)
    summary = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "model_count": len(public_rows),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "residual_norm_min": float(np.min(norms)),
        "residual_norm_median": float(np.median(norms)),
        "finite_fraction": float(np.mean(np.isfinite(units))),
        "identity_pack_sha256": sha256_file(root / "identity_pack.npz"),
        "public_manifest_sha256": sha256_file(root / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(root / "sealed_truth.jsonl"),
        "training_metrics_sha256": sha256_file(root / "training_metrics.jsonl"),
    }
    checks = {
        "behavior_accuracy": summary["behavior_accuracy_min"] >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": summary["behavior_min_probability_min"] >= THRESHOLDS["behavior_min_probability_min"],
        "residual_norm": summary["residual_norm_min"] >= THRESHOLDS["residual_norm_min"],
        "finite": summary["finite_fraction"] == 1.0,
        "public_blind": all("architecture" not in row for row in public_rows),
        "expected_count": len(public_rows) == len(ARCHITECTURES) * REPLICATES,
    }
    summary["checks"] = checks
    summary["behavior_and_measurement_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(root / "summary.json", summary)
    print(canonical(summary))


def fit_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/discovery"
    summary = read_json(root / "summary.json")
    public = read_jsonl(root / "public_manifest.jsonl")
    with np.load(root / "identity_pack.npz") as pack:
        units = np.asarray(pack["units"], dtype=np.float64)
        norms = np.asarray(pack["norms"], dtype=np.float64)
    fit_indices = [index for index, row in enumerate(public) if row["analysis_partition"] == "fit"]
    validation_indices = [index for index, row in enumerate(public) if row["analysis_partition"] == "validation"]
    prototypes = make_prototypes(units, fit_indices)
    evaluation = evaluate_identity(units, prototypes, validation_indices)
    checks = {
        "measurement_gate": bool(summary["behavior_and_measurement_gate_passed"]),
        "validation_correct": evaluation["correct_count"] >= THRESHOLDS["discovery_validation_correct_min"],
        "identity_assignment_count": evaluation["identity_assignment_count"] >= THRESHOLDS["discovery_identity_assignment_count_min"],
        "assignment_margin": evaluation["assignment_margin_median"] >= THRESHOLDS["discovery_assignment_margin_median_min"],
        "validation_norm": float(np.min(norms[validation_indices])) >= THRESHOLDS["residual_norm_min"],
    }
    decision = "interior_factor_identity_recoverable" if all(checks.values()) else "abstain"
    fit = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "prototypes": prototypes.tolist(),
        "validation": evaluation,
        "checks": checks,
        "decision": decision,
        "confirmation_authorized": decision != "abstain",
    }
    fit["fit_digest"] = digest(fit)
    write_json(OUT_ROOT / "analysis/fit.json", fit)
    print(canonical(fit))


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if not fit["confirmation_authorized"]:
        raise RuntimeError("discovery abstained; confirmation is forbidden")
    if (OUT_ROOT / "runs/confirmation").exists():
        raise RuntimeError("confirmation already exists")
    predictions = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "decision": fit["decision"],
        "prototypes": fit["prototypes"],
        "architecture_labels_used": False,
        "confirmation_run_absent_at_sealing": True,
    }
    predictions["prediction_digest"] = digest(predictions)
    write_json(OUT_ROOT / "predictions/confirmation_predictions.json", predictions)
    print(canonical(predictions))


def score_command() -> None:
    protocol = verify_protocol()
    predictions = read_json(OUT_ROOT / "predictions/confirmation_predictions.json")
    summary = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    truth = read_jsonl(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl")
    with np.load(OUT_ROOT / "runs/confirmation/identity_pack.npz") as pack:
        units = np.asarray(pack["units"], dtype=np.float64)
        norms = np.asarray(pack["norms"], dtype=np.float64)
    prototypes = np.asarray(predictions["prototypes"], dtype=np.float64)
    indices = list(range(len(units)))
    evaluation = evaluate_identity(units, prototypes, indices)
    predicted = np.asarray(evaluation["predicted"])
    per_architecture_correct = {}
    for architecture in ARCHITECTURES:
        model_indices = [index for index, row in enumerate(truth) if row["architecture"] == architecture]
        per_architecture_correct[architecture] = int(
            np.sum(predicted[model_indices] == np.arange(len(FACTORS))[None, :])
        )
    checks = {
        "measurement_gate": bool(summary["behavior_and_measurement_gate_passed"]),
        "correct": evaluation["correct_count"] >= THRESHOLDS["confirmation_correct_min"],
        "per_architecture_correct": min(per_architecture_correct.values())
        >= THRESHOLDS["confirmation_per_architecture_correct_min"],
        "identity_assignment_count": evaluation["identity_assignment_count"]
        >= THRESHOLDS["confirmation_identity_assignment_count_min"],
        "assignment_margin": evaluation["assignment_margin_median"]
        >= THRESHOLDS["confirmation_assignment_margin_median_min"],
        "residual_norm": float(np.min(norms)) >= THRESHOLDS["residual_norm_min"],
    }
    passed = all(checks.values())
    score = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": predictions["prediction_digest"],
        "evaluation": evaluation,
        "per_architecture_correct": per_architecture_correct,
        "residual_norm_min": float(np.min(norms)),
        "checks": checks,
        "interior_factor_identity_external_validity_passed": passed,
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/score.json", score)
    print(canonical(score))


def finalize_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if fit["confirmation_authorized"]:
        score = read_json(OUT_ROOT / "analysis/score.json")
        passed = bool(score["interior_factor_identity_external_validity_passed"])
        score_digest = score["score_digest"]
        branch = "confirmation_scored"
    else:
        passed = False
        score_digest = None
        branch = "discovery_abstention"
    final = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": "interior query factor-identity purification",
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "score_digest": score_digest,
        "branch": branch,
        "discovery_decision": fit["decision"],
        "interior_factor_identity_external_validity_passed": passed,
        "phase1159_narrow_causal_use_result_preserved": True,
        "phase1159_interpretation_upgraded_to_factor_identity": False,
        "full_mechanism_recovery_complete": False,
        "graph_hyperedge_scan_authorized": passed,
        "pretrained_model_scan_authorized": False,
        "auto_continue": False,
        "auto_continue_reason": (
            "A separate graph-recovery protocol is required."
            if passed
            else "The interior identity gate abstained or failed; adding graph complexity would amplify an unidentified shared shell."
        ),
        "new_puzzles": {
            "K118": "Endpoint-inclusive causal-use transfer and interior factor-identity transfer are distinct evidence gates.",
            "K119": "Across-factor centering removes the shared query-accumulation shell before identity claims are scored.",
        },
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol")
    run = subparsers.add_parser("run")
    run.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    subparsers.add_parser("fit")
    subparsers.add_parser("seal-predictions")
    subparsers.add_parser("score")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_split_command(args.split)
    elif args.command == "fit":
        fit_command()
    elif args.command == "seal-predictions":
        seal_predictions_command()
    elif args.command == "score":
        score_command()
    elif args.command == "finalize":
        finalize_command()
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
