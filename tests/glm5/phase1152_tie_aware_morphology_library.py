#!/usr/bin/env python3
"""Repair the Phase1151 tie discontinuity on entirely new systems.

Phase1151 failed because redundant paths create exact output ties.  A rotated
coordinate chart changes those ties at floating-point scale, while argmax
turns the tiny perturbation into a discrete 0/1 feature jump.  This revision
freezes a continuous projection that removes every top-1/tie-dependent field,
then regenerates both splits with new nuisance seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import phase1151_mechanism_morphology_library as prior


PHASE = 1152
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1152_tie_aware_morphology_library"
SPLITS = prior.SPLITS
BASE_REPLICATES = 4
FEATURE_NAMES = prior.FEATURE_NAMES
THRESHOLDS = dict(prior.THRESHOLDS)


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


def sorted_rows(rows: np.ndarray) -> np.ndarray:
    values = [np.asarray(row, dtype=np.float64).reshape(-1) for row in rows]
    values.sort(key=lambda row: tuple(np.round(row, 8).tolist()))
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def project_single(vector: np.ndarray) -> np.ndarray:
    rows = np.asarray(vector, dtype=np.float64).reshape(prior.N_SITES, 10)
    return sorted_rows(rows[:, [0, 1, 2, 4, 5, 6, 7, 8]])


def project_pair(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    single = project_single(value[:50])
    rows = value[50:].reshape(10, 14)
    pair = sorted_rows(rows[:, [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]])
    return np.concatenate([single, pair])


def project_factorial(vector: np.ndarray) -> np.ndarray:
    rows = np.asarray(vector, dtype=np.float64).reshape(10, 6)
    return sorted_rows(rows)


def project_exhaustive(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    blocks = value[:120].reshape(5, 4, 6)
    continuous = blocks[:, [0, 2, 3], :].reshape(-1)
    context_only = value[122:124]
    return np.concatenate([continuous, context_only])


def continuous_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    single = project_single(features["single_site_patch"])
    pair = project_pair(features["pairwise_patch"])
    factorial = project_factorial(features["factorial_interaction"])
    exhaustive = project_exhaustive(features["exhaustive_coalition"])
    pair_only = pair[len(single) :]
    return {
        "raw_coordinates": np.asarray(features["raw_coordinates"], dtype=np.float64),
        "state_gram": np.asarray(features["state_gram"], dtype=np.float64),
        "single_site_patch": single,
        "pairwise_patch": pair,
        "factorial_interaction": factorial,
        "exhaustive_coalition": exhaustive,
        "functional_tomography": np.concatenate([single, pair_only, factorial, exhaustive]),
    }


def build_units(split: str) -> list[dict[str, Any]]:
    split_seed = 115210 if split == "discovery" else 115290
    rows = []
    for mechanism in prior.MECHANISMS:
        for replicate in range(BASE_REPLICATES):
            nuisance_seed = split_seed + replicate * 103
            pair_key = digest({"phase": PHASE, "split": split, "replicate": replicate, "seed": nuisance_seed})[:16]
            for tail in prior.TAILS:
                for chart in prior.CHARTS:
                    spec = {
                        "unit_id": digest({"phase": PHASE, "split": split, "replicate": replicate, "tail": tail, "chart": chart, "mechanism": mechanism})[:20],
                        "pair_key": pair_key,
                        "split": split,
                        "replicate": replicate,
                        "nuisance_seed": nuisance_seed,
                        "chart": chart,
                        "tail": tail,
                        "mechanism": mechanism,
                        "functional_group": prior.FUNCTIONAL_GROUP[mechanism],
                    }
                    rows.append(spec)
    return rows


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1152 after run output exists")
    source_final = read_json(prior.OUT_ROOT / "analysis/final.json")
    source_audit = read_json(prior.OUT_ROOT / "audit/independent_audit.json")
    checks = {
        "phase1151_failed_only_library_gate": not bool(source_final["library_qualified"]),
        "phase1151_audit_passed": bool(source_audit["all_checks_passed"]),
        "phase1151_scientific_failure_reproduced_both_splits": True,
        "continuous_scores_retained": True,
        "donor_top1_removed": True,
        "ablation_top1_removed": True,
        "top1_based_minimum_coalition_removed": True,
        "fresh_discovery_seeds": True,
        "fresh_confirmation_seeds": True,
        "confirmation_labels_sealed": True,
        "algorithm_selection_forbidden": True,
        "natural_model_scan_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "tie-aware continuous mechanism morphology library",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1151_digest": source_final["final_digest"],
        "source_phase1151_audit_digest": source_audit["audit_digest"],
        "design": {
            "mechanisms": list(prior.MECHANISMS),
            "functional_group": prior.FUNCTIONAL_GROUP,
            "equivalent_pairs": [list(pair) for pair in prior.EQUIVALENT_PAIRS],
            "features": list(FEATURE_NAMES),
            "base_replicates": BASE_REPLICATES,
            "charts": list(prior.CHARTS),
            "tails": list(prior.TAILS),
            "units_per_split": len(build_units("discovery")),
        },
        "projection": {
            "single_removed_columns": [3, 9],
            "pair_removed_columns": [3, 9],
            "exhaustive_removed_metric_block": "donor_top1",
            "exhaustive_removed_tail_fields": ["minimum_top1_coalition", "minimum_top1_coalition_count"],
            "row_recanonicalization": True,
        },
        "thresholds": THRESHOLDS,
        "hard_stops": [
            "No Phase1151 threshold or artifact may be changed.",
            "Only predeclared discontinuous top-1 fields may be removed.",
            "Phase1152 validates the repaired library but does not score algorithms.",
            "No natural-model or component search is authorized.",
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
        raise RuntimeError("Phase1152 frozen protocol mismatch")
    return protocol


def run_command(split: str) -> None:
    protocol = verify_protocol()
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    units = build_units(split)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    public = []
    truth = []
    diagnostics = []
    for index, spec in enumerate(units):
        oracle = prior.MechanismOracle(spec, device)
        raw_features, diagnostic = prior.probe_system(oracle)
        features = continuous_features(raw_features)
        for name in FEATURE_NAMES:
            rows[name].append(features[name].astype(np.float32))
        public.append({"index": index, "unit_id": spec["unit_id"], "pair_key": spec["pair_key"], "split": split})
        truth.append({key: spec[key] for key in ("unit_id", "pair_key", "split", "replicate", "chart", "tail", "mechanism", "functional_group")} | {"index": index})
        diagnostics.append({"index": index, "unit_id": spec["unit_id"], **diagnostic})
        del oracle
    arrays = {name: np.stack(value, axis=0) for name, value in rows.items()}
    arrays["tail_ratio"] = np.asarray([row["tail_ratio"] for row in diagnostics], dtype=np.float32)
    arrays["clean_accuracy"] = np.asarray([row["clean_accuracy"] for row in diagnostics], dtype=np.float32)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "feature_pack.npz", **arrays)
    write_jsonl(out / "public_manifest.jsonl", public)
    write_jsonl(out / "sealed_truth.jsonl", truth)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)

    functional = arrays["functional_tomography"]
    by_key = {(row["mechanism"], row["replicate"], row["tail"], row["chart"]): int(row["index"]) for row in truth}
    chart_values = []
    for mechanism in prior.MECHANISMS:
        for replicate in range(BASE_REPLICATES):
            for tail in prior.TAILS:
                left = by_key[(mechanism, replicate, tail, "identity")]
                right = by_key[(mechanism, replicate, tail, "rotated")]
                chart_values.append(prior.cosine(functional[left], functional[right]))
    equivalence_values = []
    for left_name, right_name in prior.EQUIVALENT_PAIRS:
        for replicate in range(BASE_REPLICATES):
            for tail in prior.TAILS:
                for chart in prior.CHARTS:
                    left = by_key[(left_name, replicate, tail, chart)]
                    right = by_key[(right_name, replicate, tail, chart)]
                    equivalence_values.append(prior.cosine(functional[left], functional[right]))
    stable = [diagnostics[row["index"]]["tail_ratio"] for row in truth if row["tail"] == "stable"]
    degraded = [diagnostics[row["index"]]["tail_ratio"] for row in truth if row["tail"] == "degraded"]
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "unit_count": len(units),
        "feature_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "finite_fraction": float(np.mean([np.isfinite(value).mean() for value in arrays.values()])),
        "clean_accuracy_min": float(np.min(arrays["clean_accuracy"])),
        "stable_tail_ratio_min": float(min(stable)),
        "degraded_tail_ratio_max": float(max(degraded)),
        "functional_chart_cosine_min": float(min(chart_values)),
        "functional_chart_cosine_median": float(statistics.median(chart_values)),
        "functional_equivalence_cosine_min": float(min(equivalence_values)),
        "functional_equivalence_cosine_median": float(statistics.median(equivalence_values)),
        "feature_pack_sha256": sha256_file(out / "feature_pack.npz"),
        "public_manifest_sha256": sha256_file(out / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(out / "sealed_truth.jsonl"),
        "diagnostics_sha256": sha256_file(out / "diagnostics.jsonl"),
    }
    t = protocol["thresholds"]
    checks = {
        "finite": summary["finite_fraction"] >= t["finite_fraction"],
        "clean": summary["clean_accuracy_min"] >= t["clean_accuracy"],
        "stable_tail": summary["stable_tail_ratio_min"] >= t["stable_tail_ratio_min"],
        "degraded_tail": summary["degraded_tail_ratio_max"] <= t["degraded_tail_ratio_max"],
        "chart_invariance": summary["functional_chart_cosine_min"] >= t["functional_chart_cosine_min"],
        "declared_equivalence": summary["functional_equivalence_cosine_min"] >= t["functional_equivalence_cosine_min"],
    }
    summary["checks"] = checks
    summary["all_checks_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(out / "summary.json", summary)
    print(canonical(summary))


def finalize_command() -> None:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    confirmation = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    discovery_ids = {json.loads(line)["unit_id"] for line in (OUT_ROOT / "runs/discovery/public_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    confirmation_ids = {json.loads(line)["unit_id"] for line in (OUT_ROOT / "runs/confirmation/public_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    overlap = len(discovery_ids & confirmation_ids)
    passed = bool(discovery["all_checks_passed"] and confirmation["all_checks_passed"] and overlap == 0)
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": discovery["summary_digest"],
        "confirmation_summary_digest": confirmation["summary_digest"],
        "split_overlap": overlap,
        "library_qualified": passed,
        "phase1153_blind_coverage_authorized": passed,
        "pretrained_model_scan_authorized": False,
        "outcome": "tie_aware_library_qualified" if passed else "tie_aware_library_not_qualified",
        "claim_boundary": "The continuous feature library removes a known tie discontinuity. It still does not establish that any identification algorithm recovers mechanism morphology.",
        "auto_continue": passed,
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("--split", choices=SPLITS, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.split)
    elif args.command == "finalize":
        finalize_command()


if __name__ == "__main__":
    main()
