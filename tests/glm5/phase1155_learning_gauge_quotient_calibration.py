#!/usr/bin/env python3
"""Calibrate learned-mechanism cameras against explicit gauge transformations.

Phase1154 showed that a camera calibrated on hand-written equations did not
recover one architecture class after gradient learning.  This phase separates
three questions that were previously conflated:

1. Do whole-site functional interventions commute with a general invertible
   change of coordinates inside each site?
2. What happens when the gauge mixes the physical sites themselves, thereby
   changing the intervention algebra?
3. Can a deliberately coarser factor-dependency signature survive both kinds
   of gauge without claiming a full six-way causal morphology?

The primary candidate uses only ranks of controlled finite-difference spans.
For a flattened mediator state s(r,c,k), every difference matrix transforms as
D' = D T^T under an invertible gauge T, so its rank is invariant.  The candidate
is intentionally limited to three coarse dependency classes; exact six-way
architecture labels remain descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import phase1151_mechanism_morphology_library as library
import phase1152_tie_aware_morphology_library as tie_aware
import phase1153_blind_algorithm_coverage as coverage
import phase1154_learned_morphology_external_validity as learned


PHASE = 1155
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1155_learning_gauge_quotient_calibration"
SPLITS = ("discovery", "confirmation")
GROUPS = learned.GROUPS
GAUGES = ("identity", "site_gl", "cross_site_orthogonal", "cross_site_gl")
CROSS_SITE_GAUGES = ("cross_site_orthogonal", "cross_site_gl")
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
ALGORITHMS = ("state_gram", "functional_tomography", "dependency_rank_quotient")
CANDIDATE = "dependency_rank_quotient"
COARSE_LABEL = {
    "single_joint_carrier": "joint_context_invariant",
    "payload_with_gate": "joint_context_invariant",
    "factorized_roles": "factor_separable",
    "joint_coalition": "joint_context_invariant",
    "redundant_paths": "joint_context_invariant",
    "context_switched_paths": "context_augmented_joint",
}
RANK_RELATIVE_TOLERANCE = 1e-6
RANK_ABSOLUTE_TOLERANCE = 1e-8
GAUGE_CONDITION = 5.0
THRESHOLDS = {
    "model_accuracy_min": 1.0,
    "model_min_probability_min": 0.98,
    "finite_fraction_min": 1.0,
    "clean_output_abs_error_max": 1e-5,
    "site_gl_functional_cosine_min": 0.999999,
    "candidate_gauge_match_fraction_min": 1.0,
    "cross_site_physical_break_count_min": 1,
    "cross_site_physical_break_cosine_max": 0.99,
    "discovery_coarse_accuracy_min": 1.0,
    "discovery_min_coarse_accuracy_min": 1.0,
    "confirmation_coarse_accuracy_min": 1.0,
    "confirmation_min_coarse_accuracy_min": 1.0,
    "candidate_gauge_accuracy_gap_max": 0.0,
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


def seed_for(split: str, group: str, replicate: int) -> int:
    base = 115510 if split == "discovery" else 115590
    return base + GROUPS.index(group) * 1009 + int(replicate) * 107


def make_orthogonal(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def make_invertible(size: int, seed: int, condition: float = GAUGE_CONDITION) -> np.ndarray:
    left = make_orthogonal(size, seed)
    right = make_orthogonal(size, seed + 7919)
    singular = np.geomspace(condition ** -0.5, condition ** 0.5, size)
    return left @ np.diag(singular) @ right.T


def gauge_matrix(gauge: str, seed: int) -> np.ndarray:
    sites = library.N_SITES
    width = library.STATE_DIM
    total = sites * width
    if gauge == "identity":
        return np.eye(total, dtype=np.float64)
    if gauge == "site_gl":
        value = np.zeros((total, total), dtype=np.float64)
        for site in range(sites):
            start = site * width
            value[start : start + width, start : start + width] = make_invertible(width, seed + 1013 * site)
        return value
    if gauge == "cross_site_orthogonal":
        return np.kron(make_orthogonal(sites, seed), np.eye(width, dtype=np.float64))
    if gauge == "cross_site_gl":
        site_map = make_invertible(sites, seed)
        channel_map = make_orthogonal(width, seed + 1543)
        return np.kron(site_map, channel_map)
    raise ValueError(gauge)


class GaugeOracle:
    def __init__(self, model: learned.LearnedMechanism, gauge: str, nuisance_seed: int, device: torch.device) -> None:
        self.model = model
        self.gauge = gauge
        self.device = device
        matrix = gauge_matrix(gauge, nuisance_seed)
        self.matrix = torch.tensor(matrix, dtype=torch.float64, device=device)
        self.inverse = torch.linalg.inv(self.matrix)

    def _tensors(self, inputs: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([row for row, _col, _ctx in inputs], dtype=torch.long, device=self.device),
            torch.tensor([col for _row, col, _ctx in inputs], dtype=torch.long, device=self.device),
            torch.tensor([ctx for _row, _col, ctx in inputs], dtype=torch.long, device=self.device),
        )

    def states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        rows, cols, contexts = self._tensors(inputs)
        with torch.no_grad():
            logical = self.model.logical_states(rows, cols, contexts).to(torch.float64).reshape(len(inputs), -1)
        observed = logical @ self.matrix.T
        return observed.reshape(len(inputs), library.N_SITES, library.STATE_DIM)

    def decode(self, observed: torch.Tensor) -> torch.Tensor:
        flattened = observed.to(torch.float64).reshape(len(observed), -1)
        logical = flattened @ self.inverse.T
        return logical.reshape(len(observed), library.N_SITES, library.STATE_DIM).to(torch.float32)

    def output(self, observed: torch.Tensor, receivers: list[tuple[int, int, int]]) -> torch.Tensor:
        del receivers
        with torch.no_grad():
            logits = self.model.logits_from_states(self.decode(observed))
            return torch.softmax(logits, dim=1).to(torch.float64)


def factor_difference_matrices(states: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(states, dtype=np.float64).reshape(2, library.N_ROWS, library.N_COLS, -1)
    row = np.stack(
        [value[ctx, r, c] - value[ctx, 0, c] for ctx in range(2) for c in range(library.N_COLS) for r in range(1, library.N_ROWS)]
    )
    col = np.stack(
        [value[ctx, r, c] - value[ctx, r, 0] for ctx in range(2) for r in range(library.N_ROWS) for c in range(1, library.N_COLS)]
    )
    context = np.stack(
        [value[1, r, c] - value[0, r, c] for r in range(library.N_ROWS) for c in range(library.N_COLS)]
    )
    row_col = np.stack(
        [
            value[ctx, r, c] - value[ctx, r, 0] - value[ctx, 0, c] + value[ctx, 0, 0]
            for ctx in range(2)
            for r in range(1, library.N_ROWS)
            for c in range(1, library.N_COLS)
        ]
    )
    row_context = np.stack(
        [
            (value[1, r, c] - value[1, 0, c]) - (value[0, r, c] - value[0, 0, c])
            for c in range(library.N_COLS)
            for r in range(1, library.N_ROWS)
        ]
    )
    col_context = np.stack(
        [
            (value[1, r, c] - value[1, r, 0]) - (value[0, r, c] - value[0, r, 0])
            for r in range(library.N_ROWS)
            for c in range(1, library.N_COLS)
        ]
    )
    triple = np.stack(
        [
            (value[1, r, c] - value[1, r, 0] - value[1, 0, c] + value[1, 0, 0])
            - (value[0, r, c] - value[0, r, 0] - value[0, 0, c] + value[0, 0, 0])
            for r in range(1, library.N_ROWS)
            for c in range(1, library.N_COLS)
        ]
    )
    centered = value.reshape(-1, value.shape[-1]) - np.mean(value.reshape(-1, value.shape[-1]), axis=0, keepdims=True)
    return {
        "row": row,
        "col": col,
        "context": context,
        "row_col": row_col,
        "row_context": row_context,
        "col_context": col_context,
        "triple": triple,
        "centered": centered,
    }


def matrix_rank(matrix: np.ndarray, threshold: float) -> int:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return int(np.sum(singular > threshold))


def dependency_rank_feature(states: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    matrices = factor_difference_matrices(states.detach().cpu().numpy())
    first_order = np.concatenate([matrices["row"], matrices["col"], matrices["context"]], axis=0)
    singular = np.linalg.svd(first_order, compute_uv=False)
    scale = float(singular[0]) if len(singular) else 0.0
    threshold = max(RANK_ABSOLUTE_TOLERANCE, RANK_RELATIVE_TOLERANCE * scale)
    rank = {name: matrix_rank(matrix, threshold) for name, matrix in matrices.items()}
    rank["row_col_union"] = matrix_rank(np.concatenate([matrices["row"], matrices["col"]], axis=0), threshold)
    rank["row_context_union"] = matrix_rank(np.concatenate([matrices["row"], matrices["context"]], axis=0), threshold)
    rank["col_context_union"] = matrix_rank(np.concatenate([matrices["col"], matrices["context"]], axis=0), threshold)
    rank["all_first_order_union"] = matrix_rank(first_order, threshold)
    rank["row_col_intersection"] = rank["row"] + rank["col"] - rank["row_col_union"]
    rank["row_context_intersection"] = rank["row"] + rank["context"] - rank["row_context_union"]
    rank["col_context_intersection"] = rank["col"] + rank["context"] - rank["col_context_union"]
    names = (
        "row",
        "col",
        "context",
        "row_col",
        "row_context",
        "col_context",
        "triple",
        "centered",
        "row_col_union",
        "row_context_union",
        "col_context_union",
        "all_first_order_union",
        "row_col_intersection",
        "row_context_intersection",
        "col_context_intersection",
    )
    feature = np.asarray([rank[name] for name in names], dtype=np.float64)
    return feature, {"rank_threshold": threshold, "rank_scale": scale, "rank_names": list(names), "ranks": rank}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return library.cosine(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))


def classification_metrics(predicted: list[str], truth: list[dict[str, Any]], indices: list[int], label_key: str) -> dict[str, Any]:
    labels = sorted({str(truth[index][label_key]) for index in indices})
    correct = [predicted[offset] == str(truth[index][label_key]) for offset, index in enumerate(indices)]
    per_label = {}
    for label in labels:
        selected = [offset for offset, index in enumerate(indices) if str(truth[index][label_key]) == label]
        per_label[label] = float(np.mean([correct[offset] for offset in selected]))
    per_gauge = {}
    for gauge in GAUGES:
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


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to rewrite Phase1155 artifacts")
    source_final = read_json(learned.OUT_ROOT / "analysis/final.json")
    source_audit = read_json(learned.OUT_ROOT / "audit/independent_audit.json")
    checks = {
        "phase1154_failure_frozen": not bool(source_final["learned_morphology_external_validity_confirmed"]),
        "phase1154_confirmation_was_denied": not bool(source_final["phase1155_free_network_tomography_authorized"]),
        "phase1154_audit_passed": bool(source_audit["all_checks_passed"]),
        "new_object_is_gauge_calibration": True,
        "fresh_split_seeds": True,
        "fit_uses_identity_only": True,
        "candidate_predeclared": CANDIDATE == "dependency_rank_quotient",
        "coarse_claim_only": len(set(COARSE_LABEL.values())) == 3,
        "confirmation_truth_forbidden_in_predict": True,
        "pretrained_model_scan_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "learning gauge boundary and factor-dependency quotient calibration",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1154_digest": source_final["final_digest"],
        "source_phase1154_audit_digest": source_audit["audit_digest"],
        "groups": list(GROUPS),
        "coarse_labels": COARSE_LABEL,
        "gauges": list(GAUGES),
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "algorithms": list(ALGORITHMS),
        "candidate": CANDIDATE,
        "gauge_condition": GAUGE_CONDITION,
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "rank_absolute_tolerance": RANK_ABSOLUTE_TOLERANCE,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "three-way factor-dependency class under identity-to-general-linear gauge transfer",
        "secondary_endpoint": "six-way architecture label, descriptive and non-authorizing",
        "hard_stops": [
            "Site-wise GL and cross-site GL are separate hypotheses and may not be pooled.",
            "A cross-site gauge changes the physical-site intervention algebra; failure there is not a failure of the underlying function.",
            "The dependency-rank candidate may claim only three coarse factor-dependency classes, not six causal morphologies.",
            "Confirmation predictions must be sealed before confirmation truth is read.",
            "Passing this calibration does not authorize a free Transformer or pretrained-LLM scan.",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    body = dict(protocol)
    protocol["protocol_digest"] = digest(body)
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
    if digest(body) != stored or sha256_file(SCRIPT) != protocol["script_sha256"]:
        raise RuntimeError("Phase1155 frozen protocol mismatch")
    return protocol


def run_command(split: str) -> None:
    protocol = verify_protocol()
    if split == "confirmation":
        fit = read_json(OUT_ROOT / "analysis/fit.json")
        if not fit["confirmation_run_authorized"]:
            raise RuntimeError("confirmation run denied by discovery")
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in ALGORITHMS}
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    models_root = out / "models"
    models_root.mkdir(parents=True, exist_ok=False)
    index = 0
    complete = library.all_inputs()
    tensor_rows, tensor_cols, tensor_contexts, _targets = learned.input_tensors(device)
    for group in GROUPS:
        for replicate in range(REPLICATES):
            seed = seed_for(split, group, replicate)
            model, train_metrics = learned.train_model(group, seed, device)
            model_id = digest({"phase": PHASE, "split": split, "group": group, "replicate": replicate})[:18]
            model_path = models_root / f"{model_id}.pt"
            torch.save(
                {"group": group, "replicate": replicate, "seed": seed, "state_dict": model.state_dict(), "metrics": train_metrics},
                model_path,
            )
            training_rows.append(
                {
                    "model_id": model_id,
                    "group": group,
                    "replicate": replicate,
                    "seed": seed,
                    "model_sha256": sha256_file(model_path),
                    **train_metrics,
                }
            )
            with torch.no_grad():
                reference = torch.softmax(model(tensor_rows, tensor_cols, tensor_contexts), dim=1).to(torch.float64)
            nuisance_seed = seed + 17011
            for gauge in GAUGES:
                oracle = GaugeOracle(model, gauge, nuisance_seed + 313 * GAUGES.index(gauge), device)
                raw_features, _probe_diagnostics = library.probe_system(oracle)
                continuous = tie_aware.continuous_features(raw_features)
                observed = oracle.states(complete)
                rank_feature, rank_diagnostics = dependency_rank_feature(observed)
                gauge_output = oracle.output(observed, complete)
                output_error = float(torch.max(torch.abs(gauge_output - reference)).item())
                feature_rows["state_gram"].append(np.asarray(continuous["state_gram"], dtype=np.float32))
                feature_rows["functional_tomography"].append(np.asarray(continuous["functional_tomography"], dtype=np.float32))
                feature_rows[CANDIDATE].append(rank_feature.astype(np.float32))
                unit_id = digest({"phase": PHASE, "split": split, "model_id": model_id, "gauge": gauge})[:20]
                public.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "model_id": model_id,
                        "split": split,
                        "replicate": replicate,
                        "gauge": gauge,
                    }
                )
                truth.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "model_id": model_id,
                        "split": split,
                        "functional_group": group,
                        "coarse_group": COARSE_LABEL[group],
                        "replicate": replicate,
                        "gauge": gauge,
                    }
                )
                diagnostics.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "output_abs_error_max": output_error,
                        **rank_diagnostics,
                    }
                )
                index += 1
            del model
            torch.cuda.empty_cache()
    arrays = {name: np.stack(rows, axis=0) for name, rows in feature_rows.items()}
    np.savez_compressed(out / "feature_pack.npz", **arrays)
    write_jsonl(out / "public_manifest.jsonl", public)
    write_jsonl(out / "sealed_truth.jsonl", truth)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)
    write_jsonl(out / "training_metrics.jsonl", training_rows)

    by_key = {(row["functional_group"], int(row["replicate"]), row["gauge"]): int(row["index"]) for row in truth}
    site_gl_functional = []
    cross_site_functional = []
    site_gl_state_gram = []
    candidate_matches = []
    for group in GROUPS:
        for replicate in range(REPLICATES):
            base = by_key[(group, replicate, "identity")]
            for gauge in GAUGES[1:]:
                current = by_key[(group, replicate, gauge)]
                candidate_matches.append(bool(np.array_equal(arrays[CANDIDATE][base], arrays[CANDIDATE][current])))
            site_index = by_key[(group, replicate, "site_gl")]
            site_gl_functional.append(cosine(arrays["functional_tomography"][base], arrays["functional_tomography"][site_index]))
            site_gl_state_gram.append(cosine(arrays["state_gram"][base], arrays["state_gram"][site_index]))
            for gauge in CROSS_SITE_GAUGES:
                current = by_key[(group, replicate, gauge)]
                cross_site_functional.append(cosine(arrays["functional_tomography"][base], arrays["functional_tomography"][current]))
    t = protocol["thresholds"]
    cross_break_count = int(sum(value < t["cross_site_physical_break_cosine_max"] for value in cross_site_functional))
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "model_count": len(training_rows),
        "unit_count": len(public),
        "accuracy_min": float(min(row["accuracy"] for row in training_rows)),
        "min_probability_min": float(min(row["min_probability"] for row in training_rows)),
        "clean_output_abs_error_max": float(max(row["output_abs_error_max"] for row in diagnostics)),
        "candidate_gauge_match_fraction": float(np.mean(candidate_matches)),
        "site_gl_functional_cosine_min": float(min(site_gl_functional)),
        "site_gl_functional_cosine_median": float(np.median(site_gl_functional)),
        "site_gl_state_gram_cosine_median": float(np.median(site_gl_state_gram)),
        "cross_site_functional_cosine_min": float(min(cross_site_functional)),
        "cross_site_functional_cosine_median": float(np.median(cross_site_functional)),
        "cross_site_physical_break_count": cross_break_count,
        "cross_site_comparison_count": len(cross_site_functional),
        "finite_fraction": float(np.mean([np.isfinite(array).mean() for array in arrays.values()])),
        "feature_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "feature_pack_sha256": sha256_file(out / "feature_pack.npz"),
        "public_manifest_sha256": sha256_file(out / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(out / "sealed_truth.jsonl"),
        "diagnostics_sha256": sha256_file(out / "diagnostics.jsonl"),
        "training_metrics_sha256": sha256_file(out / "training_metrics.jsonl"),
    }
    checks = {
        "behavior_accuracy": summary["accuracy_min"] >= t["model_accuracy_min"],
        "behavior_probability": summary["min_probability_min"] >= t["model_min_probability_min"],
        "finite": summary["finite_fraction"] >= t["finite_fraction_min"],
        "clean_function_equivalence": summary["clean_output_abs_error_max"] <= t["clean_output_abs_error_max"],
        "site_gl_functional_commutation": summary["site_gl_functional_cosine_min"] >= t["site_gl_functional_cosine_min"],
        "candidate_full_gauge_invariance": summary["candidate_gauge_match_fraction"] >= t["candidate_gauge_match_fraction_min"],
        "cross_site_intervention_semantics_changed": summary["cross_site_physical_break_count"] >= t["cross_site_physical_break_count_min"],
    }
    summary["checks"] = checks
    summary["run_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(out / "summary.json", summary)
    print(canonical(summary))


def fit_command() -> None:
    protocol = verify_protocol()
    summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    if not summary["run_gate_passed"]:
        raise RuntimeError("discovery run gate failed")
    root = OUT_ROOT / "runs/discovery"
    truth = read_jsonl(root / "sealed_truth.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    fit_indices = [
        index
        for index, row in enumerate(truth)
        if int(row["replicate"]) in FIT_REPLICATES and row["gauge"] == "identity"
    ]
    validation_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in VALIDATION_REPLICATES]
    prototypes: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        coarse_labels, coarse_proto = coverage.build_prototypes(arrays[algorithm], truth, fit_indices, "coarse_group")
        exact_labels, exact_proto = coverage.build_prototypes(arrays[algorithm], truth, fit_indices, "functional_group")
        prototypes[f"{algorithm}__coarse"] = coarse_proto.astype(np.float32)
        prototypes[f"{algorithm}__exact"] = exact_proto.astype(np.float32)
        metadata[algorithm] = {"coarse_labels": coarse_labels, "exact_labels": exact_labels}
        coarse_prediction, _ = coverage.predict(arrays[algorithm][validation_indices], coarse_labels, coarse_proto)
        exact_prediction, _ = coverage.predict(arrays[algorithm][validation_indices], exact_labels, exact_proto)
        metrics[algorithm] = {
            "coarse": classification_metrics(coarse_prediction, truth, validation_indices, "coarse_group"),
            "exact": classification_metrics(exact_prediction, truth, validation_indices, "functional_group"),
        }
    candidate = metrics[CANDIDATE]["coarse"]
    t = protocol["thresholds"]
    checks = {
        "coarse_accuracy": candidate["accuracy"] >= t["discovery_coarse_accuracy_min"],
        "min_coarse_accuracy": candidate["min_label_accuracy"] >= t["discovery_min_coarse_accuracy_min"],
        "gauge_accuracy_gap": candidate["gauge_accuracy_gap"] <= t["candidate_gauge_accuracy_gap_max"],
        "discovery_run_gate": bool(summary["run_gate_passed"]),
    }
    analysis = OUT_ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(analysis / "frozen_prototypes.npz", **prototypes)
    write_json(analysis / "prototype_labels.json", metadata)
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_count": len(fit_indices),
        "validation_count": len(validation_indices),
        "algorithm_metrics": metrics,
        "candidate_checks": checks,
        "candidate_qualified": all(checks.values()),
        "confirmation_run_authorized": all(checks.values()),
        "prototype_sha256": sha256_file(analysis / "frozen_prototypes.npz"),
        "labels_sha256": sha256_file(analysis / "prototype_labels.json"),
    }
    result["fit_digest"] = digest(result)
    write_json(analysis / "fit.json", result)
    print(canonical(result))


def predict_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if not fit["confirmation_run_authorized"]:
        raise RuntimeError("confirmation prediction denied")
    if (OUT_ROOT / "predictions").exists():
        raise RuntimeError("refusing to overwrite confirmation predictions")
    root = OUT_ROOT / "runs/confirmation"
    public = read_jsonl(root / "public_manifest.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    metadata = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as stored:
        prototypes = {name: np.asarray(stored[name]) for name in stored.files}
    rows = []
    for algorithm in ALGORITHMS:
        coarse, coarse_score = coverage.predict(
            arrays[algorithm], metadata[algorithm]["coarse_labels"], prototypes[f"{algorithm}__coarse"]
        )
        exact, exact_score = coverage.predict(
            arrays[algorithm], metadata[algorithm]["exact_labels"], prototypes[f"{algorithm}__exact"]
        )
        for index, public_row in enumerate(public):
            if algorithm == ALGORITHMS[0]:
                rows.append(
                    {
                        "index": index,
                        "unit_id": public_row["unit_id"],
                        "replicate": public_row["replicate"],
                        "gauge": public_row["gauge"],
                        "algorithms": {},
                    }
                )
            rows[index]["algorithms"][algorithm] = {
                "coarse": coarse[index],
                "coarse_cosine": float(coarse_score[index]),
                "exact": exact[index],
                "exact_cosine": float(exact_score[index]),
            }
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
    predictions = read_jsonl(OUT_ROOT / "predictions/confirmation_predictions.jsonl")
    truth = read_jsonl(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl")
    if len(predictions) != len(truth):
        raise RuntimeError("prediction/truth length mismatch")
    indices = list(range(len(truth)))
    metrics = {}
    for algorithm in ALGORITHMS:
        coarse = [row["algorithms"][algorithm]["coarse"] for row in predictions]
        exact = [row["algorithms"][algorithm]["exact"] for row in predictions]
        metrics[algorithm] = {
            "coarse": classification_metrics(coarse, truth, indices, "coarse_group"),
            "exact": classification_metrics(exact, truth, indices, "functional_group"),
        }
    candidate = metrics[CANDIDATE]["coarse"]
    t = protocol["thresholds"]
    checks = {
        "coarse_accuracy": candidate["accuracy"] >= t["confirmation_coarse_accuracy_min"],
        "min_coarse_accuracy": candidate["min_label_accuracy"] >= t["confirmation_min_coarse_accuracy_min"],
        "gauge_accuracy_gap": candidate["gauge_accuracy_gap"] <= t["candidate_gauge_accuracy_gap_max"],
        "prediction_was_blind": manifest["confirmation_truth_read"] is False,
        "prediction_hash": sha256_file(OUT_ROOT / "predictions/confirmation_predictions.jsonl") == manifest["prediction_sha256"],
    }
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_digest": manifest["prediction_digest"],
        "count": len(truth),
        "algorithm_metrics": metrics,
        "candidate_checks": checks,
        "candidate_confirmed": all(checks.values()),
    }
    result["score_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/score.json", result)
    print(canonical(result))


def finalize_command() -> None:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    confirmation = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    score = read_json(OUT_ROOT / "analysis/score.json")
    discovery_ids = {row["unit_id"] for row in read_jsonl(OUT_ROOT / "runs/discovery/public_manifest.jsonl")}
    confirmation_ids = {row["unit_id"] for row in read_jsonl(OUT_ROOT / "runs/confirmation/public_manifest.jsonl")}
    overlap = len(discovery_ids & confirmation_ids)
    passed = bool(
        discovery["run_gate_passed"]
        and fit["candidate_qualified"]
        and confirmation["run_gate_passed"]
        and score["candidate_confirmed"]
        and overlap == 0
    )
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": discovery["summary_digest"],
        "fit_digest": fit["fit_digest"],
        "confirmation_summary_digest": confirmation["summary_digest"],
        "score_digest": score["score_digest"],
        "split_overlap": overlap,
        "simple_sitewise_gl_as_phase1154_failure_explanation_supported": False if passed else None,
        "sitewise_general_linear_commutation_confirmed": bool(
            discovery["checks"]["site_gl_functional_commutation"]
            and confirmation["checks"]["site_gl_functional_commutation"]
        ),
        "cross_site_intervention_algebra_boundary_confirmed": bool(
            discovery["checks"]["cross_site_intervention_semantics_changed"]
            and confirmation["checks"]["cross_site_intervention_semantics_changed"]
        ),
        "coarse_dependency_quotient_confirmed": passed,
        "full_six_way_morphology_identification_claim": False,
        "free_transformer_scan_authorized": False,
        "pretrained_model_scan_authorized": False,
        "outcome": "coarse_gauge_quotient_confirmed" if passed else "coarse_gauge_quotient_not_confirmed",
        "claim_boundary": (
            "The confirmed object, if passed, is a three-way factor-dependency rank signature under bounded invertible linear gauges. "
            "It is not a six-way causal morphology, does not identify redundant paths under unrestricted site mixing, and does not cover nonlinear gauges."
        ),
        "next_legal_task": (
            "Define and calibrate a causal-use observable on the recovered factor subspaces before any free-Transformer or pretrained-model scan."
        ),
        "auto_continue": False,
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
    sub.add_parser("fit")
    sub.add_parser("predict")
    sub.add_parser("score")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.split)
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
