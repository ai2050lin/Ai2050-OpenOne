#!/usr/bin/env python3
"""Calibrate additive main effects versus an interaction-only causal hyperedge.

All five systems share the same representation and clean input-output function.
They differ only in how row and column mediator changes are used after an
intervention: neither, row only, column only, independent row+column, or a
synergy gate that uses the mediator only when both factors changed together.
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

import phase1153_blind_algorithm_coverage as coverage
import phase1156_causal_use_quotient_calibration as base_phase


PHASE = 1157
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1157_causal_hyperedge_calibration"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1156_causal_use_quotient_calibration"
SPLITS = ("discovery", "confirmation")
MORPHOLOGIES = (
    "bypass_none",
    "row_main",
    "col_main",
    "additive_row_col",
    "synergy_row_col",
)
GAUGES = base_phase.GAUGES
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
ALGORITHMS = ("state_gram", "dependency_rank_quotient", "matched_hyperedge_tomography")
CANDIDATE = "matched_hyperedge_tomography"
THRESHOLDS = {
    "clean_accuracy_min": 1.0,
    "clean_target_probability_min": 0.9999,
    "clean_function_abs_error_max": 1e-10,
    "finite_fraction_min": 1.0,
    "representation_morphology_match_fraction_min": 1.0,
    "dependency_gauge_match_fraction_min": 1.0,
    "candidate_gauge_abs_error_max": 1e-10,
    "matched_null_tv_max": 1e-12,
    "synergy_gap_min": 0.9999,
    "non_synergy_gap_abs_max": 1e-6,
    "expected_target_probability_min": 0.9999,
    "discovery_accuracy_min": 1.0,
    "discovery_min_label_accuracy_min": 1.0,
    "confirmation_accuracy_min": 1.0,
    "confirmation_min_label_accuracy_min": 1.0,
    "candidate_gauge_accuracy_gap_max": 0.0,
    "representation_control_accuracy_max": 0.2,
}


canonical = base_phase.canonical
digest = base_phase.digest
sha256_file = base_phase.sha256_file
write_json = base_phase.write_json
read_json = base_phase.read_json
write_jsonl = base_phase.write_jsonl
read_jsonl = base_phase.read_jsonl


def seed_for(split: str, replicate: int) -> int:
    return (115710 if split == "discovery" else 115790) + int(replicate) * 1009


def changed_pair(
    inputs: list[tuple[int, int, int]], change_row: bool, change_col: bool
) -> list[tuple[int, int, int]]:
    return [
        ((row + 1) % base_phase.N_ROWS if change_row else row,
         (col + 1) % base_phase.N_COLS if change_col else col,
         context)
        for row, col, context in inputs
    ]


class HyperedgeOracle(base_phase.GaugeOracle):
    def output_morphology(
        self,
        observed: torch.Tensor,
        receivers: list[tuple[int, int, int]],
        morphology: str,
    ) -> torch.Tensor:
        logical = self.decode(observed)
        mediated_scores = self.base.decoded_factor_scores(logical)
        row_index = torch.tensor([row for row, _col, _context in receivers], dtype=torch.long, device=self.device)
        col_index = torch.tensor([col for _row, col, _context in receivers], dtype=torch.long, device=self.device)
        context_index = torch.tensor([context for _row, _col, context in receivers], dtype=torch.long, device=self.device)
        raw_scores = {
            "row": torch.nn.functional.one_hot(row_index, base_phase.N_ROWS).to(torch.float64),
            "col": torch.nn.functional.one_hot(col_index, base_phase.N_COLS).to(torch.float64),
            "context": torch.nn.functional.one_hot(context_index, base_phase.N_CONTEXTS).to(torch.float64),
        }
        if morphology == "bypass_none":
            row_scores, col_scores = raw_scores["row"], raw_scores["col"]
        elif morphology == "row_main":
            row_scores, col_scores = mediated_scores["row"], raw_scores["col"]
        elif morphology == "col_main":
            row_scores, col_scores = raw_scores["row"], mediated_scores["col"]
        elif morphology == "additive_row_col":
            row_scores, col_scores = mediated_scores["row"], mediated_scores["col"]
        elif morphology == "synergy_row_col":
            mediated_row = torch.argmax(mediated_scores["row"], dim=1)
            mediated_col = torch.argmax(mediated_scores["col"], dim=1)
            gate = ((mediated_row != row_index) & (mediated_col != col_index)).to(torch.float64)[:, None]
            row_scores = gate * mediated_scores["row"] + (1.0 - gate) * raw_scores["row"]
            col_scores = gate * mediated_scores["col"] + (1.0 - gate) * raw_scores["col"]
        else:
            raise ValueError(morphology)
        probabilities = {
            "row": torch.softmax(base_phase.LOGIT_SCALE * row_scores, dim=1),
            "col": torch.softmax(base_phase.LOGIT_SCALE * col_scores, dim=1),
            "context": torch.softmax(base_phase.LOGIT_SCALE * raw_scores["context"], dim=1),
        }
        joint = (
            probabilities["context"][:, :, None, None]
            * probabilities["row"][:, None, :, None]
            * probabilities["col"][:, None, None, :]
        )
        return joint.reshape(len(receivers), -1)


def marginals(distribution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shaped = distribution.reshape(
        len(distribution), base_phase.N_CONTEXTS, base_phase.N_ROWS, base_phase.N_COLS
    )
    return shaped.sum(dim=(1, 3)), shaped.sum(dim=(1, 2))


def hyperedge_feature(
    oracle: HyperedgeOracle,
    inputs: list[tuple[int, int, int]],
    states: torch.Tensor,
    morphology: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_donors = changed_pair(inputs, True, False)
    col_donors = changed_pair(inputs, False, True)
    joint_donors = changed_pair(inputs, True, True)
    baseline = oracle.output_morphology(states, inputs, morphology)
    row_state = oracle.states(row_donors)
    col_state = oracle.states(col_donors)
    joint_state = oracle.states(joint_donors)
    row_output = oracle.output_morphology(states + (row_state - states), inputs, morphology)
    col_output = oracle.output_morphology(states + (col_state - states), inputs, morphology)
    joint_output = oracle.output_morphology(states + (joint_state - states), inputs, morphology)
    null_output = oracle.output_morphology(states + (states - states), inputs, morphology)
    batch = torch.arange(len(inputs), device=oracle.device)
    row_target_index = torch.tensor([row for row, _col, _context in row_donors], dtype=torch.long, device=oracle.device)
    col_target_index = torch.tensor([col for _row, col, _context in col_donors], dtype=torch.long, device=oracle.device)
    joint_targets = base_phase.target_indices(joint_donors, oracle.device)
    receiver_targets = base_phase.target_indices(inputs, oracle.device)
    row_marginal, _ = marginals(row_output)
    _, col_marginal = marginals(col_output)
    joint_row_marginal, joint_col_marginal = marginals(joint_output)
    row_donor_probability = float(torch.mean(row_marginal[batch, row_target_index]).item())
    col_donor_probability = float(torch.mean(col_marginal[batch, col_target_index]).item())
    joint_target_probability = float(torch.mean(joint_output[batch, joint_targets]).item())
    joint_row_probability = float(torch.mean(joint_row_marginal[batch, row_target_index]).item())
    joint_col_probability = float(torch.mean(joint_col_marginal[batch, col_target_index]).item())
    synergy_gap = joint_target_probability - row_donor_probability * col_donor_probability
    null_tv = float(torch.mean(0.5 * torch.sum(torch.abs(null_output - baseline), dim=1)).item())

    if morphology == "bypass_none":
        expected = receiver_targets
    elif morphology == "row_main":
        expected = base_phase.target_indices(row_donors, oracle.device)
    elif morphology == "col_main":
        expected = base_phase.target_indices(col_donors, oracle.device)
    else:
        expected = joint_targets
    expected_probability = float(torch.mean(joint_output[batch, expected]).item())
    feature = np.asarray(
        [
            row_donor_probability,
            col_donor_probability,
            joint_target_probability,
            synergy_gap,
            joint_row_probability,
            joint_col_probability,
        ],
        dtype=np.float64,
    )
    return feature, {
        "row_donor_probability": row_donor_probability,
        "col_donor_probability": col_donor_probability,
        "joint_target_probability": joint_target_probability,
        "joint_row_probability": joint_row_probability,
        "joint_col_probability": joint_col_probability,
        "synergy_gap": synergy_gap,
        "matched_null_tv": null_tv,
        "expected_target_probability": expected_probability,
    }


def classification_metrics(predicted: list[str], truth: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    correct = [predicted[offset] == str(truth[index]["morphology"]) for offset, index in enumerate(indices)]
    labels = sorted({str(truth[index]["morphology"]) for index in indices})
    per_label = {}
    for label in labels:
        selected = [offset for offset, index in enumerate(indices) if str(truth[index]["morphology"]) == label]
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
        raise RuntimeError("refusing to rewrite Phase1157 artifacts")
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks = {
        "source_causal_use_confirmed": bool(source_final["matched_transport_causal_use_camera_confirmed"]),
        "source_auto_continue": bool(source_final["auto_continue"]),
        "source_audit_passed": bool(source_audit["all_checks_passed"]),
        "same_representation_required": True,
        "same_clean_function_required": True,
        "single_and_joint_transports_predeclared": True,
        "synergy_formula_predeclared": True,
        "candidate_predeclared": CANDIDATE == "matched_hyperedge_tomography",
        "confirmation_truth_forbidden_in_predict": True,
        "redundancy_gate_learned_network_claims_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "matched additive-versus-synergistic causal hyperedge calibration",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1156_digest": source_final["final_digest"],
        "source_phase1156_audit_digest": source_audit["audit_digest"],
        "morphologies": list(MORPHOLOGIES),
        "gauges": list(GAUGES),
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "algorithms": list(ALGORITHMS),
        "candidate": CANDIDATE,
        "synergy_formula": "P_RC(y_r'c') - P_R(r') * P_C(c')",
        "thresholds": THRESHOLDS,
        "primary_endpoint": "blind five-way recovery of none, row, column, additive, and interaction-only row-column use",
        "hard_stops": [
            "The synergy gate compares mediator and raw calibration variables; it is not a claimed natural-network gate.",
            "The output-class interaction of a joint label is not itself called synergy; the predeclared marginal-adjusted gap must pass.",
            "This phase does not identify redundant paths, necessity, learned-network external validity, or natural-language mechanisms.",
            "Confirmation predictions must be sealed before confirmation truth is read.",
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
        raise RuntimeError("Phase1157 frozen protocol mismatch")
    return protocol


def run_command(split: str) -> None:
    protocol = verify_protocol()
    if split == "confirmation":
        fit = read_json(OUT_ROOT / "analysis/fit.json")
        if not fit["confirmation_run_authorized"]:
            raise RuntimeError("confirmation run denied")
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    inputs = base_phase.all_inputs()
    batch = torch.arange(len(inputs), device=device)
    targets = base_phase.target_indices(inputs, device)
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in ALGORITHMS}
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    representation_manifest: list[dict[str, Any]] = []
    representations = out / "representations"
    representations.mkdir(parents=True, exist_ok=False)
    by_key: dict[tuple[str, int, str], int] = {}
    index = 0
    for replicate in range(REPLICATES):
        seed = seed_for(split, replicate)
        base = base_phase.BaseRepresentation(seed, device)
        representation_id = digest({"phase": PHASE, "split": split, "replicate": replicate, "seed": seed})[:18]
        representation_path = representations / f"{representation_id}.npz"
        np.savez_compressed(representation_path, **base.export())
        representation_manifest.append(
            {
                "representation_id": representation_id,
                "replicate": replicate,
                "seed": seed,
                "representation_sha256": sha256_file(representation_path),
            }
        )
        for gauge in GAUGES:
            oracle = HyperedgeOracle(base, gauge, seed + 17011 + 313 * GAUGES.index(gauge))
            states = oracle.states(inputs)
            gram = base_phase.state_gram_feature(states)
            rank, rank_diagnostics = base_phase.dependency_rank_feature(states)
            reference = oracle.output_morphology(states, inputs, "bypass_none")
            for morphology in MORPHOLOGIES:
                clean = oracle.output_morphology(states, inputs, morphology)
                accuracy = float(torch.mean((torch.argmax(clean, dim=1) == targets).to(torch.float64)).item())
                probability = float(torch.min(clean[batch, targets]).item())
                clean_error = float(torch.max(torch.abs(clean - reference)).item())
                causal, causal_diagnostics = hyperedge_feature(oracle, inputs, states, morphology)
                feature_rows["state_gram"].append(gram.astype(np.float32))
                feature_rows["dependency_rank_quotient"].append(rank.astype(np.float32))
                feature_rows[CANDIDATE].append(causal.astype(np.float64))
                unit_id = digest(
                    {"phase": PHASE, "split": split, "representation_id": representation_id, "gauge": gauge, "morphology": morphology}
                )[:20]
                public.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "representation_id": representation_id,
                        "split": split,
                        "replicate": replicate,
                        "gauge": gauge,
                    }
                )
                truth.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "representation_id": representation_id,
                        "split": split,
                        "replicate": replicate,
                        "gauge": gauge,
                        "morphology": morphology,
                    }
                )
                diagnostics.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "clean_accuracy": accuracy,
                        "clean_target_probability_min": probability,
                        "clean_function_abs_error_max": clean_error,
                        "rank_diagnostics": rank_diagnostics,
                        "causal_diagnostics": causal_diagnostics,
                    }
                )
                by_key[(morphology, replicate, gauge)] = index
                index += 1
        del base
        torch.cuda.empty_cache()

    arrays = {name: np.stack(rows, axis=0) for name, rows in feature_rows.items()}
    np.savez_compressed(out / "feature_pack.npz", **arrays)
    write_jsonl(out / "public_manifest.jsonl", public)
    write_jsonl(out / "sealed_truth.jsonl", truth)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)
    write_jsonl(out / "representation_manifest.jsonl", representation_manifest)

    representation_matches = []
    dependency_matches = []
    candidate_errors = []
    for replicate in range(REPLICATES):
        for gauge in GAUGES:
            base_index = by_key[("bypass_none", replicate, gauge)]
            for morphology in MORPHOLOGIES:
                current = by_key[(morphology, replicate, gauge)]
                representation_matches.append(
                    bool(
                        np.array_equal(arrays["state_gram"][base_index], arrays["state_gram"][current])
                        and np.array_equal(
                            arrays["dependency_rank_quotient"][base_index], arrays["dependency_rank_quotient"][current]
                        )
                    )
                )
        for morphology in MORPHOLOGIES:
            identity = by_key[(morphology, replicate, "identity")]
            for gauge in GAUGES[1:]:
                current = by_key[(morphology, replicate, gauge)]
                dependency_matches.append(
                    bool(
                        np.array_equal(
                            arrays["dependency_rank_quotient"][identity], arrays["dependency_rank_quotient"][current]
                        )
                    )
                )
                candidate_errors.append(float(np.max(np.abs(arrays[CANDIDATE][identity] - arrays[CANDIDATE][current]))))

    synergy = [
        row["causal_diagnostics"]["synergy_gap"]
        for row, truth_row in zip(diagnostics, truth)
        if truth_row["morphology"] == "synergy_row_col"
    ]
    non_synergy = [
        abs(row["causal_diagnostics"]["synergy_gap"])
        for row, truth_row in zip(diagnostics, truth)
        if truth_row["morphology"] != "synergy_row_col"
    ]
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "representation_count": len(representation_manifest),
        "unit_count": len(public),
        "clean_accuracy_min": min(row["clean_accuracy"] for row in diagnostics),
        "clean_target_probability_min": min(row["clean_target_probability_min"] for row in diagnostics),
        "clean_function_abs_error_max": max(row["clean_function_abs_error_max"] for row in diagnostics),
        "finite_fraction": float(np.mean([np.isfinite(array).mean() for array in arrays.values()])),
        "representation_morphology_match_fraction": float(np.mean(representation_matches)),
        "dependency_gauge_match_fraction": float(np.mean(dependency_matches)),
        "candidate_gauge_abs_error_max": max(candidate_errors),
        "matched_null_tv_max": max(row["causal_diagnostics"]["matched_null_tv"] for row in diagnostics),
        "synergy_gap_min": min(synergy),
        "non_synergy_gap_abs_max": max(non_synergy),
        "expected_target_probability_min": min(
            row["causal_diagnostics"]["expected_target_probability"] for row in diagnostics
        ),
        "feature_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "feature_pack_sha256": sha256_file(out / "feature_pack.npz"),
        "public_manifest_sha256": sha256_file(out / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(out / "sealed_truth.jsonl"),
        "diagnostics_sha256": sha256_file(out / "diagnostics.jsonl"),
        "representation_manifest_sha256": sha256_file(out / "representation_manifest.jsonl"),
    }
    t = protocol["thresholds"]
    checks = {
        "clean_accuracy": summary["clean_accuracy_min"] >= t["clean_accuracy_min"],
        "clean_probability": summary["clean_target_probability_min"] >= t["clean_target_probability_min"],
        "same_clean_function": summary["clean_function_abs_error_max"] <= t["clean_function_abs_error_max"],
        "finite": summary["finite_fraction"] >= t["finite_fraction_min"],
        "same_representation": summary["representation_morphology_match_fraction"] >= t["representation_morphology_match_fraction_min"],
        "dependency_gauge_invariance": summary["dependency_gauge_match_fraction"] >= t["dependency_gauge_match_fraction_min"],
        "candidate_gauge_equivariance": summary["candidate_gauge_abs_error_max"] <= t["candidate_gauge_abs_error_max"],
        "matched_null": summary["matched_null_tv_max"] <= t["matched_null_tv_max"],
        "synergy_positive": summary["synergy_gap_min"] >= t["synergy_gap_min"],
        "non_synergy_zero": summary["non_synergy_gap_abs_max"] <= t["non_synergy_gap_abs_max"],
        "expected_target": summary["expected_target_probability_min"] >= t["expected_target_probability_min"],
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
    truth = read_jsonl(OUT_ROOT / "runs/discovery/sealed_truth.jsonl")
    with np.load(OUT_ROOT / "runs/discovery/feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    fit_indices = [
        index for index, row in enumerate(truth) if int(row["replicate"]) in FIT_REPLICATES and row["gauge"] == "identity"
    ]
    validation_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in VALIDATION_REPLICATES]
    prototypes: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        labels, proto = coverage.build_prototypes(arrays[algorithm], truth, fit_indices, "morphology")
        prototypes[algorithm] = proto.astype(np.float64)
        metadata[algorithm] = {"labels": labels}
        prediction, _ = coverage.predict(arrays[algorithm][validation_indices], labels, proto)
        metrics[algorithm] = classification_metrics(prediction, truth, validation_indices)
    candidate = metrics[CANDIDATE]
    t = protocol["thresholds"]
    checks = {
        "candidate_accuracy": candidate["accuracy"] >= t["discovery_accuracy_min"],
        "candidate_min_label_accuracy": candidate["min_label_accuracy"] >= t["discovery_min_label_accuracy_min"],
        "candidate_gauge_gap": candidate["gauge_accuracy_gap"] <= t["candidate_gauge_accuracy_gap_max"],
        "state_control_at_ceiling": metrics["state_gram"]["accuracy"] <= t["representation_control_accuracy_max"],
        "rank_control_at_ceiling": metrics["dependency_rank_quotient"]["accuracy"] <= t["representation_control_accuracy_max"],
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
    public = read_jsonl(OUT_ROOT / "runs/confirmation/public_manifest.jsonl")
    with np.load(OUT_ROOT / "runs/confirmation/feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    metadata = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as pack:
        prototypes = {name: np.asarray(pack[name]) for name in pack.files}
    rows = []
    for algorithm in ALGORITHMS:
        predicted, scores = coverage.predict(arrays[algorithm], metadata[algorithm]["labels"], prototypes[algorithm])
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
            rows[index]["algorithms"][algorithm] = {"prediction": predicted[index], "cosine": float(scores[index])}
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
    indices = list(range(len(truth)))
    metrics = {}
    for algorithm in ALGORITHMS:
        predicted = [row["algorithms"][algorithm]["prediction"] for row in predictions]
        metrics[algorithm] = classification_metrics(predicted, truth, indices)
    candidate = metrics[CANDIDATE]
    t = protocol["thresholds"]
    checks = {
        "candidate_accuracy": candidate["accuracy"] >= t["confirmation_accuracy_min"],
        "candidate_min_label_accuracy": candidate["min_label_accuracy"] >= t["confirmation_min_label_accuracy_min"],
        "candidate_gauge_gap": candidate["gauge_accuracy_gap"] <= t["candidate_gauge_accuracy_gap_max"],
        "state_control_at_ceiling": metrics["state_gram"]["accuracy"] <= t["representation_control_accuracy_max"],
        "rank_control_at_ceiling": metrics["dependency_rank_quotient"]["accuracy"] <= t["representation_control_accuracy_max"],
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
        "matched_causal_hyperedge_camera_confirmed": passed,
        "additive_versus_interaction_only_separation_confirmed": passed,
        "redundancy_identification_confirmed": False,
        "gate_external_validity_confirmed": False,
        "learned_network_external_validity_confirmed": False,
        "free_transformer_scan_authorized": False,
        "pretrained_model_scan_authorized": False,
        "outcome": "controlled_causal_hyperedge_confirmed" if passed else "controlled_causal_hyperedge_not_confirmed",
        "claim_boundary": (
            "The confirmed object, if passed, separates independent row/column main effects from an explicit interaction-only calibration gate under matched single and joint transports. "
            "The gate is known truth and uses raw calibration variables; this is not evidence that a learned or natural network implements the same gate."
        ),
        "next_legal_task": (
            "Calibrate redundant sufficient paths and conditional gates, including single-ablation false negatives and joint-ablation recovery, before any learned-system transfer."
        ),
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
