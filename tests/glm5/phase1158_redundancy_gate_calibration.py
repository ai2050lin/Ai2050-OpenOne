#!/usr/bin/env python3
"""Calibrate redundant sufficient paths and a context-selected gate.

Five systems share one dual-path representation and one clean row-classification
function.  Path-specific and joint ablations distinguish a bypass, either
single path, two individually sufficient redundant paths, and a context gate.
Path-disagreement transports make the gate observable without changing the
clean function.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase1153_blind_algorithm_coverage as coverage
import phase1156_causal_use_quotient_calibration as base_phase


PHASE = 1158
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1158_redundancy_gate_calibration"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1157_causal_hyperedge_calibration"
SPLITS = ("discovery", "confirmation")
MORPHOLOGIES = ("bypass_none", "single_path_a", "single_path_b", "redundant_paths", "context_gate")
GAUGES = base_phase.GAUGES
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
ALGORITHMS = ("state_gram", "dependency_rank_quotient", "redundancy_gate_tomography")
CANDIDATE = "redundancy_gate_tomography"
THRESHOLDS = {
    "clean_accuracy_min": 1.0,
    "clean_target_probability_min": 0.9999,
    "clean_function_abs_error_max": 1e-10,
    "finite_fraction_min": 1.0,
    "representation_morphology_match_fraction_min": 1.0,
    "dependency_gauge_match_fraction_min": 1.0,
    "candidate_gauge_abs_error_max": 1e-10,
    "matched_null_tv_max": 1e-12,
    "redundant_single_ablation_probability_min": 0.9999,
    "redundant_joint_ablation_probability_max": 0.2501,
    "redundancy_joint_gap_min": 0.749,
    "gate_path_selectivity_min": 0.9999,
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
    return (115810 if split == "discovery" else 115890) + int(replicate) * 1009


class DualPathRepresentation:
    def __init__(self, seed: int, device: torch.device) -> None:
        self.seed = int(seed)
        self.device = device
        self.path_a_embedding = torch.tensor(
            base_phase.make_orthogonal(base_phase.STATE_DIM, seed + 11)[: base_phase.N_ROWS],
            dtype=torch.float64,
            device=device,
        )
        self.path_b_embedding = torch.tensor(
            base_phase.make_orthogonal(base_phase.STATE_DIM, seed + 29)[: base_phase.N_ROWS],
            dtype=torch.float64,
            device=device,
        )
        self.context_embedding = torch.tensor(
            base_phase.make_orthogonal(base_phase.STATE_DIM, seed + 43)[: base_phase.N_CONTEXTS],
            dtype=torch.float64,
            device=device,
        )
        rng = np.random.default_rng(seed + 61)
        self.col_nuisance = torch.tensor(
            rng.normal(scale=0.35, size=(base_phase.N_COLS, base_phase.STATE_DIM)), dtype=torch.float64, device=device
        )
        self.joint_nuisance = torch.tensor(
            rng.normal(
                scale=0.30,
                size=(base_phase.N_CONTEXTS, base_phase.N_ROWS, base_phase.N_COLS, base_phase.STATE_DIM),
            ),
            dtype=torch.float64,
            device=device,
        )

    def export(self) -> dict[str, np.ndarray]:
        return {
            "path_a_embedding": self.path_a_embedding.detach().cpu().numpy(),
            "path_b_embedding": self.path_b_embedding.detach().cpu().numpy(),
            "context_embedding": self.context_embedding.detach().cpu().numpy(),
            "col_nuisance": self.col_nuisance.detach().cpu().numpy(),
            "joint_nuisance": self.joint_nuisance.detach().cpu().numpy(),
        }

    def logical_states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        rows = torch.tensor([row for row, _col, _context in inputs], dtype=torch.long, device=self.device)
        cols = torch.tensor([col for _row, col, _context in inputs], dtype=torch.long, device=self.device)
        contexts = torch.tensor([context for _row, _col, context in inputs], dtype=torch.long, device=self.device)
        states = torch.zeros(
            (len(inputs), base_phase.N_SITES, base_phase.STATE_DIM), dtype=torch.float64, device=self.device
        )
        states[:, 0] = self.path_a_embedding[rows]
        states[:, 1] = self.path_b_embedding[rows]
        states[:, 2] = self.context_embedding[contexts]
        states[:, 3] = self.col_nuisance[cols]
        states[:, 4] = self.joint_nuisance[contexts, rows, cols]
        return states

    def scores(self, logical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            logical[:, 0] @ self.path_a_embedding.T,
            logical[:, 1] @ self.path_b_embedding.T,
            logical[:, 2] @ self.context_embedding.T,
        )


class RedundancyOracle:
    def __init__(self, base: DualPathRepresentation, gauge: str, gauge_seed: int) -> None:
        self.base = base
        self.device = base.device
        self.gauge = gauge
        self.matrix = torch.tensor(base_phase.gauge_matrix(gauge, gauge_seed), dtype=torch.float64, device=self.device)
        self.inverse = torch.linalg.inv(self.matrix)

    def observe(self, logical: torch.Tensor) -> torch.Tensor:
        flattened = logical.reshape(len(logical), -1)
        return (flattened @ self.matrix.T).reshape(len(logical), base_phase.N_SITES, base_phase.STATE_DIM)

    def states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        return self.observe(self.base.logical_states(inputs))

    def decode(self, observed: torch.Tensor) -> torch.Tensor:
        flattened = observed.reshape(len(observed), -1)
        return (flattened @ self.inverse.T).reshape(len(observed), base_phase.N_SITES, base_phase.STATE_DIM)

    def output(
        self, observed: torch.Tensor, receivers: list[tuple[int, int, int]], morphology: str
    ) -> torch.Tensor:
        logical = self.decode(observed)
        path_a, path_b, context_scores = self.base.scores(logical)
        row_index = torch.tensor([row for row, _col, _context in receivers], dtype=torch.long, device=self.device)
        raw = torch.nn.functional.one_hot(row_index, base_phase.N_ROWS).to(torch.float64)
        if morphology == "bypass_none":
            scores = raw
        elif morphology == "single_path_a":
            scores = path_a
        elif morphology == "single_path_b":
            scores = path_b
        elif morphology == "redundant_paths":
            scores = torch.maximum(path_a, path_b)
        elif morphology == "context_gate":
            gate = torch.argmax(context_scores, dim=1).to(torch.float64)[:, None]
            scores = (1.0 - gate) * path_a + gate * path_b
        else:
            raise ValueError(morphology)
        return torch.softmax(base_phase.LOGIT_SCALE * scores, dim=1)

    def ablated_states(self, inputs: list[tuple[int, int, int]], paths: tuple[str, ...]) -> torch.Tensor:
        logical = self.base.logical_states(inputs).clone()
        if "a" in paths:
            logical[:, 0] = 0.0
        if "b" in paths:
            logical[:, 1] = 0.0
        return self.observe(logical)

    def disagreement_states(
        self, receivers: list[tuple[int, int, int]], donor_path: str
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        donors = base_phase.changed_inputs(receivers, "row")
        logical = self.base.logical_states(receivers).clone()
        donor_logical = self.base.logical_states(donors)
        if donor_path == "a":
            logical[:, 0] = donor_logical[:, 0]
        elif donor_path == "b":
            logical[:, 1] = donor_logical[:, 1]
        else:
            raise ValueError(donor_path)
        return self.observe(logical), donors


def target_probability(
    distribution: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor
) -> float:
    return float(torch.mean(distribution[indices, targets[indices]]).item())


def redundancy_gate_feature(
    oracle: RedundancyOracle,
    inputs: list[tuple[int, int, int]],
    states: torch.Tensor,
    morphology: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    receiver_targets = torch.tensor([row for row, _col, _context in inputs], dtype=torch.long, device=oracle.device)
    all_indices = torch.arange(len(inputs), device=oracle.device)
    contexts = torch.tensor([context for _row, _col, context in inputs], dtype=torch.long, device=oracle.device)
    context_zero = torch.where(contexts == 0)[0]
    context_one = torch.where(contexts == 1)[0]
    baseline = oracle.output(states, inputs, morphology)
    ablate_a = oracle.output(oracle.ablated_states(inputs, ("a",)), inputs, morphology)
    ablate_b = oracle.output(oracle.ablated_states(inputs, ("b",)), inputs, morphology)
    ablate_joint = oracle.output(oracle.ablated_states(inputs, ("a", "b")), inputs, morphology)
    null_output = oracle.output(states + (states - states), inputs, morphology)
    disagreement_a, donors = oracle.disagreement_states(inputs, "a")
    disagreement_b, _ = oracle.disagreement_states(inputs, "b")
    output_a = oracle.output(disagreement_a, inputs, morphology)
    output_b = oracle.output(disagreement_b, inputs, morphology)
    donor_targets = torch.tensor([row for row, _col, _context in donors], dtype=torch.long, device=oracle.device)

    a_all = target_probability(ablate_a, receiver_targets, all_indices)
    b_all = target_probability(ablate_b, receiver_targets, all_indices)
    joint_all = target_probability(ablate_joint, receiver_targets, all_indices)
    a_ctx0 = target_probability(ablate_a, receiver_targets, context_zero)
    a_ctx1 = target_probability(ablate_a, receiver_targets, context_one)
    b_ctx0 = target_probability(ablate_b, receiver_targets, context_zero)
    b_ctx1 = target_probability(ablate_b, receiver_targets, context_one)
    patch_a_ctx0 = target_probability(output_a, donor_targets, context_zero)
    patch_a_ctx1 = target_probability(output_a, donor_targets, context_one)
    patch_b_ctx0 = target_probability(output_b, donor_targets, context_zero)
    patch_b_ctx1 = target_probability(output_b, donor_targets, context_one)
    null_tv = float(torch.mean(0.5 * torch.sum(torch.abs(null_output - baseline), dim=1)).item())
    feature = np.asarray(
        [
            a_all,
            b_all,
            joint_all,
            a_ctx0,
            a_ctx1,
            b_ctx0,
            b_ctx1,
            patch_a_ctx0,
            patch_a_ctx1,
            patch_b_ctx0,
            patch_b_ctx1,
        ],
        dtype=np.float64,
    )
    return feature, {
        "ablate_a_target_probability": a_all,
        "ablate_b_target_probability": b_all,
        "joint_ablation_target_probability": joint_all,
        "ablate_a_context0": a_ctx0,
        "ablate_a_context1": a_ctx1,
        "ablate_b_context0": b_ctx0,
        "ablate_b_context1": b_ctx1,
        "patch_a_donor_context0": patch_a_ctx0,
        "patch_a_donor_context1": patch_a_ctx1,
        "patch_b_donor_context0": patch_b_ctx0,
        "patch_b_donor_context1": patch_b_ctx1,
        "matched_null_tv": null_tv,
    }


def classification_metrics(predicted: list[str], truth: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    correct = [predicted[offset] == truth[index]["morphology"] for offset, index in enumerate(indices)]
    labels = sorted({truth[index]["morphology"] for index in indices})
    per_label = {
        label: float(
            np.mean([correct[offset] for offset, index in enumerate(indices) if truth[index]["morphology"] == label])
        )
        for label in labels
    }
    per_gauge = {
        gauge: float(np.mean([correct[offset] for offset, index in enumerate(indices) if truth[index]["gauge"] == gauge]))
        for gauge in GAUGES
    }
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "per_gauge_accuracy": per_gauge,
        "gauge_accuracy_gap": float(max(per_gauge.values()) - min(per_gauge.values())),
        "count": len(indices),
    }


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to rewrite Phase1158 artifacts")
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks = {
        "source_hyperedge_confirmed": bool(source_final["matched_causal_hyperedge_camera_confirmed"]),
        "source_auto_continue": bool(source_final["auto_continue"]),
        "source_audit_passed": bool(source_audit["all_checks_passed"]),
        "same_representation_required": True,
        "same_clean_function_required": True,
        "single_and_joint_ablations_predeclared": True,
        "path_disagreement_gate_probe_predeclared": True,
        "candidate_predeclared": CANDIDATE == "redundancy_gate_tomography",
        "confirmation_truth_forbidden_in_predict": True,
        "learned_and_natural_claims_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "redundant sufficient paths and conditional gate calibration",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1157_digest": source_final["final_digest"],
        "source_phase1157_audit_digest": source_audit["audit_digest"],
        "morphologies": list(MORPHOLOGIES),
        "gauges": list(GAUGES),
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "algorithms": list(ALGORITHMS),
        "candidate": CANDIDATE,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "blind five-way recovery of bypass, either single path, redundant paths, and context gate",
        "hard_stops": [
            "A null single-path necessity effect does not license an unused-path claim when joint ablation remains untested.",
            "The context gate and dual paths are explicit known-truth scaffolds, not discovered natural-network modules.",
            "Passing known-truth calibration does not by itself authorize a pretrained-model scan.",
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
        raise RuntimeError("Phase1158 frozen protocol mismatch")
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
    targets = torch.tensor([row for row, _col, _context in inputs], dtype=torch.long, device=device)
    batch = torch.arange(len(inputs), device=device)
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
        base = DualPathRepresentation(seed, device)
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
            oracle = RedundancyOracle(base, gauge, seed + 17011 + 313 * GAUGES.index(gauge))
            states = oracle.states(inputs)
            gram = base_phase.state_gram_feature(states)
            rank, rank_diagnostics = base_phase.dependency_rank_feature(states)
            reference = oracle.output(states, inputs, "bypass_none")
            for morphology in MORPHOLOGIES:
                clean = oracle.output(states, inputs, morphology)
                accuracy = float(torch.mean((torch.argmax(clean, dim=1) == targets).to(torch.float64)).item())
                probability = float(torch.min(clean[batch, targets]).item())
                clean_error = float(torch.max(torch.abs(clean - reference)).item())
                causal, causal_diagnostics = redundancy_gate_feature(oracle, inputs, states, morphology)
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

    redundant = [
        row["causal_diagnostics"] for row, truth_row in zip(diagnostics, truth) if truth_row["morphology"] == "redundant_paths"
    ]
    gates = [row["causal_diagnostics"] for row, truth_row in zip(diagnostics, truth) if truth_row["morphology"] == "context_gate"]
    redundant_single_min = min(
        min(row["ablate_a_target_probability"], row["ablate_b_target_probability"]) for row in redundant
    )
    redundant_joint_max = max(row["joint_ablation_target_probability"] for row in redundant)
    redundancy_gap_min = min(
        min(row["ablate_a_target_probability"], row["ablate_b_target_probability"])
        - row["joint_ablation_target_probability"]
        for row in redundant
    )
    gate_selectivity_min = min(
        min(
            row["patch_a_donor_context0"] - row["patch_a_donor_context1"],
            row["patch_b_donor_context1"] - row["patch_b_donor_context0"],
        )
        for row in gates
    )
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
        "redundant_single_ablation_probability_min": redundant_single_min,
        "redundant_joint_ablation_probability_max": redundant_joint_max,
        "redundancy_joint_gap_min": redundancy_gap_min,
        "gate_path_selectivity_min": gate_selectivity_min,
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
        "redundant_single_paths_sufficient": summary["redundant_single_ablation_probability_min"] >= t["redundant_single_ablation_probability_min"],
        "redundant_joint_ablation_destructive": summary["redundant_joint_ablation_probability_max"] <= t["redundant_joint_ablation_probability_max"],
        "redundancy_joint_gap": summary["redundancy_joint_gap_min"] >= t["redundancy_joint_gap_min"],
        "gate_path_selectivity": summary["gate_path_selectivity_min"] >= t["gate_path_selectivity_min"],
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
        index for index, row in enumerate(truth) if row["replicate"] in FIT_REPLICATES and row["gauge"] == "identity"
    ]
    validation_indices = [index for index, row in enumerate(truth) if row["replicate"] in VALIDATION_REPLICATES]
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
        "redundancy_and_gate_camera_confirmed": passed,
        "single_ablation_false_negative_recovered_by_joint_ablation": passed,
        "known_truth_camera_stack_complete": passed,
        "learned_network_external_validity_confirmed": False,
        "free_micro_transformer_scan_authorized": passed,
        "pretrained_model_scan_authorized": False,
        "outcome": "controlled_redundancy_gate_camera_confirmed" if passed else "controlled_redundancy_gate_camera_not_confirmed",
        "claim_boundary": (
            "The confirmed object, if passed, distinguishes explicit known-truth bypass, single paths, saturated redundant paths, and a context-selected gate. "
            "It demonstrates why single-ablation necessity can be false negative, but it does not establish the same structures in independently learned networks or language models."
        ),
        "next_legal_task": (
            "Run one predeclared external-validity transfer on independently trained free micro-networks, with an abstain outcome for mechanisms not identifiable by the calibrated intervention family."
        ),
        "auto_continue": False,
        "auto_continue_reason": (
            "The controlled camera stack is complete, but a free-network transfer requires a new architecture/task protocol rather than a mechanical extension of the known-truth scaffold."
        ),
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
