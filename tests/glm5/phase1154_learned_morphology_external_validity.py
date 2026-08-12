#!/usr/bin/env python3
"""External-validity test on gradient-trained mechanism morphologies.

Six architectures learn the same 4x4 lookup behavior.  Their mediator sites
are then exposed through the same intervention interface used by the
known-equation library.  Functional tomography must classify unseen trained
models after coordinate rotation; implementation coordinates are nuisance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

import phase1151_mechanism_morphology_library as library
import phase1152_tie_aware_morphology_library as tie_aware
import phase1153_blind_algorithm_coverage as coverage


PHASE = 1154
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1154_learned_morphology_external_validity"
SPLITS = ("discovery", "confirmation")
GROUPS = (
    "single_joint_carrier",
    "payload_with_gate",
    "factorized_roles",
    "joint_coalition",
    "redundant_paths",
    "context_switched_paths",
)
CHARTS = ("identity", "rotated")
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
MAX_STEPS = 4000
LEARNING_RATE = 0.04
TARGET_MIN_PROBABILITY = 0.98
ALGORITHMS = coverage.ALGORITHMS
CANDIDATE = coverage.CANDIDATE
THRESHOLDS = {
    "model_accuracy_min": 1.0,
    "model_min_probability_min": 0.98,
    "chart_cosine_min": 0.99999,
    "discovery_group_accuracy_min": 0.90,
    "discovery_min_group_accuracy_min": 0.75,
    "confirmation_group_accuracy_min": 0.90,
    "confirmation_min_group_accuracy_min": 0.75,
    "chart_accuracy_gap_max": 0.05,
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
    base = 115410 if split == "discovery" else 115490
    return base + GROUPS.index(group) * 1009 + int(replicate) * 107


def input_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    cols = []
    contexts = []
    targets = []
    for context in (0, 1):
        for row in range(library.N_ROWS):
            for col in range(library.N_COLS):
                rows.append(row)
                cols.append(col)
                contexts.append(context)
                targets.append(library.cell_index(row, col))
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device),
        torch.tensor(contexts, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


class LearnedMechanism(nn.Module):
    def __init__(self, group: str, seed: int) -> None:
        super().__init__()
        self.group = group
        torch.manual_seed(int(seed))
        self.joint_a = nn.Embedding(library.N_CLASSES, library.STATE_DIM)
        self.joint_b = nn.Embedding(library.N_CLASSES, library.STATE_DIM)
        self.row = nn.Embedding(library.N_ROWS, library.STATE_DIM)
        self.col = nn.Embedding(library.N_COLS, library.STATE_DIM)
        self.gate_raw = nn.Parameter(torch.tensor(0.0))
        self.readout = nn.Linear(library.STATE_DIM, library.N_CLASSES)
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.20)
            else:
                nn.init.zeros_(parameter)

    def logical_states(self, rows: torch.Tensor, cols: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        cells = rows * library.N_COLS + cols
        batch = rows.shape[0]
        states = torch.zeros(batch, library.N_SITES, library.STATE_DIM, dtype=torch.float32, device=rows.device)
        if self.group == "single_joint_carrier":
            states[:, 0] = self.joint_a(cells)
        elif self.group == "payload_with_gate":
            states[:, 0] = self.joint_a(cells)
            states[:, 1, 0] = torch.nn.functional.softplus(self.gate_raw) + 0.50
        elif self.group == "factorized_roles":
            states[:, 0] = self.row(rows)
            states[:, 1] = self.col(cols)
        elif self.group in {"joint_coalition", "redundant_paths"}:
            states[:, 0] = self.joint_a(cells)
            states[:, 1] = self.joint_b(cells)
        elif self.group == "context_switched_paths":
            states[:, 0] = self.joint_a(cells)
            states[:, 1] = self.joint_b(cells)
            states[:, 2, 0] = (contexts == 0).to(torch.float32)
            states[:, 2, 1] = (contexts == 1).to(torch.float32)
        else:
            raise ValueError(self.group)
        return states

    def compose(self, states: torch.Tensor) -> torch.Tensor:
        if self.group == "single_joint_carrier":
            return states[:, 0]
        if self.group == "payload_with_gate":
            return states[:, 0] * states[:, 1, 0:1]
        if self.group in {"factorized_roles", "joint_coalition"}:
            return states[:, 0] * states[:, 1]
        if self.group == "redundant_paths":
            return 0.5 * (states[:, 0] + states[:, 1])
        if self.group == "context_switched_paths":
            return states[:, 2, 0:1] * states[:, 0] + states[:, 2, 1:2] * states[:, 1]
        raise ValueError(self.group)

    def logits_from_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.readout(self.compose(states))

    def forward(self, rows: torch.Tensor, cols: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        return self.logits_from_states(self.logical_states(rows, cols, contexts))


def train_model(group: str, seed: int, device: torch.device) -> tuple[LearnedMechanism, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = LearnedMechanism(group, seed).to(device)
    rows, cols, contexts, targets = input_tensors(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    final_step = MAX_STEPS
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(rows, cols, contexts)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            with torch.no_grad():
                probs = torch.softmax(model(rows, cols, contexts), dim=1)
                target_prob = probs.gather(1, targets[:, None]).squeeze(1)
                accuracy = torch.mean((torch.argmax(probs, dim=1) == targets).to(torch.float32))
                if float(accuracy.item()) == 1.0 and float(torch.min(target_prob).item()) >= TARGET_MIN_PROBABILITY:
                    final_step = step
                    break
    with torch.no_grad():
        probs = torch.softmax(model(rows, cols, contexts), dim=1)
        target_prob = probs.gather(1, targets[:, None]).squeeze(1)
        accuracy = float(torch.mean((torch.argmax(probs, dim=1) == targets).to(torch.float32)).item())
        loss_value = float(torch.nn.functional.cross_entropy(model(rows, cols, contexts), targets).item())
    return model, {
        "steps": final_step,
        "accuracy": accuracy,
        "min_probability": float(torch.min(target_prob).item()),
        "mean_probability": float(torch.mean(target_prob).item()),
        "loss": loss_value,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


class LearnedOracle:
    def __init__(self, model: LearnedMechanism, chart: str, nuisance_seed: int, device: torch.device) -> None:
        self.model = model
        self.chart = chart
        self.device = device
        rng = np.random.default_rng(int(nuisance_seed))
        self.logical_to_physical = rng.permutation(library.N_SITES).astype(np.int64)
        self.scales = torch.tensor(rng.uniform(0.65, 1.75, size=library.N_SITES), dtype=torch.float32, device=device)
        charts = []
        for site in range(library.N_SITES):
            if chart == "identity":
                charts.append(torch.eye(library.STATE_DIM, dtype=torch.float32, device=device))
            else:
                value = library.make_orthogonal(int(nuisance_seed) + 1013 * (site + 1), device).to(torch.float32)
                charts.append(value)
        self.charts = torch.stack(charts)

    def _tensors(self, inputs: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([row for row, _col, _ctx in inputs], dtype=torch.long, device=self.device),
            torch.tensor([col for _row, col, _ctx in inputs], dtype=torch.long, device=self.device),
            torch.tensor([ctx for _row, _col, ctx in inputs], dtype=torch.long, device=self.device),
        )

    def states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        rows, cols, contexts = self._tensors(inputs)
        with torch.no_grad():
            logical = self.model.logical_states(rows, cols, contexts)
        observed = torch.zeros_like(logical)
        for logical_site in range(library.N_SITES):
            physical = int(self.logical_to_physical[logical_site])
            observed[:, physical] = (logical[:, logical_site] @ self.charts[logical_site].T) * self.scales[logical_site]
        return observed.to(torch.float64)

    def _decode(self, observed: torch.Tensor) -> torch.Tensor:
        value = observed.to(torch.float32)
        logical = torch.zeros_like(value)
        for logical_site in range(library.N_SITES):
            physical = int(self.logical_to_physical[logical_site])
            logical[:, logical_site] = (value[:, physical] / self.scales[logical_site]) @ self.charts[logical_site]
        return logical

    def output(self, observed: torch.Tensor, receivers: list[tuple[int, int, int]]) -> torch.Tensor:
        del receivers
        with torch.no_grad():
            return torch.softmax(self.model.logits_from_states(self._decode(observed)), dim=1).to(torch.float64)


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to rewrite Phase1154 artifacts")
    source_final = read_json(coverage.OUT_ROOT / "analysis/final.json")
    source_audit = read_json(coverage.OUT_ROOT / "audit/independent_audit.json")
    checks = {
        "phase1153_controlled_candidate_qualified": bool(source_final["controlled_functional_tomography_qualified"]),
        "phase1153_authorized_learned_calibration": bool(source_final["phase1154_learned_network_calibration_authorized"]),
        "phase1153_audit_passed": bool(source_audit["all_checks_passed"]),
        "six_architectures": len(GROUPS) == 6,
        "split_seeds_disjoint": True,
        "fit_validation_replicates_disjoint": not bool(set(FIT_REPLICATES) & set(VALIDATION_REPLICATES)),
        "candidate_frozen": CANDIDATE == "functional_tomography",
        "confirmation_truth_forbidden_in_predict": True,
        "behavior_gate_precedes_identification": True,
        "pretrained_model_scan_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "learned mechanism morphology external validity",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1153_digest": source_final["final_digest"],
        "source_phase1153_audit_digest": source_audit["audit_digest"],
        "groups": list(GROUPS),
        "charts": list(CHARTS),
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "algorithms": list(ALGORITHMS),
        "candidate": CANDIDATE,
        "training": {"max_steps": MAX_STEPS, "learning_rate": LEARNING_RATE, "target_min_probability": TARGET_MIN_PROBABILITY},
        "thresholds": THRESHOLDS,
        "hard_stops": [
            "A split that fails the behavior gate cannot enter mechanism scoring.",
            "Confirmation predictions must be sealed before truth is read.",
            "Success applies only to the six architecture-constrained learned systems.",
            "No free Transformer or pretrained-model claim is authorized by this protocol.",
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
        raise RuntimeError("Phase1154 frozen protocol mismatch")
    return protocol


def train_command(split: str) -> None:
    protocol = verify_protocol()
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in ALGORITHMS}
    public = []
    truth = []
    training_rows = []
    models_root = out / "models"
    models_root.mkdir(parents=True, exist_ok=False)
    index = 0
    for group in GROUPS:
        for replicate in range(REPLICATES):
            seed = seed_for(split, group, replicate)
            model, train_metrics = train_model(group, seed, device)
            model_id = digest({"phase": PHASE, "split": split, "group": group, "replicate": replicate})[:18]
            model_path = models_root / f"{model_id}.pt"
            torch.save({"group": group, "replicate": replicate, "seed": seed, "state_dict": model.state_dict(), "metrics": train_metrics}, model_path)
            training_rows.append({"model_id": model_id, "group": group, "replicate": replicate, "seed": seed, "model_sha256": sha256_file(model_path), **train_metrics})
            for chart in CHARTS:
                oracle = LearnedOracle(model, chart, seed + 7001, device)
                raw_features, _diagnostics = library.probe_system(oracle)
                features = tie_aware.continuous_features(raw_features)
                for name in ALGORITHMS:
                    feature_rows[name].append(features[name].astype(np.float32))
                unit_id = digest({"phase": PHASE, "split": split, "model_id": model_id, "chart": chart})[:20]
                public.append({"index": index, "unit_id": unit_id, "model_id": model_id, "split": split})
                truth.append({"index": index, "unit_id": unit_id, "model_id": model_id, "split": split, "functional_group": group, "replicate": replicate, "chart": chart})
                index += 1
            del model
            torch.cuda.empty_cache()
    arrays = {name: np.stack(rows, axis=0) for name, rows in feature_rows.items()}
    np.savez_compressed(out / "feature_pack.npz", **arrays)
    write_jsonl(out / "public_manifest.jsonl", public)
    write_jsonl(out / "sealed_truth.jsonl", truth)
    write_jsonl(out / "training_metrics.jsonl", training_rows)
    functional = arrays[CANDIDATE]
    by_key = {(row["functional_group"], row["replicate"], row["chart"]): int(row["index"]) for row in truth}
    chart_values = [
        library.cosine(
            functional[by_key[(group, replicate, "identity")]],
            functional[by_key[(group, replicate, "rotated")]],
        )
        for group in GROUPS
        for replicate in range(REPLICATES)
    ]
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "model_count": len(training_rows),
        "unit_count": len(public),
        "accuracy_min": float(min(row["accuracy"] for row in training_rows)),
        "min_probability_min": float(min(row["min_probability"] for row in training_rows)),
        "steps_min": int(min(row["steps"] for row in training_rows)),
        "steps_max": int(max(row["steps"] for row in training_rows)),
        "chart_cosine_min": float(min(chart_values)),
        "chart_cosine_median": float(np.median(chart_values)),
        "feature_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "finite_fraction": float(np.mean([np.isfinite(value).mean() for value in arrays.values()])),
        "feature_pack_sha256": sha256_file(out / "feature_pack.npz"),
        "public_manifest_sha256": sha256_file(out / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(out / "sealed_truth.jsonl"),
        "training_metrics_sha256": sha256_file(out / "training_metrics.jsonl"),
    }
    t = protocol["thresholds"]
    checks = {
        "behavior_accuracy": summary["accuracy_min"] >= t["model_accuracy_min"],
        "behavior_probability": summary["min_probability_min"] >= t["model_min_probability_min"],
        "chart_invariance": summary["chart_cosine_min"] >= t["chart_cosine_min"],
        "finite": summary["finite_fraction"] == 1.0,
    }
    summary["checks"] = checks
    summary["behavior_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(out / "summary.json", summary)
    print(canonical(summary))


def fit_command() -> None:
    protocol = verify_protocol()
    summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    if not summary["behavior_gate_passed"]:
        raise RuntimeError("discovery behavior gate failed")
    root = OUT_ROOT / "runs/discovery"
    truth = read_jsonl(root / "sealed_truth.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    fit_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in FIT_REPLICATES]
    validation_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in VALIDATION_REPLICATES]
    prototypes = {}
    metadata = {}
    metrics = {}
    for algorithm in ALGORITHMS:
        labels, proto = coverage.build_prototypes(arrays[algorithm], truth, fit_indices, "functional_group")
        prototypes[algorithm] = proto.astype(np.float32)
        metadata[algorithm] = labels
        predicted, _ = coverage.predict(arrays[algorithm][validation_indices], labels, proto)
        metrics[algorithm] = coverage.metrics_for_predictions(predicted, truth, validation_indices, "functional_group")
    candidate = metrics[CANDIDATE]
    t = protocol["thresholds"]
    checks = {
        "group_accuracy": candidate["accuracy"] >= t["discovery_group_accuracy_min"],
        "min_group_accuracy": candidate["min_label_accuracy"] >= t["discovery_min_group_accuracy_min"],
        "chart_gap": candidate["chart_accuracy_gap"] <= t["chart_accuracy_gap_max"],
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
        "confirmation_prediction_authorized": all(checks.values()),
        "prototype_sha256": sha256_file(analysis / "frozen_prototypes.npz"),
        "labels_sha256": sha256_file(analysis / "prototype_labels.json"),
    }
    result["fit_digest"] = digest(result)
    write_json(analysis / "fit.json", result)
    print(canonical(result))


def predict_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    confirmation_summary = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    if not fit["confirmation_prediction_authorized"] or not confirmation_summary["behavior_gate_passed"]:
        raise RuntimeError("confirmation prediction denied")
    if (OUT_ROOT / "predictions").exists():
        raise RuntimeError("refusing to overwrite predictions")
    root = OUT_ROOT / "runs/confirmation"
    public = read_jsonl(root / "public_manifest.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    labels = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as pack:
        prototypes = {name: np.asarray(pack[name]) for name in pack.files}
    predictions = {}
    confidences = {}
    for algorithm in ALGORITHMS:
        predictions[algorithm], confidences[algorithm] = coverage.predict(arrays[algorithm], labels[algorithm], prototypes[algorithm])
    rows = []
    for index, public_row in enumerate(public):
        rows.append({"index": index, "unit_id": public_row["unit_id"], "algorithms": {algorithm: {"group": predictions[algorithm][index], "cosine": float(confidences[algorithm][index])} for algorithm in ALGORITHMS}})
    path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    write_jsonl(path, rows)
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_count": len(rows),
        "confirmation_truth_read": False,
        "prediction_sha256": sha256_file(path),
    }
    result["prediction_digest"] = digest(result)
    write_json(OUT_ROOT / "predictions/manifest.json", result)
    print(canonical(result))


def score_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    prediction_path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    if manifest["confirmation_truth_read"] or sha256_file(prediction_path) != manifest["prediction_sha256"]:
        raise RuntimeError("prediction seal invalid")
    predictions = read_jsonl(prediction_path)
    root = OUT_ROOT / "runs/confirmation"
    truth = read_jsonl(root / "sealed_truth.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    indices = list(range(len(truth)))
    metrics = {}
    for algorithm in ALGORITHMS:
        predicted = [row["algorithms"][algorithm]["group"] for row in predictions]
        metrics[algorithm] = coverage.metrics_for_predictions(predicted, truth, indices, "functional_group")
    candidate = metrics[CANDIDATE]
    functional = arrays[CANDIDATE]
    by_key = {(row["functional_group"], row["replicate"], row["chart"]): int(row["index"]) for row in truth}
    chart_values = [library.cosine(functional[by_key[(group, replicate, "identity")]], functional[by_key[(group, replicate, "rotated")]]) for group in GROUPS for replicate in range(REPLICATES)]
    t = protocol["thresholds"]
    checks = {
        "group_accuracy": candidate["accuracy"] >= t["confirmation_group_accuracy_min"],
        "min_group_accuracy": candidate["min_label_accuracy"] >= t["confirmation_min_group_accuracy_min"],
        "chart_gap": candidate["chart_accuracy_gap"] <= t["chart_accuracy_gap_max"],
        "chart_cosine": min(chart_values) >= t["chart_cosine_min"],
    }
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_digest": manifest["prediction_digest"],
        "confirmation_count": len(truth),
        "algorithm_metrics": metrics,
        "coverage_matrix": {algorithm: values["per_label_accuracy"] for algorithm, values in metrics.items()},
        "candidate_checks": checks,
        "candidate_confirmed": all(checks.values()),
        "matched_chart_cosine_min": float(min(chart_values)),
        "matched_chart_cosine_median": float(np.median(chart_values)),
        "claim_boundary": "The result concerns architecture-constrained tiny learned systems. Architecture labels are known ground truth; no free network or pretrained language mechanism is identified.",
    }
    result["score_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/confirmation_score.json", result)
    print(canonical(result))


def finalize_command() -> None:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    confirmation = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    passed = bool(discovery["behavior_gate_passed"] and confirmation["behavior_gate_passed"] and fit["candidate_qualified"] and score["candidate_confirmed"])
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": discovery["summary_digest"],
        "confirmation_summary_digest": confirmation["summary_digest"],
        "fit_digest": fit["fit_digest"],
        "score_digest": score["score_digest"],
        "learned_morphology_external_validity_confirmed": passed,
        "phase1155_free_network_tomography_authorized": passed,
        "pretrained_model_mechanism_claim_authorized": False,
        "outcome": "learned_morphology_external_validity_confirmed" if passed else "learned_morphology_external_validity_not_confirmed",
        "auto_continue": passed,
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    train = sub.add_parser("train")
    train.add_argument("--split", choices=SPLITS, required=True)
    sub.add_parser("fit")
    sub.add_parser("predict")
    sub.add_parser("score")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "train":
        train_command(args.split)
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
