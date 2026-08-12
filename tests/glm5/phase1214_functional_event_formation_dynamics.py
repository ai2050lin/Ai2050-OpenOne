#!/usr/bin/env python3
"""Compute-normalized formation dynamics of a behavioral quotient event.

Phase 1214 is not a longer rerun of Phase 1213.  It freezes disjoint tasks,
seeds, and architectures, trains every model to the same prespecified horizon,
and treats formation time and right censoring as the outcomes.  The external
target is the task's sealed input-output contract; no hidden state defines a
functional class.  At every frozen checkpoint we measure behavior and the
earliest layer jointly passing a predictive camera, same-function preservation,
and wrong-function transfer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1213_free_transformer_behavioral_quotient_event as p1213  # noqa: E402
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


PHASE = 1214
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1214_functional_event_formation_dynamics_audit.py")
OUT_ROOT = TEST_ROOT / "result/phase1214_functional_event_formation_dynamics"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
SOURCE1213 = TEST_ROOT / "result/phase1213_free_transformer_behavioral_quotient_event"

EXPECTED_1213_FINAL = "d48f58ca5dbb0c458b84d0d31fe4173c9a8af28c90b4b60d35fa212f7eb92736"
EXPECTED_1213_AUDIT = "60b980849be3f2f739ba67d2464d2dc02ad5b719eb5010df519f71148aff6ff7"

VALUE_COUNT = p1213.VALUE_COUNT
ROLES = p1213.ROLES
QUERIES = p1213.QUERIES
FUNCTION_QUERIES = p1213.FUNCTION_QUERIES
TEMPLATES = p1213.TEMPLATES
ALL_COMBINATIONS = p1213.ALL_COMBINATIONS
VOCAB_SIZE = p1213.VOCAB_SIZE
SEQUENCE_LENGTH = p1213.SEQUENCE_LENGTH
REPLICATES = 2
RIDGE = 1.0e-3

ARCHITECTURES = {
    "discovery": {
        "d5_w88": ModelConfig(5, 88, 4, 176, SEQUENCE_LENGTH, VOCAB_SIZE),
        "d7_w120": ModelConfig(7, 120, 4, 240, SEQUENCE_LENGTH, VOCAB_SIZE),
    },
    "confirmation": {
        "d9_w104": ModelConfig(9, 104, 4, 208, SEQUENCE_LENGTH, VOCAB_SIZE),
        "d11_w136": ModelConfig(11, 136, 4, 272, SEQUENCE_LENGTH, VOCAB_SIZE),
    },
}

TASKS = {
    "discovery": (
        {"name": "fd_a", "coefficients": (2, 1, 3), "bias": 1, "lexicon_seed": 1_214_101},
        {"name": "fd_b", "coefficients": (3, 3, 1), "bias": 0, "lexicon_seed": 1_214_103},
        {"name": "fd_c", "coefficients": (1, 1, 3), "bias": 2, "lexicon_seed": 1_214_107},
    ),
    "confirmation": (
        {"name": "fc_a", "coefficients": (3, 1, 1), "bias": 3, "lexicon_seed": 1_214_211},
        {"name": "fc_b", "coefficients": (1, 3, 3), "bias": 1, "lexicon_seed": 1_214_223},
        {"name": "fc_c", "coefficients": (3, 2, 3), "bias": 2, "lexicon_seed": 1_214_227},
    ),
}

TRAINING = {
    "learning_rate": 0.0015,
    "weight_decay": 0.001,
    "gradient_clip_norm": 1.0,
    "batch_size": 512,
    "evaluation_batch_size": 4096,
    "maximum_steps": 2400,
    "evaluation_interval": 100,
    "required_consecutive_passes": 2,
    "post_formation_stability_min": 0.80,
    "no_early_stopping": True,
}

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "behavior_accuracy_min": 1.0,
    "behavior_minimum_probability_min": 0.95,
    "camera_validation_accuracy_min": 0.95,
    "camera_holdout_accuracy_min": 0.95,
    "random_camera_accuracy_max": 0.25,
    "patch_same_baseline_match_min": 0.98,
    "patch_same_preservation_min": 0.98,
    "patch_wrong_eligible_fraction_min": 0.50,
    "patch_wrong_transfer_min": 0.90,
    "behavior_models_per_split_min": 8,
    "behavior_models_per_architecture_min": 3,
    "event_models_per_split_min": 6,
    "conditional_event_fraction_min": 0.80,
    "temporal_coupling_fraction_min": 0.75,
    "temporal_coupling_window_steps": 200,
    "median_absolute_delta_steps_max": 200,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch for {field}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters()))


def split_combinations(task: dict[str, Any]) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    return p1213.split_combinations(task)


def expected_signatures(task: dict[str, Any]) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    lexicon = p1213.make_lexicon(task)
    return {
        combination: tuple(p1213.target_slot(task, combination, query, lexicon) for query in FUNCTION_QUERIES)
        for combination in ALL_COMBINATIONS
    }


def candidate_ids(task: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(p1213.make_lexicon(task)["answers"], dtype=torch.long, device=device)


def predict_slots(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    task: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    device = next(model.parameters()).device
    candidates = candidate_ids(task, device)
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    finite: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), int(TRAINING["evaluation_batch_size"])):
            ids = inputs[start : start + int(TRAINING["evaluation_batch_size"])].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)[:, -1].float().index_select(-1, candidates)
            predictions.append(torch.argmax(logits, dim=-1).cpu())
            probabilities.append(torch.softmax(logits, dim=-1).cpu())
            finite.append(torch.isfinite(logits).to(torch.float32).cpu())
    return torch.cat(predictions).numpy(), torch.cat(probabilities).numpy(), float(torch.cat(finite).mean())


def evaluate_behavior(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    inputs, targets, metadata = p1213.build_examples(task, combinations, range(len(TEMPLATES)))
    predicted, probabilities, finite_fraction = predict_slots(model, inputs, task)
    truth = targets.numpy()
    correct = predicted == truth
    target_probability = probabilities[np.arange(len(truth)), truth]
    per_query = {
        query: float(np.mean(correct[np.asarray([row[2] == query for row in metadata], dtype=bool)]))
        for query in QUERIES
    }
    per_template = {
        str(template): float(np.mean(correct[np.asarray([row[1] == template for row in metadata], dtype=bool)]))
        for template in range(len(TEMPLATES))
    }
    return {
        "case_count": int(len(truth)),
        "accuracy": float(np.mean(correct)),
        "minimum_probability": float(np.min(target_probability)),
        "mean_probability": float(np.mean(target_probability)),
        "finite_fraction": finite_fraction,
        "per_query_accuracy": per_query,
        "per_template_accuracy": per_template,
    }


def behavior_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["finite_fraction"] >= THRESHOLDS["finite_fraction_min"]
        and metrics["accuracy"] >= THRESHOLDS["behavior_accuracy_min"]
        and metrics["minimum_probability"] >= THRESHOLDS["behavior_minimum_probability_min"]
        and min(metrics["per_query_accuracy"].values()) >= THRESHOLDS["behavior_accuracy_min"]
        and min(metrics["per_template_accuracy"].values()) >= THRESHOLDS["behavior_accuracy_min"]
    )


def collect_features_all_layers(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    templates: tuple[int, ...],
) -> tuple[list[np.ndarray], list[tuple[tuple[int, int, int], int, str]]]:
    inputs, _, metadata = p1213.build_examples(task, combinations, templates, FUNCTION_QUERIES)
    blocks: list[list[torch.Tensor]] = [[] for _ in range(model.config.layers + 1)]
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), int(TRAINING["evaluation_batch_size"])):
            ids = inputs[start : start + int(TRAINING["evaluation_batch_size"])].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, states = model(ids, return_states=True)
            for index, state in enumerate(states):
                blocks[index].append(state[:, -1].float().cpu())
    return [torch.cat(parts).numpy() for parts in blocks], list(metadata)


def target_vector(task: dict[str, Any], metadata: list[tuple[tuple[int, int, int], int, str]]) -> np.ndarray:
    lexicon = p1213.make_lexicon(task)
    return np.asarray(
        [p1213.target_slot(task, combination, query, lexicon) for combination, _, query in metadata],
        dtype=np.int64,
    )


def fit_decoder(
    features: np.ndarray,
    metadata: list[tuple[tuple[int, int, int], int, str]],
    task: dict[str, Any],
) -> np.ndarray:
    design = np.concatenate((features.astype(np.float64), np.ones((len(features), 1))), axis=1)
    targets = np.zeros((len(features), VALUE_COUNT), dtype=np.float64)
    truth = target_vector(task, metadata)
    targets[np.arange(len(features)), truth] = 1.0
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def decoder_metrics(
    features: np.ndarray,
    metadata: list[tuple[tuple[int, int, int], int, str]],
    task: dict[str, Any],
    weights: np.ndarray,
) -> dict[str, Any]:
    design = np.concatenate((features.astype(np.float64), np.ones((len(features), 1))), axis=1)
    predicted = np.argmax(design @ weights, axis=1)
    truth = target_vector(task, metadata)
    grouped: dict[tuple[tuple[int, int, int], int], dict[str, tuple[int, int]]] = defaultdict(dict)
    for estimate, target, (combination, template, query) in zip(predicted, truth, metadata):
        grouped[(combination, template)][query] = (int(estimate), int(target))
    combined = [
        all(values[query][0] == values[query][1] for query in FUNCTION_QUERIES)
        for values in grouped.values()
    ]
    return {
        "row_accuracy": float(np.mean(predicted == truth)),
        "combined_accuracy": float(np.mean(combined)),
        "combined_case_count": int(len(combined)),
    }


def initial_camera_controls(
    model: TinyCausalTransformer,
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    train, _ = split_combinations(task)
    fit_values, fit_meta = collect_features_all_layers(model, task, train, (0, 1))
    validation_values, validation_meta = collect_features_all_layers(model, task, train, (2, 3))
    rows: list[dict[str, Any]] = []
    for layer in range(model.config.layers + 1):
        weights = fit_decoder(fit_values[layer], fit_meta, task)
        rows.append(decoder_metrics(validation_values[layer], validation_meta, task, weights))
    return rows


def patch_metrics(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    layer: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    lexicon = p1213.make_lexicon(task)
    candidates = candidate_ids(task, device)
    same_baseline: list[bool] = []
    same_patch: list[bool] = []
    wrong_transfer: list[bool] = []
    eligible_total = 0
    case_total = 0
    model.eval()
    with torch.inference_mode():
        for recipient_template, donor_template in ((4, 0), (5, 1)):
            for query in FUNCTION_QUERIES:
                recipients = torch.tensor(
                    [p1213.encode(value, recipient_template, query, lexicon) for value in combinations],
                    dtype=torch.long,
                    device=device,
                )
                donors = torch.tensor(
                    [p1213.encode(value, donor_template, query, lexicon) for value in combinations],
                    dtype=torch.long,
                    device=device,
                )
                swapped = tuple((value[1], value[0], value[2]) for value in combinations)
                wrong_donors = torch.tensor(
                    [p1213.encode(value, donor_template, query, lexicon) for value in swapped],
                    dtype=torch.long,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recipient_logits, recipient_states = model(recipients, return_states=True)
                    donor_logits, donor_states = model(donors, return_states=True)
                    wrong_logits, wrong_states = model(wrong_donors, return_states=True)
                recipient_base = recipient_logits[:, -1].float().index_select(-1, candidates).argmax(-1)
                donor_base = donor_logits[:, -1].float().index_select(-1, candidates).argmax(-1)
                wrong_base = wrong_logits[:, -1].float().index_select(-1, candidates).argmax(-1)
                same_hidden = recipient_states[layer].clone()
                wrong_hidden = recipient_states[layer].clone()
                same_hidden[:, -1] = donor_states[layer][:, -1]
                wrong_hidden[:, -1] = wrong_states[layer][:, -1]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    same_prediction = model.forward_from(same_hidden, layer)[:, -1].float().index_select(-1, candidates).argmax(-1)
                    wrong_prediction = model.forward_from(wrong_hidden, layer)[:, -1].float().index_select(-1, candidates).argmax(-1)
                same_baseline.extend((recipient_base == donor_base).cpu().tolist())
                same_patch.extend((same_prediction == recipient_base).cpu().tolist())
                eligible = recipient_base != wrong_base
                wrong_transfer.extend((wrong_prediction[eligible] == wrong_base[eligible]).cpu().tolist())
                eligible_total += int(torch.sum(eligible).item())
                case_total += int(len(combinations))
    return {
        "case_count": case_total,
        "same_baseline_match": float(np.mean(same_baseline)),
        "same_preservation": float(np.mean(same_patch)),
        "wrong_eligible_count": eligible_total,
        "wrong_eligible_fraction": float(eligible_total / case_total),
        "wrong_transfer": float(np.mean(wrong_transfer)) if wrong_transfer else 0.0,
    }


def event_layer_pass(row: dict[str, Any]) -> bool:
    patch = row.get("patch")
    return bool(
        row["validation"]["combined_accuracy"] >= THRESHOLDS["camera_validation_accuracy_min"]
        and row["holdout"]["combined_accuracy"] >= THRESHOLDS["camera_holdout_accuracy_min"]
        and row["initial_validation"]["combined_accuracy"] <= THRESHOLDS["random_camera_accuracy_max"]
        and patch is not None
        and patch["same_baseline_match"] >= THRESHOLDS["patch_same_baseline_match_min"]
        and patch["same_preservation"] >= THRESHOLDS["patch_same_preservation_min"]
        and patch["wrong_eligible_fraction"] >= THRESHOLDS["patch_wrong_eligible_fraction_min"]
        and patch["wrong_transfer"] >= THRESHOLDS["patch_wrong_transfer_min"]
    )


def scan_checkpoint(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    initial_controls: list[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    train, holdout = split_combinations(task)
    train_behavior = evaluate_behavior(model, task, train)
    holdout_behavior = evaluate_behavior(model, task, holdout)
    current_behavior_pass = behavior_pass(train_behavior) and behavior_pass(holdout_behavior)

    fit_values, fit_meta = collect_features_all_layers(model, task, train, (0, 1))
    validation_values, validation_meta = collect_features_all_layers(model, task, train, (2, 3))
    holdout_values, holdout_meta = collect_features_all_layers(model, task, holdout, (4, 5))
    layers: list[dict[str, Any]] = []
    first_event: int | None = None
    for layer in range(model.config.layers + 1):
        weights = fit_decoder(fit_values[layer], fit_meta, task)
        row = {
            "layer": layer,
            "relative_depth": float(layer / model.config.layers),
            "validation": decoder_metrics(validation_values[layer], validation_meta, task, weights),
            "holdout": decoder_metrics(holdout_values[layer], holdout_meta, task, weights),
            "initial_validation": initial_controls[layer],
            "patch": None,
            "event_pass": False,
        }
        camera_eligible = bool(
            row["validation"]["combined_accuracy"] >= THRESHOLDS["camera_validation_accuracy_min"]
            and row["holdout"]["combined_accuracy"] >= THRESHOLDS["camera_holdout_accuracy_min"]
            and row["initial_validation"]["combined_accuracy"] <= THRESHOLDS["random_camera_accuracy_max"]
        )
        if camera_eligible and first_event is None:
            row["patch"] = patch_metrics(model, task, holdout, layer)
            row["event_pass"] = event_layer_pass(row)
            if row["event_pass"]:
                first_event = layer
        layers.append(row)
    return {
        "step": int(step),
        "samples_seen": int(step * TRAINING["batch_size"]),
        "tokens_seen": int(step * TRAINING["batch_size"] * SEQUENCE_LENGTH),
        "behavior_pass": current_behavior_pass,
        "train_behavior": train_behavior,
        "holdout_behavior": holdout_behavior,
        "event_pass": first_event is not None,
        "earliest_event_layer": first_event,
        "layers": layers,
    }


def first_stable_time(trajectory: list[dict[str, Any]], field: str) -> dict[str, Any]:
    required = int(TRAINING["required_consecutive_passes"])
    threshold = float(TRAINING["post_formation_stability_min"])
    for index in range(0, len(trajectory) - required + 1):
        window = trajectory[index : index + required]
        if all(bool(row[field]) for row in window):
            tail = trajectory[index:]
            stability = float(np.mean([bool(row[field]) for row in tail]))
            if stability >= threshold and bool(trajectory[-1][field]):
                return {
                    "status": "observed",
                    "step": int(trajectory[index]["step"]),
                    "interval_width_steps": int(TRAINING["evaluation_interval"]),
                    "post_formation_stability": stability,
                }
    return {
        "status": "right_censored",
        "lower_bound_step": int(TRAINING["maximum_steps"]),
        "post_formation_stability": None,
    }


def normalized_time(step: int, count: int) -> dict[str, Any]:
    return {
        "updates": int(step),
        "samples_seen": int(step * TRAINING["batch_size"]),
        "tokens_seen": int(step * TRAINING["batch_size"] * SEQUENCE_LENGTH),
        "parameter_token_proxy": int(step * TRAINING["batch_size"] * SEQUENCE_LENGTH * count),
    }


def summarize_trajectory(trajectory: list[dict[str, Any]], count: int) -> dict[str, Any]:
    tau_b = first_stable_time(trajectory, "behavior_pass")
    tau_e_raw = first_stable_time(trajectory, "event_pass")
    tau_e = dict(tau_e_raw)
    if tau_b["status"] != "observed":
        tau_e = {
            "status": "not_authorized_behavior_right_censored",
            "raw_event_status": tau_e_raw["status"],
            "raw_event_step": tau_e_raw.get("step"),
        }
    for value in (tau_b, tau_e_raw):
        if value["status"] == "observed":
            value["normalized"] = normalized_time(int(value["step"]), count)
    if tau_e.get("status") == "observed":
        tau_e["normalized"] = normalized_time(int(tau_e["step"]), count)
    both = tau_b["status"] == "observed" and tau_e.get("status") == "observed"
    delta = int(tau_e["step"] - tau_b["step"]) if both else None
    selected_layers = [
        int(row["earliest_event_layer"])
        for row in trajectory
        if row["event_pass"] and row["earliest_event_layer"] is not None
    ]
    return {
        "tau_B": tau_b,
        "tau_E_raw": tau_e_raw,
        "tau_E": tau_e,
        "delta_tau_steps": delta,
        "coupled_within_window": bool(
            both and abs(delta) <= int(THRESHOLDS["temporal_coupling_window_steps"])
        ),
        "event_layer_mode": max(set(selected_layers), key=selected_layers.count) if selected_layers else None,
        "endpoint_behavior_pass": bool(trajectory[-1]["behavior_pass"]),
        "endpoint_event_pass": bool(trajectory[-1]["event_pass"]),
    }


def model_seed(split: str, task_index: int, architecture_index: int, replicate: int) -> int:
    base = 1_214_300_000 if split == "discovery" else 1_214_700_000
    return base + task_index * 100_003 + architecture_index * 10_007 + replicate * 1_009


def run_id(split: str, task: dict[str, Any], architecture: str, replicate: int) -> str:
    return f"{split}__{task['name']}__{architecture}__r{replicate:02d}"


def checkpoint_payload(
    model: TinyCausalTransformer,
    config: ModelConfig,
    identifier: str,
    step: int,
    protocol_digest: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "run_id": identifier,
        "step": int(step),
        "protocol_digest": protocol_digest,
        "config": asdict(config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> TinyCausalTransformer:
    value = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyCausalTransformer(ModelConfig(**value["config"]))
    model.load_state_dict(value["state_dict"])
    return model.to(device)


def execute_run(
    split: str,
    task_index: int,
    architecture_index: int,
    replicate: int,
    device: torch.device,
) -> dict[str, Any]:
    protocol = verify_protocol()
    task = dict(TASKS[split][task_index])
    architecture, config = list(ARCHITECTURES[split].items())[architecture_index]
    identifier = run_id(split, task, architecture, replicate)
    run_root = OUT_ROOT / "runs" / split / identifier
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        existing = read_json(metrics_path)
        validate_digest(existing, "metrics_digest")
        if existing["protocol_digest"] != protocol["protocol_digest"]:
            raise RuntimeError(f"stale metrics for {identifier}")
        return existing

    seed = model_seed(split, task_index, architecture_index, replicate)
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    count = parameter_count(model)
    train_combinations, _ = split_combinations(task)
    train_inputs, train_targets, _ = p1213.build_examples(task, train_combinations, range(len(TEMPLATES)))
    candidates = candidate_ids(task, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        weight_decay=float(TRAINING["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 37)
    controls = initial_camera_controls(model, task)
    trajectory: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    def record(step: int, loss: float | None, gradient_norm: float | None) -> None:
        checkpoint_path = run_root / "checkpoints" / f"step_{step:04d}.pt"
        write_checkpoint(
            checkpoint_path,
            checkpoint_payload(model, config, identifier, step, protocol["protocol_digest"]),
        )
        scan = scan_checkpoint(model, task, controls, step)
        scan["loss"] = loss
        scan["gradient_norm"] = gradient_norm
        scan["parameter_token_proxy"] = int(scan["tokens_seen"] * count)
        trajectory.append(scan)
        checkpoint_manifest.append(
            {
                "step": int(step),
                "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(checkpoint_path),
            }
        )
        print(
            f"[{utc_now()}] {identifier} step={step} "
            f"B={int(scan['behavior_pass'])} E={int(scan['event_pass'])} "
            f"holdout={scan['holdout_behavior']['accuracy']:.4f} "
            f"pmin={scan['holdout_behavior']['minimum_probability']:.4f}",
            flush=True,
        )

    record(0, None, None)
    last_loss: float | None = None
    last_gradient: float | None = None
    for step in range(1, int(TRAINING["maximum_steps"]) + 1):
        model.train()
        indices = torch.randint(0, len(train_inputs), (int(TRAINING["batch_size"]),), generator=generator)
        batch_inputs = train_inputs[indices].to(device, non_blocking=True)
        batch_targets = train_targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_inputs)[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), batch_targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss in {identifier} at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(TRAINING["gradient_clip_norm"]))
        if not bool(torch.isfinite(torch.as_tensor(gradient))):
            raise RuntimeError(f"nonfinite gradient in {identifier} at step {step}")
        optimizer.step()
        last_loss = float(loss.item())
        last_gradient = float(gradient)
        if step % int(TRAINING["evaluation_interval"]) == 0:
            record(step, last_loss, last_gradient)

    summary = summarize_trajectory(trajectory, count)
    signatures = expected_signatures(task)
    metrics = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_id": identifier,
        "split": split,
        "task": task,
        "architecture": architecture,
        "config": asdict(config),
        "replicate": replicate,
        "seed": seed,
        "parameter_count": count,
        "functional_contract": {
            "class_count": len(set(signatures.values())),
            "signature_digest": digest({str(key): value for key, value in signatures.items()}),
        },
        "initial_camera_controls": controls,
        "trajectory": trajectory,
        "formation": summary,
        "checkpoint_manifest": checkpoint_manifest,
    }
    metrics["metrics_digest"] = digest(metrics)
    write_json(metrics_path, metrics)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def group_summary(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    behavior = [row for row in rows if row["formation"]["tau_B"]["status"] == "observed"]
    event = [row for row in behavior if row["formation"]["tau_E"]["status"] == "observed"]
    coupled = [row for row in event if row["formation"]["coupled_within_window"]]
    deltas = [int(row["formation"]["delta_tau_steps"]) for row in event]
    per_architecture: dict[str, Any] = {}
    for architecture in ARCHITECTURES[split]:
        subset = [row for row in rows if row["architecture"] == architecture]
        b_count = sum(row["formation"]["tau_B"]["status"] == "observed" for row in subset)
        e_count = sum(row["formation"]["tau_E"]["status"] == "observed" for row in subset)
        per_architecture[architecture] = {
            "run_count": len(subset),
            "behavior_observed": b_count,
            "event_observed": e_count,
            "behavior_gate": b_count >= int(THRESHOLDS["behavior_models_per_architecture_min"]),
        }
    behavior_gate = bool(
        len(behavior) >= int(THRESHOLDS["behavior_models_per_split_min"])
        and all(value["behavior_gate"] for value in per_architecture.values())
    )
    event_fraction = float(len(event) / len(behavior)) if behavior else 0.0
    event_gate = bool(
        behavior_gate
        and len(event) >= int(THRESHOLDS["event_models_per_split_min"])
        and event_fraction >= float(THRESHOLDS["conditional_event_fraction_min"])
    )
    coupling_fraction = float(len(coupled) / len(event)) if event else 0.0
    median_abs = float(np.median(np.abs(deltas))) if deltas else None
    coupling_gate = bool(
        event_gate
        and coupling_fraction >= float(THRESHOLDS["temporal_coupling_fraction_min"])
        and median_abs is not None
        and median_abs <= float(THRESHOLDS["median_absolute_delta_steps_max"])
    )
    return {
        "split": split,
        "run_count": len(rows),
        "behavior_observed": len(behavior),
        "behavior_right_censored": len(rows) - len(behavior),
        "event_observed_given_behavior": len(event),
        "event_right_censored_given_behavior": len(behavior) - len(event),
        "conditional_event_fraction": event_fraction,
        "coupled_count": len(coupled),
        "temporal_coupling_fraction": coupling_fraction,
        "delta_tau_steps": deltas,
        "median_delta_tau_steps": float(np.median(deltas)) if deltas else None,
        "median_absolute_delta_tau_steps": median_abs,
        "per_architecture": per_architecture,
        "behavior_gate": behavior_gate,
        "event_gate": event_gate,
        "coupling_gate": coupling_gate,
        "formation_dynamics_gate": bool(behavior_gate and event_gate and coupling_gate),
    }


def source_gate() -> dict[str, bool]:
    final = read_json(SOURCE1213 / "analysis/final.json")
    audit = read_json(SOURCE1213 / "audit/independent_result_audit.json")
    validate_digest(final, "final_digest")
    validate_digest(audit, "audit_digest")
    return {
        "phase1213_final_digest": final["final_digest"] == EXPECTED_1213_FINAL,
        "phase1213_audit_digest": audit["audit_digest"] == EXPECTED_1213_AUDIT,
        "phase1213_audit_passed": audit["all_checks_passed"] is True,
        "phase1213_overall_not_confirmed": final["claims"]["free_behavioral_quotient"] == "not_confirmed",
        "phase1213_auto_continue_was_false": final["auto_continue"] is False,
    }


def script_hashes() -> dict[str, str]:
    return {
        "phase1214_main": sha256_file(SCRIPT),
        "phase1214_audit": sha256_file(AUDIT_SCRIPT),
        "phase1213_source": sha256_file(Path(p1213.__file__)),
        "tiny_transformer_source": sha256_file(TEST_ROOT / "phase1146_learned_composition_benchmark.py"),
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "title": "Compute-normalized behavioral quotient event formation dynamics",
        "created_at": utc_now(),
        "source_gate": source_gate(),
        "script_hashes": script_hashes(),
        "scientific_object": {
            "tau_B": "first checkpoint in a two-checkpoint behavior-pass run that remains >=0.80 stable through the frozen endpoint",
            "tau_E": "first checkpoint in a two-checkpoint predictive-transfer-event run that remains >=0.80 stable through the frozen endpoint",
            "delta_tau": "tau_E - tau_B",
            "censoring": "absence by the frozen horizon is right censoring; tau_E is unauthorized when tau_B is censored",
            "target": "sealed task input-output contract, never a hidden-state cluster or posthoc label",
        },
        "tasks": {split: [dict(task) for task in values] for split, values in TASKS.items()},
        "architectures": {
            split: {name: asdict(config) for name, config in values.items()}
            for split, values in ARCHITECTURES.items()
        },
        "replicates": REPLICATES,
        "formal_run_count": sum(
            len(TASKS[split]) * len(ARCHITECTURES[split]) * REPLICATES for split in TASKS
        ),
        "training": TRAINING,
        "thresholds": THRESHOLDS,
        "checkpoint_steps": list(range(0, int(TRAINING["maximum_steps"]) + 1, int(TRAINING["evaluation_interval"]))),
        "normalization": {
            "updates": "optimizer steps",
            "samples_seen": "updates * batch_size",
            "tokens_seen": "updates * batch_size * sequence_length",
            "parameter_token_proxy": "tokens_seen * trainable_parameter_count; descriptive, not literal FLOPs",
        },
        "event_conjunction": [
            "validation camera",
            "heldout camera",
            "random-initialization negative camera",
            "same-function baseline match",
            "same-function patch preservation",
            "wrong-function eligible fraction",
            "wrong-function patch transfer",
        ],
        "claim_gate": {
            "discovery_and_confirmation": "behavior breadth AND conditional event breadth AND temporal coupling",
            "independent_result_audit_required": True,
            "failure_scope": "failure does not imply the event is absent in right-censored models",
        },
        "forbidden": [
            "continue any Phase1213 formal run",
            "extend the 2400-step horizon after seeing outcomes",
            "change learning rate or thresholds after preregistration",
            "drop a task, seed, architecture, or censored run",
            "treat tau_E as negative when tau_B is censored",
            "select only behavior-qualified runs and claim a population law",
            "start Qwen3, GLM4, or DS7B before both split gates and independent audit pass",
            "claim necessity, global minimality, natural semantics, or new mathematics",
        ],
        "authorized_next_on_pass": "Phase1215 time-resolved necessity and redundancy on new tasks",
        "authorized_next_on_failure": "stop this formation-law registry and retain typed censoring boundary",
    }


def preregister() -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        raise RuntimeError(f"protocol already exists: {PROTOCOL_PATH}")
    payload = protocol_payload()
    if not all(payload["source_gate"].values()):
        raise RuntimeError(f"source gate failed: {payload['source_gate']}")
    payload["protocol_digest"] = digest(payload)
    write_json(PROTOCOL_PATH, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL_PATH)
    validate_digest(value, "protocol_digest")
    if value["script_hashes"] != script_hashes():
        raise RuntimeError("frozen script hash mismatch")
    if not all(source_gate().values()):
        raise RuntimeError("source gate changed")
    return value


def require_preaudit() -> dict[str, Any]:
    value = read_json(PREAUDIT_PATH)
    validate_digest(value, "audit_digest")
    if not value["all_checks_passed"] or value["protocol_digest"] != verify_protocol()["protocol_digest"]:
        raise RuntimeError("independent preaudit is absent, stale, or failed")
    return value


def execute_split(split: str) -> list[dict[str, Any]]:
    require_preaudit()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    for task_index in range(len(TASKS[split])):
        for architecture_index in range(len(ARCHITECTURES[split])):
            for replicate in range(REPLICATES):
                rows.append(execute_run(split, task_index, architecture_index, replicate, device))
    return rows


def collect_run_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("discovery", "confirmation"):
        for task in TASKS[split]:
            for architecture in ARCHITECTURES[split]:
                for replicate in range(REPLICATES):
                    path = OUT_ROOT / "runs" / split / run_id(split, task, architecture, replicate) / "metrics.json"
                    if not path.exists():
                        raise RuntimeError(f"missing formal result: {path}")
                    row = read_json(path)
                    validate_digest(row, "metrics_digest")
                    rows.append(row)
    return rows


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    preaudit = require_preaudit()
    rows = collect_run_rows()
    summaries = {
        split: group_summary(split, [row for row in rows if row["split"] == split])
        for split in ("discovery", "confirmation")
    }
    experimental_gate = all(value["formation_dynamics_gate"] for value in summaries.values())
    manifest: list[dict[str, Any]] = []
    for row in rows:
        path = OUT_ROOT / "runs" / row["split"] / row["run_id"] / "metrics.json"
        manifest.append(
            {
                "run_id": row["run_id"],
                "metrics": str(path.relative_to(ROOT)).replace("\\", "/"),
                "metrics_sha256": sha256_file(path),
                "metrics_digest": row["metrics_digest"],
                "checkpoint_count": len(row["checkpoint_manifest"]),
            }
        )
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "preaudit_digest": preaudit["audit_digest"],
        "summaries": summaries,
        "run_manifest": manifest,
        "claims": {
            "formation_dynamics": "experimental_gate_passed_pending_independent_audit" if experimental_gate else "not_confirmed",
            "phase1213_rescue": "not_attempted",
            "necessity": "not_tested",
            "natural_language_transfer": "not_tested",
            "new_mathematics_required": "not_supported",
        },
        "candidate": {
            "identifier": "C1214",
            "status": "E2_pending_audit" if experimental_gate else "not_registered",
            "statement": "behavior and predictive-transfer event formation times are coupled under the frozen free-Transformer protocol",
        },
        "authorized_next": protocol["authorized_next_on_pass"] if experimental_gate else protocol["authorized_next_on_failure"],
        "auto_continue": bool(experimental_gate),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    return final


def selftest() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for split, tasks in TASKS.items():
        for task in tasks:
            train, holdout = split_combinations(task)
            checks[f"{split}_{task['name']}_balanced"] = len(train) == 384 and len(holdout) == 128
            signatures = expected_signatures(task)
            checks[f"{split}_{task['name']}_512_classes"] = len(set(signatures.values())) == 512
    checks["formal_runs_24"] = sum(
        len(TASKS[split]) * len(ARCHITECTURES[split]) * REPLICATES for split in TASKS
    ) == 24
    checks["checkpoint_count_25"] = len(
        range(0, int(TRAINING["maximum_steps"]) + 1, int(TRAINING["evaluation_interval"]))
    ) == 25
    checks["cuda_available"] = torch.cuda.is_available()
    return {"checks": checks, "all_checks_passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "preregister", "discovery", "confirmation", "finalize"))
    args = parser.parse_args()
    if args.command == "selftest":
        value = selftest()
    elif args.command == "preregister":
        value = preregister()
    elif args.command in ("discovery", "confirmation"):
        value = {"split": args.command, "run_count": len(execute_split(args.command))}
    else:
        value = finalize()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
