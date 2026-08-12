#!/usr/bin/env python3
"""Free TinyTransformer external-validity test of a behavioral quotient event.

The target quotient is frozen from future observable behavior, never from a
hidden-state readout.  A freely trained causal Transformer must retrieve three
role-bound values for held-out value combinations.  Six record permutations
are nuisance realizations of the same function.  The camera searches for the
first layer where a linear response probe predicts the sealed behavior class
and a query-state patch transfers the donor behavior.  Discovery and
confirmation use disjoint tasks, seeds, depths, and widths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
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

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


PHASE = 1213
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1213_free_transformer_behavioral_quotient_event_audit.py")
OUT_ROOT = TEST_ROOT / "result/phase1213_free_transformer_behavioral_quotient_event"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
SOURCE1212 = TEST_ROOT / "result/phase1212_functional_target_quotient_minimal_observation"

EXPECTED_1212_FINAL = "b8ef2fb0c992c2da6969637f26e25b0fce017815ac57652c455fcecf14f88cb3"
EXPECTED_1212_AUDIT = "7afe0b311901a4ba6c42e50e159901bcfc34be11d7b034bf0aa8f12a236fdd86"

VALUE_COUNT = 8
ROLES = ("row", "column", "context")
QUERIES = ("row", "column", "context", "null")
FUNCTION_QUERIES = ("row", "column", "context")
TEMPLATES = tuple(itertools.permutations(ROLES))
ALL_COMBINATIONS = tuple(itertools.product(range(VALUE_COUNT), repeat=3))
VOCAB_SIZE = 48
SEQUENCE_LENGTH = 8
REPLICATES = 2
RIDGE = 1.0e-3
FUTURE_STEPS = 32

ARCHITECTURES = {
    "discovery": {
        "d6_w96": ModelConfig(6, 96, 4, 192, SEQUENCE_LENGTH, VOCAB_SIZE),
        "d8_w128": ModelConfig(8, 128, 4, 256, SEQUENCE_LENGTH, VOCAB_SIZE),
    },
    "confirmation": {
        "d10_w112": ModelConfig(10, 112, 4, 224, SEQUENCE_LENGTH, VOCAB_SIZE),
        "d12_w144": ModelConfig(12, 144, 4, 288, SEQUENCE_LENGTH, VOCAB_SIZE),
    },
}

TASKS = {
    "discovery": (
        {"name": "d_a", "coefficients": (1, 2, 3), "bias": 0, "lexicon_seed": 1_213_101},
        {"name": "d_b", "coefficients": (3, 1, 2), "bias": 1, "lexicon_seed": 1_213_103},
        {"name": "d_c", "coefficients": (1, 3, 2), "bias": 2, "lexicon_seed": 1_213_107},
    ),
    "confirmation": (
        {"name": "c_a", "coefficients": (3, 2, 1), "bias": 1, "lexicon_seed": 1_213_211},
        {"name": "c_b", "coefficients": (2, 3, 1), "bias": 2, "lexicon_seed": 1_213_223},
        {"name": "c_c", "coefficients": (1, 2, 1), "bias": 3, "lexicon_seed": 1_213_227},
    ),
}

TRAINING = {
    "learning_rate": 0.0015,
    "weight_decay": 0.001,
    "gradient_clip_norm": 1.0,
    "batch_size": 512,
    "evaluation_batch_size": 2048,
    "maximum_steps": 1200,
    "minimum_steps": 400,
    "evaluation_interval": 100,
    "required_consecutive_passes": 2,
    "future_steps": FUTURE_STEPS,
}

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "behavior_accuracy_min": 1.0,
    "behavior_minimum_probability_min": 0.95,
    "signature_template_invariance_min": 1.0,
    "signature_class_count": VALUE_COUNT**3,
    "future_signature_stability_min": 1.0,
    "camera_validation_accuracy_min": 0.95,
    "camera_holdout_accuracy_min": 0.95,
    "camera_future_accuracy_min": 0.90,
    "random_camera_accuracy_max": 0.25,
    "patch_same_preservation_min": 0.98,
    "patch_wrong_transfer_min": 0.90,
    "state_rms_distance_min": 1.0e-4,
    "bag_majority_accuracy_max": 0.25,
    "qualified_models_per_split_min": 10,
    "qualified_models_per_architecture_min": 5,
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


def task_holdout(task: dict[str, Any], combination: tuple[int, int, int]) -> bool:
    value = sum(int(a) * int(b) for a, b in zip(task["coefficients"], combination)) + int(task["bias"])
    return value % 4 == 0


def split_combinations(task: dict[str, Any]) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    train = tuple(value for value in ALL_COMBINATIONS if not task_holdout(task, value))
    holdout = tuple(value for value in ALL_COMBINATIONS if task_holdout(task, value))
    if len(train) != 384 or len(holdout) != 128:
        raise RuntimeError(f"unbalanced combination split for {task['name']}: {len(train)}/{len(holdout)}")
    return train, holdout


def make_lexicon(task: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(task["lexicon_seed"]))
    ids = rng.permutation(np.arange(1, VOCAB_SIZE)).tolist()
    answer_permutation = rng.permutation(VALUE_COUNT).tolist()
    return {
        "bos": 0,
        "roles": {role: int(ids[index]) for index, role in enumerate(ROLES)},
        "values": [int(value) for value in ids[3:11]],
        "queries": {query: int(ids[11 + index]) for index, query in enumerate(QUERIES)},
        "answers": [int(value) for value in ids[15:23]],
        "answer_permutation": [int(value) for value in answer_permutation],
    }


def raw_target(combination: tuple[int, int, int], query: str) -> int:
    if query == "null":
        return 0
    return int(combination[ROLES.index(query)])


def target_slot(
    task: dict[str, Any],
    combination: tuple[int, int, int],
    query: str,
    lexicon: dict[str, Any] | None = None,
) -> int:
    value = lexicon or make_lexicon(task)
    return int(value["answer_permutation"][raw_target(combination, query)])


def encode(
    combination: tuple[int, int, int],
    template_index: int,
    query: str,
    lexicon: dict[str, Any],
) -> list[int]:
    values = dict(zip(ROLES, combination))
    tokens = [int(lexicon["bos"])]
    for role in TEMPLATES[int(template_index)]:
        tokens.extend((int(lexicon["roles"][role]), int(lexicon["values"][values[role]])))
    tokens.append(int(lexicon["queries"][query]))
    return tokens


def build_examples(
    task: dict[str, Any],
    combinations: Iterable[tuple[int, int, int]],
    templates: Iterable[int],
    queries: Iterable[str] = QUERIES,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[tuple[int, int, int], int, str]]]:
    lexicon = make_lexicon(task)
    rows: list[list[int]] = []
    targets: list[int] = []
    metadata: list[tuple[tuple[int, int, int], int, str]] = []
    for template in templates:
        for combination in combinations:
            for query in queries:
                rows.append(encode(combination, int(template), query, lexicon))
                targets.append(target_slot(task, combination, query, lexicon))
                metadata.append((combination, int(template), query))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(targets, dtype=torch.long), metadata


def candidate_ids(task: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(make_lexicon(task)["answers"], dtype=torch.long, device=device)


def predict_slots(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    task: dict[str, Any],
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    device = next(model.parameters()).device
    batch_size = int(batch_size or TRAINING["evaluation_batch_size"])
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    finite: list[torch.Tensor] = []
    candidates = candidate_ids(task, device)
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            ids = inputs[start : start + batch_size].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)[:, -1].float().index_select(-1, candidates)
            predictions.append(torch.argmax(logits, dim=-1).cpu())
            probabilities.append(torch.softmax(logits, dim=-1).cpu())
            finite.append(torch.isfinite(logits).to(torch.float32).cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(probabilities).numpy(),
        float(torch.cat(finite).mean().item()),
    )


def evaluate_behavior(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    inputs, targets, metadata = build_examples(task, combinations, range(len(TEMPLATES)))
    predicted, probabilities, finite_fraction = predict_slots(model, inputs, task)
    truth = targets.numpy()
    correct = predicted == truth
    per_query: dict[str, float] = {}
    per_template: dict[str, float] = {}
    for query in QUERIES:
        mask = np.asarray([value[2] == query for value in metadata], dtype=bool)
        per_query[query] = float(np.mean(correct[mask]))
    for template in range(len(TEMPLATES)):
        mask = np.asarray([value[1] == template for value in metadata], dtype=bool)
        per_template[str(template)] = float(np.mean(correct[mask]))
    target_probability = probabilities[np.arange(len(truth)), truth]
    return {
        "case_count": int(len(truth)),
        "accuracy": float(np.mean(correct)),
        "minimum_probability": float(np.min(target_probability)),
        "mean_probability": float(np.mean(target_probability)),
        "finite_fraction": finite_fraction,
        "per_query_accuracy": per_query,
        "per_template_accuracy": per_template,
    }


def behavior_qualified(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["finite_fraction"] >= THRESHOLDS["finite_fraction_min"]
        and metrics["accuracy"] >= THRESHOLDS["behavior_accuracy_min"]
        and metrics["minimum_probability"] >= THRESHOLDS["behavior_minimum_probability_min"]
        and min(metrics["per_query_accuracy"].values()) >= THRESHOLDS["behavior_accuracy_min"]
        and min(metrics["per_template_accuracy"].values()) >= THRESHOLDS["behavior_accuracy_min"]
    )


def model_seed(split: str, task_index: int, architecture_index: int, replicate: int) -> int:
    base = 1_213_300_000 if split == "discovery" else 1_213_700_000
    return base + task_index * 100_003 + architecture_index * 10_007 + replicate * 1_009


def run_id(split: str, task: dict[str, Any], architecture: str, replicate: int) -> str:
    return f"{split}__{task['name']}__{architecture}__r{replicate:02d}"


def train_model(
    config: ModelConfig,
    task: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[TinyCausalTransformer, torch.optim.Optimizer, torch.Generator, dict[str, Any]]:
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    train_combinations, holdout_combinations = split_combinations(task)
    inputs, targets, _ = build_examples(task, train_combinations, range(len(TEMPLATES)))
    candidates = candidate_ids(task, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        weight_decay=float(TRAINING["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 37)
    logs: list[dict[str, Any]] = []
    consecutive = 0
    final_step = 0
    for step in range(1, int(TRAINING["maximum_steps"]) + 1):
        model.train()
        indices = torch.randint(0, len(inputs), (int(TRAINING["batch_size"]),), generator=generator)
        batch_inputs = inputs[indices].to(device, non_blocking=True)
        batch_targets = targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_inputs)[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), batch_targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(TRAINING["gradient_clip_norm"]))
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"nonfinite gradient at step {step}")
        optimizer.step()
        final_step = step
        if step % int(TRAINING["evaluation_interval"]) == 0:
            train_metrics = evaluate_behavior(model, task, train_combinations)
            holdout_metrics = evaluate_behavior(model, task, holdout_combinations)
            qualified = behavior_qualified(train_metrics) and behavior_qualified(holdout_metrics)
            consecutive = consecutive + 1 if qualified else 0
            logs.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "gradient_norm": float(gradient_norm),
                    "train_accuracy": train_metrics["accuracy"],
                    "train_minimum_probability": train_metrics["minimum_probability"],
                    "holdout_accuracy": holdout_metrics["accuracy"],
                    "holdout_minimum_probability": holdout_metrics["minimum_probability"],
                    "qualified": qualified,
                }
            )
            if step >= int(TRAINING["minimum_steps"]) and consecutive >= int(TRAINING["required_consecutive_passes"]):
                break
    train_metrics = evaluate_behavior(model, task, train_combinations)
    holdout_metrics = evaluate_behavior(model, task, holdout_combinations)
    return model, optimizer, generator, {
        "steps": final_step,
        "consecutive_passes": consecutive,
        "qualified": bool(
            behavior_qualified(train_metrics)
            and behavior_qualified(holdout_metrics)
            and consecutive >= int(TRAINING["required_consecutive_passes"])
        ),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "logs": logs,
    }


def continue_training(
    model: TinyCausalTransformer,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    task: dict[str, Any],
    device: torch.device,
) -> None:
    train_combinations, _ = split_combinations(task)
    inputs, targets, _ = build_examples(task, train_combinations, range(len(TEMPLATES)))
    candidates = candidate_ids(task, device)
    for _ in range(FUTURE_STEPS):
        indices = torch.randint(0, len(inputs), (int(TRAINING["batch_size"]),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs[indices].to(device))[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), targets[indices].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(TRAINING["gradient_clip_norm"]))
        optimizer.step()


def signature_map(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...] = ALL_COMBINATIONS,
    templates: tuple[int, ...] = (4, 5),
) -> tuple[dict[tuple[int, int, int], tuple[int, int, int]], dict[str, Any]]:
    inputs, _, metadata = build_examples(task, combinations, templates, FUNCTION_QUERIES)
    predicted, _, finite_fraction = predict_slots(model, inputs, task)
    grouped: dict[tuple[int, int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for prediction, (combination, _, query) in zip(predicted.tolist(), metadata):
        grouped[combination][query].append(int(prediction))
    signatures: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    invariant = 0
    for combination in combinations:
        values = grouped[combination]
        stable = all(len(set(values[query])) == 1 for query in FUNCTION_QUERIES)
        invariant += int(stable)
        signatures[combination] = tuple(int(values[query][0]) for query in FUNCTION_QUERIES)
    return signatures, {
        "combination_count": len(combinations),
        "template_invariance_fraction": float(invariant / len(combinations)),
        "class_count": len(set(signatures.values())),
        "finite_fraction": finite_fraction,
        "signature_digest": digest({str(key): value for key, value in signatures.items()}),
    }


def collect_response_features(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    templates: tuple[int, ...],
    layer: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    values, combinations_by_row = collect_response_features_all_layers(
        model, task, combinations, templates
    )
    return values[layer], combinations_by_row


def collect_response_features_all_layers(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    templates: tuple[int, ...],
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    device = next(model.parameters()).device
    feature_blocks: list[list[np.ndarray]] = [[] for _ in range(model.config.layers + 1)]
    reference: list[tuple[int, int, int]] | None = None
    model.eval()
    for query in FUNCTION_QUERIES:
        inputs, _, metadata = build_examples(task, combinations, templates, (query,))
        parts: list[list[torch.Tensor]] = [[] for _ in range(model.config.layers + 1)]
        with torch.inference_mode():
            for start in range(0, len(inputs), int(TRAINING["evaluation_batch_size"])):
                ids = inputs[start : start + int(TRAINING["evaluation_batch_size"])].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, states = model(ids, return_states=True)
                for state_index, state in enumerate(states):
                    parts[state_index].append(state[:, -1].float().cpu())
        for state_index in range(model.config.layers + 1):
            feature_blocks[state_index].append(torch.cat(parts[state_index]).numpy())
        labels = [value[0] for value in metadata]
        if reference is None:
            reference = labels
        elif reference != labels:
            raise RuntimeError("response feature ordering mismatch")
    return [np.concatenate(value, axis=1) for value in feature_blocks], list(reference or [])


def fit_decoder(
    features: np.ndarray,
    combinations: list[tuple[int, int, int]],
    signatures: dict[tuple[int, int, int], tuple[int, int, int]],
) -> list[np.ndarray]:
    design = np.concatenate((features.astype(np.float64), np.ones((len(features), 1))), axis=1)
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    target_blocks: list[np.ndarray] = []
    for coordinate in range(3):
        targets = np.zeros((len(features), VALUE_COUNT), dtype=np.float64)
        targets[np.arange(len(features)), [signatures[value][coordinate] for value in combinations]] = 1.0
        target_blocks.append(targets)
    combined = np.linalg.solve(gram, design.T @ np.concatenate(target_blocks, axis=1))
    return [combined[:, index * VALUE_COUNT : (index + 1) * VALUE_COUNT] for index in range(3)]


def decoder_accuracy(
    features: np.ndarray,
    combinations: list[tuple[int, int, int]],
    signatures: dict[tuple[int, int, int], tuple[int, int, int]],
    weights: list[np.ndarray],
) -> dict[str, Any]:
    design = np.concatenate((features.astype(np.float64), np.ones((len(features), 1))), axis=1)
    predicted = np.stack([np.argmax(design @ value, axis=1) for value in weights], axis=1)
    truth = np.asarray([signatures[value] for value in combinations], dtype=np.int64)
    coordinate = [float(np.mean(predicted[:, index] == truth[:, index])) for index in range(3)]
    return {"combined_accuracy": float(np.mean(np.all(predicted == truth, axis=1))), "coordinate_accuracy": coordinate}


def camera_for_layer(
    model: TinyCausalTransformer,
    initial_model: TinyCausalTransformer,
    task: dict[str, Any],
    signatures: dict[tuple[int, int, int], tuple[int, int, int]],
    layer: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    train, holdout = split_combinations(task)
    fit_features, fit_combinations = collect_response_features(model, task, train, (0, 1), layer)
    weights = fit_decoder(fit_features, fit_combinations, signatures)
    validation = decoder_accuracy(*collect_response_features(model, task, train, (2, 3), layer), signatures, weights)
    heldout = decoder_accuracy(*collect_response_features(model, task, holdout, (4, 5), layer), signatures, weights)
    initial_fit, initial_combinations = collect_response_features(initial_model, task, train, (0, 1), layer)
    initial_weights = fit_decoder(initial_fit, initial_combinations, signatures)
    initial_validation = decoder_accuracy(
        *collect_response_features(initial_model, task, train, (2, 3), layer), signatures, initial_weights
    )
    initial_holdout = decoder_accuracy(
        *collect_response_features(initial_model, task, holdout, (4, 5), layer), signatures, initial_weights
    )
    return {
        "layer": layer,
        "validation": validation,
        "heldout": heldout,
        "initial_validation": initial_validation,
        "initial_holdout": initial_holdout,
    }, weights


def query_patch_metrics(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    recipient_templates: tuple[int, int],
    donor_templates: tuple[int, int],
    layer: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    lexicon = make_lexicon(task)
    candidates = candidate_ids(task, device)
    same_results: list[bool] = []
    transfer_results: list[bool] = []
    same_count = 0
    eligible_count = 0
    model.eval()
    with torch.inference_mode():
        for recipient_template, donor_template in zip(recipient_templates, donor_templates):
            for query in FUNCTION_QUERIES:
                recipients = torch.tensor(
                    [encode(value, recipient_template, query, lexicon) for value in combinations],
                    dtype=torch.long,
                    device=device,
                )
                donors = torch.tensor(
                    [encode(value, donor_template, query, lexicon) for value in combinations],
                    dtype=torch.long,
                    device=device,
                )
                swapped_values = tuple((value[1], value[0], value[2]) for value in combinations)
                wrong_donors = torch.tensor(
                    [encode(value, donor_template, query, lexicon) for value in swapped_values],
                    dtype=torch.long,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recipient_logits, recipient_states = model(recipients, return_states=True)
                    donor_logits, donor_states = model(donors, return_states=True)
                    wrong_logits, wrong_states = model(wrong_donors, return_states=True)
                recipient_baseline = recipient_logits[:, -1].float().index_select(-1, candidates).argmax(-1)
                donor_baseline = donor_logits[:, -1].float().index_select(-1, candidates).argmax(-1)
                wrong_baseline = wrong_logits[:, -1].float().index_select(-1, candidates).argmax(-1)
                same_hidden = recipient_states[layer].clone()
                same_hidden[:, -1] = donor_states[layer][:, -1]
                wrong_hidden = recipient_states[layer].clone()
                wrong_hidden[:, -1] = wrong_states[layer][:, -1]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    same_prediction = model.forward_from(same_hidden, layer)[:, -1].float().index_select(-1, candidates).argmax(-1)
                    wrong_prediction = model.forward_from(wrong_hidden, layer)[:, -1].float().index_select(-1, candidates).argmax(-1)
                same_results.extend((same_prediction == recipient_baseline).cpu().tolist())
                same_count += len(combinations)
                if not bool(torch.all(recipient_baseline == donor_baseline)):
                    raise RuntimeError("same-function donor changed baseline behavior")
                eligible = recipient_baseline != wrong_baseline
                transfer_results.extend((wrong_prediction[eligible] == wrong_baseline[eligible]).cpu().tolist())
                eligible_count += int(torch.sum(eligible).item())
    return {
        "same_case_count": same_count,
        "wrong_eligible_case_count": eligible_count,
        "same_preservation": float(np.mean(same_results)) if same_results else 0.0,
        "wrong_transfer": float(np.mean(transfer_results)) if transfer_results else 0.0,
    }


def full_state_distance(
    model: TinyCausalTransformer,
    task: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    layer: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    lexicon = make_lexicon(task)
    inputs = torch.tensor(
        [encode(value, template, "null", lexicon) for template in range(6) for value in combinations],
        dtype=torch.long,
    )
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), int(TRAINING["evaluation_batch_size"])):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, states = model(inputs[start : start + int(TRAINING["evaluation_batch_size"])].to(device), True)
            parts.append(states[layer].float().cpu())
    values = torch.cat(parts).reshape(6, len(combinations), SEQUENCE_LENGTH, -1).permute(1, 0, 2, 3)
    distances: list[float] = []
    for item in values:
        for left in range(6):
            for right in range(left + 1, 6):
                distances.append(float(torch.sqrt(torch.mean((item[left] - item[right]) ** 2)).item()))
    return {
        "functional_twin_pair_count": len(distances),
        "minimum_rms_distance": float(min(distances)),
        "median_rms_distance": float(np.median(distances)),
    }


def bag_control(signatures: dict[tuple[int, int, int], tuple[int, int, int]]) -> dict[str, Any]:
    groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for combination in ALL_COMBINATIONS:
        groups[tuple(sorted(combination))].append(signatures[combination])
    correct = sum(max(Counter(values).values()) for values in groups.values())
    unique = sum(len(values) for values in groups.values() if len(set(values)) == 1)
    return {
        "group_count": len(groups),
        "majority_accuracy_upper_bound": float(correct / len(ALL_COMBINATIONS)),
        "identifiable_fraction": float(unique / len(ALL_COMBINATIONS)),
    }


def probe_registry(signatures: dict[tuple[int, int, int], tuple[int, int, int]]) -> dict[str, Any]:
    rows = []
    candidates = (
        ("none", 0, ()),
        ("null", 1, ()),
        ("row", 1, (0,)),
        ("column", 1, (1,)),
        ("context", 1, (2,)),
        ("row_column", 2, (0, 1)),
        ("row_context", 2, (0, 2)),
        ("column_context", 2, (1, 2)),
        ("primitive_triple", 3, (0, 1, 2)),
    )
    for name, cost, coordinates in candidates:
        values = {tuple(signatures[item][index] for index in coordinates) for item in ALL_COMBINATIONS}
        rows.append({"name": name, "cost": cost, "class_count": len(values), "identifies_target": len(values) == len(ALL_COMBINATIONS)})
    successful = [value for value in rows if value["identifies_target"]]
    return {
        "rows": rows,
        "minimum_registered_probe": min(successful, key=lambda value: (value["cost"], value["name"])) if successful else None,
    }


def analyze_models(
    endpoint: TinyCausalTransformer,
    future: TinyCausalTransformer,
    initial: TinyCausalTransformer,
    task: dict[str, Any],
) -> dict[str, Any]:
    train_combinations, holdout_combinations = split_combinations(task)
    endpoint_signatures, endpoint_signature_metrics = signature_map(endpoint, task)
    future_signatures, future_signature_metrics = signature_map(future, task)
    future_stability = float(
        np.mean([endpoint_signatures[value] == future_signatures[value] for value in ALL_COMBINATIONS])
    )
    bag = bag_control(endpoint_signatures)
    registry = probe_registry(endpoint_signatures)
    endpoint_fit_all, fit_combinations = collect_response_features_all_layers(
        endpoint, task, train_combinations, (0, 1)
    )
    endpoint_validation_all, validation_combinations = collect_response_features_all_layers(
        endpoint, task, train_combinations, (2, 3)
    )
    endpoint_holdout_all, holdout_rows = collect_response_features_all_layers(
        endpoint, task, holdout_combinations, (4, 5)
    )
    initial_fit_all, initial_fit_combinations = collect_response_features_all_layers(
        initial, task, train_combinations, (0, 1)
    )
    initial_validation_all, initial_validation_combinations = collect_response_features_all_layers(
        initial, task, train_combinations, (2, 3)
    )
    initial_holdout_all, initial_holdout_combinations = collect_response_features_all_layers(
        initial, task, holdout_combinations, (4, 5)
    )
    future_holdout_all, future_holdout_rows = collect_response_features_all_layers(
        future, task, holdout_combinations, (4, 5)
    )
    layer_rows: list[dict[str, Any]] = []
    selected_layer: int | None = None
    selected_weights: list[np.ndarray] | None = None
    for layer in range(endpoint.config.layers + 1):
        weights = fit_decoder(endpoint_fit_all[layer], fit_combinations, endpoint_signatures)
        initial_weights = fit_decoder(initial_fit_all[layer], initial_fit_combinations, endpoint_signatures)
        camera = {
            "layer": layer,
            "validation": decoder_accuracy(
                endpoint_validation_all[layer], validation_combinations, endpoint_signatures, weights
            ),
            "heldout": decoder_accuracy(endpoint_holdout_all[layer], holdout_rows, endpoint_signatures, weights),
            "initial_validation": decoder_accuracy(
                initial_validation_all[layer], initial_validation_combinations, endpoint_signatures, initial_weights
            ),
            "initial_holdout": decoder_accuracy(
                initial_holdout_all[layer], initial_holdout_combinations, endpoint_signatures, initial_weights
            ),
        }
        row = dict(camera)
        row["patch"] = None
        camera_eligible = bool(
            camera["validation"]["combined_accuracy"] >= THRESHOLDS["camera_validation_accuracy_min"]
            and camera["initial_validation"]["combined_accuracy"] <= THRESHOLDS["random_camera_accuracy_max"]
        )
        if camera_eligible and selected_layer is None:
            patch = query_patch_metrics(endpoint, task, train_combinations, (2, 3), (0, 1), layer)
            row["patch"] = patch
            if (
                patch["same_preservation"] >= THRESHOLDS["patch_same_preservation_min"]
                and patch["wrong_transfer"] >= THRESHOLDS["patch_wrong_transfer_min"]
            ):
                selected_layer = layer
                selected_weights = weights
        layer_rows.append(row)
    if selected_layer is None or selected_weights is None:
        return {
            "endpoint_signature": endpoint_signature_metrics,
            "future_signature": future_signature_metrics,
            "future_signature_stability": future_stability,
            "bag_control": bag,
            "probe_registry": registry,
            "layers": layer_rows,
            "selected_layer": None,
            "event_qualified": False,
        }
    heldout_camera = decoder_accuracy(
        endpoint_holdout_all[selected_layer], holdout_rows, endpoint_signatures, selected_weights
    )
    future_camera = decoder_accuracy(
        future_holdout_all[selected_layer], future_holdout_rows, endpoint_signatures, selected_weights
    )
    heldout_patch = query_patch_metrics(endpoint, task, holdout_combinations, (4, 5), (0, 1), selected_layer)
    state_distance = full_state_distance(endpoint, task, holdout_combinations, selected_layer)
    event_qualified = bool(
        endpoint_signature_metrics["template_invariance_fraction"] >= THRESHOLDS["signature_template_invariance_min"]
        and endpoint_signature_metrics["class_count"] == THRESHOLDS["signature_class_count"]
        and future_stability >= THRESHOLDS["future_signature_stability_min"]
        and heldout_camera["combined_accuracy"] >= THRESHOLDS["camera_holdout_accuracy_min"]
        and future_camera["combined_accuracy"] >= THRESHOLDS["camera_future_accuracy_min"]
        and heldout_patch["same_preservation"] >= THRESHOLDS["patch_same_preservation_min"]
        and heldout_patch["wrong_transfer"] >= THRESHOLDS["patch_wrong_transfer_min"]
        and state_distance["minimum_rms_distance"] >= THRESHOLDS["state_rms_distance_min"]
        and bag["majority_accuracy_upper_bound"] <= THRESHOLDS["bag_majority_accuracy_max"]
    )
    return {
        "endpoint_signature": endpoint_signature_metrics,
        "future_signature": future_signature_metrics,
        "future_signature_stability": future_stability,
        "bag_control": bag,
        "probe_registry": registry,
        "layers": layer_rows,
        "selected_layer": selected_layer,
        "selected_relative_depth": float(selected_layer / endpoint.config.layers),
        "heldout_camera": heldout_camera,
        "future_camera": future_camera,
        "heldout_patch": heldout_patch,
        "state_distance": state_distance,
        "decoder_digest": digest([value.tolist() for value in selected_weights]),
        "event_qualified": event_qualified,
    }


def script_hashes() -> dict[str, str]:
    return {"main": sha256_file(SCRIPT), "audit": sha256_file(AUDIT_SCRIPT)}


def source_gate() -> dict[str, Any]:
    final = read_json(SOURCE1212 / "analysis/final.json")
    audit = read_json(SOURCE1212 / "audit/independent_result_audit.json")
    validate_digest(final, "final_digest")
    validate_digest(audit, "audit_digest")
    checks = {
        "phase1212_final_frozen": final["final_digest"] == EXPECTED_1212_FINAL,
        "phase1212_audit_frozen": audit["audit_digest"] == EXPECTED_1212_AUDIT,
        "phase1212_audit_passed": audit["all_checks_passed"] is True,
        "phase1212_free_transfer_was_denied": final["free_transformer_transfer_authorized"] is False,
        "phase1212_authorized_this_protocol_type": "free-TinyTransformer" in final["authorized_next"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1212 source gate failed: {checks}")
    return checks


def protocol_payload() -> dict[str, Any]:
    tasks = {
        split: [
            {
                **task,
                "lexicon_digest": digest(make_lexicon(task)),
                "train_combination_count": len(split_combinations(task)[0]),
                "holdout_combination_count": len(split_combinations(task)[1]),
            }
            for task in TASKS[split]
        ]
        for split in TASKS
    }
    return {
        "phase": PHASE,
        "title": "Free TinyTransformer behavioral quotient and first predictive-transfer event",
        "created_at": utc_now(),
        "source_gate": source_gate(),
        "source_phase1212_final_digest": EXPECTED_1212_FINAL,
        "source_phase1212_audit_digest": EXPECTED_1212_AUDIT,
        "script_hashes": script_hashes(),
        "object_contract": {
            "target": "actual endpoint plus short-future response signature on sealed query battery",
            "not_target": "hidden coordinate, hand-written hidden readout, or post-hoc coarsening",
            "semantic_states": VALUE_COUNT**3,
            "nuisance_templates": len(TEMPLATES),
            "exact_hidden_instances_per_model": 128 * len(TEMPLATES),
            "surface_matched_control": "row/column swaps preserve the complete token multiset but change behavior",
            "event": "earliest layer passing a calibration camera and donor-transfer conjunction",
        },
        "architectures": {
            split: {name: asdict(config) for name, config in values.items()} for split, values in ARCHITECTURES.items()
        },
        "tasks": tasks,
        "replicates": REPLICATES,
        "training": TRAINING,
        "thresholds": THRESHOLDS,
        "probe_registry": [
            "none",
            "null",
            "row",
            "column",
            "context",
            "row_column",
            "row_context",
            "column_context",
            "primitive_triple",
        ],
        "evidence_order": [
            "behavior qualification",
            "external behavior quotient seal",
            "internal predictive camera",
            "same-function preservation and wrong-function transfer",
            "future and confirmation replication",
        ],
        "forbidden": [
            "define functional classes from hidden results",
            "change the holdout masks after model results",
            "drop failed tasks or architectures",
            "add probe types after discovery",
            "claim natural-language semantics",
            "claim global minimality from the typed registry",
        ],
    }


def preregister() -> dict[str, Any]:
    payload = protocol_payload()
    payload["protocol_digest"] = digest(payload)
    write_json(PROTOCOL_PATH, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL_PATH)
    validate_digest(value, "protocol_digest")
    if value["script_hashes"] != script_hashes():
        raise RuntimeError("script changed after preregistration")
    if not all(source_gate().values()):
        raise RuntimeError("source gate changed")
    return value


def require_preaudit() -> dict[str, Any]:
    value = read_json(PREAUDIT_PATH)
    validate_digest(value, "audit_digest")
    if not value["all_checks_passed"] or value["protocol_digest"] != verify_protocol()["protocol_digest"]:
        raise RuntimeError("independent preaudit failed")
    return value


def checkpoint_payload(model: TinyCausalTransformer, config: ModelConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "config": asdict(config),
        "metadata": metadata,
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
    task: dict[str, Any],
    architecture_index: int,
    architecture: str,
    config: ModelConfig,
    replicate: int,
    device: torch.device,
) -> dict[str, Any]:
    identifier = run_id(split, task, architecture, replicate)
    run_root = OUT_ROOT / "runs" / split / identifier
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        validate_digest(metrics, "metrics_digest")
        return metrics
    seed = model_seed(split, task_index, architecture_index, replicate)
    set_seed(seed)
    initial = TinyCausalTransformer(config).to(device)
    model, optimizer, generator, training_metrics = train_model(config, task, seed, device)
    endpoint_path = run_root / "endpoint.pt"
    future_path = run_root / "future.pt"
    write_checkpoint(endpoint_path, checkpoint_payload(model, config, {"run_id": identifier, "kind": "endpoint"}))
    if training_metrics["qualified"]:
        continue_training(model, optimizer, generator, task, device)
    write_checkpoint(future_path, checkpoint_payload(model, config, {"run_id": identifier, "kind": "future"}))
    endpoint = load_checkpoint(endpoint_path, device)
    future = load_checkpoint(future_path, device)
    train_combinations, holdout_combinations = split_combinations(task)
    future_behavior = {
        "train": evaluate_behavior(future, task, train_combinations),
        "holdout": evaluate_behavior(future, task, holdout_combinations),
    }
    analysis = analyze_models(endpoint, future, initial, task) if training_metrics["qualified"] else {
        "event_qualified": False,
        "not_tested_reason": "behavior_gate_failed",
    }
    metrics = {
        "phase": PHASE,
        "run_id": identifier,
        "split": split,
        "task": task,
        "architecture": architecture,
        "config": asdict(config),
        "replicate": replicate,
        "seed": seed,
        "training": training_metrics,
        "future_behavior": future_behavior,
        "analysis": analysis,
        "files": {
            "endpoint": str(endpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "endpoint_sha256": sha256_file(endpoint_path),
            "future": str(future_path.relative_to(ROOT)).replace("\\", "/"),
            "future_sha256": sha256_file(future_path),
        },
    }
    metrics["metrics_digest"] = digest(metrics)
    write_json(metrics_path, metrics)
    print(
        canonical(
            {
                "run": identifier,
                "steps": training_metrics["steps"],
                "behavior": training_metrics["qualified"],
                "event": analysis.get("event_qualified", False),
                "layer": analysis.get("selected_layer"),
            }
        ),
        flush=True,
    )
    del model, endpoint, future, initial, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def group_summary(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    behavior_rows = [value for value in rows if value["training"]["qualified"]]
    event_rows = [value for value in rows if value["analysis"].get("event_qualified")]
    by_architecture: dict[str, dict[str, int]] = {}
    for architecture in ARCHITECTURES[split]:
        values = [value for value in rows if value["architecture"] == architecture]
        by_architecture[architecture] = {
            "total": len(values),
            "behavior_qualified": sum(value["training"]["qualified"] for value in values),
            "event_qualified": sum(value["analysis"].get("event_qualified", False) for value in values),
        }
    behavior_gate = bool(
        len(behavior_rows) >= THRESHOLDS["qualified_models_per_split_min"]
        and all(value["behavior_qualified"] >= THRESHOLDS["qualified_models_per_architecture_min"] for value in by_architecture.values())
    )
    event_gate = bool(
        len(event_rows) >= THRESHOLDS["qualified_models_per_split_min"]
        and all(value["event_qualified"] >= THRESHOLDS["qualified_models_per_architecture_min"] for value in by_architecture.values())
    )
    selected = [value["analysis"]["selected_relative_depth"] for value in event_rows]
    return {
        "split": split,
        "model_count": len(rows),
        "behavior_qualified_count": len(behavior_rows),
        "event_qualified_count": len(event_rows),
        "by_architecture": by_architecture,
        "behavior_gate": behavior_gate,
        "event_gate": event_gate,
        "selected_relative_depth_median": float(np.median(selected)) if selected else None,
        "selected_relative_depth_range": [float(min(selected)), float(max(selected))] if selected else None,
    }


def execute() -> dict[str, Any]:
    protocol = verify_protocol()
    preaudit = require_preaudit()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("discovery", "confirmation"):
        rows: list[dict[str, Any]] = []
        for task_index, task in enumerate(TASKS[split]):
            for architecture_index, (architecture, config) in enumerate(ARCHITECTURES[split].items()):
                for replicate in range(REPLICATES):
                    rows.append(
                        execute_run(
                            split,
                            task_index,
                            task,
                            architecture_index,
                            architecture,
                            config,
                            replicate,
                            device,
                        )
                    )
        all_rows[split] = rows
    summaries = {split: group_summary(split, rows) for split, rows in all_rows.items()}
    confirmed = bool(
        summaries["discovery"]["behavior_gate"]
        and summaries["confirmation"]["behavior_gate"]
        and summaries["discovery"]["event_gate"]
        and summaries["confirmation"]["event_gate"]
    )
    run_manifest = []
    for rows in all_rows.values():
        for row in rows:
            path = OUT_ROOT / "runs" / row["split"] / row["run_id"] / "metrics.json"
            run_manifest.append(
                {
                    "run_id": row["run_id"],
                    "metrics": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "metrics_sha256": sha256_file(path),
                    "metrics_digest": row["metrics_digest"],
                    **row["files"],
                }
            )
    final = {
        "phase": PHASE,
        "title": protocol["title"],
        "completed_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "preaudit_digest": preaudit["audit_digest"],
        "summaries": summaries,
        "run_manifest": run_manifest,
        "claims": {
            "free_behavioral_quotient": "confirmed" if confirmed else "not_confirmed",
            "first_predictive_transfer_event": "confirmed" if confirmed else "not_confirmed",
            "natural_language_transfer": "not_tested",
            "global_minimality": "not_claimed",
            "new_mathematics_required": "not_supported",
        },
        "candidate_puzzle": "K193" if confirmed else None,
        "limitations": [
            "The task is a synthetic role-value binding language, not natural language.",
            "The target quotient is relative to a frozen three-query battery and a 32-step horizon.",
            "The linear camera establishes predictability, not uniqueness of representation.",
            "A query-state patch establishes local sufficiency in this architecture, not global necessity.",
            "The earliest event is criterion-relative and may move with architecture or optimization.",
            "The cost-three probe minimum is only within the preregistered typed registry.",
        ],
        "next_authorization": (
            "independent natural-material Qwen3 single-model protocol may be designed, but must be separately preregistered"
            if confirmed
            else "stop transfer; revise the free-network object without inspecting new hidden coordinates"
        ),
        "auto_continue": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical({"final_digest": final["final_digest"], "confirmed": confirmed, "summaries": summaries}), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "execute", "verify"))
    args = parser.parse_args()
    if args.command == "preregister":
        print(canonical(preregister()))
    elif args.command == "execute":
        execute()
    else:
        print(canonical(verify_protocol()))


if __name__ == "__main__":
    main()
