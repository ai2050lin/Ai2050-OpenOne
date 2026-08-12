#!/usr/bin/env python3
"""Factorial free-Transformer transfer of the calibrated formation clocks.

Phase 1217 crosses abstract task rule, physical lexicon, architecture, and
training seed.  It records full response curves for rule, confidence,
decodability, predictive transfer, single-component necessity, and joint
component necessity.  Discovery and confirmation use disjoint factor levels.
The experiment tests construct transfer of the Phase 1216 clock instrument;
it does not claim that a clock ordering is a language mechanism.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
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


PHASE = 1217
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1217_factorial_free_formation_clock_transfer_audit.py")
OUT_ROOT = TEST_ROOT / "result/phase1217_factorial_free_formation_clock_transfer"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
SOURCE1216 = TEST_ROOT / "result/phase1216_known_truth_three_clock_zoo"

EXPECTED_1216_FINAL = "9b3682327e7d81b938b4ee9046fe9fef8d1d0a37dced34989424d28dd4361ba6"
EXPECTED_1216_AUDIT = "7528184948bb47a8a2e2c1eb7c12d373e17ff8fbad6b09b2a122e54575e319f4"

VALUE_COUNT = p1213.VALUE_COUNT
ROLES = p1213.ROLES
QUERIES = p1213.QUERIES
FUNCTION_QUERIES = p1213.FUNCTION_QUERIES
TEMPLATES = p1213.TEMPLATES
ALL_COMBINATIONS = p1213.ALL_COMBINATIONS
VOCAB_SIZE = p1213.VOCAB_SIZE
SEQUENCE_LENGTH = p1213.SEQUENCE_LENGTH
CLOCKS = ("R", "C", "D", "E", "U1", "UJ")
REPLICATES = 2
RIDGE = 1.0e-3
INTERVENTION_TEMPLATES = (4, 5)

TASKS = {
    "discovery": (
        {
            "name": "task_identity",
            "source_roles": {"row": "row", "column": "column", "context": "context"},
        },
        {
            "name": "task_cycle_forward",
            "source_roles": {"row": "column", "column": "context", "context": "row"},
        },
    ),
    "confirmation": (
        {
            "name": "task_cycle_reverse",
            "source_roles": {"row": "context", "column": "row", "context": "column"},
        },
        {
            "name": "task_swap_row_column",
            "source_roles": {"row": "column", "column": "row", "context": "context"},
        },
    ),
}

LEXICONS = {
    "discovery": (
        {"name": "lexicon_d0", "seed": 1_217_101},
        {"name": "lexicon_d1", "seed": 1_217_103},
    ),
    "confirmation": (
        {"name": "lexicon_c0", "seed": 1_217_211},
        {"name": "lexicon_c1", "seed": 1_217_223},
    ),
}

ARCHITECTURES = {
    "discovery": {
        "d4_w80": ModelConfig(4, 80, 4, 160, SEQUENCE_LENGTH, VOCAB_SIZE),
        "d6_w112": ModelConfig(6, 112, 4, 224, SEQUENCE_LENGTH, VOCAB_SIZE),
    },
    "confirmation": {
        "d5_w96": ModelConfig(5, 96, 4, 192, SEQUENCE_LENGTH, VOCAB_SIZE),
        "d7_w128": ModelConfig(7, 128, 4, 256, SEQUENCE_LENGTH, VOCAB_SIZE),
    },
}

HOLDOUT_RULES = {
    "discovery": {"coefficients": (2, 1, 3), "bias": 1},
    "confirmation": {"coefficients": (3, 2, 1), "bias": 2},
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

THRESHOLD_PROFILES = {
    "lenient": {"R": 0.98, "C": 0.90, "D": 0.90, "E": 0.85, "U1": 0.10, "UJ": 0.10},
    "primary": {"R": 0.99, "C": 0.95, "D": 0.95, "E": 0.90, "U1": 0.15, "UJ": 0.15},
    "strict": {"R": 1.00, "C": 0.97, "D": 0.98, "E": 0.95, "U1": 0.20, "UJ": 0.20},
}

FIXED_CONTROLS = {
    "finite_fraction_min": 1.0,
    "initial_decode_accuracy_max": 0.25,
    "same_baseline_match_min": 0.98,
    "wrong_eligible_fraction_min": 0.50,
    "zero_drift_max": 1.0e-6,
}

GROUP_GATES = {
    "behavior_observed_per_split_min": 12,
    "behavior_observed_per_binary_level_min": 6,
    "decode_observed_per_split_min": 8,
    "interface_observed_per_split_min": 8,
    "threshold_status_stability_min": 0.75,
    "minimum_pre_behavior_prefixes_for_next_stage": 8,
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


def make_condition(split: str, task_index: int, lexicon_index: int) -> dict[str, Any]:
    task = dict(TASKS[split][task_index])
    lexicon = dict(LEXICONS[split][lexicon_index])
    return {
        **task,
        "lexicon_name": lexicon["name"],
        "lexicon_seed": int(lexicon["seed"]),
        "holdout": dict(HOLDOUT_RULES[split]),
    }


def make_lexicon(condition: dict[str, Any]) -> dict[str, Any]:
    return p1213.make_lexicon({"lexicon_seed": int(condition["lexicon_seed"])})


def split_combinations(condition: dict[str, Any]) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    rule = condition["holdout"]
    holdout = tuple(
        values
        for values in ALL_COMBINATIONS
        if (
            sum(int(a) * int(b) for a, b in zip(rule["coefficients"], values)) + int(rule["bias"])
        )
        % 4
        == 0
    )
    train = tuple(values for values in ALL_COMBINATIONS if values not in set(holdout))
    if len(train) != 384 or len(holdout) != 128:
        raise RuntimeError(f"unbalanced split: {len(train)}/{len(holdout)}")
    return train, holdout


def source_role(condition: dict[str, Any], query: str) -> str:
    if query == "null":
        return "row"
    return str(condition["source_roles"][query])


def raw_target(condition: dict[str, Any], combination: tuple[int, int, int], query: str) -> int:
    if query == "null":
        return 0
    return int(combination[ROLES.index(source_role(condition, query))])


def target_slot(condition: dict[str, Any], combination: tuple[int, int, int], query: str) -> int:
    lexicon = make_lexicon(condition)
    return int(lexicon["answer_permutation"][raw_target(condition, combination, query)])


def encode(
    combination: tuple[int, int, int],
    template_index: int,
    query: str,
    condition: dict[str, Any],
) -> list[int]:
    return p1213.encode(combination, template_index, query, make_lexicon(condition))


def build_examples(
    condition: dict[str, Any],
    combinations: Iterable[tuple[int, int, int]],
    templates: Iterable[int],
    queries: Iterable[str] = QUERIES,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[tuple[int, int, int], int, str]]]:
    rows: list[list[int]] = []
    targets: list[int] = []
    metadata: list[tuple[tuple[int, int, int], int, str]] = []
    for template in templates:
        for combination in combinations:
            for query in queries:
                rows.append(encode(combination, int(template), query, condition))
                targets.append(target_slot(condition, combination, query))
                metadata.append((combination, int(template), query))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(targets, dtype=torch.long), metadata


def candidate_ids(condition: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(make_lexicon(condition)["answers"], dtype=torch.long, device=device)


def predict_slots(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    condition: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    device = next(model.parameters()).device
    candidates = candidate_ids(condition, device)
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    finite: list[torch.Tensor] = []
    model.eval()
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
    condition: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    inputs, targets, metadata = build_examples(condition, combinations, range(len(TEMPLATES)))
    predicted, probabilities, finite_fraction = predict_slots(model, inputs, condition)
    truth = targets.numpy()
    correct = predicted == truth
    per_query = {
        query: float(np.mean(correct[np.asarray([row[2] == query for row in metadata], dtype=bool)]))
        for query in QUERIES
    }
    per_template = {
        str(template): float(np.mean(correct[np.asarray([row[1] == template for row in metadata], dtype=bool)]))
        for template in range(len(TEMPLATES))
    }
    target_probability = probabilities[np.arange(len(truth)), truth]
    minimum_group_accuracy = min(
        float(np.mean(correct)), min(per_query.values()), min(per_template.values())
    )
    return {
        "case_count": int(len(truth)),
        "accuracy": float(np.mean(correct)),
        "minimum_group_accuracy": minimum_group_accuracy,
        "minimum_probability": float(np.min(target_probability)),
        "mean_probability": float(np.mean(target_probability)),
        "finite_fraction": finite_fraction,
        "per_query_accuracy": per_query,
        "per_template_accuracy": per_template,
    }


def collect_features_all_layers(
    model: TinyCausalTransformer,
    condition: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    templates: tuple[int, ...],
) -> tuple[list[np.ndarray], list[tuple[tuple[int, int, int], int, str]]]:
    inputs, _, metadata = build_examples(condition, combinations, templates, FUNCTION_QUERIES)
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


def target_vector(condition: dict[str, Any], metadata: list[tuple[tuple[int, int, int], int, str]]) -> np.ndarray:
    return np.asarray(
        [target_slot(condition, combination, query) for combination, _, query in metadata],
        dtype=np.int64,
    )


def fit_decoder(
    features: np.ndarray,
    metadata: list[tuple[tuple[int, int, int], int, str]],
    condition: dict[str, Any],
) -> np.ndarray:
    design = np.concatenate((features.astype(np.float64), np.ones((len(features), 1))), axis=1)
    targets = np.zeros((len(features), VALUE_COUNT), dtype=np.float64)
    truth = target_vector(condition, metadata)
    targets[np.arange(len(features)), truth] = 1.0
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def decoder_metrics(
    features: np.ndarray,
    metadata: list[tuple[tuple[int, int, int], int, str]],
    condition: dict[str, Any],
    weights: np.ndarray,
) -> dict[str, Any]:
    design = np.concatenate((features.astype(np.float64), np.ones((len(features), 1))), axis=1)
    predicted = np.argmax(design @ weights, axis=1)
    truth = target_vector(condition, metadata)
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


def initial_camera_controls(model: TinyCausalTransformer, condition: dict[str, Any]) -> list[dict[str, Any]]:
    train, _ = split_combinations(condition)
    fit_values, fit_meta = collect_features_all_layers(model, condition, train, (0, 1))
    validation_values, validation_meta = collect_features_all_layers(model, condition, train, (2, 3))
    rows = []
    for layer in range(model.config.layers + 1):
        weights = fit_decoder(fit_values[layer], fit_meta, condition)
        rows.append(decoder_metrics(validation_values[layer], validation_meta, condition, weights))
    return rows


def counterfactual_combination(
    condition: dict[str, Any], combination: tuple[int, int, int], query: str
) -> tuple[int, int, int]:
    values = list(combination)
    index = ROLES.index(source_role(condition, query))
    values[index] = (values[index] + 1) % VALUE_COUNT
    return tuple(values)


def patch_metrics(
    model: TinyCausalTransformer,
    condition: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
    layer: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    candidates = candidate_ids(condition, device)
    same_baseline: list[bool] = []
    same_patch: list[bool] = []
    wrong_transfer: list[bool] = []
    eligible_total = 0
    case_total = 0
    model.eval()
    with torch.inference_mode():
        for recipient_template, donor_template in ((4, 0), (5, 1)):
            for query in FUNCTION_QUERIES:
                wrong_values = tuple(counterfactual_combination(condition, value, query) for value in combinations)
                recipients = torch.tensor(
                    [encode(value, recipient_template, query, condition) for value in combinations],
                    dtype=torch.long,
                    device=device,
                )
                donors = torch.tensor(
                    [encode(value, donor_template, query, condition) for value in combinations],
                    dtype=torch.long,
                    device=device,
                )
                wrong_donors = torch.tensor(
                    [encode(value, donor_template, query, condition) for value in wrong_values],
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


def forward_with_component_mask(
    model: TinyCausalTransformer,
    input_ids: torch.Tensor,
    layer: int,
    use_attention: bool,
    use_mlp: bool,
) -> torch.Tensor:
    hidden = model.embed(input_ids)
    for index, block in enumerate(model.blocks):
        if index != layer:
            hidden = block(hidden)
            continue
        if use_attention:
            hidden = hidden + block.attn(block.attn_norm(hidden))
        if use_mlp:
            hidden = hidden + block.mlp(block.mlp_norm(hidden))
    return model.lm_head(model.final_norm(hidden))


def necessity_metrics(
    model: TinyCausalTransformer,
    condition: dict[str, Any],
    combinations: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    inputs, targets, _ = build_examples(condition, combinations, INTERVENTION_TEMPLATES, FUNCTION_QUERIES)
    device = next(model.parameters()).device
    ids = inputs.to(device, non_blocking=True)
    truth = targets.to(device, non_blocking=True)
    candidates = candidate_ids(condition, device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        clean_logits = model(ids)[:, -1].float().index_select(-1, candidates)
    clean_predictions = clean_logits.argmax(-1)
    clean_accuracy = float((clean_predictions == truth).to(torch.float32).mean().item())
    clean_probability = float(torch.softmax(clean_logits, -1)[torch.arange(len(truth), device=device), truth].mean().item())
    layers: list[dict[str, Any]] = []
    zero_drift_max = 0.0
    for layer in range(model.config.layers):
        values: dict[str, dict[str, float]] = {}
        for name, use_attention, use_mlp in (
            ("clean_replay", True, True),
            ("attention_neutral", False, True),
            ("mlp_neutral", True, False),
            ("joint_neutral", False, False),
        ):
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_with_component_mask(model, ids, layer, use_attention, use_mlp)[:, -1].float().index_select(-1, candidates)
            predictions = logits.argmax(-1)
            probability = torch.softmax(logits, -1)[torch.arange(len(truth), device=device), truth]
            values[name] = {
                "accuracy": float((predictions == truth).to(torch.float32).mean().item()),
                "mean_target_probability": float(probability.mean().item()),
            }
            if name == "clean_replay":
                zero_drift_max = max(zero_drift_max, float(torch.max(torch.abs(logits - clean_logits)).item()))
        attention_damage = max(0.0, clean_accuracy - values["attention_neutral"]["accuracy"])
        mlp_damage = max(0.0, clean_accuracy - values["mlp_neutral"]["accuracy"])
        joint_damage = max(0.0, clean_accuracy - values["joint_neutral"]["accuracy"])
        layers.append(
            {
                "layer": layer,
                "relative_depth": float((layer + 1) / model.config.layers),
                "clean": values["clean_replay"],
                "attention_neutral": values["attention_neutral"],
                "mlp_neutral": values["mlp_neutral"],
                "joint_neutral": values["joint_neutral"],
                "attention_accuracy_damage": attention_damage,
                "mlp_accuracy_damage": mlp_damage,
                "max_single_accuracy_damage": max(attention_damage, mlp_damage),
                "joint_accuracy_damage": joint_damage,
                "joint_excess_over_best_single": joint_damage - max(attention_damage, mlp_damage),
            }
        )
    best_single = max(layers, key=lambda row: row["max_single_accuracy_damage"])
    best_joint = max(layers, key=lambda row: row["joint_accuracy_damage"])
    return {
        "case_count": int(len(inputs)),
        "clean_accuracy": clean_accuracy,
        "clean_mean_target_probability": clean_probability,
        "single_necessity": float(best_single["max_single_accuracy_damage"]),
        "single_layer": int(best_single["layer"]),
        "joint_necessity": float(best_joint["joint_accuracy_damage"]),
        "joint_layer": int(best_joint["layer"]),
        "zero_drift_max": zero_drift_max,
        "layers": layers,
    }


def checkpoint_gates(row: dict[str, Any], profile: dict[str, float]) -> dict[str, bool]:
    d_layers = [
        layer
        for layer in row["layers"]
        if layer["initial_validation"]["combined_accuracy"] <= FIXED_CONTROLS["initial_decode_accuracy_max"]
        and layer["validation"]["combined_accuracy"] >= profile["D"]
        and layer["holdout"]["combined_accuracy"] >= profile["D"]
    ]
    e_layers = [
        layer
        for layer in d_layers
        if layer["patch"] is not None
        and layer["patch"]["same_baseline_match"] >= FIXED_CONTROLS["same_baseline_match_min"]
        and layer["patch"]["same_preservation"] >= profile["E"]
        and layer["patch"]["wrong_eligible_fraction"] >= FIXED_CONTROLS["wrong_eligible_fraction_min"]
        and layer["patch"]["wrong_transfer"] >= profile["E"]
    ]
    return {
        "R": bool(row["metrics"]["rule_accuracy"] >= profile["R"]),
        "C": bool(row["metrics"]["minimum_correct_probability"] >= profile["C"]),
        "D": bool(d_layers),
        "E": bool(e_layers),
        "U1": bool(row["metrics"]["single_necessity"] >= profile["U1"]),
        "UJ": bool(row["metrics"]["joint_necessity"] >= profile["UJ"]),
    }


def scan_checkpoint(
    model: TinyCausalTransformer,
    condition: dict[str, Any],
    initial_controls: list[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    train, holdout = split_combinations(condition)
    train_behavior = evaluate_behavior(model, condition, train)
    holdout_behavior = evaluate_behavior(model, condition, holdout)
    rule_accuracy = min(train_behavior["minimum_group_accuracy"], holdout_behavior["minimum_group_accuracy"])
    minimum_probability = min(train_behavior["minimum_probability"], holdout_behavior["minimum_probability"])

    fit_values, fit_meta = collect_features_all_layers(model, condition, train, (0, 1))
    validation_values, validation_meta = collect_features_all_layers(model, condition, train, (2, 3))
    holdout_values, holdout_meta = collect_features_all_layers(model, condition, holdout, (4, 5))
    layers: list[dict[str, Any]] = []
    for layer in range(model.config.layers + 1):
        weights = fit_decoder(fit_values[layer], fit_meta, condition)
        validation = decoder_metrics(validation_values[layer], validation_meta, condition, weights)
        heldout = decoder_metrics(holdout_values[layer], holdout_meta, condition, weights)
        patch = None
        if (
            initial_controls[layer]["combined_accuracy"] <= FIXED_CONTROLS["initial_decode_accuracy_max"]
            and validation["combined_accuracy"] >= THRESHOLD_PROFILES["lenient"]["D"]
            and heldout["combined_accuracy"] >= THRESHOLD_PROFILES["lenient"]["D"]
        ):
            patch = patch_metrics(model, condition, holdout, layer)
        layers.append(
            {
                "layer": layer,
                "relative_depth": float(layer / model.config.layers),
                "validation": validation,
                "holdout": heldout,
                "initial_validation": initial_controls[layer],
                "patch": patch,
            }
        )
    necessity = necessity_metrics(model, condition, holdout)
    decode_accuracy = max(
        min(layer["validation"]["combined_accuracy"], layer["holdout"]["combined_accuracy"])
        if layer["initial_validation"]["combined_accuracy"] <= FIXED_CONTROLS["initial_decode_accuracy_max"]
        else 0.0
        for layer in layers
    )
    patch_rows = [layer for layer in layers if layer["patch"] is not None]
    transfer_success = max((layer["patch"]["wrong_transfer"] for layer in patch_rows), default=0.0)
    preservation_success = max((layer["patch"]["same_preservation"] for layer in patch_rows), default=0.0)
    row = {
        "step": int(step),
        "samples_seen": int(step * TRAINING["batch_size"]),
        "tokens_seen": int(step * TRAINING["batch_size"] * SEQUENCE_LENGTH),
        "metrics": {
            "rule_accuracy": float(rule_accuracy),
            "minimum_correct_probability": float(minimum_probability),
            "decode_accuracy": float(decode_accuracy),
            "transfer_success": float(transfer_success),
            "preservation_success": float(preservation_success),
            "single_necessity": necessity["single_necessity"],
            "joint_necessity": necessity["joint_necessity"],
        },
        "train_behavior": train_behavior,
        "holdout_behavior": holdout_behavior,
        "necessity": necessity,
        "layers": layers,
    }
    row["gates"] = {
        name: checkpoint_gates(row, profile) for name, profile in THRESHOLD_PROFILES.items()
    }
    return row


def stable_onset(trajectory: list[dict[str, Any]], profile: str, clock: str) -> dict[str, Any]:
    gates = [bool(row["gates"][profile][clock]) for row in trajectory]
    required = int(TRAINING["required_consecutive_passes"])
    threshold = float(TRAINING["post_formation_stability_min"])
    for index in range(len(gates) - required + 1):
        if all(gates[index : index + required]):
            tail_fraction = float(np.mean(gates[index:]))
            if gates[-1] and tail_fraction >= threshold:
                return {
                    "status": "observed",
                    "step": int(trajectory[index]["step"]),
                    "interval_width_steps": int(TRAINING["evaluation_interval"]),
                    "post_formation_stability": tail_fraction,
                }
    return {
        "status": "right_censored",
        "lower_bound_step": int(TRAINING["maximum_steps"]),
        "post_formation_stability": None,
    }


def relation_signature(clocks: dict[str, dict[str, Any]]) -> str:
    finite = sorted({int(value["step"]) for value in clocks.values() if value["status"] == "observed"})
    ranks = {value: index for index, value in enumerate(finite)}
    return "|".join(
        f"{clock}:{ranks[int(clocks[clock]['step'])] if clocks[clock]['status'] == 'observed' else 'X'}"
        for clock in CLOCKS
    )


def summarize_trajectory(trajectory: list[dict[str, Any]], count: int) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile in THRESHOLD_PROFILES:
        clocks = {clock: stable_onset(trajectory, profile, clock) for clock in CLOCKS}
        for value in clocks.values():
            if value["status"] == "observed":
                step = int(value["step"])
                value["normalized"] = {
                    "updates": step,
                    "samples_seen": int(step * TRAINING["batch_size"]),
                    "tokens_seen": int(step * TRAINING["batch_size"] * SEQUENCE_LENGTH),
                    "parameter_token_proxy": int(step * TRAINING["batch_size"] * SEQUENCE_LENGTH * count),
                }
        profiles[profile] = {"clocks": clocks, "signature": relation_signature(clocks)}
    primary = profiles["primary"]["clocks"]
    threshold_stability = {
        clock: float(
            np.mean(
                [
                    profiles[name]["clocks"][clock]["status"] == primary[clock]["status"]
                    for name in THRESHOLD_PROFILES
                ]
            )
        )
        for clock in CLOCKS
    }
    observed_spans = {}
    for clock in CLOCKS:
        steps = [
            int(profiles[name]["clocks"][clock]["step"])
            for name in THRESHOLD_PROFILES
            if profiles[name]["clocks"][clock]["status"] == "observed"
        ]
        observed_spans[clock] = int(max(steps) - min(steps)) if steps else None
    r_clock = primary["R"]
    pre_behavior_prefix_count = (
        sum(int(row["step"]) < int(r_clock["step"]) for row in trajectory)
        if r_clock["status"] == "observed"
        else len(trajectory) - 1
    )
    return {
        "profiles": profiles,
        "primary_clocks": primary,
        "primary_signature": profiles["primary"]["signature"],
        "threshold_status_stability": threshold_stability,
        "threshold_onset_span_steps": observed_spans,
        "pre_behavior_prefix_count": int(pre_behavior_prefix_count),
        "endpoint_zero_drift_max": float(trajectory[-1]["necessity"]["zero_drift_max"]),
    }


def model_seed(split: str, task_index: int, lexicon_index: int, architecture_index: int, replicate: int) -> int:
    base = 1_217_300_000 if split == "discovery" else 1_217_700_000
    return base + task_index * 1_000_003 + lexicon_index * 100_003 + architecture_index * 10_007 + replicate * 1_009


def run_id(split: str, condition: dict[str, Any], architecture: str, replicate: int) -> str:
    return f"{split}__{condition['name']}__{condition['lexicon_name']}__{architecture}__s{replicate}"


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


def source_gate() -> dict[str, bool]:
    final = read_json(SOURCE1216 / "analysis/final.json")
    audit = read_json(SOURCE1216 / "audit/independent_audit.json")
    validate_digest(final, "final_digest")
    validate_digest(audit, "audit_digest")
    return {
        "phase1216_final_digest": final["final_digest"] == EXPECTED_1216_FINAL,
        "phase1216_audit_digest": audit["audit_digest"] == EXPECTED_1216_AUDIT,
        "phase1216_overall_pass": final["summary"]["overall_pass"] is True,
        "phase1216_free_network_not_tested": final["summary"]["claims"]["free_network_external_validity"] == "not_tested",
        "phase1216_authorized_t02": final["authorized_next"]["experiment"] == "T02_FACTORIAL_FREE_FORMATION",
    }


def script_hashes() -> dict[str, str]:
    return {
        "phase1217_main": sha256_file(SCRIPT),
        "phase1217_audit": sha256_file(AUDIT_SCRIPT),
        "phase1213_material_source": sha256_file(Path(p1213.__file__)),
        "tiny_transformer_source": sha256_file(TEST_ROOT / "phase1146_learned_composition_benchmark.py"),
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "title": "Factorial free-Transformer formation-clock construct transfer",
        "created_at": utc_now(),
        "source_gate": source_gate(),
        "script_hashes": script_hashes(),
        "scientific_object": {
            "target": "freely learned role-conditioned retrieval rule under a sealed external input-output contract",
            "factorization": "TaskRule x PhysicalLexicon x Architecture x TrainingSeed, fully crossed within each split",
            "clock_types": {
                "R": "stable minimum top-1 rule accuracy",
                "C": "stable minimum correct-answer probability",
                "D": "stable held-out linear decodability with a random-initialization negative control",
                "E": "stable same-function preservation and wrong-function transfer at one query-state interface",
                "U1": "stable maximum accuracy damage from one attention or MLP update neutralization",
                "UJ": "stable maximum accuracy damage from jointly neutralizing attention and MLP updates in one block",
            },
            "censoring": "nonformation by step 2400 is right censoring, never proof of permanent absence",
            "scope": "clock construct validity in micro-Transformers, not pretrained-language validity",
        },
        "tasks": TASKS,
        "lexicons": LEXICONS,
        "architectures": {
            split: {name: asdict(config) for name, config in values.items()}
            for split, values in ARCHITECTURES.items()
        },
        "holdout_rules": HOLDOUT_RULES,
        "replicates": REPLICATES,
        "formal_run_count": 32,
        "runs_per_split": 16,
        "training": TRAINING,
        "threshold_profiles": THRESHOLD_PROFILES,
        "fixed_controls": FIXED_CONTROLS,
        "group_gates": GROUP_GATES,
        "frozen_analysis": {
            "stable_onset": "first two-checkpoint run whose tail pass fraction is at least 0.80 and endpoint passes",
            "threshold_robustness": "clock observed/censored status agreement with the primary profile over lenient, primary, strict profiles",
            "factor_effect": "matched-pair onset/status differences with all other factors fixed; no omnibus significance test",
            "full_response": "all scalar and per-layer response values retained at every checkpoint",
            "no_posthoc_clock_order": True,
        },
        "gating": {
            "external_validity": "behavior breadth AND decode breadth AND interface breadth AND threshold stability AND finite zero-drift audit",
            "next_stage": "only pre-behavior prefixes from systems with observed R may authorize a separately frozen precursor-prediction phase",
            "failure_scope": "failure constrains this finite task/lexicon/architecture/seed registry only",
        },
        "forbidden": [
            "posthoc threshold replacement",
            "dropping failed factor cells",
            "calling decodability causal use",
            "calling component neutralization semantic surgery",
            "claiming language-model external validity",
            "claiming a universal clock ordering",
        ],
    }


def preregister() -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        existing = read_json(PROTOCOL_PATH)
        validate_digest(existing, "protocol_digest")
        current_hashes = script_hashes()
        if existing["script_hashes"] != current_hashes:
            raise RuntimeError("frozen protocol script hashes differ from current scripts")
        return existing
    payload = protocol_payload()
    if not all(payload["source_gate"].values()):
        raise RuntimeError(f"source gate failed: {payload['source_gate']}")
    payload["protocol_digest"] = digest(payload)
    write_json(PROTOCOL_PATH, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_digest(protocol, "protocol_digest")
    if protocol["script_hashes"] != script_hashes():
        raise RuntimeError("script hash drift after preregistration")
    if not PREAUDIT_PATH.exists():
        raise RuntimeError("independent preaudit missing")
    preaudit = read_json(PREAUDIT_PATH)
    validate_digest(preaudit, "audit_digest")
    if preaudit["all_checks_passed"] is not True or preaudit["protocol_digest"] != protocol["protocol_digest"]:
        raise RuntimeError("independent preaudit did not authorize execution")
    return protocol


def execute_run(
    split: str,
    task_index: int,
    lexicon_index: int,
    architecture_index: int,
    replicate: int,
    device: torch.device,
) -> dict[str, Any]:
    protocol = verify_protocol()
    condition = make_condition(split, task_index, lexicon_index)
    architecture, config = list(ARCHITECTURES[split].items())[architecture_index]
    identifier = run_id(split, condition, architecture, replicate)
    run_root = OUT_ROOT / "runs" / split / identifier
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        existing = read_json(metrics_path)
        validate_digest(existing, "metrics_digest")
        if existing["protocol_digest"] != protocol["protocol_digest"]:
            raise RuntimeError(f"stale metrics for {identifier}")
        return existing

    seed = model_seed(split, task_index, lexicon_index, architecture_index, replicate)
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    count = parameter_count(model)
    train_combinations, _ = split_combinations(condition)
    train_inputs, train_targets, _ = build_examples(condition, train_combinations, range(len(TEMPLATES)))
    candidates = candidate_ids(condition, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        weight_decay=float(TRAINING["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 37)
    initial_controls = initial_camera_controls(model, condition)
    trajectory: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    def record(step: int, loss: float | None, gradient_norm: float | None) -> None:
        checkpoint_path = run_root / "checkpoints" / f"step_{step:04d}.pt"
        write_checkpoint(
            checkpoint_path,
            checkpoint_payload(model, config, identifier, step, protocol["protocol_digest"]),
        )
        scan = scan_checkpoint(model, condition, initial_controls, step)
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
        gates = scan["gates"]["primary"]
        print(
            f"[{utc_now()}] {identifier} step={step} "
            + " ".join(f"{clock}={int(gates[clock])}" for clock in CLOCKS)
            + f" holdout={scan['holdout_behavior']['accuracy']:.4f} pmin={scan['holdout_behavior']['minimum_probability']:.4f}",
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

    metrics = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_id": identifier,
        "split": split,
        "task_index": task_index,
        "task_name": condition["name"],
        "source_roles": condition["source_roles"],
        "lexicon_index": lexicon_index,
        "lexicon_name": condition["lexicon_name"],
        "lexicon_seed": condition["lexicon_seed"],
        "architecture_index": architecture_index,
        "architecture": architecture,
        "config": asdict(config),
        "replicate": replicate,
        "seed": seed,
        "parameter_count": count,
        "initial_camera_controls": initial_controls,
        "trajectory": trajectory,
        "formation": summarize_trajectory(trajectory, count),
        "checkpoint_manifest": checkpoint_manifest,
    }
    metrics["metrics_digest"] = digest(metrics)
    write_json(metrics_path, metrics)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def load_rows(split: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((OUT_ROOT / "runs" / split).glob("*/metrics.json")):
        row = read_json(path)
        validate_digest(row, "metrics_digest")
        rows.append(row)
    return rows


def clock_status(row: dict[str, Any], clock: str) -> str:
    return str(row["formation"]["primary_clocks"][clock]["status"])


def clock_step(row: dict[str, Any], clock: str) -> int | None:
    value = row["formation"]["primary_clocks"][clock]
    return int(value["step"]) if value["status"] == "observed" else None


def matched_factor_effect(rows: list[dict[str, Any]], factor: str, clock: str) -> dict[str, Any]:
    factors = ("task_index", "lexicon_index", "architecture_index", "replicate")
    others = tuple(value for value in factors if value != factor)
    groups: dict[tuple[int, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(int(row[value]) for value in others)
        groups[key][int(row[factor])] = row
    pairs = [group for group in groups.values() if set(group) == {0, 1}]
    discordant = 0
    differences = []
    absolute = []
    for pair in pairs:
        left = clock_step(pair[0], clock)
        right = clock_step(pair[1], clock)
        if (left is None) != (right is None):
            discordant += 1
        if left is not None and right is not None:
            differences.append(int(right - left))
            absolute.append(abs(int(right - left)))
    return {
        "matched_pair_count": len(pairs),
        "status_discordant_count": discordant,
        "status_discordant_fraction": float(discordant / len(pairs)) if pairs else 0.0,
        "both_observed_count": len(differences),
        "signed_step_differences_level1_minus_level0": differences,
        "median_signed_step_difference": float(np.median(differences)) if differences else None,
        "median_absolute_step_difference": float(np.median(absolute)) if absolute else None,
    }


def group_summary(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_clock = {}
    for clock in CLOCKS:
        observed = [row for row in rows if clock_status(row, clock) == "observed"]
        per_clock[clock] = {
            "observed": len(observed),
            "right_censored": len(rows) - len(observed),
            "steps": [clock_step(row, clock) for row in observed],
            "median_step": float(np.median([clock_step(row, clock) for row in observed])) if observed else None,
            "threshold_status_stability_mean": float(
                np.mean([row["formation"]["threshold_status_stability"][clock] for row in rows])
            ),
        }
    per_level = {}
    for factor in ("task_index", "lexicon_index", "architecture_index", "replicate"):
        per_level[factor] = {}
        for level in (0, 1):
            subset = [row for row in rows if int(row[factor]) == level]
            per_level[factor][str(level)] = {
                "run_count": len(subset),
                "R_observed": sum(clock_status(row, "R") == "observed" for row in subset),
                "C_observed": sum(clock_status(row, "C") == "observed" for row in subset),
            }
    factor_effects = {
        factor: {clock: matched_factor_effect(rows, factor, clock) for clock in CLOCKS}
        for factor in ("task_index", "lexicon_index", "architecture_index", "replicate")
    }
    behavior_observed = per_clock["R"]["observed"]
    behavior_breadth = bool(
        behavior_observed >= GROUP_GATES["behavior_observed_per_split_min"]
        and all(
            per_level[factor][str(level)]["R_observed"]
            >= GROUP_GATES["behavior_observed_per_binary_level_min"]
            for factor in per_level
            for level in (0, 1)
        )
    )
    decode_breadth = per_clock["D"]["observed"] >= GROUP_GATES["decode_observed_per_split_min"]
    interface_breadth = per_clock["E"]["observed"] >= GROUP_GATES["interface_observed_per_split_min"]
    threshold_stability = all(
        per_clock[clock]["threshold_status_stability_mean"]
        >= GROUP_GATES["threshold_status_stability_min"]
        for clock in CLOCKS
    )
    finite = all(
        all(
            trajectory["train_behavior"]["finite_fraction"] >= FIXED_CONTROLS["finite_fraction_min"]
            and trajectory["holdout_behavior"]["finite_fraction"] >= FIXED_CONTROLS["finite_fraction_min"]
            for trajectory in row["trajectory"]
        )
        for row in rows
    )
    zero_drift = all(
        all(trajectory["necessity"]["zero_drift_max"] <= FIXED_CONTROLS["zero_drift_max"] for trajectory in row["trajectory"])
        for row in rows
    )
    prefix_eligible = sum(
        clock_status(row, "R") == "observed"
        and row["formation"]["pre_behavior_prefix_count"] >= GROUP_GATES["minimum_pre_behavior_prefixes_for_next_stage"]
        for row in rows
    )
    return {
        "split": split,
        "run_count": len(rows),
        "per_clock": per_clock,
        "per_level": per_level,
        "factor_effects": factor_effects,
        "primary_signature_counts": dict(
            sorted(
                {
                    signature: sum(row["formation"]["primary_signature"] == signature for row in rows)
                    for signature in {row["formation"]["primary_signature"] for row in rows}
                }.items()
            )
        ),
        "pre_behavior_prefix_eligible_count": int(prefix_eligible),
        "gates": {
            "behavior_breadth": behavior_breadth,
            "decode_breadth": decode_breadth,
            "interface_breadth": interface_breadth,
            "threshold_stability": threshold_stability,
            "all_finite": finite,
            "zero_drift": zero_drift,
        },
        "clock_construct_transfer_gate": bool(
            behavior_breadth and decode_breadth and interface_breadth and threshold_stability and finite and zero_drift
        ),
    }


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    expected = 16
    rows = {split: load_rows(split) for split in ("discovery", "confirmation")}
    if any(len(values) != expected for values in rows.values()):
        raise RuntimeError(f"incomplete runs: { {split: len(values) for split, values in rows.items()} }")
    summaries = {split: group_summary(split, values) for split, values in rows.items()}
    transfer = all(summary["clock_construct_transfer_gate"] for summary in summaries.values())
    prefix_ready = all(
        summary["pre_behavior_prefix_eligible_count"] >= GROUP_GATES["minimum_pre_behavior_prefixes_for_next_stage"]
        for summary in summaries.values()
    )
    manifest = []
    for split, values in rows.items():
        for row in values:
            path = OUT_ROOT / "runs" / split / row["run_id"] / "metrics.json"
            manifest.append(
                {
                    "split": split,
                    "run_id": row["run_id"],
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "metrics_digest": row["metrics_digest"],
                }
            )
    result = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "factorial_free_clock_transfer_passed" if transfer else "factorial_free_clock_transfer_not_confirmed",
        "protocol_digest": protocol["protocol_digest"],
        "summaries": summaries,
        "run_manifest": manifest,
        "claims": {
            "known_truth_clock_construct_transferred_to_free_networks": transfer,
            "factor_effects_identified_within_frozen_registry": True,
            "universal_clock_order": "not_claimed",
            "clock_threshold_naturality": "not_claimed",
            "pretrained_language_external_validity": "not_tested",
            "semantic_mechanism": "not_tested",
        },
        "authorized_next": {
            "experiment": "T03_PRECURSOR_INCREMENTAL_PREDICTION" if transfer and prefix_ready else None,
            "scope": "separately preregistered prediction from checkpoints strictly before tau_R",
            "automatic_execution": bool(transfer and prefix_ready),
            "reason": (
                "both splits passed free-network clock transfer and contain enough pre-behavior prefixes"
                if transfer and prefix_ready
                else "clock transfer or pre-behavior prefix breadth gate failed"
            ),
            "pretrained_model_run": False,
        },
        "k_item": {
            "identifier": "K194",
            "evidence_grade": "E3-MICRO" if transfer else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "The six-clock instrument transferred prospectively to a fully crossed free micro-Transformer registry."
                if transfer
                else "The six-clock instrument did not satisfy its preregistered transfer gate in the fully crossed free micro-Transformer registry."
            ),
            "scope": "finite micro-Transformer task-rule, lexicon, architecture, and seed registry",
        },
        "new_mathematics_required": False,
    }
    result["final_digest"] = digest(result)
    write_json(FINAL_PATH, result)
    return result


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    condition = make_condition("discovery", 0, 0)
    train, holdout = split_combinations(condition)
    config = ARCHITECTURES["discovery"]["d4_w80"]
    set_seed(1_217_999)
    model = TinyCausalTransformer(config).cuda()
    controls = initial_camera_controls(model, condition)
    scan = scan_checkpoint(model, condition, controls, 0)
    result = {
        "train_count": len(train),
        "holdout_count": len(holdout),
        "layer_count": len(scan["layers"]),
        "necessity_layer_count": len(scan["necessity"]["layers"]),
        "all_gates_present": all(clock in scan["gates"]["primary"] for clock in CLOCKS),
        "zero_drift": scan["necessity"]["zero_drift_max"],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_split(split: str) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    rows = []
    for task_index, lexicon_index, architecture_index, replicate in itertools.product(range(2), repeat=4):
        rows.append(
            execute_run(split, task_index, lexicon_index, architecture_index, replicate, device)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "preregister", "run", "finalize"), required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"))
    args = parser.parse_args()
    if args.stage == "smoke":
        print(json.dumps(smoke(), indent=2))
    elif args.stage == "preregister":
        print(json.dumps(preregister(), indent=2))
    elif args.stage == "run":
        if args.split is None:
            raise SystemExit("--split is required for run")
        print(json.dumps({"split": args.split, "run_count": len(run_split(args.split))}, indent=2))
    else:
        print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
