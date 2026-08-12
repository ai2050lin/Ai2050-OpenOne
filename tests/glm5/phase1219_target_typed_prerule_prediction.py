#!/usr/bin/env python3
"""Target-typed pre-rule prediction on new free micro-Transformers.

Phase 1219 is a one-shot test of H15.P3/H16.P2 under a frozen observation
contract.  Classification and onset targets have separate qualification
rules.  Every predictor uses only a common step-0..50 prefix; checkpoints are
never treated as independent samples.  Discovery fits the finite predictor
family before any confirmation network is trained.
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

import phase1218_dense_prerule_trajectory_acquisition as source  # noqa: E402
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


core = source.core
PHASE = 1219
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1219_target_typed_prerule_prediction_audit.py")
OUT_ROOT = TEST_ROOT / "result/phase1219_target_typed_prerule_prediction"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
DISCOVERY_MODEL_PATH = OUT_ROOT / "analysis/discovery_model.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
SOURCE1218 = TEST_ROOT / "result/phase1218_dense_prerule_trajectory_acquisition"

EXPECTED_1218_FINAL = "27f482e0a43a2f53b2d5a5dcf7ac6b5b3b96478e61639b71a4fff0c3d3ae7a42"
EXPECTED_1218_AUDIT = "64f143313778497a7c62f88a1881748f340d9500b5001d5d197dba2a40d93e30"

CLOCKS = core.CLOCKS
REPLICATES = 4
LANDMARK_STEP = 50
ANCHOR_INTERVAL = 100
PREFIX_STEPS = tuple(range(0, LANDMARK_STEP + 1, 5))
ANCHOR_STEPS = tuple(range(0, 2401, ANCHOR_INTERVAL))
OBSERVATION_STEPS = tuple(sorted(set(PREFIX_STEPS) | set(ANCHOR_STEPS)))
OBSERVATION_STEP_SET = set(OBSERVATION_STEPS)
SAVED_CHECKPOINT_STEPS = (0, 50, 800, 2400)
CLASSIFICATION_HORIZON = 800
PROBE_RELATIVE_NORM = 5.0e-4
PROBE_BATCH_SIZE = 512
ROUTING_COMBINATION_COUNT = 32
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
NULL_SHIFTS = (1, 2, 3)

# New networks and split-disjoint rules.  Width is fixed at 112, so the
# architecture factor changes depth rather than bundling depth and width.
TASKS = {
    "discovery": (
        {"name": "typed_swap_row_context", "source_roles": {"row": "context", "column": "column", "context": "row"}},
        {"name": "typed_swap_column_context", "source_roles": {"row": "row", "column": "context", "context": "column"}},
    ),
    "confirmation": (
        {"name": "typed_identity", "source_roles": {"row": "row", "column": "column", "context": "context"}},
        {"name": "typed_cycle_forward", "source_roles": {"row": "column", "column": "context", "context": "row"}},
    ),
}

LEXICONS = {
    "discovery": (
        {"name": "typed_lexicon_d0", "seed": 1_219_101},
        {"name": "typed_lexicon_d1", "seed": 1_219_103},
    ),
    "confirmation": (
        {"name": "typed_lexicon_c0", "seed": 1_219_211},
        {"name": "typed_lexicon_c1", "seed": 1_219_223},
    ),
}

ARCHITECTURES = {
    "discovery": {
        "d4_w112": ModelConfig(4, 112, 4, 224, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
        "d6_w112": ModelConfig(6, 112, 4, 224, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
    },
    "confirmation": {
        "d5_w112": ModelConfig(5, 112, 4, 224, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
        "d7_w112": ModelConfig(7, 112, 4, 224, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
    },
}

HOLDOUT_RULES = {
    "discovery": {"coefficients": (1, 3, 2), "bias": 0},
    "confirmation": {"coefficients": (2, 3, 1), "bias": 3},
}

TRAINING = dict(core.TRAINING)
TRAINING.update(
    {
        "maximum_steps": 2400,
        "evaluation_interval": ANCHOR_INTERVAL,
        "observation_steps": OBSERVATION_STEPS,
        "landmark_step": LANDMARK_STEP,
        "no_early_stopping": True,
    }
)

TARGET_GATES = {
    "classification": {
        "systems_per_split": 32,
        "landmark_pre_rule_min": 32,
        "positive_min": 8,
        "negative_min": 8,
    },
    "onset": {
        "observed_min": 16,
        "observed_per_architecture_min": 6,
        "distinct_onsets_min": 3,
    },
    "finite_fraction_min": 1.0,
    "zero_drift_max": 1.0e-6,
}

CONFIRMATION_GATES = {
    "classification_balanced_accuracy_min": 0.70,
    "classification_balanced_accuracy_gain_min": 0.10,
    "classification_brier_gain_min": 0.03,
    "classification_null_balanced_accuracy_advantage_min": 0.02,
    "classification_null_brier_advantage_min": 0.01,
    "onset_mae_gain_steps_min": 100.0,
    "onset_relative_mae_max": 0.80,
    "onset_null_mae_advantage_steps_min": 50.0,
    "onset_within_200_gain_min": 0.10,
}

BASELINE_FACTOR_NAMES = tuple(
    [f"factor_route_{query}_from_{role}" for query in core.FUNCTION_QUERIES for role in core.ROLES]
    + [
        "factor_fixed_point_fraction",
        "factor_task_level",
        "factor_lexicon_level",
        "factor_depth",
        "factor_width",
        "factor_parameter_log",
        "factor_holdout_row",
        "factor_holdout_column",
        "factor_holdout_context",
        "factor_holdout_bias",
    ]
    + [f"factor_replicate_{index}" for index in range(REPLICATES)]
)

BASELINE_FAMILIES = (
    "accuracy",
    "loss",
    "confidence",
    "gradient_norm",
    "parameter_norm",
    "updates",
    "tokens",
    "parameter_token_proxy",
)
BASELINE_SCALAR_NAMES = tuple(
    name for family in BASELINE_FAMILIES for name in (f"scalar_{family}_endpoint", f"scalar_{family}_slope")
)

MECHANISM_FEATURE_NAMES = (
    "mechanism_h11_routing_advantage_endpoint",
    "mechanism_h11_routing_advantage_slope",
    "mechanism_h12_shared_value_attention_endpoint",
    "mechanism_h12_query_differential_endpoint",
    "mechanism_h13_joint_excess_endpoint",
    "mechanism_h13_redundant_layer_fraction_endpoint",
    "mechanism_h14_decode_profile_auc_delta",
    "mechanism_h14_decode_profile_centroid_endpoint",
    "mechanism_h15_functional_path_excess",
    "mechanism_h15_functional_acceleration",
    "mechanism_h16_correct_probe_progress",
    "mechanism_h16_probe_selectivity",
)


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


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch for {field}")


def install_core_overrides() -> None:
    core.PHASE = PHASE
    core.SCRIPT = SCRIPT
    core.AUDIT_SCRIPT = AUDIT_SCRIPT
    core.OUT_ROOT = OUT_ROOT
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.PREAUDIT_PATH = PREAUDIT_PATH
    core.FINAL_PATH = FINAL_PATH
    core.TASKS = TASKS
    core.LEXICONS = LEXICONS
    core.ARCHITECTURES = ARCHITECTURES
    core.HOLDOUT_RULES = HOLDOUT_RULES
    core.TRAINING = TRAINING


install_core_overrides()


def source_gate() -> dict[str, bool]:
    final = read_json(SOURCE1218 / "analysis/final.json")
    audit = read_json(SOURCE1218 / "audit/independent_result_audit.json")
    validate_digest(final, "final_digest")
    validate_digest(audit, "audit_digest")
    return {
        "phase1218_final_frozen": final["final_digest"] == EXPECTED_1218_FINAL,
        "phase1218_audit_frozen": audit["audit_digest"] == EXPECTED_1218_AUDIT,
        "phase1218_audit_passed": audit["all_checks_passed"] is True,
        "phase1218_predictor_not_fitted": final["claims"]["precursor_predictor_fitted"] is False,
        "phase1218_auto_continue_was_false": final["authorized_next"]["automatic_execution"] is False,
        "phase1218_target_type_defect_recorded": (
            final["summaries"]["confirmation"]["prediction_target_counts"]["formed_by_800"]["balanced"] is True
            and final["summaries"]["confirmation"]["prediction_target_counts"]["primary_onset"]["identifiable"] is True
            and final["summaries"]["confirmation"]["gates"]["factor_level_breadth"] is False
        ),
    }


def script_hashes() -> dict[str, str]:
    return {
        "phase1219_main": sha256_file(SCRIPT),
        "phase1219_audit": sha256_file(AUDIT_SCRIPT),
        "phase1218_source": sha256_file(Path(source.__file__)),
        "phase1217_measurement_source": sha256_file(Path(core.__file__)),
        "phase1213_material_source": sha256_file(Path(core.p1213.__file__)),
        "tiny_transformer_source": sha256_file(Path(sys.modules[TinyCausalTransformer.__module__].__file__)),
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "schema_version": "phase1219.target_typed_prerule_prediction.protocol.v1",
        "created_at": utc_now(),
        "purpose": "one-shot prediction of future rule formation from a fixed pre-rule prefix beyond strong nuisance baselines",
        "source_gate": source_gate(),
        "script_hashes": script_hashes(),
        "formal_run_count": 64,
        "runs_per_split": 32,
        "replicates": REPLICATES,
        "tasks": TASKS,
        "lexicons": LEXICONS,
        "architectures": {split: {name: asdict(config) for name, config in rows.items()} for split, rows in ARCHITECTURES.items()},
        "holdout_rules": HOLDOUT_RULES,
        "training": TRAINING,
        "observation_contract": {
            "prefix_grid": PREFIX_STEPS,
            "anchor_grid": ANCHOR_STEPS,
            "observation_grid": OBSERVATION_STEPS,
            "saved_checkpoint_steps": SAVED_CHECKPOINT_STEPS,
            "clock_outcomes_use_anchor_grid_only": True,
            "prediction_input_step_max": LANDMARK_STEP,
            "prediction_unit": "system, never checkpoint",
            "confirmation_trained_only_after_discovery_model_freeze": True,
        },
        "target_contract": {
            "classification": {
                "name": "formed_by_800",
                "eligible": "all systems whose primary R gate is closed throughout steps 0..50",
                "positive": "anchor-defined primary R onset <= 800",
                "negative": "anchor-defined primary R onset > 800 or right-censored at 2400",
            },
            "onset": {
                "name": "primary_R_onset",
                "eligible": "only systems with observed primary R onset strictly after step 50",
                "right_censored_systems": "excluded from onset regression but retained in classification",
            },
            "gates": TARGET_GATES,
        },
        "baseline_contract": {
            "factor_features": BASELINE_FACTOR_NAMES,
            "scalar_families": BASELINE_FAMILIES,
            "scalar_features": BASELINE_SCALAR_NAMES,
            "strong_baseline": "factor plus scalar",
        },
        "mechanism_contract": {
            "families": (
                "H11 conditional routing",
                "H12 shared-differential RDC",
                "H13 single-joint redundancy",
                "H14 functional quotient",
                "H15 dynamic trajectory",
                "H16 local formation sensitivity",
            ),
            "features": MECHANISM_FEATURE_NAMES,
            "probe_relative_parameter_norm": PROBE_RELATIVE_NORM,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "routing_combination_count": ROUTING_COMBINATION_COUNT,
            "matched_null_shifts_within_exact_factor_cell": NULL_SHIFTS,
        },
        "predictor_contract": {
            "model": "ridge linear probability/regression with unpenalized intercept",
            "ridge_grid": RIDGE_GRID,
            "selection": "eight leave-one-factor-cell-out discovery folds; minimize Brier for classification and MAE for onset; tie chooses stronger regularization",
            "discovery_model_frozen_before_confirmation_training": True,
            "confirmation_gates": CONFIRMATION_GATES,
        },
        "claims_allowed": (
            "whether the frozen mechanism summaries predict formed_by_800 beyond the strong nuisance baseline on unseen tasks, lexicons, depths, and seeds",
            "whether the separately qualified onset predictor improves on the same baseline among observed-onset systems",
            "a contract-scoped support or negative boundary for H15.P3/H16.P2",
        ),
        "forbidden": (
            "changing features, targets, horizon, thresholds, seeds, tasks, architectures, ridge grid, or nulls after any run",
            "using checkpoints as independent samples",
            "dropping right-censored systems from classification",
            "requiring observed onset breadth to authorize classification",
            "reading confirmation outcomes before the discovery model is frozen",
            "claiming a universal formation law, natural-language mechanism, pretrained-model validity, or new mathematics",
        ),
    }


def preregister() -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        existing = read_json(PROTOCOL_PATH)
        validate_digest(existing, "protocol_digest")
        if existing["script_hashes"] != script_hashes():
            raise RuntimeError("frozen Phase 1219 protocol script hashes differ from current scripts")
        return existing
    if DISCOVERY_MODEL_PATH.exists() or (OUT_ROOT / "runs").exists():
        raise RuntimeError("results exist before Phase 1219 preregistration")
    payload = protocol_payload()
    if not all(payload["source_gate"].values()):
        raise RuntimeError(f"Phase 1218 source gate failed: {payload['source_gate']}")
    payload["protocol_digest"] = digest(payload)
    write_json(PROTOCOL_PATH, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_digest(protocol, "protocol_digest")
    if protocol["script_hashes"] != script_hashes():
        raise RuntimeError("script hash drift after preregistration")
    preaudit = read_json(PREAUDIT_PATH)
    validate_digest(preaudit, "audit_digest")
    if preaudit["all_checks_passed"] is not True or preaudit["protocol_digest"] != protocol["protocol_digest"]:
        raise RuntimeError("independent preaudit did not authorize execution")
    return protocol


def model_seed(split: str, task_index: int, lexicon_index: int, architecture_index: int, replicate: int) -> int:
    base = 1_219_300_000 if split == "discovery" else 1_219_700_000
    return base + task_index * 1_000_003 + lexicon_index * 100_003 + architecture_index * 10_007 + replicate * 1_009


def run_id(split: str, condition: dict[str, Any], architecture: str, replicate: int) -> str:
    return f"{split}__{condition['name']}__{condition['lexicon_name']}__{architecture}__s{replicate}"


def parameter_norm(model: torch.nn.Module) -> float:
    squared = sum(float(torch.sum(value.detach().float().square()).item()) for value in model.parameters())
    return float(np.sqrt(squared))


def attention_routing_camera(model: TinyCausalTransformer, condition: dict[str, Any]) -> dict[str, Any]:
    _, holdout = core.split_combinations(condition)
    combinations = holdout[:ROUTING_COMBINATION_COUNT]
    inputs, _, metadata = core.build_examples(condition, combinations, (4, 5), core.FUNCTION_QUERIES)
    device = next(model.parameters()).device
    ids = inputs.to(device, non_blocking=True)
    source_positions = []
    queries = []
    for _, template_index, query in metadata:
        order = core.TEMPLATES[int(template_index)]
        source = core.source_role(condition, query)
        source_positions.append(2 + 2 * order.index(source))
        queries.append(query)
    source_tensor = torch.tensor(source_positions, dtype=torch.long, device=device)
    value_positions = torch.tensor((2, 4, 6), dtype=torch.long, device=device)
    hidden = model.embed(ids)
    rows: list[dict[str, Any]] = []
    per_query_values: dict[str, list[float]] = {query: [] for query in core.FUNCTION_QUERIES}
    model.eval()
    with torch.inference_mode():
        for layer_index, block in enumerate(model.blocks):
            normalized = block.attn_norm(hidden)
            batch, length, width = normalized.shape
            qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
            query_tensor, key_tensor, _ = qkv.unbind(dim=2)
            query_tensor = query_tensor.transpose(1, 2)
            key_tensor = key_tensor.transpose(1, 2)
            scores = torch.matmul(query_tensor[:, :, -1:, :].float(), key_tensor.transpose(-2, -1).float())
            scores = scores / math.sqrt(block.attn.head_dim)
            weights = torch.softmax(scores.squeeze(2), dim=-1)
            selected = weights.gather(2, source_tensor[:, None, None].expand(-1, block.attn.heads, 1)).squeeze(2)
            values = weights.index_select(2, value_positions)
            distractor = (values.sum(dim=2) - selected) / 2.0
            advantage = selected - distractor
            for head in range(block.attn.heads):
                head_advantage = advantage[:, head]
                query_means = {}
                for query_name in core.FUNCTION_QUERIES:
                    mask = torch.tensor([value == query_name for value in queries], dtype=torch.bool, device=device)
                    query_mean = float(head_advantage[mask].mean().item())
                    query_means[query_name] = query_mean
                    per_query_values[query_name].append(query_mean)
                rows.append(
                    {
                        "layer": layer_index,
                        "head": head,
                        "relative_depth": float((layer_index + 1) / model.config.layers),
                        "selected_attention": float(selected[:, head].mean().item()),
                        "distractor_attention": float(distractor[:, head].mean().item()),
                        "advantage": float(head_advantage.mean().item()),
                        "per_query_advantage": query_means,
                    }
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = block(hidden)
    advantages = np.asarray([row["advantage"] for row in rows], dtype=np.float64)
    selected = np.asarray([row["selected_attention"] for row in rows], dtype=np.float64)
    positive = np.maximum(advantages, 0.0)
    depths = np.asarray([row["relative_depth"] for row in rows], dtype=np.float64)
    per_query = {name: float(np.mean(values)) for name, values in per_query_values.items()}
    return {
        "case_count": len(metadata),
        "rows": rows,
        "mean_selected_attention": float(np.mean(selected)),
        "mean_advantage": float(np.mean(advantages)),
        "max_head_advantage": float(np.max(advantages)),
        "positive_head_fraction": float(np.mean(advantages > 0.0)),
        "advantage_centroid_relative_depth": float(np.sum(depths * positive) / np.sum(positive)) if np.sum(positive) > 0 else 0.0,
        "per_query_advantage": per_query,
        "query_differential": float(np.std(list(per_query.values()))),
    }


def mechanism_camera(model: TinyCausalTransformer, condition: dict[str, Any], scan: dict[str, Any]) -> dict[str, Any]:
    routing = attention_routing_camera(model, condition)
    decode_profile = np.asarray(
        [min(layer["validation"]["combined_accuracy"], layer["holdout"]["combined_accuracy"]) for layer in scan["layers"]],
        dtype=np.float64,
    )
    initial_profile = np.asarray(
        [layer["initial_validation"]["combined_accuracy"] for layer in scan["layers"]], dtype=np.float64
    )
    corrected = np.maximum(decode_profile - initial_profile, 0.0)
    decode_depths = np.asarray([layer["relative_depth"] for layer in scan["layers"]], dtype=np.float64)
    necessity_rows = scan["necessity"]["layers"]
    joint_excess = np.asarray([row["joint_excess_over_best_single"] for row in necessity_rows], dtype=np.float64)
    single_damage = np.asarray([row["max_single_accuracy_damage"] for row in necessity_rows], dtype=np.float64)
    joint_damage = np.asarray([row["joint_accuracy_damage"] for row in necessity_rows], dtype=np.float64)
    return {
        "routing": routing,
        "rdc": {
            "shared_value_attention": routing["mean_selected_attention"],
            "query_differential": routing["query_differential"],
        },
        "redundancy": {
            "joint_excess_max": float(np.max(joint_excess)),
            "joint_excess_mean": float(np.mean(joint_excess)),
            "redundant_layer_fraction": float(np.mean(joint_damage > single_damage + 0.01)),
            "single_damage_max": float(np.max(single_damage)),
            "joint_damage_max": float(np.max(joint_damage)),
        },
        "functional_quotient": {
            "decode_profile_auc": float(np.mean(corrected)),
            "decode_profile_max": float(np.max(corrected)),
            "decode_profile_centroid": float(np.sum(decode_depths * corrected) / np.sum(corrected)) if np.sum(corrected) > 0 else 0.0,
            "decode_profile_dispersion": float(np.std(corrected)),
        },
    }


PROGRESS_KEYS = (
    "rule_accuracy",
    "minimum_correct_probability",
    "decode_accuracy",
    "transfer_success",
    "preservation_success",
    "single_necessity",
    "joint_necessity",
    "routing_advantage",
)


def progress_vector(scan: dict[str, Any]) -> dict[str, float]:
    values = {name: float(scan["metrics"][name]) for name in PROGRESS_KEYS[:-1]}
    values["routing_advantage"] = float(scan["mechanism_camera"]["routing"]["max_head_advantage"])
    return values


def response_difference(probe: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    left, right = progress_vector(probe), progress_vector(base)
    return {name: float(left[name] - right[name]) for name in PROGRESS_KEYS}


def mean_progress(response: dict[str, float]) -> float:
    return float(np.mean([response[name] for name in PROGRESS_KEYS]))


def local_gradient_probe(
    model: TinyCausalTransformer,
    condition: dict[str, Any],
    base_scan: dict[str, Any],
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    candidates: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 91_919)
    indices = torch.randperm(len(train_inputs), generator=generator)[:PROBE_BATCH_SIZE]
    ids = train_inputs[indices].to(device, non_blocking=True)
    targets = train_targets[indices].to(device, non_blocking=True)
    params = [value for value in model.parameters() if value.requires_grad]
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(ids)[:, -1].index_select(-1, candidates)
        loss = F.cross_entropy(logits.float(), targets)
    gradients = torch.autograd.grad(loss, params)
    gradient_norm = float(torch.sqrt(sum(torch.sum(value.detach().float().square()) for value in gradients)).item())
    original = [value.detach().clone() for value in params]
    epsilon = float(PROBE_RELATIVE_NORM * parameter_norm(model))
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(seed + 92_929)
    random_values = [
        torch.randn(value.shape, generator=random_generator, device=device, dtype=torch.float32) for value in params
    ]
    random_norm = float(torch.sqrt(sum(torch.sum(value.square()) for value in random_values)).item())

    directions = {
        "correct": [(-epsilon / max(gradient_norm, 1.0e-12)) * value.detach().float() for value in gradients],
        "anti": [(epsilon / max(gradient_norm, 1.0e-12)) * value.detach().float() for value in gradients],
        "random": [(epsilon / max(random_norm, 1.0e-12)) * value for value in random_values],
    }
    responses: dict[str, Any] = {}
    for name, deltas in directions.items():
        with torch.no_grad():
            for parameter, baseline, delta in zip(params, original, deltas):
                parameter.copy_(baseline + delta.to(dtype=parameter.dtype))
        probe_scan = core.scan_checkpoint(model, condition, source_controls_for_probe(base_scan), LANDMARK_STEP)
        probe_scan["mechanism_camera"] = mechanism_camera(model, condition, probe_scan)
        responses[name] = response_difference(probe_scan, base_scan)
    with torch.no_grad():
        for parameter, baseline in zip(params, original):
            parameter.copy_(baseline)
    drift = max(float(torch.max(torch.abs(parameter.detach() - baseline)).item()) for parameter, baseline in zip(params, original))
    model.zero_grad(set_to_none=True)
    del original, random_values, directions, gradients
    gc.collect()
    torch.cuda.empty_cache()
    progress = {name: mean_progress(value) for name, value in responses.items()}
    return {
        "loss": float(loss.item()),
        "gradient_norm": gradient_norm,
        "relative_parameter_norm": PROBE_RELATIVE_NORM,
        "absolute_delta_norm": epsilon,
        "responses": responses,
        "mean_progress": progress,
        "selectivity": float(progress["correct"] - max(progress["anti"], progress["random"])),
        "restore_drift_max": drift,
    }


def source_controls_for_probe(base_scan: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(layer["initial_validation"]) for layer in base_scan["layers"]]


def summarize_trajectory(trajectory: list[dict[str, Any]], count: int) -> dict[str, Any]:
    anchors = [row for row in trajectory if int(row["step"]) in set(ANCHOR_STEPS)]
    if [int(row["step"]) for row in anchors] != list(ANCHOR_STEPS):
        raise RuntimeError("anchor trajectory is incomplete")
    result = core.summarize_trajectory(anchors, count)
    prefix = [row for row in trajectory if int(row["step"]) <= LANDMARK_STEP]
    result["landmark_pre_rule"] = bool(all(not row["gates"]["primary"]["R"] for row in prefix))
    result["landmark_observation_count"] = len(prefix)
    return result


def execute_run(
    split: str,
    task_index: int,
    lexicon_index: int,
    architecture_index: int,
    replicate: int,
    device: torch.device,
) -> dict[str, Any]:
    protocol = verify_protocol()
    if split == "confirmation" and not DISCOVERY_MODEL_PATH.exists():
        raise RuntimeError("confirmation execution forbidden before discovery model freeze")
    condition = core.make_condition(split, task_index, lexicon_index)
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
    core.set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    count = core.parameter_count(model)
    train_combinations, _ = core.split_combinations(condition)
    train_inputs, train_targets, _ = core.build_examples(condition, train_combinations, range(len(core.TEMPLATES)))
    candidates = core.candidate_ids(condition, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(TRAINING["learning_rate"]), weight_decay=float(TRAINING["weight_decay"])
    )
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(seed + 37)
    initial_controls = core.initial_camera_controls(model, condition)
    trajectory: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    def record(step: int, loss: float | None, gradient_norm: float | None) -> None:
        scan = core.scan_checkpoint(model, condition, initial_controls, step)
        scan["loss"] = loss
        scan["gradient_norm"] = gradient_norm
        scan["parameter_norm"] = parameter_norm(model)
        scan["updates"] = int(step)
        scan["parameter_token_proxy"] = int(scan["tokens_seen"] * count)
        if step <= LANDMARK_STEP:
            scan["mechanism_camera"] = mechanism_camera(model, condition, scan)
        if step == LANDMARK_STEP:
            scan["local_gradient_probe"] = local_gradient_probe(
                model, condition, scan, train_inputs, train_targets, candidates, seed
            )
        trajectory.append(scan)
        if step in SAVED_CHECKPOINT_STEPS:
            checkpoint_path = run_root / "checkpoints" / f"step_{step:04d}.pt"
            core.write_checkpoint(
                checkpoint_path,
                core.checkpoint_payload(model, config, identifier, step, protocol["protocol_digest"]),
            )
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
            + f" holdout={scan['holdout_behavior']['accuracy']:.4f}",
            flush=True,
        )

    record(0, None, None)
    last_loss: float | None = None
    last_gradient: float | None = None
    for step in range(1, int(TRAINING["maximum_steps"]) + 1):
        model.train()
        indices = torch.randint(0, len(train_inputs), (int(TRAINING["batch_size"]),), generator=batch_generator)
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
        if step in OBSERVATION_STEP_SET:
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
        "holdout_rule": HOLDOUT_RULES[split],
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


def clock_step(row: dict[str, Any], clock: str = "R") -> int | None:
    value = row["formation"]["primary_clocks"][clock]
    return int(value["step"]) if value["status"] == "observed" else None


def formed_by_800(row: dict[str, Any]) -> int:
    onset = clock_step(row)
    return int(onset is not None and onset <= CLASSIFICATION_HORIZON)


def prefix_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [value for value in row["trajectory"] if int(value["step"]) <= LANDMARK_STEP]
    if [int(value["step"]) for value in rows] != list(PREFIX_STEPS):
        raise RuntimeError(f"invalid prefix for {row['run_id']}")
    return rows


def endpoint_and_slope(points: list[tuple[int, float]]) -> tuple[float, float]:
    valid = [(step, value) for step, value in points if value is not None and math.isfinite(float(value))]
    if not valid:
        return 0.0, 0.0
    endpoint = float(valid[-1][1])
    if len(valid) < 2:
        return endpoint, 0.0
    x = np.asarray([step / LANDMARK_STEP for step, _ in valid], dtype=np.float64)
    y = np.asarray([value for _, value in valid], dtype=np.float64)
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / max(np.sum((x - x.mean()) ** 2), 1.0e-12))
    return endpoint, slope


def factor_features(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    fixed = 0
    for query in core.FUNCTION_QUERIES:
        source_role = row["source_roles"][query]
        fixed += int(query == source_role)
        for role in core.ROLES:
            values[f"factor_route_{query}_from_{role}"] = float(source_role == role)
    config = row["config"]
    holdout = row["holdout_rule"]
    values.update(
        {
            "factor_fixed_point_fraction": float(fixed / len(core.FUNCTION_QUERIES)),
            "factor_task_level": float(row["task_index"]),
            "factor_lexicon_level": float(row["lexicon_index"]),
            "factor_depth": float(config["layers"]),
            "factor_width": float(config["width"]),
            "factor_parameter_log": float(math.log(max(row["parameter_count"], 1))),
            "factor_holdout_row": float(holdout["coefficients"][0]),
            "factor_holdout_column": float(holdout["coefficients"][1]),
            "factor_holdout_context": float(holdout["coefficients"][2]),
            "factor_holdout_bias": float(holdout["bias"]),
        }
    )
    for index in range(REPLICATES):
        values[f"factor_replicate_{index}"] = float(row["replicate"] == index)
    if tuple(values) != BASELINE_FACTOR_NAMES:
        raise RuntimeError("factor feature schema drift")
    return values


def scalar_features(row: dict[str, Any]) -> dict[str, float]:
    prefix = prefix_rows(row)
    extractors = {
        "accuracy": lambda point: point["metrics"]["rule_accuracy"],
        "loss": lambda point: point["loss"],
        "confidence": lambda point: point["metrics"]["minimum_correct_probability"],
        "gradient_norm": lambda point: point["gradient_norm"],
        "parameter_norm": lambda point: point["parameter_norm"],
        "updates": lambda point: point["updates"],
        "tokens": lambda point: point["tokens_seen"],
        "parameter_token_proxy": lambda point: point["parameter_token_proxy"],
    }
    values: dict[str, float] = {}
    for family in BASELINE_FAMILIES:
        endpoint, slope = endpoint_and_slope([(int(point["step"]), extractors[family](point)) for point in prefix])
        values[f"scalar_{family}_endpoint"] = endpoint
        values[f"scalar_{family}_slope"] = slope
    if tuple(values) != BASELINE_SCALAR_NAMES:
        raise RuntimeError("scalar feature schema drift")
    return values


def mechanism_features(row: dict[str, Any]) -> dict[str, float]:
    prefix = prefix_rows(row)
    endpoint = prefix[-1]
    initial = prefix[0]
    route_endpoint, route_slope = endpoint_and_slope(
        [(int(point["step"]), point["mechanism_camera"]["routing"]["max_head_advantage"]) for point in prefix]
    )
    joint_endpoint = float(endpoint["mechanism_camera"]["redundancy"]["joint_excess_max"])
    redundant_endpoint = float(endpoint["mechanism_camera"]["redundancy"]["redundant_layer_fraction"])
    decode_delta = float(
        endpoint["mechanism_camera"]["functional_quotient"]["decode_profile_auc"]
        - initial["mechanism_camera"]["functional_quotient"]["decode_profile_auc"]
    )
    decode_centroid = float(endpoint["mechanism_camera"]["functional_quotient"]["decode_profile_centroid"])
    vectors = []
    for point in prefix:
        camera = point["mechanism_camera"]
        vectors.append(
            np.asarray(
                [
                    camera["routing"]["max_head_advantage"],
                    camera["rdc"]["shared_value_attention"],
                    camera["rdc"]["query_differential"],
                    camera["redundancy"]["joint_excess_max"],
                    camera["redundancy"]["redundant_layer_fraction"],
                    camera["functional_quotient"]["decode_profile_auc"],
                    camera["functional_quotient"]["decode_profile_centroid"],
                ],
                dtype=np.float64,
            )
        )
    path_length = float(sum(np.linalg.norm(right - left) for left, right in zip(vectors[:-1], vectors[1:])))
    displacement = float(np.linalg.norm(vectors[-1] - vectors[0]))
    path_excess = float(max(0.0, path_length - displacement))
    midpoint = vectors[PREFIX_STEPS.index(25)]
    acceleration = float(np.linalg.norm(vectors[-1] - 2.0 * midpoint + vectors[0]))
    probe = endpoint["local_gradient_probe"]
    values = {
        "mechanism_h11_routing_advantage_endpoint": route_endpoint,
        "mechanism_h11_routing_advantage_slope": route_slope,
        "mechanism_h12_shared_value_attention_endpoint": float(endpoint["mechanism_camera"]["rdc"]["shared_value_attention"]),
        "mechanism_h12_query_differential_endpoint": float(endpoint["mechanism_camera"]["rdc"]["query_differential"]),
        "mechanism_h13_joint_excess_endpoint": joint_endpoint,
        "mechanism_h13_redundant_layer_fraction_endpoint": redundant_endpoint,
        "mechanism_h14_decode_profile_auc_delta": decode_delta,
        "mechanism_h14_decode_profile_centroid_endpoint": decode_centroid,
        "mechanism_h15_functional_path_excess": path_excess,
        "mechanism_h15_functional_acceleration": acceleration,
        "mechanism_h16_correct_probe_progress": float(probe["mean_progress"]["correct"]),
        "mechanism_h16_probe_selectivity": float(probe["selectivity"]),
    }
    if tuple(values) != MECHANISM_FEATURE_NAMES:
        raise RuntimeError("mechanism feature schema drift")
    return values


def feature_record(row: dict[str, Any]) -> dict[str, Any]:
    factors = factor_features(row)
    scalars = scalar_features(row)
    mechanisms = mechanism_features(row)
    onset = clock_step(row)
    return {
        "run_id": row["run_id"],
        "split": row["split"],
        "cell": [int(row["task_index"]), int(row["lexicon_index"]), int(row["architecture_index"])],
        "replicate": int(row["replicate"]),
        "architecture_index": int(row["architecture_index"]),
        "landmark_pre_rule": bool(row["formation"]["landmark_pre_rule"]),
        "formed_by_800": formed_by_800(row),
        "primary_onset": onset,
        "factor": factors,
        "scalar": scalars,
        "mechanism": mechanisms,
    }


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def target_qualification(records: list[dict[str, Any]]) -> dict[str, Any]:
    classification = [row for row in records if row["landmark_pre_rule"]]
    positives = sum(int(row["formed_by_800"]) for row in classification)
    negatives = len(classification) - positives
    onset = [row for row in records if row["landmark_pre_rule"] and row["primary_onset"] is not None and row["primary_onset"] > LANDMARK_STEP]
    per_arch = {
        str(index): sum(row["architecture_index"] == index for row in onset) for index in range(2)
    }
    class_gate = bool(
        len(records) == TARGET_GATES["classification"]["systems_per_split"]
        and len(classification) >= TARGET_GATES["classification"]["landmark_pre_rule_min"]
        and positives >= TARGET_GATES["classification"]["positive_min"]
        and negatives >= TARGET_GATES["classification"]["negative_min"]
        and all(all_finite(row) for row in records)
    )
    onset_gate = bool(
        len(onset) >= TARGET_GATES["onset"]["observed_min"]
        and all(value >= TARGET_GATES["onset"]["observed_per_architecture_min"] for value in per_arch.values())
        and len({row["primary_onset"] for row in onset}) >= TARGET_GATES["onset"]["distinct_onsets_min"]
        and all(all_finite(row) for row in records)
    )
    factor_balance = {}
    for factor in ("task", "lexicon", "architecture"):
        index = {"task": 0, "lexicon": 1, "architecture": 2}[factor]
        factor_balance[factor] = {
            str(level): {
                "positive": sum(row["cell"][index] == level and row["formed_by_800"] == 1 for row in classification),
                "negative": sum(row["cell"][index] == level and row["formed_by_800"] == 0 for row in classification),
            }
            for level in range(2)
        }
    return {
        "classification": {
            "eligible": len(classification),
            "positive": positives,
            "negative": negatives,
            "authorized": class_gate,
            "factor_balance_descriptive_only": factor_balance,
        },
        "onset": {
            "eligible_observed": len(onset),
            "right_censored_or_late_count": len(records) - len(onset),
            "per_architecture": per_arch,
            "distinct_onsets": sorted({row["primary_onset"] for row in onset}),
            "authorized": onset_gate,
        },
    }


def feature_names(kind: str) -> tuple[str, ...]:
    if kind == "factor":
        return BASELINE_FACTOR_NAMES
    if kind == "scalar":
        return BASELINE_SCALAR_NAMES
    if kind == "baseline":
        return BASELINE_FACTOR_NAMES + BASELINE_SCALAR_NAMES
    if kind == "augmented":
        return BASELINE_FACTOR_NAMES + BASELINE_SCALAR_NAMES + MECHANISM_FEATURE_NAMES
    raise ValueError(kind)


def matrix(
    records: list[dict[str, Any]],
    kind: str,
    null_shift: int | None = None,
    null_pool: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    names = feature_names(kind)
    mechanism_sources = records
    if null_shift is not None:
        pool = null_pool if null_pool is not None else records
        lookup = {(tuple(row["cell"]), row["replicate"]): row for row in pool}
        mechanism_sources = [lookup[(tuple(row["cell"]), (row["replicate"] + null_shift) % REPLICATES)] for row in records]
    rows = []
    for row, mechanism_source in zip(records, mechanism_sources):
        merged = {**row["factor"], **row["scalar"], **mechanism_source["mechanism"]}
        rows.append([float(merged[name]) for name in names])
    return np.asarray(rows, dtype=np.float64)


def target_vector(records: list[dict[str, Any]], target: str) -> np.ndarray:
    if target == "classification":
        return np.asarray([row["formed_by_800"] for row in records], dtype=np.float64)
    return np.asarray([row["primary_onset"] for row in records], dtype=np.float64)


def standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return mean, scale


def ridge_fit(values: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, Any]:
    mean, scale = standardize_fit(values)
    normalized = (values - mean) / scale
    design = np.concatenate((np.ones((len(values), 1)), normalized), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {
        "alpha": float(alpha),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
    }


def ridge_predict(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    normalized = (values - mean) / scale
    design = np.concatenate((np.ones((len(values), 1)), normalized), axis=1)
    return design @ coefficient


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(predictions, 0.0, 1.0)
    labels = probabilities >= 0.5
    truth = targets >= 0.5
    positive = truth
    negative = ~truth
    tpr = float(np.mean(labels[positive] == truth[positive])) if np.any(positive) else 0.0
    tnr = float(np.mean(labels[negative] == truth[negative])) if np.any(negative) else 0.0
    pairwise = [float(predictions[i] > predictions[j]) + 0.5 * float(predictions[i] == predictions[j]) for i in np.where(positive)[0] for j in np.where(negative)[0]]
    return {
        "accuracy": float(np.mean(labels == truth)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "positive_recall": tpr,
        "negative_recall": tnr,
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "auc_pairwise": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def onset_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = np.abs(predictions - targets)
    return {
        "mae_steps": float(np.mean(errors)),
        "median_absolute_error_steps": float(np.median(errors)),
        "within_100_fraction": float(np.mean(errors <= 100.0)),
        "within_200_fraction": float(np.mean(errors <= 200.0)),
    }


def cell_folds(records: list[dict[str, Any]]) -> list[np.ndarray]:
    cells = sorted({tuple(row["cell"]) for row in records})
    return [np.asarray([index for index, row in enumerate(records) if tuple(row["cell"]) == cell], dtype=np.int64) for cell in cells]


def select_alpha(
    records: list[dict[str, Any]],
    kind: str,
    target: str,
    null_shift: int | None = None,
    null_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = matrix(records, kind, null_shift, null_pool)
    targets = target_vector(records, target)
    folds = cell_folds(records)
    candidates = []
    for alpha in RIDGE_GRID:
        predictions = np.zeros(len(records), dtype=np.float64)
        for test_indices in folds:
            train_mask = np.ones(len(records), dtype=bool)
            train_mask[test_indices] = False
            model = ridge_fit(values[train_mask], targets[train_mask], alpha)
            predictions[test_indices] = ridge_predict(model, values[test_indices])
        metrics = classification_metrics(targets, predictions) if target == "classification" else onset_metrics(targets, predictions)
        score = metrics["brier"] if target == "classification" else metrics["mae_steps"]
        candidates.append({"alpha": float(alpha), "score": float(score), "metrics": metrics})
    best_score = min(row["score"] for row in candidates)
    tied = [row for row in candidates if row["score"] <= best_score + (0.005 if target == "classification" else 5.0)]
    selected = max(tied, key=lambda row: row["alpha"])
    return {"selected_alpha": selected["alpha"], "selected_metrics": selected["metrics"], "candidates": candidates}


def fit_named_model(
    records: list[dict[str, Any]],
    kind: str,
    target: str,
    null_shift: int | None = None,
    null_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selection = select_alpha(records, kind, target, null_shift, null_pool)
    values = matrix(records, kind, null_shift, null_pool)
    targets = target_vector(records, target)
    model = ridge_fit(values, targets, selection["selected_alpha"])
    return {
        "kind": kind,
        "target": target,
        "null_shift": null_shift,
        "feature_names": feature_names(kind),
        "cross_validation": selection,
        "model": model,
    }


def fit_discovery() -> dict[str, Any]:
    protocol = verify_protocol()
    if DISCOVERY_MODEL_PATH.exists():
        existing = read_json(DISCOVERY_MODEL_PATH)
        validate_digest(existing, "model_digest")
        return existing
    if (OUT_ROOT / "runs/confirmation").exists():
        raise RuntimeError("confirmation data exists before discovery model freeze")
    rows = load_rows("discovery")
    if len(rows) != 32:
        raise RuntimeError(f"expected 32 discovery rows, found {len(rows)}")
    records = [feature_record(row) for row in rows]
    qualification = target_qualification(records)
    models: dict[str, Any] = {}
    for target in ("classification", "onset"):
        if not qualification[target]["authorized"]:
            continue
        eligible = records if target == "classification" else [row for row in records if row["primary_onset"] is not None and row["primary_onset"] > LANDMARK_STEP]
        models[target] = {
            "factor": fit_named_model(eligible, "factor", target),
            "scalar": fit_named_model(eligible, "scalar", target),
            "baseline": fit_named_model(eligible, "baseline", target),
            "augmented": fit_named_model(eligible, "augmented", target),
            "matched_nulls": [
                fit_named_model(eligible, "augmented", target, shift, records) for shift in NULL_SHIFTS
            ],
        }
    result = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "split": "discovery",
        "run_count": len(rows),
        "run_metrics_digests": {row["run_id"]: row["metrics_digest"] for row in rows},
        "feature_schema": {
            "factor": BASELINE_FACTOR_NAMES,
            "scalar": BASELINE_SCALAR_NAMES,
            "mechanism": MECHANISM_FEATURE_NAMES,
        },
        "records_digest": digest(records),
        "qualification": qualification,
        "models": models,
        "confirmation_seen": False,
    }
    result["model_digest"] = digest(result)
    write_json(DISCOVERY_MODEL_PATH, result)
    return result


def evaluate_named_model(
    specification: dict[str, Any],
    records: list[dict[str, Any]],
    null_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = matrix(records, specification["kind"], specification["null_shift"], null_pool)
    targets = target_vector(records, specification["target"])
    predictions = ridge_predict(specification["model"], values)
    metrics = classification_metrics(targets, predictions) if specification["target"] == "classification" else onset_metrics(targets, predictions)
    return {
        "metrics": metrics,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
    }


def target_confirmation(target: str, models: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = records if target == "classification" else [row for row in records if row["primary_onset"] is not None and row["primary_onset"] > LANDMARK_STEP]
    evaluations = {
        name: evaluate_named_model(models[name], eligible) for name in ("factor", "scalar", "baseline", "augmented")
    }
    nulls = [evaluate_named_model(value, eligible, records) for value in models["matched_nulls"]]
    baseline = evaluations["baseline"]["metrics"]
    augmented = evaluations["augmented"]["metrics"]
    if target == "classification":
        best_null_ba = max(value["metrics"]["balanced_accuracy"] for value in nulls)
        best_null_brier = min(value["metrics"]["brier"] for value in nulls)
        gates = {
            "balanced_accuracy_absolute": augmented["balanced_accuracy"] >= CONFIRMATION_GATES["classification_balanced_accuracy_min"],
            "balanced_accuracy_increment": augmented["balanced_accuracy"] - baseline["balanced_accuracy"] >= CONFIRMATION_GATES["classification_balanced_accuracy_gain_min"],
            "brier_increment": baseline["brier"] - augmented["brier"] >= CONFIRMATION_GATES["classification_brier_gain_min"],
            "matched_null_balanced_accuracy": augmented["balanced_accuracy"] - best_null_ba >= CONFIRMATION_GATES["classification_null_balanced_accuracy_advantage_min"],
            "matched_null_brier": best_null_brier - augmented["brier"] >= CONFIRMATION_GATES["classification_null_brier_advantage_min"],
        }
    else:
        best_null_mae = min(value["metrics"]["mae_steps"] for value in nulls)
        gates = {
            "mae_absolute_increment": baseline["mae_steps"] - augmented["mae_steps"] >= CONFIRMATION_GATES["onset_mae_gain_steps_min"],
            "mae_relative_increment": augmented["mae_steps"] <= baseline["mae_steps"] * CONFIRMATION_GATES["onset_relative_mae_max"],
            "matched_null_mae": best_null_mae - augmented["mae_steps"] >= CONFIRMATION_GATES["onset_null_mae_advantage_steps_min"],
            "within_200_increment": augmented["within_200_fraction"] - baseline["within_200_fraction"] >= CONFIRMATION_GATES["onset_within_200_gain_min"],
        }
    return {
        "eligible_count": len(eligible),
        "evaluations": evaluations,
        "matched_nulls": nulls,
        "gates": gates,
        "passed": all(gates.values()),
    }


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery_model = read_json(DISCOVERY_MODEL_PATH)
    validate_digest(discovery_model, "model_digest")
    rows_by_split = {split: load_rows(split) for split in ("discovery", "confirmation")}
    if any(len(rows) != 32 for rows in rows_by_split.values()):
        raise RuntimeError(f"incomplete runs: { {split: len(rows) for split, rows in rows_by_split.items()} }")
    records = {split: [feature_record(row) for row in rows] for split, rows in rows_by_split.items()}
    qualifications = {split: target_qualification(values) for split, values in records.items()}
    confirmation: dict[str, Any] = {}
    for target, models in discovery_model["models"].items():
        if qualifications["confirmation"][target]["authorized"]:
            confirmation[target] = target_confirmation(target, models, records["confirmation"])
        else:
            confirmation[target] = {"tested": False, "reason": "target-specific confirmation qualification failed"}
    class_test = confirmation.get("classification", {})
    onset_test = confirmation.get("onset", {})
    class_pass = bool(class_test.get("passed", False))
    onset_pass = bool(onset_test.get("passed", False))
    manifest = []
    for split, rows in rows_by_split.items():
        for row in rows:
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
    if class_pass and onset_pass:
        status = "classification_and_onset_incremental_prediction_confirmed"
    elif class_pass:
        status = "classification_incremental_prediction_confirmed_onset_not_confirmed"
    else:
        status = "frozen_prerule_mechanism_increment_not_confirmed"
    result = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_model_digest": discovery_model["model_digest"],
        "discovery_model_created_at": discovery_model["created_at"],
        "earliest_confirmation_created_at": min(row["created_at"] for row in rows_by_split["confirmation"]),
        "qualifications": qualifications,
        "confirmation": confirmation,
        "run_manifest": manifest,
        "claims": {
            "classification_target_and_onset_target_separately_authorized": True,
            "right_censored_retained_in_classification": True,
            "confirmation_trained_after_discovery_model_freeze": discovery_model["created_at"] < min(row["created_at"] for row in rows_by_split["confirmation"]),
            "H15_P3_contract_scoped": "supported" if class_pass else "not_supported_under_frozen_six_family_summary_contract",
            "H16_P2_contract_scoped": "supported_candidate" if class_pass else "not_supported_under_frozen_cross_task_contract",
            "universal_formation_law": "not_claimed",
            "natural_language_mechanism": "not_tested",
            "pretrained_model_external_validity": "not_tested",
        },
        "authorized_next": {
            "automatic_execution": bool(class_pass),
            "experiment": "PHASE1220_PRERULE_PREDICTOR_CAUSAL_USE" if class_pass else None,
            "reason": "classification precursor increment passed all absolute, baseline, and matched-null gates" if class_pass else "frozen precursor summaries did not pass the one-shot classification increment gate",
            "pretrained_model_run": False,
        },
        "k_item": {
            "identifier": "K196",
            "evidence_grade": "E2-PREDICTIVE-CANDIDATE" if class_pass else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "A frozen six-family pre-rule mechanism summary predicted formed_by_800 beyond factor-plus-scalar baselines and matched within-cell nulls on unseen tasks, lexicons, depths, and seeds."
                if class_pass
                else "The frozen six-family pre-rule mechanism summary did not predict formed_by_800 beyond factor-plus-scalar baselines and matched within-cell nulls under the one-shot confirmation contract."
            ),
            "scope": "64 new fixed-width free micro-Transformers; synthetic finite rules; contract-scoped precursor prediction",
        },
        "new_mathematics_required": False,
    }
    result["final_digest"] = digest(result)
    write_json(FINAL_PATH, result)
    return result


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    condition = core.make_condition("discovery", 0, 0)
    config = ARCHITECTURES["discovery"]["d4_w112"]
    core.set_seed(1_219_999)
    model = TinyCausalTransformer(config).cuda()
    controls = core.initial_camera_controls(model, condition)
    scan = core.scan_checkpoint(model, condition, controls, 0)
    scan["mechanism_camera"] = mechanism_camera(model, condition, scan)
    result = {
        "cuda_device": torch.cuda.get_device_name(0),
        "observation_count": len(OBSERVATION_STEPS),
        "prefix_count": len(PREFIX_STEPS),
        "anchor_count": len(ANCHOR_STEPS),
        "routing_case_count": scan["mechanism_camera"]["routing"]["case_count"],
        "mechanism_camera_finite": all_finite(scan["mechanism_camera"]),
        "layer_count": len(scan["layers"]),
        "zero_drift": scan["necessity"]["zero_drift_max"],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_split(split: str) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if split == "confirmation" and not DISCOVERY_MODEL_PATH.exists():
        raise RuntimeError("freeze discovery model before confirmation")
    device = torch.device("cuda:0")
    rows = []
    for task_index, lexicon_index, architecture_index, replicate in itertools.product(range(2), range(2), range(2), range(REPLICATES)):
        rows.append(execute_run(split, task_index, lexicon_index, architecture_index, replicate, device))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "preregister", "run", "fit-discovery", "finalize"), required=True)
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
    elif args.stage == "fit-discovery":
        print(json.dumps(fit_discovery(), indent=2))
    else:
        print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
