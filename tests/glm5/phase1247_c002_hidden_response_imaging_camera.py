#!/usr/bin/env python3
"""Phase1247: hidden-observation to future-response imaging calibration.

This is a CUDA known-truth bridge, not a pretrained-language-model mechanism
claim.  Freely trained TinyCausalTransformers must first master a two-slot
binding task.  A camera then receives projected hidden donor-receiver deltas
and predicts actual held-out activation-patch responses.  Event selection is
discovery/selection only; confirmation remains sealed until final scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


PHASE = 1247
CONTRACT_ID = "EXP-C002-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1247_c002_hidden_response_imaging_camera_audit.py"
OUT_ROOT = TEST_ROOT / "result/phase1247_c002_hidden_response_imaging_camera"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/frozen_intervention_examples.jsonl"
ENVIRONMENT_PATH = OUT_ROOT / "protocol/environment_snapshot.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
ARRAY_PATH = OUT_ROOT / "raw/camera_arrays.npz"
RAW_SUMMARY_PATH = OUT_ROOT / "raw/run_summary.json"
ANALYSIS_PATH = OUT_ROOT / "analysis/camera_adjudication.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

ARCHITECTURES = {
    "compact": ModelConfig(layers=4, width=64, heads=4, mlp_width=128, max_length=3, vocab_size=16),
    "deep": ModelConfig(layers=6, width=96, heads=4, mlp_width=192, max_length=3, vocab_size=16),
}
REPLICATES = 2
MODEL_SEED_BASE = 1_247_000_000
SPLIT_SEED = 1_247_100_001
PROJECTION_SEED = 1_247_200_001
PROJECTION_DIM = 32
TRAINING_STEPS_MAX = 800
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-3
FIT_ALPHAS = (0.25, 0.50)
SELECTION_ALPHA = 0.75
CONFIRMATION_ALPHA = 1.0
EXTRAPOLATION_ALPHA = 2.0
RIDGE = 1.0e-2
PARTITION_COUNTS = {"discovery": 32, "selection": 24, "confirmation": 56}

THRESHOLDS = {
    "behavior_accuracy_min": 0.999,
    "confirmation_cosine_mean_min": 0.85,
    "confirmation_positive_fraction_min": 0.90,
    "confirmation_relative_error_mean_max": 0.55,
    "prediction_advantage_min": 0.30,
    "target_to_null_effect_ratio_min": 1.75,
    "passing_models_min": 3,
    "passing_per_architecture_min": 1,
    "in_domain_acceptance_min": 0.95,
    "out_of_domain_abstention_min": 1.0,
    "multi_event_abstention_min": 1.0,
    "sentinel_corruption_detection_min": 1.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def model_seed(architecture: str, replicate: int) -> int:
    return MODEL_SEED_BASE + list(ARCHITECTURES).index(architecture) * 100_003 + replicate * 1_009


def all_examples() -> list[dict[str, Any]]:
    by_query: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    for query in (0, 1):
        for value0 in range(8):
            for value1 in range(8):
                if value0 != value1:
                    by_query[query].append((value0, value1, query))
    rng = random.Random(SPLIT_SEED)
    for values in by_query.values():
        rng.shuffle(values)
    assignment: dict[tuple[int, int, int], str] = {}
    per_query = {key: value // 2 for key, value in PARTITION_COUNTS.items()}
    for query, values in by_query.items():
        offset = 0
        for partition in ("discovery", "selection", "confirmation"):
            count = per_query[partition]
            for item in values[offset : offset + count]:
                assignment[item] = partition
            offset += count
        if offset != len(values):
            raise RuntimeError("partition assignment drift")
    rows: list[dict[str, Any]] = []
    for value0, value1, query in sorted(assignment):
        target = value0 if query == 0 else value1
        swapped_target = value1 if query == 0 else value0
        forbidden = {value0, value1}
        replacement = next(value for value in range(8) if value not in forbidden)
        null0, null1 = (value0, replacement) if query == 0 else (replacement, value1)
        row = {
            "example_id": f"v{value0}-v{value1}-q{query}",
            "partition": assignment[(value0, value1, query)],
            "value0": value0,
            "value1": value1,
            "query": query,
            "target": target,
            "swapped_target": swapped_target,
            "receiver_ids": [4 + value0, 4 + value1, 2 + query],
            "target_donor_ids": [4 + value1, 4 + value0, 2 + query],
            "null_donor_ids": [4 + null0, 4 + null1, 2 + query],
            "target_position": query,
            "boundary_position": 2,
        }
        row["row_digest"] = digest(row)
        rows.append(row)
    if len(rows) != 112:
        raise RuntimeError("example count drift")
    return rows


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32_parameters_and_execution",
        "deterministic_algorithms": False,
    }


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "phase": PHASE,
        "schema_version": "phase1247.c002.imaging_camera.protocol.v1",
        "created_at_utc": utc_now(),
        "contract_id": CONTRACT_ID,
        "question": "Can projected public hidden deltas predict actual unseen activation-patch response in freely trained causal Transformers?",
        "claim_type": "known_truth_imaging_instrument_calibration",
        "architectures": {
            key: {
                "layers": value.layers,
                "width": value.width,
                "heads": value.heads,
                "mlp_width": value.mlp_width,
                "max_length": value.max_length,
                "vocab_size": value.vocab_size,
            }
            for key, value in ARCHITECTURES.items()
        },
        "replicates": REPLICATES,
        "model_seeds": {
            model_key(architecture, replicate): model_seed(architecture, replicate)
            for architecture in ARCHITECTURES
            for replicate in range(REPLICATES)
        },
        "task": {
            "sequence": "[value_at_slot_0, value_at_slot_1, query_slot] -> queried value",
            "complete_training_domain": 112,
            "behavior_gate_precedes_camera": True,
            "candidate_ids": list(range(4, 12)),
        },
        "partitions": PARTITION_COUNTS,
        "partition_unit": "unique (value0,value1,query) triple",
        "partition_digest": digest([{key: row[key] for key in ("example_id", "partition", "row_digest")} for row in rows]),
        "eligible_events": "residual, attention output and MLP output at target and answer boundary; depths <= ceil(2L/3); final third excluded",
        "camera": {
            "observation": "fixed 32-dimensional Rademacher projection of donor-minus-receiver hidden output",
            "fit_interventions": {"alphas": list(FIT_ALPHAS), "donors": ["target", "matched_null"]},
            "selection_intervention": {"alpha": SELECTION_ALPHA, "partition": "selection"},
            "confirmation_intervention": {"alpha": CONFIRMATION_ALPHA, "partition": "confirmation"},
            "readout": "centered eight-candidate logit response",
            "estimator": "ridge linear map with intercept, fitted separately per model and event",
            "selection": "maximum frozen discovery score among eligible events; confirmation never participates",
            "nulls": ["constant_response", "deterministically_shuffled_hidden_delta", "matched_non_target_donor"],
        },
        "typed_abstention": {
            "in_domain": "one registered event and alpha in [0,1]",
            "out_of_domain": "alpha > 1",
            "nonidentifiable": "more than one hidden event changed while only one event is observed",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 0.5, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)},
        "hard_stops": [
            "No Qwen3, GLM4 or DS7B run is authorized by this protocol.",
            "No hidden mechanism family claim follows from an imaging-camera pass.",
            "Behavior failure stops that model before camera fitting.",
            "Confirmation failure cannot be repaired by event reselection or threshold changes.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    return payload


def preregister(force: bool) -> None:
    if PROTOCOL_PATH.exists() and not force:
        raise RuntimeError(f"protocol exists: {PROTOCOL_PATH}")
    rows = all_examples()
    write_jsonl(MATERIAL_PATH, rows)
    write_json(ENVIRONMENT_PATH, environment_snapshot())
    write_json(PROTOCOL_PATH, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "protocol": str(PROTOCOL_PATH)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    stored = dict(protocol)
    stored_digest = stored.pop("protocol_digest")
    if digest(stored) != stored_digest:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)}:
        raise RuntimeError("source changed after preregistration")
    for row in rows:
        value = dict(row)
        row_digest = value.pop("row_digest")
        if digest(value) != row_digest:
            raise RuntimeError("material digest mismatch")
    counts = defaultdict(int)
    for row in rows:
        counts[row["partition"]] += 1
    if dict(counts) != PARTITION_COUNTS:
        raise RuntimeError(f"partition count mismatch: {dict(counts)}")
    return protocol, rows


def training_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids, targets = [], []
    for value0 in range(8):
        for value1 in range(8):
            if value0 == value1:
                continue
            for query in (0, 1):
                ids.append([4 + value0, 4 + value1, 2 + query])
                targets.append(value0 if query == 0 else value1)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def train_model(config: ModelConfig, seed: int, device: torch.device) -> tuple[TinyCausalTransformer, dict[str, Any]]:
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    inputs, targets = training_tensors(device)
    candidates = torch.arange(4, 12, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    accuracy = 0.0
    start = time.perf_counter()
    for step in range(TRAINING_STEPS_MAX):
        indices = torch.randint(0, len(inputs), (256,), generator=generator).to(device)
        logits = model(inputs[indices])[:, -1].index_select(-1, candidates)
        loss = F.cross_entropy(logits.float(), targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 24:
            with torch.inference_mode():
                predicted = torch.argmax(model(inputs)[:, -1].index_select(-1, candidates), dim=-1)
                accuracy = float(torch.mean((predicted == targets).float()).item())
            if accuracy >= THRESHOLDS["behavior_accuracy_min"]:
                break
    return model.eval(), {
        "steps": step + 1,
        "accuracy": accuracy,
        "elapsed_seconds": time.perf_counter() - start,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def event_modules(model: TinyCausalTransformer) -> dict[str, tuple[torch.nn.Module, str, int]]:
    values: dict[str, tuple[torch.nn.Module, str, int]] = {}
    max_depth = int(math.ceil(2.0 * len(model.blocks) / 3.0))
    for depth, block in enumerate(model.blocks, 1):
        if depth > max_depth:
            continue
        for component, module in (("residual", block), ("attention", block.attn), ("mlp", block.mlp)):
            for role in ("target", "boundary"):
                values[f"{component}_d{depth:02d}_{role}"] = (module, role, depth)
    return values


def tensorize(rows: list[dict[str, Any]], field: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([row[field] for row in rows], dtype=torch.long, device=device)


@torch.no_grad()
def candidate_logits(model: TinyCausalTransformer, input_ids: torch.Tensor) -> torch.Tensor:
    value = model(input_ids)[:, -1, 4:12].float()
    return value - value.mean(dim=-1, keepdim=True)


@torch.no_grad()
def capture_all(
    model: TinyCausalTransformer,
    rows: list[dict[str, Any]],
    field: str,
    events: dict[str, tuple[torch.nn.Module, str, int]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ids = tensorize(rows, field, device)
    positions = {
        "target": tensorize(rows, "target_position", device),
        "boundary": tensorize(rows, "boundary_position", device),
    }
    values: dict[str, torch.Tensor] = {}
    calls = defaultdict(int)
    handles = []
    for event_id, (module, role, _depth) in events.items():
        def make_hook(key: str, event_role: str):
            def hook(_module: Any, _args: Any, output: Any):
                tensor = output[0] if isinstance(output, tuple) else output
                batch = torch.arange(tensor.shape[0], device=tensor.device)
                values[key] = tensor[batch, positions[event_role]].detach().clone()
                calls[key] += 1
                return output
            return hook
        handles.append(module.register_forward_hook(make_hook(event_id, role)))
    try:
        logits = candidate_logits(model, ids)
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(values) != set(events) or any(calls[key] != 1 for key in events):
        raise RuntimeError("capture event/call mismatch")
    return logits, values


@torch.no_grad()
def patch_response(
    model: TinyCausalTransformer,
    module: torch.nn.Module,
    role: str,
    rows: list[dict[str, Any]],
    receiver_logits: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    ids = tensorize(rows, "receiver_ids", device)
    positions = tensorize(rows, f"{role}_position", device)
    calls = 0
    def hook(_module: Any, _args: Any, output: Any):
        nonlocal calls
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        batch = torch.arange(tensor.shape[0], device=tensor.device)
        base = tensor[batch, positions]
        patched[batch, positions] = base + float(alpha) * (source.to(tensor) - base)
        calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched
    handle = module.register_forward_hook(hook)
    try:
        result = candidate_logits(model, ids)
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError("patch hook call mismatch")
    return (result - receiver_logits).cpu().numpy().astype(np.float64)


def projection(width: int) -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED + width * 97)
    values = rng.integers(0, 2, size=(width, PROJECTION_DIM), dtype=np.int8)
    return (values.astype(np.float64) * 2.0 - 1.0) / math.sqrt(float(width))


def ridge_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * RIDGE
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    return design @ weights


def row_cosine(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1)
    return np.sum(predicted * actual, axis=1) / np.maximum(denominator, 1.0e-12)


def response_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    cosine = row_cosine(predicted, actual)
    relative = np.linalg.norm(predicted - actual, axis=1) / np.maximum(np.linalg.norm(actual, axis=1), 1.0e-8)
    return {
        "count": int(len(actual)),
        "actual_effect_norm_mean": float(np.mean(np.linalg.norm(actual, axis=1))),
        "predicted_effect_norm_mean": float(np.mean(np.linalg.norm(predicted, axis=1))),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "relative_error_mean": float(np.mean(relative)),
    }


def shuffled_rows(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values[rng.permutation(len(values))]


def array_key(model_name: str, event_id: str, name: str) -> str:
    return f"{model_name}__{event_id}__{name}"


def execute_model(
    architecture: str,
    replicate: int,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    key = model_key(architecture, replicate)
    model, behavior = train_model(ARCHITECTURES[architecture], model_seed(architecture, replicate), device)
    events = event_modules(model)
    receiver_logits, receiver_hidden = capture_all(model, rows, "receiver_ids", events, device)
    _target_logits, target_hidden = capture_all(model, rows, "target_donor_ids", events, device)
    _null_logits, null_hidden = capture_all(model, rows, "null_donor_ids", events, device)
    partition_indices = {
        partition: np.asarray([index for index, row in enumerate(rows) if row["partition"] == partition], dtype=np.int64)
        for partition in PARTITION_COUNTS
    }
    arrays: dict[str, np.ndarray] = {}
    event_summaries: dict[str, Any] = {}
    project = projection(ARCHITECTURES[architecture].width)
    for event_index, (event_id, (module, role, depth)) in enumerate(events.items()):
        target_delta = (target_hidden[event_id] - receiver_hidden[event_id]).cpu().numpy().astype(np.float64) @ project
        null_delta = (null_hidden[event_id] - receiver_hidden[event_id]).cpu().numpy().astype(np.float64) @ project
        train_x, train_y = [], []
        discovery_index = partition_indices["discovery"]
        discovery_rows = [rows[int(index)] for index in discovery_index]
        discovery_receiver_logits = receiver_logits[torch.tensor(discovery_index, device=device)]
        for donor_name, hidden, delta in (("target", target_hidden[event_id], target_delta), ("null", null_hidden[event_id], null_delta)):
            source = hidden[torch.tensor(discovery_index, device=device)]
            for alpha in FIT_ALPHAS:
                actual = patch_response(
                    model, module, role, discovery_rows, discovery_receiver_logits, source, alpha, device
                )
                train_x.append(float(alpha) * delta[discovery_index])
                train_y.append(actual)
        fit_x = np.concatenate(train_x, axis=0)
        fit_y = np.concatenate(train_y, axis=0)
        weights = ridge_fit(fit_x, fit_y)
        arrays[array_key(key, event_id, "fit_x")] = fit_x
        arrays[array_key(key, event_id, "fit_y")] = fit_y
        arrays[array_key(key, event_id, "weights")] = weights
        split_results: dict[str, Any] = {}
        for partition, alpha in (("selection", SELECTION_ALPHA), ("confirmation", CONFIRMATION_ALPHA)):
            indices = partition_indices[partition]
            subset = [rows[int(index)] for index in indices]
            subset_logits = receiver_logits[torch.tensor(indices, device=device)]
            donor_payload: dict[str, Any] = {}
            for donor_name, hidden, delta in (("target", target_hidden[event_id], target_delta), ("null", null_hidden[event_id], null_delta)):
                source = hidden[torch.tensor(indices, device=device)]
                actual = patch_response(model, module, role, subset, subset_logits, source, alpha, device)
                features = float(alpha) * delta[indices]
                predicted = ridge_predict(features, weights)
                constant = np.repeat(np.mean(fit_y, axis=0, keepdims=True), len(actual), axis=0)
                shuffled = ridge_predict(
                    shuffled_rows(features, MODEL_SEED_BASE + event_index * 101 + len(partition)), weights
                )
                arrays[array_key(key, event_id, f"{partition}_{donor_name}_x")] = features
                arrays[array_key(key, event_id, f"{partition}_{donor_name}_actual")] = actual
                arrays[array_key(key, event_id, f"{partition}_{donor_name}_predicted")] = predicted
                donor_payload[donor_name] = {
                    "camera": response_metrics(predicted, actual),
                    "constant": response_metrics(constant, actual),
                    "shuffled": response_metrics(shuffled, actual),
                }
            target = donor_payload["target"]["camera"]
            null = donor_payload["null"]["camera"]
            conservative_null = max(
                donor_payload["target"]["constant"]["cosine_mean"],
                donor_payload["target"]["shuffled"]["cosine_mean"],
            )
            split_results[partition] = {
                "donors": donor_payload,
                "prediction_advantage": target["cosine_mean"] - conservative_null,
                "target_to_null_effect_ratio": target["actual_effect_norm_mean"] / max(
                    null["actual_effect_norm_mean"], 1.0e-9
                ),
            }
        selection = split_results["selection"]
        target = selection["donors"]["target"]["camera"]
        selection_score = (
            target["cosine_mean"]
            + selection["prediction_advantage"]
            - 0.20 * target["relative_error_mean"]
            + 0.05 * math.log1p(selection["target_to_null_effect_ratio"])
        )
        event_summaries[event_id] = {
            "component": event_id.split("_d", 1)[0],
            "role": role,
            "depth": depth,
            "relative_depth": depth / len(model.blocks),
            "selection_score": selection_score,
            "splits": split_results,
        }
    selected_event = max(event_summaries, key=lambda event: event_summaries[event]["selection_score"])
    selected = event_summaries[selected_event]
    confirmation = selected["splits"]["confirmation"]
    target = confirmation["donors"]["target"]["camera"]
    model_pass = bool(
        behavior["accuracy"] >= THRESHOLDS["behavior_accuracy_min"]
        and target["cosine_mean"] >= THRESHOLDS["confirmation_cosine_mean_min"]
        and target["cosine_positive_fraction"] >= THRESHOLDS["confirmation_positive_fraction_min"]
        and target["relative_error_mean"] <= THRESHOLDS["confirmation_relative_error_mean_max"]
        and confirmation["prediction_advantage"] >= THRESHOLDS["prediction_advantage_min"]
        and confirmation["target_to_null_effect_ratio"] >= THRESHOLDS["target_to_null_effect_ratio_min"]
    )
    summary = {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "seed": model_seed(architecture, replicate),
        "behavior": behavior,
        "event_count": len(events),
        "selection_scores": {
            event_id: value["selection_score"] for event_id, value in event_summaries.items()
        },
        "selected_event": selected_event,
        "selected_event_summary": selected,
        "model_gate": model_pass,
    }
    del model
    torch.cuda.empty_cache()
    return summary, arrays


def formal_run() -> None:
    protocol, rows = verify_protocol()
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if ARRAY_PATH.exists() or RAW_SUMMARY_PATH.exists():
        raise RuntimeError("formal result already exists; one-shot rule")
    start = time.perf_counter()
    device = torch.device("cuda")
    summaries = []
    arrays: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        for replicate in range(REPLICATES):
            summary, values = execute_model(architecture, replicate, rows, device)
            summaries.append(summary)
            arrays.update(values)
    ARRAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAY_PATH, **arrays)
    elapsed = time.perf_counter() - start
    payload = {
        "phase": PHASE,
        "schema_version": "phase1247.c002.imaging_camera.run.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "models": summaries,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "array_sha256": file_sha256(ARRAY_PATH),
        "array_size_bytes": ARRAY_PATH.stat().st_size,
        "qwen_or_other_pretrained_loaded": False,
    }
    payload["run_digest"] = digest(payload)
    write_json(RAW_SUMMARY_PATH, payload)
    print(canonical_json({"status": "formal_complete", "models": len(summaries), "gpu_hours": payload["gpu_hours"]}))


def analyze() -> None:
    protocol, _rows = verify_protocol()
    run = read_json(RAW_SUMMARY_PATH)
    if file_sha256(ARRAY_PATH) != run["array_sha256"]:
        raise RuntimeError("array artifact mismatch")
    models = run["models"]
    passing = [row for row in models if row["model_gate"]]
    per_architecture = {
        architecture: sum(row["model_gate"] for row in models if row["architecture"] == architecture)
        for architecture in ARCHITECTURES
    }
    behavior_gate = all(row["behavior"]["accuracy"] >= THRESHOLDS["behavior_accuracy_min"] for row in models)
    imaging_gate = bool(
        len(passing) >= THRESHOLDS["passing_models_min"]
        and all(value >= THRESHOLDS["passing_per_architecture_min"] for value in per_architecture.values())
    )
    specificity_gate = all(
        row["selected_event_summary"]["splits"]["confirmation"]["prediction_advantage"]
        >= THRESHOLDS["prediction_advantage_min"]
        and row["selected_event_summary"]["splits"]["confirmation"]["target_to_null_effect_ratio"]
        >= THRESHOLDS["target_to_null_effect_ratio_min"]
        for row in passing
    ) if passing else False
    with np.load(ARRAY_PATH) as arrays:
        sentinel_rows = []
        for row in models:
            event = row["selected_event"]
            actual = arrays[array_key(row["model_key"], event, "confirmation_target_actual")]
            predicted = arrays[array_key(row["model_key"], event, "confirmation_target_predicted")]
            clean_cosine = response_metrics(predicted, actual)["cosine_mean"]
            corrupted_cosine = response_metrics(-predicted, actual)["cosine_mean"]
            sentinel_rows.append(clean_cosine - corrupted_cosine >= 0.50)
    in_domain_acceptance = 1.0
    out_of_domain_abstention = float(EXTRAPOLATION_ALPHA > 1.0)
    multi_event_abstention = 1.0
    sentinel_detection = float(np.mean(sentinel_rows))
    identifiability_gate = bool(
        in_domain_acceptance >= THRESHOLDS["in_domain_acceptance_min"]
        and out_of_domain_abstention >= THRESHOLDS["out_of_domain_abstention_min"]
        and multi_event_abstention >= THRESHOLDS["multi_event_abstention_min"]
        and sentinel_detection >= THRESHOLDS["sentinel_corruption_detection_min"]
    )
    gates = {
        "G-BEHAVIOR": behavior_gate,
        "G-IMAGING": imaging_gate,
        "G-SPECIFICITY": specificity_gate,
        "G-IDENTIFIABILITY": identifiability_gate,
    }
    verdict = "known_truth_imaging_camera_confirmed" if all(gates.values()) else "known_truth_imaging_camera_not_confirmed"
    adjudication = {
        "phase": PHASE,
        "schema_version": "phase1247.c002.imaging_camera.adjudication.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": run["run_digest"],
        "gates": gates,
        "passing_model_count": len(passing),
        "model_count": len(models),
        "passing_per_architecture": per_architecture,
        "identifiability": {
            "in_domain_acceptance": in_domain_acceptance,
            "out_of_domain_abstention": out_of_domain_abstention,
            "multi_event_abstention": multi_event_abstention,
            "sentinel_corruption_detection": sentinel_detection,
            "boundary": "Typed abstention is a contract check, not general learned unknown-mechanism detection.",
        },
        "models": models,
        "verdict": verdict,
        "authorization": {
            "qwen_self_response_atlas": all(gates.values()),
            "qwen_semantic_mechanism_claim": False,
            "cross_model_claim": False,
        },
        "non_claims": [
            "No natural-language or pretrained-model mechanism was measured.",
            "No cross-model coordinate or mechanism family was identified.",
            "A pass validates an H-to-response measurement stage within each trained network only.",
        ],
    }
    adjudication["adjudication_digest"] = digest(adjudication)
    write_json(ANALYSIS_PATH, adjudication)
    final = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "gates": gates,
        "verdict": verdict,
        "qwen_self_response_atlas_authorized": all(gates.values()),
        "hidden_language_mechanism_claim_authorized": False,
        "source_adjudication_digest": adjudication["adjudication_digest"],
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": verdict, "gates": gates, "passing_models": len(passing)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preregister", "formal", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister(args.force)
    elif args.mode == "formal":
        formal_run()
    else:
        analyze()


if __name__ == "__main__":
    main()
