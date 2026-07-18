#!/usr/bin/env python3
"""Trace Phase518 world/query platforms for independently authorized models.

The script freezes role-local contiguous platforms on the discovery split before
reading the prediction split.  It is observational: no component or neuron is
intervened on, and platform predictability is not called transport or causality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import get_model_info, load_model, release_model  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")
TASKS = ("world_topology", "query_evaluation")
PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase518_world_query_platform_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase518_frozen_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase518_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase518_world_query_platform_protocol.py"
DISCOVERY_PATH = PROTOCOL_DIR / "phase518_platform_discovery_relation.jsonl"
PREDICTION_PATH = PROTOCOL_DIR / "phase518_platform_prediction_relation.jsonl"
AUTHORIZATION_PATH = (
    ROOT
    / "tests/gpt5/result/phase522_semantic_event_confirmation"
    / "phase522_physical_authorization.json"
)
OUT_DIR = ROOT / "tests/gpt5/result/phase523_world_query_platform_physical"
Z = 1.96


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * ((p * (1 - p) + Z * Z / (4 * n)) / n) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def binary_rate(predictions: np.ndarray, truths: np.ndarray) -> dict[str, Any]:
    n = int(len(truths))
    count = int((predictions == truths).sum())
    lower, upper = wilson(count, n)
    return {
        "n": n,
        "count": count,
        "rate": count / n if n else 0.0,
        "lcb95": lower,
        "ucb95": upper,
    }


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    authorization = read_json(AUTHORIZATION_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase518 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase518 protocol changed after freeze")
    if sha256_file(DISCOVERY_PATH) != contract["split_files"]["platform_discovery_relation"]["sha256"]:
        raise RuntimeError("Phase518 discovery split hash drift")
    if sha256_file(PREDICTION_PATH) != contract["split_files"]["platform_prediction_relation"]["sha256"]:
        raise RuntimeError("Phase518 prediction split hash drift")
    if authorization["stage"] != "confirmation":
        raise RuntimeError("physical authorization is not independently confirmed")
    return contract, authorization


def flatten(samples: list[dict[str, Any]], split: str, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"invalid {split} sample")
        for variant in sample["variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_pair_id": sample["source_pair_id"],
                "split": split,
                "pair_index": sample["pair_index"],
                "truth_value": sample["truth_value"],
                "world_topology": sample["world_topology"],
                "query_pattern": sample["query_pattern"],
                "length_control": sample["length_control"],
                "fact_order": sample["fact_order"],
                "relation_verb": sample["relation_verb"],
                "surface": variant["surface"],
                "prompt": variant["natural_prompt"],
                "role_char_ends": {role: variant["role_char_ends"][role] for role in roles},
            })
    return rows


def prefix_position(tokenizer: Any, prompt: str, char_end: int) -> int:
    full = tokenizer.encode(prompt, add_special_tokens=True)
    prefix = tokenizer.encode(prompt[:char_end], add_special_tokens=True)
    common = 0
    for left, right in zip(full, prefix):
        if left != right:
            break
        common += 1
    if common < len(prefix) - 1:
        raise RuntimeError(f"could not map role boundary: common={common}, prefix={len(prefix)}")
    return max(0, common - 1)


def role_positions(tokenizer: Any, row: dict[str, Any], roles: tuple[str, ...]) -> dict[str, int]:
    return {
        role: prefix_position(tokenizer, row["prompt"], int(row["role_char_ends"][role]))
        for role in roles
    }


def projection_matrix(
    d_model: int,
    dimension: int,
    seeds: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    matrices = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(0, 2, (d_model, dimension), generator=generator, dtype=torch.int8)
        matrices.append(signs.float().mul_(2).sub_(1).div_(math.sqrt(dimension)))
    return torch.cat(matrices, dim=1).to(device)


def parse_first_event(text: str) -> bool | None:
    normalized = " ".join(text.lstrip().split())
    match = re.match(
        r"^(The statement is supported\.|The statement is contradicted\.)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).lower() == "the statement is supported."


def collect(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    model_name: str,
    rows: list[dict[str, Any]],
    roles: tuple[str, ...],
    seeds: tuple[int, ...],
    dimension: int,
    batch_size: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    info = get_model_info(model, model_name)
    projection = projection_matrix(info.d_model, dimension, seeds, device)
    tokenizer.padding_side = "left"
    projected_batches = []
    norm_batches = []
    metadata = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        local_positions = [role_positions(tokenizer, row, roles) for row in batch]
        encoded_cpu = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=640,
        )
        width = int(encoded_cpu["input_ids"].shape[1])
        lengths = encoded_cpu["attention_mask"].sum(dim=1).tolist()
        padded_positions = []
        for length, positions in zip(lengths, local_positions, strict=True):
            pad = width - int(length)
            padded_positions.append([pad + positions[role] for role in roles])
        encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("model returned no hidden states")
        batch_indices = torch.arange(len(batch), device=device).unsqueeze(1)
        position_tensor = torch.tensor(padded_positions, device=device, dtype=torch.long)
        layer_projects = []
        layer_norms = []
        for hidden in hidden_states:
            selected = hidden[batch_indices, position_tensor].float()
            compressed = selected @ projection
            compressed = compressed.view(len(batch), len(roles), len(seeds), dimension)
            layer_projects.append(compressed.cpu())
            layer_norms.append(torch.linalg.vector_norm(selected, dim=-1).cpu())
        projected = torch.stack(layer_projects, dim=1).permute(0, 3, 1, 2, 4)
        norms = torch.stack(layer_norms, dim=1)
        projected_batches.append(projected.to(torch.float16).numpy())
        norm_batches.append(norms.to(torch.float32).numpy())
        del outputs, hidden_states, projected, norms

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for index, (row, positions, length) in enumerate(zip(batch, local_positions, lengths, strict=True)):
            text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
            event = parse_first_event(text)
            metadata.append({
                **{key: value for key, value in row.items() if key not in {"prompt", "role_char_ends"}},
                "token_length": int(length),
                "role_token_positions": positions,
                "generated_natural_text": text,
                "first_event_value": event,
                "first_event_recoverable": event is not None,
                "first_event_correct": event is not None and event == row["truth_value"],
            })
        del generated, encoded
        if start == 0 or start + len(batch) == len(rows) or (start // batch_size) % 8 == 7:
            log(f"{model_name} {stage} {min(start + len(batch), len(rows))}/{len(rows)}")
    projected_array = np.concatenate(projected_batches, axis=0)
    norm_array = np.concatenate(norm_batches, axis=0)
    return projected_array, norm_array, metadata, {
        "n_layers_with_embedding": int(projected_array.shape[2]),
        "d_model": int(info.d_model),
        "projection_dimension": dimension,
        "projection_seeds": list(seeds),
        "position_roles": list(roles),
    }


def unit_vectors(values: np.ndarray) -> np.ndarray:
    vectors = values.astype(np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def task_labels(metadata: list[dict[str, Any]], task: str) -> np.ndarray:
    if task == "world_topology":
        return np.asarray([row["world_topology"] == "A" for row in metadata], dtype=bool)
    if task == "query_evaluation":
        return np.asarray([row["truth_value"] for row in metadata], dtype=bool)
    raise ValueError(task)


def fit_observers(
    vectors: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = vectors[mask]
    local_labels = labels[mask]
    mean_positive = train[local_labels].mean(axis=0)
    mean_negative = train[~local_labels].mean(axis=0)
    directions = mean_positive - mean_negative
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    thresholds = np.einsum("plrd,plrd->plr", (mean_positive + mean_negative) / 2, directions)
    return directions.astype(np.float32), thresholds.astype(np.float32)


def predict_grid(
    vectors: np.ndarray,
    directions: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    scores = np.einsum("nplrd,plrd->nplr", vectors, directions) - thresholds[None, :, :, :]
    return scores > 0


def pair_rate(
    predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row["source_pair_id"]].append(index)
    records = []
    for pair_id, indices in groups.items():
        if len(indices) != 4:
            raise RuntimeError(f"incomplete four-way pair {pair_id}: {len(indices)}")
        records.append(bool(np.all(predictions[indices] == truths[indices])))
    values = np.asarray(records, dtype=bool)
    return binary_rate(values, np.ones_like(values, dtype=bool))


def metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    by_surface = {}
    for surface in SURFACES:
        mask = np.asarray([row["surface"] == surface for row in metadata], dtype=bool)
        by_surface[surface] = binary_rate(predictions[mask], truths[mask])
    return {
        "overall": binary_rate(predictions, truths),
        "by_surface": by_surface,
        "four_way_pair": pair_rate(predictions, truths, metadata),
    }


def subgroup_metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    fields = ("surface", "query_pattern", "truth_value") if task == "world_topology" else (
        "surface", "world_topology", "query_pattern"
    )
    report = {}
    values_by_field = {field: sorted({str(row[field]).lower() for row in metadata}) for field in fields}
    for surface in values_by_field[fields[0]]:
        for second in values_by_field[fields[1]]:
            for third in values_by_field[fields[2]]:
                mask = np.asarray([
                    str(row[fields[0]]).lower() == surface
                    and str(row[fields[1]]).lower() == second
                    and str(row[fields[2]]).lower() == third
                    for row in metadata
                ])
                key = f"{fields[0]}={surface}|{fields[1]}={second}|{fields[2]}={third}"
                report[key] = binary_rate(predictions[mask], truths[mask])
    return report


def gate_pass(report: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        report["overall"]["lcb95"] >= gate["overall_lcb95_min"]
        and all(item["lcb95"] >= gate["surface_lcb95_min"] for item in report["by_surface"].values())
        and report["four_way_pair"]["lcb95"] >= gate["four_way_lcb95_min"]
    )


def cross_validated_grid(
    vectors: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    folds: int,
) -> tuple[np.ndarray, list[list[list[dict[str, Any]]]]]:
    row_folds = np.asarray([int(row["pair_index"]) % folds for row in metadata], dtype=int)
    predictions = np.zeros(vectors.shape[:-1], dtype=bool)
    reports: list[list[list[dict[str, Any]]]] = [
        [[{} for _role in range(vectors.shape[3])] for _layer in range(vectors.shape[2])]
        for _projection in range(vectors.shape[1])
    ]
    for fold in range(folds):
        train_mask = row_folds != fold
        test_mask = row_folds == fold
        directions, thresholds = fit_observers(vectors, labels, train_mask)
        predictions[test_mask] = predict_grid(vectors[test_mask], directions, thresholds)
        local_truths = labels[test_mask]
        local_meta = [row for row, keep in zip(metadata, test_mask, strict=True) if keep]
        for projection in range(vectors.shape[1]):
            for layer in range(vectors.shape[2]):
                for role in range(vectors.shape[3]):
                    report = metrics(
                        predictions[test_mask, projection, layer, role],
                        local_truths,
                        local_meta,
                    )
                    reports[projection][layer][role].setdefault("folds", []).append(report)
    return predictions, reports


def discover_platforms(
    oof_predictions: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    reports: list[list[list[dict[str, Any]]]],
    roles: tuple[str, ...],
    design: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    local_gate = design["discovery_local_gate"]
    pass_mask = np.zeros(oof_predictions.shape[1:], dtype=bool)
    cell_ledger = []
    for projection in range(oof_predictions.shape[1]):
        for layer in range(oof_predictions.shape[2]):
            for role_index, role in enumerate(roles):
                fold_reports = reports[projection][layer][role_index]["folds"]
                fold_passes = [gate_pass(report, local_gate) for report in fold_reports]
                passed = sum(fold_passes) >= design["minimum_fold_passes"]
                pass_mask[projection, layer, role_index] = passed
                cell_ledger.append({
                    "projection_index": projection,
                    "layer_with_embedding": layer,
                    "position_role": role,
                    "fold_pass_count": sum(fold_passes),
                    "fold_gate_passes": fold_passes,
                    "qualifies_projection": passed,
                })
    consensus = pass_mask.sum(axis=0) >= design["projection_consensus_required"]
    platforms = []
    minimum_length = design["role_local_minimum_contiguous_layers"]
    for role_index, role in enumerate(roles):
        layers = np.flatnonzero(consensus[:, role_index]).tolist()
        runs = []
        for layer in layers:
            if not runs or layer != runs[-1][-1] + 1:
                runs.append([layer])
            else:
                runs[-1].append(layer)
        for run in runs:
            if len(run) < minimum_length:
                continue
            votes = oof_predictions[:, :, run, role_index]
            aggregate = votes.mean(axis=(1, 2)) > 0.5
            aggregate_report = metrics(aggregate, labels, metadata)
            per_projection = {
                str(projection): metrics(
                    votes[:, projection, :].mean(axis=1) > 0.5,
                    labels,
                    metadata,
                )
                for projection in range(oof_predictions.shape[1])
            }
            platforms.append({
                "platform_id": f"{role}:L{run[0]}-L{run[-1]}",
                "position_role": role,
                "role_index": role_index,
                "layers_with_embedding": run,
                "relative_depth_start": run[0] / max(1, oof_predictions.shape[2] - 1),
                "relative_depth_end": run[-1] / max(1, oof_predictions.shape[2] - 1),
                "discovery_oof": aggregate_report,
                "discovery_oof_by_projection": per_projection,
            })
    role_order = {role: index for index, role in enumerate(roles)}
    platforms.sort(
        key=lambda item: (
            -min(value["lcb95"] for value in item["discovery_oof"]["by_surface"].values()),
            -item["discovery_oof"]["four_way_pair"]["lcb95"],
            -item["discovery_oof"]["overall"]["lcb95"],
            -len(item["layers_with_embedding"]),
            item["layers_with_embedding"][0],
            role_order[item["position_role"]],
        )
    )
    for index, platform in enumerate(platforms):
        platform["selection_rank"] = index + 1
        platform["primary"] = index == 0
    return platforms, np.asarray(cell_ledger, dtype=object)


def behavior_report(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = np.asarray([row["first_event_value"] is True for row in metadata], dtype=bool)
    recoverable = np.asarray([row["first_event_recoverable"] for row in metadata], dtype=bool)
    truths = np.asarray([row["truth_value"] for row in metadata], dtype=bool)
    report = metrics(predictions, truths, metadata)
    report["recoverable"] = binary_rate(recoverable, np.ones_like(recoverable, dtype=bool))
    unrecoverable_count = int((~recoverable).sum())
    lower, upper = wilson(unrecoverable_count, len(recoverable))
    report["unrecoverable"] = {
        "n": int(len(recoverable)),
        "count": unrecoverable_count,
        "rate": unrecoverable_count / len(recoverable) if len(recoverable) else 0.0,
        "lcb95": lower,
        "ucb95": upper,
    }
    return report


def behavior_gate_pass(report: dict[str, Any], gate: dict[str, float]) -> bool:
    accuracy_gate = {
        "surface_lcb95_min": gate["surface_lcb95_min"],
        "overall_lcb95_min": gate["surface_lcb95_min"],
        "four_way_lcb95_min": gate["four_way_lcb95_min"],
    }
    return (
        gate_pass(report, accuracy_gate)
        and report["unrecoverable"]["ucb95"] <= gate["unrecoverable_ucb95_max"]
    )


def prediction_platform_report(
    grid: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    platform: dict[str, Any],
    gate: dict[str, float],
) -> dict[str, Any]:
    role_index = int(platform["role_index"])
    layers = list(platform["layers_with_embedding"])
    votes = grid[:, :, layers, role_index]
    aggregate_predictions = votes.mean(axis=(1, 2)) > 0.5
    aggregate = metrics(aggregate_predictions, labels, metadata)
    per_projection = {}
    intervals = []
    for projection in range(grid.shape[1]):
        local_predictions = votes[:, projection, :].mean(axis=1) > 0.5
        local_report = metrics(local_predictions, labels, metadata)
        local_report["gate_pass"] = gate_pass(local_report, gate)
        per_projection[str(projection)] = local_report
        intervals.append((local_report["overall"]["lcb95"], local_report["overall"]["ucb95"]))
    overlap_lower = max(lower for lower, _upper in intervals)
    overlap_upper = min(upper for _lower, upper in intervals)
    aggregate_pass = gate_pass(aggregate, gate)
    return {
        "aggregate": aggregate,
        "by_projection": per_projection,
        "projection_interval_overlap": {
            "lower": overlap_lower,
            "upper": overlap_upper,
            "nonempty": overlap_lower <= overlap_upper,
        },
        "matched_subgroups": subgroup_metrics(aggregate_predictions, labels, metadata, platform["task"]),
        "aggregate_gate_pass": aggregate_pass,
        "all_projection_gates_pass": all(item["gate_pass"] for item in per_projection.values()),
        "prediction_gate_pass": aggregate_pass and all(item["gate_pass"] for item in per_projection.values()),
    }


def save_arrays(
    model_name: str,
    stage: str,
    projected: np.ndarray,
    norms: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[Path, Path]:
    array_path = OUT_DIR / f"phase523_{model_name}_{stage}_projection.npz"
    np.savez_compressed(array_path, projected=projected, norms=norms)
    metadata_path = OUT_DIR / f"phase523_{model_name}_{stage}_metadata.jsonl"
    write_jsonl(metadata_path, metadata)
    return array_path, metadata_path


def run_model(
    model_name: str,
    contract: dict[str, Any],
    authorization: dict[str, Any],
    batch_size: int,
    use_8bit: bool,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"phase523_{model_name}_world_query_platform_summary.json"
    if model_name not in authorization["relation_models"]:
        summary = {
            "schema_version": "phase523_world_query_platform_summary.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_authorized",
            "model": model_name,
            "cuda_used": False,
            "model_weights_loaded": False,
            "prediction_split_read": False,
            "sealed_split_read": False,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
        return summary_path

    design = contract["physical_design"]
    roles = tuple(design["position_roles"])
    seeds = tuple(int(seed) for seed in design["projection_seeds"])
    model = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase523 requires CUDA")
        model, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        discovery_rows = flatten(read_jsonl(DISCOVERY_PATH), "platform_discovery", roles)
        discovery_projected, discovery_norms, discovery_meta, model_info = collect(
            model,
            tokenizer,
            device,
            model_name,
            discovery_rows,
            roles,
            seeds,
            int(design["projection_dimension"]),
            batch_size,
            "discovery",
        )
        discovery_vectors = unit_vectors(discovery_projected)
        discovery_behavior = behavior_report(discovery_meta)
        behavior_gate = behavior_gate_pass(discovery_behavior, contract["gates"]["natural_relation"])

        task_ledgers = {}
        observer_payload = {}
        for task in TASKS:
            labels = task_labels(discovery_meta, task)
            oof_grid, fold_reports = cross_validated_grid(
                discovery_vectors,
                labels,
                discovery_meta,
                int(design["group_folds"]),
            )
            platforms, cell_ledger = discover_platforms(
                oof_grid,
                labels,
                discovery_meta,
                fold_reports,
                roles,
                design,
            )
            for platform in platforms:
                platform["task"] = task
            full_mask = np.ones(len(labels), dtype=bool)
            directions, thresholds = fit_observers(discovery_vectors, labels, full_mask)
            observer_payload[f"{task}_directions"] = directions
            observer_payload[f"{task}_thresholds"] = thresholds
            task_ledgers[task] = {
                "platform_count": len(platforms),
                "platforms": platforms,
                "cell_ledger": cell_ledger.tolist(),
                "primary_platform_id": platforms[0]["platform_id"] if platforms else None,
            }

        observer_path = OUT_DIR / f"phase523_{model_name}_frozen_platform_observers.npz"
        np.savez_compressed(observer_path, **observer_payload)
        discovery_array_path, discovery_meta_path = save_arrays(
            model_name, "discovery", discovery_projected, discovery_norms, discovery_meta
        )
        ledger = {
            "schema_version": "phase523_frozen_platform_ledger.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "frozen_before_prediction_read",
            "model": model_name,
            "model_info": model_info,
            "discovery_row_count": len(discovery_rows),
            "discovery_behavior": discovery_behavior,
            "discovery_behavior_gate_pass": behavior_gate,
            "tasks": task_ledgers,
            "observer_path": str(observer_path.relative_to(ROOT)),
            "observer_sha256": sha256_file(observer_path),
            "discovery_projection_path": str(discovery_array_path.relative_to(ROOT)),
            "discovery_projection_sha256": sha256_file(discovery_array_path),
            "discovery_metadata_path": str(discovery_meta_path.relative_to(ROOT)),
            "discovery_metadata_sha256": sha256_file(discovery_meta_path),
            "prediction_split_read": False,
            "sealed_split_read": False,
            "causal": False,
        }
        ledger_path = OUT_DIR / f"phase523_{model_name}_frozen_platform_ledger.json"
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Read the prediction split only after the observer/platform ledger is durable.
        prediction_rows = flatten(read_jsonl(PREDICTION_PATH), "platform_prediction", roles)
        prediction_projected, prediction_norms, prediction_meta, prediction_model_info = collect(
            model,
            tokenizer,
            device,
            model_name,
            prediction_rows,
            roles,
            seeds,
            int(design["projection_dimension"]),
            batch_size,
            "prediction",
        )
        if prediction_model_info != model_info:
            raise RuntimeError("model info changed between discovery and prediction")
        prediction_vectors = unit_vectors(prediction_projected)
        prediction_behavior = behavior_report(prediction_meta)
        prediction_reports = {}
        for task in TASKS:
            directions = observer_payload[f"{task}_directions"]
            thresholds = observer_payload[f"{task}_thresholds"]
            grid = predict_grid(prediction_vectors, directions, thresholds)
            labels = task_labels(prediction_meta, task)
            platform_reports = []
            for platform in task_ledgers[task]["platforms"]:
                platform_reports.append({
                    "platform_id": platform["platform_id"],
                    "selection_rank": platform["selection_rank"],
                    "primary": platform["primary"],
                    "position_role": platform["position_role"],
                    "layers_with_embedding": platform["layers_with_embedding"],
                    "relative_depth_start": platform["relative_depth_start"],
                    "relative_depth_end": platform["relative_depth_end"],
                    "prediction": prediction_platform_report(
                        grid,
                        labels,
                        prediction_meta,
                        platform,
                        design["prediction_gate"],
                    ),
                })
            prediction_reports[task] = {
                "platform_count": len(platform_reports),
                "platforms": platform_reports,
                "primary_prediction_gate_pass": bool(
                    platform_reports and platform_reports[0]["prediction"]["prediction_gate_pass"]
                ),
            }
        prediction_array_path, prediction_meta_path = save_arrays(
            model_name, "prediction", prediction_projected, prediction_norms, prediction_meta
        )
        summary = {
            "schema_version": "phase523_world_query_platform_summary.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "model": model_name,
            "runtime_seconds": time.monotonic() - started,
            "cuda_used": True,
            "model_weights_loaded": True,
            "model_info": model_info,
            "discovery_row_count": len(discovery_rows),
            "prediction_row_count": len(prediction_rows),
            "discovery_behavior": discovery_behavior,
            "discovery_behavior_gate_pass": behavior_gate,
            "prediction_behavior": prediction_behavior,
            "task_predictions": prediction_reports,
            "frozen_ledger_path": str(ledger_path.relative_to(ROOT)),
            "frozen_ledger_sha256": sha256_file(ledger_path),
            "prediction_projection_path": str(prediction_array_path.relative_to(ROOT)),
            "prediction_projection_sha256": sha256_file(prediction_array_path),
            "prediction_metadata_path": str(prediction_meta_path.relative_to(ROOT)),
            "prediction_metadata_sha256": sha256_file(prediction_meta_path),
            "prediction_split_read": True,
            "sealed_split_read": False,
            "evidence_boundary": {
                "observational_platform": True,
                "compute_transport_measured": False,
                "causal_intervention": False,
                "component_head_channel_neuron_scan": False,
                "cross_model_shared_coordinates_claimed": False,
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(ledger_path)
        print(summary_path)
        return summary_path
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    contract, authorization = verify()
    run_model(args.model, contract, authorization, args.batch_size, args.use_8bit)


if __name__ == "__main__":
    main()
