#!/usr/bin/env python3
"""Discover fresh GLM4 role-normalized world geometry after behavior gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase530_glm4_fresh_world_geometry_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase530_frozen_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase530_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase530_glm4_fresh_world_geometry_protocol.py"
AUTH_PATH = ROOT / "tests/gpt5/result/phase531_glm4_fresh_world_geometry_behavior/phase531_fresh_physical_authorization.json"
OUT_DIR = ROOT / "tests/gpt5/result/phase532_glm4_role_normalized_world_geometry"
Z = 1.96


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wilson(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * ((p * (1 - p) + Z * Z / (4 * n)) / n) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=bool)
    n = int(values.size)
    count = int(values.sum())
    lower, upper = wilson(count, n)
    return {"n": n, "count": count, "rate": count / n if n else 0.0, "lcb95": lower, "ucb95": upper}


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    authorization = read_json(AUTH_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase530 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase530 protocol source drift")
    if sha256_file(CONTRACT_PATH) != authorization["contract_sha256"]:
        raise RuntimeError("Phase531 authorization contract drift")
    for split in ("discovery", "entity_prediction", "relation_prediction"):
        spec = contract["split_files"][split]
        if spec["sealed"] or sha256_file(ROOT / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"Phase530 split drift: {split}")
    return contract, authorization


def prefix_position(tokenizer: Any, prompt: str, char_end: int) -> int:
    full = tokenizer.encode(prompt, add_special_tokens=True)
    prefix = tokenizer.encode(prompt[:char_end], add_special_tokens=True)
    common = 0
    for left, right in zip(full, prefix):
        if left != right:
            break
        common += 1
    if common < len(prefix) - 1:
        raise RuntimeError(f"role boundary tokenization mismatch: {common}/{len(prefix)}")
    return max(0, common - 1)


def projection_matrix(d_model: int, dimension: int, seeds: tuple[int, ...], device: torch.device) -> torch.Tensor:
    matrices = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(0, 2, (d_model, dimension), generator=generator, dtype=torch.int8)
        matrices.append(signs.float().mul_(2).sub_(1).div_(math.sqrt(dimension)))
    return torch.cat(matrices, dim=1).to(device)


def collect(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    roles: tuple[str, ...],
    seeds: tuple[int, ...],
    dimension: int,
    batch_size: int,
    stage: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    info = get_model_info(model, "glm4")
    projection = projection_matrix(info.d_model, dimension, seeds, device)
    tokenizer.padding_side = "left"
    projected_batches = []
    metadata = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        local_positions = [
            {role: prefix_position(tokenizer, row["natural_prompt"], int(row["role_char_ends"][role])) for role in roles}
            for row in batch
        ]
        encoded_cpu = tokenizer(
            [row["natural_prompt"] for row in batch],
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
        if outputs.hidden_states is None:
            raise RuntimeError("model returned no hidden states")
        batch_indices = torch.arange(len(batch), device=device).unsqueeze(1)
        position_tensor = torch.tensor(padded_positions, device=device, dtype=torch.long)
        layers = []
        for hidden in outputs.hidden_states:
            selected = hidden[batch_indices, position_tensor].float()
            compressed = (selected @ projection).view(len(batch), len(roles), len(seeds), dimension)
            layers.append(compressed.cpu())
        projected = torch.stack(layers, dim=1).permute(0, 3, 1, 2, 4)
        projected_batches.append(projected.to(torch.float16).numpy())
        for row, positions, length in zip(batch, local_positions, lengths, strict=True):
            metadata.append({
                **{key: value for key, value in row.items() if key not in {"natural_prompt", "world_prefix", "role_char_ends"}},
                "token_length": int(length),
                "role_token_positions": positions,
            })
        del outputs, projected, encoded
        if start == 0 or start + len(batch) == len(rows) or (start // batch_size) % 8 == 7:
            log(f"glm4 {stage} {min(start + len(batch), len(rows))}/{len(rows)}")
    array = np.concatenate(projected_batches, axis=0)
    return array, metadata, {
        "n_layers_with_embedding": int(array.shape[2]),
        "d_model": int(info.d_model),
        "projection_dimension": dimension,
        "projection_seeds": list(seeds),
        "position_roles": list(roles),
    }


def canonical_endpoints(projected: np.ndarray, metadata: list[dict[str, Any]]) -> np.ndarray:
    endpoints = np.empty((len(metadata), projected.shape[1], projected.shape[2], 4, projected.shape[-1]), dtype=np.float32)
    for row_index, row in enumerate(metadata):
        register_order = [int(value) for value in row["register_order"]]
        for entity in range(4):
            slot = register_order.index(entity)
            endpoints[row_index, :, :, entity, :] = projected[row_index, :, :, slot, :].astype(np.float32)
    endpoints /= np.maximum(np.linalg.norm(endpoints, axis=-1, keepdims=True), 1e-8)
    return endpoints


def pair_feature(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    feature = np.concatenate((target - source, source * target), axis=-1)
    return feature / np.maximum(np.linalg.norm(feature, axis=-1, keepdims=True), 1e-8)


def build_examples(endpoints: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    labels = []
    kinds = []
    world_indices = []
    pair_keys = []
    source_slots = []
    target_slots = []
    for world_index, row in enumerate(metadata):
        register_order = [int(value) for value in row["register_order"]]
        entity_slot = {entity: register_order.index(entity) for entity in range(4)}
        edges = [tuple(int(item) for item in edge) for edge in row["edges"]]
        for edge_index, (source, target) in enumerate(edges):
            for kind, left, right, label in (
                ("positive", source, target, True),
                ("reverse", target, source, False),
            ):
                features.append(pair_feature(endpoints[world_index, :, :, left], endpoints[world_index, :, :, right]))
                labels.append(label)
                kinds.append(kind)
                world_indices.append(world_index)
                pair_keys.append(f"{world_index}:{edge_index}")
                source_slots.append(entity_slot[left])
                target_slots.append(entity_slot[right])
        for disconnected_index, (source, target) in enumerate(
            ((0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (1, 3), (3, 1))
        ):
            features.append(pair_feature(endpoints[world_index, :, :, source], endpoints[world_index, :, :, target]))
            labels.append(False)
            kinds.append("disconnected")
            world_indices.append(world_index)
            pair_keys.append(f"{world_index}:d{disconnected_index}")
            source_slots.append(entity_slot[source])
            target_slots.append(entity_slot[target])
    return {
        "features": np.stack(features, axis=0).transpose(0, 2, 1, 3),  # [example, layer, projection, feature]
        "labels": np.asarray(labels, dtype=bool),
        "kinds": np.asarray(kinds),
        "world_indices": np.asarray(world_indices, dtype=np.int32),
        "pair_keys": np.asarray(pair_keys),
        "source_slots": np.asarray(source_slots, dtype=np.int8),
        "target_slots": np.asarray(target_slots, dtype=np.int8),
    }


def group_folds(metadata: list[dict[str, Any]], count: int) -> np.ndarray:
    folds = []
    for row in metadata:
        digest = hashlib.sha256(row["source_pair_id"].encode("utf-8")).hexdigest()
        folds.append(int(digest[:8], 16) % count)
    return np.asarray(folds, dtype=np.int8)


def fit_direction(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    positive = features[labels].mean(axis=0)
    negative = features[~labels].mean(axis=0)
    direction = positive - negative
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    threshold = float(((positive + negative) / 2) @ direction)
    return direction.astype(np.float32), threshold


def position_baseline(examples: dict[str, Any], metadata: list[dict[str, Any]], fold_count: int) -> float:
    kinds = examples["kinds"]
    orientation = kinds != "disconnected"
    labels = examples["labels"][orientation]
    world_indices = examples["world_indices"][orientation]
    row_folds = group_folds(metadata, fold_count)
    features = np.zeros((int(orientation.sum()), 8), dtype=np.float32)
    source = examples["source_slots"][orientation]
    target = examples["target_slots"][orientation]
    features[np.arange(len(features)), source] = 1.0
    features[np.arange(len(features)), 4 + target] = 1.0
    predictions = np.zeros(len(features), dtype=bool)
    for fold in range(fold_count):
        test = row_folds[world_indices] == fold
        direction, threshold = fit_direction(features[~test], labels[~test])
        predictions[test] = features[test] @ direction > threshold
    return float((predictions == labels).mean())


def score_metrics(
    scores: np.ndarray,
    examples: dict[str, Any],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    kinds = examples["kinds"]
    labels = examples["labels"]
    orientation = kinds != "disconnected"
    orientation_correct = (scores[orientation] > 0) == labels[orientation]
    by_surface = {}
    orientation_worlds = examples["world_indices"][orientation]
    for surface in sorted({row["surface"] for row in metadata}):
        mask = np.asarray([metadata[index]["surface"] == surface for index in orientation_worlds], dtype=bool)
        by_surface[surface] = rate(orientation_correct[mask])
    pair_groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(examples["pair_keys"]):
        if kinds[index] != "disconnected":
            pair_groups[str(key)].append(index)
    direction_values = []
    for indices in pair_groups.values():
        positive = [index for index in indices if labels[index]]
        negative = [index for index in indices if not labels[index]]
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError("invalid orientation pair")
        direction_values.append(scores[positive[0]] > scores[negative[0]])
    disconnected = kinds == "disconnected"
    exact_world_values = []
    for world_index in range(len(metadata)):
        local = examples["world_indices"] == world_index
        exact_world_values.append(bool(np.all(((scores[local] > 0) == labels[local]))))
    return {
        "orientation": rate(orientation_correct),
        "by_surface": by_surface,
        "direction_pair": rate(np.asarray(direction_values, dtype=bool)),
        "disconnected_specificity": rate(scores[disconnected] <= 0),
        "exact_world": rate(np.asarray(exact_world_values, dtype=bool)),
    }


def gate_pass(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        metrics["orientation"]["lcb95"] >= gate["orientation_lcb95_min"]
        and metrics["direction_pair"]["lcb95"] >= gate["direction_pair_lcb95_min"]
        and metrics["disconnected_specificity"]["lcb95"] >= gate["disconnected_specificity_lcb95_min"]
        and metrics["exact_world"]["lcb95"] >= gate["exact_world_lcb95_min"]
        and all(item["lcb95"] >= gate["surface_lcb95_min"] for item in metrics["by_surface"].values())
    )


def discover(
    examples: dict[str, Any],
    metadata: list[dict[str, Any]],
    design: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    features = examples["features"]
    labels = examples["labels"]
    kinds = examples["kinds"]
    orientation = kinds != "disconnected"
    world_indices = examples["world_indices"]
    row_folds = group_folds(metadata, int(design["group_folds"]))
    n_layers = features.shape[1]
    n_projections = features.shape[2]
    oof_scores = np.zeros((len(labels), n_layers, n_projections), dtype=np.float32)
    full_directions = np.zeros((n_layers, n_projections, features.shape[-1]), dtype=np.float32)
    full_thresholds = np.zeros((n_layers, n_projections), dtype=np.float32)
    fold_rates = np.zeros((n_layers, n_projections, int(design["group_folds"])), dtype=np.float32)
    for layer in range(n_layers):
        for projection in range(n_projections):
            local = features[:, layer, projection]
            direction, threshold = fit_direction(local[orientation], labels[orientation])
            full_directions[layer, projection] = direction
            full_thresholds[layer, projection] = threshold
            for fold in range(int(design["group_folds"])):
                test_world = row_folds == fold
                train = orientation & ~test_world[world_indices]
                test = test_world[world_indices]
                fold_direction, fold_threshold = fit_direction(local[train], labels[train])
                oof_scores[test, layer, projection] = local[test] @ fold_direction - fold_threshold
                test_orientation = test & orientation
                fold_rates[layer, projection, fold] = float(
                    np.mean((oof_scores[test_orientation, layer, projection] > 0) == labels[test_orientation])
                )

    position_rate = position_baseline(examples, metadata, int(design["group_folds"]))
    embedding_rates = np.zeros(n_projections, dtype=np.float32)
    cell_pass = np.zeros((n_layers, n_projections), dtype=bool)
    cell_reports: list[list[dict[str, Any]]] = []
    for layer in range(n_layers):
        row_reports = []
        for projection in range(n_projections):
            metrics = score_metrics(oof_scores[:, layer, projection], examples, metadata)
            if layer == 0:
                embedding_rates[projection] = metrics["orientation"]["rate"]
            fold_passes = int((fold_rates[layer, projection] >= 0.8).sum())
            base_gate = gate_pass(metrics, design["discovery_gate"])
            embedding_gain = metrics["orientation"]["rate"] - float(embedding_rates[projection])
            position_gain = metrics["orientation"]["rate"] - position_rate
            passed = (
                layer > 0
                and base_gate
                and fold_passes >= int(design["minimum_fold_passes"])
                and embedding_gain >= float(design["discovery_gate"]["embedding_gain_min"])
                and position_gain >= float(design["discovery_gate"]["position_gain_min"])
            )
            cell_pass[layer, projection] = passed
            row_reports.append({
                "layer_with_embedding": layer,
                "projection_index": projection,
                "metrics": metrics,
                "fold_rates": fold_rates[layer, projection].tolist(),
                "fold_passes": fold_passes,
                "embedding_orientation_rate": float(embedding_rates[projection]),
                "embedding_gain": embedding_gain,
                "position_baseline_rate": position_rate,
                "position_gain": position_gain,
                "cell_gate_pass": passed,
            })
        cell_reports.append(row_reports)

    consensus = cell_pass.sum(axis=1) >= int(design["projection_consensus_required"])
    platforms = []
    start = None
    for layer, passed in enumerate(list(consensus) + [False]):
        if passed and start is None:
            start = layer
        elif not passed and start is not None:
            layers = list(range(start, layer))
            if len(layers) >= int(design["minimum_contiguous_layers"]):
                platforms.append({
                    "platform_id": f"glm4_world_geometry_L{layers[0]}_L{layers[-1]}",
                    "layers_with_embedding": layers,
                    "relative_depth_start": layers[0] / max(1, n_layers - 1),
                    "relative_depth_end": layers[-1] / max(1, n_layers - 1),
                    "projection_passes_by_layer": {
                        str(local_layer): np.where(cell_pass[local_layer])[0].astype(int).tolist()
                        for local_layer in layers
                    },
                })
            start = None
    platforms.sort(key=lambda item: (-len(item["layers_with_embedding"]), item["layers_with_embedding"][0]))
    for rank, platform in enumerate(platforms, start=1):
        platform["selection_rank"] = rank
        platform["primary"] = rank == 1
    diagnostics = {
        "position_baseline_orientation_rate": position_rate,
        "embedding_orientation_rates": embedding_rates.tolist(),
        "cell_pass": cell_pass.tolist(),
        "cell_reports": cell_reports,
        "consensus_layers": np.where(consensus)[0].astype(int).tolist(),
    }
    return platforms, full_directions, full_thresholds, diagnostics


def platform_scores(
    examples: dict[str, Any],
    directions: np.ndarray,
    thresholds: np.ndarray,
    platform: dict[str, Any],
) -> np.ndarray:
    scores = []
    for layer in platform["layers_with_embedding"]:
        for projection in platform["projection_passes_by_layer"][str(layer)]:
            local = examples["features"][:, layer, projection]
            scores.append(local @ directions[layer, projection] - thresholds[layer, projection])
    return np.mean(np.stack(scores, axis=1), axis=1)


def save_stage(stage: str, projected: np.ndarray, metadata: list[dict[str, Any]]) -> tuple[Path, Path]:
    array_path = OUT_DIR / f"phase532_glm4_{stage}_projection.npz"
    np.savez_compressed(array_path, projected=projected)
    metadata_path = OUT_DIR / f"phase532_glm4_{stage}_metadata.jsonl"
    write_jsonl(metadata_path, metadata)
    return array_path, metadata_path


def run_glm4(contract: dict[str, Any], batch_size: int, use_8bit: bool) -> Path:
    design = contract["physical_design"]
    roles = tuple(design["position_roles"])
    seeds = tuple(int(seed) for seed in design["projection_seeds"])
    model = None
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "phase532_glm4_world_geometry_summary.json"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase532 requires CUDA")
        model, tokenizer, device = load_model("glm4", use_8bit=True if use_8bit else None)
        discovery_rows = read_jsonl(ROOT / contract["split_files"]["discovery"]["path"])
        discovery_projected, discovery_meta, model_info = collect(
            model,
            tokenizer,
            device,
            discovery_rows,
            roles,
            seeds,
            int(design["projection_dimension"]),
            batch_size,
            "discovery",
        )
        discovery_examples = build_examples(canonical_endpoints(discovery_projected, discovery_meta), discovery_meta)
        platforms, directions, thresholds, diagnostics = discover(discovery_examples, discovery_meta, design)
        discovery_array_path, discovery_meta_path = save_stage("discovery", discovery_projected, discovery_meta)
        observer_path = OUT_DIR / "phase532_glm4_frozen_world_geometry_observers.npz"
        np.savez_compressed(observer_path, directions=directions, thresholds=thresholds)
        ledger_path = OUT_DIR / "phase532_glm4_frozen_world_geometry_ledger.json"
        ledger = {
            "schema_version": "phase532_glm4_frozen_world_geometry_ledger.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "frozen_before_prediction_read",
            "model": "glm4",
            "model_info": model_info,
            "discovery_row_count": len(discovery_rows),
            "platform_count": len(platforms),
            "platforms": platforms,
            "diagnostics": diagnostics,
            "observer_path": str(observer_path.relative_to(ROOT)),
            "observer_sha256": sha256_file(observer_path),
            "discovery_projection_path": str(discovery_array_path.relative_to(ROOT)),
            "discovery_projection_sha256": sha256_file(discovery_array_path),
            "discovery_metadata_path": str(discovery_meta_path.relative_to(ROOT)),
            "discovery_metadata_sha256": sha256_file(discovery_meta_path),
            "prediction_splits_read": False,
            "sealed_split_read": False,
            "causal": False,
        }
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if not platforms:
            payload = {
                "schema_version": "phase532_glm4_role_normalized_world_geometry.v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "stopped_no_discovery_platform",
                "model": "glm4",
                "runtime_seconds": time.monotonic() - started,
                "discovery_row_count": len(discovery_rows),
                "platform_count": 0,
                "frozen_ledger_path": str(ledger_path.relative_to(ROOT)),
                "prediction_splits_read": False,
                "sealed_split_read": False,
                "cuda_used": True,
                "causal": False,
            }
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(summary_path)
            return summary_path

        prediction_reports = {}
        all_prediction_pass = True
        for split in ("entity_prediction", "relation_prediction"):
            rows = read_jsonl(ROOT / contract["split_files"][split]["path"])
            projected, metadata, prediction_info = collect(
                model,
                tokenizer,
                device,
                rows,
                roles,
                seeds,
                int(design["projection_dimension"]),
                batch_size,
                split,
            )
            if prediction_info != model_info:
                raise RuntimeError("model info drift")
            examples = build_examples(canonical_endpoints(projected, metadata), metadata)
            reports = []
            for platform in platforms:
                scores = platform_scores(examples, directions, thresholds, platform)
                metrics = score_metrics(scores, examples, metadata)
                passed = gate_pass(metrics, design["prediction_gate"])
                reports.append({
                    "platform_id": platform["platform_id"],
                    "selection_rank": platform["selection_rank"],
                    "metrics": metrics,
                    "prediction_gate_pass": passed,
                })
            array_path, meta_path = save_stage(split, projected, metadata)
            primary_pass = bool(reports and reports[0]["prediction_gate_pass"])
            all_prediction_pass &= primary_pass
            prediction_reports[split] = {
                "row_count": len(rows),
                "platform_reports": reports,
                "primary_prediction_gate_pass": primary_pass,
                "projection_path": str(array_path.relative_to(ROOT)),
                "projection_sha256": sha256_file(array_path),
                "metadata_path": str(meta_path.relative_to(ROOT)),
                "metadata_sha256": sha256_file(meta_path),
            }
        payload = {
            "schema_version": "phase532_glm4_role_normalized_world_geometry.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "model": "glm4",
            "runtime_seconds": time.monotonic() - started,
            "discovery_row_count": len(discovery_rows),
            "platform_count": len(platforms),
            "platforms": platforms,
            "prediction_reports": prediction_reports,
            "primary_platform_all_predictions_pass": all_prediction_pass,
            "frozen_ledger_path": str(ledger_path.relative_to(ROOT)),
            "prediction_splits_read": True,
            "sealed_split_read": False,
            "cuda_used": True,
            "evidence_boundary": {
                "observer_scaffolded_world_geometry": True,
                "natural_unprompted_world_state": False,
                "query_platform_bridge": False,
                "compute_edge": False,
                "causal": False,
                "component_head_channel_neuron_scan": False,
            },
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
        return summary_path
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_model(model_name: str, batch_size: int, use_8bit: bool) -> Path:
    contract, authorization = verify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if model_name not in authorization["fresh_physical_authorized_models"]:
        path = OUT_DIR / f"phase532_{model_name}_world_geometry_summary.json"
        payload = {
            "schema_version": "phase532_glm4_role_normalized_world_geometry.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_authorized",
            "model": model_name,
            "cuda_used": False,
            "model_weights_loaded": False,
            "prediction_splits_read": False,
            "sealed_split_read": False,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
        return path
    if model_name != "glm4":
        raise RuntimeError("Phase530 fresh physical protocol authorizes GLM4 only")
    return run_glm4(contract, batch_size, use_8bit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    run_model(args.model, args.batch_size, args.use_8bit)


if __name__ == "__main__":
    main()
