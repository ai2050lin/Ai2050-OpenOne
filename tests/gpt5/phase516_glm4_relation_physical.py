#!/usr/bin/env python3
"""Collect and independently predict the GLM4 Phase516 relation trajectory."""

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


PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase516_relation_physical_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase516_frozen_relation_physical_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase516_relation_physical_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase516_relation_physical_protocol.py"
PHASE509_DIR = ROOT / "tests/gpt5/result/phase509_dual_contract_protocol"
FIT_PATH = PHASE509_DIR / "phase509_physical_fit_relation.jsonl"
PREDICTION_PATH = PHASE509_DIR / "phase509_physical_prediction_relation.jsonl"
OUT_DIR = ROOT / "tests/gpt5/result/phase516_glm4_relation_physical"
SURFACES = ("identity", "native_plain_candidate")
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


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def binary_rate(predictions: np.ndarray, truths: np.ndarray) -> dict[str, Any]:
    n = int(len(truths))
    k = int((predictions == truths).sum())
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def flatten(samples: list[dict[str, Any]], split: str, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"invalid {split} relation sample")
        for variant in sample["variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_pair_id": sample["source_pair_id"],
                "split": split,
                "truth_value": sample["truth_value"],
                "world_role": sample["world_role"],
                "pair_index": sample["pair_index"],
                "length_control": sample["length_control"],
                "fact_order": sample["fact_order"],
                "relation_verb": sample["relation_verb"],
                "surface": variant["surface"],
                "prompt": variant["prompt"],
                "true_candidate": variant["true_candidate"],
                "false_candidate": variant["false_candidate"],
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


def random_projection(d_model: int, dimension: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (d_model, dimension), generator=generator, dtype=torch.int8)
    return signs.float().mul_(2).sub_(1).div_(math.sqrt(dimension)).to(device)


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"candidate {text!r} is not one token: {ids}")
    return int(ids[0])


def collect(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    roles: tuple[str, ...],
    projection_seed: int,
    projection_dimension: int,
    batch_size: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    info = get_model_info(model, "glm4")
    projection = random_projection(info.d_model, projection_dimension, projection_seed, device)
    true_id = single_token_id(tokenizer, " true")
    false_id = single_token_id(tokenizer, " false")
    tokenizer.padding_side = "left"
    projected_batches = []
    norm_batches = []
    metadata = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        local_positions = [role_positions(tokenizer, row, roles) for row in batch]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        width = int(encoded["input_ids"].shape[1])
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        padded_positions = []
        for length, positions in zip(lengths, local_positions, strict=True):
            pad = width - int(length)
            padded_positions.append([pad + positions[role] for role in roles])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("model returned no hidden states")
        margins = (outputs.logits[:, -1, true_id] - outputs.logits[:, -1, false_id]).float().cpu().numpy()
        batch_indices = torch.arange(len(batch), device=device).unsqueeze(1)
        position_tensor = torch.tensor(padded_positions, device=device, dtype=torch.long)
        layer_projects = []
        layer_norms = []
        for hidden in hidden_states:
            selected = hidden[batch_indices, position_tensor].float()
            layer_projects.append((selected @ projection).cpu())
            layer_norms.append(torch.linalg.vector_norm(selected, dim=-1).cpu())
        projected = torch.stack(layer_projects, dim=1)
        norms = torch.stack(layer_norms, dim=1)
        projected_batches.append(projected.to(torch.float16).numpy())
        norm_batches.append(norms.to(torch.float32).numpy())
        for row, positions, length, margin in zip(batch, local_positions, lengths, margins, strict=True):
            metadata.append({
                **{key: value for key, value in row.items() if key not in {"prompt", "role_char_ends", "true_candidate", "false_candidate"}},
                "token_length": int(length),
                "role_token_positions": positions,
                "output_true_minus_false_margin": float(margin),
                "output_candidate_prediction": bool(margin > 0),
                "output_candidate_correct": bool((margin > 0) == row["truth_value"]),
            })
        del outputs, hidden_states, projected, norms
        if start == 0 or start + len(batch) == len(rows) or (start // batch_size) % 8 == 7:
            log(f"glm4 {stage} {min(start + len(batch), len(rows))}/{len(rows)}")
    arrays = np.concatenate(projected_batches, axis=0)
    norms = np.concatenate(norm_batches, axis=0)
    return arrays, norms, metadata, {
        "n_layers_with_embedding": int(arrays.shape[1]),
        "d_model": int(info.d_model),
        "projection_dimension": projection_dimension,
        "projection_seed": projection_seed,
        "position_roles": list(roles),
    }


def unit_vectors(values: np.ndarray) -> np.ndarray:
    vectors = values.astype(np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def fit_observers(vectors: np.ndarray, truths: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = vectors[mask]
    labels = truths[mask]
    mean_true = train[labels].mean(axis=0)
    mean_false = train[~labels].mean(axis=0)
    directions = mean_true - mean_false
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    thresholds = np.einsum("lrd,lrd->lr", (mean_true + mean_false) / 2, directions)
    return directions.astype(np.float32), thresholds.astype(np.float32)


def predict_grid(vectors: np.ndarray, directions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    scores = np.einsum("nlrd,lrd->nlr", vectors, directions) - thresholds[None, :, :]
    return scores > 0


def pair_rate(predictions: np.ndarray, metadata: list[dict[str, Any]], truths: np.ndarray) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row["source_pair_id"]].append(index)
    values = []
    for pair_id, indices in groups.items():
        if len(indices) != 4:
            raise RuntimeError(f"incomplete four-way pair {pair_id}: {len(indices)}")
        values.append(bool(np.all(predictions[indices] == truths[indices])))
    array = np.asarray(values, dtype=bool)
    return binary_rate(array, np.ones_like(array, dtype=bool))


def window_metrics(
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
        "four_way_pair": pair_rate(predictions, metadata, truths),
    }


def subset(
    predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    indices = np.flatnonzero(mask)
    return predictions[indices], truths[indices], [metadata[index] for index in indices]


def selection_grid(
    grid_predictions: np.ndarray,
    truths: np.ndarray,
    metadata: list[dict[str, Any]],
    selection_mask: np.ndarray,
    roles: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for layer in range(grid_predictions.shape[1]):
        for role_index, role in enumerate(roles):
            pred, local_truths, local_meta = subset(
                grid_predictions[:, layer, role_index], truths, metadata, selection_mask
            )
            metrics = window_metrics(pred, local_truths, local_meta)
            min_surface = min(item["rate"] for item in metrics["by_surface"].values())
            rows.append({
                "layer_with_embedding": layer,
                "position_role": role,
                "minimum_surface_accuracy": min_surface,
                "metrics": metrics,
            })
    role_order = {role: index for index, role in enumerate(roles)}
    primary = max(
        rows,
        key=lambda row: (
            row["minimum_surface_accuracy"],
            row["metrics"]["four_way_pair"]["rate"],
            row["metrics"]["overall"]["rate"],
            -row["layer_with_embedding"],
            -role_order[row["position_role"]],
        ),
    )
    return rows, primary


def distance_controls(
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    world_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    surface_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        world_groups[(row["source_pair_id"], row["surface"])].append(index)
        surface_groups[row["sample_id"]].append(index)
    world_distances = []
    for indices in world_groups.values():
        if len(indices) != 2:
            raise RuntimeError("incomplete world distance pair")
        world_distances.append(float(np.linalg.norm(vectors[indices[0]] - vectors[indices[1]])))
    surface_distances = []
    for indices in surface_groups.values():
        if len(indices) != 2:
            raise RuntimeError("incomplete surface distance pair")
        surface_distances.append(float(np.linalg.norm(vectors[indices[0]] - vectors[indices[1]])))
    return {
        "mean_world_pair_distance": float(np.mean(world_distances)),
        "mean_surface_pair_distance": float(np.mean(surface_distances)),
        "world_to_surface_distance_ratio": float(np.mean(world_distances) / max(np.mean(surface_distances), 1e-8)),
        "world_pair_count": len(world_distances),
        "surface_pair_count": len(surface_distances),
        "interpretation": "descriptive projected-distance control only",
    }


def random_label_controls(
    fit_vectors: np.ndarray,
    fit_truths: np.ndarray,
    fit_metadata: list[dict[str, Any]],
    train_mask: np.ndarray,
    prediction_vectors: np.ndarray,
    prediction_truths: np.ndarray,
    layer: int,
    role_index: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    controls = []
    pair_ids = sorted({row["source_pair_id"] for row in fit_metadata})
    for seed in seeds:
        rng = np.random.default_rng(seed)
        flips = {pair_id: bool(value) for pair_id, value in zip(pair_ids, rng.integers(0, 2, len(pair_ids)), strict=True)}
        random_truths = np.asarray([
            bool(truth) ^ flips[row["source_pair_id"]]
            for truth, row in zip(fit_truths, fit_metadata, strict=True)
        ])
        direction, threshold = fit_observers(fit_vectors, random_truths, train_mask)
        predictions = predict_grid(prediction_vectors, direction, threshold)[:, layer, role_index]
        controls.append({"seed": seed, "prediction": binary_rate(predictions, prediction_truths)})
    return controls


def verify() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase516 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase516 protocol source changed after freeze")
    if sha256_file(FIT_PATH) != contract["fit_split"]["sha256"]:
        raise RuntimeError("Phase516 fit hash drift")
    if sha256_file(PREDICTION_PATH) != contract["prediction_split"]["sha256"]:
        raise RuntimeError("Phase516 prediction hash drift")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    contract = verify()
    roles = tuple(contract["position_roles"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase516 requires CUDA")
        model, tokenizer, device = load_model("glm4", use_8bit=True if args.use_8bit else None)

        fit_rows = flatten(read_jsonl(FIT_PATH), "physical_fit", roles)
        fit_projected, fit_norms, fit_metadata, model_info = collect(
            model,
            tokenizer,
            device,
            fit_rows,
            roles,
            contract["projection"]["seed"],
            contract["projection"]["dimension"],
            args.batch_size,
            "fit",
        )
        fit_vectors = unit_vectors(fit_projected)
        fit_truths = np.asarray([row["truth_value"] for row in fit_metadata], dtype=bool)
        train_mask = np.asarray([row["pair_index"] % 2 == 0 for row in fit_metadata], dtype=bool)
        selection_mask = ~train_mask
        directions, thresholds = fit_observers(fit_vectors, fit_truths, train_mask)
        fit_predictions = predict_grid(fit_vectors, directions, thresholds)
        selection_rows, primary = selection_grid(
            fit_predictions, fit_truths, fit_metadata, selection_mask, roles
        )

        observer_npz = OUT_DIR / "phase516_glm4_frozen_observer.npz"
        np.savez_compressed(observer_npz, directions=directions, thresholds=thresholds)
        fit_npz = OUT_DIR / "phase516_glm4_fit_projection.npz"
        np.savez_compressed(fit_npz, projected=fit_projected, norms=fit_norms)
        fit_meta_path = OUT_DIR / "phase516_glm4_fit_metadata.jsonl"
        write_jsonl(fit_meta_path, fit_metadata)
        ledger = {
            "schema_version": "phase516_glm4_frozen_observer_ledger.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "glm4",
            "contract": "R",
            "model_info": model_info,
            "observer_train_rows": int(train_mask.sum()),
            "window_selection_rows": int(selection_mask.sum()),
            "selection_grid": selection_rows,
            "primary_window": primary,
            "observer_npz_path": str(observer_npz.relative_to(ROOT)),
            "observer_npz_sha256": sha256_file(observer_npz),
            "fit_projection_path": str(fit_npz.relative_to(ROOT)),
            "fit_projection_sha256": sha256_file(fit_npz),
            "fit_metadata_path": str(fit_meta_path.relative_to(ROOT)),
            "prediction_split_read": False,
            "sealed_split_read": False,
            "causal": False,
        }
        ledger_path = OUT_DIR / "phase516_glm4_frozen_observer_ledger.json"
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # The prediction file is deliberately read only after the observer ledger exists.
        prediction_rows = flatten(read_jsonl(PREDICTION_PATH), "physical_prediction", roles)
        pred_projected, pred_norms, pred_metadata, prediction_model_info = collect(
            model,
            tokenizer,
            device,
            prediction_rows,
            roles,
            contract["projection"]["seed"],
            contract["projection"]["dimension"],
            args.batch_size,
            "prediction",
        )
        if prediction_model_info != model_info:
            raise RuntimeError("model info changed between fit and prediction")
        pred_vectors = unit_vectors(pred_projected)
        pred_truths = np.asarray([row["truth_value"] for row in pred_metadata], dtype=bool)
        pred_grid = predict_grid(pred_vectors, directions, thresholds)
        layer = int(primary["layer_with_embedding"])
        role_index = roles.index(primary["position_role"])
        primary_predictions = pred_grid[:, layer, role_index]
        primary_metrics = window_metrics(primary_predictions, pred_truths, pred_metadata)
        gate = contract["prediction_gate"]
        prediction_gate = (
            primary_metrics["by_surface"]["identity"]["lcb95"] >= gate["identity_lcb95_min"]
            and primary_metrics["by_surface"]["native_plain_candidate"]["lcb95"] >= gate["native_plain_lcb95_min"]
            and primary_metrics["overall"]["lcb95"] >= gate["overall_lcb95_min"]
            and primary_metrics["four_way_pair"]["lcb95"] >= gate["four_way_pair_lcb95_min"]
        )
        role_trajectory = []
        for current_layer in range(pred_grid.shape[1]):
            metrics = window_metrics(
                pred_grid[:, current_layer, role_index], pred_truths, pred_metadata
            )
            role_trajectory.append({
                "layer_with_embedding": current_layer,
                "relative_depth": current_layer / max(1, pred_grid.shape[1] - 1),
                "position_role": primary["position_role"],
                "metrics": metrics,
            })
        primary_vectors = pred_vectors[:, layer, role_index]
        controls = distance_controls(primary_vectors, pred_metadata)
        random_controls = random_label_controls(
            fit_vectors,
            fit_truths,
            fit_metadata,
            train_mask,
            pred_vectors,
            pred_truths,
            layer,
            role_index,
            contract["random_label_controls"]["seeds"],
        )
        output_predictions = np.asarray(
            [row["output_candidate_prediction"] for row in pred_metadata], dtype=bool
        )
        hidden_output_agreement = binary_rate(primary_predictions, output_predictions)

        pred_npz = OUT_DIR / "phase516_glm4_prediction_projection.npz"
        np.savez_compressed(pred_npz, projected=pred_projected, norms=pred_norms)
        pred_meta_path = OUT_DIR / "phase516_glm4_prediction_metadata.jsonl"
        write_jsonl(pred_meta_path, pred_metadata)
        summary = {
            "schema_version": "phase516_glm4_relation_physical_summary.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "model": "glm4",
            "contract": "R",
            "scope": "model-specific observational relation trajectory",
            "runtime_seconds": time.monotonic() - started,
            "cuda_used": True,
            "model_weights_loaded": True,
            "fit_row_count": len(fit_rows),
            "prediction_row_count": len(prediction_rows),
            "model_info": model_info,
            "primary_window": primary,
            "primary_prediction": primary_metrics,
            "primary_prediction_gate_pass": prediction_gate,
            "primary_distance_controls": controls,
            "random_label_controls": random_controls,
            "hidden_output_prediction_agreement": hidden_output_agreement,
            "prediction_role_trajectory": role_trajectory,
            "prediction_projection_path": str(pred_npz.relative_to(ROOT)),
            "prediction_projection_sha256": sha256_file(pred_npz),
            "prediction_metadata_path": str(pred_meta_path.relative_to(ROOT)),
            "observer_ledger_path": str(ledger_path.relative_to(ROOT)),
            "observer_ledger_sha256": sha256_file(ledger_path),
            "evidence_boundary": {
                "shared_cross_model_claim": False,
                "compute_transport_measured": False,
                "causal_intervention": False,
                "head_channel_neuron_scan": False,
                "sealed_split_read": False,
            },
        }
        summary_path = OUT_DIR / "phase516_glm4_relation_physical_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(ledger_path)
        print(summary_path)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
