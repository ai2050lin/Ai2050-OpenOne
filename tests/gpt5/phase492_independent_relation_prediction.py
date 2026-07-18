#!/usr/bin/env python3
"""Phase492 independent prediction for frozen late relation windows.

One model per invocation. The layer, position, projection seed, direction fit,
and gates are frozen by Phase491 before this script reads the physical-
prediction split. Raw hidden states are never saved.
"""

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
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_model_info, load_model, release_model  # noqa: E402


PHASE487_DIR = ROOT / "tests" / "gpt5" / "result" / "phase487_dual_observer_native_core_protocol"
SAMPLES_PATH = PHASE487_DIR / "phase487_physical_prediction_samples.jsonl"
MANIFEST_PATH = PHASE487_DIR / "phase487_manifest.json"
FREEZE_PATH = (
    ROOT
    / "tests"
    / "gpt5"
    / "result"
    / "phase491_geometry_contamination_audit"
    / "phase491_physical_prediction_freeze.json"
)
GEOMETRY_DIR = ROOT / "tests" / "gpt5" / "result" / "phase490_open_native_relation_geometry"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase492_independent_relation_prediction"
NATIVE_TRACKS = ("identity", "native_plain_candidate")
MODEL_SEEDS = {"qwen3": 490031, "glm4": 490037}
PROJECTION_DIM = 64
Z = 1.96


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def flatten_prediction(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        if sample["split"] != "physical_prediction" or sample["sealed"]:
            raise RuntimeError("Phase492 encountered a non-prediction or sealed sample")
        if sample["label_mapping"] != "mu_ab":
            continue
        for variant in sample["surface_variants"]:
            if variant["track"] not in NATIVE_TRACKS:
                continue
            geometry_case_id = f"{sample['source_pair_id']}::{sample['pair_role']}"
            key = (geometry_case_id, variant["track"])
            if key in seen:
                raise RuntimeError(f"Duplicate prediction key {key}")
            seen.add(key)
            rows.append({
                "geometry_case_id": geometry_case_id,
                "source_pair_id": sample["source_pair_id"],
                "sample_id": sample["sample_id"],
                "family": sample["family"],
                "pair_index": sample["pair_index"],
                "pair_role": sample["pair_role"],
                "truth_value": sample["truth_value"],
                "target_slot": sample["target_slot"],
                "fact_order": sample["fact_order"],
                "track": variant["track"],
                "prompt": variant["semantic_prompt"],
            })
    if len(rows) != 512:
        raise RuntimeError(f"Expected 512 prediction rows, got {len(rows)}")
    return rows


def random_projection(d_model: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (d_model, PROJECTION_DIM), generator=generator, dtype=torch.int8)
    return signs.float().mul_(2).sub_(1).div_(math.sqrt(PROJECTION_DIM)).to(device)


def collect_window(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    model_key: str,
    layer_index: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    info = get_model_info(model, model_key)
    projection = random_projection(info.d_model, MODEL_SEEDS[model_key], device)
    tokenizer.padding_side = "left"
    projected_batches = []
    norm_batches = []
    metadata = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or layer_index >= len(hidden_states):
            raise RuntimeError(f"Frozen layer {layer_index} unavailable")
        selected = hidden_states[layer_index][:, -1, :].float()
        projected_batches.append((selected @ projection).to(torch.float16).cpu().numpy())
        norm_batches.append(torch.linalg.vector_norm(selected, dim=-1).cpu().numpy())
        for row, length in zip(batch, lengths, strict=True):
            metadata.append({
                **{key: value for key, value in row.items() if key != "prompt"},
                "token_length": int(length),
            })
        del outputs, hidden_states, selected
        log(f"{model_key} prediction {min(start + len(batch), len(rows))}/{len(rows)}")
    return (
        np.concatenate(projected_batches).astype(np.float16),
        np.concatenate(norm_batches).astype(np.float32),
        metadata,
        {
            "d_model": info.d_model,
            "n_transformer_layers": info.n_layers,
            "layer_with_embedding": layer_index,
            "position_role": "prompt_end",
            "projection_dim": PROJECTION_DIM,
            "projection_seed": MODEL_SEEDS[model_key],
        },
    )


def unit_vectors(vectors: np.ndarray) -> np.ndarray:
    values = vectors.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def geometry_pairs(metadata: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    by_case_track: dict[tuple[str, str], int] = {}
    by_pair_track: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(metadata):
        by_case_track[(row["geometry_case_id"], row["track"])] = index
        by_pair_track[(row["source_pair_id"], row["track"])][row["pair_role"]] = index
    surface = [
        (
            by_case_track[(case_id, "identity")],
            by_case_track[(case_id, "native_plain_candidate")],
        )
        for case_id in sorted({row["geometry_case_id"] for row in metadata})
    ]
    relation = []
    for key, roles in sorted(by_pair_track.items()):
        if set(roles) != {"entailed", "counterfactual"}:
            raise RuntimeError(f"Incomplete relation pair {key}")
        relation.append((roles["entailed"], roles["counterfactual"]))
    return {"surface": surface, "relation": relation}


def mean_distance(vectors: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    left = vectors[[pair[0] for pair in pairs]]
    right = vectors[[pair[1] for pair in pairs]]
    return float(np.linalg.norm(left - right, axis=-1).mean())


def filter_pairs(metadata: list[dict[str, Any]], pairs: list[tuple[int, int]], family: str) -> list[tuple[int, int]]:
    return [pair for pair in pairs if metadata[pair[0]]["family"] == family and metadata[pair[1]]["family"] == family]


def relation_geometry(vectors: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = geometry_pairs(metadata)
    d_surface = mean_distance(vectors, pairs["surface"])
    d_relation = mean_distance(vectors, pairs["relation"])
    deltas = vectors[[pair[0] for pair in pairs["relation"]]] - vectors[[pair[1] for pair in pairs["relation"]]]
    delta_units = deltas / np.maximum(np.linalg.norm(deltas, axis=-1, keepdims=True), 1e-8)
    families = sorted({row["family"] for row in metadata})
    by_family = {}
    for family in families:
        surface = filter_pairs(metadata, pairs["surface"], family)
        relation = filter_pairs(metadata, pairs["relation"], family)
        ds = mean_distance(vectors, surface)
        dr = mean_distance(vectors, relation)
        by_family[family] = {
            "d_native_surface": ds,
            "d_relation_counterfactual": dr,
            "q_native": (dr - ds) / (dr + ds + 1e-8),
        }
    return {
        "pair_counts": {key: len(value) for key, value in pairs.items()},
        "d_native_surface": d_surface,
        "d_relation_counterfactual": d_relation,
        "q_native": (d_relation - d_surface) / (d_relation + d_surface + 1e-8),
        "relation_direction_coherence": float(np.linalg.norm(delta_units.mean(axis=0))),
        "by_family": by_family,
    }


def fit_direction(vectors: np.ndarray, metadata: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    true_vectors = vectors[[index for index, row in enumerate(metadata) if row["truth_value"]]]
    false_vectors = vectors[[index for index, row in enumerate(metadata) if not row["truth_value"]]]
    mean_true = true_vectors.mean(axis=0)
    mean_false = false_vectors.mean(axis=0)
    direction = mean_true - mean_false
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    midpoint = (mean_true + mean_false) / 2
    return direction, midpoint


def rate(predictions: np.ndarray, truths: np.ndarray) -> dict[str, Any]:
    correct = int((predictions == truths).sum())
    lcb, ucb = wilson(correct, len(truths))
    return {"n": len(truths), "count": correct, "rate": correct / len(truths), "lcb95": lcb, "ucb95": ucb}


def prediction_report(
    train_vectors: np.ndarray,
    train_metadata: list[dict[str, Any]],
    test_vectors: np.ndarray,
    test_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    direction, midpoint = fit_direction(train_vectors, train_metadata)
    scores = (test_vectors - midpoint) @ direction
    predictions = scores > 0
    truths = np.asarray([row["truth_value"] for row in test_metadata], dtype=bool)
    overall = rate(predictions, truths)
    by_track = {}
    for track in NATIVE_TRACKS:
        indices = np.asarray([index for index, row in enumerate(test_metadata) if row["track"] == track])
        by_track[track] = rate(predictions[indices], truths[indices])
    by_family = {}
    for family in sorted({row["family"] for row in test_metadata}):
        indices = np.asarray([index for index, row in enumerate(test_metadata) if row["family"] == family])
        by_family[family] = rate(predictions[indices], truths[indices])
    return {
        "fit_split": "geometry_window",
        "test_split": "physical_prediction",
        "fit_rows": len(train_metadata),
        "test_rows": len(test_metadata),
        "direction_norm_after_normalization": float(np.linalg.norm(direction)),
        "overall": overall,
        "by_track": by_track,
        "by_family": by_family,
        "score_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean_true": float(scores[truths].mean()),
            "mean_false": float(scores[~truths].mean()),
        },
    }


def load_geometry_training(model_key: str, layer_index: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays = np.load(GEOMETRY_DIR / f"phase490_{model_key}_projected_states.npz")
    projected = arrays["projected"]
    metadata = load_jsonl(GEOMETRY_DIR / f"phase490_{model_key}_metadata.jsonl")
    if projected.shape[0] != len(metadata) or layer_index >= projected.shape[1]:
        raise RuntimeError("Geometry training projection ledger mismatch")
    # prompt_end is role index 2 in the frozen Phase490 role order.
    return unit_vectors(projected[:, layer_index, 2]), metadata


def verify_freeze(model_key: str) -> dict[str, Any]:
    freeze = load_json(FREEZE_PATH)
    if not freeze["authorization"]["physical_prediction_split_read"]:
        raise RuntimeError("Physical prediction split is not authorized")
    if model_key not in freeze["models_in_required_order"]:
        raise RuntimeError(f"{model_key} is not in the frozen prediction model list")
    manifest = load_json(MANIFEST_PATH)
    expected = manifest["split_files"]["physical_prediction"]["sha256"]
    if sha256_file(SAMPLES_PATH) != expected:
        raise RuntimeError("Physical-prediction split hash drift")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SEEDS), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    freeze = verify_freeze(args.model)
    window = freeze["windows"][args.model]
    layer_index = int(window["layer_with_embedding"])
    # This is the first read of the independent prediction samples.
    rows = flatten_prediction(load_jsonl(SAMPLES_PATH))
    train_vectors, train_metadata = load_geometry_training(args.model, layer_index)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
        projected, norms, metadata, run_info = collect_window(
            model, tokenizer, device, rows, args.model, layer_index, args.batch_size
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runtime = time.monotonic() - started
    test_vectors = unit_vectors(projected)
    geometry = relation_geometry(test_vectors, metadata)
    prediction = prediction_report(train_vectors, train_metadata, test_vectors, metadata)
    gate = (
        geometry["q_native"] > 0
        and all(payload["q_native"] > 0 for payload in geometry["by_family"].values())
        and geometry["relation_direction_coherence"] >= freeze["independent_gate"]["direction_coherence_min"]
        and prediction["overall"]["lcb95"] > freeze["independent_gate"]["truth_prediction_lcb95_above"]
        and all(payload["lcb95"] > 0.5 for payload in prediction["by_track"].values())
        and all(payload["lcb95"] > 0.5 for payload in prediction["by_family"].values())
    )
    npz_path = OUT_DIR / f"phase492_{args.model}_projected_window.npz"
    metadata_path = OUT_DIR / f"phase492_{args.model}_metadata.jsonl"
    summary_path = OUT_DIR / f"phase492_{args.model}_summary.json"
    np.savez_compressed(npz_path, projected=projected, hidden_norms=norms)
    write_jsonl(metadata_path, metadata)
    summary = {
        "schema_version": "phase492_independent_relation_prediction.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "independent_prediction_pass" if gate else "independent_prediction_fail",
        "model": args.model,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "runtime_seconds": runtime,
        "sealed_split_read": False,
        "input": str(SAMPLES_PATH.relative_to(ROOT)),
        "input_sha256": sha256_file(SAMPLES_PATH),
        "frozen_window": window,
        "run_info": run_info,
        "raw_hidden_states_saved": False,
        "independent_geometry": geometry,
        "independent_truth_prediction": prediction,
        "gate_pass": gate,
        "authorization": {
            "sealed_read_authorized": False,
            "causal_intervention_authorized": False,
            "head_channel_neuron_scan_authorized": False,
        },
        "allowed_claim": (
            "A passing result is an independent observational relation-state prediction at one frozen late prompt window."
        ),
        "forbidden_claim": (
            "A passing result is not causal localization, mediation, neuron closure, or universal cross-model coding."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(npz_path)
    print(metadata_path)
    print(summary_path)


if __name__ == "__main__":
    main()
