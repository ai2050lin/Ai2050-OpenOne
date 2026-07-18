#!/usr/bin/env python3
"""Phase490 open native-core relation geometry collection.

Runs one authorized model per invocation on the Phase487 geometry-window file.
Only identity and genuinely distinct native-plain tracks are collected. Compact
fixed random projections are saved; raw hidden states are discarded.
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
SAMPLES_PATH = PHASE487_DIR / "phase487_geometry_window_samples.jsonl"
MANIFEST_PATH = PHASE487_DIR / "phase487_manifest.json"
AUTH_PATH = (
    ROOT
    / "tests"
    / "gpt5"
    / "result"
    / "phase489_three_channel_behavior_analysis"
    / "phase489_open_physical_authorization.json"
)
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase490_open_native_relation_geometry"
NATIVE_TRACKS = ("identity", "native_plain_candidate")
POSITION_ROLES = ("evidence_end", "claim_end", "prompt_end")
PROJECTION_DIM = 64
MODEL_SEEDS = {"qwen3": 490031, "glm4": 490037}


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


def flatten_geometry(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        if sample["split"] != "geometry_window" or sample["sealed"]:
            raise RuntimeError("Phase490 encountered a non-geometry or sealed sample")
        # Semantic prompts are identical across label mappings. Keep one mapping
        # to avoid treating duplicated prompts as independent physical samples.
        if sample["label_mapping"] != "mu_ab":
            continue
        for variant in sample["surface_variants"]:
            if variant["track"] not in NATIVE_TRACKS:
                continue
            key = (f"{sample['source_pair_id']}::{sample['pair_role']}", variant["track"])
            if key in seen:
                raise RuntimeError(f"Duplicate geometry key {key}")
            seen.add(key)
            rows.append({
                "geometry_case_id": key[0],
                "source_pair_id": sample["source_pair_id"],
                "sample_id": sample["sample_id"],
                "family": sample["family"],
                "pair_index": sample["pair_index"],
                "pair_role": sample["pair_role"],
                "truth_value": sample["truth_value"],
                "target_slot": sample["target_slot"],
                "fact_order": sample["fact_order"],
                "track": variant["track"],
                "facts_text": " ".join(sample["facts"]),
                "claim": sample["claim"],
                "prompt": variant["semantic_prompt"],
            })
    if len(rows) != 512:
        raise RuntimeError(f"Expected 512 open native geometry rows, got {len(rows)}")
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
        raise RuntimeError(
            f"Could not map prefix boundary: common={common}, prefix={len(prefix)}"
        )
    return max(0, common - 1)


def role_positions(tokenizer: Any, row: dict[str, Any]) -> dict[str, int]:
    prompt = row["prompt"]
    fact_start = prompt.find(row["facts_text"])
    claim_start = prompt.find(row["claim"])
    if fact_start < 0 or claim_start < 0:
        raise RuntimeError("Facts or claim not found in rendered prompt")
    return {
        "evidence_end": prefix_position(tokenizer, prompt, fact_start + len(row["facts_text"])),
        "claim_end": prefix_position(tokenizer, prompt, claim_start + len(row["claim"])),
        "prompt_end": len(tokenizer.encode(prompt, add_special_tokens=True)) - 1,
    }


def random_projection(d_model: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (d_model, PROJECTION_DIM), generator=generator, dtype=torch.int8)
    matrix = signs.float().mul_(2).sub_(1).div_(math.sqrt(PROJECTION_DIM))
    return matrix.to(device)


def collect(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    model_key: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    info = get_model_info(model, model_key)
    projection = random_projection(info.d_model, MODEL_SEEDS[model_key], device)
    metadata = []
    projected_batches = []
    norm_batches = []
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        local_positions = [role_positions(tokenizer, row) for row in batch]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        width = int(encoded["input_ids"].shape[1])
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        padded_positions = []
        for length, positions in zip(lengths, local_positions, strict=True):
            pad = width - int(length)
            padded_positions.append([pad + positions[role] for role in POSITION_ROLES])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model returned no hidden states")
        layer_projects = []
        layer_norms = []
        batch_indices = torch.arange(len(batch), device=device).unsqueeze(1)
        position_tensor = torch.tensor(padded_positions, device=device, dtype=torch.long)
        for hidden in hidden_states:
            selected = hidden[batch_indices, position_tensor].float()
            layer_projects.append((selected @ projection).cpu())
            layer_norms.append(torch.linalg.vector_norm(selected, dim=-1).cpu())
        # [batch, layers, roles, projection]
        projected = torch.stack(layer_projects, dim=1)
        norms = torch.stack(layer_norms, dim=1)
        projected_batches.append(projected.to(torch.float16).numpy())
        norm_batches.append(norms.to(torch.float32).numpy())
        for row, positions, length in zip(batch, local_positions, lengths, strict=True):
            metadata.append({
                **{key: value for key, value in row.items() if key != "prompt"},
                "token_length": int(length),
                "role_token_positions": positions,
            })
        del outputs, hidden_states, projected, norms
        log(f"{model_key} geometry {min(start + len(batch), len(rows))}/{len(rows)}")
    arrays = np.concatenate(projected_batches, axis=0)
    norms = np.concatenate(norm_batches, axis=0)
    return arrays, norms, metadata, {
        "n_layers_with_embedding": arrays.shape[1],
        "d_model": info.d_model,
        "projection_dim": PROJECTION_DIM,
        "projection_seed": MODEL_SEEDS[model_key],
        "position_roles": list(POSITION_ROLES),
    }


def unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def mean_distance(vectors: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    left = vectors[[pair[0] for pair in pairs]]
    right = vectors[[pair[1] for pair in pairs]]
    return float(np.linalg.norm(left - right, axis=-1).mean())


def build_pairs(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    by_case_track: dict[tuple[str, str], int] = {}
    by_pair_track: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(metadata):
        by_case_track[(row["geometry_case_id"], row["track"])] = index
        by_pair_track[(row["source_pair_id"], row["track"])][row["pair_role"]] = index
    surface_pairs = []
    for case_id in {row["geometry_case_id"] for row in metadata}:
        surface_pairs.append((
            by_case_track[(case_id, "identity")],
            by_case_track[(case_id, "native_plain_candidate")],
        ))
    relation_pairs = []
    for key, roles in sorted(by_pair_track.items()):
        if set(roles) != {"entailed", "counterfactual"}:
            raise RuntimeError(f"Incomplete relation pair {key}")
        relation_pairs.append((roles["entailed"], roles["counterfactual"]))
    return {"surface": surface_pairs, "relation": relation_pairs}


def family_pairs(metadata: list[dict[str, Any]], pairs: list[tuple[int, int]], family: str) -> list[tuple[int, int]]:
    return [pair for pair in pairs if metadata[pair[0]]["family"] == family and metadata[pair[1]]["family"] == family]


def relation_direction_metrics(vectors: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, float]:
    deltas = vectors[[pair[0] for pair in pairs]] - vectors[[pair[1] for pair in pairs]]
    delta_norms = np.linalg.norm(deltas, axis=-1, keepdims=True)
    unit_deltas = deltas / np.maximum(delta_norms, 1e-8)
    coherence = float(np.linalg.norm(unit_deltas.mean(axis=0)))
    by_family: dict[str, list[np.ndarray]] = defaultdict(list)
    return {"coherence": coherence, "mean_delta_norm": float(delta_norms.mean())}


def analyze_geometry(projected: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = build_pairs(metadata)
    families = sorted({row["family"] for row in metadata})
    reports = []
    for layer in range(projected.shape[1]):
        for role_index, role in enumerate(POSITION_ROLES):
            vectors = unit_vectors(projected[:, layer, role_index].astype(np.float32))
            d_surface = mean_distance(vectors, pairs["surface"])
            d_relation = mean_distance(vectors, pairs["relation"])
            q = (d_relation - d_surface) / (d_relation + d_surface + 1e-8)
            family_quality = {}
            for family in families:
                surface_family = family_pairs(metadata, pairs["surface"], family)
                relation_family = family_pairs(metadata, pairs["relation"], family)
                ds = mean_distance(vectors, surface_family)
                dr = mean_distance(vectors, relation_family)
                family_quality[family] = {
                    "d_native_surface": ds,
                    "d_relation_counterfactual": dr,
                    "q_native": (dr - ds) / (dr + ds + 1e-8),
                }
            direction = relation_direction_metrics(vectors, pairs["relation"])
            qualifies = q > 0 and all(payload["q_native"] > 0 for payload in family_quality.values())
            reports.append({
                "layer_with_embedding": layer,
                "normalized_depth": layer / max(1, projected.shape[1] - 1),
                "position_role": role,
                "d_native_surface": d_surface,
                "d_relation_counterfactual": d_relation,
                "q_native": q,
                "relation_direction_coherence": direction["coherence"],
                "mean_relation_delta_norm": direction["mean_delta_norm"],
                "by_family": family_quality,
                "qualifies_open_window": qualifies,
            })
    selected = []
    for role in POSITION_ROLES:
        eligible = [row for row in reports if row["position_role"] == role and row["qualifies_open_window"]]
        if eligible:
            best = max(
                eligible,
                key=lambda row: (
                    min(payload["q_native"] for payload in row["by_family"].values()),
                    row["q_native"],
                    -row["layer_with_embedding"],
                ),
            )
            selected.append({
                "position_role": role,
                "layer_with_embedding": best["layer_with_embedding"],
                "normalized_depth": best["normalized_depth"],
                "q_native": best["q_native"],
                "selection_rule": "maximize minimum family q among q>0 in every family; tie by overall q then lower layer",
            })
    return {
        "pair_counts": {key: len(value) for key, value in pairs.items()},
        "reports": reports,
        "positive_window_count": sum(row["qualifies_open_window"] for row in reports),
        "selected_windows_for_independent_prediction": selected,
        "prediction_stage_candidate": len(selected) > 0,
    }


def verify_authorization(model_key: str) -> dict[str, Any]:
    authorization = load_json(AUTH_PATH)
    allowed = authorization["open_relation_geometry_models_in_order"]
    if model_key not in allowed:
        raise RuntimeError(f"{model_key} is not authorized for open relation geometry")
    manifest = load_json(MANIFEST_PATH)
    expected = manifest["split_files"]["geometry_window"]["sha256"]
    if sha256_file(SAMPLES_PATH) != expected:
        raise RuntimeError("Phase487 geometry-window hash drift")
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SEEDS), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    authorization = verify_authorization(args.model)
    rows = flatten_geometry(load_jsonl(SAMPLES_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
        projected, norms, metadata, run_info = collect(
            model, tokenizer, device, rows, args.model, args.batch_size
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runtime = time.monotonic() - started
    npz_path = OUT_DIR / f"phase490_{args.model}_projected_states.npz"
    metadata_path = OUT_DIR / f"phase490_{args.model}_metadata.jsonl"
    summary_path = OUT_DIR / f"phase490_{args.model}_summary.json"
    np.savez_compressed(npz_path, projected=projected, hidden_norms=norms)
    write_jsonl(metadata_path, metadata)
    geometry = analyze_geometry(projected, metadata)
    summary = {
        "schema_version": "phase490_open_native_relation_geometry.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open_geometry_window_complete",
        "model": args.model,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "runtime_seconds": runtime,
        "sealed_split_read": False,
        "input": str(SAMPLES_PATH.relative_to(ROOT)),
        "input_sha256": sha256_file(SAMPLES_PATH),
        "authorization_input": str(AUTH_PATH.relative_to(ROOT)),
        "authorization_models": authorization["open_relation_geometry_models_in_order"],
        "row_count": len(metadata),
        "raw_hidden_states_saved": False,
        "run_info": run_info,
        "geometry": geometry,
        "authorization": {
            "physical_prediction_candidate": geometry["prediction_stage_candidate"],
            "physical_prediction_authorized": False,
            "sealed_read_authorized": False,
            "head_channel_neuron_scan_authorized": False,
        },
        "limitations": [
            "Counterfactual pairs differ in claim attribute token as well as relation truth.",
            "Positive geometry is observational and cannot establish causal mediation.",
            "The geometry-window split selects candidates; independent prediction remains required.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(npz_path)
    print(metadata_path)
    print(summary_path)


if __name__ == "__main__":
    main()
