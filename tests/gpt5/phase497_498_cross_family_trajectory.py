#!/usr/bin/env python3
"""Phase497-498 train-family observer freeze and unseen-family trajectory.

One behavior-qualified CUDA model is loaded per invocation. The script first
collects only the two-family formation split, writes a frozen observer ledger,
and only then reads the open unseen-family prediction split. Raw hidden states
are never persisted; fixed 64-dimensional projections are saved.
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


PROTOCOL_DIR = ROOT / "tests" / "gpt5" / "result" / "phase494_cross_family_trajectory_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase494_frozen_contract.json"
FIT_PATH = PROTOCOL_DIR / "phase494_formation_fit_samples.jsonl"
PREDICTION_PATH = PROTOCOL_DIR / "phase494_family_prediction_samples.jsonl"
PROTOCOL_SOURCE = ROOT / "tests" / "gpt5" / "phase494_cross_family_trajectory_protocol.py"
AUTH_PATH = ROOT / "tests" / "gpt5" / "result" / "phase496_behavior_authorization" / "phase496_open_physical_authorization.json"
LEGACY_DIR = ROOT / "tests" / "gpt5" / "result" / "phase490_open_native_relation_geometry"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase497_498_cross_family_trajectory"
MODELS = ("qwen3", "glm4", "deepseek7b")
TRACKS = ("identity", "native_plain_candidate")
TRAIN_FAMILIES = ("marker_inheritance", "signal_assignment")
UNSEEN_FAMILIES = (
    "symmetric_pair",
    "directed_mentor",
    "transitive_precedence",
    "direct_nontransitive",
)
POSITION_ROLES = (
    "rules_end",
    "target_evidence_end",
    "distractor_evidence_end",
    "claim_entity_end",
    "claim_relation_end",
    "claim_end",
    "final_instruction_end",
    "prompt_end",
)
PROJECTION_DIM = 64
PROJECTION_SEEDS = {"qwen3": 490031, "glm4": 490037, "deepseek7b": 494041}
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
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def rate_from_arrays(predictions: np.ndarray, truths: np.ndarray) -> dict[str, Any]:
    n = int(len(truths))
    k = int((predictions == truths).sum())
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def flatten(samples: list[dict[str, Any]], split: str, allowed_families: set[str]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"Phase497-498 received an invalid {split} sample")
        if sample["family"] not in allowed_families:
            continue
        for variant in sample["surface_variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "world_case_id": sample["world_case_id"],
                "source_pair_id": sample["source_pair_id"],
                "split": split,
                "family": sample["family"],
                "pair_index": sample["pair_index"],
                "world_role": sample["world_role"],
                "truth_value": sample["truth_value"],
                "claim_polarity": sample["claim_polarity"],
                "length_control": sample["length_control"],
                "fact_order": sample["fact_order"],
                "track": variant["track"],
                "prompt": variant["semantic_prompt"],
                "role_char_ends": variant["role_char_ends"],
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
        raise RuntimeError(f"Could not map role boundary: common={common}, prefix={len(prefix)}")
    return max(0, common - 1)


def role_positions(tokenizer: Any, row: dict[str, Any]) -> dict[str, int]:
    return {
        role: prefix_position(tokenizer, row["prompt"], int(row["role_char_ends"][role]))
        for role in POSITION_ROLES
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
    stage: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    info = get_model_info(model, model_key)
    projection = random_projection(info.d_model, PROJECTION_SEEDS[model_key], device)
    tokenizer.padding_side = "left"
    projected_batches = []
    norm_batches = []
    metadata = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        local_positions = [role_positions(tokenizer, row) for row in batch]
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
            padded_positions.append([pad + positions[role] for role in POSITION_ROLES])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model returned no hidden states")
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
        for row, positions, length in zip(batch, local_positions, lengths, strict=True):
            metadata.append({
                **{key: value for key, value in row.items() if key not in {"prompt", "role_char_ends"}},
                "token_length": int(length),
                "role_token_positions": positions,
            })
        del outputs, hidden_states, projected, norms
        if start == 0 or start + len(batch) == len(rows) or (start // batch_size) % 8 == 7:
            log(f"{model_key} {stage} {min(start + len(batch), len(rows))}/{len(rows)}")
    arrays = np.concatenate(projected_batches, axis=0)
    norms = np.concatenate(norm_batches, axis=0)
    return arrays, norms, metadata, {
        "n_layers_with_embedding": int(arrays.shape[1]),
        "d_model": int(info.d_model),
        "projection_dim": PROJECTION_DIM,
        "projection_seed": PROJECTION_SEEDS[model_key],
        "position_roles": list(POSITION_ROLES),
    }


def unit_vectors(vectors: np.ndarray) -> np.ndarray:
    values = vectors.astype(np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def fit_grid(projected: np.ndarray, metadata: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    vectors = unit_vectors(projected)
    truths = np.asarray([row["truth_value"] for row in metadata], dtype=bool)
    mean_true = vectors[truths].mean(axis=0)
    mean_false = vectors[~truths].mean(axis=0)
    directions = mean_true - mean_false
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    thresholds = np.einsum("lrd,lrd->lr", (mean_true + mean_false) / 2, directions)
    return directions.astype(np.float32), thresholds.astype(np.float32)


def binary_observer(vectors: np.ndarray, truths: np.ndarray) -> tuple[np.ndarray, float]:
    values = unit_vectors(vectors)
    mean_true = values[truths].mean(axis=0)
    mean_false = values[~truths].mean(axis=0)
    direction = mean_true - mean_false
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    threshold = float(((mean_true + mean_false) / 2) @ direction)
    return direction.astype(np.float32), threshold


def primary_layer(contract: dict[str, Any], model_key: str, n_layers: int) -> int:
    frozen = contract["primary_windows"][model_key]
    if "layer_with_embedding" in frozen:
        layer = int(frozen["layer_with_embedding"])
    else:
        layer = int(round(float(frozen["normalized_depth"]) * (n_layers - 1)))
    if not 0 <= layer < n_layers:
        raise RuntimeError(f"Frozen primary layer {layer} is invalid for {model_key}/{n_layers}")
    return layer


def pair_ledger(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    by_world_track: dict[tuple[str, str], int] = {}
    by_pair_track: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_world_track[(row["world_case_id"], row["track"])] = index
        by_pair_track[(row["source_pair_id"], row["track"])].append(index)
    surface = []
    for world_case_id in sorted({row["world_case_id"] for row in metadata}):
        surface.append((
            by_world_track[(world_case_id, "identity")],
            by_world_track[(world_case_id, "native_plain_candidate")],
        ))
    relation = []
    for key, indices in sorted(by_pair_track.items()):
        if len(indices) != 2 or {metadata[index]["truth_value"] for index in indices} != {False, True}:
            raise RuntimeError(f"Incomplete relation pair {key}")
        relation.append((indices[0], indices[1]))
    return {"surface": surface, "relation": relation}


def geometry_report(vectors: np.ndarray, metadata: list[dict[str, Any]], pairs: dict[str, Any]) -> dict[str, Any]:
    values = unit_vectors(vectors)

    def mean_distance(selected: list[tuple[int, int]]) -> float:
        if not selected:
            return 0.0
        left = values[[pair[0] for pair in selected]]
        right = values[[pair[1] for pair in selected]]
        return float(np.linalg.norm(left - right, axis=-1).mean())

    surface = mean_distance(pairs["surface"])
    relation = mean_distance(pairs["relation"])
    by_family = {}
    for family in sorted({row["family"] for row in metadata}):
        surface_pairs = [pair for pair in pairs["surface"] if metadata[pair[0]]["family"] == family]
        relation_pairs = [pair for pair in pairs["relation"] if metadata[pair[0]]["family"] == family]
        ds = mean_distance(surface_pairs)
        dr = mean_distance(relation_pairs)
        by_family[family] = {"d_native_surface": ds, "d_relation_world_pair": dr, "q_native": (dr - ds) / (dr + ds + 1e-8)}
    return {
        "d_native_surface": surface,
        "d_relation_world_pair": relation,
        "q_native": (relation - surface) / (relation + surface + 1e-8),
        "by_family": by_family,
    }


def prediction_report(scores: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    truths = np.asarray([row["truth_value"] for row in metadata], dtype=bool)
    predictions = scores > 0
    report = {"overall": rate_from_arrays(predictions, truths)}
    for key, values in (
        ("by_family", sorted({row["family"] for row in metadata})),
        ("by_track", list(TRACKS)),
        ("by_claim_polarity", ["positive", "negative"]),
        ("by_length", ["short", "medium", "long"]),
        ("by_fact_order", ["target_first", "distractor_first", "interleaved"]),
    ):
        field = {
            "by_family": "family",
            "by_track": "track",
            "by_claim_polarity": "claim_polarity",
            "by_length": "length_control",
            "by_fact_order": "fact_order",
        }[key]
        report[key] = {}
        for value in values:
            indices = np.asarray([index for index, row in enumerate(metadata) if row[field] == value], dtype=int)
            report[key][value] = rate_from_arrays(predictions[indices], truths[indices])
    report["score_summary"] = {
        "mean_true": float(scores[truths].mean()),
        "mean_false": float(scores[~truths].mean()),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
    }
    return report


def report_pass(prediction: dict[str, Any], geometry: dict[str, Any], families: list[str], gate: dict[str, Any]) -> bool:
    return (
        geometry["q_native"] > 0
        and all(geometry["by_family"][family]["q_native"] > 0 for family in families)
        and all(prediction["by_family"][family]["lcb95"] >= gate["per_unseen_family_prediction_lcb95_min"] for family in families)
        and prediction["overall"]["lcb95"] >= gate["overall_unseen_prediction_lcb95_min"]
    )


def evaluate_grid(
    test_projected: np.ndarray,
    metadata: list[dict[str, Any]],
    directions: np.ndarray,
    thresholds: np.ndarray,
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    vectors = unit_vectors(test_projected)
    pairs = pair_ledger(metadata)
    families = sorted({row["family"] for row in metadata})
    reports = []
    for layer in range(vectors.shape[1]):
        for role_index, role in enumerate(POSITION_ROLES):
            local = vectors[:, layer, role_index]
            scores = local @ directions[layer, role_index] - thresholds[layer, role_index]
            prediction = prediction_report(scores, metadata)
            geometry = geometry_report(local, metadata, pairs)
            reports.append({
                "layer_with_embedding": layer,
                "normalized_depth": layer / max(1, vectors.shape[1] - 1),
                "position_role": role,
                "prediction": prediction,
                "geometry": geometry,
                "cross_family_gate_pass": report_pass(prediction, geometry, families, gate),
            })
    return reports


def trajectory_summary(reports: list[dict[str, Any]], n_layers: int, stable_count: int) -> dict[str, Any]:
    out = {}
    for role in POSITION_ROLES:
        rows = sorted((row for row in reports if row["position_role"] == role), key=lambda row: row["layer_with_embedding"])
        flags = [bool(row["cross_family_gate_pass"]) for row in rows]
        formation = None
        for index in range(0, len(flags) - stable_count + 1):
            if all(flags[index:index + stable_count]):
                formation = rows[index]["layer_with_embedding"]
                break
        events = []
        seen = False
        previous = False
        for row, flag in zip(rows, flags, strict=True):
            event = "below_gate"
            if flag and not seen:
                event = "first_readable_layer"
                seen = True
            elif flag and previous:
                event = "sustained_readability"
            elif flag and seen and not previous:
                event = "reformed_readability"
            elif not flag and previous:
                event = "readability_loss"
            events.append({
                "layer_with_embedding": row["layer_with_embedding"],
                "event": event,
                "gate_pass": flag,
                "accuracy": row["prediction"]["overall"]["rate"],
                "lcb95": row["prediction"]["overall"]["lcb95"],
                "q_native": row["geometry"]["q_native"],
            })
            previous = flag
        terminal_amplification = False
        if formation is not None:
            formation_rate = rows[formation]["prediction"]["overall"]["rate"]
            terminal_amplification = rows[-1]["prediction"]["overall"]["rate"] >= formation_rate + 0.05
        out[role] = {
            "stable_formation_layer": formation,
            "stable_formation_normalized_depth": formation / max(1, n_layers - 1) if formation is not None else None,
            "terminal_amplification": terminal_amplification,
            "events": events,
        }
    return out


def random_feature_parameters(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(PROJECTION_DIM, PROJECTION_DIM))
    matrix /= math.sqrt(PROJECTION_DIM)
    bias = rng.uniform(-0.5, 0.5, size=(PROJECTION_DIM,)).astype(np.float32)
    return matrix, bias


def rbf_scores(train: np.ndarray, train_truths: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, float]:
    reference = train[:: max(1, len(train) // 128)][:128]
    distances = np.sum((reference[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
    positive = distances[distances > 1e-8]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    sigma2 = max(sigma2, 1e-6)
    true_train = train[train_truths]
    false_train = train[~train_truths]
    margins = []
    for start in range(0, len(test), 128):
        batch = test[start:start + 128]
        true_distance = np.sum((batch[:, None, :] - true_train[None, :, :]) ** 2, axis=-1)
        false_distance = np.sum((batch[:, None, :] - false_train[None, :, :]) ** 2, axis=-1)
        true_similarity = np.exp(-true_distance / (2 * sigma2)).mean(axis=1)
        false_similarity = np.exp(-false_distance / (2 * sigma2)).mean(axis=1)
        margins.append(true_similarity - false_similarity)
    return np.concatenate(margins).astype(np.float32), sigma2


def legacy_observer(model_key: str, layer: int) -> tuple[np.ndarray, float] | None:
    if model_key not in {"qwen3", "glm4"}:
        return None
    arrays = np.load(LEGACY_DIR / f"phase490_{model_key}_projected_states.npz")
    metadata = load_jsonl(LEGACY_DIR / f"phase490_{model_key}_metadata.jsonl")
    projected = arrays["projected"]
    if layer >= projected.shape[1]:
        raise RuntimeError("Legacy observer layer is outside the saved Phase490 grid")
    truths = np.asarray([row["truth_value"] for row in metadata], dtype=bool)
    # Phase490 prompt_end was role index 2.
    return binary_observer(projected[:, layer, 2], truths)


def verify_authorization(model_key: str) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    contract = load_json(CONTRACT_PATH)
    authorization = load_json(AUTH_PATH)
    if model_key not in authorization["physical_models_in_required_execution_order"]:
        raise RuntimeError(f"{model_key} is not behavior-authorized for Phase497-498")
    model_auth = authorization["models"][model_key]
    families = list(model_auth["passed_unseen_families"])
    if not families:
        raise RuntimeError(f"{model_key} has no behavior-qualified unseen family")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase494 protocol source changed after freeze")
    if sha256_file(FIT_PATH) != contract["split_files"]["formation_fit"]["sha256"]:
        raise RuntimeError("Phase494 formation-fit hash drift")
    return contract, authorization, families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    contract, authorization, prediction_families = verify_authorization(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase497-498 requires CUDA")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)

        # Phase497: read, collect, fit, and persist the training-family observer first.
        fit_rows = flatten(load_jsonl(FIT_PATH), "formation_fit", set(TRAIN_FAMILIES))
        fit_projected, fit_norms, fit_metadata, run_info = collect(
            model, tokenizer, device, fit_rows, args.model, args.batch_size, "formation-fit"
        )
        directions, thresholds = fit_grid(fit_projected, fit_metadata)
        primary = primary_layer(contract, args.model, fit_projected.shape[1])
        primary_role_index = POSITION_ROLES.index("prompt_end")
        fit_truths = np.asarray([row["truth_value"] for row in fit_metadata], dtype=bool)
        fit_primary = unit_vectors(fit_projected[:, primary, primary_role_index])
        primary_fit_scores = fit_primary @ directions[primary, primary_role_index] - thresholds[primary, primary_role_index]
        primary_fit_prediction = prediction_report(primary_fit_scores, fit_metadata)
        rf_matrix, rf_bias = random_feature_parameters(PROJECTION_SEEDS[args.model] + 4947)
        fit_rf = np.tanh(fit_primary @ rf_matrix + rf_bias)
        rf_direction, rf_threshold = binary_observer(fit_rf, fit_truths)
        legacy = legacy_observer(args.model, primary)

        fit_npz_path = OUT_DIR / f"phase497_{args.model}_fit_and_observers.npz"
        fit_metadata_path = OUT_DIR / f"phase497_{args.model}_fit_metadata.jsonl"
        np.savez_compressed(
            fit_npz_path,
            projected=fit_projected,
            hidden_norms=fit_norms,
            linear_directions=directions,
            linear_thresholds=thresholds,
            random_feature_matrix=rf_matrix,
            random_feature_bias=rf_bias,
            random_feature_direction=rf_direction,
            random_feature_threshold=np.asarray(rf_threshold, dtype=np.float32),
            legacy_direction=legacy[0] if legacy else np.asarray([], dtype=np.float32),
            legacy_threshold=np.asarray(legacy[1] if legacy else np.nan, dtype=np.float32),
        )
        write_jsonl(fit_metadata_path, fit_metadata)
        freeze = {
            "schema_version": "phase497_train_family_observer_freeze.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "observer_frozen_before_unseen_split_read",
            "model": args.model,
            "fit_families": list(TRAIN_FAMILIES),
            "fit_rows": len(fit_metadata),
            "fit_input_sha256": sha256_file(FIT_PATH),
            "observer_npz": str(fit_npz_path.relative_to(ROOT)),
            "observer_npz_sha256_before_unseen_read": sha256_file(fit_npz_path),
            "primary_window": {
                "layer_with_embedding": primary,
                "normalized_depth": primary / max(1, fit_projected.shape[1] - 1),
                "position_role": "prompt_end",
                "source": contract["primary_windows"][args.model]["source"],
            },
            "primary_fit_prediction": primary_fit_prediction,
            "legacy_phase492_observer_available": legacy is not None,
            "unseen_split_read_at_freeze_time": False,
            "sealed_split_read": False,
        }
        freeze_path = OUT_DIR / f"phase497_{args.model}_observer_freeze.json"
        freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"{args.model} observer freeze persisted before unseen read")

        # Phase498: verify and then perform the first content read of the open
        # unseen-family split only after the observer artifact is immutable.
        if sha256_file(PREDICTION_PATH) != contract["split_files"]["family_prediction"]["sha256"]:
            raise RuntimeError("Phase494 family-prediction hash drift")
        prediction_rows = flatten(load_jsonl(PREDICTION_PATH), "family_prediction", set(prediction_families))
        test_projected, test_norms, test_metadata, test_run_info = collect(
            model, tokenizer, device, prediction_rows, args.model, args.batch_size, "unseen-prediction"
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        torch.cuda.empty_cache()

    runtime = time.monotonic() - started
    gate = contract["physical_gate"]
    reports = evaluate_grid(test_projected, test_metadata, directions, thresholds, gate)
    trajectory = trajectory_summary(reports, test_projected.shape[1], int(gate["stable_layer_count"]))
    primary_report = next(
        row for row in reports
        if row["layer_with_embedding"] == primary and row["position_role"] == "prompt_end"
    )
    test_primary = unit_vectors(test_projected[:, primary, primary_role_index])
    primary_scores = test_primary @ directions[primary, primary_role_index] - thresholds[primary, primary_role_index]
    test_truths = np.asarray([row["truth_value"] for row in test_metadata], dtype=bool)

    test_rf = unit_vectors(np.tanh(test_primary @ rf_matrix + rf_bias))
    random_feature_scores = test_rf @ rf_direction - rf_threshold
    random_feature_report = prediction_report(random_feature_scores, test_metadata)
    rbf_margin, rbf_sigma2 = rbf_scores(fit_primary, fit_truths, test_primary)
    rbf_report = prediction_report(rbf_margin, test_metadata)

    legacy_report = None
    legacy_scores = np.full(len(test_metadata), np.nan, dtype=np.float32)
    if legacy is not None:
        legacy_scores = test_primary @ legacy[0] - legacy[1]
        legacy_report = prediction_report(legacy_scores, test_metadata)

    all_unseen_behavior_pass = authorization["models"][args.model]["all_unseen_families_behavior_pass"]
    all_four_collected = set(prediction_families) == set(UNSEEN_FAMILIES)
    primary_cross_family_pass = bool(all_unseen_behavior_pass and all_four_collected and primary_report["cross_family_gate_pass"])
    formation_layer = trajectory["prompt_end"]["stable_formation_layer"]
    formation_pass = formation_layer is not None

    prediction_npz_path = OUT_DIR / f"phase498_{args.model}_unseen_projected_states.npz"
    prediction_metadata_path = OUT_DIR / f"phase498_{args.model}_unseen_metadata.jsonl"
    score_rows_path = OUT_DIR / f"phase498_{args.model}_functional_scores.jsonl"
    summary_path = OUT_DIR / f"phase498_{args.model}_summary.json"
    np.savez_compressed(prediction_npz_path, projected=test_projected, hidden_norms=test_norms)
    write_jsonl(prediction_metadata_path, test_metadata)
    score_rows = []
    for index, row in enumerate(test_metadata):
        score_rows.append({
            "model": args.model,
            "sample_id": row["sample_id"],
            "world_case_id": row["world_case_id"],
            "source_pair_id": row["source_pair_id"],
            "family": row["family"],
            "track": row["track"],
            "truth_value": row["truth_value"],
            "claim_polarity": row["claim_polarity"],
            "primary_linear_score": float(primary_scores[index]),
            "legacy_phase492_score": float(legacy_scores[index]) if np.isfinite(legacy_scores[index]) else None,
            "fixed_random_feature_score": float(random_feature_scores[index]),
            "local_rbf_margin": float(rbf_margin[index]),
        })
    write_jsonl(score_rows_path, score_rows)

    summary = {
        "schema_version": "phase498_cross_family_trajectory.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "cross_family_pass" if primary_cross_family_pass else "cross_family_fail_or_partial",
        "model": args.model,
        "cuda_used": True,
        "model_weights_loaded": True,
        "runtime_seconds": runtime,
        "fit_rows": len(fit_metadata),
        "prediction_rows": len(test_metadata),
        "behavior_qualified_prediction_families": prediction_families,
        "all_four_unseen_families_collected": all_four_collected,
        "sealed_split_read": False,
        "run_info": run_info,
        "prediction_run_info": test_run_info,
        "primary_window": freeze["primary_window"],
        "primary_train_family_observer": primary_report,
        "legacy_phase492_observer": legacy_report,
        "nonlinear_controls": {
            "fixed_tanh_random_features": random_feature_report,
            "local_rbf_class_kernel": {"bandwidth_squared": rbf_sigma2, "prediction": rbf_report},
        },
        "trajectory_by_position_role": trajectory,
        "grid_reports": reports,
        "gates": {
            "all_unseen_behavior_pass": all_unseen_behavior_pass,
            "primary_cross_family_prediction_pass": primary_cross_family_pass,
            "stable_prompt_end_formation_found": formation_pass,
            "compute_transport_measured": False,
            "path_causal_eligible": False,
        },
        "evidence_boundary": {
            "role_state_sequence_is_observational_precursor_only": True,
            "source_to_terminal_write_not_measured": True,
            "compute_edge_claim": False,
            "causal_claim": False,
            "single_neuron_claim": False,
            "strict_mechanism_closure": False,
        },
        "artifacts": {
            "fit_freeze": str(freeze_path.relative_to(ROOT)),
            "fit_observer_sha256_after_prediction": sha256_file(fit_npz_path),
            "prediction_npz": str(prediction_npz_path.relative_to(ROOT)),
            "functional_scores": str(score_rows_path.relative_to(ROOT)),
        },
    }
    if summary["artifacts"]["fit_observer_sha256_after_prediction"] != freeze["observer_npz_sha256_before_unseen_read"]:
        raise RuntimeError("Frozen observer changed after unseen-family read")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(freeze_path)
    print(summary_path)


if __name__ == "__main__":
    main()
