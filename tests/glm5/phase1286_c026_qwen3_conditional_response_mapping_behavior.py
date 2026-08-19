#!/usr/bin/env python3
"""Phase1286: Qwen3 behavior and conditional response-map adjudication for C026."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
from phase1285_c026_conditional_response_mapping_contract import (  # noqa: E402
    PANELS, ROLE_ORDER, ROLE_PERMUTATION_NULLS, SURFACE_ORDER,
)


PHASE = 1286
CAMPAIGN = "C026"
CONTRACT_ID = "EXP-C026-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1286_c026_qwen3_conditional_response_mapping_behavior_audit.py"
INPUT = ROOT / "tests/glm5/result/phase1285_c026_conditional_response_mapping_contract"
INPUT_PROTOCOL = INPUT / "protocol/preregistration.json"
INPUT_MATERIAL = INPUT / "material/frozen_binary_status_worlds.jsonl"
INPUT_FINAL = INPUT / "analysis/final.json"
INPUT_AUDIT = INPUT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1286_c026_qwen3_conditional_response_mapping_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/response_scores.jsonl"
GENERATIONS = OUT / "raw/confirmation_generations.jsonl"
SELECTION_DECISION = OUT / "analysis/frozen_selection_decision.json"
RUN_SUMMARY = OUT / "analysis/run_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

SCORE_BATCH_SIZE = 32
GENERATION_BATCH_SIZE = 16
MAX_NEW_TOKENS = 8
FAMILY_ORDER = ("H0_constant", "H1_identity", "H2_diagonal_affine", "H3_full_affine")
SOURCE_SURFACE = SURFACE_ORDER[0]
TARGET_SURFACES = SURFACE_ORDER[1:]


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("Phase1286 protocol already exists")
    parent = read_json(INPUT_PROTOCOL)
    parent_final = read_json(INPUT_FINAL)
    parent_audit = read_json(INPUT_AUDIT)
    if parent_final.get("authorization") != "phase1286_qwen3_conditional_response_mapping_behavior" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1285 authorization missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1286.c026.qwen.behavior_mapping.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "row_count": parent["counts"]["worlds"],
        "context_count": parent["counts"]["contexts"],
        "scored_sequence_count": parent["counts"]["scored_sequences"],
        "generation_count": parent["counts"]["confirmation_generations"],
        "surface_order": list(SURFACE_ORDER),
        "role_order": list(ROLE_ORDER),
        "panels": list(PANELS),
        "source_surface": SOURCE_SURFACE,
        "target_surfaces": list(TARGET_SURFACES),
        "family_order": list(FAMILY_ORDER),
        "map_fit": parent["map_fit"],
        "zero_models": parent["zero_models"],
        "thresholds": parent["thresholds"],
        "score_batch_size": SCORE_BATCH_SIZE,
        "generation": {
            "partition": "confirmation",
            "panels": ["consistency", "reversal"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "batch_size": GENERATION_BATCH_SIZE,
            "parser": "exact frozen expected_label versus opposite_label word boundary",
        },
        "dependencies": {
            "phase1285_protocol": file_sha256(INPUT_PROTOCOL),
            "phase1285_material": file_sha256(INPUT_MATERIAL),
            "phase1285_final": file_sha256(INPUT_FINAL),
            "phase1285_audit": file_sha256(INPUT_AUDIT),
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "formal_run_budget": 1,
        "unblinding_order": [
            "score all frozen sequences and save raw scores",
            "generate all frozen confirmation continuations and save raw text",
            "fit discovery maps and evaluate only selection",
            "write immutable selected-family decision",
            "evaluate behavior, generation, confirmation mapping, and null specificity",
        ],
        "hard_stops": [
            "The selected mapping family is written before confirmation mapping metrics are computed.",
            "Mean log probability per continuation token is primary; total log probability is a frozen sensitivity account.",
            "No row, surface, candidate, null, parser term, threshold, or hypothesis may change after this protocol.",
            "Any behavior, generation, mapping, or specificity failure stops C026 before hidden hooks and before other models.",
            "Formal Qwen3 execution is one-shot and the complete marker forbids reruns.",
        ],
    }
    protocol = {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    print(canonical_json({"status": "preregistered", "protocol_digest": protocol["protocol_digest"]}))


def score_examples(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        for surface in SURFACE_ORDER:
            for panel in PANELS:
                context = row["contexts"][surface][panel]
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                for role in ROLE_ORDER:
                    continuation = row["candidate_continuations"][role]
                    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
                    if full_ids[:len(context_ids)] != context_ids:
                        raise RuntimeError("candidate prefix drift")
                    continuation_length = len(full_ids) - len(context_ids)
                    if continuation_length <= 0:
                        raise RuntimeError("empty candidate continuation")
                    examples.append({
                        "row_id": row["row_id"], "partition": row["partition"], "axis": row["axis"],
                        "surface": surface, "panel": panel, "role": role,
                        "full_ids": full_ids, "context_length": len(context_ids),
                        "continuation_length": continuation_length,
                    })
    scored: dict[tuple[str, str, str], dict[str, Any]] = {}
    for start in range(0, len(examples), SCORE_BATCH_SIZE):
        batch = examples[start:start + SCORE_BATCH_SIZE]
        maximum = max(len(value["full_ids"]) for value in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(batch), maximum), int(pad_id), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        offsets = []
        for index, value in enumerate(batch):
            offset = maximum - len(value["full_ids"])
            offsets.append(offset)
            ids[index, offset:] = torch.tensor(value["full_ids"], dtype=torch.long, device=device)
            mask[index, offset:] = 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        for index, value in enumerate(batch):
            offset = offsets[index]
            first = offset + value["context_length"]
            last = offset + len(value["full_ids"])
            positions = torch.arange(first - 1, last - 1, device=device)
            token_ids = ids[index, first:last]
            total = float(log_probs[index, positions, token_ids].sum().item())
            key = (value["row_id"], value["surface"], value["panel"])
            entry = scored.setdefault(key, {
                "row_id": value["row_id"], "partition": value["partition"], "axis": value["axis"],
                "surface": value["surface"], "panel": value["panel"],
                "total_log_prob": {}, "mean_log_prob": {}, "continuation_length": {},
            })
            entry["total_log_prob"][value["role"]] = total
            entry["mean_log_prob"][value["role"]] = total / value["continuation_length"]
            entry["continuation_length"][value["role"]] = value["continuation_length"]
        if (start // SCORE_BATCH_SIZE + 1) % 100 == 0:
            print(canonical_json({"scored_sequences": min(start + SCORE_BATCH_SIZE, len(examples)), "total": len(examples)}), flush=True)
    output = []
    for value in scored.values():
        numbers = [value[account][role] for account in ("total_log_prob", "mean_log_prob") for role in ROLE_ORDER]
        value["finite"] = bool(np.isfinite(numbers).all())
        output.append(value)
    return sorted(output, key=lambda value: (value["row_id"], value["surface"], value["panel"]))


def response(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    row_id: str,
    surface: str,
    right: str,
    left: str,
    account: str,
) -> np.ndarray:
    right_values = np.asarray([by_key[(row_id, surface, right)][account][role] for role in ROLE_ORDER], dtype=np.float64)
    left_values = np.asarray([by_key[(row_id, surface, left)][account][role] for role in ROLE_ORDER], dtype=np.float64)
    value = right_values - left_values
    return value - value.mean()


def build_signatures(raw: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    by_key = {(value["row_id"], value["surface"], value["panel"]): value for value in raw}
    output: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for account in ("mean_log_prob", "total_log_prob"):
        signatures: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            for surface in SURFACE_ORDER:
                active = response(by_key, row["row_id"], surface, "reversal", "consistency", account)
                lexical = response(by_key, row["row_id"], surface, "lexical_reversal", "lexical_consistency", account)
                role = response(by_key, row["row_id"], surface, "role_reversal", "role_consistency", account)
                target_scale = max(float(np.mean(np.abs(active[:4]))), 1.0e-12)
                signatures[(row["row_id"], surface)] = {
                    "active": active,
                    "lexical": lexical,
                    "role": role,
                    "effect": float(np.mean(active[2:4]) - np.mean(active[0:2])),
                    "active_norm": float(np.linalg.norm(active)),
                    "lexical_norm": float(np.linalg.norm(lexical)),
                    "role_norm": float(np.linalg.norm(role)),
                    "control_leakage": float(np.mean(np.abs(active[4:6])) / target_scale),
                }
        output[account] = signatures
    return output


def behavior_summary(
    raw: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    signatures: dict[str, dict[tuple[str, str], dict[str, Any]]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    primary = signatures["mean_log_prob"]
    secondary = signatures["total_log_prob"]
    surface_cells: dict[str, Any] = {}
    axis_cells: dict[str, Any] = {}
    for partition in ("discovery", "selection", "confirmation"):
        partition_rows = [row for row in rows if row["partition"] == partition]
        axes = sorted({row["axis"] for row in partition_rows})
        for surface in SURFACE_ORDER:
            values = [primary[(row["row_id"], surface)] for row in partition_rows]
            total_values = [secondary[(row["row_id"], surface)] for row in partition_rows]
            median_active = float(np.median([value["active_norm"] for value in values]))
            median_lexical = float(np.median([value["lexical_norm"] for value in values]))
            median_role = float(np.median([value["role_norm"] for value in values]))
            cell = {
                "n_worlds": len(values),
                "positive_fraction": float(np.mean([value["effect"] > 0 for value in values])),
                "median_effect": float(np.median([value["effect"] for value in values])),
                "median_active_norm": median_active,
                "median_lexical_norm": median_lexical,
                "median_role_norm": median_role,
                "lexical_norm_ratio": median_lexical / max(median_active, 1.0e-12),
                "role_norm_ratio": median_role / max(median_active, 1.0e-12),
                "median_active_minus_lexical_norm": float(np.median([value["active_norm"] - value["lexical_norm"] for value in values])),
                "median_active_minus_role_norm": float(np.median([value["active_norm"] - value["role_norm"] for value in values])),
                "median_control_leakage": float(np.median([value["control_leakage"] for value in values])),
                "total_mean_effect_sign_agreement": float(np.mean([
                    (value["effect"] > 0) == (total_value["effect"] > 0)
                    for value, total_value in zip(values, total_values)
                ])),
            }
            surface_cells[f"{partition}.{surface}"] = cell
            for axis_name in axes:
                axis_rows = [row for row in partition_rows if row["axis"] == axis_name]
                axis_values = [primary[(row["row_id"], surface)] for row in axis_rows]
                axis_cells[f"{partition}.{surface}.{axis_name}"] = {
                    "n_worlds": len(axis_values),
                    "positive_fraction": float(np.mean([value["effect"] > 0 for value in axis_values])),
                    "median_effect": float(np.median([value["effect"] for value in axis_values])),
                    "median_active_norm": float(np.median([value["active_norm"] for value in axis_values])),
                }
    axis_pass_counts: dict[str, int] = {}
    for partition in ("discovery", "selection", "confirmation"):
        axes = sorted({row["axis"] for row in rows if row["partition"] == partition})
        axis_pass_counts[partition] = sum(
            min(axis_cells[f"{partition}.{surface}.{axis_name}"]["positive_fraction"] for surface in SURFACE_ORDER)
            >= thresholds["axis_positive_fraction_min"]
            for axis_name in axes
        )
    gates = {
        "finite": float(np.mean([value["finite"] for value in raw])) >= thresholds["finite_fraction_min"],
        "positive_fraction": min(value["positive_fraction"] for value in surface_cells.values()) >= thresholds["partition_surface_positive_fraction_min"],
        "median_effect": min(value["median_effect"] for value in surface_cells.values()) >= thresholds["partition_surface_median_effect_min"],
        "active_norm": min(value["median_active_norm"] for value in surface_cells.values()) >= thresholds["partition_surface_median_active_norm_min"],
        "axis_coverage": min(axis_pass_counts.values()) >= thresholds["axis_pass_count_per_partition_min"],
        "lexical_null": max(value["lexical_norm_ratio"] for value in surface_cells.values()) <= thresholds["lexical_null_norm_ratio_max"],
        "role_null": max(value["role_norm_ratio"] for value in surface_cells.values()) <= thresholds["role_null_norm_ratio_max"],
        "control_leakage": max(value["median_control_leakage"] for value in surface_cells.values()) <= thresholds["control_leakage_ratio_max"],
        "length_sensitivity": min(value["total_mean_effect_sign_agreement"] for value in surface_cells.values()) >= thresholds["total_mean_effect_sign_agreement_min"],
    }
    return {
        "finite_fraction": float(np.mean([value["finite"] for value in raw])),
        "surface_cells": surface_cells,
        "axis_cells": axis_cells,
        "axis_pass_counts": axis_pass_counts,
        "gate_extrema": {
            "positive_fraction_min": min(value["positive_fraction"] for value in surface_cells.values()),
            "median_effect_min": min(value["median_effect"] for value in surface_cells.values()),
            "active_norm_min": min(value["median_active_norm"] for value in surface_cells.values()),
            "lexical_ratio_max": max(value["lexical_norm_ratio"] for value in surface_cells.values()),
            "role_ratio_max": max(value["role_norm_ratio"] for value in surface_cells.values()),
            "control_leakage_max": max(value["median_control_leakage"] for value in surface_cells.values()),
            "length_agreement_min": min(value["total_mean_effect_sign_agreement"] for value in surface_cells.values()),
            "axis_pass_count_min": min(axis_pass_counts.values()),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def generate_confirmation(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row["partition"] != "confirmation":
            continue
        for surface in SURFACE_ORDER:
            for panel in ("consistency", "reversal"):
                examples.append({
                    "row_id": row["row_id"], "axis": row["axis"], "surface": surface, "panel": panel,
                    "context": row["contexts"][surface][panel],
                    "expected_label": row["expected_label"], "opposite_label": row["opposite_label"],
                })
    output = []
    for start in range(0, len(examples), GENERATION_BATCH_SIZE):
        batch = examples[start:start + GENERATION_BATCH_SIZE]
        encoded = [tokenizer.encode(value["context"], add_special_tokens=False) for value in batch]
        maximum = max(map(len, encoded))
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(batch), maximum), int(pad_id), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        for index, value in enumerate(encoded):
            ids[index, -len(value):] = torch.tensor(value, dtype=torch.long, device=device)
            mask[index, -len(value):] = 1
        generated = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, use_cache=True, pad_token_id=int(pad_id), eos_token_id=tokenizer.eos_token_id,
        )[:, maximum:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for value, text in zip(batch, texts):
            lower = text.lower()
            found_expected = bool(re.search(rf"\b{re.escape(value['expected_label'].lower())}\b", lower))
            found_opposite = bool(re.search(rf"\b{re.escape(value['opposite_label'].lower())}\b", lower))
            parsed = found_expected ^ found_opposite
            prediction = "expected" if found_expected and not found_opposite else "opposite" if found_opposite and not found_expected else None
            gold = "expected" if value["panel"] == "consistency" else "opposite"
            output.append({
                **value, "generation": text, "matched_expected": found_expected, "matched_opposite": found_opposite,
                "parsed": parsed, "prediction": prediction, "gold": gold, "correct": bool(parsed and prediction == gold),
            })
    return output


def generation_summary(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    cells = {}
    for surface in SURFACE_ORDER:
        for panel in ("consistency", "reversal"):
            subset = [value for value in rows if value["surface"] == surface and value["panel"] == panel]
            parsed = [value for value in subset if value["parsed"]]
            cells[f"{surface}.{panel}"] = {
                "n": len(subset),
                "coverage": float(np.mean([value["parsed"] for value in subset])),
                "accuracy_given_parsed": float(np.mean([value["correct"] for value in parsed])) if parsed else 0.0,
            }
    gates = {
        "coverage": min(value["coverage"] for value in cells.values()) >= thresholds["generation_coverage_min"],
        "accuracy": min(value["accuracy_given_parsed"] for value in cells.values()) >= thresholds["generation_accuracy_min"],
    }
    return {
        "cells": cells,
        "coverage_min": min(value["coverage"] for value in cells.values()),
        "accuracy_min": min(value["accuracy_given_parsed"] for value in cells.values()),
        "gates": gates,
        "passed": all(gates.values()),
    }


def matrix_for(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    partition: str,
    surface: str,
    kind: str = "active",
) -> np.ndarray:
    selected = sorted((row for row in rows if row["partition"] == partition), key=lambda row: row["row_id"])
    return np.stack([signatures[(row["row_id"], surface)][kind] for row in selected], axis=0)


def fit_family(family: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if family == "H0_constant":
        return {"family": family, "mean": y.mean(axis=0).tolist()}
    if family == "H1_identity":
        return {"family": family}
    if family == "H2_diagonal_affine":
        gains, offsets = [], []
        for column in range(y.shape[1]):
            design = np.column_stack([x[:, column], np.ones(len(x))])
            regularizer = np.diag([1.0e-3, 0.0])
            weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y[:, column])
            gains.append(float(weights[0]))
            offsets.append(float(weights[1]))
        return {"family": family, "gains": gains, "offsets": offsets}
    if family == "H3_full_affine":
        design = np.column_stack([x, np.ones(len(x))])
        regularizer = np.diag([1.0e-2] * x.shape[1] + [0.0])
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return {"family": family, "matrix": weights[:-1].tolist(), "offset": weights[-1].tolist()}
    raise ValueError(f"unknown family {family}")


def predict_family(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    family = model["family"]
    if family == "H0_constant":
        return np.repeat(np.asarray(model["mean"], dtype=np.float64)[None, :], len(x), axis=0)
    if family == "H1_identity":
        return x.copy()
    if family == "H2_diagonal_affine":
        return x * np.asarray(model["gains"], dtype=np.float64) + np.asarray(model["offsets"], dtype=np.float64)
    if family == "H3_full_affine":
        return x @ np.asarray(model["matrix"], dtype=np.float64) + np.asarray(model["offset"], dtype=np.float64)
    raise ValueError(f"unknown family {family}")


def prediction_metrics(y: np.ndarray, prediction: np.ndarray, h0_prediction: np.ndarray | None = None) -> dict[str, Any]:
    squared_error = float(np.square(y - prediction).sum())
    energy = max(float(np.square(y).sum()), 1.0e-12)
    cosines = [cosine(left, right) for left, right in zip(y, prediction)]
    result = {
        "n": len(y),
        "squared_error": squared_error,
        "energy": energy,
        "nrmse": float(np.sqrt(squared_error / energy)),
        "median_cosine": float(np.median(cosines)),
        "positive_fraction": float(np.mean(np.asarray(cosines) > 0.0)),
        "minimum_cosine": float(np.min(cosines)),
    }
    if h0_prediction is not None:
        h0_error = max(float(np.square(y - h0_prediction).sum()), 1.0e-12)
        result["gain_over_h0"] = 1.0 - squared_error / h0_error
        result["h0_nrmse"] = float(np.sqrt(h0_error / energy))
        result["nrmse_improvement_over_h0"] = result["h0_nrmse"] - result["nrmse"]
    return result


def fit_and_select(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    tolerance: float,
    account: str,
) -> dict[str, Any]:
    fits: dict[str, dict[str, Any]] = {}
    selection: dict[str, dict[str, Any]] = {}
    pooled: dict[str, float] = {}
    for family in FAMILY_ORDER:
        fits[family] = {}
        selection[family] = {}
        total_error = 0.0
        total_energy = 0.0
        for target in TARGET_SURFACES:
            x_discovery = matrix_for(rows, signatures, "discovery", SOURCE_SURFACE)
            y_discovery = matrix_for(rows, signatures, "discovery", target)
            model = fit_family(family, x_discovery, y_discovery)
            fits[family][target] = model
            x_selection = matrix_for(rows, signatures, "selection", SOURCE_SURFACE)
            y_selection = matrix_for(rows, signatures, "selection", target)
            h0 = predict_family(fit_family("H0_constant", x_discovery, y_discovery), x_selection)
            metrics = prediction_metrics(y_selection, predict_family(model, x_selection), h0)
            selection[family][target] = metrics
            total_error += metrics["squared_error"]
            total_energy += metrics["energy"]
        pooled[family] = float(np.sqrt(total_error / max(total_energy, 1.0e-12)))
    best_value = min(pooled.values())
    eligible = [family for family in FAMILY_ORDER if pooled[family] <= best_value + tolerance]
    selected_family = eligible[0]
    return {
        "account": account,
        "family_order": list(FAMILY_ORDER),
        "discovery_fits": fits,
        "selection_metrics": selection,
        "pooled_selection_nrmse": pooled,
        "minimum_selection_nrmse": best_value,
        "simplicity_tolerance": tolerance,
        "eligible_within_tolerance": eligible,
        "selected_family": selected_family,
    }


def evaluate_selected(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    selected_family: str,
    thresholds: dict[str, float],
    account: str,
) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    refits: dict[str, Any] = {}
    for target in TARGET_SURFACES:
        x_train = np.concatenate([
            matrix_for(rows, signatures, "discovery", SOURCE_SURFACE),
            matrix_for(rows, signatures, "selection", SOURCE_SURFACE),
        ], axis=0)
        y_train = np.concatenate([
            matrix_for(rows, signatures, "discovery", target),
            matrix_for(rows, signatures, "selection", target),
        ], axis=0)
        model = fit_family(selected_family, x_train, y_train)
        refits[target] = model
        x = matrix_for(rows, signatures, "confirmation", SOURCE_SURFACE)
        y = matrix_for(rows, signatures, "confirmation", target)
        h0_model = fit_family("H0_constant", x_train, y_train)
        h0 = predict_family(h0_model, x)
        prediction = predict_family(model, x)
        metrics = prediction_metrics(y, prediction, h0)
        permutation_nrmses = []
        for permutation in ROLE_PERMUTATION_NULLS:
            permuted = prediction[:, list(permutation)]
            permutation_nrmses.append(prediction_metrics(y, permuted)["nrmse"])
        metrics["role_permutation_nrmse"] = permutation_nrmses
        metrics["best_role_permutation_nrmse"] = min(permutation_nrmses)
        metrics["nrmse_improvement_over_best_role_permutation"] = metrics["best_role_permutation_nrmse"] - metrics["nrmse"]

        null_metrics = {}
        for kind in ("lexical", "role"):
            x_null = matrix_for(rows, signatures, "confirmation", SOURCE_SURFACE, kind)
            y_null = matrix_for(rows, signatures, "confirmation", target, kind)
            null_h0_train = np.concatenate([
                matrix_for(rows, signatures, "discovery", target, kind),
                matrix_for(rows, signatures, "selection", target, kind),
            ], axis=0)
            null_h0 = np.repeat(null_h0_train.mean(axis=0, keepdims=True), len(x_null), axis=0)
            null_metrics[kind] = prediction_metrics(y_null, predict_family(model, x_null), null_h0)
        metrics["nulls"] = null_metrics
        metrics["active_minus_lexical_gain"] = metrics["gain_over_h0"] - null_metrics["lexical"]["gain_over_h0"]
        metrics["active_minus_role_gain"] = metrics["gain_over_h0"] - null_metrics["role"]["gain_over_h0"]
        per_target[target] = metrics
    map_gates = {
        "median_cosine": min(value["median_cosine"] for value in per_target.values()) >= thresholds["mapping_confirmation_median_cosine_min"],
        "positive_fraction": min(value["positive_fraction"] for value in per_target.values()) >= thresholds["mapping_confirmation_positive_fraction_min"],
        "nrmse": max(value["nrmse"] for value in per_target.values()) <= thresholds["mapping_confirmation_nrmse_max"],
        "h0_improvement": min(value["nrmse_improvement_over_h0"] for value in per_target.values()) >= thresholds["mapping_nrmse_improvement_over_h0_min"],
        "role_permutation_improvement": min(value["nrmse_improvement_over_best_role_permutation"] for value in per_target.values()) >= thresholds["mapping_nrmse_improvement_over_role_permutation_min"],
    }
    specificity_gates = {
        "active_gain": min(value["gain_over_h0"] for value in per_target.values()) >= thresholds["mapping_active_gain_min"],
        "lexical_advantage": min(value["active_minus_lexical_gain"] for value in per_target.values()) >= thresholds["mapping_active_minus_lexical_gain_min"],
        "role_advantage": min(value["active_minus_role_gain"] for value in per_target.values()) >= thresholds["mapping_active_minus_role_gain_min"],
    }
    return {
        "account": account,
        "selected_family": selected_family,
        "refits": refits,
        "confirmation": per_target,
        "map_gates": map_gates,
        "mapping_passed": all(map_gates.values()),
        "specificity_gates": specificity_gates,
        "specificity_passed": all(specificity_gates.values()),
    }


def run_model() -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal Phase1286 run already complete")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent Phase1286 preaudit missing")
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(INPUT_MATERIAL)
    model = None
    started = time.perf_counter()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if device.type != "cuda" or set(precision["parameter_dtypes"]) != {"float16"} or precision["has_quantized_modules"] or precision["has_bf16_parameters"]:
            raise RuntimeError(f"FP16 qualification failed: {precision}")
        with torch.inference_mode():
            raw = score_examples(model, tokenizer, device, rows)
            generations = generate_confirmation(model, tokenizer, device, rows)
        write_jsonl(RAW, raw)
        write_jsonl(GENERATIONS, generations)

        signatures = build_signatures(raw, rows)
        primary_selection = fit_and_select(
            rows, signatures["mean_log_prob"],
            protocol["thresholds"]["selection_simplicity_tolerance"], "mean_log_prob",
        )
        selection_artifact = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "written_at_utc": utc_now(),
            "confirmation_mapping_metrics_read": False,
            **primary_selection,
        }
        selection_artifact["decision_digest"] = digest(selection_artifact)
        atomic_json(SELECTION_DECISION, selection_artifact)

        behavior = behavior_summary(raw, rows, signatures, protocol["thresholds"])
        generation = generation_summary(generations, protocol["thresholds"])
        primary_mapping = evaluate_selected(
            rows, signatures["mean_log_prob"], primary_selection["selected_family"],
            protocol["thresholds"], "mean_log_prob",
        )
        total_selection = fit_and_select(
            rows, signatures["total_log_prob"],
            protocol["thresholds"]["selection_simplicity_tolerance"], "total_log_prob",
        )
        total_mapping = evaluate_selected(
            rows, signatures["total_log_prob"], total_selection["selected_family"],
            protocol["thresholds"], "total_log_prob",
        )
        all_passed = behavior["passed"] and generation["passed"] and primary_mapping["mapping_passed"] and primary_mapping["specificity_passed"]
        runtime = time.perf_counter() - started
        summary = {
            "behavior": behavior,
            "generation": generation,
            "primary_selection": primary_selection,
            "primary_mapping": primary_mapping,
            "total_log_prob_sensitivity": {"selection": total_selection, "mapping": total_mapping},
            "runtime_seconds": runtime,
            "precision_audit": precision,
            "placement": placement,
        }
        atomic_json(RUN_SUMMARY, summary)
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "verdict": "qwen3_conditional_response_mapping_qualified" if all_passed else "qwen3_conditional_response_mapping_gate_failed",
            "behavior": behavior,
            "generation": generation,
            "mapping": primary_mapping,
            "selection": {
                "selected_family": primary_selection["selected_family"],
                "pooled_selection_nrmse": primary_selection["pooled_selection_nrmse"],
                "decision_digest": selection_artifact["decision_digest"],
            },
            "length_sensitivity": {
                "selected_family": total_selection["selected_family"],
                "mapping_passed": total_mapping["mapping_passed"],
                "specificity_passed": total_mapping["specificity_passed"],
            },
            "runtime_seconds": runtime,
            "precision_audit": precision,
            "authorization": "phase1287_qwen3_hidden_conditional_mapping" if all_passed else "stop_c026_at_qwen_behavior_mapping",
            "scope": (
                "Qwen3-4B FP16; 24 disjoint explicit closed-binary axes; four fixed surfaces; "
                "world-level behavior response mapping with lexical, role, constant-center, identity, and role-permutation controls"
            ),
        }
        atomic_json(FINAL, final)
        atomic_json(COMPLETE, {
            "phase": PHASE,
            "completed_at_utc": utc_now(),
            "raw_sha256": file_sha256(RAW),
            "generation_sha256": file_sha256(GENERATIONS),
            "selection_decision_sha256": file_sha256(SELECTION_DECISION),
            "final_sha256": file_sha256(FINAL),
        })
        print(canonical_json({
            "verdict": final["verdict"],
            "authorization": final["authorization"],
            "behavior_gates": behavior["gates"],
            "generation_gates": generation["gates"],
            "selected_family": primary_selection["selected_family"],
            "mapping_gates": primary_mapping["map_gates"],
            "specificity_gates": primary_mapping["specificity_gates"],
        }))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.action == "preregister" else run_model()
