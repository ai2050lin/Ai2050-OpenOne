#!/usr/bin/env python3
"""Phase1288: one-shot Qwen3 C027 world-residual behavior adjudication."""

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
from phase1287_c027_world_residual_transport_contract import (  # noqa: E402
    FAMILIES, PANELS, PARTITIONS, ROLE_ORDER, ROLE_PERMUTATIONS, SOURCE_FAMILY,
    SURFACE_ORDER, TARGET_FAMILIES, VARIANTS, WRONG_WORLD_OFFSETS,
)


PHASE = 1288
CAMPAIGN = "C027"
CONTRACT_ID = "EXP-C027-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1288_c027_qwen3_world_residual_transport_audit.py"
INPUT = ROOT / "tests/glm5/result/phase1287_c027_world_residual_transport_contract"
INPUT_PROTOCOL = INPUT / "protocol/preregistration.json"
INPUT_MATERIAL = INPUT / "material/frozen_world_residual_material.jsonl"
INPUT_FINAL = INPUT / "analysis/final.json"
INPUT_AUDIT = INPUT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1288_c027_qwen3_world_residual_transport"
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
FAMILY_ORDER = ("H0_zero", "HC_content", "H1_identity", "H2_diagonal", "H3_full")


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
        raise RuntimeError("Phase1288 protocol already exists")
    parent = read_json(INPUT_PROTOCOL)
    parent_final = read_json(INPUT_FINAL)
    parent_audit = read_json(INPUT_AUDIT)
    if parent_final.get("authorization") != "phase1288_qwen3_world_residual_behavior_after_audit":
        raise RuntimeError("Phase1287 final authorization missing")
    if not parent_audit.get("all_checks_passed") or parent_audit.get("authorization") != "phase1288_qwen3_world_residual_behavior":
        raise RuntimeError("Phase1287 pure replay audit missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1288.c027.qwen.world_residual_transport.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "counts": parent["counts"],
        "surface_families": parent["surface_families"],
        "surface_variants": parent["surface_variants"],
        "source_family": parent["source_family"],
        "target_families": parent["target_families"],
        "roles": parent["roles"],
        "panels": parent["panels"],
        "hypotheses": parent["hypotheses"],
        "map_fit": parent["map_fit"],
        "zero_models": parent["zero_models"],
        "thresholds": parent["thresholds"],
        "content_feature_order": parent["content_feature_order"],
        "score_batch_size": SCORE_BATCH_SIZE,
        "generation": {
            "partition": "confirmation",
            "panels": ["consistency", "reversal"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "batch_size": GENERATION_BATCH_SIZE,
            "parser": "exact frozen expected/opposite label word-boundary XOR",
        },
        "dependencies": {
            "phase1287_protocol": file_sha256(INPUT_PROTOCOL),
            "phase1287_material": file_sha256(INPUT_MATERIAL),
            "phase1287_final": file_sha256(INPUT_FINAL),
            "phase1287_audit": file_sha256(INPUT_AUDIT),
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "formal_run_budget": 1,
        "unblinding_order": [
            "save all frozen candidate scores and generation text",
            "fit discovery models and read selection only",
            "write the selected-family artifact with confirmation_unread=true",
            "compute confirmation behavior, generation, reliability, transport, specificity, and total-account ledgers",
        ],
        "hard_stops": parent["hard_stops"],
        "claims_forbidden": parent["claims_forbidden"],
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
        offsets: list[int] = []
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
            entry["mean_log_prob"][value["role"]] = total / continuation_length
            entry["continuation_length"][value["role"]] = continuation_length
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
    for partition in PARTITIONS:
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
                "median_control_leakage": float(np.median([value["control_leakage"] for value in values])),
                "total_mean_effect_sign_agreement": float(np.mean([
                    (value["effect"] > 0) == (total_value["effect"] > 0)
                    for value, total_value in zip(values, total_values)
                ])),
            }
            surface_cells[f"{partition}.{surface}"] = cell
            for axis_name in axes:
                axis_values = [primary[(row["row_id"], surface)] for row in partition_rows if row["axis"] == axis_name]
                axis_cells[f"{partition}.{surface}.{axis_name}"] = {
                    "n_worlds": len(axis_values),
                    "positive_fraction": float(np.mean([value["effect"] > 0 for value in axis_values])),
                    "median_effect": float(np.median([value["effect"] for value in axis_values])),
                }
    axis_pass_counts: dict[str, int] = {}
    for partition in PARTITIONS:
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
                "parsed": parsed, "prediction": prediction, "gold": gold,
                "correct": bool(parsed and prediction == gold),
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


def sorted_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return sorted((row for row in rows if row["partition"] == partition), key=lambda row: row["row_id"])


def discovery_centers(
    rows: list[dict[str, Any]], signatures: dict[tuple[str, str], dict[str, Any]], kind: str,
) -> dict[str, np.ndarray]:
    discovery = sorted_rows(rows, "discovery")
    return {
        surface: np.stack([signatures[(row["row_id"], surface)][kind] for row in discovery]).mean(axis=0)
        for surface in SURFACE_ORDER
    }


def variant_matrix(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    centers: dict[str, np.ndarray],
    partition: str,
    family: str,
    variant: str,
    kind: str,
) -> np.ndarray:
    surface = f"{family}_{variant}"
    return np.stack([
        signatures[(row["row_id"], surface)][kind] - centers[surface]
        for row in sorted_rows(rows, partition)
    ])


def family_matrix(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    centers: dict[str, np.ndarray],
    partition: str,
    family: str,
    kind: str,
) -> np.ndarray:
    return np.mean([
        variant_matrix(rows, signatures, centers, partition, family, variant, kind)
        for variant in VARIANTS
    ], axis=0)


def raw_family_matrix(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    partition: str,
    family: str,
    kind: str,
) -> np.ndarray:
    selected = sorted_rows(rows, partition)
    return np.mean([
        np.stack([signatures[(row["row_id"], f"{family}_{variant}")][kind] for row in selected])
        for variant in VARIANTS
    ], axis=0)


def content_matrix(rows: list[dict[str, Any]], partition: str, feature_order: list[str]) -> np.ndarray:
    return np.asarray([
        [row["content_features"][feature] for feature in feature_order]
        for row in sorted_rows(rows, partition)
    ], dtype=np.float64)


def reliability_summary(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    centers: dict[str, np.ndarray],
    thresholds: dict[str, float],
    kind: str = "active",
) -> dict[str, Any]:
    per_family = {}
    for family in FAMILIES:
        left = variant_matrix(rows, signatures, centers, "confirmation", family, "a", kind)
        right = variant_matrix(rows, signatures, centers, "confirmation", family, "b", kind)
        same = np.asarray([cosine(a, b) for a, b in zip(left, right)])
        wrong_medians = []
        for offset in WRONG_WORLD_OFFSETS:
            wrong_medians.append(float(np.median([cosine(a, b) for a, b in zip(left, np.roll(right, offset, axis=0))])))
        residual = (left + right) / 2.0
        raw = raw_family_matrix(rows, signatures, "confirmation", family, kind)
        energy_ratio = float(np.median(np.linalg.norm(residual, axis=1)) / max(np.median(np.linalg.norm(raw, axis=1)), 1.0e-12))
        per_family[family] = {
            "n": len(same),
            "median_cosine": float(np.median(same)),
            "positive_fraction": float(np.mean(same > 0)),
            "wrong_world_median_cosines": wrong_medians,
            "cosine_advantage_over_best_wrong_world": float(np.median(same) - max(wrong_medians)),
            "residual_energy_ratio": energy_ratio,
        }
    gates = {
        "energy": min(value["residual_energy_ratio"] for value in per_family.values()) >= thresholds["residual_energy_ratio_min"],
        "median_cosine": min(value["median_cosine"] for value in per_family.values()) >= thresholds["residual_reliability_median_cosine_min"],
        "positive_fraction": min(value["positive_fraction"] for value in per_family.values()) >= thresholds["residual_reliability_positive_fraction_min"],
        "wrong_world_advantage": min(value["cosine_advantage_over_best_wrong_world"] for value in per_family.values()) >= thresholds["residual_reliability_gain_over_wrong_world_min"],
    }
    return {"kind": kind, "per_family": per_family, "gates": gates, "passed": all(gates.values())}


def fit_family(
    family: str,
    source: np.ndarray,
    content: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    if family == "H0_zero":
        return {"family": family}
    if family == "HC_content":
        mean = content.mean(axis=0)
        scale = content.std(axis=0)
        scale[scale < 1.0e-8] = 1.0
        standardized = (content - mean) / scale
        design = np.column_stack([standardized, np.ones(len(content))])
        regularizer = np.diag([1.0e-2] * standardized.shape[1] + [0.0])
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
        return {
            "family": family, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
            "matrix": weights[:-1].tolist(), "offset": weights[-1].tolist(),
        }
    if family == "H1_identity":
        return {"family": family}
    if family == "H2_diagonal":
        gains = []
        for column in range(target.shape[1]):
            denominator = float(np.dot(source[:, column], source[:, column]) + 1.0e-3)
            gains.append(float(np.dot(source[:, column], target[:, column]) / denominator))
        return {"family": family, "gains": gains}
    if family == "H3_full":
        regularizer = 1.0e-2 * np.eye(source.shape[1])
        matrix = np.linalg.solve(source.T @ source + regularizer, source.T @ target)
        return {"family": family, "matrix": matrix.tolist()}
    raise ValueError(f"unknown family {family}")


def predict_family(model: dict[str, Any], source: np.ndarray, content: np.ndarray) -> np.ndarray:
    family = model["family"]
    if family == "H0_zero":
        return np.zeros((len(source), source.shape[1]), dtype=np.float64)
    if family == "HC_content":
        standardized = (
            content - np.asarray(model["feature_mean"], dtype=np.float64)
        ) / np.asarray(model["feature_scale"], dtype=np.float64)
        return standardized @ np.asarray(model["matrix"], dtype=np.float64) + np.asarray(model["offset"], dtype=np.float64)
    if family == "H1_identity":
        return source.copy()
    if family == "H2_diagonal":
        return source * np.asarray(model["gains"], dtype=np.float64)
    if family == "H3_full":
        return source @ np.asarray(model["matrix"], dtype=np.float64)
    raise ValueError(f"unknown family {family}")


def prediction_metrics(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray | None = None) -> dict[str, Any]:
    squared_error = float(np.square(target - prediction).sum())
    energy = max(float(np.square(target).sum()), 1.0e-12)
    cosines = np.asarray([cosine(left, right) for left, right in zip(target, prediction)])
    result = {
        "n": len(target),
        "squared_error": squared_error,
        "energy": energy,
        "nrmse": float(np.sqrt(squared_error / energy)),
        "risk_gain_over_zero": float(1.0 - squared_error / energy),
        "median_cosine": float(np.median(cosines)),
        "positive_fraction": float(np.mean(cosines > 0)),
    }
    if baseline is not None:
        baseline_error = max(float(np.square(target - baseline).sum()), 1.0e-12)
        result["risk_gain_over_baseline"] = float(1.0 - squared_error / baseline_error)
        result["baseline_nrmse"] = float(np.sqrt(baseline_error / energy))
    return result


def matrices(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    centers: dict[str, np.ndarray],
    partition: str,
    feature_order: list[str],
    kind: str = "active",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    source = family_matrix(rows, signatures, centers, partition, SOURCE_FAMILY, kind)
    targets = {
        target: family_matrix(rows, signatures, centers, partition, target, kind)
        for target in TARGET_FAMILIES
    }
    content = content_matrix(rows, partition, feature_order)
    return source, targets, content


def fit_and_select(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    centers: dict[str, np.ndarray],
    feature_order: list[str],
    tolerance: float,
    account: str,
) -> dict[str, Any]:
    source_d, targets_d, content_d = matrices(rows, signatures, centers, "discovery", feature_order)
    source_s, targets_s, content_s = matrices(rows, signatures, centers, "selection", feature_order)
    fits: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    pooled: dict[str, float] = {}
    for family in FAMILY_ORDER:
        fits[family] = {}
        selection[family] = {}
        total_error = 0.0
        total_energy = 0.0
        for target in TARGET_FAMILIES:
            model = fit_family(family, source_d, content_d, targets_d[target])
            fits[family][target] = model
            prediction = predict_family(model, source_s, content_s)
            metrics = prediction_metrics(targets_s[target], prediction)
            selection[family][target] = metrics
            total_error += metrics["squared_error"]
            total_energy += metrics["energy"]
        pooled[family] = float(np.sqrt(total_error / max(total_energy, 1.0e-12)))
    best = min(pooled.values())
    eligible = [family for family in FAMILY_ORDER if pooled[family] <= best + tolerance]
    return {
        "account": account,
        "family_order": list(FAMILY_ORDER),
        "discovery_fits": fits,
        "selection_metrics": selection,
        "pooled_selection_nrmse": pooled,
        "minimum_selection_nrmse": best,
        "simplicity_tolerance": tolerance,
        "eligible_within_tolerance": eligible,
        "selected_family": eligible[0],
    }


def evaluate_selected(
    rows: list[dict[str, Any]],
    signatures: dict[tuple[str, str], dict[str, Any]],
    centers_by_kind: dict[str, dict[str, np.ndarray]],
    feature_order: list[str],
    selected_family: str,
    thresholds: dict[str, float],
    account: str,
) -> dict[str, Any]:
    train_parts = ("discovery", "selection")
    source_train = np.concatenate([
        family_matrix(rows, signatures, centers_by_kind["active"], partition, SOURCE_FAMILY, "active")
        for partition in train_parts
    ])
    content_train = np.concatenate([content_matrix(rows, partition, feature_order) for partition in train_parts])
    source_c, targets_c, content_c = matrices(
        rows, signatures, centers_by_kind["active"], "confirmation", feature_order, "active",
    )
    per_target: dict[str, Any] = {}
    refits: dict[str, Any] = {}
    for target in TARGET_FAMILIES:
        target_train = np.concatenate([
            family_matrix(rows, signatures, centers_by_kind["active"], partition, target, "active")
            for partition in train_parts
        ])
        model = fit_family(selected_family, source_train, content_train, target_train)
        content_model = fit_family("HC_content", source_train, content_train, target_train)
        refits[target] = model
        prediction = predict_family(model, source_c, content_c)
        content_prediction = predict_family(content_model, source_c, content_c)
        metrics = prediction_metrics(targets_c[target], prediction, content_prediction)
        metrics["risk_gain_over_content"] = metrics.pop("risk_gain_over_baseline")
        wrong_errors = []
        for offset in WRONG_WORLD_OFFSETS:
            wrong_prediction = predict_family(model, np.roll(source_c, offset, axis=0), content_c)
            wrong_errors.append(float(np.square(targets_c[target] - wrong_prediction).sum()))
        best_wrong_error = min(wrong_errors)
        metrics["wrong_world_squared_errors"] = wrong_errors
        metrics["risk_gain_over_best_wrong_world"] = float(1.0 - metrics["squared_error"] / max(best_wrong_error, 1.0e-12))
        role_errors = [
            float(np.square(targets_c[target] - prediction[:, list(permutation)]).sum())
            for permutation in ROLE_PERMUTATIONS
        ]
        metrics["role_permutation_squared_errors"] = role_errors
        metrics["risk_gain_over_best_role_permutation"] = float(
            1.0 - metrics["squared_error"] / max(min(role_errors), 1.0e-12)
        )
        nulls = {}
        for kind in ("lexical", "role"):
            null_source_train = np.concatenate([
                family_matrix(rows, signatures, centers_by_kind[kind], partition, SOURCE_FAMILY, kind)
                for partition in train_parts
            ])
            null_target_train = np.concatenate([
                family_matrix(rows, signatures, centers_by_kind[kind], partition, target, kind)
                for partition in train_parts
            ])
            null_source_c = family_matrix(rows, signatures, centers_by_kind[kind], "confirmation", SOURCE_FAMILY, kind)
            null_target_c = family_matrix(rows, signatures, centers_by_kind[kind], "confirmation", target, kind)
            # Apply the active map. HC needs the same content; all source-dependent maps use null source residuals.
            null_prediction = predict_family(model, null_source_c, content_c)
            nulls[kind] = prediction_metrics(null_target_c, null_prediction)
            nulls[kind]["train_target_energy"] = float(np.square(null_target_train).sum())
            nulls[kind]["train_source_energy"] = float(np.square(null_source_train).sum())
        metrics["nulls"] = nulls
        metrics["active_minus_lexical_gain"] = metrics["risk_gain_over_zero"] - nulls["lexical"]["risk_gain_over_zero"]
        metrics["active_minus_role_gain"] = metrics["risk_gain_over_zero"] - nulls["role"]["risk_gain_over_zero"]
        per_target[target] = metrics
    map_gates = {
        "source_dependent_family": selected_family in ("H1_identity", "H2_diagonal", "H3_full"),
        "zero_gain": min(value["risk_gain_over_zero"] for value in per_target.values()) >= thresholds["transport_risk_gain_over_zero_min"],
        "content_gain": min(value["risk_gain_over_content"] for value in per_target.values()) >= thresholds["transport_risk_gain_over_content_min"],
        "median_cosine": min(value["median_cosine"] for value in per_target.values()) >= thresholds["transport_median_cosine_min"],
        "positive_fraction": min(value["positive_fraction"] for value in per_target.values()) >= thresholds["transport_positive_fraction_min"],
        "wrong_world_gain": min(value["risk_gain_over_best_wrong_world"] for value in per_target.values()) >= thresholds["transport_gain_over_wrong_world_min"],
        "role_permutation_gain": min(value["risk_gain_over_best_role_permutation"] for value in per_target.values()) >= thresholds["transport_gain_over_role_permutation_min"],
    }
    specificity_gates = {
        "lexical_advantage": min(value["active_minus_lexical_gain"] for value in per_target.values()) >= thresholds["transport_active_minus_lexical_gain_min"],
        "role_advantage": min(value["active_minus_role_gain"] for value in per_target.values()) >= thresholds["transport_active_minus_role_gain_min"],
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
        raise RuntimeError("formal Phase1288 run already complete")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent Phase1288 preaudit missing")
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(INPUT_MATERIAL)
    model = None
    started = time.perf_counter()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if device.type != "cuda" or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError(f"FP16 qualification failed: {precision}")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"]:
            raise RuntimeError(f"quantization/BF16 qualification failed: {precision}")
        with torch.inference_mode():
            raw = score_examples(model, tokenizer, device, rows)
            generations = generate_confirmation(model, tokenizer, device, rows)
        write_jsonl(RAW, raw)
        write_jsonl(GENERATIONS, generations)

        signatures = build_signatures(raw, rows)
        centers_mean = {
            kind: discovery_centers(rows, signatures["mean_log_prob"], kind)
            for kind in ("active", "lexical", "role")
        }
        primary_selection = fit_and_select(
            rows, signatures["mean_log_prob"], centers_mean["active"], protocol["content_feature_order"],
            protocol["thresholds"]["selection_simplicity_tolerance"], "mean_log_prob",
        )
        selection_artifact = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "written_at_utc": utc_now(),
            "confirmation_metrics_read": False,
            **primary_selection,
        }
        selection_artifact["decision_digest"] = digest(selection_artifact)
        atomic_json(SELECTION_DECISION, selection_artifact)

        behavior = behavior_summary(raw, rows, signatures, protocol["thresholds"])
        generation = generation_summary(generations, protocol["thresholds"])
        reliability = reliability_summary(
            rows, signatures["mean_log_prob"], centers_mean["active"], protocol["thresholds"], "active",
        )
        primary_mapping = evaluate_selected(
            rows, signatures["mean_log_prob"], centers_mean, protocol["content_feature_order"],
            primary_selection["selected_family"], protocol["thresholds"], "mean_log_prob",
        )

        centers_total = {
            kind: discovery_centers(rows, signatures["total_log_prob"], kind)
            for kind in ("active", "lexical", "role")
        }
        total_mapping = evaluate_selected(
            rows, signatures["total_log_prob"], centers_total, protocol["content_feature_order"],
            primary_selection["selected_family"], protocol["thresholds"], "total_log_prob_same_family",
        )
        total_account_passed = min(
            value["risk_gain_over_zero"] for value in total_mapping["confirmation"].values()
        ) > protocol["thresholds"]["total_account_transport_gain_min"]

        ledgers = {
            "behavior": behavior["passed"],
            "generation": generation["passed"],
            "residual_reliability": reliability["passed"],
            "transport": primary_mapping["mapping_passed"],
            "specificity": primary_mapping["specificity_passed"],
            "total_account": total_account_passed,
        }
        all_passed = all(ledgers.values())
        runtime = time.perf_counter() - started
        summary = {
            "behavior": behavior,
            "generation": generation,
            "reliability": reliability,
            "primary_selection": primary_selection,
            "primary_mapping": primary_mapping,
            "total_log_prob_sensitivity": total_mapping,
            "total_account_passed": total_account_passed,
            "ledgers": ledgers,
            "runtime_seconds": runtime,
            "precision_audit": precision,
            "placement": placement,
        }
        atomic_json(RUN_SUMMARY, summary)
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "verdict": "qwen3_world_residual_transport_qualified" if all_passed else "qwen3_world_residual_transport_gate_failed",
            "ledgers": ledgers,
            "behavior": behavior,
            "generation": generation,
            "reliability": reliability,
            "selection": {
                "selected_family": primary_selection["selected_family"],
                "pooled_selection_nrmse": primary_selection["pooled_selection_nrmse"],
                "decision_digest": selection_artifact["decision_digest"],
            },
            "mapping": primary_mapping,
            "total_log_prob_sensitivity": {
                "selected_family": primary_selection["selected_family"],
                "transport_gains": {
                    target: value["risk_gain_over_zero"]
                    for target, value in total_mapping["confirmation"].items()
                },
                "passed": total_account_passed,
            },
            "runtime_seconds": runtime,
            "precision_audit": precision,
            "authorization": "phase1289_qwen3_hidden_world_residual_path" if all_passed else "close_c027_without_hidden",
            "scope": (
                "Qwen3-4B FP16; 27 disjoint closed-binary axes; four new surface families with two wordings each; "
                "world-indexed behavior residual reliability and cross-surface transport."
            ),
        }
        atomic_json(FINAL, final)
        atomic_json(COMPLETE, {
            "phase": PHASE,
            "completed_at_utc": utc_now(),
            "raw_sha256": file_sha256(RAW),
            "generation_sha256": file_sha256(GENERATIONS),
            "selection_decision_sha256": file_sha256(SELECTION_DECISION),
            "run_summary_sha256": file_sha256(RUN_SUMMARY),
            "final_sha256": file_sha256(FINAL),
        })
        print(canonical_json({
            "verdict": final["verdict"],
            "authorization": final["authorization"],
            "ledgers": ledgers,
            "selected_family": primary_selection["selected_family"],
            "behavior_gates": behavior["gates"],
            "generation_gates": generation["gates"],
            "reliability_gates": reliability["gates"],
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
    arguments = parser.parse_args()
    preregister(arguments.force) if arguments.action == "preregister" else run_model()
