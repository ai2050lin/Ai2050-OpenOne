#!/usr/bin/env python3
"""Phase1284: Qwen3 behavior gate for the C025 response-signature object."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
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


PHASE = 1284
CAMPAIGN = "C025"
CONTRACT_ID = "EXP-C025-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1284_c025_qwen3_response_signature_behavior_audit.py"
INPUT = ROOT / "tests/glm5/result/phase1283_c025_response_signature_contract"
INPUT_PROTOCOL = INPUT / "protocol/preregistration.json"
INPUT_MATERIAL = INPUT / "material/frozen_response_worlds.jsonl"
INPUT_FINAL = INPUT / "analysis/final.json"
INPUT_AUDIT = INPUT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1284_c025_qwen3_response_signature_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/response_library_scores.jsonl"
GENERATION = OUT / "raw/confirmation_generations.jsonl"
RUN_SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"

ROLE_ORDER = ("expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1")
PANELS = ("consistency", "reversal", "carrier_consistency", "lexical_consistency", "role_consistency", "role_reversal")
PARTITION_SURFACES = {
    "discovery": ("test_confirmation", "forecast_agreement"),
    "selection": ("evidence_support", "outcome_match"),
    "confirmation": ("measurement_validation", "finding_consistency"),
}
TEMPLATE = np.asarray([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
SCORE_BATCH_SIZE = 24
GENERATION_BATCH_SIZE = 16
MAX_NEW_TOKENS = 12


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


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-12 else np.zeros_like(value)


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    parent = read_json(INPUT_PROTOCOL)
    final = read_json(INPUT_FINAL)
    audit = read_json(INPUT_AUDIT)
    if final.get("authorization") != "phase1284_qwen3_response_signature_behavior" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1283 authorization missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1284.c025.qwen.response_signature.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "row_count": 192,
        "context_count": 2304,
        "scored_sequence_count": 13824,
        "role_order": ROLE_ORDER,
        "panels": PANELS,
        "partition_surfaces": PARTITION_SURFACES,
        "score_batch_size": SCORE_BATCH_SIZE,
        "generation": {
            "partition": "confirmation", "panels": ["consistency", "reversal"],
            "accepted_terms": "row.expected_terms versus row.opposite_terms",
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "batch_size": GENERATION_BATCH_SIZE,
        },
        "thresholds": parent["thresholds"],
        "dependencies": {
            "phase1283_protocol": file_sha256(INPUT_PROTOCOL),
            "phase1283_material": file_sha256(INPUT_MATERIAL),
            "phase1283_final": file_sha256(INPUT_FINAL),
            "phase1283_audit": file_sha256(INPUT_AUDIT),
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "formal_run_budget": 1,
        "hard_stops": [
            "All 13,824 full candidate sequences are scored before any behavior gate is evaluated.",
            "The response signature is centered within the frozen six-role candidate library.",
            "Axis-level minima, not expanded sequence rows, determine the scientific gates.",
            "The discovery centroid is frozen before selection and confirmation holdout metrics are read.",
            "Greedy generation uses only frozen confirmation axes and surfaces and a predeclared term parser.",
            "Any behavior or generation failure stops C025 before hidden-state hooks.",
            "No threshold, candidate, parser term, surface, axis, panel, or null may be changed after this protocol.",
        ],
    }
    protocol = {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    print(canonical_json({"status": "preregistered", "protocol_digest": protocol["protocol_digest"]}))


def score_examples(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        for surface, panels in row["contexts"].items():
            for panel, context in panels.items():
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                for role in ROLE_ORDER:
                    continuation = row["candidate_continuations"][role]
                    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
                    if full_ids[:len(context_ids)] != context_ids:
                        raise RuntimeError("candidate prefix drift")
                    examples.append({
                        "row_id": row["row_id"], "partition": row["partition"], "axis": row["axis"],
                        "surface": surface, "panel": panel, "role": role,
                        "full_ids": full_ids, "context_length": len(context_ids),
                        "continuation_length": len(full_ids) - len(context_ids),
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
            score = float(log_probs[index, positions, token_ids].sum().item())
            key = (value["row_id"], value["surface"], value["panel"])
            entry = scored.setdefault(key, {
                "row_id": value["row_id"], "partition": value["partition"], "axis": value["axis"],
                "surface": value["surface"], "panel": value["panel"], "log_prob": {}, "continuation_length": {},
            })
            entry["log_prob"][value["role"]] = score
            entry["continuation_length"][value["role"]] = value["continuation_length"]
        if (start // SCORE_BATCH_SIZE + 1) % 100 == 0:
            print(canonical_json({"scored_sequences": min(start + SCORE_BATCH_SIZE, len(examples)), "total": len(examples)}), flush=True)
    output = []
    for value in scored.values():
        value["finite"] = bool(np.isfinite([value["log_prob"][role] for role in ROLE_ORDER]).all())
        output.append(value)
    return sorted(output, key=lambda value: (value["row_id"], value["surface"], value["panel"]))


def response(by_key: dict[tuple[str, str, str], dict[str, Any]], row_id: str, surface: str, right: str, left: str) -> np.ndarray:
    right_values = np.asarray([by_key[(row_id, surface, right)]["log_prob"][role] for role in ROLE_ORDER], dtype=np.float64)
    left_values = np.asarray([by_key[(row_id, surface, left)]["log_prob"][role] for role in ROLE_ORDER], dtype=np.float64)
    value = right_values - left_values
    return value - value.mean()


def behavior_summary(scored: list[dict[str, Any]], rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    by_key = {(value["row_id"], value["surface"], value["panel"]): value for value in scored}
    row_meta = {value["row_id"]: value for value in rows}
    signatures: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        for surface in row["contexts"]:
            active = response(by_key, row["row_id"], surface, "reversal", "consistency")
            lexical = response(by_key, row["row_id"], surface, "lexical_consistency", "carrier_consistency")
            role = response(by_key, row["row_id"], surface, "role_reversal", "role_consistency")
            target_scale = max(float(np.mean(np.abs(active[:4]))), 1.0e-12)
            signatures[(row["row_id"], surface)] = {
                "active": active,
                "effect": float(np.mean(active[2:4]) - np.mean(active[0:2])),
                "template_cosine": cosine(active, TEMPLATE),
                "lexical_ratio": float(np.linalg.norm(lexical) / max(np.linalg.norm(active), 1.0e-12)),
                "role_ratio": float(np.linalg.norm(role) / max(np.linalg.norm(active), 1.0e-12)),
                "control_leakage": float(np.mean(np.abs(active[4:6])) / target_scale),
            }
    axis_cells = {}
    for partition, surfaces in PARTITION_SURFACES.items():
        axes = sorted({row["axis"] for row in rows if row["partition"] == partition})
        for surface in surfaces:
            for axis_name in axes:
                ids = [row["row_id"] for row in rows if row["partition"] == partition and row["axis"] == axis_name]
                values = [signatures[(row_id, surface)] for row_id in ids]
                axis_cells[f"{partition}.{surface}.{axis_name}"] = {
                    "n_worlds": len(ids),
                    "positive_fraction": float(np.mean([value["effect"] > 0 for value in values])),
                    "median_effect": float(np.median([value["effect"] for value in values])),
                    "median_template_cosine": float(np.median([value["template_cosine"] for value in values])),
                    "median_lexical_ratio": float(np.median([value["lexical_ratio"] for value in values])),
                    "median_role_ratio": float(np.median([value["role_ratio"] for value in values])),
                    "median_control_leakage": float(np.median([value["control_leakage"] for value in values])),
                }
    paired_surface = {}
    for partition, surfaces in PARTITION_SURFACES.items():
        axes = sorted({row["axis"] for row in rows if row["partition"] == partition})
        for axis_name in axes:
            ids = [row["row_id"] for row in rows if row["partition"] == partition and row["axis"] == axis_name]
            values = [cosine(signatures[(row_id, surfaces[0])]["active"], signatures[(row_id, surfaces[1])]["active"]) for row_id in ids]
            paired_surface[f"{partition}.{axis_name}"] = {"n_worlds": len(ids), "median_cosine": float(np.median(values)), "minimum": float(np.min(values))}
    discovery_units = [
        unit(signatures[(row["row_id"], surface)]["active"])
        for row in rows if row["partition"] == "discovery" for surface in PARTITION_SURFACES["discovery"]
    ]
    discovery_centroid = unit(np.mean(discovery_units, axis=0))
    holdout = {}
    for partition in ("selection", "confirmation"):
        axes = sorted({row["axis"] for row in rows if row["partition"] == partition})
        for axis_name in axes:
            values = [
                cosine(signatures[(row["row_id"], surface)]["active"], discovery_centroid)
                for row in rows if row["partition"] == partition and row["axis"] == axis_name
                for surface in PARTITION_SURFACES[partition]
            ]
            holdout[f"{partition}.{axis_name}"] = {
                "n_signatures": len(values), "median_cosine": float(np.median(values)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0)), "minimum": float(np.min(values)),
            }
    gates = {
        "finite": float(np.mean([value["finite"] for value in scored])) >= thresholds["finite_fraction_min"],
        "active_positive": min(value["positive_fraction"] for value in axis_cells.values()) >= thresholds["active_positive_fraction_min"],
        "active_axis_median": min(value["median_effect"] for value in axis_cells.values()) >= thresholds["active_axis_median_min"],
        "template_cosine": min(value["median_template_cosine"] for value in axis_cells.values()) >= thresholds["template_cosine_axis_median_min"],
        "paired_surface": min(value["median_cosine"] for value in paired_surface.values()) >= thresholds["paired_surface_cosine_median_min"],
        "holdout_centroid_cosine": min(value["median_cosine"] for value in holdout.values()) >= thresholds["holdout_centroid_cosine_axis_median_min"],
        "holdout_centroid_positive": min(value["positive_fraction"] for value in holdout.values()) >= thresholds["holdout_centroid_positive_fraction_min"],
        "lexical_null": max(value["median_lexical_ratio"] for value in axis_cells.values()) <= thresholds["lexical_null_norm_ratio_max"],
        "role_null": max(value["median_role_ratio"] for value in axis_cells.values()) <= thresholds["role_null_norm_ratio_max"],
        "control_leakage": max(value["median_control_leakage"] for value in axis_cells.values()) <= thresholds["control_leakage_ratio_max"],
    }
    return {
        "raw_context_count": len(scored),
        "finite_fraction": float(np.mean([value["finite"] for value in scored])),
        "axis_cells": axis_cells,
        "paired_surface": paired_surface,
        "discovery_centroid": discovery_centroid.tolist(),
        "holdout": holdout,
        "gate_extrema": {
            "active_positive_min": min(value["positive_fraction"] for value in axis_cells.values()),
            "active_effect_min": min(value["median_effect"] for value in axis_cells.values()),
            "template_cosine_min": min(value["median_template_cosine"] for value in axis_cells.values()),
            "paired_surface_cosine_min": min(value["median_cosine"] for value in paired_surface.values()),
            "holdout_cosine_min": min(value["median_cosine"] for value in holdout.values()),
            "holdout_positive_min": min(value["positive_fraction"] for value in holdout.values()),
            "lexical_ratio_max": max(value["median_lexical_ratio"] for value in axis_cells.values()),
            "role_ratio_max": max(value["median_role_ratio"] for value in axis_cells.values()),
            "control_leakage_max": max(value["median_control_leakage"] for value in axis_cells.values()),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "independent_axis_count": len({row_meta[value["row_id"]]["axis"] for value in scored}),
    }


def generate_confirmation(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row["partition"] != "confirmation":
            continue
        for surface in PARTITION_SURFACES["confirmation"]:
            for panel in ("consistency", "reversal"):
                examples.append({
                    "row_id": row["row_id"], "axis": row["axis"], "surface": surface, "panel": panel,
                    "context": row["contexts"][surface][panel],
                    "expected_terms": row["expected_terms"], "opposite_terms": row["opposite_terms"],
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
            found_expected = [term for term in value["expected_terms"] if re.search(rf"\b{re.escape(term.lower())}\b", lower)]
            found_opposite = [term for term in value["opposite_terms"] if re.search(rf"\b{re.escape(term.lower())}\b", lower)]
            parsed = bool(found_expected) ^ bool(found_opposite)
            prediction = "expected" if found_expected and not found_opposite else "opposite" if found_opposite and not found_expected else None
            gold = "expected" if value["panel"] == "consistency" else "opposite"
            output.append({
                **value, "generation": text, "matched_expected": found_expected, "matched_opposite": found_opposite,
                "parsed": parsed, "prediction": prediction, "gold": gold, "correct": parsed and prediction == gold,
            })
    return output


def generation_summary(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    cells = {}
    for surface in PARTITION_SURFACES["confirmation"]:
        for panel in ("consistency", "reversal"):
            subset = [value for value in rows if value["surface"] == surface and value["panel"] == panel]
            coverage = float(np.mean([value["parsed"] for value in subset]))
            parsed = [value for value in subset if value["parsed"]]
            accuracy = float(np.mean([value["correct"] for value in parsed])) if parsed else 0.0
            cells[f"{surface}.{panel}"] = {"n": len(subset), "coverage": coverage, "accuracy_given_parsed": accuracy}
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


def run_model() -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal run already complete")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("preaudit missing")
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
            scored = score_examples(model, tokenizer, device, rows)
            behavior = behavior_summary(scored, rows, protocol["thresholds"])
            generations = generate_confirmation(model, tokenizer, device, rows)
            generation = generation_summary(generations, protocol["thresholds"])
        write_jsonl(RAW, scored)
        write_jsonl(GENERATION, generations)
        passed = behavior["passed"] and generation["passed"]
        runtime = time.perf_counter() - started
        summary = {
            "behavior": behavior, "generation": generation, "runtime_seconds": runtime,
            "precision_audit": precision, "placement": placement,
        }
        atomic_json(RUN_SUMMARY, summary)
        final = {
            "phase": PHASE, "campaign": CAMPAIGN,
            "verdict": "qwen3_response_signature_behavior_qualified" if passed else "qwen3_response_signature_behavior_gate_failed",
            "behavior": behavior, "generation": generation, "runtime_seconds": runtime, "precision_audit": precision,
            "authorization": "phase1285_qwen3_typed_multievent_response_causality" if passed else "stop_c025_at_response_signature_behavior",
            "scope": "Qwen3-4B FP16; 24 disjoint explicit expectation axes; six frozen response-library roles; six held-out surface families",
        }
        atomic_json(FINAL, final)
        atomic_json(COMPLETE, {
            "phase": PHASE, "completed_at_utc": utc_now(), "raw_sha256": file_sha256(RAW),
            "generation_sha256": file_sha256(GENERATION), "final_sha256": file_sha256(FINAL),
        })
        print(canonical_json(final))
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
