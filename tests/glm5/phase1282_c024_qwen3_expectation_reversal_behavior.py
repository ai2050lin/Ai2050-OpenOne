#!/usr/bin/env python3
"""Phase1282: Qwen3 full-continuation and generation gate for C024."""

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


PHASE = 1282
CAMPAIGN = "C024"
CONTRACT_ID = "EXP-C024-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1282_c024_qwen3_expectation_reversal_behavior_audit.py"
INPUT = ROOT / "tests/glm5/result/phase1281_c024_expectation_reversal_contract"
INPUT_PROTOCOL = INPUT / "protocol/preregistration.json"
INPUT_MATERIAL = INPUT / "material/frozen_expectation_worlds.jsonl"
INPUT_FINAL = INPUT / "analysis/final.json"
INPUT_AUDIT = INPUT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1282_c024_qwen3_expectation_reversal_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/full_continuation_scores.jsonl"
GENERATION = OUT / "raw/confirmation_generations.jsonl"
RUN_SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

SURFACES = ("coordination", "adverbial", "expectation", "evaluation", "report")
PANELS = ("consistency", "contrast", "carrier_consistency", "lexical_consistency", "carrier_contrast", "lexical_contrast")
CONSISTENCY = ("consistency", "carrier_consistency", "lexical_consistency")
CONTRAST = ("contrast", "carrier_contrast", "lexical_contrast")
SCORE_BATCH_SIZE = 24
GENERATION_BATCH_SIZE = 16
MAX_NEW_TOKENS = 6


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


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    parent = read_json(INPUT_PROTOCOL)
    final = read_json(INPUT_FINAL)
    audit = read_json(INPUT_AUDIT)
    if final.get("authorization") != "phase1282_qwen3_multitoken_behavior_and_generation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1281 authorization missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1282.c024.qwen.behavior.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "row_count": final["row_count"],
        "context_count": final["context_count"],
        "scored_sequence_count": final["scored_sequence_count"],
        "surfaces": list(SURFACES),
        "panels": list(PANELS),
        "score_batch_size": SCORE_BATCH_SIZE,
        "generation": {"partition": "confirmation", "panels": ["consistency", "contrast"], "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "batch_size": GENERATION_BATCH_SIZE},
        "thresholds": parent["thresholds"],
        "dependencies": {
            "phase1281_protocol": file_sha256(INPUT_PROTOCOL),
            "phase1281_material": file_sha256(INPUT_MATERIAL),
            "phase1281_final": file_sha256(INPUT_FINAL),
            "phase1281_audit": file_sha256(INPUT_AUDIT),
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "formal_run_budget": 1,
        "hard_stops": [
            "All 15,360 continuation sequences are scored before the behavior gate is evaluated.",
            "Primary scores sum token log probabilities over the entire frozen continuation.",
            "Generation is greedy and evaluated only on the untouched confirmation partition.",
            "Quoted-cue effects are compared against equal-token carrier notes in both relation modes.",
            "Behavior or generation failure denies hidden-state intervention.",
            "A pass authorizes Qwen3 typed causal study only and does not prove all contrast semantics.",
            "No repair, threshold change, surface deletion or adjective-pair deletion is allowed after this preregistration.",
        ],
    }
    protocol = {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    print(canonical_json({"status": "preregistered", "protocol_digest": protocol["protocol_digest"]}))


def score_examples(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        for surface in SURFACES:
            for panel in PANELS:
                context = row["contexts"][surface][panel]
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                for identity in ("expected", "opposite"):
                    continuation = row["continuations"][identity]
                    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
                    if full_ids[:len(context_ids)] != context_ids:
                        raise RuntimeError("continuation prefix drift")
                    examples.append({
                        "row_id": row["row_id"], "partition": row["partition"], "surface": surface,
                        "panel": panel, "identity": identity, "full_ids": full_ids,
                        "context_length": len(context_ids), "continuation_length": len(full_ids) - len(context_ids),
                    })
    scored: dict[tuple[str, str, str], dict[str, Any]] = {}
    for start in range(0, len(examples), SCORE_BATCH_SIZE):
        batch = examples[start:start + SCORE_BATCH_SIZE]
        maximum = max(len(row["full_ids"]) for row in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(batch), maximum), int(pad_id), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        offsets = []
        for index, row in enumerate(batch):
            offset = maximum - len(row["full_ids"])
            offsets.append(offset)
            ids[index, offset:] = torch.tensor(row["full_ids"], dtype=torch.long, device=device)
            mask[index, offset:] = 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        for index, row in enumerate(batch):
            offset = offsets[index]
            first = offset + row["context_length"]
            last = offset + len(row["full_ids"])
            token_ids = ids[index, first:last]
            positions = torch.arange(first - 1, last - 1, device=device)
            value = float(log_probs[index, positions, token_ids].sum().item())
            key = (row["row_id"], row["surface"], row["panel"])
            entry = scored.setdefault(key, {
                "row_id": row["row_id"], "partition": row["partition"], "surface": row["surface"],
                "panel": row["panel"], "log_prob": {}, "continuation_length": {},
            })
            entry["log_prob"][row["identity"]] = value
            entry["continuation_length"][row["identity"]] = row["continuation_length"]
        if (start // SCORE_BATCH_SIZE + 1) % 100 == 0:
            print(canonical_json({"scored_sequences": min(start + SCORE_BATCH_SIZE, len(examples)), "total": len(examples)}), flush=True)
    output = []
    for value in scored.values():
        value["D_opposite_minus_expected"] = value["log_prob"]["opposite"] - value["log_prob"]["expected"]
        value["finite"] = bool(np.isfinite(list(value["log_prob"].values())).all())
        value["predicted_identity"] = "opposite" if value["D_opposite_minus_expected"] > 0 else "expected"
        output.append(value)
    return sorted(output, key=lambda row: (row["row_id"], row["surface"], row["panel"]))


def behavior_summary(scored: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    by_key = {(row["row_id"], row["surface"], row["panel"]): row for row in scored}
    partitions = ("discovery", "selection", "confirmation")
    sign_accuracy: dict[str, float] = {}
    effects: dict[str, Any] = {}
    lexical: dict[str, Any] = {}
    for partition in partitions:
        ids = sorted({row["row_id"] for row in scored if row["partition"] == partition})
        for surface in SURFACES:
            for panel in PANELS:
                values = [by_key[(row_id, surface, panel)]["D_opposite_minus_expected"] for row_id in ids]
                correct = [value < 0 for value in values] if panel in CONSISTENCY else [value > 0 for value in values]
                sign_accuracy[f"{partition}.{surface}.{panel}"] = float(np.mean(correct))
            delta = np.asarray([
                by_key[(row_id, surface, "contrast")]["D_opposite_minus_expected"]
                - by_key[(row_id, surface, "consistency")]["D_opposite_minus_expected"] for row_id in ids
            ], dtype=np.float64)
            lc = np.asarray([
                by_key[(row_id, surface, "lexical_consistency")]["D_opposite_minus_expected"]
                - by_key[(row_id, surface, "carrier_consistency")]["D_opposite_minus_expected"] for row_id in ids
            ], dtype=np.float64)
            lr = np.asarray([
                by_key[(row_id, surface, "lexical_contrast")]["D_opposite_minus_expected"]
                - by_key[(row_id, surface, "carrier_contrast")]["D_opposite_minus_expected"] for row_id in ids
            ], dtype=np.float64)
            effect_median = float(np.median(delta))
            ratio = float(np.median(np.maximum(np.abs(lc), np.abs(lr))) / max(abs(effect_median), 1.0e-12))
            effects[f"{partition}.{surface}"] = {
                "positive_fraction": float(np.mean(delta > 0)),
                "median": effect_median,
                "minimum": float(np.min(delta)),
                "maximum": float(np.max(delta)),
            }
            lexical[f"{partition}.{surface}"] = {
                "consistency_median": float(np.median(lc)),
                "contrast_median": float(np.median(lr)),
                "interaction_median": float(np.median(lr - lc)),
                "specific_abs_ratio": ratio,
            }
    core_min = min(value for key, value in sign_accuracy.items() if key.split(".")[-1] in ("consistency", "contrast"))
    null_min = min(value for key, value in sign_accuracy.items() if key.split(".")[-1] not in ("consistency", "contrast"))
    effect_positive_min = min(value["positive_fraction"] for value in effects.values())
    effect_median_min = min(value["median"] for value in effects.values())
    lexical_ratio_max = max(value["specific_abs_ratio"] for value in lexical.values())
    finite_fraction = float(np.mean([row["finite"] for row in scored]))
    gates = {
        "finite": finite_fraction >= thresholds["finite_fraction_min"],
        "core_sign": core_min >= thresholds["core_sign_accuracy_min"],
        "null_sign": null_min >= thresholds["null_sign_accuracy_min"],
        "effect_positive": effect_positive_min >= thresholds["effect_positive_fraction_min"],
        "effect_median": effect_median_min >= thresholds["median_functional_effect_min"],
        "lexical_specificity": lexical_ratio_max <= thresholds["lexical_specific_ratio_max"],
    }
    return {
        "context_count": len(scored), "finite_fraction": finite_fraction,
        "sign_accuracy": sign_accuracy, "functional_effect": effects, "lexical_effect": lexical,
        "core_sign_accuracy_min": core_min, "null_sign_accuracy_min": null_min,
        "effect_positive_fraction_min": effect_positive_min, "median_functional_effect_min": effect_median_min,
        "lexical_specific_ratio_max": lexical_ratio_max, "gates": gates, "passed": all(gates.values()),
    }


def generate_confirmation(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row["partition"] != "confirmation":
            continue
        for surface in SURFACES:
            for panel in ("consistency", "contrast"):
                examples.append({
                    "row_id": row["row_id"], "surface": surface, "panel": panel,
                    "context": row["contexts"][surface][panel],
                    "expected_adjective": row["expected_adjective"], "opposite_adjective": row["opposite_adjective"],
                })
    output = []
    for start in range(0, len(examples), GENERATION_BATCH_SIZE):
        batch = examples[start:start + GENERATION_BATCH_SIZE]
        encoded = [tokenizer.encode(row["context"], add_special_tokens=False) for row in batch]
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
        for row, text in zip(batch, texts):
            lower = text.lower()
            found_expected = bool(re.search(rf"\b{re.escape(row['expected_adjective'].lower())}\b", lower))
            found_opposite = bool(re.search(rf"\b{re.escape(row['opposite_adjective'].lower())}\b", lower))
            parsed = found_expected ^ found_opposite
            predicted = "expected" if found_expected and not found_opposite else "opposite" if found_opposite and not found_expected else None
            gold = "expected" if row["panel"] == "consistency" else "opposite"
            output.append({**row, "generation": text, "parsed": parsed, "prediction": predicted, "gold": gold, "correct": parsed and predicted == gold})
    return output


def generation_summary(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    cells = {}
    for surface in SURFACES:
        for panel in ("consistency", "contrast"):
            subset = [row for row in rows if row["surface"] == surface and row["panel"] == panel]
            coverage = float(np.mean([row["parsed"] for row in subset]))
            parsed = [row for row in subset if row["parsed"]]
            accuracy = float(np.mean([row["correct"] for row in parsed])) if parsed else 0.0
            cells[f"{surface}.{panel}"] = {"coverage": coverage, "accuracy_given_parsed": accuracy, "n": len(subset)}
    coverage_min = min(value["coverage"] for value in cells.values())
    accuracy_min = min(value["accuracy_given_parsed"] for value in cells.values())
    gates = {
        "coverage": coverage_min >= thresholds["generation_parse_coverage_min"],
        "accuracy": accuracy_min >= thresholds["generation_sign_accuracy_min"],
    }
    return {"cells": cells, "coverage_min": coverage_min, "accuracy_min": accuracy_min, "gates": gates, "passed": all(gates.values())}


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
            behavior = behavior_summary(scored, protocol["thresholds"])
            generations = generate_confirmation(model, tokenizer, device, rows)
            generation = generation_summary(generations, protocol["thresholds"])
        write_jsonl(RAW, scored)
        write_jsonl(GENERATION, generations)
        passed = behavior["passed"] and generation["passed"]
        runtime = time.perf_counter() - started
        summary = {"behavior": behavior, "generation": generation, "runtime_seconds": runtime, "precision_audit": precision, "placement": placement}
        atomic_json(RUN_SUMMARY, summary)
        final = {
            "phase": PHASE, "campaign": CAMPAIGN,
            "verdict": "qwen3_expectation_reversal_behavior_qualified" if passed else "qwen3_expectation_reversal_behavior_gate_failed",
            "behavior": behavior, "generation": generation, "precision_audit": precision,
            "runtime_seconds": runtime,
            "authorization": "phase1283_qwen3_typed_multievent_causal_closure" if passed else "stop_c024_at_natural_use_behavior",
            "scope": "Qwen3-4B FP16; explicit expectation satisfaction/violation; full multi-token scoring plus greedy confirmation generation",
        }
        atomic_json(FINAL, final)
        atomic_json(COMPLETE, {"phase": PHASE, "completed_at_utc": utc_now(), "raw_sha256": file_sha256(RAW), "generation_sha256": file_sha256(GENERATION), "final_sha256": file_sha256(FINAL)})
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
