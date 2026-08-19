#!/usr/bin/env python3
"""Phase1290: one-shot Qwen3 behavior qualification for C028."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
from phase1289_c028_typed_complement_contract import (  # noqa: E402
    ACTIVE_PANELS, FAMILIES, PANELS, PARTITIONS, SURFACES,
)


PHASE = 1290
CAMPAIGN = "C028"
CONTRACT_ID = "EXP-C028-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1290_c028_qwen3_typed_complement_behavior_audit.py"
PARENT = ROOT / "tests/glm5/result/phase1289_c028_typed_complement_contract"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_MATERIAL = PARENT / "material/frozen_typed_complement_material.jsonl"
PARENT_REVIEW = PARENT / "material/pre_model_semantic_naturalness_review.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1290_c028_qwen3_typed_complement_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GENERATIONS = OUT / "raw/confirmation_generations.jsonl"
RUN_SUMMARY = OUT / "analysis/run_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

SCORE_BATCH_SIZE = 16
GENERATION_BATCH_SIZE = 8
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


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("Phase1290 protocol already exists")
    parent = read_json(PARENT_PROTOCOL)
    parent_final = read_json(PARENT_FINAL)
    parent_audit = read_json(PARENT_AUDIT)
    if parent_final.get("authorization") != "phase1290_qwen3_typed_complement_behavior":
        raise RuntimeError("Phase1289 authorization missing")
    if not parent_audit.get("all_checks_passed") or parent_audit.get("authorization") != "phase1290_qwen3_typed_complement_behavior":
        raise RuntimeError("Phase1289 independent audit missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1290.c028.qwen3.behavior.v1",
        "research_object": parent["research_object"],
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "formal_run_budget": 1,
        "counts": parent["counts"],
        "partitions": list(PARTITIONS),
        "families": list(FAMILIES),
        "surfaces": list(SURFACES),
        "panels": list(PANELS),
        "active_panels": list(ACTIVE_PANELS),
        "thresholds": parent["thresholds"],
        "zero_models": parent["zero_models"],
        "metric_scopes": {
            "partition_candidate_accuracy": "minimum over all five panels within each partition",
            "surface_candidate_accuracy": "minimum over all five panels within each surface",
            "active_panel_accuracy_and_margin": "minimum over identity, single, and double across all partitions and surfaces",
            "surface_variant_both_correct": "minimum over family x panel cells, paired by row",
            "generation_coverage_and_exact_accuracy": "minimum over confirmation surface x active-panel cells",
            "shortcut_program_ceiling": "maximum accuracy on active panels only",
        },
        "scoring": {
            "primary": "total continuation log probability",
            "sensitivity": "mean continuation log probability",
            "tie_policy": "a tie is incorrect and prediction is null",
            "batch_size": SCORE_BATCH_SIZE,
        },
        "generation": {
            "partition": "confirmation",
            "panels": list(ACTIVE_PANELS),
            "batch_size": GENERATION_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "coverage_parser": "exactly one of the two state labels occurs by word boundary",
            "exact_parser": "normalized first nonempty line equals 'the final state is <gold>.'",
        },
        "unblinding_order": [
            "save all candidate scores and raw confirmation generation text",
            "write raw file hashes",
            "compute candidate, composition, null, surface, generation, and shortcut ledgers",
            "write final authorization without modifying the frozen contract",
        ],
        "dependencies": {
            "phase1289_protocol": file_sha256(PARENT_PROTOCOL),
            "phase1289_material": file_sha256(PARENT_MATERIAL),
            "phase1289_review": file_sha256(PARENT_REVIEW),
            "phase1289_final": file_sha256(PARENT_FINAL),
            "phase1289_audit": file_sha256(PARENT_AUDIT),
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "branching": parent["branching"],
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
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "model_weights_loaded": False,
    })
    print(canonical_json({"status": "preregistered", "protocol_digest": protocol["protocol_digest"]}))


def score_candidates(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        for surface in SURFACES:
            for panel in PANELS:
                context = row["contexts"][surface][panel]
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                for role, continuation in row["candidate_continuations"].items():
                    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
                    if full_ids[:len(context_ids)] != context_ids:
                        raise RuntimeError("candidate prefix drift")
                    length = len(full_ids) - len(context_ids)
                    if length <= 0:
                        raise RuntimeError("empty candidate")
                    examples.append({
                        "row_id": row["row_id"], "partition": row["partition"], "axis": row["axis"],
                        "surface": surface, "panel": panel, "role": role,
                        "full_ids": full_ids, "context_length": len(context_ids), "continuation_length": length,
                    })
    scored: dict[tuple[str, str, str], dict[str, Any]] = {}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for start in range(0, len(examples), SCORE_BATCH_SIZE):
        batch = examples[start:start + SCORE_BATCH_SIZE]
        maximum = max(len(value["full_ids"]) for value in batch)
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
            entry["mean_log_prob"][value["role"]] = total / value["continuation_length"]
            entry["continuation_length"][value["role"]] = value["continuation_length"]
        if (start // SCORE_BATCH_SIZE + 1) % 100 == 0:
            print(canonical_json({"scored": min(start + SCORE_BATCH_SIZE, len(examples)), "total": len(examples)}), flush=True)
    return sorted(scored.values(), key=lambda value: (value["row_id"], value["surface"], value["panel"]))


def first_nonempty_line(text: str) -> str:
    for line in text.replace("\r", "\n").split("\n"):
        normalized = " ".join(line.strip().lower().strip('"\' ').split())
        if normalized:
            return normalized
    return ""


def generate_confirmation(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row["partition"] != "confirmation":
            continue
        for surface in SURFACES:
            for panel in ACTIVE_PANELS:
                examples.append({
                    "row_id": row["row_id"], "axis": row["axis"], "surface": surface, "panel": panel,
                    "context": row["contexts"][surface][panel], "left_label": row["left_label"],
                    "right_label": row["right_label"], "gold_label": row["gold_by_panel"][panel],
                })
    output: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for start in range(0, len(examples), GENERATION_BATCH_SIZE):
        batch = examples[start:start + GENERATION_BATCH_SIZE]
        encoded = [tokenizer.encode(value["context"], add_special_tokens=False) for value in batch]
        maximum = max(map(len, encoded))
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
            left = bool(re.search(rf"\b{re.escape(value['left_label'].lower())}\b", lower))
            right = bool(re.search(rf"\b{re.escape(value['right_label'].lower())}\b", lower))
            covered = left ^ right
            predicted = value["left_label"] if left and not right else value["right_label"] if right and not left else None
            expected_sentence = f"the final state is {value['gold_label']}."
            exact_sentence = first_nonempty_line(text) == expected_sentence
            output.append({
                **{key: value[key] for key in ("row_id", "axis", "surface", "panel", "left_label", "right_label", "gold_label")},
                "generation": text, "covered": covered, "prediction": predicted,
                "label_correct": bool(covered and predicted == value["gold_label"]),
                "exact_sentence": bool(exact_sentence), "expected_sentence": expected_sentence,
            })
        if (start // GENERATION_BATCH_SIZE + 1) % 25 == 0:
            print(canonical_json({"generated": min(start + GENERATION_BATCH_SIZE, len(examples)), "total": len(examples)}), flush=True)
    return output


def shortcut_predictions(row: dict[str, Any], panel: str, context: str) -> dict[str, str]:
    base = row["base_label"]
    opposite = row["opposite_label"]
    left, right = row["left_label"], row["right_label"]
    has_complement_wording = any(phrase in context.lower() for phrase in (
        "does not apply", "fails to describe", "does not match", "unassigned alternative",
        "is not true of", "does not belong to",
    ))
    return {
        "constant_left": left,
        "constant_right": right,
        "source_only": base,
        "always_complement": opposite,
        "surface_not_heuristic": opposite if has_complement_wording else base,
        "listed_first": row["listed_order"][0],
        "listed_second": row["listed_order"][1],
        "target_blind_operation": right if panel == "single_complement" else left,
    }


def annotate_raw(raw: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["row_id"]: row for row in rows}
    annotated = []
    for value in raw:
        row = by_id[value["row_id"]]
        left_score = value["total_log_prob"]["left"]
        right_score = value["total_log_prob"]["right"]
        prediction = row["left_label"] if left_score > right_score else row["right_label"] if right_score > left_score else None
        gold = row["gold_by_panel"][value["panel"]]
        gold_score = left_score if gold == row["left_label"] else right_score
        other_score = right_score if gold == row["left_label"] else left_score
        annotated.append({
            **value,
            "left_label": row["left_label"], "right_label": row["right_label"],
            "base_side": row["base_side"], "base_label": row["base_label"], "opposite_label": row["opposite_label"],
            "listed_order": row["listed_order"], "gold_label": gold, "prediction": prediction,
            "correct": bool(prediction == gold), "gold_margin": float(gold_score - other_score),
            "shortcut_predictions": shortcut_predictions(row, value["panel"], row["contexts"][value["surface"]][value["panel"]]),
        })
    return annotated


def mean_bool(values: Iterable[bool]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def behavior_summary(raw: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    partitions = {part: mean_bool(v["correct"] for v in raw if v["partition"] == part) for part in PARTITIONS}
    surfaces = {surface: mean_bool(v["correct"] for v in raw if v["surface"] == surface) for surface in SURFACES}
    active = {panel: {
        "accuracy": mean_bool(v["correct"] for v in raw if v["panel"] == panel),
        "median_gold_margin": float(np.median([v["gold_margin"] for v in raw if v["panel"] == panel])),
    } for panel in ACTIVE_PANELS}
    base_sides = {str(side): mean_bool(v["correct"] for v in raw if v["base_side"] == side) for side in (0, 1)}
    grouped = defaultdict(dict)
    for value in raw:
        grouped[(value["row_id"], value["surface"])][value["panel"]] = value
    triple = []
    identity_double = []
    identity_single = []
    lexical = []
    scope = []
    for values in grouped.values():
        triple.append(all(values[panel]["correct"] for panel in ACTIVE_PANELS))
        identity_double.append(values["identity"]["correct"] and values["double_complement"]["correct"] and values["identity"]["prediction"] == values["double_complement"]["prediction"])
        identity_single.append(values["identity"]["correct"] and values["single_complement"]["correct"] and values["identity"]["prediction"] != values["single_complement"]["prediction"])
        lexical.append(values["identity"]["correct"] and values["lexical_null"]["correct"] and values["identity"]["prediction"] == values["lexical_null"]["prediction"])
        scope.append(values["identity"]["correct"] and values["scope_null"]["correct"] and values["identity"]["prediction"] == values["scope_null"]["prediction"])
    variant_cells: dict[str, float] = {}
    by_key = {(v["row_id"], v["surface"], v["panel"]): v for v in raw}
    for family in FAMILIES:
        for panel in PANELS:
            values = []
            for row_id in {v["row_id"] for v in raw}:
                a = by_key[(row_id, f"{family}_a", panel)]
                b = by_key[(row_id, f"{family}_b", panel)]
                values.append(a["correct"] and b["correct"] and a["prediction"] == b["prediction"])
            variant_cells[f"{family}.{panel}"] = mean_bool(values)
    shortcut = {}
    active_rows = [v for v in raw if v["panel"] in ACTIVE_PANELS]
    for name in next(iter(active_rows))["shortcut_predictions"]:
        shortcut[name] = mean_bool(v["shortcut_predictions"][name] == v["gold_label"] for v in active_rows)
    finite = mean_bool(
        np.isfinite([*v["total_log_prob"].values(), *v["mean_log_prob"].values()]).all()
        for v in raw
    )
    metrics = {
        "finite_fraction": finite,
        "overall_candidate_accuracy": mean_bool(v["correct"] for v in raw),
        "partition_accuracy": partitions,
        "surface_accuracy": surfaces,
        "active_panel": active,
        "active_triple_all_correct_rate": mean_bool(triple),
        "identity_double_both_correct_rate": mean_bool(identity_double),
        "identity_single_opposition_both_correct_rate": mean_bool(identity_single),
        "lexical_null_preservation_rate": mean_bool(lexical),
        "scope_null_preservation_rate": mean_bool(scope),
        "surface_variant_cells": variant_cells,
        "base_side_accuracy": base_sides,
        "shortcut_program_accuracy_active": shortcut,
        "shortcut_program_ceiling_active": max(shortcut.values()),
    }
    gates = {
        "finite": finite >= thresholds["finite_fraction_min"],
        "overall_candidate": metrics["overall_candidate_accuracy"] >= thresholds["overall_candidate_accuracy_min"],
        "partition_candidate": min(partitions.values()) >= thresholds["partition_candidate_accuracy_min"],
        "surface_candidate": min(surfaces.values()) >= thresholds["surface_candidate_accuracy_min"],
        "active_panel_accuracy": min(value["accuracy"] for value in active.values()) >= thresholds["active_panel_accuracy_min"],
        "active_panel_margin": min(value["median_gold_margin"] for value in active.values()) >= thresholds["median_gold_margin_per_active_panel_min"],
        "active_triple": metrics["active_triple_all_correct_rate"] >= thresholds["active_triple_all_correct_rate_min"],
        "identity_double": metrics["identity_double_both_correct_rate"] >= thresholds["identity_double_both_correct_rate_min"],
        "identity_single": metrics["identity_single_opposition_both_correct_rate"] >= thresholds["identity_single_opposition_both_correct_rate_min"],
        "lexical_null": metrics["lexical_null_preservation_rate"] >= thresholds["lexical_null_preservation_rate_min"],
        "scope_null": metrics["scope_null_preservation_rate"] >= thresholds["scope_null_preservation_rate_min"],
        "surface_variants": min(variant_cells.values()) >= thresholds["surface_variant_both_correct_rate_min"],
        "base_sides": min(base_sides.values()) >= thresholds["base_side_accuracy_min"],
        "shortcut_ceiling": metrics["shortcut_program_ceiling_active"] <= thresholds["shortcut_program_accuracy_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def generation_summary(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    cells = {}
    for surface in SURFACES:
        for panel in ACTIVE_PANELS:
            subset = [value for value in rows if value["surface"] == surface and value["panel"] == panel]
            cells[f"{surface}.{panel}"] = {
                "n": len(subset),
                "coverage": mean_bool(value["covered"] for value in subset),
                "label_accuracy": mean_bool(value["label_correct"] for value in subset),
                "exact_sentence_accuracy": mean_bool(value["exact_sentence"] for value in subset),
            }
    grouped = defaultdict(dict)
    for value in rows:
        grouped[(value["row_id"], value["surface"])][value["panel"]] = value
    triple = mean_bool(all(values[panel]["exact_sentence"] for panel in ACTIVE_PANELS) for values in grouped.values())
    coverage_min = min(value["coverage"] for value in cells.values())
    exact_min = min(value["exact_sentence_accuracy"] for value in cells.values())
    gates = {
        "coverage": coverage_min >= thresholds["generation_coverage_min"],
        "exact_accuracy": exact_min >= thresholds["generation_exact_accuracy_min"],
        "active_triple": triple >= thresholds["generation_active_triple_rate_min"],
    }
    return {
        "cells": cells, "coverage_min": coverage_min, "exact_accuracy_min": exact_min,
        "active_triple_exact_rate": triple, "gates": gates, "passed": all(gates.values()),
    }


def run_model() -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal Phase1290 run already complete")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent Phase1290 preaudit missing")
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(PARENT_MATERIAL)
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
            raw = annotate_raw(score_candidates(model, tokenizer, device, rows), rows)
            generations = generate_confirmation(model, tokenizer, device, rows)
        write_jsonl(RAW, raw)
        write_jsonl(GENERATIONS, generations)
        raw_hashes = {"candidate_scores": file_sha256(RAW), "confirmation_generations": file_sha256(GENERATIONS)}
        behavior = behavior_summary(raw, protocol["thresholds"])
        generation = generation_summary(generations, protocol["thresholds"])
        ledgers = {
            "candidate_behavior": behavior["passed"],
            "natural_generation": generation["passed"],
        }
        all_passed = all(ledgers.values())
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "runtime_seconds": time.perf_counter() - started,
            "precision_audit": precision,
            "placement": placement,
            "raw_hashes": raw_hashes,
            "raw_counts": {"candidate_contexts": len(raw), "generations": len(generations)},
            "behavior": behavior,
            "generation": generation,
            "ledgers": ledgers,
            "all_ledgers_passed": all_passed,
        }
        atomic_json(RUN_SUMMARY, summary)
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "verdict": "qwen3_typed_complement_behavior_qualified" if all_passed else "qwen3_typed_complement_behavior_gate_failed",
            "protocol_digest": protocol["protocol_digest"],
            "raw_hashes": raw_hashes,
            "ledgers": ledgers,
            "behavior_gate": behavior["passed"],
            "generation_gate": generation["passed"],
            "authorization": "phase1291_multievent_future_response_contract" if all_passed else "close_c028_without_hidden",
            "hidden_measured": False,
            "other_models_run": False,
            "new_mathematics_required": False,
        }
        atomic_json(FINAL, final)
        atomic_json(COMPLETE, {
            "completed_at_utc": utc_now(), "formal_runs_used": 1, "protocol_digest": protocol["protocol_digest"],
            "run_summary_sha256": file_sha256(RUN_SUMMARY), "final_sha256": file_sha256(FINAL), **raw_hashes,
        })
        print(canonical_json({"verdict": final["verdict"], "ledgers": ledgers, "authorization": final["authorization"]}))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister(args.force)
    else:
        run_model()
