#!/usr/bin/env python3
"""Phase 1295: one-shot Qwen3 behavior gate for frozen C030."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402


PHASE = 1295
CAMPAIGN = "C030"
SCRIPT = Path(__file__).resolve()
AUDITOR = TEST_ROOT / "phase1295_c030_qwen3_grounded_lookup_behavior_audit.py"
PARENT = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_MATERIAL = PARENT / "material/frozen_grounded_lookup_cases.jsonl"
PARENT_NATURALNESS = PARENT / "material/pre_model_grammar_type_review.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
OUT = TEST_ROOT / "result/phase1295_c030_qwen3_grounded_lookup_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GENERATIONS = OUT / "raw/list_free_generations.jsonl"
SUMMARY = OUT / "analysis/run_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("discovery", "confirmation", "holdout")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SURFACES = ("catalog_prose", "inventory_ledger")
STATES = (0, 1)
SCORE_BATCH_SIZE = 16
GENERATION_BATCH_SIZE = 8
MAX_NEW_TOKENS = 4


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    os.replace(tmp, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chat_render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Use only the supplied catalog. Reply exactly as requested and do not explain."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("Phase1295 protocol already exists")
    parent = load(PARENT_PROTOCOL)
    parent_final = load(PARENT_FINAL)
    parent_audit = load(PARENT_AUDIT)
    if parent_final.get("authorization") != "phase1295_qwen3_behavior_only":
        raise RuntimeError("Phase1294 authorization missing")
    if not parent_audit.get("all_checks_passed") or parent_audit.get("authorization") != "phase1295_qwen3_behavior_only":
        raise RuntimeError("Phase1294 independent audit missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "experiment_id": "EXP-C030-WP01-001",
        "schema_version": "phase1295.c030.behavior.v1",
        "research_object": "grounded object-attribute inverse lookup under externally generated binding perturbations",
        "type_signature": "(WorldState, Attribute, Value) -> Entity",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "formal_run_budget": 1,
        "material_count": parent["material"]["case_count"],
        "generation_count": parent["material"]["generation_cases"],
        "partitions": list(PARTITIONS),
        "panels": list(PANELS),
        "surfaces": list(SURFACES),
        "thresholds": parent["thresholds"],
        "zero_models": parent["zero_models"],
        "candidate_scoring": {
            "context": "native non-thinking chat template",
            "continuation": "one leading-space entity-name token",
            "score": "next-token log probability",
            "tie_policy": "tie is incorrect and prediction is null",
            "batch_size": SCORE_BATCH_SIZE,
        },
        "generation": {
            "partitions": ["confirmation", "holdout"],
            "candidate_order": 0,
            "candidate_list_present": False,
            "batch_size": GENERATION_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "coverage_parser": "exactly one candidate occurs as an ASCII word-boundary match",
            "accuracy_parser": "first nonempty line stripped only of whitespace, quotes, terminal period, comma, colon, or semicolon equals gold ignoring case",
        },
        "metric_scopes": {
            "candidate": ["overall", "minimum partition", "minimum panel", "minimum surface", "minimum binding state"],
            "paired": ["each panel both-state success", "candidate-order triple success", "cross-surface pair success"],
            "generation": ["coverage", "exact accuracy", "both-state pair success"],
            "shortcut": "frozen external program ceiling",
        },
        "unblinding_order": [
            "write all candidate log probabilities",
            "write all raw list-free generation text",
            "hash both raw ledgers",
            "compute every frozen metric and gate",
            "write authorization without changing contract, parser, or threshold",
        ],
        "dependencies": {
            "phase1294_protocol": sha(PARENT_PROTOCOL),
            "phase1294_material": sha(PARENT_MATERIAL),
            "phase1294_naturalness": sha(PARENT_NATURALNESS),
            "phase1294_final": sha(PARENT_FINAL),
            "phase1294_audit": sha(PARENT_AUDIT),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "authorization_if_pass": "phase1296_multievent_response_preregistration_only",
        "authorization_if_fail": "close_c030_without_hidden",
        "hard_stops": [
            "No hidden state is read in Phase1295.",
            "Any candidate, pair, generation, finite, or shortcut gate failure closes C030.",
            "No prompt repair, panel deletion, threshold change, parser change, seed rerun, or other-model vote is allowed after unblinding.",
        ],
    }
    frozen = {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}
    save(PROTOCOL, frozen)
    save(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "model_weights_loaded": False,
    })
    print(canonical({"status": "preregistered", "protocol_digest": frozen["protocol_digest"]}))


def prepare_scoring(tokenizer: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        rendered = chat_render(tokenizer, row["candidate_prompt"])
        context_ids = tokenizer.encode(rendered, add_special_tokens=False)
        candidate_ids = []
        for candidate in row["candidates"]:
            full = tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)
            if full[:len(context_ids)] != context_ids or len(full) != len(context_ids) + 1:
                raise RuntimeError(f"candidate continuation contract drift: {row['case_id']} {candidate}")
            candidate_ids.append(full[-1])
        prepared.append({"row": row, "context_ids": context_ids, "candidate_ids": candidate_ids})
    return prepared


@torch.inference_mode()
def score_candidates(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = prepare_scoring(tokenizer, rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    supports_last = "logits_to_keep" in inspect.signature(model.forward).parameters
    output: list[dict[str, Any]] = []
    for start in range(0, len(prepared), SCORE_BATCH_SIZE):
        batch = prepared[start:start + SCORE_BATCH_SIZE]
        maximum = max(len(item["context_ids"]) for item in batch)
        ids = torch.full((len(batch), maximum), int(pad_id), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        for index, item in enumerate(batch):
            values = item["context_ids"]
            ids[index, -len(values):] = torch.tensor(values, dtype=torch.long, device=device)
            mask[index, -len(values):] = 1
        kwargs = {"input_ids": ids, "attention_mask": mask, "use_cache": False, "return_dict": True}
        if supports_last:
            kwargs["logits_to_keep"] = 1
        logits = model(**kwargs).logits[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        for index, item in enumerate(batch):
            row = item["row"]
            scores = {
                candidate: float(log_probs[index, token_id].item())
                for candidate, token_id in zip(row["candidates"], item["candidate_ids"])
            }
            ordered = sorted(scores, key=lambda name: (-scores[name], name))
            prediction = ordered[0] if scores[ordered[0]] > scores[ordered[1]] else None
            gold = row["gold_candidate"]
            other_best = max(score for name, score in scores.items() if name != gold)
            output.append({
                "case_id": row["case_id"], "group_id": row["group_id"],
                "partition": row["partition"], "profile_index": row["profile_index"],
                "attribute": row["attribute"], "panel": row["panel"], "surface": row["surface"],
                "candidate_order": row["candidate_order"], "binding_state": row["binding_state"],
                "entities": row["entities"], "record_order": row["record_order"], "candidates": row["candidates"],
                "gold_candidate": gold, "candidate_token_ids": item["candidate_ids"],
                "candidate_log_prob": scores, "prediction": prediction,
                "correct": prediction == gold, "gold_margin": float(scores[gold] - other_best),
                "finite": bool(all(np.isfinite(list(scores.values())))),
            })
        if (start // SCORE_BATCH_SIZE + 1) % 50 == 0:
            print(canonical({"candidate_scored": min(start + SCORE_BATCH_SIZE, len(prepared)), "total": len(prepared)}), flush=True)
    return output


def normalize_first_line(text: str) -> str:
    for line in text.replace("\r", "\n").split("\n"):
        value = line.strip().strip("\"' ").strip(".,:; ").strip()
        if value:
            return value.lower()
    return ""


@torch.inference_mode()
def generate_list_free(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["partition"] in {"confirmation", "holdout"} and row["candidate_order"] == 0
    ]
    prepared = [{"row": row, "ids": tokenizer.encode(chat_render(tokenizer, row["generation_prompt"]), add_special_tokens=False)} for row in selected]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    output = []
    for start in range(0, len(prepared), GENERATION_BATCH_SIZE):
        batch = prepared[start:start + GENERATION_BATCH_SIZE]
        maximum = max(len(item["ids"]) for item in batch)
        ids = torch.full((len(batch), maximum), int(pad_id), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        for index, item in enumerate(batch):
            values = item["ids"]
            ids[index, -len(values):] = torch.tensor(values, dtype=torch.long, device=device)
            mask[index, -len(values):] = 1
        generated = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=int(pad_id),
            eos_token_id=tokenizer.eos_token_id,
        )[:, maximum:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for item, text in zip(batch, texts):
            row = item["row"]
            hits = [
                candidate for candidate in row["candidates"]
                if re.search(rf"\b{re.escape(candidate)}\b", text, flags=re.IGNORECASE)
            ]
            prediction = hits[0] if len(hits) == 1 else None
            normalized = normalize_first_line(text)
            exact = normalized == row["gold_candidate"].lower()
            output.append({
                "case_id": row["case_id"], "group_id": row["group_id"],
                "partition": row["partition"], "profile_index": row["profile_index"],
                "attribute": row["attribute"], "panel": row["panel"], "surface": row["surface"],
                "candidate_order": row["candidate_order"], "binding_state": row["binding_state"],
                "candidates": row["candidates"], "gold_candidate": row["gold_candidate"],
                "generation": text, "normalized_first_line": normalized, "candidate_hits": hits,
                "covered": len(hits) == 1, "prediction": prediction,
                "label_correct": prediction == row["gold_candidate"], "exact_correct": exact,
            })
        if (start // GENERATION_BATCH_SIZE + 1) % 25 == 0:
            print(canonical({"generated": min(start + GENERATION_BATCH_SIZE, len(prepared)), "total": len(prepared)}), flush=True)
    return output


def rate(values: Iterable[bool]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def candidate_summary(raw: list[dict[str, Any]], thresholds: dict[str, float], shortcut: float) -> dict[str, Any]:
    partition = {key: rate(row["correct"] for row in raw if row["partition"] == key) for key in PARTITIONS}
    panel = {key: rate(row["correct"] for row in raw if row["panel"] == key) for key in PANELS}
    surface = {key: rate(row["correct"] for row in raw if row["surface"] == key) for key in SURFACES}
    states = {str(key): rate(row["correct"] for row in raw if row["binding_state"] == key) for key in STATES}

    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    surface_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        pair_groups[row["group_id"]].append(row)
        order_groups[(row["partition"], row["profile_index"], row["attribute"], row["panel"], row["surface"], row["binding_state"])].append(row)
        surface_groups[(row["partition"], row["profile_index"], row["attribute"], row["panel"], row["candidate_order"], row["binding_state"])].append(row)
    pair_success = {
        panel_name: rate(
            len(group) == 2 and all(row["correct"] for row in group)
            for group in pair_groups.values() if group and group[0]["panel"] == panel_name
        ) for panel_name in PANELS
    }
    order_success = rate(len(group) == 3 and all(row["correct"] for row in group) for group in order_groups.values())
    cross_surface = rate(len(group) == 2 and all(row["correct"] for row in group) for group in surface_groups.values())
    finite = rate(row["finite"] for row in raw)
    overall = rate(row["correct"] for row in raw)
    metrics = {
        "finite_fraction": finite,
        "overall_candidate_accuracy": overall,
        "partition_candidate_accuracy": partition,
        "panel_candidate_accuracy": panel,
        "surface_candidate_accuracy": surface,
        "binding_state_accuracy": states,
        "panel_pair_success": pair_success,
        "candidate_order_triple_success": order_success,
        "cross_surface_pair_success": cross_surface,
        "median_gold_margin": float(np.median([row["gold_margin"] for row in raw])),
        "shortcut_program_ceiling": shortcut,
    }
    gates = {
        "finite": finite >= thresholds["finite_fraction_min"],
        "overall_candidate": overall >= thresholds["overall_candidate_accuracy_min"],
        "partition_candidate": min(partition.values()) >= thresholds["partition_candidate_accuracy_min"],
        "panel_candidate": min(panel.values()) >= thresholds["panel_candidate_accuracy_min"],
        "surface_candidate": min(surface.values()) >= thresholds["surface_candidate_accuracy_min"],
        "binding_state": min(states.values()) >= thresholds["base_side_accuracy_min"],
        "active_pair": pair_success["active"] >= thresholds["active_pair_success_min"],
        "matched_null_pair": pair_success["matched_null"] >= thresholds["matched_null_pair_success_min"],
        "surface_only_pair": pair_success["surface_only"] >= thresholds["surface_only_pair_success_min"],
        "semantic_neighbor_pair": pair_success["semantic_neighbor"] >= thresholds["semantic_neighbor_pair_success_min"],
        "candidate_order_triple": order_success >= thresholds["candidate_order_triple_success_min"],
        "cross_surface_pair": cross_surface >= thresholds["cross_surface_pair_success_min"],
        "shortcut": shortcut <= thresholds["shortcut_program_accuracy_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def generation_summary(raw: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[(row["partition"], row["profile_index"], row["attribute"], row["panel"], row["surface"])].append(row)
    coverage = rate(row["covered"] for row in raw)
    exact = rate(row["exact_correct"] for row in raw)
    label = rate(row["label_correct"] for row in raw)
    pair_success = rate(len(group) == 2 and all(row["exact_correct"] for row in group) for group in groups.values())
    cells = {}
    for partition in ("confirmation", "holdout"):
        for panel in PANELS:
            for surface in SURFACES:
                subset = [row for row in raw if row["partition"] == partition and row["panel"] == panel and row["surface"] == surface]
                cells[f"{partition}|{panel}|{surface}"] = {
                    "coverage": rate(row["covered"] for row in subset),
                    "exact_accuracy": rate(row["exact_correct"] for row in subset),
                }
    metrics = {
        "coverage": coverage,
        "exact_accuracy": exact,
        "label_accuracy": label,
        "both_state_pair_success": pair_success,
        "cells": cells,
    }
    gates = {
        "coverage": coverage >= thresholds["generation_coverage_min"],
        "accuracy": exact >= thresholds["generation_accuracy_min"],
        "pair_success": pair_success >= thresholds["generation_pair_success_min"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def run() -> None:
    protocol = load(PROTOCOL)
    preaudit = load(PREAUDIT)
    if not preaudit.get("all_checks_passed") or preaudit.get("authorization") != "run_phase1295_once":
        raise RuntimeError("independent preaudit authorization missing")
    if COMPLETE.exists() or RAW.exists() or GENERATIONS.exists():
        raise RuntimeError("formal run budget already consumed or partial raw output exists")
    rows = read_jsonl(PARENT_MATERIAL)
    started = time.time()
    model = tokenizer = None
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(f"FP16 qualification failed: {qa}")
        candidate_rows = score_candidates(model, tokenizer, device, rows)
        write_jsonl(RAW, candidate_rows)
        generation_rows = generate_list_free(model, tokenizer, device, rows)
        write_jsonl(GENERATIONS, generation_rows)
        raw_hashes = {"candidate_scores": sha(RAW), "list_free_generations": sha(GENERATIONS)}
        thresholds = protocol["thresholds"]
        shortcut = float(protocol["zero_models"]["shortcut_ceiling"])
        candidate = candidate_summary(candidate_rows, thresholds, shortcut)
        generation = generation_summary(generation_rows, thresholds)
        passed = candidate["passed"] and generation["passed"]
        authorization = "phase1296_multievent_response_preregistration_only" if passed else "close_c030_without_hidden"
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "protocol_digest": protocol["protocol_digest"],
            "candidate": candidate,
            "generation": generation,
            "all_behavior_gates_passed": passed,
            "authorization": authorization,
            "raw_hashes": raw_hashes,
            "counts": {"candidate": len(candidate_rows), "generation": len(generation_rows)},
            "model_audit": qa,
            "placement": placement,
            "runtime_seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "hidden_states_read": False,
        }
        save(SUMMARY, summary)
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "verdict": "behavior_qualified" if passed else "behavior_gate_failed_c030_closed",
            "protocol_digest": protocol["protocol_digest"],
            "raw_hashes": raw_hashes,
            "all_behavior_gates_passed": passed,
            "authorization": authorization,
            "hidden_states_read": False,
        }
        save(FINAL, final)
        save(COMPLETE, {"completed_at_utc": utc_now(), "formal_runs_consumed": 1, "protocol_digest": protocol["protocol_digest"]})
        print(canonical({
            "verdict": final["verdict"], "candidate_accuracy": candidate["metrics"]["overall_candidate_accuracy"],
            "generation_accuracy": generation["metrics"]["exact_accuracy"], "authorization": authorization,
        }))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()


if __name__ == "__main__":
    main()
