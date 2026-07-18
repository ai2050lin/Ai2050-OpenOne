#!/usr/bin/env python3
"""Run one Phase500 staged behavior split for one CUDA model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402


PROTOCOL_DIR = ROOT / "tests" / "gpt5" / "result" / "phase500_native_relation_contract_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase500_frozen_contract.json"
AUDIT_PATH = PROTOCOL_DIR / "phase500_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests" / "gpt5" / "phase500_native_relation_contract_protocol.py"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "native_plain_candidate")
OBSERVERS = ("true_false", "mapped_ab", "mapped_01")
Z = 1.96

STAGES = {
    "calibration": {
        "phase": 501,
        "split": "function_polarity_calibration",
        "input": PROTOCOL_DIR / "phase500_function_polarity_calibration.jsonl",
        "authorization": None,
        "out": ROOT / "tests" / "gpt5" / "result" / "phase501_function_polarity_calibration",
    },
    "contract": {
        "phase": 503,
        "split": "vocab_observer_calibration",
        "input": PROTOCOL_DIR / "phase500_vocab_observer_calibration.jsonl",
        "authorization": ROOT / "tests" / "gpt5" / "result" / "phase502_staged_behavior_authorization" / "phase502_calibration_authorization.json",
        "out": ROOT / "tests" / "gpt5" / "result" / "phase503_vocab_observer_calibration",
    },
    "confirmation": {
        "phase": 505,
        "split": "independent_confirmation",
        "input": PROTOCOL_DIR / "phase500_independent_confirmation.jsonl",
        "authorization": ROOT / "tests" / "gpt5" / "result" / "phase504_staged_behavior_authorization" / "phase504_contract_authorization.json",
        "out": ROOT / "tests" / "gpt5" / "result" / "phase505_independent_confirmation",
    },
}


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


def rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(row[field]) for row in rows)
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def selection_for_stage(stage: str, model: str) -> list[dict[str, str]] | None:
    auth_path = STAGES[stage]["authorization"]
    if auth_path is None:
        return None
    authorization = load_json(auth_path)
    key = "stage_b_cells_by_model" if stage == "contract" else "stage_c_contracts_by_model"
    return list(authorization[key].get(model, []))


def sample_selected(sample: dict[str, Any], stage: str, selections: list[dict[str, str]] | None) -> bool:
    if selections is None:
        return True
    if stage == "contract":
        return any(
            sample["function_class"] == item["function_class"] and sample["polarity"] == item["polarity"]
            for item in selections
        )
    return any(
        sample["function_class"] == item["function_class"]
        and sample["polarity"] == item["polarity"]
        and sample["vocab_system"] == item["vocab_system"]
        for item in selections
    )


def flatten(samples: list[dict[str, Any]], stage: str, selections: list[dict[str, str]] | None) -> list[dict[str, Any]]:
    split = STAGES[stage]["split"]
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"Invalid sample in {split}")
        if not sample_selected(sample, stage, selections):
            continue
        for variant in sample["variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_pair_id": sample["source_pair_id"],
                "function_class": sample["function_class"],
                "polarity": sample["polarity"],
                "vocab_system": sample["vocab_system"],
                "truth_value": sample["truth_value"],
                "world_role": sample["world_role"],
                "length_control": sample["length_control"],
                "fact_order": sample["fact_order"],
                "surface": variant["surface"],
                "observer": variant["observer"],
                "mapping_flip": variant["mapping_flip"],
                "prompt": variant["prompt"],
                "true_candidate": variant["true_candidate"],
                "false_candidate": variant["false_candidate"],
            })
    return rows


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Candidate {text!r} is not one token: {ids}")
    return int(ids[0])


def score_rows(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]], batch_size: int) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["true_candidate"], row["false_candidate"])].append(row)
    tokenizer.padding_side = "left"
    completed = 0
    for (true_candidate, false_candidate), group in sorted(groups.items()):
        true_id = single_token_id(tokenizer, true_candidate)
        false_id = single_token_id(tokenizer, false_candidate)
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
            true_scores = logits[:, true_id].float().cpu()
            false_scores = logits[:, false_id].float().cpu()
            for index, row in enumerate(batch):
                margin = float(true_scores[index] - false_scores[index])
                prediction = margin > 0
                row.update({
                    "semantic_margin_true_minus_false": margin,
                    "semantic_prediction": prediction,
                    "correct": prediction == row["truth_value"],
                })
            completed += len(batch)
            if completed == len(rows) or completed % 512 < len(batch):
                log(f"candidate scoring {completed}/{len(rows)}")


def surface_report(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    by_surface = {surface: rate([row for row in rows if row["surface"] == surface], "correct") for surface in SURFACES}
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
        by_pair[row["source_pair_id"]].append(row)
    intersections = []
    for sample_id, items in by_sample.items():
        if len(items) != 2 or {item["surface"] for item in items} != set(SURFACES):
            raise RuntimeError(f"Incomplete surface intersection {sample_id}")
        intersections.append({"correct": all(item["correct"] for item in items)})
    pairs = []
    for pair_id, items in by_pair.items():
        if len(items) != 4 or {item["truth_value"] for item in items} != {False, True}:
            raise RuntimeError(f"Incomplete paired world {pair_id}")
        pairs.append({"correct": all(item["correct"] for item in items)})
    intersection = rate(intersections, "correct")
    paired = rate(pairs, "correct")
    passed = (
        by_surface["identity"]["lcb95"] >= gate["identity_lcb95_min"]
        and by_surface["native_plain_candidate"]["lcb95"] >= gate["native_plain_lcb95_min"]
        and intersection["lcb95"] >= gate["surface_intersection_lcb95_min"]
        and paired["lcb95"] >= gate["paired_world_lcb95_min"]
    )
    return {"by_surface": by_surface, "surface_intersection": intersection, "paired_world": paired, "gate_pass": passed}


def observer_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sample_id"], row["surface"])].append(row)
    records = []
    for key, items in grouped.items():
        if len(items) != len(OBSERVERS) or {item["observer"] for item in items} != set(OBSERVERS):
            raise RuntimeError(f"Incomplete observer group {key}")
        records.append({
            "consistent": len({item["semantic_prediction"] for item in items}) == 1,
            "consistent_and_correct": len({item["semantic_prediction"] for item in items}) == 1 and all(item["correct"] for item in items),
        })
    return {"semantic_consistency": rate(records, "consistent"), "consistent_and_correct": rate(records, "consistent_and_correct")}


def build_summary(stage: str, model_key: str, rows: list[dict[str, Any]], runtime: float, loaded: bool) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    gate = contract["behavior_gate"]
    cells = {}
    if stage == "calibration":
        keys = sorted({(row["function_class"], row["polarity"]) for row in rows})
        for function_class, polarity in keys:
            selected = [row for row in rows if row["function_class"] == function_class and row["polarity"] == polarity]
            report = surface_report(selected, gate)
            cells[f"{function_class}|{polarity}"] = {
                "function_class": function_class,
                "polarity": polarity,
                "vocab_system": "natural_names",
                "observer": "true_false",
                **report,
            }
        passed = [
            {"function_class": item["function_class"], "polarity": item["polarity"]}
            for item in cells.values() if item["gate_pass"]
        ]
        selection_key = "passed_function_polarity_cells"
    else:
        keys = sorted({(row["function_class"], row["polarity"], row["vocab_system"]) for row in rows})
        for function_class, polarity, vocab in keys:
            selected = [
                row for row in rows
                if row["function_class"] == function_class and row["polarity"] == polarity and row["vocab_system"] == vocab
            ]
            observers = {
                observer: surface_report([row for row in selected if row["observer"] == observer], gate)
                for observer in OBSERVERS
            }
            consistency = observer_consistency(selected)
            passed_cell = (
                all(payload["gate_pass"] for payload in observers.values())
                and consistency["semantic_consistency"]["lcb95"] >= gate["observer_consistency_lcb95_min"]
                and consistency["consistent_and_correct"]["lcb95"] >= gate["observer_consistency_lcb95_min"]
            )
            cells[f"{function_class}|{polarity}|{vocab}"] = {
                "function_class": function_class,
                "polarity": polarity,
                "vocab_system": vocab,
                "observers": observers,
                "observer_consistency": consistency,
                "gate_pass": passed_cell,
            }
        passed = [
            {"function_class": item["function_class"], "polarity": item["polarity"], "vocab_system": item["vocab_system"]}
            for item in cells.values() if item["gate_pass"]
        ]
        selection_key = "passed_native_contracts" if stage == "contract" else "confirmed_native_contracts"
    return {
        "schema_version": f"phase{STAGES[stage]['phase']}_staged_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if rows else "no_authorized_cells",
        "stage": stage,
        "split": STAGES[stage]["split"],
        "model": model_key,
        "cuda_used": loaded,
        "model_weights_loaded": loaded,
        "runtime_seconds": runtime,
        "row_count": len(rows),
        "sealed_split_read": False,
        "cells": cells,
        selection_key: passed,
    }


def verify(stage: str) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    audit = load_json(AUDIT_PATH)
    if audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase500 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase500 protocol source changed after freeze")
    split = STAGES[stage]["split"]
    if sha256_file(STAGES[stage]["input"]) != contract["split_files"][split]["sha256"]:
        raise RuntimeError(f"Phase500 {split} hash drift")
    auth_path = STAGES[stage]["authorization"]
    if auth_path is not None and not auth_path.exists():
        raise RuntimeError(f"Missing prior-stage authorization: {auth_path}")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    verify(args.stage)
    selections = selection_for_stage(args.stage, args.model)
    samples = load_jsonl(STAGES[args.stage]["input"])
    rows = flatten(samples, args.stage, selections)
    out_dir = STAGES[args.stage]["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    loaded = False
    started = time.monotonic()
    if rows:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase500 staged behavior requires CUDA")
        try:
            model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
            loaded = True
            score_rows(model, tokenizer, device, rows, args.batch_size)
        finally:
            if model is not None:
                release_model(model)
            gc.collect()
            torch.cuda.empty_cache()
    runtime = time.monotonic() - started
    for row in rows:
        row["model"] = args.model
        row.pop("prompt", None)
    phase = STAGES[args.stage]["phase"]
    rows_path = out_dir / f"phase{phase}_{args.model}_rows.jsonl"
    summary_path = out_dir / f"phase{phase}_{args.model}_summary.json"
    write_jsonl(rows_path, rows)
    summary = build_summary(args.stage, args.model, rows, runtime, loaded)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(rows_path)
    print(summary_path)


if __name__ == "__main__":
    main()
