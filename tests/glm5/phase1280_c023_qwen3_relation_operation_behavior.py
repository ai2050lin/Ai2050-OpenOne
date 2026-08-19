#!/usr/bin/env python3
"""Phase1280: Qwen3 FP16 behavior gate for C023 relation operations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402


PHASE = 1280
CAMPAIGN = "C023"
CONTRACT_ID = "EXP-C023-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1280_c023_qwen3_relation_operation_behavior_audit.py"
INPUT = ROOT / "tests/glm5/result/phase1279_c023_relation_operation_behavior_contract"
INPUT_PROTOCOL = INPUT / "protocol/preregistration.json"
INPUT_MATERIAL = INPUT / "material/frozen_relation_worlds.jsonl"
INPUT_FINAL = INPUT / "analysis/final.json"
INPUT_AUDIT = INPUT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1280_c023_qwen3_relation_operation_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
RUN_SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

OPERATIONS = ("contrast", "addition", "cause", "sequence")
PANELS = ("base", "target", "wrong", "null", "joint", "surface", "implicit")
FACTORIAL = ("base", "target", "wrong", "null", "joint")
BATCH_SIZE = 16


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


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
    parent_final = read_json(INPUT_FINAL)
    parent_audit = read_json(INPUT_AUDIT)
    if parent_final.get("authorization") != "phase1280_qwen3_behavior_only" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1279 authorization missing")
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1280.c023.qwen.behavior.v1",
        "model": {"name": "qwen3", "path": MODEL_CONFIGS["qwen3"]["path"], "precision": "fp16_cuda_no_quantization"},
        "row_count": parent_final["row_count"],
        "prompt_count": parent_final["prompt_count"],
        "operations": list(OPERATIONS),
        "panels": list(PANELS),
        "factorial_panels": list(FACTORIAL),
        "thresholds": parent["thresholds"],
        "batch_size": BATCH_SIZE,
        "dependencies": {
            "phase1279_protocol": file_sha256(INPUT_PROTOCOL),
            "phase1279_material": file_sha256(INPUT_MATERIAL),
            "phase1279_final": file_sha256(INPUT_FINAL),
            "phase1279_audit": file_sha256(INPUT_AUDIT),
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "formal_run_budget": 1,
        "hard_stops": [
            "Every frozen prompt is scored before a behavior verdict.",
            "Candidate scores are next-token FP32 views of an FP16 CUDA forward pass; no free generation enters the gate.",
            "No prompt, candidate, partition, threshold or denominator changes after preregistration.",
            "Behavior failure denies all hidden-state capture and intervention.",
            "Behavior success authorizes only Qwen3 typed causal testing; it does not establish abstract relation semantics.",
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


def encode_batch(tokenizer: Any, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    maximum = max(map(len, values))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(values), maximum), int(pad), dtype=torch.long)
    mask = torch.zeros((len(values), maximum), dtype=torch.long)
    for index, row in enumerate(values):
        ids[index, -len(row):] = torch.tensor(row, dtype=torch.long)
        mask[index, -len(row):] = 1
    return ids.to(device), mask.to(device)


def score_panel(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]], panel: str, candidate_ids: torch.Tensor) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        ids, mask = encode_batch(tokenizer, [row["panels"][panel] for row in batch], device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()
        scores = logits.index_select(-1, candidate_ids).detach().cpu().numpy()
        for row, values in zip(batch, scores):
            expected = row["expected"][panel]
            gold_index = OPERATIONS.index(expected)
            prediction_index = int(np.argmax(values))
            alternatives = np.delete(values, gold_index)
            output.append({
                "row_id": row["row_id"],
                "partition": row["partition"],
                "panel": panel,
                "expected": expected,
                "content_operation": row["content_operation"],
                "prediction": OPERATIONS[prediction_index],
                "correct": prediction_index == gold_index,
                "scores": {operation: float(values[index]) for index, operation in enumerate(OPERATIONS)},
                "gold_margin": float(values[gold_index] - float(np.max(alternatives))),
                "finite": bool(np.isfinite(values).all()),
            })
    return output


def summarize(scored: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    partitions = ("discovery", "selection", "confirmation")
    cells: dict[str, float] = {}
    margins: dict[str, float] = {}
    for partition in partitions:
        for panel in PANELS:
            subset = [row for row in scored if row["partition"] == partition and row["panel"] == panel]
            cells[f"{partition}.{panel}"] = float(np.mean([row["correct"] for row in subset]))
            margins[f"{partition}.{panel}"] = float(np.median([row["gold_margin"] for row in subset]))
    operation_macro = {
        operation: float(np.mean([row["correct"] for row in scored if row["expected"] == operation]))
        for operation in OPERATIONS
    }
    finite_fraction = float(np.mean([row["finite"] for row in scored]))
    factorial_min = min(value for key, value in cells.items() if key.split(".")[1] in FACTORIAL)
    surface_min = min(value for key, value in cells.items() if key.endswith(".surface"))
    implicit_min = min(value for key, value in cells.items() if key.endswith(".implicit"))
    median_margin = float(np.median([row["gold_margin"] for row in scored]))
    gates = {
        "finite": finite_fraction >= thresholds["candidate_finite_fraction_min"],
        "factorial": factorial_min >= thresholds["factorial_cell_accuracy_min"],
        "surface": surface_min >= thresholds["surface_cell_accuracy_min"],
        "implicit": implicit_min >= thresholds["implicit_cell_accuracy_min"],
        "operation_macro": min(operation_macro.values()) >= thresholds["operation_macro_accuracy_min"],
        "margin": median_margin >= thresholds["gold_margin_median_min"],
    }
    return {
        "prompt_count": len(scored),
        "cell_accuracy": cells,
        "cell_median_gold_margin": margins,
        "operation_macro_accuracy": operation_macro,
        "candidate_finite_fraction": finite_fraction,
        "factorial_minimum_accuracy": factorial_min,
        "surface_minimum_accuracy": surface_min,
        "implicit_minimum_accuracy": implicit_min,
        "overall_median_gold_margin": median_margin,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_model() -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal run already complete")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit missing")
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(INPUT_MATERIAL)
    started = time.perf_counter()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if device.type != "cuda" or precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError(f"FP16 qualification failed: {precision}")
        candidate_map = read_json(INPUT_PROTOCOL)["token_audit"]["candidate_token_ids"]
        candidate_ids = torch.tensor([candidate_map[operation] for operation in OPERATIONS], dtype=torch.long, device=device)
        scored: list[dict[str, Any]] = []
        with torch.inference_mode():
            for panel in PANELS:
                scored.extend(score_panel(model, tokenizer, device, rows, panel, candidate_ids))
                print(canonical_json({"panel_complete": panel, "scored": len(scored)}), flush=True)
        behavior = summarize(scored, protocol["thresholds"])
        write_jsonl(RAW, scored)
        runtime = time.perf_counter() - started
        run_summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "runtime_seconds": runtime,
            "precision_audit": precision, "placement": placement, "behavior": behavior,
        }
        atomic_json(RUN_SUMMARY, run_summary)
        verdict = "qwen3_relation_operation_behavior_qualified" if behavior["passed"] else "qwen3_relation_operation_behavior_gate_failed"
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "verdict": verdict,
            "behavior": behavior,
            "precision_audit": precision,
            "runtime_seconds": runtime,
            "authorization": "phase1281_qwen3_typed_causal_closure" if behavior["passed"] else "stop_c023_at_behavior_object",
            "scope": "Qwen3-4B FP16 next-token candidate scoring over the frozen C023 English relation-operation contract",
        }
        atomic_json(FINAL, final)
        atomic_json(COMPLETE, {"phase": PHASE, "completed_at_utc": utc_now(), "raw_sha256": file_sha256(RAW), "final_sha256": file_sha256(FINAL)})
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
    if args.action == "preregister":
        preregister(args.force)
    else:
        run_model()
