#!/usr/bin/env python3
"""Phase485 GLM4 behavior gate for Phase484 core-surface protocol.

Runs open splits only. Sealed split is never read. This is a behavior
qualification gate before any Phase484 physical geometry collection.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase451_glm4_v2_pilot_behavior import classify, generate_batch, prompt_for, write_jsonl  # noqa: E402


PHASE484_DIR = ROOT / "tests" / "gpt5" / "result" / "phase484_core_surface_protocol"
SAMPLES_PATH = PHASE484_DIR / "phase484_core_surface_samples.jsonl"
AUDIT_PATH = PHASE484_DIR / "phase484_core_surface_static_audit.json"

OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase485_core_surface_behavior_gate"
GENERATIONS_PATH = OUT_DIR / "phase485_core_surface_behavior_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase485_core_surface_behavior_summary.json"

OPEN_SPLITS = {"geometry_window_freeze", "physical_prediction_holdout"}
SEALED_SPLIT = "sealed_physical_holdout"
Z = 1.96


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def iter_open_variants(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["sealed"] or sample["split"] == SEALED_SPLIT:
            continue
        if sample["split"] not in OPEN_SPLITS:
            continue
        for variant in sample["surface_variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_sample_id": sample["source_sample_id"],
                "source_pair_id": sample["source_pair_id"],
                "split": sample["split"],
                "pair_index": sample["pair_index"],
                "pair_role": sample["pair_role"],
                "subprotocol": sample["subprotocol"],
                "label_mapping": sample["label_mapping"],
                "truth_value": sample["truth_value"],
                "expected_label": sample["canonical_answer"],
                "variant_track": variant["track"],
                "variant_class": variant["variant_class"],
                "topology_signature": variant["topology_signature"],
                "text": variant["text"],
            })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = Counter((row["split"], row["variant_class"], row["classification"]) for row in rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for fields in (
            ("split", "variant_class"),
            ("split", "variant_track"),
            ("split", "subprotocol", "variant_class"),
            ("split", "truth_value", "variant_class"),
            ("split", "label_mapping", "variant_class"),
        ):
            groups[(fields, tuple(row[field] for field in fields))].append(row)
    group_reports = []
    for (fields, key), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(items)
        correct = sum(1 for item in items if item["classification"] == "semantic")
        other = sum(1 for item in items if item["classification"] == "other")
        lcb, ucb = wilson(correct, n)
        other_lcb, other_ucb = wilson(other, n)
        report = {field: value for field, value in zip(fields, key, strict=True)}
        report.update({
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "accuracy_lcb95": lcb,
            "accuracy_ucb95": ucb,
            "other": other,
            "other_rate": other / n if n else 0.0,
            "other_ucb95": other_ucb,
        })
        group_reports.append(report)
    core_rows = [row for row in rows if row["variant_class"] in {"identity", "core_surface"}]
    core_n = len(core_rows)
    core_correct = sum(1 for row in core_rows if row["classification"] == "semantic")
    core_other = sum(1 for row in core_rows if row["classification"] == "other")
    core_lcb, _core_ucb = wilson(core_correct, core_n)
    _other_lcb, core_other_ucb = wilson(core_other, core_n)
    core_gate_pass = core_lcb >= 0.90 and core_other_ucb <= 0.05
    return {
        "schema_version": "phase485_core_surface_behavior_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_gate_pass" if core_gate_pass else "behavior_gate_fail",
        "model": "glm4",
        "sealed_split_read": False,
        "open_variant_records": len(rows),
        "classification_counts": {str(key): value for key, value in counters.items()},
        "group_reports": group_reports,
        "core_behavior_gate": {
            "n": core_n,
            "correct": core_correct,
            "accuracy": core_correct / core_n if core_n else 0.0,
            "accuracy_lcb95": core_lcb,
            "other": core_other,
            "other_rate": core_other / core_n if core_n else 0.0,
            "other_ucb95": core_other_ucb,
            "pass": core_gate_pass,
            "criteria": "LCB95(core accuracy) >= 0.90 and UCB95(other rate) <= 0.05",
        },
        "authorization": {
            "physical_geometry_authorized": core_gate_pass,
            "sealed_split_read_authorized": False,
            "head_channel_neuron_scan_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-open-variants", type=int, default=0, help="Debug only; 0 runs all open variants.")
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase484 static audit has not passed; refusing behavior run.")
    samples = load_jsonl(SAMPLES_PATH)
    rows = iter_open_variants(samples)
    if args.max_open_variants:
        rows = rows[: args.max_open_variants]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model("glm4", use_8bit=args.use_8bit)
    out = []
    try:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [prompt_for(row["text"]) for row in batch]
            generations = generate_batch(model, tokenizer, device, prompts, max_new_tokens=4)
            for row, generated in zip(batch, generations, strict=True):
                item = dict(row)
                item["generated_text"] = generated
                item["normalized_generated"] = generated.strip()[:1].upper() if generated.strip() else ""
                item["classification"] = classify(row["expected_label"], generated)
                out.append(item)
            if len(out) % 256 == 0:
                print(f"[phase485] generated {len(out)}/{len(rows)}", flush=True)
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(GENERATIONS_PATH, out)
    SUMMARY_PATH.write_text(json.dumps(summarize(out), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(GENERATIONS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
