#!/usr/bin/env python3
"""Phase1334: C044 BF16 score/margin execution-factorial qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16  # noqa: E402

PHASE, CAMPAIGN = 1334, "C044"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1334_c044_numeric_factorial_audit.py"
UTIL = T / "phase1332_bf16_utils.py"
PARENT = T / "result/phase1333_c044_relational_measurement_contract"
OUT = T / "result/phase1334_c044_numeric_factorial"
MODELS = ("qwen3", "glm4", "deepseek7b")


def parent_ok() -> dict[str, Any]:
    protocol = core.load(PARENT / "protocol/preregistration.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if protocol["authorization"] != "run_phase1334_c044_numeric_factorial" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1333 does not authorize Phase1334")
    return protocol


def prepare(force: bool) -> None:
    protocol = parent_ok()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists() and not force:
        raise RuntimeError(f"{path} exists")
    if any((OUT / f"raw/{model}_factorial.jsonl").exists() for model in MODELS):
        raise RuntimeError("cannot rewrite manifest after model output exists")
    case_ids = set(protocol["numeric"]["case_ids"])
    widths = {}
    cohort_ids = {}
    for model in MODELS:
        selected = [row for row in core.rows(PARENT / f"compiled/{model}_behavior.jsonl") if row["case_id"] in case_ids]
        widths[model] = max(len(row["prompt_ids"]) for row in selected)
        cohort_ids[model] = [[row["case_id"] for row in selected[index:index + 8]] for index in range(0, len(selected), 8)]
    frozen = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1334.c044.numeric_factorial.v1",
        "parent_protocol_sha256": core.sha(PARENT / "protocol/preregistration.json"),
        "parent_contract_sha256": protocol["contract_sha256"],
        "model_order": list(MODELS), "precision": "bfloat16-no-quantization",
        "score_dtype": "float32_log_softmax", "padding_side": "right",
        "explicit_position_ids": True, "fixed_width_by_model": widths,
        "conditions": protocol["numeric"]["conditions"],
        "case_ids": protocol["numeric"]["case_ids"], "cohorts_by_model": cohort_ids,
        "gate": protocol["numeric"]["gate"], "overwrite_after_run": False,
        "script_sha256": core.sha(SCRIPT), "auditor_sha256": core.sha(AUDITOR), "util_sha256": core.sha(UTIL),
    }
    frozen["manifest_sha256"] = core.digest(frozen)
    frozen["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(path, frozen)
    print(json.dumps(frozen, indent=2))


def score_batch(model, device, batch: list[dict[str, Any]], width: int, pad_id: int) -> list[list[float]]:
    input_ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(batch), width), dtype=torch.long, device=device)
    lengths = []
    for index, row in enumerate(batch):
        ids = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        input_ids[index, :len(ids)] = ids
        attention[index, :len(ids)] = 1
        lengths.append(len(ids))
    position_ids = attention.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=False).logits
        output = []
        for index, row in enumerate(batch):
            log_probs = torch.log_softmax(logits[index, lengths[index] - 1].float(), dim=-1)
            output.append([float(log_probs[candidate[0]].item()) for candidate in row["candidate_ids"]])
    del input_ids, attention, position_ids, logits
    return output


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def comparison(left: list[list[float]], right: list[list[float]]) -> dict[str, float]:
    absolute = []
    common = []
    margin_absolute = []
    margin_normalized = []
    ranks = []
    for a, b in zip(left, right):
        delta = [b[index] - a[index] for index in range(2)]
        absolute.extend(abs(value) for value in delta)
        common.append(abs(sum(delta) / 2))
        margin_a = a[0] - a[1]
        margin_b = b[0] - b[1]
        drift = abs(margin_b - margin_a)
        margin_absolute.append(drift)
        margin_normalized.append(drift / (1 + abs(margin_a)))
        ranks.append((a[0] > a[1]) == (b[0] > b[1]))
    return {
        "rank_agreement": sum(ranks) / len(ranks),
        "absolute_score_drift_max": max(absolute),
        "common_drift_p95": quantile(common, 0.95),
        "absolute_margin_drift_p95": quantile(margin_absolute, 0.95),
        "normalized_margin_drift_p95": quantile(margin_normalized, 0.95),
        "normalized_margin_drift_max": max(margin_normalized),
    }


def run_model(model_name: str) -> None:
    if model_name not in MODELS:
        raise ValueError(model_name)
    protocol = parent_ok()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    if core.digest(frozen) != manifest["manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    result_path = OUT / f"analysis/{model_name}_summary.json"
    if result_path.exists():
        raise RuntimeError(f"formal result exists: {result_path}")
    selected_ids = set(manifest["case_ids"])
    source = [row for row in core.rows(PARENT / "material/frozen_behavior_cases.jsonl") if row["case_id"] in selected_ids]
    compiled = [row for row in core.rows(PARENT / f"compiled/{model_name}_behavior.jsonl") if row["case_id"] in selected_ids]
    if len(source) != len(compiled) != 48 or any(a["case_id"] != b["case_id"] for a, b in zip(source, compiled)):
        raise RuntimeError("source/compiled mismatch")
    if len(source) != 48 or len(compiled) != 48:
        raise RuntimeError("numeric case count mismatch")

    print(f"[Phase1334] loading {model_name}", flush=True)
    model = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = int(manifest["fixed_width_by_model"][model_name])
        solo = [score_batch(model, device, [row], width, int(pad_id))[0] for row in compiled]
        replicated = [score_batch(model, device, [row] * 8, width, int(pad_id))[0] for row in compiled]
        cohort = []
        for start in range(0, len(compiled), 8):
            cohort.extend(score_batch(model, device, compiled[start:start + 8], width, int(pad_id)))
        cohort_repeat = []
        for start in range(0, len(compiled), 8):
            cohort_repeat.extend(score_batch(model, device, compiled[start:start + 8], width, int(pad_id)))
        raw = []
        for src, a, b, c, d in zip(source, solo, replicated, cohort, cohort_repeat):
            raw.append({"case_id": src["case_id"], "solo_fixed_width": a, "replicated_batch8": b,
                        "cohort_batch8": c, "cohort_batch8_repeat": d})
        all_scores = [value for row in raw for condition in manifest["conditions"] for value in row[condition]]
        shape = comparison(solo, replicated)
        composition = comparison(replicated, cohort)
        repeat_values = [abs(a - b) for left, right in zip(cohort, cohort_repeat) for a, b in zip(left, right)]
        metrics = {"finite_fraction": sum(math.isfinite(value) for value in all_scores) / len(all_scores),
                   "shape": shape, "composition": composition,
                   "repeat_max_abs_score_diff": max(repeat_values), "case_count": len(raw)}
        threshold = manifest["gate"]
        gates = {
            "finite_fraction": metrics["finite_fraction"] >= threshold["finite_fraction_min"],
            "shape_rank_agreement": shape["rank_agreement"] >= threshold["shape_rank_agreement_min"],
            "composition_rank_agreement": composition["rank_agreement"] >= threshold["composition_rank_agreement_min"],
            "shape_margin_p95": shape["normalized_margin_drift_p95"] <= threshold["shape_normalized_margin_drift_p95_max"],
            "composition_margin_p95": composition["normalized_margin_drift_p95"] <= threshold["composition_normalized_margin_drift_p95_max"],
            "shape_margin_max": shape["normalized_margin_drift_max"] <= threshold["shape_normalized_margin_drift_max"],
            "composition_margin_max": composition["normalized_margin_drift_max"] <= threshold["composition_normalized_margin_drift_max"],
            "repeat": metrics["repeat_max_abs_score_diff"] <= threshold["repeat_max_abs_score_diff_max"],
            "case_count": metrics["case_count"] == 48,
        }
        qualified = all(gates.values())
        core.write_rows(OUT / f"raw/{model_name}_factorial.jsonl", raw)
        runtime = {"model": model_name, "device": str(device), "placement": placement,
                   "quantization_audit": quantization_audit(model),
                   "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        core.save(OUT / f"runtime/{model_name}.json", runtime)
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "metrics": metrics,
                   "gates": gates, "qualified": qualified,
                   "raw_sha256": core.sha(OUT / f"raw/{model_name}_factorial.jsonl"),
                   "runtime_sha256": core.sha(OUT / f"runtime/{model_name}.json"),
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        core.save(result_path, summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        print(f"[Phase1334] released {model_name}", flush=True)


def finalize() -> None:
    protocol = parent_ok()
    summaries = {model: core.load(OUT / f"analysis/{model}_summary.json") for model in MODELS}
    qualified = [model for model in MODELS if summaries[model]["qualified"]]
    passed = len(qualified) >= protocol["numeric"]["gate"]["minimum_authorized_models"]
    final = {"phase": PHASE, "campaign": CAMPAIGN, "qualified_models": qualified,
             "qualified_model_count": len(qualified), "all_gates_passed": passed,
             "authorization": "run_phase1335_c044_multi_interface_behavior" if passed else "close_c044_numeric_factorial",
             "model_summary_sha256": {model: core.sha(OUT / f"analysis/{model}_summary.json") for model in MODELS},
             "manifest_sha256": core.sha(OUT / "protocol/execution_manifest.json"),
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.force)
    elif args.model:
        run_model(args.model)
    else:
        finalize()
