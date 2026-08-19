#!/usr/bin/env python3
"""Phase1332: frozen BF16 numerical qualification for C043."""
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
from phase1331_relational_measurement_core import canonical, load, rows, save, sha, write_rows  # noqa: E402
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16  # noqa: E402

PHASE, CAMPAIGN = 1332, "C043"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1332_c043_bf16_numeric_qualification_audit.py"
UTIL = T / "phase1332_bf16_utils.py"
PARENT = T / "result/phase1331_c043_native_relational_contract"
OUT = T / "result/phase1332_c043_bf16_numeric_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")


def parent_ok() -> dict[str, Any]:
    protocol = load(PARENT / "protocol/preregistration.json")
    audit = load(PARENT / "audit/independent_final_audit.json")
    if protocol["authorization"] != "run_phase1332_bf16_numeric_qualification" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1331 does not authorize Phase1332")
    return protocol


def prepare(force: bool) -> None:
    protocol = parent_ok()
    manifest = OUT / "protocol/execution_manifest.json"
    if manifest.exists() and not force:
        raise RuntimeError(f"{manifest} exists")
    if any((OUT / f"raw/{model}_scores.jsonl").exists() for model in MODELS):
        raise RuntimeError("cannot rewrite manifest after model output exists")
    frozen = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "phase1332.c043.bf16_numeric_qualification.v1",
        "parent_protocol_sha256": sha(PARENT / "protocol/preregistration.json"),
        "parent_contract_sha256": protocol["contract_sha256"],
        "model_order": list(MODELS),
        "precision": "bfloat16-no-quantization",
        "score_dtype": "float32_log_softmax",
        "runs": [
            {"name": "single", "batch_size": 1},
            {"name": "batch", "batch_size": 8},
            {"name": "batch_repeat", "batch_size": 8},
        ],
        "sentinel_case_ids": protocol["numeric"]["sentinel_case_ids"],
        "gate": protocol["numeric"]["gate"],
        "model_failure": protocol["numeric"]["failure"],
        "minimum_models_for_behavior": protocol["behavior"]["gate"]["minimum_authorized_models"],
        "overwrite_after_run": False,
        "script_sha256": sha(SCRIPT),
        "auditor_sha256": sha(AUDITOR),
        "util_sha256": sha(UTIL),
    }
    frozen["manifest_sha256"] = hashlib.sha256(canonical(frozen).encode()).hexdigest()
    frozen["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(manifest, frozen)
    print(json.dumps(frozen, indent=2))


def score_sequences(model, device, jobs: list[dict[str, Any]], pad_id: int, batch_size: int) -> list[float]:
    output: list[float] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        lengths = [len(job["sequence"]) for job in batch]
        width = max(lengths)
        input_ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long, device=device)
        attention = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for index, job in enumerate(batch):
            sequence = torch.tensor(job["sequence"], dtype=torch.long, device=device)
            input_ids[index, :len(sequence)] = sequence
            attention[index, :len(sequence)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        for index, job in enumerate(batch):
            prompt_len = job["prompt_len"]
            values = [
                float(log_probs[index, prompt_len + offset - 1, token].item())
                for offset, token in enumerate(job["candidate"])
            ]
            output.append(sum(values) / len(values))
        del input_ids, attention, logits, log_probs
    return output


def grouped(values: list[float]) -> list[list[float]]:
    return [values[index:index + 2] for index in range(0, len(values), 2)]


def run_model(model_name: str) -> None:
    if model_name not in MODELS:
        raise ValueError(model_name)
    protocol = parent_ok()
    manifest = load(OUT / "protocol/execution_manifest.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    if hashlib.sha256(canonical(frozen).encode()).hexdigest() != manifest["manifest_sha256"]:
        raise RuntimeError("execution manifest hash mismatch")
    result_path = OUT / f"analysis/{model_name}_summary.json"
    if result_path.exists():
        raise RuntimeError(f"formal result already exists: {result_path}")

    sentinel_ids = set(manifest["sentinel_case_ids"])
    source = [row for row in rows(PARENT / "material/frozen_behavior_cases.jsonl") if row["case_id"] in sentinel_ids]
    compiled = [row for row in rows(PARENT / f"compiled/{model_name}_behavior.jsonl") if row["case_id"] in sentinel_ids]
    if len(source) != len(compiled) or len(source) != manifest["gate"]["sentinel_case_count"] or any(
        left["case_id"] != right["case_id"] for left, right in zip(source, compiled)
    ):
        raise RuntimeError("compiled sentinel mismatch")

    print(f"[Phase1332] loading {model_name}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        jobs: list[dict[str, Any]] = []
        for row in compiled:
            for candidate in row["candidate_ids"]:
                jobs.append({
                    "sequence": row["prompt_ids"] + candidate,
                    "prompt_len": len(row["prompt_ids"]),
                    "candidate": candidate,
                })
        runs = {
            spec["name"]: grouped(score_sequences(model, device, jobs, int(pad_id), spec["batch_size"]))
            for spec in manifest["runs"]
        }
        raw = []
        finite_values = []
        rank_matches = []
        batch_diffs = []
        repeat_diffs = []
        for row, single, batch, repeat in zip(source, runs["single"], runs["batch"], runs["batch_repeat"]):
            finite = all(math.isfinite(value) for value in single + batch + repeat)
            finite_values.extend(single + batch + repeat)
            rank_match = (single[0] > single[1]) == (batch[0] > batch[1])
            rank_matches.append(rank_match)
            batch_abs = [abs(a - b) for a, b in zip(single, batch)]
            repeat_abs = [abs(a - b) for a, b in zip(batch, repeat)]
            batch_diffs.extend(batch_abs)
            repeat_diffs.extend(repeat_abs)
            raw.append({
                "case_id": row["case_id"],
                "single_scores": single,
                "batch_scores": batch,
                "batch_repeat_scores": repeat,
                "finite": finite,
                "rank_match": rank_match,
                "batch_abs_diff": batch_abs,
                "repeat_abs_diff": repeat_abs,
            })
        metrics = {
            "finite_fraction": sum(math.isfinite(value) for value in finite_values) / len(finite_values),
            "batch_rank_agreement": sum(rank_matches) / len(rank_matches),
            "batch_max_abs_score_diff": max(batch_diffs),
            "repeat_max_abs_score_diff": max(repeat_diffs),
            "sentinel_case_count": len(raw),
        }
        thresholds = manifest["gate"]
        gates = {
            "finite_fraction": metrics["finite_fraction"] >= thresholds["finite_fraction_min"],
            "batch_rank_agreement": metrics["batch_rank_agreement"] >= thresholds["batch_rank_agreement_min"],
            "batch_max_abs_score_diff": metrics["batch_max_abs_score_diff"] <= thresholds["batch_max_abs_score_diff_max"],
            "repeat_max_abs_score_diff": metrics["repeat_max_abs_score_diff"] <= thresholds["repeat_max_abs_score_diff_max"],
            "sentinel_case_count": metrics["sentinel_case_count"] == thresholds["sentinel_case_count"],
        }
        qualified = all(gates.values())
        write_rows(OUT / f"raw/{model_name}_scores.jsonl", raw)
        runtime = {
            "model": model_name,
            "device": str(device),
            "placement": placement,
            "quantization_audit": quantization_audit(model),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save(OUT / f"runtime/{model_name}.json", runtime)
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "model": model_name,
            "metrics": metrics,
            "gates": gates,
            "qualified": qualified,
            "raw_sha256": sha(OUT / f"raw/{model_name}_scores.jsonl"),
            "runtime_sha256": sha(OUT / f"runtime/{model_name}.json"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save(result_path, summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        print(f"[Phase1332] released {model_name}", flush=True)


def finalize() -> None:
    protocol = parent_ok()
    summaries = {model: load(OUT / f"analysis/{model}_summary.json") for model in MODELS}
    qualified = [model for model in MODELS if summaries[model]["qualified"]]
    passed = len(qualified) >= protocol["behavior"]["gate"]["minimum_authorized_models"]
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "qualified_models": qualified,
        "qualified_model_count": len(qualified),
        "all_gates_passed": passed,
        "authorization": "run_phase1333_bf16_behavior" if passed else "close_c043_numeric_ineligible",
        "model_summary_sha256": {
            model: sha(OUT / f"analysis/{model}_summary.json") for model in MODELS
        },
        "execution_manifest_sha256": sha(OUT / "protocol/execution_manifest.json"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(OUT / "analysis/final.json", final)
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
