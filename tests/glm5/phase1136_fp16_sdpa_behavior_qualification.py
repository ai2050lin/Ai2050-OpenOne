#!/usr/bin/env python3
"""One-shot FP16 SDPA qualification for the Phase1135 GLM4 behavior object."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1135_temporal_binding_intervention as base  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402
from phase1023_fp16_utils import parameter_dtype_counts, quantization_audit  # noqa: E402


PHASE = 1136
MODEL = "glm4"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1135_temporal_binding_intervention"
OUT_ROOT = ROOT / "tests/glm5/result/phase1136_fp16_sdpa_behavior_qualification"
GATED_STATES = base.GATED_STATES
THRESHOLDS = base.BEHAVIOR_THRESHOLDS


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def conditional_accuracy(rows: list[dict[str, Any]], split: str, state: str) -> float:
    selected = [row for row in rows if row["split"] == split and row["state"] == state and row["finite"]]
    return sum(bool(row["correct"]) for row in selected) / max(len(selected), 1)


def protocol_command() -> None:
    source_protocol = read_json(SOURCE_ROOT / "protocol/preregistration.json")
    source_behavior = read_json(SOURCE_ROOT / "analysis/behavior_authorization.json")
    glm_decisions = read_jsonl(SOURCE_ROOT / "analysis/behavior_decisions.glm4.jsonl")
    ds_decisions = read_jsonl(SOURCE_ROOT / "analysis/behavior_decisions.deepseek7b.jsonl")
    conditional = {
        model: {
            split: {
                state: conditional_accuracy(rows, split, state)
                for state in GATED_STATES
            }
            for split in ("discovery", "confirmation")
        }
        for model, rows in (("glm4", glm_decisions), ("deepseek7b", ds_decisions))
    }
    glm_eligible = all(
        conditional["glm4"][split][state] >= 0.90
        for split in conditional["glm4"]
        for state in GATED_STATES
    )
    ds_eligible = all(
        conditional["deepseek7b"][split][state] >= 0.90
        for split in conditional["deepseek7b"]
        for state in GATED_STATES
    )
    checks = {
        "phase1135_audit_passed": read_json(SOURCE_ROOT / "audit/independent_result_audit.json")["all_checks_passed"],
        "phase1135_hidden_denied": source_behavior["hidden_scan_authorized"] is False,
        "qwen3_only_authorized": source_behavior["authorized_models"] == ["qwen3"],
        "glm4_numeric_failure": source_behavior["models"]["glm4"]["splits"]["discovery"]["finite_fraction"] < 0.99,
        "glm4_conditional_behavior_eligible": glm_eligible,
        "deepseek7b_not_eligible": not ds_eligible,
    }
    protocol = {
        "schema_version": "phase1136_fp16_sdpa_preregistration.v2",
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase": 1135,
        "source_protocol_digest": source_protocol["protocol_digest"],
        "source_authorization_digest": source_behavior["authorization_digest"],
        "objective": "test whether a mathematically equivalent SDPA attention backend restores finite FP16 GLM4 scoring without changing weights, materials, prompts, or behavior thresholds",
        "authorized_model": MODEL,
        "excluded_model": "deepseek7b",
        "selection_rule": "numeric failure plus at least 0.90 finite-conditional accuracy in every frozen discovery/confirmation state",
        "conditional_accuracy_diagnostic": conditional,
        "weights_precision": "float16",
        "quantization": "none",
        "attention_backend": "sdpa",
        "sdpa_kernel_policy": "math_only",
        "engineering_revision": "Revision 1 produced no scores and crashed in the fused PyTorch c10 path; Revision 2 freezes the mathematically equivalent SDPA math kernel and permits no further backend retry",
        "smoke_item_ids": source_protocol["causal_items"],
        "smoke_gate": {
            "finite_fraction": 0.99,
            "state_accuracy": 0.80,
            "all_four_accuracy": 0.65,
            "required_splits": ["discovery", "confirmation"],
        },
        "full_gate": THRESHOLDS,
        "hard_stop": "if smoke fails, do not run full corpus; if full fails, do not authorize hidden intervention",
        "claim_boundary": "numerical backend qualification only; no semantic or mechanism claim",
    }
    protocol["protocol_digest"] = digest(protocol)
    audit = {
        "schema_version": "phase1136_protocol_audit.v1",
        "phase": PHASE,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "passed_count": sum(checks.values()),
        "check_count": len(checks),
        "protocol_digest": protocol["protocol_digest"],
    }
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1136 protocol audit failed: {checks}")
    print(json.dumps({"phase": PHASE, "protocol_audit": f"{audit['passed_count']}/{audit['check_count']}", "protocol_digest": protocol["protocol_digest"]}), flush=True)


def load_sdpa_glm4():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    path = MODEL_CONFIGS[MODEL]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_memory = {0: "11GiB", "cpu": "24GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    return model, tokenizer, device, {
        "placement": "accelerate_auto_cpu_gpu",
        "max_memory": {"cuda:0": "11GiB", "cpu": "24GiB"},
        "device_map": {str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()},
        "parameter_dtypes": parameter_dtype_counts(model),
        "requested_attention_backend": "sdpa",
        "effective_attention_backend": getattr(model.config, "_attn_implementation", None),
        "sdpa_kernel_policy": "math_only",
        "quantization": "none",
    }


def decisions(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        grouped[str(row["case_id"])][str(row["candidate_key"])] = row
    result = []
    for case_id, pair in sorted(grouped.items()):
        old, new = pair["old"], pair["new"]
        finite = bool(old["finite"] and new["finite"])
        margin = float(new["logp_mean"] - old["logp_mean"]) if finite else None
        expected = str(old["expected_key"])
        correct_margin = margin if expected == "new" and finite else (-margin if expected == "old" and finite else None)
        result.append({
            "case_id": case_id,
            "item_id": old["item_id"],
            "split": old["split"],
            "state": old["state"],
            "expected_key": expected,
            "finite": finite,
            "margin_new_minus_old": margin,
            "correct_margin": correct_margin,
            "correct": finite and correct_margin is not None and correct_margin > 0.0,
        })
    return result


def metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    state_metrics = {}
    for state in GATED_STATES:
        state_rows = [row for row in selected if row["state"] == state]
        finite_margins = [row["correct_margin"] for row in state_rows if row["correct_margin"] is not None]
        state_metrics[state] = {
            "count": len(state_rows),
            "finite_fraction": sum(row["finite"] for row in state_rows) / max(len(state_rows), 1),
            "accuracy": sum(bool(row["correct"]) for row in state_rows) / max(len(state_rows), 1),
            "median_correct_margin": statistics.median(finite_margins) if finite_margins else None,
        }
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_item[str(row["item_id"])][str(row["state"])] = row
    complete = [states for states in by_item.values() if all(state in states for state in GATED_STATES)]
    all_four = sum(all(states[state]["correct"] for state in GATED_STATES) for states in complete) / max(len(complete), 1)
    finite_fraction = sum(row["finite"] for row in selected) / max(len(selected), 1)
    passed = (
        finite_fraction >= 0.99
        and all(state_metrics[state]["accuracy"] >= 0.80 for state in GATED_STATES)
        and all(state_metrics[state]["median_correct_margin"] is not None and state_metrics[state]["median_correct_margin"] > 0.0 for state in GATED_STATES)
        and all_four >= 0.65
    )
    return {
        "split": split,
        "count": len(selected),
        "finite_fraction": finite_fraction,
        "state_metrics": state_metrics,
        "all_four_accuracy": all_four,
        "passed": passed,
    }


def run_command() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    logical = read_jsonl(SOURCE_ROOT / "protocol/logical_cases.jsonl")
    logical = [row for row in logical if row["state"] in GATED_STATES]
    smoke_ids = set(protocol["smoke_item_ids"]["discovery"]) | set(protocol["smoke_item_ids"]["confirmation"])
    smoke_cases = [row for row in logical if row["item_id"] in smoke_ids]
    model = None
    started = time.time()
    try:
        print(json.dumps({"phase": PHASE, "stage": "load_start", "backend": "sdpa_math"}), flush=True)
        model, tokenizer, device, placement = load_sdpa_glm4()
        print(json.dumps({"phase": PHASE, "stage": "load_complete", "effective_backend": placement["effective_attention_backend"]}), flush=True)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"FP16/no-quantization audit failed: {precision}")
        if placement["effective_attention_backend"] != "sdpa":
            raise RuntimeError(f"SDPA request not honored: {placement}")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def score_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            expanded = [base.tokenize_case(tokenizer, case, key) for case in cases for key in ("old", "new")]
            scored = base.score_rows(model, expanded, int(pad_id), device, base.BATCH_SIZE[MODEL])
            return scored, decisions(scored)

        print(json.dumps({"phase": PHASE, "stage": "smoke_start", "case_count": len(smoke_cases)}), flush=True)
        smoke_scores, smoke_decisions = score_cases(smoke_cases)
        print(json.dumps({"phase": PHASE, "stage": "smoke_scored", "score_count": len(smoke_scores)}), flush=True)
        smoke_metrics = {split: metrics(smoke_decisions, split) for split in ("discovery", "confirmation")}
        smoke_passed = all(row["passed"] for row in smoke_metrics.values())
        write_jsonl(OUT_ROOT / "smoke/scores.jsonl", smoke_scores)
        write_jsonl(OUT_ROOT / "smoke/decisions.jsonl", smoke_decisions)
        if smoke_passed:
            print(json.dumps({"phase": PHASE, "stage": "full_start", "case_count": len(logical)}), flush=True)
            full_scores, full_decisions = score_cases(logical)
            print(json.dumps({"phase": PHASE, "stage": "full_scored", "score_count": len(full_scores)}), flush=True)
            full_metrics = {split: metrics(full_decisions, split) for split in ("discovery", "confirmation", "natural_use")}
            full_passed = all(full_metrics[split]["passed"] for split in ("discovery", "confirmation"))
            write_jsonl(OUT_ROOT / "full/scores.jsonl", full_scores)
            write_jsonl(OUT_ROOT / "full/decisions.jsonl", full_decisions)
        else:
            full_scores, full_decisions, full_metrics, full_passed = [], [], {}, False
        qwen_authorized = read_json(SOURCE_ROOT / "analysis/behavior_authorization.json")["models"]["qwen3"]["authorized_for_hidden_scan"]
        authorized_models = ["qwen3", "glm4"] if qwen_authorized and full_passed else ["qwen3"] if qwen_authorized else []
        result = {
            "schema_version": "phase1136_fp16_sdpa_qualification.v1",
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "model": MODEL,
            "precision": precision,
            "placement": placement,
            "smoke_case_count": len(smoke_cases),
            "smoke_score_count": len(smoke_scores),
            "smoke_metrics": smoke_metrics,
            "smoke_passed": smoke_passed,
            "full_case_count": len(logical) if smoke_passed else 0,
            "full_score_count": len(full_scores),
            "full_metrics": full_metrics,
            "full_passed": full_passed,
            "authorized_models_for_new_causal_phase": authorized_models,
            "new_causal_phase_authorized": len(authorized_models) >= 2,
            "elapsed_seconds": time.time() - started,
            "claim_boundary": "backend qualification only; Phase1135 eager result remains unchanged",
            "human_annotation_eligible": False,
        }
        result["result_digest"] = digest(result)
        write_json(OUT_ROOT / "analysis/final.json", result)
        print(json.dumps({
            "phase": PHASE,
            "smoke_passed": smoke_passed,
            "full_passed": full_passed,
            "authorized_models_for_new_causal_phase": authorized_models,
            "new_causal_phase_authorized": result["new_causal_phase_authorized"],
            "elapsed_seconds": result["elapsed_seconds"],
            "result_digest": result["result_digest"],
        }), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def failure_command() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    result = {
        "schema_version": "phase1136_fp16_sdpa_qualification.v2",
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "model": MODEL,
        "engineering_failure": True,
        "failure_stage": "before_load_complete",
        "failure_signature": "Windows application error 1000; python.exe fault in torch c10.dll; exception 0xc0000005",
        "attempts": [
            {
                "revision": 1,
                "kernel_policy": "default_fused_selection",
                "scores_observed": 0,
                "outcome": "native c10.dll crash",
            },
            {
                "revision": 2,
                "kernel_policy": "math_only",
                "scores_observed": 0,
                "outcome": "native c10.dll crash before load_complete",
            },
        ],
        "smoke_case_count": 0,
        "smoke_score_count": 0,
        "smoke_metrics": {},
        "smoke_passed": False,
        "full_case_count": 0,
        "full_score_count": 0,
        "full_metrics": {},
        "full_passed": False,
        "authorized_models_for_new_causal_phase": ["qwen3"],
        "new_causal_phase_authorized": False,
        "next_action": "close local SDPA repair; do not run hidden intervention under a one-model gate",
        "claim_boundary": "engineering hard stop only; no GLM4 SDPA behavior result and no semantic or mechanism conclusion",
        "human_annotation_eligible": False,
    }
    result["result_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/final.json", result)
    print(json.dumps({
        "phase": PHASE,
        "engineering_failure": True,
        "new_causal_phase_authorized": False,
        "next_action": result["next_action"],
        "result_digest": result["result_digest"],
    }), flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("protocol", "run", "record-failure"):
        raise SystemExit("usage: phase1136_fp16_sdpa_behavior_qualification.py {protocol|run|record-failure}")
    if sys.argv[1] == "protocol":
        protocol_command()
    elif sys.argv[1] == "run":
        run_command()
    else:
        failure_command()


if __name__ == "__main__":
    main()
