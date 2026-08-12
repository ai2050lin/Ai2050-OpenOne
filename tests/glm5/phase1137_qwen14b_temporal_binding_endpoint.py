#!/usr/bin/env python3
"""Prospective Qwen3-14B endpoint for the frozen Phase1135 temporal task.

This phase does not reopen or amend the Phase1135 three-model gate. It asks a
new, narrower question: does the already audited Qwen3-14B FP16 endpoint repeat
the Qwen3-4B four-state behavior on exactly the same machine-consensus carrier?
No hidden-state result is produced in this phase.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Iterable

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import quantization_audit  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402


PHASE = 1137
MODEL_NAME = "qwen3_14b"
MODEL_ROOT = ROOT / "models" / "hf" / "Qwen3-14B"
OUT_ROOT = ROOT / "tests/glm5/result/phase1137_qwen14b_temporal_binding_endpoint"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1135_temporal_binding_intervention"
PHASE1118_ROOT = ROOT / "tests/glm5/result/phase1118_qwen3_14b_fp16_offload_smoke"
PHASE1119_ROOT = ROOT / "tests/glm5/result/phase1119_qwen3_4b_14b_scale"

EXPECTED_PARAMETER_COUNT = 14_768_307_200
EXPECTED_MANIFEST_DIGEST = "92ae3a1b0dbf063ac1ecaca5bab4d95f32a52fe1de19277c418d42866974417c"
EXPECTED_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
BATCH_SIZE = 16
BUCKET_WIDTH = 8
SECONDARY_INTERACTION_THRESHOLDS = {
    "median_min": 0.0,
    "positive_fraction_min": 0.95,
    "required_splits": ("discovery", "confirmation", "natural_use"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def frozen_sources() -> dict[str, Any]:
    source_prereg = read_json(SOURCE_ROOT / "protocol/preregistration.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_result_audit.json")
    source_authorization = read_json(SOURCE_ROOT / "analysis/behavior_authorization.json")
    source_qwen_summary = read_json(SOURCE_ROOT / "behavior/qwen3/summary.json")
    phase1118_protocol = read_json(PHASE1118_ROOT / "protocol/protocol.json")
    phase1118_audit = read_json(PHASE1118_ROOT / "audit/result_audit.json")
    phase1119_prereg = read_json(PHASE1119_ROOT / "protocol/preregistration.json")
    phase1119_audit = read_json(PHASE1119_ROOT / "audit/result_audit.json")
    return {
        "source_prereg": source_prereg,
        "source_audit": source_audit,
        "source_authorization": source_authorization,
        "source_qwen_summary": source_qwen_summary,
        "phase1118_protocol": phase1118_protocol,
        "phase1118_audit": phase1118_audit,
        "phase1119_prereg": phase1119_prereg,
        "phase1119_audit": phase1119_audit,
    }


def protocol_command() -> None:
    behavior_summary = OUT_ROOT / "behavior/qwen3_14b/summary.json"
    if behavior_summary.exists():
        raise RuntimeError("refusing to rewrite the protocol after Qwen3-14B output exists")

    frozen = frozen_sources()
    logical_cases = read_jsonl(SOURCE_ROOT / "protocol/logical_cases.jsonl")
    source_score_path = SOURCE_ROOT / "behavior/qwen3/scores.jsonl"
    source_audit_path = SOURCE_ROOT / "audit/independent_result_audit.json"
    qwen4_tokenizer = ROOT / "models/hf/qwen3-4b/tokenizer.json"
    qwen14_tokenizer = MODEL_ROOT / "tokenizer.json"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = [
        source.tokenize_case(tokenizer, case, candidate_key)
        for case in logical_cases
        for candidate_key in ("old", "new")
    ]
    tokenized.sort(key=lambda row: (str(row["case_id"]), str(row["candidate_key"])))
    lengths = sorted(len(row["input_ids"]) for row in tokenized)

    checks = {
        "phase1135_protocol_identity": frozen["source_prereg"]["protocol_digest"]
        == "aa3ca4b6750b2e4df1917dfa2920ba7b6c2ca50de3efc6e53b11d79e7e883f7b",
        "phase1135_audit_passed": bool(frozen["source_audit"]["all_checks_passed"]),
        "phase1135_hidden_remains_denied": frozen["source_authorization"]["hidden_scan_authorized"] is False,
        "phase1135_only_qwen4_authorized": frozen["source_authorization"]["authorized_models"] == ["qwen3"],
        "phase1135_qwen4_summary_intact": digest({
            key: value
            for key, value in frozen["source_qwen_summary"].items()
            if key != "summary_digest"
        }) == frozen["source_qwen_summary"]["summary_digest"],
        "source_count_491": frozen["source_prereg"]["source_count"] == 491,
        "logical_case_count_2946": len(logical_cases) == 2946,
        "candidate_score_count_5892": len(tokenized) == 5892,
        "machine_only_scope_preserved": frozen["source_prereg"]["human_annotation_eligible"] is False,
        "phase1118_engineering_audit_passed": bool(frozen["phase1118_audit"]["all_checks_passed"]),
        "phase1119_result_audit_passed": bool(frozen["phase1119_audit"]["all_checks_passed"]),
        "qwen14_parameter_count_frozen": int(
            frozen["phase1119_prereg"]["expected_parameter_counts"][MODEL_NAME]
        ) == EXPECTED_PARAMETER_COUNT,
        "qwen14_manifest_frozen": frozen["phase1119_prereg"]["model_manifest_digests"][MODEL_NAME]
        == EXPECTED_MANIFEST_DIGEST,
        "qwen_tokenizers_identical": sha256_file(qwen4_tokenizer)
        == sha256_file(qwen14_tokenizer)
        == EXPECTED_TOKENIZER_SHA256,
        "no_qwen14_outputs_before_freeze": not behavior_summary.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1137 protocol audit failed: {checks}")

    case_digest = digest(logical_cases)
    carrier_digest = digest(tokenized)
    device_map = frozen["phase1118_protocol"]["device_map"]
    prereg_core = {
        "schema_version": "phase1137_qwen14b_temporal_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "Prospectively test Qwen3-14B FP16 on the unchanged Phase1135 four-state temporal carrier. "
            "This is a same-family endpoint and cannot reopen the frozen Phase1135 cross-model gate."
        ),
        "source": {
            "phase1135_protocol_digest": frozen["source_prereg"]["protocol_digest"],
            "phase1135_authorization_digest": frozen["source_authorization"]["authorization_digest"],
            "phase1135_audit_file_sha256": sha256_file(source_audit_path),
            "phase1135_qwen4_summary_digest": frozen["source_qwen_summary"]["summary_digest"],
            "phase1135_qwen4_scores_file_sha256": sha256_file(source_score_path),
            "logical_case_digest": case_digest,
            "logical_case_count": len(logical_cases),
            "evidence_scope": "external_machine_consensus_not_human_gold",
        },
        "model": {
            "name": MODEL_NAME,
            "repo": frozen["phase1118_protocol"]["repo"],
            "commit": frozen["phase1118_protocol"]["expected_commit"],
            "manifest_digest": EXPECTED_MANIFEST_DIGEST,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "precision": "fp16",
            "quantization": "none",
            "placement": "frozen Phase1118 CUDA-plus-disk map",
            "device_map": device_map,
        },
        "carrier": {
            "candidate_case_digest": carrier_digest,
            "candidate_score_count": len(tokenized),
            "minimum_tokens": lengths[0],
            "median_tokens": lengths[len(lengths) // 2],
            "p95_tokens": lengths[int(0.95 * len(lengths))],
            "maximum_tokens": lengths[-1],
            "batch_size": BATCH_SIZE,
            "bucket_width": BUCKET_WIDTH,
            "states": list(source.BEHAVIOR_STATES),
            "gated_states": list(source.GATED_STATES),
            "splits": ["discovery", "confirmation", "natural_use"],
        },
        "primary_behavior_thresholds": dict(source.BEHAVIOR_THRESHOLDS),
        "secondary_interaction_thresholds": {
            **SECONDARY_INTERACTION_THRESHOLDS,
            "required_splits": list(SECONDARY_INTERACTION_THRESHOLDS["required_splits"]),
            "authorization_role": "prospective secondary replication; cannot authorize hidden alone",
        },
        "predictions": {
            "P1": "all source, model, tokenizer, carrier, and no-output-before-freeze checks pass",
            "P2": "Qwen3-14B produces finite FP16 candidate scores at rate at least 0.99",
            "P3": "Qwen3-14B passes every unchanged Phase1135 discovery and confirmation behavior gate",
            "P4": "the prospective binding interaction has positive median and at least 0.95 positive fraction in every split",
            "P5": "Qwen3-4B and Qwen3-14B form a same-family behavior replication if and only if P3 and P4 pass",
            "P6": "Phase1135 remains closed and no cross-architecture or causal claim is made",
        },
        "hard_stops": [
            "do not edit or recalculate Phase1135 as a four-model gate",
            "do not count Qwen3-14B as cross-architecture replication",
            "do not use BF16, FP32, quantization, changed materials, changed states, or relaxed thresholds",
            "do not inspect hidden states, attention, MLP, heads, SAE, or neurons in Phase1137",
            "if either primary behavior or prospective interaction replication fails, stop this endpoint without panel deletion",
            "if both pass, only a separately frozen same-family causal phase may be designed",
            "machine consensus cannot be upgraded to human gold",
        ],
        "auto_continue_rule": (
            "authorize a separate, prospectively frozen same-family causal protocol only when P3 and P4 both pass; "
            "otherwise stop"
        ),
        "model_outputs_read_before_protocol": False,
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    audit_core = {
        "schema_version": "phase1137_qwen14b_temporal_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_jsonl(OUT_ROOT / "protocol/logical_cases.jsonl", logical_cases)
    write_jsonl(OUT_ROOT / "protocol/cases.qwen3_14b.jsonl", tokenized)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "command": "protocol",
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "protocol_digest": prereg["protocol_digest"],
        "candidate_scores": len(tokenized),
        "token_length_max": lengths[-1],
    }, ensure_ascii=False), flush=True)


def load_model(prereg: dict[str, Any]) -> tuple[Any, Any, dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(MODEL_ROOT, local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch.float16,
            trust_remote_code=True,
        )
    model.tie_weights()
    device_map = {
        key: (int(value) if str(value).isdigit() else value)
        for key, value in prereg["model"]["device_map"].items()
    }
    offload_root = OUT_ROOT / "offload/qwen3_14b"
    offload_root.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(MODEL_ROOT),
        device_map=device_map,
        no_split_module_classes=list(model._no_split_modules),
        offload_folder=str(offload_root),
        offload_buffers=False,
        dtype=torch.float16,
        offload_state_dict=True,
        force_hooks=True,
        strict=True,
    )
    model.eval()
    return model, tokenizer, {str(key): str(value) for key, value in model.hf_device_map.items()}


def behavior_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    audit = read_json(OUT_ROOT / "protocol/audit.json")
    rows = read_jsonl(OUT_ROOT / "protocol/cases.qwen3_14b.jsonl")
    if not audit["all_checks_passed"] or audit["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1137 protocol is not authorized")
    if digest(rows) != prereg["carrier"]["candidate_case_digest"]:
        raise RuntimeError("Phase1137 carrier digest mismatch")

    started = time.time()
    model = None
    try:
        model, tokenizer, device_map = load_model(prereg)
        precision = quantization_audit(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("Qwen3-14B parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"Phase1137 FP16/no-quantization audit failed: {precision}")

        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            width = ((len(row["input_ids"]) + BUCKET_WIDTH - 1) // BUCKET_WIDTH) * BUCKET_WIDTH
            buckets[width].append(row)

        torch.cuda.reset_peak_memory_stats()
        scores: list[dict[str, Any]] = []
        batch_seconds: list[float] = []
        completed = 0
        with torch.inference_mode():
            for bucket in sorted(buckets):
                panel = buckets[bucket]
                for start in range(0, len(panel), BATCH_SIZE):
                    batch = panel[start : start + BATCH_SIZE]
                    input_ids, attention_mask = source.pad_sequences(
                        batch,
                        int(tokenizer.pad_token_id),
                        torch.device("cuda:0"),
                    )
                    before = time.time()
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    scored = source.scores_from_logits(output.logits, batch)
                    batch_seconds.append(time.time() - before)
                    for row, score in zip(batch, scored):
                        finite = bool(score["finite"])
                        scores.append({
                            **{
                                key: value
                                for key, value in row.items()
                                if key not in ("input_ids", "prompt_ids", "continuation_ids")
                            },
                            "token_count": int(score["token_count"]),
                            "logp_sum": float(score["logp_sum"]) if finite else None,
                            "logp_mean": float(score["logp_mean"]) if finite else None,
                            "finite": finite,
                        })
                    completed += len(batch)
                    print(json.dumps({
                        "phase": PHASE,
                        "model": MODEL_NAME,
                        "completed": completed,
                        "total": len(rows),
                        "bucket": bucket,
                    }), flush=True)
                    del output, scored, input_ids, attention_mask

        scores.sort(key=lambda row: (str(row["case_id"]), str(row["candidate_key"])))
        finite_count = sum(bool(row["finite"]) for row in scores)
        core = {
            "schema_version": "phase1137_qwen14b_temporal_behavior_summary.v1",
            "phase": PHASE,
            "model": MODEL_NAME,
            "protocol_digest": prereg["protocol_digest"],
            "carrier_digest": prereg["carrier"]["candidate_case_digest"],
            "model_manifest_digest": prereg["model"]["manifest_digest"],
            "candidate_score_count": len(scores),
            "finite_count": finite_count,
            "finite_fraction": finite_count / max(len(scores), 1),
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": "cuda_disk_offload",
            "device_map": device_map,
            "batch_size": BATCH_SIZE,
            "bucket_width": BUCKET_WIDTH,
            "batch_count": len(batch_seconds),
            "batch_seconds": batch_seconds,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.time() - started,
            "score_digest": digest(scores),
            "evidence_scope": "same_family_machine_consensus_endpoint",
        }
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        write_jsonl(OUT_ROOT / "behavior/qwen3_14b/scores.jsonl", scores)
        write_json(OUT_ROOT / "behavior/qwen3_14b/summary.json", summary)
        print(json.dumps({
            "phase": PHASE,
            "command": "behavior",
            "scores": len(scores),
            "finite_fraction": summary["finite_fraction"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def decisions_from_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        grouped[str(row["case_id"])][str(row["candidate_key"])] = row
    decisions = []
    for case_id, candidates in sorted(grouped.items()):
        if set(candidates) != {"old", "new"}:
            raise RuntimeError(f"candidate coverage drift for {case_id}")
        old = candidates["old"]
        new = candidates["new"]
        finite = bool(old["finite"] and new["finite"])
        margin = float(new["logp_mean"] - old["logp_mean"]) if finite else None
        expected = old["expected_key"]
        if expected == "new":
            correct_margin = margin if finite else None
            correct = finite and margin is not None and margin > 0.0
        elif expected == "old":
            correct_margin = -margin if finite and margin is not None else None
            correct = finite and margin is not None and margin < 0.0
        else:
            correct_margin = None
            correct = None
        decisions.append({
            "schema_version": "phase1137_qwen14b_temporal_decision.v1",
            "phase": PHASE,
            "model": MODEL_NAME,
            "case_id": case_id,
            "item_id": old["item_id"],
            "split": old["split"],
            "property_id": old["property_id"],
            "domain": old["domain"],
            "state": old["state"],
            "expected_key": expected,
            "candidate_order": old["candidate_order"],
            "finite": finite,
            "old_logp_mean": old["logp_mean"] if old["finite"] else None,
            "new_logp_mean": new["logp_mean"] if new["finite"] else None,
            "margin_new_minus_old": margin,
            "correct_margin": correct_margin,
            "correct": correct,
        })
    return decisions


def finalize_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    summary = read_json(OUT_ROOT / "behavior/qwen3_14b/summary.json")
    scores = read_jsonl(OUT_ROOT / "behavior/qwen3_14b/scores.jsonl")
    if summary["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1137 summary protocol drift")
    if digest(scores) != summary["score_digest"]:
        raise RuntimeError("Phase1137 score digest mismatch")

    decisions = decisions_from_scores(scores)
    metrics = {
        split: source.behavior_metrics(decisions, split)
        for split in ("discovery", "confirmation", "natural_use")
    }
    primary_passed = all(
        metrics[split]["passed"]
        for split in source.BEHAVIOR_THRESHOLDS["required_splits"]
    )
    interaction_split_pass = {
        split: bool(
            metrics[split]["posthoc_binding_interaction"]["median"] is not None
            and metrics[split]["posthoc_binding_interaction"]["median"]
            > SECONDARY_INTERACTION_THRESHOLDS["median_min"]
            and metrics[split]["posthoc_binding_interaction"]["positive_fraction"]
            >= SECONDARY_INTERACTION_THRESHOLDS["positive_fraction_min"]
        )
        for split in SECONDARY_INTERACTION_THRESHOLDS["required_splits"]
    }
    interaction_passed = all(interaction_split_pass.values())
    same_family_replication = bool(primary_passed and interaction_passed)

    source_authorization = read_json(SOURCE_ROOT / "analysis/behavior_authorization.json")
    core = {
        "schema_version": "phase1137_qwen14b_temporal_final.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model": MODEL_NAME,
        "metrics": metrics,
        "primary_behavior_passed": primary_passed,
        "interaction_split_pass": interaction_split_pass,
        "prospective_interaction_passed": interaction_passed,
        "source_qwen4_authorized": source_authorization["authorized_models"] == ["qwen3"],
        "same_family_behavior_replication": same_family_replication,
        "phase1135_gate_reopened": False,
        "cross_architecture_replication": False,
        "hidden_scanned": False,
        "same_family_causal_protocol_authorized": same_family_replication,
        "auto_continue": same_family_replication,
        "next_action": (
            "freeze a separate same-family residual intervention protocol; do not claim cross-architecture conservation"
            if same_family_replication
            else "stop the Qwen3-14B temporal endpoint without hidden-state search"
        ),
        "evidence_scope": "same_family_machine_consensus_behavior_only",
        "human_annotation_eligible": False,
        "claim_boundary": (
            "A pass is same-family behavior replication only. It is not evidence for a temporal-binding module, "
            "a shared physical coordinate, cross-architecture conservation, or causal mechanism."
        ),
    }
    result = dict(core)
    result["final_digest"] = digest(core)
    write_jsonl(OUT_ROOT / "analysis/behavior_decisions.qwen3_14b.jsonl", decisions)
    write_json(OUT_ROOT / "analysis/final_summary.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize",
        "primary_behavior_passed": primary_passed,
        "prospective_interaction_passed": interaction_passed,
        "same_family_behavior_replication": same_family_replication,
        "auto_continue": result["auto_continue"],
        "final_digest": result["final_digest"],
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "behavior", "finalize"))
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "behavior":
        behavior_command()
    else:
        finalize_command()


if __name__ == "__main__":
    main()
