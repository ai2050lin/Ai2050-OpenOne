#!/usr/bin/env python3
"""Phase1232: correct the Qwen3 native assistant first-token interface.

Phase1231 formally remains a failed leading-space interface experiment.  This
separate phase keeps every prompt, batch, denominator, threshold, and ledger
fixed, but preregisters the tokenizer-native no-leading-space direction tokens
that a generated assistant response actually emits at its first position.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1231_qwen3_clock_compass_behavior_execution as p1231
from model_utils import MODEL_CONFIGS
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1232
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1232_qwen3_native_boundary_behavior_correction_audit.py"

PHASE1231_ROOT = TEST_ROOT / "result/phase1231_qwen3_clock_compass_behavior_execution"
PHASE1231_FINAL = PHASE1231_ROOT / "analysis/final.json"
PHASE1231_AUDIT = PHASE1231_ROOT / "audit/independent_final_audit.json"
EXPECTED_PHASE1231_FINAL = "6ba50a8126788cd16d6d7d277f80f1d1da929eec77e36b73b38392f2c09d710a"
EXPECTED_PHASE1231_AUDIT = "93384982ab5f5a88dd9d2987b0b690f8bea03e232f3ddf18d9ef2716b74aa07e"

OUT_ROOT = TEST_ROOT / "result/phase1232_qwen3_native_boundary_behavior_correction"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

CANDIDATES = p1231.CANDIDATES
TIE_TOLERANCE = p1231.TIE_TOLERANCE


def source_hashes() -> dict[str, str]:
    return {
        "execution": p1231.file_sha256(SCRIPT),
        "independent_audit": p1231.file_sha256(AUDIT_SCRIPT),
    }


def verify_phase1231() -> tuple[dict[str, Any], dict[str, Any]]:
    final = p1231.read_json(PHASE1231_FINAL)
    audit = p1231.read_json(PHASE1231_AUDIT)
    if final.get("final_digest") != EXPECTED_PHASE1231_FINAL:
        raise RuntimeError("Phase1231 final digest mismatch")
    if audit.get("audit_digest") != EXPECTED_PHASE1231_AUDIT or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1231 audit mismatch")
    if final.get("status") != "candidate_behavior_gate_failed":
        raise RuntimeError("unexpected Phase1231 status")
    return final, audit


def derive_native_candidate_ids() -> tuple[dict[str, int], dict[str, int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    native: dict[str, int] = {}
    spaced: dict[str, int] = {}
    for candidate in CANDIDATES:
        native_ids = tokenizer.encode(candidate, add_special_tokens=False)
        spaced_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        if len(native_ids) != 1 or len(spaced_ids) != 1:
            raise RuntimeError("candidate is not a single token in both interfaces")
        if native_ids[0] == spaced_ids[0]:
            raise RuntimeError("interface correction unexpectedly has identical IDs")
        if tokenizer.decode(native_ids) != candidate or tokenizer.decode(spaced_ids) != " " + candidate:
            raise RuntimeError("candidate roundtrip failed")
        native[candidate] = int(native_ids[0])
        spaced[candidate] = int(spaced_ids[0])
    if len(set(native.values())) != 4:
        raise RuntimeError("native candidate IDs are not distinct")
    return native, spaced


def preregister() -> None:
    if CONTRACT_PATH.exists() or BATCH_PLAN_PATH.exists():
        raise RuntimeError("Phase1232 preregistration already exists")
    previous, _audit = verify_phase1231()
    _upstream, manifest, _material = p1231.verify_upstream()
    native_ids, spaced_ids = derive_native_candidate_ids()
    plan = p1231.build_batch_plan(manifest)
    p1231.write_json(BATCH_PLAN_PATH, plan)
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1232.qwen3.native_boundary.behavior_correction.v1",
        "created_at_utc": p1231.utc_now(),
        "objective": "Independently rerun the frozen clock-compass behavior contract using tokenizer-native assistant first-token IDs.",
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1231_final_digest": EXPECTED_PHASE1231_FINAL,
            "phase1231_audit_digest": EXPECTED_PHASE1231_AUDIT,
            "phase1231_status": previous["status"],
            "phase1230_manifest_digest": p1231.EXPECTED_MANIFEST_DIGEST,
            "phase1231_formal_result_is_not_rewritten": True,
        },
        "interface_correction": {
            "old_continuation": "one ASCII space followed by the direction",
            "new_continuation": "direction begins immediately at the native assistant generation boundary",
            "old_spaced_ids": spaced_ids,
            "new_native_ids": native_ids,
            "selection_rule": "tokenize the exact unprefixed lowercase response string before corrected model execution",
            "thresholds_selected_from_output": False,
        },
        "execution": {
            "model": "qwen3",
            "device": "cuda",
            "precision": "float16",
            "quantization": "none",
            "manifest": "exact Phase1230 input_ids",
            "batch_plan_digest": plan["plan_digest"],
            "batch_size": p1231.BATCH_SIZE,
            "adaptive_batch_fallback": False,
            "hidden_states": False,
            "attentions": False,
            "intervention": False,
        },
        "thresholds": p1231.THRESHOLDS,
        "ledgers": "Q0-Q5 are exactly the Phase1230/1231 definitions",
        "shortcut_boundary": {
            "phase1231_shortcut_audit_digest": previous["shortcut_audit_digest"],
            "three_non_target_complement_accuracy": previous["shortcut_boundary"]["three_non_target_lookup_accuracy"],
            "target_record_use_identifiable": False,
        },
        "forbidden": [
            "change any prompt, case, split, panel, batch membership, threshold, denominator, or tie rule",
            "drop Phase1231 because its interface failed",
            "save hidden states or attentions",
            "perform intervention",
            "claim target-record use from behavior",
            "run another model",
        ],
    }
    contract["contract_digest"] = p1231.digest(contract)
    p1231.write_json(CONTRACT_PATH, contract)
    print(p1231.canonical_json({
        "status": "phase1232_preregistered",
        "contract_digest": contract["contract_digest"],
        "native_ids": native_ids,
    }))


def verify_frozen() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    verify_phase1231()
    _upstream, manifest, material = p1231.verify_upstream()
    contract = p1231.read_json(CONTRACT_PATH)
    plan = p1231.read_json(BATCH_PLAN_PATH)
    if contract["contract_digest"] != p1231.digest(p1231.strip_digest(contract, "contract_digest")):
        raise RuntimeError("Phase1232 contract drift")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("Phase1232 source changed after freeze")
    if plan["plan_digest"] != p1231.digest(p1231.strip_digest(plan, "plan_digest")):
        raise RuntimeError("Phase1232 batch plan drift")
    if contract["execution"]["batch_plan_digest"] != plan["plan_digest"]:
        raise RuntimeError("Phase1232 plan link mismatch")
    preaudit = p1231.read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("Phase1232 preaudit failed")
    return contract, plan, manifest, material


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1232 outputs already exist")
    contract, plan, manifest, _material = verify_frozen()
    manifest_by_id = {row["item_id"]: row for row in manifest}
    native_ids = {key: int(value) for key, value in contract["interface_correction"]["new_native_ids"].items()}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    if device.type != "cuda" or precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
        release_fp16(model)
        raise RuntimeError("Phase1232 numerical contract failed")
    raw: list[dict[str, Any]] = []
    try:
        for batch_number, batch in enumerate(plan["batches"], start=1):
            members = [manifest_by_id[item_id] for item_id in batch["item_ids"]]
            input_ids = torch.tensor([row["input_ids"] for row in members], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=1,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            logits = output.logits[:, -1, :].float()
            finite_rows = torch.isfinite(logits).all(dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            top1_ids = logits.argmax(dim=-1)
            for row_index, source in enumerate(members):
                scores = {candidate: float(log_probs[row_index, native_ids[candidate]].item()) for candidate in CANDIDATES}
                order = sorted(CANDIDATES, key=lambda candidate: scores[candidate], reverse=True)
                top_margin = scores[order[0]] - scores[order[1]]
                finite = bool(finite_rows[row_index].item()) and all(math.isfinite(value) for value in scores.values())
                prediction = None if (not finite or top_margin <= TIE_TOLERANCE) else order[0]
                gold = source["gold_candidate"]
                wrong_best = max(value for candidate, value in scores.items() if candidate != gold)
                top1_id = int(top1_ids[row_index].item())
                row: dict[str, Any] = {
                    "phase": PHASE,
                    "schema_version": "phase1232.qwen3.behavior.row.v1",
                    "contract_digest": contract["contract_digest"],
                    "item_id": source["item_id"],
                    "manifest_row_digest": source["manifest_row_digest"],
                    "execution_index": int(source["execution_index"]),
                    "split": source["split"],
                    "panel": source["panel"],
                    "bundle_id": source["bundle_id"],
                    "world_id": source["world_id"],
                    "template_id": source["template_id"],
                    "target_entity": source["target_entity"],
                    "gold_candidate": gold,
                    "all_vocab_logits_finite": finite,
                    "candidate_scores": scores,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "unresolved_tie": finite and top_margin <= TIE_TOLERANCE,
                    "top_candidate_margin": top_margin,
                    "gold_margin": scores[gold] - wrong_best,
                    "full_vocab_top1_id": top1_id,
                    "full_vocab_top1_text": tokenizer.decode([top1_id], skip_special_tokens=False),
                    "full_vocab_top1_is_gold_candidate": top1_id == native_ids[gold],
                    "input_length": int(source["input_length"]),
                    "runtime_batch_size": len(members),
                    "batch_index": int(batch["batch_index"]),
                }
                row["behavior_row_digest"] = p1231.digest(row)
                raw.append(row)
            del output, logits, log_probs, top1_ids, finite_rows, input_ids
            if batch_number % 50 == 0 or batch_number == len(plan["batches"]):
                print(f"[phase1232] {batch_number}/{len(plan['batches'])} batches", flush=True)
        raw.sort(key=lambda row: row["execution_index"])
        p1231.write_jsonl(RAW_PATH, raw)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1232.qwen3.run_summary.v1",
            "created_at_utc": p1231.utc_now(),
            "model": "qwen3",
            "contract_digest": contract["contract_digest"],
            "case_count": len(raw),
            "raw_digest": p1231.digest(raw),
            "precision_audit": precision,
            "placement": placement,
            "batch_plan_digest": plan["plan_digest"],
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hidden_states_saved": False,
            "attentions_saved": False,
            "interventions_performed": False,
        }
        summary["summary_digest"] = p1231.digest(summary)
        p1231.write_json(RUN_SUMMARY_PATH, summary)
        print(p1231.canonical_json({"status": "behavior_complete", "rows": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1232 final already exists")
    contract, plan, manifest, material = verify_frozen()
    raw = p1231.read_jsonl(RAW_PATH)
    summary = p1231.read_json(RUN_SUMMARY_PATH)
    if len(raw) != p1231.EXPECTED_ROWS or summary["raw_digest"] != p1231.digest(raw):
        raise RuntimeError("Phase1232 raw mismatch")
    if {row["item_id"] for row in raw} != {row["item_id"] for row in manifest}:
        raise RuntimeError("Phase1232 manifest coverage mismatch")
    ledgers = p1231.adjudicate(raw, material)
    candidate_gate = bool(ledgers["candidate_behavior_gate"])
    first_token_gate = bool(ledgers["natural_first_token_gate"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1232.qwen3.native_boundary.final.v1",
        "created_at_utc": p1231.utc_now(),
        "status": "native_candidate_behavior_passed_construct_ambiguous" if candidate_gate else "native_candidate_behavior_gate_failed",
        "contract_digest": contract["contract_digest"],
        "batch_plan_digest": plan["plan_digest"],
        "run_summary_digest": summary["summary_digest"],
        "raw_digest": summary["raw_digest"],
        "ledgers": ledgers,
        "phase1231_correction": {
            "phase1231_final_digest": EXPECTED_PHASE1231_FINAL,
            "phase1231_formal_failure_retained": True,
            "old_spaced_ids": contract["interface_correction"]["old_spaced_ids"],
            "new_native_ids": contract["interface_correction"]["new_native_ids"],
        },
        "construct_boundary": {
            "three_non_target_complement_accuracy": contract["shortcut_boundary"]["three_non_target_complement_accuracy"],
            "target_record_use_identifiable": False,
            "C4_to_S4_alone_sufficient_to_fix": False,
        },
        "k_item": {
            "identifier": "K207",
            "evidence_grade": "E3-BEHAVIOR-CONSTRUCT-BOUNDARY" if candidate_gate else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "The native assistant first-token interface passed Q0-Q4, but bijective complement shortcuts prevent target-record attribution."
                if candidate_gate else
                "The corrected native assistant first-token interface did not pass all frozen Q0-Q4 ledgers."
            ),
            "scope": "Qwen3-4B; CUDA FP16; exact Phase1229-1230 inputs; corrected unprefixed first-token candidate interface; behavior only",
        },
        "authorization": {
            "candidate_behavior_object": candidate_gate,
            "natural_first_token_claim": first_token_gate,
            "target_record_specific_mechanism_claim": False,
            "hidden_scan": False,
            "next_experiment": (
                "A separately frozen record-indexed response protocol that treats direct-target and global-complement strategies as competing programs"
                if candidate_gate else None
            ),
            "auto_continue": False,
            "cross_model_run": False,
        },
        "claim_boundary": [
            "Phase1231 remains a valid negative result for its preregistered leading-space interface.",
            "Phase1232 is an output-informed interface correction and is not an untouched confirmation split.",
            "A behavior pass cannot identify target-record use because the non-target complement is exact.",
            "No hidden, attention, intervention, rescue, generation continuation, or cross-model mechanism was tested.",
        ],
        "new_mathematics_required": False,
    }
    final["final_digest"] = p1231.digest(final)
    p1231.write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    verify_phase1231()
    native, spaced = derive_native_candidate_ids()
    assert native == {"north": 61895, "east": 60501, "south": 66484, "west": 11039}
    assert all(native[key] != spaced[key] for key in CANDIDATES)
    print(p1231.canonical_json({"status": "selftest_passed", "native_ids": native, "spaced_ids": spaced}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "preregister", "run", "finalize"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "preregister":
        preregister()
    elif args.stage == "run":
        run_qwen3()
    elif args.stage == "finalize":
        finalize()


if __name__ == "__main__":
    main()
