#!/usr/bin/env python3
"""Independent audits for Phase1232 native-boundary correction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
import sys
sys.path.insert(0, str(TEST_ROOT))

import phase1231_qwen3_clock_compass_behavior_execution as p1231


MAIN_SCRIPT = TEST_ROOT / "phase1232_qwen3_native_boundary_behavior_correction.py"
OUT_ROOT = TEST_ROOT / "result/phase1232_qwen3_native_boundary_behavior_correction"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"
PHASE1231_FINAL = TEST_ROOT / "result/phase1231_qwen3_clock_compass_behavior_execution/analysis/final.json"
PHASE1231_AUDIT = TEST_ROOT / "result/phase1231_qwen3_clock_compass_behavior_execution/audit/independent_final_audit.json"


def audit_value(kind: str, checks: dict[str, bool], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": 1232,
        "audit_type": kind,
        "created_at_utc": p1231.utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    if extra:
        value.update(extra)
    value["audit_digest"] = p1231.digest(value)
    return value


def preaudit() -> None:
    if PREAUDIT_PATH.exists():
        raise RuntimeError("preaudit already exists")
    contract = p1231.read_json(CONTRACT_PATH)
    plan = p1231.read_json(BATCH_PLAN_PATH)
    phase1231 = p1231.read_json(PHASE1231_FINAL)
    phase1231_audit = p1231.read_json(PHASE1231_AUDIT)
    _upstream, manifest, _material = p1231.verify_upstream()
    planned = [item for batch in plan["batches"] for item in batch["item_ids"]]
    new_ids = contract["interface_correction"]["new_native_ids"]
    old_ids = contract["interface_correction"]["old_spaced_ids"]
    checks = {
        "contract_self_digest": contract["contract_digest"] == p1231.digest(p1231.strip_digest(contract, "contract_digest")),
        "main_script_hash": contract["source_hashes"]["execution"] == p1231.file_sha256(MAIN_SCRIPT),
        "audit_script_hash": contract["source_hashes"]["independent_audit"] == p1231.file_sha256(Path(__file__).resolve()),
        "phase1231_final_retained": phase1231["final_digest"] == contract["upstream"]["phase1231_final_digest"],
        "phase1231_audit_pass": phase1231_audit.get("all_checks_passed") is True,
        "phase1231_was_formal_failure": phase1231["status"] == "candidate_behavior_gate_failed",
        "native_ids_exact": new_ids == {"north": 61895, "east": 60501, "south": 66484, "west": 11039},
        "old_ids_exact": old_ids == {"north": 10200, "east": 10984, "south": 9806, "west": 9710},
        "interfaces_distinct": all(new_ids[key] != old_ids[key] for key in p1231.CANDIDATES),
        "thresholds_unchanged": contract["thresholds"] == p1231.THRESHOLDS,
        "plan_self_digest": plan["plan_digest"] == p1231.digest(p1231.strip_digest(plan, "plan_digest")),
        "plan_exact_partition": len(planned) == 9216 and len(set(planned)) == 9216 and set(planned) == {row["item_id"] for row in manifest},
        "preoutput": not RAW_PATH.exists() and not RUN_SUMMARY_PATH.exists(),
        "no_hidden": contract["execution"]["hidden_states"] is False and contract["execution"]["attentions"] is False,
        "no_intervention": contract["execution"]["intervention"] is False,
        "construct_boundary_retained": contract["shortcut_boundary"]["target_record_use_identifiable"] is False,
    }
    value = audit_value("independent_preaudit", checks)
    p1231.write_json(PREAUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def result_audit() -> None:
    if RESULT_AUDIT_PATH.exists():
        raise RuntimeError("result audit already exists")
    contract = p1231.read_json(CONTRACT_PATH)
    raw = p1231.read_jsonl(RAW_PATH)
    summary = p1231.read_json(RUN_SUMMARY_PATH)
    final = p1231.read_json(FINAL_PATH)
    _upstream, manifest, material = p1231.verify_upstream()
    manifest_by_id = {row["item_id"]: row for row in manifest}
    native_ids = {key: int(value) for key, value in contract["interface_correction"]["new_native_ids"].items()}
    row_ok = []
    for row in raw:
        source = manifest_by_id[row["item_id"]]
        scores = row["candidate_scores"]
        ordered = sorted(scores, key=lambda candidate: scores[candidate], reverse=True)
        margin = scores[ordered[0]] - scores[ordered[1]]
        prediction = None if (not row["all_vocab_logits_finite"] or margin <= p1231.TIE_TOLERANCE) else ordered[0]
        gold = source["gold_candidate"]
        row_ok.append(
            row["prediction"] == prediction
            and row["correct"] == (prediction == gold)
            and row["full_vocab_top1_is_gold_candidate"] == (row["full_vocab_top1_id"] == native_ids[gold])
            and row["behavior_row_digest"] == p1231.digest(p1231.strip_digest(row, "behavior_row_digest"))
        )
    independently = p1231.adjudicate(raw, material)
    checks = {
        "case_count": len(raw) == 9216,
        "case_identity": set(manifest_by_id) == {row["item_id"] for row in raw},
        "row_recomputation": all(row_ok),
        "raw_digest": summary["raw_digest"] == p1231.digest(raw),
        "summary_self_digest": summary["summary_digest"] == p1231.digest(p1231.strip_digest(summary, "summary_digest")),
        "final_self_digest": final["final_digest"] == p1231.digest(p1231.strip_digest(final, "final_digest")),
        "ledger_recomputation": independently == final["ledgers"],
        "precision": set(summary["precision_audit"]["parameter_dtypes"]) == {"float16"} and not summary["precision_audit"]["has_quantized_modules"],
        "no_hidden": summary["hidden_states_saved"] is False and summary["attentions_saved"] is False,
        "no_intervention": summary["interventions_performed"] is False,
        "phase1231_retained": final["phase1231_correction"]["phase1231_formal_failure_retained"] is True,
        "construct_boundary": final["authorization"]["target_record_specific_mechanism_claim"] is False,
    }
    value = audit_value("independent_result_audit", checks, {
        "independent_metrics": {
            "candidate_accuracy": independently["overall_candidate_accuracy"],
            "Q5": independently["Q5"]["natural_first_token_accuracy"],
            "prediction_counts": dict(Counter(str(row["prediction"]) for row in raw)),
        }
    })
    p1231.write_json(RESULT_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def final_audit() -> None:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("final audit already exists")
    pre = p1231.read_json(PREAUDIT_PATH)
    result = p1231.read_json(RESULT_AUDIT_PATH)
    final = p1231.read_json(FINAL_PATH)
    checks = {
        "preaudit_pass": pre.get("all_checks_passed") is True,
        "result_audit_pass": result.get("all_checks_passed") is True,
        "final_self_digest": final["final_digest"] == p1231.digest(p1231.strip_digest(final, "final_digest")),
        "phase1231_not_rewritten": final["phase1231_correction"]["phase1231_formal_failure_retained"] is True,
        "no_hidden_authorization": final["authorization"]["hidden_scan"] is False,
        "no_target_record_claim": final["authorization"]["target_record_specific_mechanism_claim"] is False,
        "k207_scope": final["k_item"]["identifier"] == "K207" and "behavior only" in final["k_item"]["scope"],
    }
    value = audit_value("independent_final_audit", checks, {
        "final_digest": final["final_digest"],
        "result_audit_digest": result["audit_digest"],
    })
    p1231.write_json(FINAL_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    elif args.stage == "result":
        result_audit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
