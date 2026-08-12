#!/usr/bin/env python3
"""Independent audits for Phase1231 Qwen3 clock-compass execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN_SCRIPT = TEST_ROOT / "phase1231_qwen3_clock_compass_behavior_execution.py"
OUT_ROOT = TEST_ROOT / "result/phase1231_qwen3_clock_compass_behavior_execution"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
SHORTCUT_PATH = OUT_ROOT / "protocol/preoutput_shortcut_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

UPSTREAM_ROOT = TEST_ROOT / "result/phase1230_qwen3_clock_compass_behavior_protocol"
UPSTREAM_MANIFEST = UPSTREAM_ROOT / "protocol/qwen3_manifest.jsonl"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_FINAL_AUDIT = UPSTREAM_ROOT / "audit/independent_final_audit.json"
MATERIAL_PATH = TEST_ROOT / "result/phase1229_deanswer_clock_compass_material_contract/material/clock_compass_binding.jsonl"

EXPECTED_ROWS = 9216
EXPECTED_UPSTREAM_FINAL = "ebee860364c036ff9700d4a8af30b6a2f7309a5d24c03fccd1e451f895d4923b"
EXPECTED_UPSTREAM_AUDIT = "00bc4facb62df8cfa4fe89d41f849bb6650dd27dcabadc9624e21cf8703e0003"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_digest(value: dict[str, Any]) -> str:
    return digest(value)


def preaudit() -> None:
    if PREAUDIT_PATH.exists():
        raise RuntimeError("preaudit already exists")
    contract = read_json(CONTRACT_PATH)
    plan = read_json(BATCH_PLAN_PATH)
    shortcut = read_json(SHORTCUT_PATH)
    manifest = read_jsonl(UPSTREAM_MANIFEST)
    upstream_final = read_json(UPSTREAM_FINAL)
    upstream_audit = read_json(UPSTREAM_FINAL_AUDIT)
    item_ids = [row["item_id"] for row in manifest]
    manifest_by_id = {row["item_id"]: row for row in manifest}
    planned = [item for batch in plan["batches"] for item in batch["item_ids"]]
    checks = {
        "contract_self_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "main_script_hash": contract["source_hashes"]["execution"] == file_sha256(MAIN_SCRIPT),
        "audit_script_hash": contract["source_hashes"]["independent_audit"] == file_sha256(Path(__file__).resolve()),
        "upstream_final_digest": upstream_final.get("final_digest") == EXPECTED_UPSTREAM_FINAL,
        "upstream_audit_digest": upstream_audit.get("audit_digest") == EXPECTED_UPSTREAM_AUDIT,
        "upstream_audit_pass": upstream_audit.get("all_checks_passed") is True,
        "manifest_count": len(manifest) == EXPECTED_ROWS,
        "manifest_unique": len(set(item_ids)) == EXPECTED_ROWS,
        "plan_self_digest": plan["plan_digest"] == digest(strip_digest(plan, "plan_digest")),
        "plan_contract_link": contract["execution"]["batch_plan_digest"] == plan["plan_digest"],
        "plan_exact_partition": len(planned) == EXPECTED_ROWS and len(set(planned)) == EXPECTED_ROWS and set(planned) == set(item_ids),
        "plan_exact_lengths": all(
            all(len(manifest_by_id[item]["input_ids"]) == int(batch["input_length"]) for item in batch["item_ids"])
            for batch in plan["batches"]
        ),
        "plan_batch_cap": all(1 <= int(batch["runtime_batch_size"]) <= 16 and int(batch["runtime_batch_size"]) == len(batch["item_ids"]) for batch in plan["batches"]),
        "shortcut_self_digest": shortcut["shortcut_audit_digest"] == digest(strip_digest(shortcut, "shortcut_audit_digest")),
        "shortcut_contract_link": contract["shortcut_sidecar"]["digest"] == shortcut["shortcut_audit_digest"],
        "shortcut_preoutput": not RAW_PATH.exists() and not RUN_SUMMARY_PATH.exists(),
        "shortcut_constant_chance": shortcut["results"]["constant"]["accuracy"] == 0.25,
        "shortcut_three_non_target_exact": shortcut["results"]["target_index_plus_three_non_targets"]["accuracy"] == 1.0,
        "frozen_thresholds": contract["thresholds"] == {
            "Q0_finite_rate": 1.0,
            "Q1_panel_accuracy": 0.9,
            "Q1_active_worst_marginal": 0.8,
            "Q2_active_quartet": 0.75,
            "Q3_control_invariant_bundle": 0.8,
            "Q4_template_pair": 0.85,
            "Q5_natural_first_token": 0.8,
        },
        "scope_no_hidden": contract["execution"]["hidden_states"] is False and contract["execution"]["attentions"] is False,
        "scope_no_intervention": contract["execution"]["intervention"] is False,
        "q0_q5_unchanged_by_shortcut": shortcut["does_not_change_Q0_Q5"] is True,
    }
    value: dict[str, Any] = {
        "phase": 1231,
        "audit_type": "independent_preaudit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    value["audit_digest"] = audit_digest(value)
    write_json(PREAUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def independent_result_audit() -> None:
    if RESULT_AUDIT_PATH.exists():
        raise RuntimeError("result audit already exists")
    contract = read_json(CONTRACT_PATH)
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    final = read_json(FINAL_PATH)
    manifest = read_jsonl(UPSTREAM_MANIFEST)
    material = {row["item_id"]: row for row in read_jsonl(MATERIAL_PATH)}
    manifest_map = {row["item_id"]: row for row in manifest}

    row_checks = []
    for row in raw:
        source = manifest_map[row["item_id"]]
        scores = row["candidate_scores"]
        ordered = sorted(scores, key=lambda candidate: scores[candidate], reverse=True)
        margin = scores[ordered[0]] - scores[ordered[1]]
        finite = bool(row["all_vocab_logits_finite"])
        prediction = None if (not finite or margin <= 1e-7) else ordered[0]
        gold = source["gold_candidate"]
        wrong_best = max(value for candidate, value in scores.items() if candidate != gold)
        row_checks.append(
            row["manifest_row_digest"] == source["manifest_row_digest"]
            and row["prediction"] == prediction
            and row["correct"] == (prediction == gold)
            and abs(row["top_candidate_margin"] - margin) <= 1e-10
            and abs(row["gold_margin"] - (scores[gold] - wrong_best)) <= 1e-10
            and row["full_vocab_top1_is_gold_candidate"] == (row["full_vocab_top1_id"] == source["gold_candidate_token_id"])
            and row["behavior_row_digest"] == digest(strip_digest(row, "behavior_row_digest"))
        )

    q0_overall = rate(raw, "all_vocab_logits_finite")
    q0_split = {split: rate([r for r in raw if r["split"] == split], "all_vocab_logits_finite") for split in ("discovery", "confirmation", "natural_use")}
    q1_panel = {
        f"{split}|{panel}": rate([r for r in raw if r["split"] == split and r["panel"] == panel], "correct")
        for split in ("discovery", "confirmation", "natural_use")
        for panel in ("active", "matched_null", "surface_order")
    }
    natural = [row for row in raw if row["split"] == "natural_use"]
    q5 = rate(natural, "full_vocab_top1_is_gold_candidate")
    checks = {
        "case_count": len(raw) == EXPECTED_ROWS,
        "case_identity": {row["item_id"] for row in raw} == set(manifest_map),
        "row_recomputation": all(row_checks),
        "raw_digest": summary["raw_digest"] == digest(raw),
        "summary_self_digest": summary["summary_digest"] == digest(strip_digest(summary, "summary_digest")),
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "contract_link": final["contract_digest"] == contract["contract_digest"] == summary["contract_digest"],
        "no_hidden": summary["hidden_states_saved"] is False and summary["attentions_saved"] is False,
        "no_intervention": summary["interventions_performed"] is False,
        "precision": set(summary["precision_audit"]["parameter_dtypes"]) == {"float16"} and not summary["precision_audit"]["has_quantized_modules"],
        "Q0_overall": abs(final["ledgers"]["Q0"]["overall_finite_rate"] - q0_overall) <= 1e-12,
        "Q0_split": final["ledgers"]["Q0"]["split_finite_rates"] == q0_split,
        "Q1_panel": final["ledgers"]["Q1"]["split_panel_candidate_accuracy"] == q1_panel,
        "Q5": abs(final["ledgers"]["Q5"]["natural_first_token_accuracy"] - q5) <= 1e-12,
        "shortcut_not_target_claim": final["authorization"]["target_record_specific_mechanism_claim"] is False,
        "next_scope_consistent": final["authorization"]["automatic_hidden_scan"] is False,
    }
    value: dict[str, Any] = {
        "phase": 1231,
        "audit_type": "independent_result_audit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "independent_metrics": {
            "Q0_overall": q0_overall,
            "Q0_split": q0_split,
            "Q1_split_panel": q1_panel,
            "Q5_natural_first_token": q5,
            "prediction_counts": dict(Counter(str(row["prediction"]) for row in raw)),
        },
    }
    value["audit_digest"] = audit_digest(value)
    write_json(RESULT_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def final_audit() -> None:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("final audit already exists")
    pre = read_json(PREAUDIT_PATH)
    result = read_json(RESULT_AUDIT_PATH)
    final = read_json(FINAL_PATH)
    checks = {
        "preaudit_pass": pre.get("all_checks_passed") is True,
        "result_audit_pass": result.get("all_checks_passed") is True,
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "preaudit_linked_contract": final["contract_digest"] == read_json(CONTRACT_PATH)["contract_digest"],
        "shortcut_link": final["shortcut_audit_digest"] == read_json(SHORTCUT_PATH)["shortcut_audit_digest"],
        "behavior_only": final["authorization"]["hidden_state_execution_in_this_phase"] is False,
        "no_target_record_overclaim": final["authorization"]["target_record_specific_mechanism_claim"] is False,
        "k_scope": final["k_item"]["identifier"] == "K206" and "behavior only" in final["k_item"]["scope"],
    }
    value: dict[str, Any] = {
        "phase": 1231,
        "audit_type": "independent_final_audit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "final_digest": final["final_digest"],
        "result_audit_digest": result["audit_digest"],
    }
    value["audit_digest"] = audit_digest(value)
    write_json(FINAL_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    elif args.stage == "result":
        independent_result_audit()
    elif args.stage == "final":
        final_audit()


if __name__ == "__main__":
    main()
