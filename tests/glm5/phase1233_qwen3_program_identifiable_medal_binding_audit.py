#!/usr/bin/env python3
"""Independent audits for Phase1233 program-identifiable medal binding.

This file deliberately does not import the Phase1233 execution module.  It
recomputes artifact digests, registered-program ceilings, collision ledgers,
and final claim permissions from the frozen files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN_SCRIPT = TEST_ROOT / "phase1233_qwen3_program_identifiable_medal_binding.py"
OUT_ROOT = TEST_ROOT / "result/phase1233_qwen3_program_identifiable_medal_binding"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/medal_binding.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
PROGRAM_AUDIT_PATH = OUT_ROOT / "protocol/competing_program_audit.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"
UPSTREAM_FINAL = TEST_ROOT / "result/phase1232_qwen3_native_boundary_behavior_correction/analysis/final.json"
UPSTREAM_AUDIT = TEST_ROOT / "result/phase1232_qwen3_native_boundary_behavior_correction/audit/independent_final_audit.json"

PHASE = 1233
EXPECTED_ROWS = 6912
EXPECTED_ROWS_PER_SPLIT = 2304
SPLITS = ("discovery", "confirmation", "natural_use")
VALUES = ("gold", "silver", "bronze")
CANDIDATES = ("first", "second", "third")
VALUE_TO_ANSWER = dict(zip(VALUES, CANDIDATES))
EXPECTED_CANDIDATE_IDS = {"first": [3896], "second": [5569], "third": [31727]}
EXPECTED_UPSTREAM_FINAL = "885d40152f901335204af5f87b491acbe56ddfa719fbc4200da5f752bee3d190"
EXPECTED_UPSTREAM_AUDIT = "73c50bbfc05147209c4e8fa674c4b09627d88a5e33d22fc04fdd34cee83eec92"
TIE_TOLERANCE = 1e-7
THRESHOLDS = {
    "Q0_finite_rate": 1.0,
    "Q1_split_accuracy": 0.90,
    "Q1_worst_marginal": 0.80,
    "Q1_program_ceiling_margin": 0.15,
    "Q2_target_change_triplet": 0.70,
    "Q3_non_target_null_triplet": 0.75,
    "Q4_query_switch_pair": 0.80,
    "Q4_binding_swap_pair": 0.80,
    "Q5_order_pair": 0.85,
    "Q5_template_pair": 0.85,
    "Q6_natural_first_token": 0.80,
}
MARGINAL_AXES = ("gold_candidate", "template_id", "world_id", "query_index", "order_variant")


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


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            groups[str(row[key])].append(row)
    return groups


def empirical_lookup_accuracy(rows: list[dict[str, Any]], feature: Callable[[dict[str, Any]], Any]) -> float:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[canonical_json(feature(row))][row["gold_candidate"]] += 1
    return sum(max(cell.values()) for cell in table.values()) / len(rows)


def independent_program_metrics(material: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, Callable[[dict[str, Any]], Any]] = {
        "constant": lambda row: "constant",
        "query_entity": lambda row: row["query_entity"],
        "query_index": lambda row: row["query_index"],
        "other_record_value": lambda row: row["other_value"],
        "query_plus_other_value": lambda row: (row["query_index"], row["other_value"]),
        "first_record_value": lambda row: row["values"][row["record_order_indices"][0]],
        "last_record_value": lambda row: row["values"][row["record_order_indices"][-1]],
        "value_bag": lambda row: tuple(sorted(row["values"])),
        "target_record_position": lambda row: row["target_record_position"],
        "template": lambda row: row["template_id"],
        "world": lambda row: row["world_id"],
        "order": lambda row: row["order_variant"],
        "intended_target_value": lambda row: row["target_value"],
    }
    accuracies = {name: empirical_lookup_accuracy(material, function) for name, function in features.items()}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    target = grouped(material, "target_triplet_id")
    null = grouped(material, "null_triplet_id")
    query = grouped(material, "query_pair_id")
    order = grouped(material, "order_pair_id")
    template = grouped(material, "template_pair_id")
    swap = grouped(material, "binding_swap_pair_id")
    collision_checks = {
        "target_change_complete": all(
            len(cell) == 3
            and len({row["other_value"] for row in cell}) == 1
            and {row["gold_candidate"] for row in cell} == set(CANDIDATES)
            for cell in target.values()
        ),
        "non_target_null_complete": all(
            len(cell) == 3
            and len({row["target_value"] for row in cell}) == 1
            and len({row["gold_candidate"] for row in cell}) == 1
            for cell in null.values()
        ),
        "query_switch_discriminating": all(
            len(cell) == 2
            and len({tuple(row["values"]) for row in cell}) == 1
            and (
                len({row["gold_candidate"] for row in cell}) == 2
                if cell[0]["values"][0] != cell[0]["values"][1]
                else len({row["gold_candidate"] for row in cell}) == 1
            )
            for cell in query.values()
        ),
        "order_invariant": all(len(cell) == 2 and len({row["gold_candidate"] for row in cell}) == 1 for cell in order.values()),
        "template_invariant": all(len(cell) == 2 and len({row["gold_candidate"] for row in cell}) == 1 for cell in template.values()),
        "binding_swap_discriminating": all(
            len(cell) == 2
            and len({row["gold_candidate"] for row in cell}) == 2
            and len({row["prompt_lexical_multiset_digest"] for row in cell}) == 1
            and len({manifest_by_id[row["item_id"]]["input_id_multiset_digest"] for row in cell}) == 1
            for cell in swap.values()
        ),
    }
    alternatives = {name: value for name, value in accuracies.items() if name != "intended_target_value"}
    return {
        "empirical_bayes_accuracy": accuracies,
        "maximum_registered_alternative_accuracy": max(alternatives.values()),
        "collision_group_counts": {
            "target_change_triplets": len(target),
            "non_target_null_triplets": len(null),
            "query_pairs": len(query),
            "order_pairs": len(order),
            "template_pairs": len(template),
            "binding_swap_pairs": len(swap),
        },
        "collision_checks": collision_checks,
        "repeat_value_row_count": sum(row["values"][0] == row["values"][1] for row in material),
        "unused_value_min": min(len(set(VALUES) - set(row["values"])) for row in material),
    }


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def collision_rate(rows: list[dict[str, Any]], key: str, expected_size: int, mode: str) -> float:
    outcomes: list[bool] = []
    for cell in grouped(rows, key).values():
        success = len(cell) == expected_size and all(row["all_vocab_logits_finite"] and row["correct"] for row in cell)
        predictions = [row["prediction"] for row in cell]
        if mode == "cover":
            success = success and set(predictions) == set(CANDIDATES)
        elif mode == "invariant":
            success = success and len(set(predictions)) == 1
        elif mode == "different":
            success = success and len(set(predictions)) == expected_size
        else:
            raise ValueError(mode)
        outcomes.append(success)
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


def independent_ledgers(raw: list[dict[str, Any]], alternative_ceiling: float) -> dict[str, Any]:
    q0_split = {split: rate([row for row in raw if row["split"] == split], "all_vocab_logits_finite") for split in SPLITS}
    q0_overall = rate(raw, "all_vocab_logits_finite")
    q0_pass = q0_overall >= THRESHOLDS["Q0_finite_rate"] and min(q0_split.values()) >= THRESHOLDS["Q0_finite_rate"]
    split_accuracy = {split: rate([row for row in raw if row["split"] == split], "correct") for split in SPLITS}
    marginal_cells: dict[str, dict[str, float]] = {}
    marginal_worst: dict[str, float] = {}
    for axis in MARGINAL_AXES:
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw:
            cells[canonical_json(row[axis])].append(row)
        values = {key: rate(cell, "correct") for key, cell in cells.items()}
        marginal_cells[axis] = values
        marginal_worst[axis] = min(values.values())
    overall_accuracy = rate(raw, "correct")
    program_margin = overall_accuracy - alternative_ceiling
    q1_pass = (
        min(split_accuracy.values()) >= THRESHOLDS["Q1_split_accuracy"]
        and min(marginal_worst.values()) >= THRESHOLDS["Q1_worst_marginal"]
        and program_margin >= THRESHOLDS["Q1_program_ceiling_margin"]
    )
    q2 = {split: collision_rate([row for row in raw if row["split"] == split], "target_triplet_id", 3, "cover") for split in SPLITS}
    q3 = {split: collision_rate([row for row in raw if row["split"] == split], "null_triplet_id", 3, "invariant") for split in SPLITS}
    q4_query: dict[str, float] = {}
    q4_swap: dict[str, float] = {}
    for split in SPLITS:
        selected = [row for row in raw if row["split"] == split and row["target_value"] != row["other_value"]]
        q4_query[split] = collision_rate(selected, "query_pair_id", 2, "different")
        q4_swap[split] = collision_rate(selected, "binding_swap_pair_id", 2, "different")
    q5_order = {split: collision_rate([row for row in raw if row["split"] == split], "order_pair_id", 2, "invariant") for split in SPLITS}
    q5_template = {split: collision_rate([row for row in raw if row["split"] == split], "template_pair_id", 2, "invariant") for split in SPLITS}
    natural = [row for row in raw if row["split"] == "natural_use"]
    q6_accuracy = rate(natural, "full_vocab_top1_is_gold_candidate")
    q2_pass = min(q2.values()) >= THRESHOLDS["Q2_target_change_triplet"]
    q3_pass = min(q3.values()) >= THRESHOLDS["Q3_non_target_null_triplet"]
    q4_pass = min(q4_query.values()) >= THRESHOLDS["Q4_query_switch_pair"] and min(q4_swap.values()) >= THRESHOLDS["Q4_binding_swap_pair"]
    q5_pass = min(q5_order.values()) >= THRESHOLDS["Q5_order_pair"] and min(q5_template.values()) >= THRESHOLDS["Q5_template_pair"]
    q6_pass = q6_accuracy >= THRESHOLDS["Q6_natural_first_token"]
    behavior_gate = q0_pass and q1_pass and q2_pass and q3_pass and q4_pass and q5_pass
    return {
        "Q0": {"overall_finite_rate": q0_overall, "split_finite_rates": q0_split, "passed": q0_pass},
        "Q1": {
            "split_accuracy": split_accuracy,
            "worst_marginal_by_axis": marginal_worst,
            "marginal_cells": marginal_cells,
            "overall_accuracy": overall_accuracy,
            "registered_alternative_ceiling": alternative_ceiling,
            "program_ceiling_margin": program_margin,
            "passed": q1_pass,
        },
        "Q2": {"target_change_triplet_success": q2, "passed": q2_pass},
        "Q3": {"non_target_null_triplet_success": q3, "passed": q3_pass},
        "Q4": {"query_switch_pair_success": q4_query, "binding_swap_pair_success": q4_swap, "passed": q4_pass},
        "Q5": {"order_pair_success": q5_order, "template_pair_success": q5_template, "passed": q5_pass},
        "Q6": {"natural_first_token_accuracy": q6_accuracy, "passed": q6_pass},
        "construct_gate": True,
        "behavior_gate": behavior_gate,
        "hidden_eligibility": behavior_gate,
        "natural_first_token_gate": behavior_gate and q6_pass,
        "overall_candidate_accuracy": overall_accuracy,
        "tie_count": sum(row["unresolved_tie"] for row in raw),
        "nonfinite_count": sum(not row["all_vocab_logits_finite"] for row in raw),
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in raw)),
    }


def audit_value(kind: str, checks: dict[str, bool], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(item) for item in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    if extra:
        value.update(extra)
    value["audit_digest"] = digest(value)
    return value


def preaudit() -> None:
    if PREAUDIT_PATH.exists():
        raise RuntimeError("preaudit already exists")
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_AUDIT_PATH)
    plan = read_json(BATCH_PLAN_PATH)
    upstream = read_json(UPSTREAM_FINAL)
    upstream_audit = read_json(UPSTREAM_AUDIT)
    manifest_by_id = {row["item_id"]: row for row in manifest}
    program_metrics = independent_program_metrics(material, manifest)
    planned = [item for batch in plan["batches"] for item in batch["item_ids"]]
    split_counts = Counter(row["split"] for row in material)
    factorial_cells: dict[tuple[Any, ...], int] = Counter(
        (row["split"], row["world_index"], row["template_index"], row["order_variant"], row["query_index"])
        for row in material
    )
    lexical_leak = sum(
        bool(set(CANDIDATES) & set(re.findall(r"[a-z]+", row["prompt"].lower())))
        for row in material
    )
    row_integrity = all(row["row_digest"] == digest(strip_digest(row, "row_digest")) for row in material)
    manifest_integrity = all(row["manifest_row_digest"] == digest(strip_digest(row, "manifest_row_digest")) for row in manifest)
    manifest_links = all(
        row["item_id"] in manifest_by_id
        and manifest_by_id[row["item_id"]]["material_row_digest"] == row["row_digest"]
        and manifest_by_id[row["item_id"]]["gold_candidate"] == row["gold_candidate"]
        for row in material
    )
    token_rows = all(
        row["candidate_token_ids"] == EXPECTED_CANDIDATE_IDS
        and not ({ids[0] for ids in EXPECTED_CANDIDATE_IDS.values()} & set(row["input_ids"]))
        and row["input_ids_digest"] == digest(row["input_ids"])
        and row["input_id_multiset_digest"] == digest(sorted(row["input_ids"]))
        and row["prediction_token_index"] == len(row["input_ids"]) - 1
        and row["input_length"] == len(row["input_ids"])
        and bool(row["role_token_spans"])
        for row in manifest
    )
    expected_program = {
        "constant": 1 / 3,
        "query_entity": 1 / 3,
        "query_index": 1 / 3,
        "other_record_value": 1 / 3,
        "query_plus_other_value": 1 / 3,
        "first_record_value": 2 / 3,
        "last_record_value": 2 / 3,
        "value_bag": 2 / 3,
        "target_record_position": 1 / 3,
        "template": 1 / 3,
        "world": 1 / 3,
        "order": 1 / 3,
        "intended_target_value": 1.0,
    }
    computed_program = program_metrics["empirical_bayes_accuracy"]
    checks = {
        "contract_self_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "main_script_hash": contract["source_hashes"]["execution"] == file_sha256(MAIN_SCRIPT),
        "audit_script_hash": contract["source_hashes"]["independent_audit"] == file_sha256(Path(__file__).resolve()),
        "upstream_final_digest": upstream["final_digest"] == EXPECTED_UPSTREAM_FINAL == contract["upstream"]["phase1232_final_digest"],
        "upstream_audit_pass": upstream_audit.get("all_checks_passed") is True and upstream_audit["audit_digest"] == EXPECTED_UPSTREAM_AUDIT,
        "old_family_closed": upstream["authorization"]["hidden_scan"] is False,
        "material_digest": contract["material"]["material_digest"] == digest(material),
        "material_row_digests": row_integrity,
        "material_cardinality": len(material) == EXPECTED_ROWS and len({row["item_id"] for row in material}) == EXPECTED_ROWS,
        "split_cardinality": split_counts == Counter({split: EXPECTED_ROWS_PER_SPLIT for split in SPLITS}),
        "full_assignment_factorial": len(factorial_cells) == 768 and set(factorial_cells.values()) == {9},
        "split_world_separation": all(len({row["world_id"] for row in material if row["split"] == split}) == 32 for split in SPLITS),
        "mapping_exact": contract["material"]["mapping"] == VALUE_TO_ANSWER,
        "non_bijective_support": program_metrics["repeat_value_row_count"] == 2304 and program_metrics["unused_value_min"] >= 1,
        "candidate_lexical_leak_zero": lexical_leak == 0,
        "manifest_digest": contract["interface"]["manifest_digest"] == digest(manifest),
        "manifest_row_digests": manifest_integrity,
        "manifest_exact_coverage": len(manifest) == EXPECTED_ROWS and set(manifest_by_id) == {row["item_id"] for row in material},
        "manifest_links": manifest_links,
        "native_token_contract": token_rows and contract["interface"]["candidate_token_ids"] == EXPECTED_CANDIDATE_IDS,
        "tokenizer_gate_summary": contract["source_artifacts"]["tokenizer_summary"]["tokenizer_gate"] is True,
        "program_audit_self_digest": program["program_audit_digest"] == digest(strip_digest(program, "program_audit_digest")),
        "program_accuracies_recomputed": all(abs(computed_program[name] - expected_program[name]) < 1e-12 for name in expected_program),
        "program_artifact_matches": computed_program == program["empirical_bayes_accuracy"],
        "alternative_ceiling": abs(program_metrics["maximum_registered_alternative_accuracy"] - 2 / 3) < 1e-12,
        "program_margin_separated": program_metrics["maximum_registered_alternative_accuracy"] + THRESHOLDS["Q1_program_ceiling_margin"] < THRESHOLDS["Q1_split_accuracy"],
        "collision_group_counts": program_metrics["collision_group_counts"] == program["collision_groups"],
        "collision_constructs": all(program_metrics["collision_checks"].values()),
        "program_gate": program["program_identifiability_gate"] is True,
        "batch_plan_self_digest": plan["plan_digest"] == digest(strip_digest(plan, "plan_digest")),
        "batch_plan_partition": len(planned) == EXPECTED_ROWS and len(set(planned)) == EXPECTED_ROWS and set(planned) == set(manifest_by_id),
        "batch_plan_fixed": plan["batch_size"] == 16 and plan["adaptive_fallback"] is False,
        "thresholds_exact": contract["thresholds"] == THRESHOLDS,
        "behavior_outputs_absent": not RAW_PATH.exists() and not RUN_SUMMARY_PATH.exists(),
        "no_hidden_or_attention": contract["execution"]["hidden_states"] is False and contract["execution"]["attentions"] is False,
        "no_intervention": contract["execution"]["intervention"] is False,
        "behavior_only_claim": any("does not prove" in item for item in contract["claim_boundary"]),
    }
    value = audit_value(
        "independent_preaudit",
        checks,
        {
            "independent_program_metrics": program_metrics,
            "contract_digest": contract["contract_digest"],
            "material_digest": digest(material),
            "manifest_digest": digest(manifest),
        },
    )
    write_json(PREAUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def result_audit() -> None:
    if RESULT_AUDIT_PATH.exists():
        raise RuntimeError("result audit already exists")
    contract = read_json(CONTRACT_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_AUDIT_PATH)
    plan = read_json(BATCH_PLAN_PATH)
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    manifest_by_id = {row["item_id"]: row for row in manifest}
    candidate_ids = {candidate: ids[0] for candidate, ids in EXPECTED_CANDIDATE_IDS.items()}
    row_checks: list[bool] = []
    for row in raw:
        source = manifest_by_id.get(row["item_id"])
        if source is None:
            row_checks.append(False)
            continue
        scores = row["candidate_scores"]
        finite_scores = all(math.isfinite(float(value)) for value in scores.values())
        ordered = sorted(CANDIDATES, key=lambda candidate: scores[candidate], reverse=True)
        top_margin = scores[ordered[0]] - scores[ordered[1]]
        finite = bool(row["all_vocab_logits_finite"]) and finite_scores
        prediction = None if (not finite or top_margin <= TIE_TOLERANCE) else ordered[0]
        gold = source["gold_candidate"]
        wrong_best = max(scores[candidate] for candidate in CANDIDATES if candidate != gold)
        row_checks.append(
            row["manifest_row_digest"] == source["manifest_row_digest"]
            and row["execution_index"] == source["execution_index"]
            and row["prediction"] == prediction
            and row["correct"] == (prediction == gold)
            and row["unresolved_tie"] == (finite and top_margin <= TIE_TOLERANCE)
            and abs(row["top_candidate_margin"] - top_margin) < 1e-6
            and abs(row["gold_margin"] - (scores[gold] - wrong_best)) < 1e-6
            and row["full_vocab_top1_is_gold_candidate"] == (row["full_vocab_top1_id"] == candidate_ids[gold])
            and row["behavior_row_digest"] == digest(strip_digest(row, "behavior_row_digest"))
        )
    ledgers = independent_ledgers(raw, float(program["maximum_registered_alternative_accuracy"]))
    precision = summary["precision_audit"]
    checks = {
        "preaudit_pass": read_json(PREAUDIT_PATH).get("all_checks_passed") is True,
        "contract_self_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "sources_still_frozen": contract["source_hashes"] == {"execution": file_sha256(MAIN_SCRIPT), "independent_audit": file_sha256(Path(__file__).resolve())},
        "case_count": len(raw) == EXPECTED_ROWS,
        "case_identity": {row["item_id"] for row in raw} == set(manifest_by_id) and len({row["item_id"] for row in raw}) == EXPECTED_ROWS,
        "execution_order": [row["execution_index"] for row in raw] == list(range(EXPECTED_ROWS)),
        "row_recomputation": all(row_checks),
        "raw_digest": summary["raw_digest"] == digest(raw),
        "summary_self_digest": summary["summary_digest"] == digest(strip_digest(summary, "summary_digest")),
        "contract_link": summary["contract_digest"] == contract["contract_digest"],
        "batch_plan_link": summary["batch_plan_digest"] == plan["plan_digest"],
        "cuda_runtime": bool(summary.get("cuda_runtime")) and bool(summary.get("gpu")),
        "fp16_unquantized": set(precision["parameter_dtypes"]) == {"float16"} and not precision["has_quantized_modules"],
        "no_hidden_or_attention": summary["hidden_states_saved"] is False and summary["attentions_saved"] is False,
        "no_intervention": summary["interventions_performed"] is False,
        "ledger_denominators_defined": all(not math.isnan(value) for value in ledgers["Q2"]["target_change_triplet_success"].values()),
    }
    value = audit_value(
        "independent_result_audit",
        checks,
        {
            "recomputed_ledgers": ledgers,
            "ledgers_digest": digest(ledgers),
            "raw_digest": digest(raw),
            "behavior_gate_passed": ledgers["behavior_gate"],
            "hidden_eligibility": ledgers["hidden_eligibility"],
        },
    )
    write_json(RESULT_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def final_audit() -> None:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("final audit already exists")
    preaudit_value = read_json(PREAUDIT_PATH)
    result_value = read_json(RESULT_AUDIT_PATH)
    final = read_json(FINAL_PATH)
    passed = bool(result_value["recomputed_ledgers"]["hidden_eligibility"])
    natural_passed = bool(result_value["recomputed_ledgers"]["natural_first_token_gate"])
    checks = {
        "preaudit_pass": preaudit_value.get("all_checks_passed") is True,
        "result_audit_pass": result_value.get("all_checks_passed") is True,
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "ledger_recomputation": final["ledgers"] == result_value["recomputed_ledgers"],
        "result_audit_link": final["result_audit_digest"] == result_value["audit_digest"],
        "status_typed": final["status"] == ("behavior_and_construct_gate_passed" if passed else "one_shot_behavior_gate_failed"),
        "k208_typed": final["k_item"]["identifier"] == "K208" and final["k_item"]["evidence_grade"] == ("E3-BEHAVIOR-CONSTRUCT" if passed else "E3-NEGATIVE-BOUNDARY"),
        "scope_is_behavior_only": "behavior only" in final["k_item"]["scope"],
        "behavior_authorization": final["authorization"]["behavior_object"] is passed,
        "construct_authorization": final["authorization"]["registered_program_construct"] is True,
        "natural_authorization": final["authorization"]["natural_first_token_claim"] is natural_passed,
        "no_unique_algorithm_claim": final["authorization"]["unique_neural_algorithm_claim"] is False,
        "no_hidden_in_phase": final["authorization"]["hidden_scan_in_this_phase"] is False,
        "no_cross_model": final["authorization"]["cross_model_run"] is False,
        "no_auto_continue": final["authorization"]["auto_continue"] is False,
        "next_experiment_typed": (final["authorization"]["next_experiment"] is not None) is passed,
        "no_new_mathematics_claim": final["new_mathematics_required"] is False,
    }
    value = audit_value(
        "independent_final_audit",
        checks,
        {
            "final_digest": final["final_digest"],
            "result_audit_digest": result_value["audit_digest"],
            "behavior_gate_passed": passed,
            "next_experiment_authorized": final["authorization"]["next_experiment"],
        },
    )
    write_json(FINAL_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result", "final"))
    stage = parser.parse_args().stage
    {"preaudit": preaudit, "result": result_audit, "final": final_audit}[stage]()


if __name__ == "__main__":
    main()
