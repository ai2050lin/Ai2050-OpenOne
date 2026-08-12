#!/usr/bin/env python3
"""Independent zero-model audit for Phase 1229."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1229
SCRIPT = TEST_ROOT / "phase1229_deanswer_clock_compass_material_contract.py"
AUDIT_SCRIPT = Path(__file__).resolve()
SOURCE_ROOT = TEST_ROOT / "result/phase1228_known_truth_automorphism_quotient_camera_revision1"
SOURCE_FINAL = SOURCE_ROOT / "analysis/final.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit/independent_result_audit.json"
EXPECTED_SOURCE_FINAL = "3c884d130da36ddc6d6a3208e080de8bb83f4ec3493161b803d82058e7752b81"
EXPECTED_SOURCE_AUDIT = "0332256a24c33624b9ec0d646264885a1f38ad53e573b3831070948c1a90abdd"

OUT_ROOT = TEST_ROOT / "result/phase1229_deanswer_clock_compass_material_contract"
CONTRACT_PATH = OUT_ROOT / "protocol/material_contract.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
MATERIAL_PATH = OUT_ROOT / "material/clock_compass_binding.jsonl"
DONOR_PATH = OUT_ROOT / "material/donor_registry.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/readiness_summary.json"
MATERIAL_AUDIT_PATH = OUT_ROOT / "audit/independent_material_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use")
PANELS = ("active", "matched_null", "surface_order")
STATES = (0, 1, 2, 3)
CLOCK_VALUES = ("twelve o'clock", "three o'clock", "six o'clock", "nine o'clock")
COMPASS_VALUES = ("north", "east", "south", "west")
CLOCK_TO_COMPASS = dict(zip(CLOCK_VALUES, COMPASS_VALUES))
EXPECTED_ROW_COUNT = 9216
EXPECTED_ROWS_PER_SPLIT = 3072
EXPECTED_ACTIVE_COUNT = 3072
EXPECTED_DONOR_COUNT = 3072
CHANCE = 0.25
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def token_multiset_digest(text: str) -> str:
    return digest(sorted(lexical_tokens(text)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bayes_accuracy(rows: list[dict[str, Any]], feature: Any) -> float:
    groups: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = feature(row)
        if isinstance(value, (list, dict, tuple)):
            key: Hashable = canonical_json(value)
        else:
            key = value
        groups[key][row["gold_candidate"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def record_clock_sequence(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(row["assignments"][entity] for entity in row["record_order_entities"])


def bundle_semantics(rows: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["bundle_id"]].append(row)
    active_ok = []
    null_ok = []
    surface_ok = []
    group_size_ok = []
    unique_prompt_ok = []
    for values in groups.values():
        values.sort(key=lambda row: int(row["binding_state"]))
        group_size_ok.append(len(values) == 4 and {int(row["binding_state"]) for row in values} == set(STATES))
        unique_prompt_ok.append(len({row["prompt"] for row in values}) == 4)
        panel = values[0]["panel"]
        bags = {row["prompt_token_multiset_digest"] for row in values}
        lengths = {int(row["prompt_char_length"]) for row in values}
        golds = {row["gold_candidate"] for row in values}
        if panel == "active":
            active_ok.append(bags == {values[0]["prompt_token_multiset_digest"]} and len(lengths) == 1 and golds == set(COMPASS_VALUES))
        elif panel == "matched_null":
            target_clocks = {row["gold_clock_value"] for row in values}
            null_ok.append(len(bags) == 1 and len(lengths) == 1 and len(golds) == 1 and len(target_clocks) == 1)
        elif panel == "surface_order":
            assignments = {canonical_json(row["assignments"]) for row in values}
            orders = {tuple(row["record_order_indices"]) for row in values}
            surface_ok.append(len(bags) == 1 and len(lengths) == 1 and len(golds) == 1 and len(assignments) == 1 and len(orders) == 4)
    checks = {
        "bundle_size_and_states": all(group_size_ok),
        "bundle_prompts_unique": all(unique_prompt_ok),
        "active_four_answer_counterfactual": all(active_ok),
        "matched_null_invariance": all(null_ok),
        "surface_order_invariance": all(surface_ok),
    }
    metrics = {
        "bundle_count": len(groups),
        "active_bundle_count": len(active_ok),
        "matched_null_bundle_count": len(null_ok),
        "surface_order_bundle_count": len(surface_ok),
    }
    return checks, metrics


def span_checks(rows: list[dict[str, Any]]) -> tuple[bool, int]:
    failures = 0
    required = {
        "record_full": 4,
        "record_object": 4,
        "record_anchor": 4,
        "record_relation": 4,
        "record_value": 4,
        "query_full": 1,
        "query_subject": 1,
        "query_anchor": 1,
        "query_relation": 1,
        "answer_boundary": 1,
    }
    for row in rows:
        prompt = row["prompt"]
        spans = row["spans"]
        if set(spans) != set(required) or any(len(spans[key]) != count for key, count in required.items()):
            failures += 1
            continue
        valid = True
        for values in spans.values():
            for start, end in values:
                valid = valid and 0 <= int(start) < int(end) <= len(prompt)
        full_records = [prompt[start:end] for start, end in spans["record_full"]]
        record_objects = [prompt[start:end] for start, end in spans["record_object"]]
        record_anchors = [prompt[start:end] for start, end in spans["record_anchor"]]
        record_values = [prompt[start:end] for start, end in spans["record_value"]]
        query_full = prompt[spans["query_full"][0][0]:spans["query_full"][0][1]]
        query_subject = prompt[spans["query_subject"][0][0]:spans["query_subject"][0][1]]
        query_anchor = prompt[spans["query_anchor"][0][0]:spans["query_anchor"][0][1]]
        answer_boundary = prompt[spans["answer_boundary"][0][0]:spans["answer_boundary"][0][1]]
        valid = valid and full_records == row["rendered_records"]
        valid = valid and record_objects == row["record_order_entities"]
        valid = valid and set(record_anchors) == {row["anchor"]}
        valid = valid and record_values == list(record_clock_sequence(row))
        valid = valid and query_full == row["query"]
        valid = valid and query_subject == row["target_entity"] and query_anchor == row["anchor"]
        valid = valid and answer_boundary == "Answer:"
        if not valid:
            failures += 1
    return failures == 0, failures


def donor_checks(rows: list[dict[str, Any]], donors: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, int]]:
    by_id = {row["item_id"]: row for row in rows}
    recipients = {row["recipient_id"] for row in donors}
    active_ids = {row["item_id"] for row in rows if row["panel"] == "active"}
    link_ids_exist = True
    counterfactual_ok = True
    wrong_ok = True
    same_answer_ok = True
    null_ok = True
    surface_ok = True
    no_cross_split = True
    for donor in donors:
        recipient = by_id[donor["recipient_id"]]
        link_names = [
            *donor["counterfactual_active_ids"],
            donor["wrong_answer_same_bundle_id"],
            donor["same_answer_wrong_binding_id"],
            donor["matched_null_id"],
            donor["surface_order_id"],
        ]
        if any(item_id not in by_id for item_id in link_names):
            link_ids_exist = False
            continue
        counterfactuals = [by_id[item_id] for item_id in donor["counterfactual_active_ids"]]
        counterfactual_ok = counterfactual_ok and len(counterfactuals) == 3
        counterfactual_ok = counterfactual_ok and all(
            row["panel"] == "active" and row["bundle_id"] == recipient["bundle_id"]
            and row["gold_candidate"] != recipient["gold_candidate"]
            for row in counterfactuals
        )
        counterfactual_ok = counterfactual_ok and {
            row["gold_candidate"] for row in counterfactuals
        } == set(COMPASS_VALUES) - {recipient["gold_candidate"]}
        wrong = by_id[donor["wrong_answer_same_bundle_id"]]
        wrong_ok = wrong_ok and wrong["bundle_id"] == recipient["bundle_id"] and wrong["gold_candidate"] != recipient["gold_candidate"]
        same = by_id[donor["same_answer_wrong_binding_id"]]
        same_answer_ok = same_answer_ok and (
            same["panel"] == "active"
            and same["split"] == recipient["split"]
            and same["template_id"] == recipient["template_id"]
            and same["gold_candidate"] == recipient["gold_candidate"]
            and same["world_index"] != recipient["world_index"]
            and same["target_entity"] != recipient["target_entity"]
        )
        matched = by_id[donor["matched_null_id"]]
        surface = by_id[donor["surface_order_id"]]
        shared_fields = ("split", "world_index", "target_index", "template_id", "order_variant", "mapping_variant", "binding_state")
        null_ok = null_ok and matched["panel"] == "matched_null" and all(matched[key] == recipient[key] for key in shared_fields)
        surface_ok = surface_ok and surface["panel"] == "surface_order" and all(surface[key] == recipient[key] for key in shared_fields)
        no_cross_split = no_cross_split and all(by_id[item_id]["split"] == recipient["split"] for item_id in link_names)
    checks = {
        "donor_row_digests": all(row["row_digest"] == digest(strip_digest(row, "row_digest")) for row in donors),
        "donor_recipients_exact_active": recipients == active_ids,
        "donor_link_ids_exist": link_ids_exist,
        "counterfactual_links": counterfactual_ok,
        "wrong_answer_same_bundle": wrong_ok,
        "same_answer_wrong_binding": same_answer_ok,
        "matched_null_links": null_ok,
        "surface_order_links": surface_ok,
        "donor_no_cross_split": no_cross_split,
    }
    return checks, {"recipient_count": len(recipients), "active_id_count": len(active_ids)}


def preaudit() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    source_final = read_json(SOURCE_FINAL)
    source_audit = read_json(SOURCE_AUDIT)
    source_text = SCRIPT.read_text(encoding="utf-8")
    checks = {
        "phase": contract.get("phase") == PHASE,
        "contract_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "generator_hash": contract["source_hashes"]["generator"] == file_sha256(SCRIPT),
        "audit_hash": contract["source_hashes"]["independent_audit"] == file_sha256(AUDIT_SCRIPT),
        "source_final": source_final.get("final_digest") == EXPECTED_SOURCE_FINAL and source_final["result"]["camera_gate"] is True,
        "source_audit": source_audit.get("audit_digest") == EXPECTED_SOURCE_AUDIT and source_audit.get("all_checks_passed") is True,
        "source_sha256": (
            contract["source"]["phase1228_final_sha256"] == file_sha256(SOURCE_FINAL)
            and contract["source"]["phase1228_audit_sha256"] == file_sha256(SOURCE_AUDIT)
        ),
        "zero_model_contract": (
            contract["execution_scope"]["zero_model"] is True
            and contract["execution_scope"]["tokenizer_loaded"] is False
            and contract["execution_scope"]["model_loaded"] is False
        ),
        "expected_geometry": (
            contract["expected"]["row_count"] == EXPECTED_ROW_COUNT
            and contract["expected"]["rows_per_split"] == EXPECTED_ROWS_PER_SPLIT
            and contract["expected"]["donor_count"] == EXPECTED_DONOR_COUNT
        ),
        "four_state_active_contract": (
            contract["active_contract"]["same_prompt_token_multiset_across_states"] is True
            and contract["active_contract"]["gold_set_per_bundle"] == list(COMPASS_VALUES)
        ),
        "answer_load_prohibited": "place north/east/south/west in any prompt or record" in contract["prohibited"],
        "subgroup_correction": "need not be a subgroup" in contract["mathematical_corrections"]["threshold_permutations"],
        "quotient_types_separate": "distinct types" in contract["mathematical_corrections"]["quotient_types"],
        "basis_refinement_correct": "monotonically refines" in contract["mathematical_corrections"]["basis_refinement"],
        "left_action_inverse": "pi^{-1}" in contract["mathematical_corrections"]["group_action"],
        "gauge_scope_corrected": "not independently trained" in contract["mathematical_corrections"]["gauge_scope"],
        "no_model_imports": not any(term in source_text for term in ("import torch", "transformers", "AutoTokenizer", "AutoModel")),
        "formal_outputs_absent": not any(path.exists() for path in (MATERIAL_PATH, DONOR_PATH, SUMMARY_PATH, MATERIAL_AUDIT_PATH, FINAL_PATH, FINAL_AUDIT_PATH)),
        "pass_scope_limited": "protocol materialization only" in contract["authorization"]["pass_authorizes"],
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_preaudit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "contract_digest": contract["contract_digest"],
    }
    result["audit_digest"] = digest(result)
    write_json(PREAUDIT_PATH, result)
    return result


def material_audit() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    preaudit_result = read_json(PREAUDIT_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    donors = read_jsonl(DONOR_PATH)
    summary = read_json(SUMMARY_PATH)
    active = [row for row in rows if row["panel"] == "active"]
    split_counts = Counter(row["split"] for row in rows)
    panel_counts = Counter(row["panel"] for row in rows)
    split_panel_counts = Counter((row["split"], row["panel"]) for row in rows)
    split_panel_gold = Counter((row["split"], row["panel"], row["gold_candidate"]) for row in rows)
    split_panel_state = Counter((row["split"], row["panel"], int(row["binding_state"])) for row in rows)
    bundle_checks, bundle_metrics = bundle_semantics(rows)
    donor_result_checks, donor_metrics = donor_checks(rows, donors)
    spans_ok, span_failures = span_checks(rows)
    record_overlap = sum(
        int(any(candidate in lexical_tokens(row["records_text"]) for candidate in COMPASS_VALUES))
        for row in rows
    )
    prompt_overlap = sum(
        int(any(candidate in lexical_tokens(row["prompt"]) for candidate in COMPASS_VALUES))
        for row in rows
    )
    symbolic_failures = sum(
        int(
            row["gold_clock_value"] != row["assignments"][row["target_entity"]]
            or row["gold_candidate"] != CLOCK_TO_COMPASS[row["assignments"][row["target_entity"]]]
            or set(row["assignments"].values()) != set(CLOCK_VALUES)
        )
        for row in rows
    )
    token_digest_failures = sum(
        int(row["prompt_token_multiset_digest"] != token_multiset_digest(row["prompt"]))
        for row in rows
    )
    heuristic_features = {
        "constant": lambda row: "constant",
        "token_multiset": lambda row: row["prompt_token_multiset_digest"],
        "prompt_length": lambda row: row["prompt_char_length"],
        "target_entity": lambda row: row["target_entity"],
        "target_record_position": lambda row: row["target_record_position"],
        "first_clock_value": lambda row: record_clock_sequence(row)[0],
        "last_clock_value": lambda row: record_clock_sequence(row)[-1],
        "template": lambda row: row["template_id"],
        "world": lambda row: row["world_id"],
        "order_variant": lambda row: row["order_variant"],
        "binding_state": lambda row: row["binding_state"],
        "mapping_variant": lambda row: row["mapping_variant"],
    }
    heuristic_accuracy = {name: bayes_accuracy(active, function) for name, function in heuristic_features.items()}
    split_entities: dict[str, set[str]] = {}
    split_templates: dict[str, set[str]] = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        split_entities[split] = {
            name for row in selected for name in [row["anchor"], *row["entities"]]
        }
        split_templates[split] = {row["template_id"] for row in selected}
    entities_disjoint = all(
        not (split_entities[left] & split_entities[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    )
    templates_disjoint = all(
        not (split_templates[left] & split_templates[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    )
    checks: dict[str, bool] = {
        "preaudit_pass": preaudit_result.get("all_checks_passed") is True,
        "preaudit_digest": preaudit_result["audit_digest"] == digest(strip_digest(preaudit_result, "audit_digest")),
        "contract_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "source_immutability": (
            contract["source_hashes"]["generator"] == file_sha256(SCRIPT)
            and contract["source_hashes"]["independent_audit"] == file_sha256(AUDIT_SCRIPT)
        ),
        "row_count": len(rows) == EXPECTED_ROW_COUNT,
        "donor_count": len(donors) == EXPECTED_DONOR_COUNT,
        "row_ids_unique": len({row["item_id"] for row in rows}) == len(rows),
        "bundle_ids_typed": all(row["bundle_id"].startswith("b1229-") for row in rows),
        "row_self_digests": all(row["row_digest"] == digest(strip_digest(row, "row_digest")) for row in rows),
        "summary_self_digest": summary["summary_digest"] == digest(strip_digest(summary, "summary_digest")),
        "summary_material_digest": summary["material_digest"] == digest(rows),
        "summary_donor_digest": summary["donor_digest"] == digest(donors),
        "summary_file_hashes": summary["material_sha256"] == file_sha256(MATERIAL_PATH) and summary["donor_sha256"] == file_sha256(DONOR_PATH),
        "split_counts": set(split_counts.values()) == {EXPECTED_ROWS_PER_SPLIT} and set(split_counts) == set(SPLITS),
        "panel_counts": set(panel_counts.values()) == {EXPECTED_ACTIVE_COUNT} and set(panel_counts) == set(PANELS),
        "split_panel_balance": set(split_panel_counts.values()) == {1024},
        "split_panel_gold_balance": set(split_panel_gold.values()) == {256},
        "split_panel_state_balance": set(split_panel_state.values()) == {256},
        "candidate_registry_fixed": all(tuple(row["candidates"]) == COMPASS_VALUES for row in rows),
        "symbolic_solver": symbolic_failures == 0,
        "record_answer_overlap_zero": record_overlap == 0,
        "prompt_answer_overlap_zero": prompt_overlap == 0,
        "token_multiset_digests": token_digest_failures == 0,
        "span_registry": spans_ok,
        "entity_split_isolation": entities_disjoint,
        "template_split_isolation": templates_disjoint,
        "active_heuristics_at_chance": all(value <= CHANCE + 1e-12 for value in heuristic_accuracy.values()),
        "no_arbitrary_code": all(not any(code in row["prompt"].lower() for code in ("candidate a", "candidate b", "alpha", "beta")) for row in rows),
        "no_model_artifact": summary["model_loaded"] is False and summary["tokenizer_loaded"] is False,
        **bundle_checks,
        **donor_result_checks,
    }
    metrics: dict[str, Any] = {
        **bundle_metrics,
        **donor_metrics,
        "row_count": len(rows),
        "donor_count": len(donors),
        "record_answer_overlap_count": record_overlap,
        "prompt_answer_overlap_count": prompt_overlap,
        "symbolic_failure_count": symbolic_failures,
        "span_failure_count": span_failures,
        "token_digest_failure_count": token_digest_failures,
        "active_heuristic_bayes_accuracy": heuristic_accuracy,
        "split_counts": dict(split_counts),
        "panel_counts": dict(panel_counts),
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_material_audit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "metrics": metrics,
        "contract_digest": contract["contract_digest"],
        "material_digest": digest(rows),
        "donor_digest": digest(donors),
    }
    result["audit_digest"] = digest(result)
    write_json(MATERIAL_AUDIT_PATH, result)
    return result


def final_audit() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    material = read_json(MATERIAL_AUDIT_PATH)
    summary = read_json(SUMMARY_PATH)
    final = read_json(FINAL_PATH)
    checks = {
        "material_audit_pass": material.get("all_checks_passed") is True,
        "material_audit_digest": material["audit_digest"] == digest(strip_digest(material, "audit_digest")),
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "contract_link": final["contract_digest"] == contract["contract_digest"],
        "summary_link": final["summary_digest"] == summary["summary_digest"],
        "audit_link": final["material_audit_digest"] == material["audit_digest"],
        "formal_gate": final["result"]["material_gate"] is True and final["status"] == "deanswer_material_contract_passed",
        "no_k_inflation": final["k_ledger"]["new_item"] is None,
        "claim_boundary": "No tokenizer, model, hidden state, behavior, or causal mechanism was tested." in final["claim_boundary"],
        "auto_scope": (
            final["authorization"]["auto_continue"] == 1
            and final["authorization"]["model_execution_authorized"] is False
        ),
        "source_immutability": (
            contract["source_hashes"]["generator"] == file_sha256(SCRIPT)
            and contract["source_hashes"]["independent_audit"] == file_sha256(AUDIT_SCRIPT)
        ),
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_final_audit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "contract_digest": contract["contract_digest"],
        "final_digest": final["final_digest"],
    }
    result["audit_digest"] = digest(result)
    write_json(FINAL_AUDIT_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "material", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        result = preaudit()
    elif args.stage == "material":
        result = material_audit()
    else:
        result = final_audit()
    print(canonical_json({
        "stage": args.stage,
        "all_checks_passed": result["all_checks_passed"],
        "passed": result["passed_count"],
        "total": result["check_count"],
        "audit_digest": result["audit_digest"],
    }))


if __name__ == "__main__":
    main()
