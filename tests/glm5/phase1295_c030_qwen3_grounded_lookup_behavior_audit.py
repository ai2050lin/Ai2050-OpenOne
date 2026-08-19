#!/usr/bin/env python3
"""Independent pre- and post-run audit for Phase 1295/C030."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN = TEST_ROOT / "phase1295_c030_qwen3_grounded_lookup_behavior.py"
AUDITOR = Path(__file__).resolve()
PARENT = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_MATERIAL = PARENT / "material/frozen_grounded_lookup_cases.jsonl"
PARENT_NATURALNESS = PARENT / "material/pre_model_grammar_type_review.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
OUT = TEST_ROOT / "result/phase1295_c030_qwen3_grounded_lookup_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
POSTAUDIT = OUT / "audit/independent_final_audit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GENERATIONS = OUT / "raw/list_free_generations.jsonl"
SUMMARY = OUT / "analysis/run_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("discovery", "confirmation", "holdout")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SURFACES = ("catalog_prose", "inventory_ledger")
STATES = (0, 1)
EXPECTED_CANDIDATE = 6912
EXPECTED_GENERATION = 1536
EXPECTED_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "overall_candidate_accuracy_min": 0.95,
    "partition_candidate_accuracy_min": 0.94,
    "panel_candidate_accuracy_min": 0.93,
    "surface_candidate_accuracy_min": 0.93,
    "base_side_accuracy_min": 0.93,
    "active_pair_success_min": 0.90,
    "matched_null_pair_success_min": 0.90,
    "surface_only_pair_success_min": 0.90,
    "semantic_neighbor_pair_success_min": 0.90,
    "candidate_order_triple_success_min": 0.90,
    "cross_surface_pair_success_min": 0.90,
    "generation_coverage_min": 0.95,
    "generation_accuracy_min": 0.90,
    "generation_pair_success_min": 0.85,
    "shortcut_program_accuracy_max": 0.70,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def base_checks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    parent = load(PARENT_PROTOCOL)
    parent_final = load(PARENT_FINAL)
    parent_audit = load(PARENT_AUDIT)
    naturalness = load(PARENT_NATURALNESS)
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest_recomputes", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes_match", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(AUDITOR)}, protocol["source_hashes"])
    add(checks, "phase_campaign_exact", (protocol["phase"], protocol["campaign"]) == (1295, "C030"), [protocol["phase"], protocol["campaign"]])
    add(checks, "parent_authorization_exact", parent_final.get("authorization") == "phase1295_qwen3_behavior_only" and parent_audit.get("all_checks_passed") is True, parent_final)
    expected_dependencies = {
        "phase1294_protocol": sha(PARENT_PROTOCOL),
        "phase1294_material": sha(PARENT_MATERIAL),
        "phase1294_naturalness": sha(PARENT_NATURALNESS),
        "phase1294_final": sha(PARENT_FINAL),
        "phase1294_audit": sha(PARENT_AUDIT),
    }
    add(checks, "dependencies_match", protocol["dependencies"] == expected_dependencies, protocol["dependencies"])
    add(checks, "parent_material_hash_frozen", parent["material"]["material_sha256"] == sha(PARENT_MATERIAL), sha(PARENT_MATERIAL))
    add(checks, "naturalness_prequalified", naturalness["reviewed_before_any_c030_weight_load"] is True and naturalness["all_checks_passed"] is True and not naturalness["issues"], naturalness["limitation"])
    add(checks, "type_signature_exact", protocol["type_signature"] == "(WorldState, Attribute, Value) -> Entity", protocol["type_signature"])
    add(checks, "counts_exact", protocol["material_count"] == EXPECTED_CANDIDATE and protocol["generation_count"] == EXPECTED_GENERATION, [protocol["material_count"], protocol["generation_count"]])
    add(checks, "thresholds_exact", protocol["thresholds"] == EXPECTED_THRESHOLDS, protocol["thresholds"])
    add(checks, "single_qwen_fp16_run", protocol["model"] == "qwen3-4b-fp16-cuda-no-quantization" and protocol["formal_run_budget"] == 1, [protocol["model"], protocol["formal_run_budget"]])
    add(checks, "candidate_parser_frozen", protocol["candidate_scoring"]["continuation"] == "one leading-space entity-name token" and protocol["candidate_scoring"]["tie_policy"].startswith("tie is incorrect"), protocol["candidate_scoring"])
    add(checks, "generation_parser_frozen", protocol["generation"]["candidate_list_present"] is False and protocol["generation"]["partitions"] == ["confirmation", "holdout"] and protocol["generation"]["candidate_order"] == 0, protocol["generation"])
    add(checks, "behavior_before_hidden", any("No hidden state" in item for item in protocol["hard_stops"]), protocol["hard_stops"])
    add(checks, "failure_closes", protocol["authorization_if_fail"] == "close_c030_without_hidden" and any("failure closes C030" in item for item in protocol["hard_stops"]), protocol["authorization_if_fail"])
    return checks


def preaudit() -> None:
    protocol = load(PROTOCOL)
    checks = base_checks(protocol)
    no_outputs = not any(path.exists() for path in (RAW, GENERATIONS, SUMMARY, FINAL, COMPLETE))
    add(checks, "formal_run_not_started", no_outputs, no_outputs)
    passed = all(check["passed"] for check in checks)
    result = {
        "phase": 1295,
        "campaign": "C030",
        "audit_stage": "pre_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "total_count": len(checks),
        "all_checks_passed": passed,
        "authorization": "run_phase1295_once" if passed else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    save(PREAUDIT, result)
    print(canonical({"stage": "preaudit", "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


def rate(values: Iterable[bool]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def recompute_candidate(raw: list[dict[str, Any]], thresholds: dict[str, float], shortcut: float) -> dict[str, Any]:
    normalized = []
    for row in raw:
        scores = row["candidate_log_prob"]
        ordered = sorted(scores, key=lambda name: (-scores[name], name))
        prediction = ordered[0] if scores[ordered[0]] > scores[ordered[1]] else None
        gold = row["gold_candidate"]
        other = max(score for name, score in scores.items() if name != gold)
        normalized.append({**row, "prediction": prediction, "correct": prediction == gold, "gold_margin": float(scores[gold] - other), "finite": bool(all(np.isfinite(list(scores.values()))))})
    partition = {key: rate(row["correct"] for row in normalized if row["partition"] == key) for key in PARTITIONS}
    panel = {key: rate(row["correct"] for row in normalized if row["panel"] == key) for key in PANELS}
    surface = {key: rate(row["correct"] for row in normalized if row["surface"] == key) for key in SURFACES}
    states = {str(key): rate(row["correct"] for row in normalized if row["binding_state"] == key) for key in STATES}
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    surface_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        pair_groups[row["group_id"]].append(row)
        order_groups[(row["partition"], row["profile_index"], row["attribute"], row["panel"], row["surface"], row["binding_state"])].append(row)
        surface_groups[(row["partition"], row["profile_index"], row["attribute"], row["panel"], row["candidate_order"], row["binding_state"])].append(row)
    pairs = {
        panel_name: rate(len(group) == 2 and all(row["correct"] for row in group) for group in pair_groups.values() if group and group[0]["panel"] == panel_name)
        for panel_name in PANELS
    }
    order_success = rate(len(group) == 3 and all(row["correct"] for row in group) for group in order_groups.values())
    cross_surface = rate(len(group) == 2 and all(row["correct"] for row in group) for group in surface_groups.values())
    finite = rate(row["finite"] for row in normalized)
    overall = rate(row["correct"] for row in normalized)
    metrics = {
        "finite_fraction": finite,
        "overall_candidate_accuracy": overall,
        "partition_candidate_accuracy": partition,
        "panel_candidate_accuracy": panel,
        "surface_candidate_accuracy": surface,
        "binding_state_accuracy": states,
        "panel_pair_success": pairs,
        "candidate_order_triple_success": order_success,
        "cross_surface_pair_success": cross_surface,
        "median_gold_margin": float(np.median([row["gold_margin"] for row in normalized])),
        "shortcut_program_ceiling": shortcut,
    }
    gates = {
        "finite": finite >= thresholds["finite_fraction_min"],
        "overall_candidate": overall >= thresholds["overall_candidate_accuracy_min"],
        "partition_candidate": min(partition.values()) >= thresholds["partition_candidate_accuracy_min"],
        "panel_candidate": min(panel.values()) >= thresholds["panel_candidate_accuracy_min"],
        "surface_candidate": min(surface.values()) >= thresholds["surface_candidate_accuracy_min"],
        "binding_state": min(states.values()) >= thresholds["base_side_accuracy_min"],
        "active_pair": pairs["active"] >= thresholds["active_pair_success_min"],
        "matched_null_pair": pairs["matched_null"] >= thresholds["matched_null_pair_success_min"],
        "surface_only_pair": pairs["surface_only"] >= thresholds["surface_only_pair_success_min"],
        "semantic_neighbor_pair": pairs["semantic_neighbor"] >= thresholds["semantic_neighbor_pair_success_min"],
        "candidate_order_triple": order_success >= thresholds["candidate_order_triple_success_min"],
        "cross_surface_pair": cross_surface >= thresholds["cross_surface_pair_success_min"],
        "shortcut": shortcut <= thresholds["shortcut_program_accuracy_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def normalize_first_line(text: str) -> str:
    for line in text.replace("\r", "\n").split("\n"):
        value = line.strip().strip("\"' ").strip(".,:; ").strip()
        if value:
            return value.lower()
    return ""


def recompute_generation(raw: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    normalized = []
    for row in raw:
        hits = [candidate for candidate in row["candidates"] if re.search(rf"\b{re.escape(candidate)}\b", row["generation"], flags=re.IGNORECASE)]
        prediction = hits[0] if len(hits) == 1 else None
        first = normalize_first_line(row["generation"])
        normalized.append({**row, "covered": len(hits) == 1, "prediction": prediction, "label_correct": prediction == row["gold_candidate"], "exact_correct": first == row["gold_candidate"].lower()})
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        groups[(row["partition"], row["profile_index"], row["attribute"], row["panel"], row["surface"])].append(row)
    coverage = rate(row["covered"] for row in normalized)
    exact = rate(row["exact_correct"] for row in normalized)
    label = rate(row["label_correct"] for row in normalized)
    pair_success = rate(len(group) == 2 and all(row["exact_correct"] for row in group) for group in groups.values())
    cells = {}
    for partition in ("confirmation", "holdout"):
        for panel in PANELS:
            for surface in SURFACES:
                subset = [row for row in normalized if row["partition"] == partition and row["panel"] == panel and row["surface"] == surface]
                cells[f"{partition}|{panel}|{surface}"] = {"coverage": rate(row["covered"] for row in subset), "exact_accuracy": rate(row["exact_correct"] for row in subset)}
    metrics = {"coverage": coverage, "exact_accuracy": exact, "label_accuracy": label, "both_state_pair_success": pair_success, "cells": cells}
    gates = {
        "coverage": coverage >= thresholds["generation_coverage_min"],
        "accuracy": exact >= thresholds["generation_accuracy_min"],
        "pair_success": pair_success >= thresholds["generation_pair_success_min"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def postaudit() -> None:
    protocol = load(PROTOCOL)
    checks = base_checks(protocol)
    raw = read_jsonl(RAW)
    generations = read_jsonl(GENERATIONS)
    summary = load(SUMMARY)
    final = load(FINAL)
    complete = load(COMPLETE)
    material = read_jsonl(PARENT_MATERIAL)
    material_by_id = {row["case_id"]: row for row in material}

    add(checks, "raw_counts_exact", len(raw) == EXPECTED_CANDIDATE and len(generations) == EXPECTED_GENERATION, [len(raw), len(generations)])
    add(checks, "raw_ids_unique", len({row["case_id"] for row in raw}) == EXPECTED_CANDIDATE and len({row["case_id"] for row in generations}) == EXPECTED_GENERATION, "candidate and generation")
    add(checks, "raw_hashes_match", summary["raw_hashes"] == {"candidate_scores": sha(RAW), "list_free_generations": sha(GENERATIONS)} and final["raw_hashes"] == summary["raw_hashes"], summary["raw_hashes"])
    candidate_material_match = all(
        row["case_id"] in material_by_id
        and row["gold_candidate"] == material_by_id[row["case_id"]]["gold_candidate"]
        and row["candidates"] == material_by_id[row["case_id"]]["candidates"]
        for row in raw
    )
    add(checks, "candidate_rows_match_frozen_material", candidate_material_match, "all candidate rows")
    generation_material_match = all(
        row["case_id"] in material_by_id
        and material_by_id[row["case_id"]]["candidate_order"] == 0
        and material_by_id[row["case_id"]]["partition"] in {"confirmation", "holdout"}
        and row["gold_candidate"] == material_by_id[row["case_id"]]["gold_candidate"]
        for row in generations
    )
    add(checks, "generation_rows_match_frozen_subset", generation_material_match, "all generation rows")
    raw_fields_recompute = True
    for row in raw:
        scores = row["candidate_log_prob"]
        ordered = sorted(scores, key=lambda name: (-scores[name], name))
        prediction = ordered[0] if scores[ordered[0]] > scores[ordered[1]] else None
        other = max(value for name, value in scores.items() if name != row["gold_candidate"])
        raw_fields_recompute &= (
            row["prediction"] == prediction
            and row["correct"] == (prediction == row["gold_candidate"])
            and abs(row["gold_margin"] - (scores[row["gold_candidate"]] - other)) < 1e-10
            and row["finite"] is True
        )
    add(checks, "candidate_fields_recompute", raw_fields_recompute, "all rows")
    generation_fields_recompute = True
    for row in generations:
        hits = [candidate for candidate in row["candidates"] if re.search(rf"\b{re.escape(candidate)}\b", row["generation"], flags=re.IGNORECASE)]
        prediction = hits[0] if len(hits) == 1 else None
        generation_fields_recompute &= (
            row["candidate_hits"] == hits
            and row["covered"] == (len(hits) == 1)
            and row["prediction"] == prediction
            and row["label_correct"] == (prediction == row["gold_candidate"])
            and row["exact_correct"] == (normalize_first_line(row["generation"]) == row["gold_candidate"].lower())
        )
    add(checks, "generation_fields_recompute", generation_fields_recompute, "all rows")

    candidate = recompute_candidate(raw, protocol["thresholds"], float(protocol["zero_models"]["shortcut_ceiling"]))
    generation = recompute_generation(generations, protocol["thresholds"])
    add(checks, "candidate_summary_recomputes", canonical(candidate) == canonical(summary["candidate"]), candidate)
    add(checks, "generation_summary_recomputes", canonical(generation) == canonical(summary["generation"]), generation)
    expected_pass = candidate["passed"] and generation["passed"]
    expected_auth = "phase1296_multievent_response_preregistration_only" if expected_pass else "close_c030_without_hidden"
    add(checks, "verdict_and_authorization_recompute", summary["all_behavior_gates_passed"] == expected_pass and final["all_behavior_gates_passed"] == expected_pass and summary["authorization"] == expected_auth and final["authorization"] == expected_auth, expected_auth)
    model_audit = summary["model_audit"]
    add(checks, "fp16_no_quantization", model_audit["has_fp16_parameters"] is True and model_audit["has_quantized_modules"] is False and not model_audit["suspicious_quantized_module_classes"], model_audit)
    add(checks, "hidden_not_read", summary["hidden_states_read"] is False and final["hidden_states_read"] is False, False)
    add(checks, "formal_run_consumed_once", complete["formal_runs_consumed"] == 1 and complete["protocol_digest"] == protocol["protocol_digest"], complete)

    passed = all(check["passed"] for check in checks)
    result = {
        "phase": 1295,
        "campaign": "C030",
        "audit_stage": "post_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "total_count": len(checks),
        "all_checks_passed": passed,
        "scientific_authorization": expected_auth if passed else "none_due_to_audit_failure",
        "protocol_digest": protocol["protocol_digest"],
    }
    save(POSTAUDIT, result)
    print(canonical({"stage": "postaudit", "passed": result["passed_count"], "total": result["total_count"], "authorization": result["scientific_authorization"]}))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()


if __name__ == "__main__":
    main()
