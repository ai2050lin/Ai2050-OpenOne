#!/usr/bin/env python3
"""Independent pre/final audit for Phase1290 C028 behavior qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "tests/glm5/result/phase1289_c028_typed_complement_contract"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_MATERIAL = PARENT / "material/frozen_typed_complement_material.jsonl"
PARENT_REVIEW = PARENT / "material/pre_model_semantic_naturalness_review.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1290_c028_qwen3_typed_complement_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GENERATIONS = OUT / "raw/confirmation_generations.jsonl"
RUN_SUMMARY = OUT / "analysis/run_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"
AUDIT = OUT / "audit/independent_final_audit.json"

PARTITIONS = ("discovery", "selection", "confirmation")
FAMILIES = ("case_record", "lab_log", "field_report")
SURFACES = tuple(f"{family}_{variant}" for family in FAMILIES for variant in ("a", "b"))
PANELS = ("identity", "single_complement", "double_complement", "lexical_null", "scope_null")
ACTIVE = ("identity", "single_complement", "double_complement")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def mean_bool(values: Iterable[bool]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def preaudit() -> None:
    protocol = read_json(PROTOCOL)
    parent = read_json(PARENT_PROTOCOL)
    parent_final = read_json(PARENT_FINAL)
    parent_audit = read_json(PARENT_AUDIT)
    environment = read_json(ENVIRONMENT)
    checks = []
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    checks.append(check("protocol_digest_recomputes", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"]))
    checks.append(check("parent_audit_passed", parent_audit["all_checks_passed"] and parent_audit["authorization"] == "phase1290_qwen3_typed_complement_behavior", parent_audit["authorization"]))
    checks.append(check("parent_final_authorizes", parent_final["authorization"] == "phase1290_qwen3_typed_complement_behavior", parent_final["authorization"]))
    checks.append(check("dependencies_exact", protocol["dependencies"] == {
        "phase1289_protocol": file_sha256(PARENT_PROTOCOL),
        "phase1289_material": file_sha256(PARENT_MATERIAL),
        "phase1289_review": file_sha256(PARENT_REVIEW),
        "phase1289_final": file_sha256(PARENT_FINAL),
        "phase1289_audit": file_sha256(PARENT_AUDIT),
    }, protocol["dependencies"]))
    checks.append(check("thresholds_inherited_exact", protocol["thresholds"] == parent["thresholds"], protocol["thresholds"]))
    checks.append(check("object_inherited_exact", protocol["research_object"] == parent["research_object"], protocol["research_object"]))
    checks.append(check("schema_exact", protocol["partitions"] == list(PARTITIONS) and protocol["surfaces"] == list(SURFACES) and protocol["panels"] == list(PANELS) and protocol["active_panels"] == list(ACTIVE), "frozen dimensions"))
    checks.append(check("single_fp16_model", protocol["model"] == "qwen3-4b-fp16-cuda-no-quantization" and protocol["formal_run_budget"] == 1, [protocol["model"], protocol["formal_run_budget"]]))
    checks.append(check("batch_sizes_frozen", protocol["scoring"]["batch_size"] == 16 and protocol["generation"]["batch_size"] == 8, [protocol["scoring"]["batch_size"], protocol["generation"]["batch_size"]]))
    checks.append(check("generation_parser_frozen", protocol["generation"]["max_new_tokens"] == 12 and protocol["generation"]["do_sample"] is False, protocol["generation"]))
    checks.append(check("behavior_before_hidden", any("Behavior and exact free generation" in value for value in protocol["hard_stops"]), protocol["hard_stops"]))
    checks.append(check("failure_closes", protocol["branching"]["any_phase1290_behavior_or_generation_ledger_fails"] == "close_c028_without_hidden", protocol["branching"]))
    checks.append(check("source_hashes_match", protocol["source_hashes"] == {
        "main": file_sha256(ROOT / "tests/glm5/phase1290_c028_qwen3_typed_complement_behavior.py"),
        "auditor": file_sha256(Path(__file__).resolve()),
    }, protocol["source_hashes"]))
    checks.append(check("cuda_fp16_environment", environment["cuda_available"] and torch.cuda.is_available(), environment))
    checks.append(check("unblinded_outputs_absent", not any(path.exists() for path in (RAW, GENERATIONS, RUN_SUMMARY, FINAL, COMPLETE)), [str(path) for path in (RAW, GENERATIONS, RUN_SUMMARY, FINAL, COMPLETE) if path.exists()]))
    result = {
        "phase": 1290, "campaign": "C028", "mode": "preaudit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
        "checks": checks, "passed_count": sum(value["passed"] for value in checks), "total_count": len(checks),
        "all_checks_passed": all(value["passed"] for value in checks),
        "authorization": "one_formal_qwen3_behavior_run" if all(value["passed"] for value in checks) else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    atomic_json(PREAUDIT, result)
    print(canonical_json({"mode": "preaudit", "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def recompute_behavior(raw: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    partitions = {part: mean_bool(v["correct"] for v in raw if v["partition"] == part) for part in PARTITIONS}
    surfaces = {surface: mean_bool(v["correct"] for v in raw if v["surface"] == surface) for surface in SURFACES}
    active = {panel: {
        "accuracy": mean_bool(v["correct"] for v in raw if v["panel"] == panel),
        "median_gold_margin": float(np.median([v["gold_margin"] for v in raw if v["panel"] == panel])),
    } for panel in ACTIVE}
    base = {str(side): mean_bool(v["correct"] for v in raw if v["base_side"] == side) for side in (0, 1)}
    grouped = defaultdict(dict)
    for value in raw:
        grouped[(value["row_id"], value["surface"])][value["panel"]] = value
    triple = mean_bool(all(values[panel]["correct"] for panel in ACTIVE) for values in grouped.values())
    ident_double = mean_bool(values["identity"]["correct"] and values["double_complement"]["correct"] and values["identity"]["prediction"] == values["double_complement"]["prediction"] for values in grouped.values())
    ident_single = mean_bool(values["identity"]["correct"] and values["single_complement"]["correct"] and values["identity"]["prediction"] != values["single_complement"]["prediction"] for values in grouped.values())
    lexical = mean_bool(values["identity"]["correct"] and values["lexical_null"]["correct"] and values["identity"]["prediction"] == values["lexical_null"]["prediction"] for values in grouped.values())
    scope = mean_bool(values["identity"]["correct"] and values["scope_null"]["correct"] and values["identity"]["prediction"] == values["scope_null"]["prediction"] for values in grouped.values())
    by_key = {(v["row_id"], v["surface"], v["panel"]): v for v in raw}
    variant = {}
    row_ids = {v["row_id"] for v in raw}
    for family in FAMILIES:
        for panel in PANELS:
            variant[f"{family}.{panel}"] = mean_bool(
                by_key[(row_id, f"{family}_a", panel)]["correct"]
                and by_key[(row_id, f"{family}_b", panel)]["correct"]
                and by_key[(row_id, f"{family}_a", panel)]["prediction"] == by_key[(row_id, f"{family}_b", panel)]["prediction"]
                for row_id in row_ids
            )
    active_rows = [value for value in raw if value["panel"] in ACTIVE]
    shortcut = {name: mean_bool(value["shortcut_predictions"][name] == value["gold_label"] for value in active_rows) for name in active_rows[0]["shortcut_predictions"]}
    finite = mean_bool(np.isfinite([*v["total_log_prob"].values(), *v["mean_log_prob"].values()]).all() for v in raw)
    metrics = {
        "finite_fraction": finite,
        "overall_candidate_accuracy": mean_bool(v["correct"] for v in raw),
        "partition_accuracy": partitions, "surface_accuracy": surfaces, "active_panel": active,
        "active_triple_all_correct_rate": triple,
        "identity_double_both_correct_rate": ident_double,
        "identity_single_opposition_both_correct_rate": ident_single,
        "lexical_null_preservation_rate": lexical, "scope_null_preservation_rate": scope,
        "surface_variant_cells": variant, "base_side_accuracy": base,
        "shortcut_program_accuracy_active": shortcut, "shortcut_program_ceiling_active": max(shortcut.values()),
    }
    gates = {
        "finite": finite >= thresholds["finite_fraction_min"],
        "overall_candidate": metrics["overall_candidate_accuracy"] >= thresholds["overall_candidate_accuracy_min"],
        "partition_candidate": min(partitions.values()) >= thresholds["partition_candidate_accuracy_min"],
        "surface_candidate": min(surfaces.values()) >= thresholds["surface_candidate_accuracy_min"],
        "active_panel_accuracy": min(value["accuracy"] for value in active.values()) >= thresholds["active_panel_accuracy_min"],
        "active_panel_margin": min(value["median_gold_margin"] for value in active.values()) >= thresholds["median_gold_margin_per_active_panel_min"],
        "active_triple": triple >= thresholds["active_triple_all_correct_rate_min"],
        "identity_double": ident_double >= thresholds["identity_double_both_correct_rate_min"],
        "identity_single": ident_single >= thresholds["identity_single_opposition_both_correct_rate_min"],
        "lexical_null": lexical >= thresholds["lexical_null_preservation_rate_min"],
        "scope_null": scope >= thresholds["scope_null_preservation_rate_min"],
        "surface_variants": min(variant.values()) >= thresholds["surface_variant_both_correct_rate_min"],
        "base_sides": min(base.values()) >= thresholds["base_side_accuracy_min"],
        "shortcut_ceiling": max(shortcut.values()) <= thresholds["shortcut_program_accuracy_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def recompute_generation(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    cells = {}
    for surface in SURFACES:
        for panel in ACTIVE:
            subset = [value for value in rows if value["surface"] == surface and value["panel"] == panel]
            cells[f"{surface}.{panel}"] = {
                "n": len(subset),
                "coverage": mean_bool(value["covered"] for value in subset),
                "label_accuracy": mean_bool(value["label_correct"] for value in subset),
                "exact_sentence_accuracy": mean_bool(value["exact_sentence"] for value in subset),
            }
    grouped = defaultdict(dict)
    for value in rows:
        grouped[(value["row_id"], value["surface"])][value["panel"]] = value
    triple = mean_bool(all(values[panel]["exact_sentence"] for panel in ACTIVE) for values in grouped.values())
    coverage = min(value["coverage"] for value in cells.values())
    exact = min(value["exact_sentence_accuracy"] for value in cells.values())
    gates = {
        "coverage": coverage >= thresholds["generation_coverage_min"],
        "exact_accuracy": exact >= thresholds["generation_exact_accuracy_min"],
        "active_triple": triple >= thresholds["generation_active_triple_rate_min"],
    }
    return {"cells": cells, "coverage_min": coverage, "exact_accuracy_min": exact, "active_triple_exact_rate": triple, "gates": gates, "passed": all(gates.values())}


def final_audit() -> None:
    protocol = read_json(PROTOCOL)
    raw = read_jsonl(RAW)
    generations = read_jsonl(GENERATIONS)
    summary = read_json(RUN_SUMMARY)
    final = read_json(FINAL)
    complete = read_json(COMPLETE)
    thresholds = protocol["thresholds"]
    behavior = recompute_behavior(raw, thresholds)
    generation = recompute_generation(generations, thresholds)
    ledgers = {"candidate_behavior": behavior["passed"], "natural_generation": generation["passed"]}
    checks = []
    checks.append(check("preaudit_passed", read_json(PREAUDIT)["all_checks_passed"], read_json(PREAUDIT)["authorization"]))
    checks.append(check("protocol_unchanged", digest({k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}) == protocol["protocol_digest"], protocol["protocol_digest"]))
    checks.append(check("raw_counts", len(raw) == 4320 and len(generations) == 864, [len(raw), len(generations)]))
    checks.append(check("raw_unique_keys", len({(v["row_id"], v["surface"], v["panel"]) for v in raw}) == 4320 and len({(v["row_id"], v["surface"], v["panel"]) for v in generations}) == 864, "candidate and generation keys"))
    checks.append(check("raw_hashes", summary["raw_hashes"] == {"candidate_scores": file_sha256(RAW), "confirmation_generations": file_sha256(GENERATIONS)} == final["raw_hashes"], summary["raw_hashes"]))
    checks.append(check("complete_hashes", complete["run_summary_sha256"] == file_sha256(RUN_SUMMARY) and complete["final_sha256"] == file_sha256(FINAL), complete))
    checks.append(check("formal_run_budget", complete["formal_runs_used"] == 1, complete["formal_runs_used"]))
    checks.append(check("all_candidate_numbers_finite", all(np.isfinite([*v["total_log_prob"].values(), *v["mean_log_prob"].values(), v["gold_margin"]]).all() for v in raw), "all rows"))
    checks.append(check("candidate_prediction_recomputes", all((v["left_label"] if v["total_log_prob"]["left"] > v["total_log_prob"]["right"] else v["right_label"] if v["total_log_prob"]["right"] > v["total_log_prob"]["left"] else None) == v["prediction"] for v in raw), "all rows"))
    checks.append(check("candidate_correctness_recomputes", all((v["prediction"] == v["gold_label"]) == v["correct"] for v in raw), "all rows"))
    checks.append(check("generation_parser_fields_consistent", all((v["prediction"] == v["gold_label"] and v["covered"]) == v["label_correct"] and (v["expected_sentence"] == f"the final state is {v['gold_label']}.") for v in generations), "all rows"))
    checks.append(check("behavior_metrics_match", canonical_json(behavior) == canonical_json(summary["behavior"]), behavior))
    checks.append(check("generation_metrics_match", canonical_json(generation) == canonical_json(summary["generation"]), generation))
    checks.append(check("ledgers_match", ledgers == summary["ledgers"] == final["ledgers"], ledgers))
    expected_auth = "phase1291_multievent_future_response_contract" if all(ledgers.values()) else "close_c028_without_hidden"
    checks.append(check("authorization_matches_frozen_branch", final["authorization"] == expected_auth, [final["authorization"], expected_auth]))
    checks.append(check("no_hidden_or_other_models", final["hidden_measured"] is False and final["other_models_run"] is False, [final["hidden_measured"], final["other_models_run"]]))
    checks.append(check("precision_fp16_cuda", set(summary["precision_audit"]["parameter_dtypes"]) == {"float16"} and not summary["precision_audit"]["has_quantized_modules"] and not summary["precision_audit"]["has_bf16_parameters"], summary["precision_audit"]))
    all_passed = all(value["passed"] for value in checks)
    result = {
        "phase": 1290, "campaign": "C028", "mode": "final_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
        "checks": checks, "passed_count": sum(value["passed"] for value in checks), "total_count": len(checks),
        "all_checks_passed": all_passed,
        "recomputed_ledgers": ledgers,
        "authorization": expected_auth if all_passed else "audit_failure_no_authorization",
        "protocol_digest": protocol["protocol_digest"],
    }
    atomic_json(AUDIT, result)
    print(canonical_json({"mode": "final", "passed": result["passed_count"], "total": result["total_count"], "ledgers": ledgers, "authorization": result["authorization"]}))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.mode == "preaudit":
        preaudit()
    else:
        final_audit()
