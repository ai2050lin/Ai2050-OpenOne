#!/usr/bin/env python3
"""Independent audit for Phase1440 C073 preregistration."""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1440, "C073"
OUT = TESTS / "result/phase1440_c073_side_phase_contract"
ROLES = ("record_target", "record_family", "query_target", "query_family")


def ba(truths: list[bool], predictions: list[bool]) -> float:
    positive = [pred for truth, pred in zip(truths, predictions) if truth]
    negative = [pred for truth, pred in zip(truths, predictions) if not truth]
    return 0.5 * (sum(positive) / len(positive) + sum(not value for value in negative) / len(negative))


def main() -> None:
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    registry = core.rows(OUT / "material/permutation_registry.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    contrast = core.load(OUT / "material/matched_contrast.json")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    truths = [row["truth"] for row in active]
    by_id = {row["permutation_id"]: row for row in registry}
    p07, p23 = by_id["p07"], by_id["p23"]
    matched_fields = protocol["permutations"]["matched_fields"]
    order = {
        "evidence_first": all(max(row["role_positions"][r][0] for r in ROLES[:2]) < min(row["role_positions"][r][0] for r in ROLES[2:]) for row in compiled if row["surface"] == "evidence_first"),
        "question_first": all(max(row["role_positions"][r][0] for r in ROLES[2:]) < min(row["role_positions"][r][0] for r in ROLES[:2]) for row in compiled if row["surface"] == "question_first"),
    }
    zero = {
        "person": ba(truths, [row["record_target"] == row["query_target"] for row in active]),
        "group": ba(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact": ba(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
    }
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "active": len(active) == 2880 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in protocol["surfaces"]},
        "truth": Counter(truths) == {True: 720, False: 2160},
        "semantic": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "naturalness": all(row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {"response_discovery": 24, "confirmation": 24, "lockbox": 24},
        "compiled": len(compiled) == 2880 and all(len(row["role_positions"]) == 5 for row in compiled),
        "reversed_order": all(order.values()),
        "permutations": {tuple(row["source_indices_by_target"]) for row in registry} == {tuple(value) for value in permutations(range(4))},
        "matched_invariants": all(p07[key] == p23[key] for key in matched_fields),
        "semantic_opposition": p07["semantic_side_preserving"] and p23["semantic_side_crossing"],
        "physical_reversal": all(p07["physical_phase_preserved_by_route"][route] == 0 and p23["physical_phase_preserved_by_route"][route] == 4 for route in protocol["mechanism"]["reversed_routes"]),
        "same_order_control": all(p07["physical_phase_preserved_by_route"][route] == 4 and p23["physical_phase_preserved_by_route"][route] == 0 for route in protocol["mechanism"]["same_order_routes"]),
        "contrast_copy": contrast["semantic_side_arm"]["permutation_id"] == "p07" and contrast["physical_phase_arm"]["permutation_id"] == "p23",
        "zero_models": abs(zero["person"] - 5 / 6) < 1e-12 and abs(zero["group"] - 5 / 6) < 1e-12 and zero["exact"] == 1.0,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl") and protocol["permutations"]["registry_sha256"] == core.sha(OUT / "material/permutation_registry.jsonl"),
        "fixed_object": protocol["state_index"] == 16 and protocol["model"] == "qwen3-bfloat16-cuda-no-quantization",
        "classification": set(protocol["classification"]) == {"semantic_side_confirmed", "physical_phase_confirmed", "conditional_semantic_side", "conditional_physical_phase", "mixed_or_no_stable_separation", "executor_failed"},
        "forbidden": all(term in protocol["forbidden"] for term in ("attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "layer search", "coordinate search")),
        "hidden_not_accessed": preaudit["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1441_c073_behavior" and final["contract_sha256"] == protocol["contract_sha256"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
