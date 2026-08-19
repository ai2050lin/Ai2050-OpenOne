#!/usr/bin/env python3
"""Independent audit for Phase1435 C072 preregistration."""
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

OUT = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
ROLES = ("record_target", "record_family", "query_target", "query_family")


def ba(truths: list[bool], predictions: list[bool]) -> float:
    return 0.5 * (sum(pred for truth, pred in zip(truths, predictions) if truth) / sum(truths) + sum(not pred for truth, pred in zip(truths, predictions) if not truth) / sum(not truth for truth in truths))


def main() -> None:
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    registry = core.rows(OUT / "material/permutation_registry.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    truths = [row["truth"] for row in active]
    independent_zero = {
        "person": ba(truths, [row["record_target"] == row["query_target"] for row in active]),
        "circle": ba(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact": ba(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
    }
    expected = {tuple(value) for value in permutations(range(4))}
    observed = {tuple(row["source_indices_by_target"]) for row in registry}
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in protocol["surfaces"]}
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "active": len(active) == 2880 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in protocol["surfaces"]},
        "truth": Counter(truths) == {True: 720, False: 2160},
        "semantic": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {"response_discovery": 24, "confirmation": 24, "lockbox": 24},
        "compiled": len(compiled) == 2880 and all(len(row["role_positions"]) == 5 for row in compiled),
        "shapes": all(len(value) == 1 for value in lengths.values()) and len({next(iter(value)) for value in lengths.values()}) == 2,
        "permutations": len(registry) == 24 and observed == expected and sum(row["identity"] for row in registry) == 1,
        "strata": Counter(row["parity"] for row in registry) == {"even": 12, "odd": 12} and Counter(row["cycle_type"] for row in registry) == {"1-1-1-1": 1, "2-1-1": 6, "3-1": 8, "2-2": 3, "4": 6},
        "zero_models": abs(independent_zero["person"] - 5 / 6) < 1e-12 and abs(independent_zero["circle"] - 5 / 6) < 1e-12 and independent_zero["exact"] == 1.0,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl") and protocol["permutations"]["registry_sha256"] == core.sha(OUT / "material/permutation_registry.jsonl"),
        "fixed_object": protocol["state_index"] == 16 and protocol["permutations"]["count"] == 24,
        "classification": set(protocol["classification"]) == {"role_order_selective", "permutation_symmetric_multiset", "subgroup_structured", "heterogeneous_or_executor_failed"},
        "forbidden": all(term in protocol["forbidden"] for term in ("attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "layer search", "coordinate search")),
        "hidden_not_accessed": preaudit["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1436_c072_behavior" and final["contract_sha256"] == protocol["contract_sha256"],
    }
    result = {"phase": 1435, "campaign": "C072", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
