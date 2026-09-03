#!/usr/bin/env python3
"""Independent audit for Phase1463 C079 contract."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1463_c079_aggregate_observation_contract"


def main() -> None:
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "active": len(active) == 3456,
        "truth": Counter(row["truth"] for row in active) == {True: 1728, False: 1728},
        "composition": len(composition) == 216,
        "compiled": len(compiled) == 3456 and all(len(row["role_positions"]) == 9 for row in compiled),
        "hashes": core.sha(OUT / "material/active_cases.jsonl") == protocol["material"]["active_sha256"] and core.sha(OUT / "material/composition_sets.jsonl") == protocol["material"]["composition_sha256"],
        "aggregate": all(key in protocol["behavior"] for key in ("global_surface_balanced_accuracy_min", "surface_partition_balanced_accuracy_min", "surface_truth_accuracy_min", "relation_surface_balanced_accuracy_min")),
        "no_sparse_conjunction": "excluded_redundant_gate" in protocol["behavior"] and "family_relation_surface_accuracy_min" not in protocol["behavior"],
        "eligible_exact": protocol["capture"]["eligible_rule"].startswith("all sixteen"),
        "observables": set(protocol["allowed_observables"]) == {"input embeddings", "all full-dimensional Hidden States", "yes/no logits"},
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "PCA", "TDA", "learned probe")),
        "hidden": preaudit["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1464_c079_behavior",
    }
    result = {"phase": 1463, "campaign": "C079", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
