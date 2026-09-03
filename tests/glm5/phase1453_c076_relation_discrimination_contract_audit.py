#!/usr/bin/env python3
"""Independent audit for Phase1453 C076 contract."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1453, "C076"
OUT = TESTS / "result/phase1453_c076_relation_discrimination_contract"


def main() -> None:
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    graph = core.load(OUT / "material/frozen_concept_graph.json")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "families": len(graph["families"]) == 6 and all(len(values) == 6 for values in graph["families"].values()),
        "relations": len(graph["relations"]) == 6 and graph["relations"] == protocol["relations"],
        "active": len(active) == 3456 and Counter(row["surface"] for row in active) == {surface: 1728 for surface in protocol["surfaces"]},
        "truth": Counter(row["truth"] for row in active) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_relation_id"] == row["query_relation_id"]) for row in active),
        "nuisance": all(Counter(row[key] for row in active) == {True: 1728, False: 1728} for key in ("entity_match", "object_match", "relation_match")),
        "composition": len(composition) == 216 and Counter(row["partition"] for row in composition) == {"response_discovery": 72, "confirmation": 72, "lockbox": 72},
        "compiled": len(compiled) == 3456 and all(len(row["role_positions"]) == 7 and all(len(value) == 1 for value in row["role_positions"].values()) for row in compiled),
        "hashes": core.sha(OUT / "material/active_cases.jsonl") == protocol["material"]["active_sha256"] and core.sha(OUT / "material/composition_sets.jsonl") == protocol["material"]["composition_sha256"],
        "discovery": protocol["discovery_capture"]["expected_case_count"] == 1152 and protocol["discovery_capture"]["no_holdout_access"] and protocol["discovery_capture"]["no_pooling"],
        "freeze": protocol["discovery_description"]["candidate_freeze_before_holdout"],
        "forbidden": all(term in protocol["forbidden"] for term in ("attention", "MLP", "parameters", "gradients", "PCA", "learned probe")),
        "hidden_not_accessed": preaudit["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1454_c076_behavior" and final["contract_sha256"] == protocol["contract_sha256"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
