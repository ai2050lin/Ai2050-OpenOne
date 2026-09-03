#!/usr/bin/env python3
"""Independent audit for Phase1460 C078 contract."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1460_c078_colon_label_contract"


def main() -> None:
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "active": len(active) == protocol["material"]["active_count"] == 3456,
        "surfaces": Counter(row["surface"] for row in active) == {surface: 1728 for surface in protocol["surfaces"]},
        "truth": Counter(row["truth"] for row in active) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_label"] == row["query_label"]) for row in active),
        "composition": len(composition) == protocol["material"]["composition_count"] == 216,
        "compiled": len(compiled) == 3456 and all(len(row["role_positions"]) == 9 for row in compiled),
        "hashes": core.sha(OUT / "material/active_cases.jsonl") == protocol["material"]["active_sha256"] and core.sha(OUT / "material/composition_sets.jsonl") == protocol["material"]["composition_sha256"],
        "eligible_rule": protocol["capture"]["eligible_rule"].startswith("all sixteen"),
        "raw_scope": protocol["capture"]["no_pooling"] and protocol["capture"]["no_coordinate_selection"],
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "PCA", "TDA", "learned probe")),
        "claim_boundary": "unlabeled relation semantics" in protocol["claim_boundary"]["forbidden"],
        "hidden": preaudit["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == protocol["authorization"] == "run_phase1461_c078_behavior",
    }
    result = {"phase": 1460, "campaign": "C078", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
