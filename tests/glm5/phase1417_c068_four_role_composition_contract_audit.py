#!/usr/bin/env python3
"""Independent audit for Phase1417."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1417_c068_four_role_composition_contract"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    roles = {"record_target", "record_family", "query_target", "query_family"}
    checks = {
        "preaudit": pre["all_checks_passed"],
        "active": len(active) == 4320 and Counter(row["cell"] for row in active) == {cell: 540 for cell in ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")},
        "truth": Counter(row["truth"] for row in active) == {True: 1080, False: 3240},
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in protocol["material"]["partitions"]},
        "compiled": len(compiled) == 4320 and all(all(len(row["role_positions"][role]) == 1 for role in roles) for row in compiled),
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl"),
        "state16_catalog": protocol["mechanism"]["state_index"] == 16 and protocol["mechanism"]["surface"] == "catalog",
        "bidirectional": set(protocol["mechanism"]["directions"]) == {"true_recipient", "false_recipient"},
        "quartet": set(protocol["camera"]["roles"]) == roles,
        "dual_ledger": set(protocol["evidence_levels"]) == {"graded", "discrete", "strong"},
        "no_search": all(item in protocol["forbidden"] for item in ("layer search", "subset search", "candidate search")),
        "hidden_not_accessed": pre["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1418_c068_behavior",
    }
    result = {"phase": 1417, "campaign": "C068", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
