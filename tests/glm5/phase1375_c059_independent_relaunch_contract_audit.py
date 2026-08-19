#!/usr/bin/env python3
"""Independent audit for Phase1375 C059 relaunch contract."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1375_c059_independent_relaunch_contract"
SCRIPT = TESTS / "phase1375_c059_independent_relaunch_contract.py"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    final = core.load(OUT / "analysis/final.json")
    concepts = core.load(OUT / "material/frozen_concept_graph.json")["concepts"]
    active = core.rows(OUT / "material/active_membership_cases.jsonl")
    status = core.rows(OUT / "material/status_cases.jsonl")
    pairs = core.rows(OUT / "material/candidate_pairs.jsonl")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    checks = {
        "concepts": len(concepts) == 48 and len({r["word"] for r in concepts}) == 48,
        "active": len(active) == 864 and Counter(r["truth"] for r in active) == {True: 432, False: 432},
        "status": len(status) == 288 and Counter(r["truth"] for r in status) == {True: 144, False: 144},
        "pairs": len(pairs) == 432,
        "preaudit": pre["all_checks_passed"] and pre["passed"] == pre["total"] == 16,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_membership_cases.jsonl") and
                  protocol["material"]["status_sha256"] == core.sha(OUT / "material/status_cases.jsonl") and
                  protocol["material"]["pair_sha256"] == core.sha(OUT / "material/candidate_pairs.jsonl"),
        "quota_frozen": protocol["material"]["eligible_cases_per_cell"] == 6 and
                        protocol["material"]["eligible_case_target"] == 216,
        "independent_branches": protocol["coordinate_groups"]["sufficiency_qualification_independent_of_reverse_qualification"],
        "hidden_only": "attention" in protocol["forbidden"] and "MLP" in protocol["forbidden"],
        "authorization": final["authorization"] == "run_phase1376_c059_behavior_qualification",
        "scripts_compile": True,
    }
    audit = {"phase": 1375, "campaign": "C059", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
