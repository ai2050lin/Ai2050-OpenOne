#!/usr/bin/env python3
"""Independent audit for Phase1380 C060 contract."""
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

OUT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
SCRIPT = TESTS / "phase1380_c060_conditional_coalition_campaign_contract.py"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    final = core.load(OUT / "analysis/final.json")
    concepts = core.load(OUT / "material/frozen_concept_graph.json")["concepts"]
    active = core.rows(OUT / "material/active_membership_cases.jsonl")
    status = core.rows(OUT / "material/status_cases.jsonl")
    pairs = core.rows(OUT / "material/candidate_pairs.jsonl")
    coalitions = core.load(OUT / "protocol/fixed_coalitions.json")["groups"]
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    inherited, complement = set(coalitions["inherited_S1024"]), set(coalitions["inherited_C1536"])
    checks = {
        "concepts": len(concepts) == 48 and len({r["word"] for r in concepts}) == 48,
        "active": len(active) == 864 and Counter(r["truth"] for r in active) == {True: 432, False: 432},
        "status": len(status) == 288 and Counter(r["truth"] for r in status) == {True: 144, False: 144},
        "pairs": len(pairs) == 432,
        "preaudit": pre["all_checks_passed"] and pre["passed"] == pre["total"] == 18,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_membership_cases.jsonl") and
                  protocol["material"]["status_sha256"] == core.sha(OUT / "material/status_cases.jsonl") and
                  protocol["material"]["pair_sha256"] == core.sha(OUT / "material/candidate_pairs.jsonl"),
        "coalition_hash": protocol["fixed_coalitions"]["artifact_sha256"] ==
                          core.sha(OUT / "protocol/fixed_coalitions.json"),
        "inherited_complement": len(inherited) == 1024 and len(complement) == 1536 and
                                not inherited & complement and inherited | complement == set(range(2560)),
        "new_random_pairs": all(set(coalitions[f"new_random_{i}_S1024"]).isdisjoint(
                                coalitions[f"new_random_{i}_C1536"]) and
                                set(coalitions[f"new_random_{i}_S1024"]) |
                                set(coalitions[f"new_random_{i}_C1536"]) == set(range(2560))
                                for i in range(1, 5)),
        "quota": protocol["material"]["eligible_cases_per_cell"] == 6 and
                 protocol["material"]["eligible_case_target"] == 216,
        "branches": set(protocol["branching"]) == {"phase1381", "phase1382", "phase1383",
                                                       "phase1384", "phase1385", "phase1386"},
        "hidden_only": "attention" in protocol["forbidden"] and "MLP" in protocol["forbidden"] and
                       "PCA" in protocol["forbidden"] and "learned probe" in protocol["forbidden"],
        "authorization": final["authorization"] == "run_phase1381_c060_behavior_qualification",
        "scripts_compile": True,
    }
    audit = {"phase": 1380, "campaign": "C060", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks),
             "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
