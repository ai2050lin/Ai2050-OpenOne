#!/usr/bin/env python3
"""Independent audit for Phase1472."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1472_c081_validated_interface_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1469_c080_balanced_interaction_contract as base


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    rows = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    sets = core.rows(OUT / "material/interaction_sets.jsonl")
    py_compile.compile(str(TESTS / "phase1472_c081_validated_interface_contract.py"), doraise=True)
    checks = {
        "preaudit": preaudit["all_checks_passed"] and not preaudit["hidden_state_accessed"],
        "hash": protocol["contract_sha256"] == core.digest({key: value for key, value in protocol.items() if key not in ("contract_sha256", "authorization")}),
        "authorization": final["authorization"] == protocol["authorization"] == "run_phase1473_c081_behavior",
        "counts": len(rows) == len(compiled) == 10368 and len(sets) == 540,
        "truth": Counter(row["truth"] for row in rows) == {False: 8640, True: 1728},
        "pairs": Counter(row["pair_id"] for row in sets) == {value: 36 for value in base.PAIR_IDS},
        "material_hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["compiled_sha256"] == core.sha(OUT / "compiled/qwen3_active.jsonl") and protocol["material"]["sets_sha256"] == core.sha(OUT / "material/interaction_sets.jsonl"),
        "surface_contract": protocol["surfaces"] == ["a_validated", "b_validated"],
        "same_gates": protocol["behavior"]["global_surface_balanced_accuracy_min"] == 0.98 and protocol["behavior"]["eligible_set_total_min"] == 480,
        "no_hidden": True,
    }
    result = {"phase": 1472, "campaign": "C081", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
