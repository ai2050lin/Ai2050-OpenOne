#!/usr/bin/env python3
"""Independent audit for Phase1512."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1512_c088_cross_root_semantic_code_factorial_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    resource = core.load(OUT / "audit/fresh_resource_audit.json")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    groups = core.rows(OUT / "material/composition_sets.jsonl")
    selected = core.rows(OUT / "material/selected_instances.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    py_compile.compile(str(TESTS / "phase1512_c088_cross_root_semantic_code_factorial_contract.py"), doraise=True)
    checks = {
        "contract_hash": protocol["contract_sha256"] == core.digest({key: value for key, value in protocol.items() if key not in ("contract_sha256", "authorization")}),
        "counts": len(cases) == len(compiled) == 1984 and len(groups) == 248 and len(selected) == 248,
        "factor": Counter((row["semantic_sign"], row["code_sign"]) for row in cases) == {(1, 1): 496, (-1, 1): 496, (1, -1): 496, (-1, -1): 496},
        "groups": all(sum(key.startswith(("a_code_", "b_code_")) for key in row) == 8 for row in groups),
        "fresh": resource["strict_fresh_available_items"] == 8 and not resource["strict_fresh_target_met"],
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["compiled_sha256"] == core.sha(OUT / "compiled/qwen3_active.jsonl"),
        "zero_models": all(value == 0.5 for value in audit["zero_models"].values()),
        "single_forward": protocol["authoritative_forward"]["single_pass_behavior_and_hidden_capture"],
        "scope": "universal comparator" in protocol["claim_boundary"]["forbidden"],
        "preaudit": all(audit["checks"].values()),
    }
    result = {"phase": 1512, "campaign": "C088", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
