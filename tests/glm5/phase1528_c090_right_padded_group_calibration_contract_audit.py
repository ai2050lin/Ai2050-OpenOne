#!/usr/bin/env python3
"""Independent audit for Phase1528."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1528_c090_right_padded_group_calibration_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_engine_audit.json")
    batches = core.rows(OUT / "protocol/right_padded_calibration_batches.jsonl")
    py_compile.compile(str(TESTS / "phase1528_c090_right_padded_group_calibration_contract.py"), doraise=True)
    checks = {
        "status": final["status"] == "right_padded_group_calibration_contract_frozen",
        "batches": len(batches) == 18 and all(row["cells"] == ["aa", "ab", "ba", "bb"] for row in batches),
        "hash": core.sha(OUT / "protocol/right_padded_calibration_batches.jsonl") == protocol["batches_sha256"],
        "contract": core.digest({key: value for key, value in protocol.items() if key != "contract_sha256"}) == protocol["contract_sha256"] == final["contract_sha256"],
        "gates": protocol["gates"] == {"repeat_hidden_max_abs": 1e-6, "repeat_logit_max_abs": 1e-6, "causal_prefix_relative_l2": 1e-6},
        "no_model": audit["checks"]["no_model"],
        "scope": not protocol["hidden_semantic_claim"],
        "authorization": final["authorization"] == "run_phase1529_c090_right_padded_group_calibration",
    }
    result = {"phase": 1528, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
