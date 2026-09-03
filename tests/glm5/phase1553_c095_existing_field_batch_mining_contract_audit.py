#!/usr/bin/env python3
"""Independent audit for Phase1553."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1553_c095_existing_field_batch_mining_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1553_c095_existing_field_batch_mining_contract.py"), doraise=True)
    final = core.load(OUT / "analysis/final.json")
    contract = core.load(OUT / "protocol/preregistration.json")
    unsigned = {key: value for key, value in contract.items() if key not in {"contract_sha256", "authorization"}}
    forbidden = set(contract["forbidden"])
    checks = {
        "digest": contract["contract_sha256"] == core.digest(unsigned),
        "axes": len(contract["axes"]["partitions"]) == 3 and len(contract["axes"]["family_pairs"]) == 3 and len(contract["axes"]["states"]) == 37 and len(contract["axes"]["roles"]) == 4,
        "full_coordinates": contract["axes"]["coordinates"] == 2560 and contract["raw_coordinate_branch"]["support_counts"] == [16, 64, 256, 1024],
        "triadic": contract["triadic_interaction_branch"]["definition"].startswith("C_fg = 0.5"),
        "decomposition": len(contract["three_by_three_decomposition"]["definitions"]) == 4,
        "behavior_missingness": "M_CELL" in contract["predefined_missingness"],
        "no_hard_pattern_stop": contract["stop_rule"].startswith("Only source-integrity"),
        "queue": len(contract["route_queue"]) == 3,
        "forbidden": {"new model run", "attention", "MLP", "PCA", "learned probe", "causal claim", "new mathematics"}.issubset(forbidden),
        "retrospective_scope": contract["claim_boundary"]["allowed"].startswith("retrospective descriptive"),
        "authorization": final["authorization"] == "run_phase1554_c095_triadic_interaction_and_field_atlas",
    }
    result = {"phase": 1553, "campaign": "C095", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
