#!/usr/bin/env python3
"""Independent audit for Phase1484."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1484_c084_batch_deep_mining_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    contract = core.load(OUT / "protocol/preregistration.json")
    py_compile.compile(str(TESTS / "phase1484_c084_batch_deep_mining_contract.py"), doraise=True)
    unsigned = {key: value for key, value in contract.items() if key not in {"contract_sha256", "authorization"}}
    checks = {
        "digest": contract["contract_sha256"] == core.digest(unsigned),
        "axes": len(contract["axes"]["relations"]) == 6 and len(contract["axes"]["states"]) == 37 and contract["axes"]["coordinates"] == 2560,
        "support": contract["coordinate_branch"]["support_counts"] == [13, 26, 51, 128],
        "factorial": len(contract["factorial_branch"]["contrasts"]) == 7,
        "queue": len(contract["route_queue"]) == 3,
        "scope": contract["evidence_typing"]["pattern"].startswith("L3 exploratory"),
        "forbidden": {"attention", "MLP", "PCA", "learned probe"}.issubset(set(contract["forbidden"])),
        "final": final["status"] == "batch_deep_mining_preregistered",
    }
    result = {"phase": 1484, "campaign": "C084", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
