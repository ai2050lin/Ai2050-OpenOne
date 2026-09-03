#!/usr/bin/env python3
"""Independent audit for the C111 read-only observation contract."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1612_c111_value_identity_role_coalition_contract.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "audit/internal_contract_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"],
        "internal": report["all_checks_passed"],
        "protocol": core.sha(OUT / "protocol/preregistration.json") == report["protocol_sha256"],
        "sources": all(Path(protocol["source_paths"][name]).exists() and core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "read_only": protocol["model_run"] == "forbidden; read-only archive analysis",
        "forbidden": {"PCA", "attention decomposition", "MLP decomposition", "new model execution"}.issubset(protocol["forbidden"]),
        "missingness": len(protocol["planned_missingness"]) == 4,
        "authorization": protocol["authorization"] == "run_phase1613_c111_read_only_observation",
    }
    result = {"phase": 1612, "campaign": "C111", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_contract_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
