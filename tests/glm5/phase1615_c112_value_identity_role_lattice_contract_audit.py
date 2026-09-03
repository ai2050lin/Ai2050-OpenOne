#!/usr/bin/env python3
"""Independent audit for the C112 value-identity and role-lattice contract."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1615_c112_value_identity_role_lattice"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1615_c112_value_identity_role_lattice_contract.py"
    py_compile.compile(str(producer), doraise=True)
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {
        "producer": core.sha(producer) == internal["producer_sha256"],
        "internal": internal["all_checks_passed"],
        "protocol": core.sha(OUT / "protocol/preregistration.json") == internal["protocol_sha256"],
        "sources": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "permutations": len(protocol["movement_permutations"]["attribute_binding"]) == 8 and len(protocol["movement_permutations"]["agent_patient"]) == 8,
        "roles": len(protocol["single_roles"]) == 7 and len(protocol["role_coalitions"]) == 4,
        "modes": len(protocol["modes"]) == 20,
        "authorization": protocol["authorization"] == "run_phase1616_c112_cuda_batch_interventions",
    }
    result = {"phase": 1615, "campaign": "C112", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_contract_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
