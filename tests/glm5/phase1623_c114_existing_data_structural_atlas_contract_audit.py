#!/usr/bin/env python3
"""Independent audit for Phase1623 / C114 contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1623_c114_existing_data_structural_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    checks = {
        "internal": internal["all_checks_passed"],
        "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1623_c114_existing_data_structural_atlas_contract.py"),
        "sources": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "scope": protocol["datasets"] == ["C112", "C113"] and protocol["cells"] == 16,
        "policy": "no PCA" in protocol["analysis_policy"] and "C113-only" in protocol["missingness"],
        "boundary": "no independent replication" in protocol["claim_boundary"] and "attention/MLP" in protocol["claim_boundary"],
        "authorization": protocol["authorization"] == "execute_phase1624_c114_structural_atlas_and_freeze_c115_predictions",
    }
    report = {"phase": 1623, "campaign": "C114", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": protocol["authorization"]}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
