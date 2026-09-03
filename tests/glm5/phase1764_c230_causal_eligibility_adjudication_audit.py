#!/usr/bin/env python3
"""Independent audit for C230."""
from __future__ import annotations

import json
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C230"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    headline = final["headline"]
    checks = {"final": final["all_checks_passed"], "typed_not_tested": headline["status"] == "typed_not_tested", "no_tests": headline["tests_executed"] == [], "failed_inputs_preserved": not headline["ledger"]["transport_confirmation"] and not headline["ledger"]["transport_lockbox"] and headline["ledger"]["composition_families_passed"] == 2, "producer_hash": core.sha(Path(__file__).with_name("phase1764_c230_causal_eligibility_adjudication.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1764, "campaign": "C230", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
