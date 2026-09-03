#!/usr/bin/env python3
"""Independent audit for C182 typed-not-tested adjudication."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1716_c182_cross_model_hidden_topology_adjudication"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    producer = Path(__file__).with_name("phase1716_c182_cross_model_hidden_topology_adjudication.py")
    checks = {
        "closed_typed": final["status"] == "closed_typed_not_tested",
        "zero_tests": final["tests_run"] == 0,
        "no_hidden": final["hidden_states_loaded"] is False,
        "gate_failed": final["checks"]["fewer_than_four_common_families"],
        "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {
        "phase": 1716,
        "campaign": "C182",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "authorization": final["next_authorization"],
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
