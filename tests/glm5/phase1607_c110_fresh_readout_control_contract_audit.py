#!/usr/bin/env python3
"""Independent audit for the C110 fresh contract."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1607_c110_fresh_readout_control_contract.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_audit.json")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "source_checks": audit["all_checks_passed"],
        "shape": protocol["archive"]["shape"] == [37, len(manifest), 2560],
        "manifest": all(row["occurrence_index"] == i for i, row in enumerate(manifest)),
        "fresh": audit["checks"]["freshness"],
        "predictions": protocol["frozen_field_prediction"]["cross_fresh_partition_cosine_min"] == 0.9 and len(protocol["frozen_leverage_prediction"]) == 2,
        "energy_match": "exactly the target-support L2" in protocol["energy_match"],
        "multi_role": protocol["multi_role"] == {"state": 19, "roles": ["query_anchor", "focus_record"], "coordinates": "all_2560"},
        "no_hard_stop": "do not stop" in protocol["completion_rule"],
        "authorization": protocol["authorization"] == "execute_phase1608_c110_exact_field_capture",
    }
    result = {"phase": 1607, "campaign": "C110", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_model_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
