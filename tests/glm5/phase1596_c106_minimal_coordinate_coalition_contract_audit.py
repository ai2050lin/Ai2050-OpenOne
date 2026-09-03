#!/usr/bin/env python3
"""Independent pre-model audit for C106."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1596_c106_minimal_coordinate_coalition_contract.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    source = core.load(OUT / "audit/pre_model_audit.json")
    manifest = core.rows(OUT / "protocol/pair_manifest.jsonl")
    discovery = core.rows(OUT / "analysis/discovery_nested_support_observation.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "source": source["all_checks_passed"] and source["passed"] == source["total"] == 9,
        "ranking": all(sorted(protocol["rankings"][family]) == list(range(2560)) for family in protocol["families"]),
        "nested": protocol["nested_k"] == [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560],
        "manifest": len(manifest) == protocol["pairs"] == 96 and core.sha(OUT / "protocol/pair_manifest.jsonl") == protocol["pair_manifest_sha256"],
        "discovery": len(discovery) == 20 and core.sha(OUT / "analysis/discovery_nested_support_observation.jsonl") == protocol["discovery_observation_sha256"],
        "candidate_order": protocol["candidate_order"] == ["yes", "no"] and "candidate[0]-candidate[1]" in protocol["readout"],
        "authorization": protocol["authorization"] == "execute_phase1597_c106_nested_coordinate_interventions",
    }
    result = {"phase": 1596, "campaign": "C106", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_model_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
