#!/usr/bin/env python3
"""Independent pre-model audit for Phase1600 / C108."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1600_c108_fresh_coordinate_causality_contract.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_audit.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    checks = {
        "producer_compiles": py_compile.compile(str(producer), doraise=True) is not None,
        "source_checks": all(audit["checks"].values()),
        "counts": len(units) == 24 and len(cases) == len(compiled) == 384,
        "partitions": Counter((row["family"], row["partition"]) for row in units) == {(family, partition): 6 for family in protocol["families"] for partition in protocol["partitions"]},
        "truth_code": all(row["output_yes"] == ((row["truth_factor"] == 1) == (row["code"] == 1)) for row in cases),
        "candidate_order": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "frozen_k": protocol["frozen_k"] == {"attribute_binding": 256, "agent_patient": 128},
        "rankings": all(sorted(protocol["rankings"][family]) == list(range(2560)) for family in protocol["families"]),
        "no_reselection": protocol["no_reselection"].startswith("no family"),
        "authorization": protocol["authorization"] == "execute_phase1601_c108_fresh_coordinate_interventions",
    }
    result = {"phase": 1600, "campaign": "C108", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_model_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
