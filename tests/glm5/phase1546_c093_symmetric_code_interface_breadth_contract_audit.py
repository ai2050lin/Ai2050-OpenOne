#!/usr/bin/env python3
"""Independent audit for Phase1546."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1546_c093_symmetric_code_interface_breadth_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_audit.json")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    cells = Counter((row["partition"], row["surface"], row["interface"], row["truth_sign"], row["codebook_sign"]) for row in cases)
    checks = {
        "pre_model": audit["all_checks_passed"],
        "hashes": protocol["files"] == {"cases_sha256": core.sha(OUT / "material/active_cases.jsonl"), "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl")},
        "coverage": len(cases) == len(compiled) == 960,
        "cells": len(cells) == 96 and set(cells.values()) == {10},
        "case_identity": [row["case_id"] for row in cases] == [row["case_id"] for row in compiled],
        "tokens": len({ids[0] for mapping in audit["token_ids"].values() for ids in mapping.values()}) == 8,
        "zero_models": all(value == 0.5 for mapping in audit["zero_models"].values() for value in mapping.values()),
        "route_policy": "retire failures" in protocol["route_policy"] and "passing both partitions" in protocol["route_policy"],
        "hidden_scope": "no attention" in protocol["hidden_scope_if_qualified"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1547_c093_discovery_behavior_breadth_screen",
    }
    result = {"phase": 1546, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
