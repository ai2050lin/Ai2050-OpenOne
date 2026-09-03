#!/usr/bin/env python3
"""Independent audit for Phase1549."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1549_c094_demonstrated_codebook_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_audit.json")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    cells = Counter((row["partition"], row["surface"], row["truth_sign"], row["codebook_sign"], row["answer_sign"]) for row in cases)
    checks = {
        "pre_model": audit["all_checks_passed"],
        "hashes": protocol["files"] == {"cases_sha256": core.sha(OUT / "material/active_cases.jsonl"), "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl")},
        "coverage": len(cases) == len(compiled) == 240,
        "cells": len(cells) == 24 and set(cells.values()) == {10},
        "single_token": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "case_identity": [row["case_id"] for row in cases] == [row["case_id"] for row in compiled],
        "demonstrations": protocol["demonstrations"]["present_in_both_codebooks"],
        "sequence": protocol["sequence"][0] == "discovery behavior" and "all-state capture only after all behavior partitions pass" in protocol["sequence"],
        "scope": {"attention", "MLP", "PCA", "learned probes"}.issubset(protocol["forbidden"]),
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1550_c094_discovery_behavior_qualification",
    }
    result = {"phase": 1549, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
