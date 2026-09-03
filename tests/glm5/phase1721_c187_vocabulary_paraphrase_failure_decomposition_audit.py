#!/usr/bin/env python3
"""Independent audit for C187 factorial failure decomposition."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1721_c187_vocabulary_paraphrase_failure_decomposition"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    atlas = core.load(OUT / "analysis/factorial_atlas.json")
    final = core.load(OUT / "analysis/final.json")
    producer = Path(__file__).with_name("phase1721_c187_vocabulary_paraphrase_failure_decomposition.py")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "28_cells": len(atlas["rows"]) == 28, "27_observed": sum(row["observed"] for row in atlas["rows"]) == 27, "one_missing": len(atlas["registered_missing"]) == 1, "no_imputation": protocol["missing_policy"].startswith("no imputation"), "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1721, "campaign": "C187", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
