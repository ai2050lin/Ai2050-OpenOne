#!/usr/bin/env python3
"""Independent audit for C188 scaffold prediction."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1722_c188_new_material_generic_scaffold_prediction"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); atlas = core.load(OUT / "analysis/scaffold_prediction_atlas.json"); final = core.load(OUT / "analysis/final.json"); producer = Path(__file__).with_name("phase1722_c188_new_material_generic_scaffold_prediction.py")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "four_cells": len(atlas["rows"]) == 4, "missing_denominator": sorted(row["observed_family_count"] for row in atlas["rows"]) == [6, 7, 7, 7], "same_thresholds": protocol["labels"] == core.load(TESTS / "result/phase1718_c184_response_ecology_invariant_discovery/protocol/preregistration.json")["observation_labels"], "no_cosine": "cosine" not in json.dumps(protocol["metrics"]).lower(), "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1722, "campaign": "C188", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
