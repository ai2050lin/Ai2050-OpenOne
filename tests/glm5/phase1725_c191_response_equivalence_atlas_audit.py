#!/usr/bin/env python3
"""Independent audit for C191 response-equivalence atlas."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1725_c191_response_equivalence_atlas"; PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c191_response_equivalence_atlas.json"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/response_equivalence_atlas.json"); payload = core.load(PUBLIC); producer = Path(__file__).with_name("phase1725_c191_response_equivalence_atlas.py")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "cells": report["observed_cells"] == len(payload["rows"]) == 52,
        "missing": len(report["registered_missing"]) == 4,
        "all_2560": len(payload["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in payload["rows"]),
        "matrix": np.asarray(payload["similarity_matrix"]).shape == (52, 52),
        "labels": set(report["nearest_neighbor_summary"]) == {"family", "unit", "phrase_variant", "wrapper_variant"},
        "asset_hash": core.sha(PUBLIC) == final["asset"]["sha256"],
        "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {"phase": 1725, "campaign": "C191", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
