#!/usr/bin/env python3
"""Independent audit for C220."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.RESULT / "phase1754_c220_response_state_minimality_collision_controls"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c220_response_state_minimality_atlas.json"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = final["headline"]
    asset = core.load(ASSET)
    matrix = np.asarray([row["values"] for row in asset["rows"]], np.float32)
    checks = {
        "final": final["all_checks_passed"],
        "frozen_before_reveal": "before_C219_reveal" in protocol["status"],
        "selection_then_fresh": set(protocol["sources"]) == {"templates", "selection", "single_reveal"},
        "five_subsets": len(report["subset_ladder"]) == 5,
        "three_negative_controls": len(report["negative_controls"]) == 3,
        "fresh_support": report["single_reveal_fresh"]["support"] == 20,
        "physical_coordinates": matrix.ndim == 2 and matrix.shape[1] == 2560,
        "finite": bool(np.isfinite(matrix).all()),
        "producer_hash": core.sha(Path(__file__).with_name("phase1754_c220_response_state_minimality_collision_controls.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1754, "campaign": "C220", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
