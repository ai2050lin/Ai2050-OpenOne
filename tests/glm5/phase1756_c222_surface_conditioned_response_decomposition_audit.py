#!/usr/bin/env python3
"""Independent audit for C222."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.RESULT / "phase1756_c222_surface_conditioned_response_decomposition"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c222_surface_conditioned_response_atlas.json"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    summary = final["headline"]["summary"]
    asset = core.load(ASSET)
    matrix = np.asarray([row["values"] for row in asset["rows"]], np.float32)
    checks = {
        "final": final["all_checks_passed"],
        "post_reveal": protocol["epistemic_status"].startswith("post-reveal"),
        "within_surface": summary["within_C221_surface"]["support"] == 20,
        "four_banks": len(summary["per_prior_bank"]) == 4,
        "oracle_labeled": any("oracle" in name for name in summary["exact_field"]),
        "physical_coordinates": matrix.shape == (240, 2560),
        "finite": bool(np.isfinite(matrix).all()),
        "producer_hash": core.sha(Path(__file__).with_name("phase1756_c222_surface_conditioned_response_decomposition.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1756, "campaign": "C222", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
