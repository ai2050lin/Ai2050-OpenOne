#!/usr/bin/env python3
"""Independent audit for C218."""
from __future__ import annotations

import json
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.RESULT / "phase1752_c218_cross_surface_response_state_atlas"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c218_cross_surface_response_state_atlas.json"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); asset = core.load(ASSET)
    checks = {"final": final["all_checks_passed"], "descriptive": protocol["epistemic_status"].startswith("post-reveal"), "schema": asset["schema"] == "c218_cross_surface_response_state_atlas.v1", "full_coordinates": asset["dimensions"] == list(range(2560)), "rows": len(asset["rows"]) == 180 and all(len(row["values"]) == 2560 for row in asset["rows"]), "three_sources": {row["source"] for row in asset["rows"]} == {"C216_original", "C217_reworded", "cross_surface_difference"}, "producer_hash": core.sha(Path(__file__).with_name("phase1752_c218_cross_surface_response_state_atlas.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1752, "campaign": "C218", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
