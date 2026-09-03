#!/usr/bin/env python3
"""Independent audit for C215."""
from __future__ import annotations

import json
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C215
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c215_response_interval_composition_atlas.json"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    asset = core.load(ASSET)
    kinds = {row["kind"] for row in asset["rows"]}
    checks = {
        "final": final["all_checks_passed"],
        "schema": asset["schema"] == "c215_response_interval_composition_atlas.v1",
        "physical_coordinates": asset["dimensions"] == list(range(2560)),
        "row_width": len(asset["rows"]) == 180 and all(len(row["values"]) == 2560 for row in asset["rows"]),
        "panels": {"fresh_baseline", "dose1_intervention_response", "path_combined", "path_additive_prediction", "path_interaction"} <= kinds,
        "upstream": all(core.load(({
            205: common.C205, 206: common.C206, 207: common.C207, 208: common.C208, 209: common.C209,
            210: common.C210, 211: common.C211, 212: common.C212, 213: common.C213, 214: common.C214,
        }[campaign]) / "audit/independent_final_audit.json")["all_checks_passed"] for campaign in range(205, 215)),
        "producer_hash": core.sha(Path(__file__).with_name("phase1749_c215_campaign_synthesis_heatmap.py")) == protocol["producer_sha256"],
        "claim_boundary": "not weights" in asset["claim_boundary"],
    }
    audit = {"phase": 1749, "campaign": "C215", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
