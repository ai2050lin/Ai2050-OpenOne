#!/usr/bin/env python3
"""Independent audit for C243."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C243"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c243_conditional_event_atlas.json"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    manifest = final["headline"]["asset_manifest"]
    with ASSET.open("r", encoding="utf-8") as handle:
        asset = json.load(handle)
    rows = asset["rows"]
    sample_ids = (0, 1, len(rows) // 2, len(rows) - 1)
    checks = {
        "internal": final["all_checks_passed"],
        "parent_chain": all(
            core.load(common.OUTS[campaign] / "audit/independent_final_audit.json")["all_checks_passed"]
            for campaign in tuple(common.OUTS)[:-1]
        ),
        "schema": asset["schema"] == "c243_conditional_event_atlas.v1",
        "all_rows": len(rows) == manifest["rows"] == 3330,
        "all_dimensions": asset["dimensions"] == list(range(2560)),
        "sample_width": all(len(rows[index]["values"]) == 2560 for index in sample_ids),
        "embedding_and_hidden": {row["checkpoint_type"] for row in rows} == {"embedding", "hidden_state"},
        "families_effects_roles": {row["family"] for row in rows} == set(common.FAMILIES) and {row["effect"] for row in rows} == set(common.EFFECTS) and {row["role"] for row in rows} == set(common.ROLES),
        "asset_hash": core.sha(ASSET) == manifest["sha256"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1777_c243_campaign_theory_heatmap_closure.py")) == protocol["producer_sha256"],
    }
    audit = {
        "phase": 1777,
        "campaign": "C243",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": final["next_authorization"],
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
