#!/usr/bin/env python3
"""Independent audit for C233 and the C223-C233 chain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C233"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c233_surface_transport_composition_atlas.json"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    asset = core.load(ASSET)
    matrix = np.asarray([row["values"] for row in asset["rows"]], np.float32)
    chain = {campaign: core.load(common.OUTS[campaign] / "audit/independent_final_audit.json")["all_checks_passed"] for campaign in ("C223", "C224", "C225", "C226", "C227", "C228", "C229", "C230", "C231", "C232")}
    checks = {"final": final["all_checks_passed"], "prior_chain": all(chain.values()), "rows": matrix.shape == (840, 2560), "embedding_hidden": set(row["checkpoint"] for row in asset["rows"]) == set(common.CHECKPOINTS), "all_coordinates": asset["dimensions"] == list(range(2560)), "strict_failures_preserved": not asset["summary"]["transport_lockbox_passed"] and not asset["summary"]["composition_campaign_passed"] and not asset["summary"]["cross_model_passed"] and not asset["summary"]["new_mathematics_authorized"], "producer_hash": core.sha(Path(__file__).with_name("phase1767_c233_campaign_synthesis_heatmap.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1767, "campaign": "C233", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "prior_chain": chain, "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
