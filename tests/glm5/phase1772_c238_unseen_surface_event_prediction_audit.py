#!/usr/bin/env python3
"""Independent audit for C238."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C238"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/prediction_rows.jsonl")
    families = core.rows(OUT / "analysis/family_results.jsonl")
    checks = {
        "final": final["all_checks_passed"],
        "rows": len(rows) == 240,
        "partitions": {row["partition"] for row in rows} == {"confirmation", "lockbox", "fresh"},
        "families": len([row for row in families if row["partition"] == "final"]) == 5,
        "controls": set(protocol["controls"]) == {"best_wrong_family", "surface_only_generic", "relation_role_only", "nearest_length_discovery_group", "zero"},
        "no_refit": protocol["no_refit"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1772_c238_unseen_surface_event_prediction.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1772, "campaign": "C238", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
