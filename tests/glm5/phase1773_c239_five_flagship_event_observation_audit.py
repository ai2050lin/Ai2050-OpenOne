#!/usr/bin/env python3
"""Independent audit for C239."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C239"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    routes = core.load(OUT / "analysis/route_profiles.json")
    checks = {"final": final["all_checks_passed"], "routes": len(routes) == 5, "families": {row["family"] for row in routes} == set(common.FAMILIES), "three_effects": all(len(row["effects"]) == 3 for row in routes), "no_new_run": protocol["no_new_model_run"], "producer_hash": core.sha(Path(__file__).with_name("phase1773_c239_five_flagship_event_observation.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1773, "campaign": "C239", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
