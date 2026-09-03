#!/usr/bin/env python3
"""Independent audit for C240."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C240"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/prediction_rows.jsonl")
    checks = {"final": final["all_checks_passed"], "rows": len(rows) == 50, "partitions": {row["partition"] for row in rows} == {"lockbox", "fresh"}, "families": {row["family"] for row in rows} == set(common.FAMILIES), "caveat": "not an untouched" in protocol["caveat"], "producer_hash": core.sha(Path(__file__).with_name("phase1774_c240_factor_composition_event_prediction.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1774, "campaign": "C240", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
