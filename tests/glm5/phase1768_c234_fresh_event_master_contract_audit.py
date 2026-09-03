#!/usr/bin/env python3
"""Independent audit for C234."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C234"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    rows = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    checks = {
        "internal": internal["all_checks_passed"],
        "rows": len(rows) == len(compiled) == 640,
        "partitions": {row["partition"] for row in rows} == set(common.PARTITIONS),
        "surface_disjoint": all(row["partition"] == common.SURFACE_PARTITION[row["surface"]] for row in rows),
        "all_coordinates": protocol["physical_coordinates"] == 2560,
        "all_orders": {row["order"] for row in rows} == {1, -1},
        "human_missing_explicit": internal["human_blind_audit_missing"] and "unavailable" in protocol["naturalness_audit"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1768_c234_fresh_event_master_contract.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1768, "campaign": "C234", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
