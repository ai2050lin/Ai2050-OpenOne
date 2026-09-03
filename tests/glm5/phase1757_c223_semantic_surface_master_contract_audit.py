#!/usr/bin/env python3
"""Independent audit for C223."""
from __future__ import annotations

import json
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C223"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    checks = {
        "internal": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "rows": len(rows) == len(compiled) == 2304,
        "families": set(row["family"] for row in rows) == set(common.FAMILIES),
        "surfaces": set(row["surface"] for row in rows) == set(common.SURFACES),
        "partitions": set(row["partition"] for row in rows) == {"discovery", "confirmation", "lockbox"},
        "lockbox_rule": "lockbox" in protocol["surface_transport_fit"],
        "route_policy": "does not stop" in protocol["route_policy"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1757_c223_semantic_surface_master_contract.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1757, "campaign": "C223", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
