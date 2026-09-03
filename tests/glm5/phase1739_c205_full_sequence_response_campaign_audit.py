#!/usr/bin/env python3
"""Independent audit for C205."""
from __future__ import annotations

import json
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C205


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    anchors = core.rows(OUT / "compiled/qwen3_anchors.jsonl")
    checks = {
        "final": final["all_checks_passed"],
        "phase": final["phase"] == 1739 and final["campaign"] == "C205",
        "anchors": len(anchors) == 36,
        "breadth": len({row["program"] for row in anchors}) == 9,
        "routes": len(protocol["routes"]) == 10,
        "full_token_object": "full-token" in protocol["object"],
        "route_level_elimination": "failure eliminates only that route" in protocol["route_policy"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1739_c205_full_sequence_response_campaign.py")) == protocol["producer_sha256"],
    }
    report = {"phase": 1739, "campaign": "C205", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
