#!/usr/bin/env python3
"""Independent audit for C242."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C242"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = final["headline"]
    checks = {"final": final["all_checks_passed"], "models": set(report["models"]) == {"qwen3", "glm4", "deepseek7b"}, "participants": set(report["participants"]) <= {"qwen3", "glm4", "deepseek7b"}, "pairs": len(report["pair_tests"]) == len(report["participants"]) * (len(report["participants"]) - 1) // 2, "sequential": protocol["sequential_loading"], "no_coordinate_alignment": "No physical coordinate" in protocol["claim_boundary"], "producer_hash": core.sha(Path(__file__).with_name("phase1776_c242_cross_model_abstract_event_graph.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1776, "campaign": "C242", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
