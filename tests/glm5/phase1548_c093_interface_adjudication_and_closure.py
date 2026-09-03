#!/usr/bin/env python3
"""Phase1548: adjudicate the C093 breadth screen and authorize a demonstrated codebook route."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1547_c093_discovery_behavior_breadth_screen"
OUT = RESULT / "phase1548_c093_interface_adjudication_and_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1548 exists")
    parent = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    summary = core.load(PARENT / "analysis/discovery_behavior_summary.json")
    if parent["authorization"] != "run_phase1548_c093_discovery_interface_adjudication" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1547 authorization missing")
    passing = [name for name, value in summary["interface_results"].items() if value["discovery_pass"]]
    if passing:
        raise RuntimeError(("unexpected passing interfaces", passing))
    reversed_post = {
        name: value["codebooks"]["reversed"]["surface"]["postquery"]
        for name, value in summary["interface_results"].items()
    }
    next_campaign = {
        "campaign": "C094",
        "authorization": "run_phase1549_c094_demonstrated_codebook_contract",
        "objective": "test whether balanced in-context demonstrations establish a reversible A/B output compiler before hidden-state factorial analysis",
        "frozen_interface": "latin A/B selected before C094 data because it had the highest C093 reversed true/false balance among alphabetic interfaces",
        "design": "one obvious whole-part true example and one obvious false example, both present under both codebooks; discovery then confirmation; hidden only after both pass",
    }
    report = {
        "phase": 1548,
        "campaign": "C093",
        "status": "closed_no_discovery_interface_passed",
        "passing_interfaces": passing,
        "reversed_postquery_balanced_accuracy": reversed_post,
        "hidden_states_accessed": False,
        "core_puzzle_update": "none",
        "conclusion": "all four zero-shot direct code interfaces failed the frozen reversed-code behavior gate",
        "not_concluded": ["arbitrary mapping impossible", "K267 false", "truth and output identity inseparable in hidden states"],
        "next_campaign": next_campaign,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/c093_closure.json", report)
    core.save(OUT / "protocol/next_campaign_authorization.json", next_campaign)
    core.save(OUT / "analysis/final.json", {"phase": 1548, "campaign": "C093", "status": report["status"], "authorization": next_campaign["authorization"]})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
