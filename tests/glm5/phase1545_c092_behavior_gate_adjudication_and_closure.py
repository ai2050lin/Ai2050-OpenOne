#!/usr/bin/env python3
"""Phase1545: independently adjudicate and close C092; authorize interface breadth screening."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1543_c092_truth_output_code_factorial_contract"
PARENT = RESULT / "phase1544_c092_behavior_only_qualification"
OUT = RESULT / "phase1545_c092_behavior_gate_adjudication_and_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def recall(rows: list[dict], truth: bool) -> float:
    subset = [row for row in rows if row["semantic_truth"] is truth]
    return sum(row["semantic_correct"] for row in subset) / len(subset)


def ba(rows: list[dict]) -> float:
    return 0.5 * (recall(rows, True) + recall(rows, False))


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1545 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    rows = core.rows(PARENT / "raw/behavior_logits.jsonl")
    if parent["authorization"] != "run_phase1545_c092_behavior_gate_adjudication" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1544 authorization missing")
    gate = protocol["behavior_gate"]
    adjudication = {}
    for codebook in protocol["codebooks"]:
        subset = [row for row in rows if row["codebook"] == codebook and row["partition"] == "response_discovery"]
        metrics = {
            "semantic_balanced_accuracy": ba(subset),
            "semantic_true_recall": recall(subset, True),
            "semantic_false_recall": recall(subset, False),
            "surface": {surface: ba([row for row in subset if row["surface"] == surface]) for surface in protocol["surfaces"]},
        }
        checks = {
            "balanced_accuracy": metrics["semantic_balanced_accuracy"] >= gate["discovery_each_codebook_semantic_balanced_accuracy"],
            "each_surface": all(value >= gate["discovery_each_codebook_each_surface_semantic_balanced_accuracy"] for value in metrics["surface"].values()),
            "true_recall": metrics["semantic_true_recall"] >= gate["discovery_each_codebook_true_recall"],
            "false_recall": metrics["semantic_false_recall"] >= gate["discovery_each_codebook_false_recall"],
        }
        adjudication[codebook] = {"metrics": metrics, "checks": checks, "qualified": all(checks.values())}
    both = all(value["qualified"] for value in adjudication.values())
    next_campaign = {
        "campaign": "C093",
        "authorization": "run_phase1546_c093_symmetric_code_interface_breadth_contract",
        "objective": "pre-register and behavior-screen several semantically neutral symmetric output alphabets before any hidden-state access",
        "route_policy": "screen all frozen interfaces on discovery; confirm every passing interface without threshold mutation; hidden access only for interfaces that pass both",
        "reason": "C092 failed specifically at reversed yes/no code, so a semantically neutral output alphabet is the nearest construct-valid alternative",
    }
    report = {
        "phase": 1545,
        "campaign": "C092",
        "status": "closed_at_behavior_gate",
        "adjudication": adjudication,
        "both_codebooks_qualified": both,
        "hidden_states_accessed": False,
        "core_puzzle_update": "none",
        "conclusion": "C092 did not test its hidden-state factorial because reversed yes/no failed behavior qualification",
        "not_concluded": ["K267 false", "whole-part truth absent internally", "truth and answer code inseparable in principle"],
        "next_campaign": next_campaign,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if both:
        raise RuntimeError("unexpected gate outcome; C092 closure branch invalid")
    core.save(OUT / "analysis/c092_closure.json", report)
    core.save(OUT / "protocol/next_campaign_authorization.json", next_campaign)
    core.save(OUT / "analysis/final.json", {"phase": 1545, "campaign": "C092", "status": report["status"], "authorization": next_campaign["authorization"]})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
