#!/usr/bin/env python3
"""C179 behavior-boundary closure before HiddenState capture."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1713_c179_visible_codebook_natural_ecology"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    lock = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    rows = core.rows(OUT / "raw/behavior_index.jsonl")
    by_truth = {str(v): float(np.mean([r["correct"] for r in rows if r["truth"] == v])) for v in (-1, 1)}
    by_surface = {str(v): float(np.mean([r["correct"] for r in rows if r["surface"] == v])) for v in (-1, 1)}
    by_codebook = {str(v): float(np.mean([r["correct"] for r in rows if r["codebook"] == v])) for v in (-1, 1)}
    checks = {
        "visible_repair_improved": lock["global_accuracy"] > 0.5,
        "global_gate_failed": lock["global_accuracy"] < 0.80,
        "no_family_eligible": lock["eligible_families"] == [],
        "positive_negative_split": by_truth["1"] > 0.95 and by_truth["-1"] < 0.30,
        "anchors_correct": all(lock["anchor_correct"].values()),
        "hidden_not_run": not (OUT / "raw/anchor_role_response.float16.npy").exists(),
    }
    result = {"phase": 1713, "campaign": "C179", "status": "behavior_boundary_closed_hidden_not_tested", "checks": checks, "all_checks_passed": all(checks.values()), "behavior": {"global": lock["global_accuracy"], "truth": by_truth, "surface": by_surface, "codebook": by_codebook}, "interpretation": "Visible candidate repair helped, but broken-path negatives remained unqualified. Positive anchors alone are vulnerable to an always-reachable shortcut.", "authorization": "C180_reachable_target_choice_ecology_new_contract"}
    core.save(OUT / "analysis/behavior_boundary_adjudication.json", result)
    final = {"phase": 1713, "campaign": "C179", "status": "closed_behavior_ineligible", "checks": checks, "all_checks_passed": all(checks.values()), "scientific_result_valid": True, "hidden_tested": False, "headline": {"behavior": result["behavior"], "interpretation": result["interpretation"]}, "next_authorization": result["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
