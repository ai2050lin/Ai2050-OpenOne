#!/usr/bin/env python3
"""C178 behavior-interface adjudication before any HiddenState capture."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1712_c178_natural_knowledge_ecology"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    hidden_files = [OUT / "raw/eligible_six_role_all_checkpoint.bf16.npy", OUT / "raw/anchor_role_response.float16.npy"]
    invisible = all("(A) yes (B) no" in row["prompt"] for row in compiled if row["codebook"] == -1)
    checks = {
        "behavior_exact_half": eligibility["global_accuracy"] == 0.5,
        "no_eligible_family": eligibility["eligible_families"] == [],
        "anchors_normal_codebook_correct": all(eligibility["anchor_correct"].values()),
        "codebook_not_rendered": invisible,
        "hidden_not_run": not any(path.exists() for path in hidden_files),
    }
    result = {
        "phase": 1712,
        "campaign": "C178",
        "status": "behavior_interface_invalid_hidden_not_tested",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "reason": "The frozen codebook factor changed the gold label but did not change the visible candidate meanings in the prompt.",
        "scientific_behavior_result": "invalid instrument, not model failure",
        "authorization": "C179_visible_codebook_repair_same_object",
    }
    core.save(OUT / "analysis/behavior_interface_adjudication.json", result)
    final = {"phase": 1712, "campaign": "C178", "status": "closed_behavior_interface_invalid", "checks": checks, "all_checks_passed": all(checks.values()), "scientific_result_valid": False, "hidden_tested": False, "headline": {"global_accuracy": eligibility["global_accuracy"], "reason": result["reason"]}, "next_authorization": result["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
