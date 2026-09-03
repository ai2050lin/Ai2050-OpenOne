#!/usr/bin/env python3
"""Independent audit for Phase1542."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1542_c091_final_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/c091_final_adjudication.json")
    puzzle = core.load(OUT / "theory/k267.json")
    next_campaign = core.load(OUT / "protocol/next_campaign_authorization.json")
    checks = {
        "seven_audits": set(report["audits"]) == {str(value) for value in range(1535, 1542)} and all(value["passed"] == value["total"] for value in report["audits"].values()),
        "semantic_scope": report["behavior_scope"]["qualified"] == ["whole_part"] and set(report["behavior_scope"]["retired"]) == {"similarity", "class_inclusion"},
        "holdout_pass": report["holdout_gate"]["passed"] and report["holdout_gate"]["min_centroid_cosine_to_discovery"] >= 0.5,
        "puzzle_identical": puzzle == report["core_puzzle"] and puzzle["id"] == "K267",
        "claim_boundary": "答案代码与任务终止已被排除" in puzzle["not_claimed"] and "新数学理论" in puzzle["not_claimed"],
        "theory_name_stable": report["theory_update"]["theory_name_unchanged"] == "条件化输出场闭合理论",
        "next_contract_identical": next_campaign == report["next_campaign"],
        "next_scope": next_campaign["frozen_constraints"]["analysis_scope"] == ["embeddings", "all_hidden_states", "candidate_logits"],
        "forbidden_components": {"attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes"}.issubset(next_campaign["frozen_constraints"]["forbidden"]),
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1543_c092_truth_output_code_factorial_contract",
    }
    result = {"phase": 1542, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
