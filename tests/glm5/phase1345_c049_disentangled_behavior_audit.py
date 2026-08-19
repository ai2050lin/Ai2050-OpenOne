#!/usr/bin/env python3
"""Independent audit for Phase1345 C049 behavior ledgers."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PARENT = TESTS / "result/phase1344_c049_disentangled_relation_contract"
OUT = TESTS / "result/phase1345_c049_disentangled_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def close(left, right, tolerance=1e-10):
    return abs(float(left) - float(right)) <= tolerance


def main():
    protocol = core.load(PARENT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "contract": core.load(OUT / "protocol/execution_manifest.json")["contract_sha256"]
        == protocol["contract_sha256"],
        "authorization": final["authorization"]
        in ("run_phase1346_c049_full_interaction_field", "close_c049_behavior"),
    }
    recomputed_relation, recomputed_joint = [], []
    for model in MODELS:
        rows = core.rows(OUT / f"raw/{model}_behavior.jsonl")
        summary = core.load(OUT / f"analysis/{model}_summary.json")
        groups = defaultdict(list)
        for row in rows:
            groups[row["quartet_key"]].append(row)
        interactions, pairwise, all_correct = [], [], []
        for group in groups.values():
            cells = {row["cell"]: row for row in group}
            interactions.append(
                cells["aa"]["semantic_margin"]
                - cells["ab"]["semantic_margin"]
                - cells["ba"]["semantic_margin"]
                + cells["bb"]["semantic_margin"]
            )
            pairwise.extend(
                [
                    cells["aa"]["semantic_margin"] > cells["ab"]["semantic_margin"],
                    cells["bb"]["semantic_margin"] > cells["ba"]["semantic_margin"],
                ]
            )
            all_correct.append(all(row["correct"] for row in group))
        metrics = summary["relation_interaction_metrics"]
        joint = summary["quartet_joint_reliability"]
        checks[f"{model}_counts"] = len(rows) == 1728 and len(groups) == 432
        checks[f"{model}_accuracy"] = close(metrics["accuracy"], sum(row["correct"] for row in rows) / len(rows))
        checks[f"{model}_interaction"] = close(metrics["median_interaction"], median(interactions)) and close(
            metrics["positive_interaction_fraction"], sum(value > 0 for value in interactions) / len(interactions)
        )
        checks[f"{model}_pairwise"] = close(metrics["pairwise_true_over_false"], sum(pairwise) / len(pairwise))
        checks[f"{model}_joint"] = close(joint["quartet_all_correct"], sum(all_correct) / len(all_correct))
        checks[f"{model}_executor"] = summary["executor"]["qualified"]
        recomputed_relation.append(model) if summary["relation_interaction_qualified"] else None
        recomputed_joint.append(model) if summary["quartet_joint_qualified"] else None
    checks["relation_list"] = recomputed_relation == final["relation_interaction_qualified_models"]
    checks["joint_list"] = recomputed_joint == final["quartet_joint_qualified_models"]
    checks["ledger_separation"] = protocol["behavior_ledgers"]["quartet_joint_reliability_report"][
        "authorization_effect"
    ].startswith("reported independently")
    checks["finite"] = all(math.isfinite(float(value)) for model in MODELS for row in core.rows(OUT / f"raw/{model}_behavior.jsonl") for value in row["scores"])
    result = {
        "phase": 1345,
        "campaign": "C049",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
