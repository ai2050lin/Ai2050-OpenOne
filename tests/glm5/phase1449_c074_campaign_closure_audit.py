#!/usr/bin/env python3
"""Independent audit for Phase1449 C074 closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1449, "C074"
OUT = TESTS / "result/phase1449_c074_campaign_closure"
P1448 = TESTS / "result/phase1448_c074_directional_domain_map"


def main() -> None:
    result = core.load(OUT / "analysis/final.json")
    domain = core.load(P1448 / "analysis/directional_domain_summary.json")
    robust = core.rows(P1448 / "analysis/robust_edges.jsonl")
    checks = {
        "closure": result["all_checks_passed"] and all(result["checks"].values()),
        "status": result["status"] == "closed_with_sparse_directional_transport_domain",
        "classes": domain["class_counts"] == {"robust": 10, "split_specific": 2, "rejected": 20},
        "robust": len(robust) == 10 and sorted(row["edge_id"] for row in robust) == sorted(domain["robust_edge_ids"]),
        "evidence": sum(row["source"].endswith("evidence_first") and row["target"].endswith("evidence_first") for row in robust) == 8,
        "cross_order": not any(not row["same_order"] for row in robust),
        "question": sum(row["source"].endswith("question_first") for row in robust) == 2 and all(row["direction"] == "false_to_true" for row in robust if row["source"].endswith("question_first")),
        "boundary": "semantic neuron group discovered" in result["claim_boundary"]["forbidden"],
        "authorization": result["authorization"] == "preregister_c075_full_hiddenstate_observation_atlas_on_c074_robust_edges",
    }
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
