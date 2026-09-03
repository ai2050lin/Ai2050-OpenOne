#!/usr/bin/env python3
"""Independent audit for the C112 CUDA batch interventions."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1615_c112_value_identity_role_lattice"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1616_c112_cuda_batch_interventions.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "analysis/adjudication.json")
    rows = core.rows(OUT / "analysis/batch_intervention_results.jsonl")
    summary = core.rows(OUT / "analysis/batch_intervention_summary.jsonl")
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"],
        "checks": all(report["checks"].values()),
        "hashes": core.sha(OUT / "analysis/batch_intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/batch_intervention_summary.jsonl") == report["summary_sha256"],
        "rows": len(rows) == 192 and len({row["pair_id"] for row in rows}) == 192,
        "modes": all(len(row["modes"]) == 20 for row in rows),
        "summary": len(summary) == 8 and all(row["pairs"] == 24 for row in summary),
        "l2": report["max_permutation_l2_relative_error"] <= 0.02,
        "predictions": set(report["predictions"]) == {"attribute_frozen_gt_permutation_median_cells", "attribute_frozen_gt_all_permutation_cells", "agent_focus_record_positive_cells", "agent_record_path_gt_query_cells"},
        "boundary": "no attention" in report["claim_boundary"],
        "authorization": report["authorization"] == "run_phase1617_c112_synthesis_heatmap_and_closure",
    }
    result = {"phase": 1616, "campaign": "C112", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_batch_intervention_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
