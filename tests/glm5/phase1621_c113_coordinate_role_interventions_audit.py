#!/usr/bin/env python3
"""Independent audit for Phase1621 / C113 interventions."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/intervention_adjudication.json")
    rows = core.rows(OUT / "analysis/intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/intervention_summary.jsonl")
    checks = {
        "producer": report["producer_sha256"] == core.sha(TESTS / "phase1621_c113_coordinate_role_interventions.py"),
        "rows": len(rows) == 192 and len({row["pair_id"] for row in rows}) == 192,
        "modes": all(set(row["modes"]) == set(protocol["modes"]) for row in rows),
        "summary": len(summaries) == 8 and all(row["pairs"] == 24 and row["independent_units"] == 6 for row in summaries),
        "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in rows for mode in protocol["modes"]),
        "l2": report["max_permutation_l2_relative_error"] <= protocol["numeric"]["movement_permutation_actual_l2_relative_tolerance"],
        "hashes": core.sha(OUT / "analysis/intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/intervention_summary.jsonl") == report["summary_sha256"],
        "predictions": set(report["predictions"]) == {"attribute_frozen_gt_all_permutation_cells", "agent_record_path_gt_query_cells", "agent_all_roles_gt_path_cells", "agent_leave_query_anchor_lowers_cells", "agent_leave_query_focus_lowers_cells"},
        "runtime": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"],
        "authorization": report["authorization"] == "run_phase1622_c113_synthesis_heatmap_and_closure",
    }
    audit = {"phase": 1621, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": report["authorization"]}
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(OUT / "audit/independent_intervention_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
