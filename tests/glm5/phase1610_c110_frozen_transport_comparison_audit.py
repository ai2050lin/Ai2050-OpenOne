#!/usr/bin/env python3
"""Independent audit for the C110 frozen transport comparison."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1610_c110_frozen_transport_comparison.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "analysis/transport_adjudication.json")
    adapter = core.load(OUT / "protocol/transport_adapter.json")
    results = core.rows(OUT / "analysis/fresh_transport_results.jsonl")
    summaries = core.rows(OUT / "analysis/fresh_transport_summary.jsonl")
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"] == adapter["producer_sha256"], "source": all(report["checks"].values()),
        "adapter": adapter["contract_sha256"] == core.sha(OUT / "protocol/preregistration.json") and adapter["field_report_sha256"] == core.sha(OUT / "analysis/field_prediction_adjudication.json"),
        "hashes": core.sha(OUT / "analysis/fresh_transport_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/fresh_transport_summary.jsonl") == report["summary_sha256"],
        "rows": len(results) == 192 and len(summaries) == 8, "modes": all(set(row["modes"]) == set(adapter["modes"]) for row in results),
        "energy": max(row["energy_match_relative_error"] for row in results) <= adapter["energy_match_relative_tolerance_bf16"],
        "prediction": set(report["prediction"]) == {"attribute_target_efficiency_gt_wrong_cells", "agent_target_efficiency_lt_wrong_cells", "attribute_prediction_passed", "agent_prediction_passed"},
        "boundary": "separate" in report["claim_boundary"],
        "authorization": report["authorization"] == "run_phase1611_c110_synthesis_heatmap_and_major_stage_closure",
    }
    result = {"phase": 1610, "campaign": "C110", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_transport_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
