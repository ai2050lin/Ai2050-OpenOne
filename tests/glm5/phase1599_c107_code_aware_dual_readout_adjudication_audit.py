#!/usr/bin/env python3
"""Independent audit for Phase1599 / C107."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1599_c107_code_aware_dual_readout_adjudication"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1599_c107_code_aware_dual_readout_adjudication.py"
    py_compile.compile(str(producer), doraise=True)
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/c106_code_aware_results.jsonl")
    summary = core.rows(OUT / "analysis/c106_code_aware_summary.jsonl")
    rollup = core.rows(OUT / "analysis/c106_code_aware_family_rollup.jsonl")
    asset = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    payload = core.load(asset)
    standard = [row for row in summary if row["code"] == 1]
    reversed_rows = [row for row in summary if row["code"] == -1]
    checks = {
        "producer_compiles": py_compile.compile(str(producer), doraise=True) is not None,
        "source_checks": all(final["checks"].values()),
        "row_counts": len(rows) == 96 and len(summary) == 80 and len(rollup) == 2,
        "candidate_order": final["candidate_order"]["normalized"] == ["yes", "no"],
        "code_sign": all(
            abs(entry["code_aligned_task_gain"] - row["code"] * entry["truth_direction_gain"]) < 1e-12
            for row in rows for nested in row["nested"].values() for entry in nested.values()
        ),
        "strata": len(standard) == len(reversed_rows) == 40,
        "raw_scale": final["c106_first_tested_k"]["truth_direction"] == {"agent_patient": 128, "attribute_binding": 256},
        "no_task_scale": final["c106_first_tested_k"]["code_aligned_task"] == {"agent_patient": None, "attribute_binding": None},
        "claim_repair": "minimality" in final["claim_adjudication"]["K279-R1"],
        "heatmap_identity": core.sha(asset) == core.sha(PUBLIC) == final["heatmap"]["sha256"],
        "heatmap_scope": payload["headline"]["legacy_minimal_k_retracted"] and all(value is None for value in payload["headline"]["task_aligned_all_four_k"].values()),
        "no_model_claim": final["claim_boundary"].startswith("deterministic reanalysis"),
    }
    result = {
        "phase": 1599,
        "campaign": "C107",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
