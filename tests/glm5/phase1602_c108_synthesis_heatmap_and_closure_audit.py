#!/usr/bin/env python3
"""Independent closure and heatmap audit for Phase1602 / C108."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
OUT = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1602_c108_synthesis_heatmap_and_closure.py"
    py_compile.compile(str(producer), doraise=True)
    audit = core.load(OUT / "audit/independent_intervention_audit.json")
    closure = core.load(OUT / "analysis/closure.json")
    asset = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    payload = core.load(asset)
    family = {row["family"]: row for row in payload["fresh_c108"]["family_rollup"]}
    checks = {
        "producer_compiles": py_compile.compile(str(producer), doraise=True) is not None,
        "source_audit": audit["all_checks_passed"],
        "attribute": (family["attribute_binding"]["truth_direction_write_cells"], family["attribute_binding"]["truth_direction_delete_cells"]) == (4, 4),
        "agent": (family["agent_patient"]["truth_direction_write_cells"], family["agent_patient"]["truth_direction_delete_cells"]) == (0, 3),
        "task_boundary": (family["attribute_binding"]["code_aligned_task_write_cells"], family["agent_patient"]["code_aligned_task_write_cells"]) == (2, 0),
        "puzzles": set(closure["claim_adjudication"]) == {"K281", "K282"},
        "identity": core.sha(asset) == core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "scope": "functionally sufficient" in closure["claim_boundary"] and "Neither set is minimal" in closure["claim_boundary"],
        "authorization": closure["next_authorization"].startswith("C109 observation-first"),
    }
    result = {"phase": 1602, "campaign": "C108", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_closure_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
