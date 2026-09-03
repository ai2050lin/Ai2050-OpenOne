#!/usr/bin/env python3
"""Phase1617 / C112: synthesize coordinate assignment and role-lattice results."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1615_c112_value_identity_role_lattice"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def family_rollup(rows: list[dict], summaries: list[dict], family: str) -> dict:
    selected = [row for row in rows if row["family"] == family]
    cells = [row for row in summaries if row["family"] == family]
    query_mode = "single_query_anchor"
    path_mode = "coalition_record_to_query_path"
    all_mode = "coalition_all_registered_roles"
    return {
        "pairs": len(selected),
        "frozen_support_gt_permutation_median_cells": int(sum(row["frozen_support_gt_permutation_median"] for row in cells)),
        "frozen_support_gt_all_permutation_medians_cells": int(sum(row["frozen_support_gt_all_permutation_medians"] for row in cells)),
        "frozen_support_median_gain_range": [min(row["frozen_support_median_gain"] for row in cells), max(row["frozen_support_median_gain"] for row in cells)],
        "permutation_median_of_medians_range": [min(row["movement_permutation_median_of_medians"] for row in cells), max(row["movement_permutation_median_of_medians"] for row in cells)],
        "single_role_median_ranges": {
            role: [min(row["single_role_median_gains"][role] for row in cells), max(row["single_role_median_gains"][role] for row in cells)]
            for role in cells[0]["single_role_median_gains"]
        },
        "coalition_median_ranges": {
            name: [min(row["coalition_median_gains"][name] for row in cells), max(row["coalition_median_gains"][name] for row in cells)]
            for name in cells[0]["coalition_median_gains"]
        },
        "query_truth_flips": int(sum(row["modes"][query_mode]["truth_flip"] for row in selected)),
        "record_path_truth_flips": int(sum(row["modes"][path_mode]["truth_flip"] for row in selected)),
        "all_role_truth_flips": int(sum(row["modes"][all_mode]["truth_flip"] for row in selected)),
        "record_path_additional_truth_flips": int(sum(row["modes"][path_mode]["truth_flip"] and not row["modes"][query_mode]["truth_flip"] for row in selected)),
        "all_role_additional_truth_flips": int(sum(row["modes"][all_mode]["truth_flip"] and not row["modes"][query_mode]["truth_flip"] for row in selected)),
    }


def main() -> None:
    adjudication = core.load(OUT / "analysis/adjudication.json")
    source_audit = core.load(OUT / "audit/independent_batch_intervention_audit.json")
    if adjudication["authorization"] != "run_phase1617_c112_synthesis_heatmap_and_closure" or not source_audit["all_checks_passed"]:
        raise RuntimeError("C112 closure authorization missing")
    rows = core.rows(OUT / "analysis/batch_intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/batch_intervention_summary.jsonl")
    rollup = {family: family_rollup(rows, summaries, family) for family in ("attribute_binding", "agent_patient")}

    payload = core.load(PUBLIC)
    payload.update({
        "phase": 1617,
        "campaign": "C109-C112",
        "title": "C109-C112 Coordinate Assignment / Role-Lattice Atlas",
        "c112_batch": {
            "predictions": adjudication["predictions"],
            "max_permutation_l2_relative_error": adjudication["max_permutation_l2_relative_error"],
            "summaries": summaries,
            "family_rollup": rollup,
        },
        "claim_boundary": "C112 shows that physical assignment of the frozen truth movement matters for attribute output leverage and that agent truth transport is distributed across a record-to-query role path. This is Qwen3 controlled-English activation transport at state19: it does not establish a minimal or necessary circuit, natural-language universality, weights, attention/MLP components, or semantic neurons.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c112_coordinate_role_lattice_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)

    closure = {
        "phase": 1617,
        "campaign": "C112",
        "created_at_utc": now(),
        "status": "coordinate_assignment_role_lattice_major_stage_complete",
        "headline": {"predictions": adjudication["predictions"], "max_permutation_l2_relative_error": adjudication["max_permutation_l2_relative_error"], "family_rollup": rollup},
        "new_puzzles": {
            "K292-R1": "exact-energy coordinate assignment: attribute frozen K256 movement beats the median and every one of eight within-support movement-permutation medians in all four cells; physical assignment inside the support matters for output leverage",
            "K293-R1": "relation-conditioned role lattice: attribute transport is dominated by query_anchor, while agent has positive single-role leverage at focus_record, query_focus, query_anchor, and code_instruction and its record-to-query coalition beats query alone in all four cells",
            "K294-R1": "distributed agent rescue: agent query alone flips 6/96 pairs, the frozen record-to-query path flips 18/96 with 12 additional flips, and all seven roles flip 29/96 with 23 additional flips; attribute is already concentrated at query with 76/96, 77/96, and 78/96 respectively",
        },
        "theory_update": "RDC gains two experimentally separated structures: coordinate assignment V inside a readable support S, and a relation-typed role coalition C_f. Attribute behaves like a query-centered field with coordinate-specific sparse leverage; agent behaves like a distributed record-to-query field whose components combine before boundary compilation.",
        "unified_formula": "y = O_c(L_{S,V,C_f,s}(R[f,r,s]))",
        "claim_boundary": payload["claim_boundary"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC)},
        "next_authorization": "C113 fourth-lexicon prospective replication: freeze K292 attribute coordinate-assignment and K294 agent role-lattice predictions, add leave-one-role-out necessity candidates, preserve observation-first reporting, and continue embedding/HiddenState only",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "source": source_audit["all_checks_passed"],
        "rows": len(rows) == 192,
        "summaries": len(summaries) == 8,
        "l2": adjudication["max_permutation_l2_relative_error"] < 0.001,
        "attribute_assignment": rollup["attribute_binding"]["frozen_support_gt_all_permutation_medians_cells"] == 4,
        "agent_roles": adjudication["predictions"]["agent_focus_record_positive_cells"] == 4 and adjudication["predictions"]["agent_record_path_gt_query_cells"] == 4,
        "flips": rollup["attribute_binding"]["query_truth_flips"] == 76 and rollup["agent_patient"]["record_path_additional_truth_flips"] == 12 and rollup["agent_patient"]["all_role_additional_truth_flips"] == 23,
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": "does not establish a minimal" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1617, "campaign": "C112", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(canonical), "authorization": "audit_frontend_append_c112_memo_and_close_current_major_stage"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "new_puzzles": closure["new_puzzles"], "heatmap": closure["heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
