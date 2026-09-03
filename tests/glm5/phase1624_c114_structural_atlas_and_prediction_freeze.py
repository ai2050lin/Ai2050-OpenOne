#!/usr/bin/env python3
"""Phase1624 / C114: build the descriptive C112-C113 structural atlas."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1623_c114_existing_data_structural_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def value_range(rows: list[dict], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [min(values), max(values)]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/independent_contract_audit.json")
    if protocol["authorization"] != "execute_phase1624_c114_structural_atlas_and_freeze_c115_predictions" or not audit["all_checks_passed"]:
        raise RuntimeError("C114 authorization missing")
    source_summaries = {
        "C112": core.rows(Path(protocol["source_paths"]["c112_summary"])),
        "C113": core.rows(Path(protocol["source_paths"]["c113_summary"])),
    }
    cells = []
    for dataset, summaries in source_summaries.items():
        for row in summaries:
            correct = float(row["frozen_support_median_gain"])
            permutations = [float(value) for value in row["movement_permutation_median_gains"]]
            query = float(row["single_role_median_gains"]["query_anchor"])
            path = float(row["coalition_median_gains"]["record_to_query_path"])
            all_roles = float(row["coalition_median_gains"]["all_registered_roles"])
            cells.append({
                "dataset": dataset, "family": row["family"], "partition": row["partition"], "code": int(row["code"]),
                "pairs": int(row["pairs"]), "independent_units": int(row["independent_units"]),
                "frozen_support_median_gain": correct, "permutation_median": float(np.median(permutations)),
                "permutation_max": max(permutations), "correct_minus_permutation_median": correct - float(np.median(permutations)),
                "correct_minus_permutation_max": correct - max(permutations),
                "conservative_rank": 1 + sum(value >= correct for value in permutations),
                "beats_permutation_median": correct > float(np.median(permutations)), "strictly_beats_all_permutations": correct > max(permutations),
                "query_gain": query, "path_gain": path, "all_role_gain": all_roles,
                "path_minus_query": path - query, "all_minus_path": all_roles - path,
                "single_role_gains": {role: float(value) for role, value in row["single_role_median_gains"].items()},
            })
    cell_path = OUT / "analysis/structural_cells.jsonl"
    core.write_rows(cell_path, cells)
    rollups = {}
    for family in protocol["families"]:
        selected = [row for row in cells if row["family"] == family]
        rollups[family] = {
            "cells": len(selected),
            "beats_permutation_median_cells": int(sum(row["beats_permutation_median"] for row in selected)),
            "strictly_beats_all_permutations_cells": int(sum(row["strictly_beats_all_permutations"] for row in selected)),
            "conservative_rank_counts": dict(sorted(Counter(str(row["conservative_rank"]) for row in selected).items())),
            "correct_minus_permutation_median_range": value_range(selected, "correct_minus_permutation_median"),
            "path_gt_query_cells": int(sum(row["path_minus_query"] > 0 for row in selected)),
            "all_gt_path_cells": int(sum(row["all_minus_path"] > 0 for row in selected)),
            "path_minus_query_range": value_range(selected, "path_minus_query"),
            "all_minus_path_range": value_range(selected, "all_minus_path"),
            "single_role_positive_cells": {role: int(sum(row["single_role_gains"][role] > 0 for row in selected)) for role in selected[0]["single_role_gains"]},
            "by_dataset": {
                dataset: {
                    "beats_median": int(sum(row["beats_permutation_median"] for row in selected if row["dataset"] == dataset)),
                    "strict_win_all": int(sum(row["strictly_beats_all_permutations"] for row in selected if row["dataset"] == dataset)),
                    "path_gt_query": int(sum(row["path_minus_query"] > 0 for row in selected if row["dataset"] == dataset)),
                    "all_gt_path": int(sum(row["all_minus_path"] > 0 for row in selected if row["dataset"] == dataset)),
                }
                for dataset in protocol["datasets"]
            },
        }
    c113 = source_summaries["C113"]
    composition = {}
    for family in protocol["families"]:
        selected = [row for row in c113 if row["family"] == family]
        composition[family] = {
            "leave_one_positive_loss_cells": {role: int(sum(float(row["leave_one_path_median_losses"][role]) > 0 for row in selected)) for role in selected[0]["leave_one_path_median_losses"]},
            "leave_one_loss_ranges": {role: [min(float(row["leave_one_path_median_losses"][role]) for row in selected), max(float(row["leave_one_path_median_losses"][role]) for row in selected)] for role in selected[0]["leave_one_path_median_losses"]},
            "staged_positive_increment_cells": {name: int(sum(float(row["staged_increments"][name]) > 0 for row in selected)) for name in selected[0]["staged_increments"]},
            "staged_increment_ranges": {name: [min(float(row["staged_increments"][name]) for row in selected), max(float(row["staged_increments"][name]) for row in selected)] for name in selected[0]["staged_increments"]},
        }
    atlas = {
        "phase": 1624, "campaign": "C114", "created_at_utc": now(), "status": "existing_data_structural_atlas_complete",
        "cells": cells, "rollups": rollups, "c113_only_composition": composition,
        "structural_readout": {
            "stable": [
                "both families beat the median of eight exact-energy coordinate permutations in all 8/8 exposed C112-C113 cells",
                "agent record/query coalition exceeds query alone in all 8/8 cells",
                "agent query_anchor and query_focus single-role gains are positive in all 8/8 cells",
                "C113 agent leave-one query_anchor, query_focus, and focus_post losses are positive in all 4/4 cells",
            ],
            "nonstationary": [
                "attribute strict victory over every permutation falls from C112 4/4 to C113 2/4",
                "agent focus_record single-role gain and leave-one contribution change sign across lexical partitions",
                "boundary increments change sign, while focus_pre contributes exactly zero in all C113 cells",
            ],
            "protocol_stage": "C113 code_instruction adds positive agent gain in 4/4 cells; it explains most all-role-over-path gain and must not be counted as upstream relation transport",
        },
        "c115_frozen_prediction_template": {
            "field": "both families pass the same cross-partition, reference-cosine, and support-overlap gates at query_anchor@state19",
            "coordinate_assignment": "both families' correct support movement exceeds the median of eight exact-energy within-support permutations in all four cells; strict victory over all permutations is descriptive only",
            "agent_multi_position": "record/query coalition exceeds query alone in all four cells",
            "agent_role_candidates": "query_anchor and query_focus single-role gains are positive in all four cells; leave-one query_anchor/query_focus/focus_post tests remain prospective candidates",
            "protocol": "code_instruction gain is reported separately and cannot upgrade an upstream relation-field claim",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    atlas_path = OUT / "analysis/structural_atlas.json"
    core.save(atlas_path, atlas)
    payload = core.load(PUBLIC)
    payload.update({"phase": 1624, "campaign": "C109-C114", "title": "C109-C114 Coordinate Assignment / Multi-Position Structural Atlas", "c114_structural_atlas": atlas, "created_at_utc": now()})
    canonical = OUT / "visualization/c109_c114_structural_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    closure = {
        "phase": 1624, "campaign": "C114", "created_at_utc": now(), "status": "descriptive_structural_atlas_and_c115_prediction_freeze_complete",
        "headline": {"rollups": rollups, "c113_only_composition": composition, "structural_readout": atlas["structural_readout"]},
        "new_puzzles": {
            "K298-OBS": "graded assignment law: correct support movement beats exact-energy permutation medians in 8/8 exposed cells across C112-C113, while strict victory over every permutation is 6/8; coordinate assignment is structured but equivalence classes remain",
            "K299-OBS": "stable agent query coalition: path exceeds query and query_anchor/query_focus single gains are positive in 8/8 exposed cells; C113 leave-one additionally implicates query_anchor/query_focus/focus_post in 4/4",
            "K300-BOUNDARY": "record and protocol separation: focus_record is lexicon/partition-dependent, code_instruction is consistently positive for agent in C113, boundary is sign-unstable, and focus_pre adds zero; simultaneous multi-position rescue is not a natural path",
        },
        "theory_update": "RDC now treats the candidate mechanism as a graded coordinate-assignment equivalence class plus a query-centered response coalition, with protocol state factored separately. This is a descriptive compression of exposed data, not a new confirmatory mechanism result.",
        "unified_formula": "y = O_c(L_{[S,V],C_q,P,s}(R[f,r,s]))",
        "c115_frozen_prediction_template": atlas["c115_frozen_prediction_template"],
        "claim_boundary": protocol["claim_boundary"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC)},
        "next_authorization": "C115 fifth-lexicon prospective test using the frozen C114 prediction template; retain observation-first embedding/HiddenState capture and treat every failed route as route-level evidence",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "sources": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "cells": len(cells) == 16 and sum(row["pairs"] for row in cells) == 384,
        "rollups": set(rollups) == set(protocol["families"]) and all(row["cells"] == 8 for row in rollups.values()),
        "composition": set(composition) == set(protocol["families"]),
        "stable_counts": rollups["attribute_binding"]["beats_permutation_median_cells"] == 8 and rollups["agent_patient"]["beats_permutation_median_cells"] == 8 and rollups["agent_patient"]["path_gt_query_cells"] == 8,
        "graded": rollups["attribute_binding"]["strictly_beats_all_permutations_cells"] == 6,
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": "not a new confirmatory" in closure["theory_update"] and "descriptive only" in atlas["c115_frozen_prediction_template"]["coordinate_assignment"],
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "rollups": rollups})
    report = {"phase": 1624, "campaign": "C114", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "atlas_sha256": core.sha(atlas_path), "asset_sha256": core.sha(canonical), "authorization": "audit_client_append_c114_memo_and_close_current_major_stage"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "rollups": rollups, "composition": composition, "heatmap": closure["heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
