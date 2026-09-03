#!/usr/bin/env python3
"""Phase1622 / C113: synthesize evidence, extend the coordinate heatmap, and close."""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 32, 36)
RAW_STATES = (0, 8, 16, 19, 24, 32, 36)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def family_rollup(rows: list[dict], summaries: list[dict], family: str) -> dict:
    selected = [row for row in rows if row["family"] == family]
    cells = [row for row in summaries if row["family"] == family]
    modes = {
        "query": "single_query_anchor", "path": "coalition_record_to_query_path",
        "path_code": "coalition_path_plus_code", "path_code_boundary": "coalition_path_plus_code_boundary",
        "all": "coalition_all_registered_roles",
    }
    return {
        "pairs": len(selected),
        "frozen_support_gt_permutation_median_cells": int(sum(row["frozen_support_gt_permutation_median"] for row in cells)),
        "frozen_support_gt_all_permutation_medians_cells": int(sum(row["frozen_support_gt_all_permutation_medians"] for row in cells)),
        "frozen_support_median_gain_range": [min(row["frozen_support_median_gain"] for row in cells), max(row["frozen_support_median_gain"] for row in cells)],
        "single_role_median_ranges": {role: [min(row["single_role_median_gains"][role] for row in cells), max(row["single_role_median_gains"][role] for row in cells)] for role in cells[0]["single_role_median_gains"]},
        "coalition_median_ranges": {name: [min(row["coalition_median_gains"][name] for row in cells), max(row["coalition_median_gains"][name] for row in cells)] for name in cells[0]["coalition_median_gains"]},
        "leave_one_path_loss_ranges": {role: [min(row["leave_one_path_median_losses"][role] for row in cells), max(row["leave_one_path_median_losses"][role] for row in cells)] for role in cells[0]["leave_one_path_median_losses"]},
        "staged_increment_ranges": {name: [min(row["staged_increments"][name] for row in cells), max(row["staged_increments"][name] for row in cells)] for name in cells[0]["staged_increments"]},
        **{f"{name}_truth_flips": int(sum(row["modes"][mode]["truth_flip"] for row in selected)) for name, mode in modes.items()},
    }


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    field = core.load(OUT / "analysis/field_adjudication.json")
    intervention = core.load(OUT / "analysis/intervention_adjudication.json")
    audit = core.load(OUT / "audit/independent_intervention_audit.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    if intervention["authorization"] != "run_phase1622_c113_synthesis_heatmap_and_closure" or not audit["all_checks_passed"]:
        raise RuntimeError("C113 closure authorization missing")
    rows = core.rows(OUT / "analysis/intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/intervention_summary.jsonl")
    rollup = {family: family_rollup(rows, summaries, family) for family in protocol["families"]}

    payload = core.load(PUBLIC)
    mean = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    role_index = {role: index for index, role in enumerate(protocol["roles"])}
    c113_effect_rows = []
    for family_index, family in enumerate(protocol["families"]):
        for partition_index, partition in enumerate(protocol["partitions"]):
            for role in protocol["roles"]:
                for state in KEY_STATES:
                    c113_effect_rows.append({
                        "dataset": "C113", "family": family, "partition": partition, "role": role,
                        "state": state, "state_kind": "embedding" if state == 0 else "hidden_state",
                        "effect": "balanced_truth_walsh", "values": np.asarray(mean[family_index, partition_index, role_index[role], state], dtype=np.float32).tolist(),
                    })
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    raw_field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    c113_raw_rows = []
    for family in protocol["families"]:
        for partition in protocol["partitions"]:
            row_index = next(index for index, row in enumerate(compiled) if row["family"] == family and row["partition"] == partition)
            row = compiled[row_index]
            occurrence = lookup[(row_index, "query_anchor")][0]
            occurrence_index = int(occurrence["occurrence_index"])
            for state in RAW_STATES:
                c113_raw_rows.append({
                    "dataset": "C113", "case_id": row["case_id"], "family": family, "partition": partition,
                    "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"], "code": row["code"], "role": "query_anchor",
                    "subtoken": int(occurrence["subtoken"]), "token_position": int(occurrence["token_position"]),
                    "token_id": int(occurrence["token_id"]), "token_text": occurrence["token_text"],
                    "state": state, "state_kind": "embedding" if state == 0 else "hidden_state",
                    "values": decode_bf16(raw_field[state, occurrence_index]).tolist(),
                })
    payload["effect_rows"] = [row for row in payload["effect_rows"] if row.get("dataset") != "C113"] + c113_effect_rows
    payload["raw_rows"] = [row for row in payload["raw_rows"] if row.get("dataset") != "C113"] + c113_raw_rows
    candidate_vectors = [np.asarray(row["values"], dtype=np.float32) for row in c113_effect_rows if row["role"] == "query_anchor" and row["state"] == 19]
    payload["default_coordinates"] = np.argsort(-np.mean(np.stack([np.abs(vector) for vector in candidate_vectors]), axis=0), kind="stable")[:64].astype(int).tolist()
    payload["scale"] = {
        "effect_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]]), 0.99)),
        "raw_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]]), 0.99)),
    }
    payload.update({
        "phase": 1622, "campaign": "C109-C113", "title": "C109-C113 Coordinate Assignment / Multi-Position Field Atlas",
        "c113_batch": {
            "field_prediction": field, "behavior": capture["behavior"], "predictions": intervention["predictions"],
            "max_permutation_l2_relative_error": intervention["max_permutation_l2_relative_error"],
            "summaries": summaries, "family_rollup": rollup,
        },
        "claim_boundary": "C113 prospectively replicates fourth-lexicon full truth fields. Attribute exact coordinate assignment beats permutation medians in all cells but beats every permutation in only 2/4 cells. Agent multi-position gain is query-centered and code-modulated: query_anchor/query_focus are stable leave-one candidates, focus_record is partition-dependent, boundary is mixed, and focus_pre adds zero. These are controlled-English Qwen3 activation interventions, not weights, semantic neurons, an endogenous transport path, attention/MLP components, or a universal language mechanism.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c113_coordinate_multi_position_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)

    closure = {
        "phase": 1622, "campaign": "C113", "created_at_utc": now(),
        "status": "fourth_lexicon_field_and_role_lattice_replication_complete",
        "headline": {"field_results": field["results"], "predictions": intervention["predictions"], "family_rollup": rollup, "behavior": capture["behavior"], "max_permutation_l2_relative_error": intervention["max_permutation_l2_relative_error"]},
        "new_puzzles": {
            "K295": "fourth-lexicon full-field replication: query_anchor@state19 truth fields pass cross-partition, C110-reference, and frozen-support-overlap gates for both attribute binding and agent-patient",
            "K296": "graded coordinate assignment boundary: attribute K256 correct movement exceeds the median of eight exact-energy within-support permutations in 4/4 cells but exceeds every permutation in only 2/4; physical assignment matters statistically, but a unique exact coordinate dictionary is not established",
            "K297": "query-centered multi-position agent field: path exceeds query alone in 4/4 cells, and leave-one query_anchor/query_focus lowers median gain in 4/4; focus_record is helpful only in lockbox, code_instruction supplies most all-over-path gain, boundary is mixed, and focus_pre adds zero",
        },
        "theory_update": "RDC is narrowed from a presumed record-to-query route to a context-indexed multi-position response field. A readable relation field R can be stable across lexicons, while output leverage depends on coordinate assignment V, a query-centered coalition C, and later protocol state P; neither V nor C is yet unique or minimal.",
        "unified_formula": "y = O_c(L_{S,V,C,P,s}(R[f,r,s]))",
        "claim_corrections": {
            "C112_path": "simultaneous multi-position rescue did not establish an endogenous record-to-query path; C113 localizes the stable agent contribution to query_anchor/query_focus, with partition-dependent focus_record and code-driven late gain",
            "coordinate_dictionary": "winning against permutation medians is evidence for structured coordinate assignment, not a one-coordinate-one-meaning dictionary or a unique physical ordering",
        },
        "problems": [
            "controlled synthetic lexicon and fixed English prompt; no human naturalness audit or cross-model replication",
            "192 intervention pairs arise from 24 lexical units and are not 192 independent lexical replications",
            "state19 is frozen from prior work; no layer-general necessity result",
            "simultaneous activation patching can be off-manifold and cannot reveal natural computation order",
            "standard code succeeds while reversed code fails, so task-aligned rescue is typed missingness",
        ],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": 2560, "includes_embedding": True, "hidden_states": list(RAW_STATES[1:])},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C114 existing-data structural atlas: jointly mine C110-C113 without new model runs to separate stable query-centered roles, protocol-stage gains, and coordinate-assignment equivalence classes; freeze any fifth-lexicon predictions only after this descriptive pass",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "source": audit["all_checks_passed"], "rows": len(rows) == 192, "summaries": len(summaries) == 8,
        "effects": len(c113_effect_rows) == 280 and all(len(row["values"]) == 2560 for row in c113_effect_rows),
        "raw": len(c113_raw_rows) == 28 and all(len(row["values"]) == 2560 for row in c113_raw_rows),
        "embedding_hidden": {row["state_kind"] for row in c113_raw_rows} == {"embedding", "hidden_state"},
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "field": len(field["passed_families"]) == 2,
        "graded_assignment": intervention["predictions"]["attribute_frozen_gt_all_permutation_cells"] == 2 and rollup["attribute_binding"]["frozen_support_gt_permutation_median_cells"] == 4,
        "agent": all(intervention["predictions"][key] == 4 for key in ("agent_record_path_gt_query_cells", "agent_all_roles_gt_path_cells", "agent_leave_query_anchor_lowers_cells", "agent_leave_query_focus_lowers_cells")),
        "boundary": "not weights" in payload["claim_boundary"] and "not establish an endogenous" in closure["claim_corrections"]["C112_path"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1622, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(canonical), "authorization": "audit_frontend_build_append_c113_memo_then_execute_c114_existing_data_atlas"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "new_puzzles": closure["new_puzzles"], "heatmap": closure["heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
