#!/usr/bin/env python3
"""Independent stage audits for the C115 fifth-lexicon campaign."""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1625_c115_fifth_lexicon_prospective_replication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def decode(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cos(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def save(name: str, phase: int, checks: dict, authorization: str) -> None:
    report = {"phase": phase, "campaign": "C115", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": authorization}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / f"audit/{name}.json", report)
    print(json.dumps(report, indent=2))


def contract_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_pre_model_audit.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    cells = Counter((row["family"], row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    checks = {
        "internal": internal["all_checks_passed"],
        "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1625_c115_fifth_lexicon_common.py"),
        "material_digest": protocol["material_digest"] == core.digest([*units, *cases]),
        "counts": len(units) == 48 and len(cases) == 768 and len(compiled) == 768 and len(manifest) == protocol["occurrences"],
        "partitions": Counter((row["family"], row["partition"]) for row in units) == {(family, partition): 12 for family in protocol["families"] for partition in protocol["partitions"]},
        "factorial": len(cells) == 64 and set(cells.values()) == {12},
        "unique": len({row["prompt"] for row in cases}) == 768 and len({value.casefold() for row in units for value in row["values"]}) == 240,
        "roles": all(set(row["role_positions"]) == set(protocol["roles"]) for row in compiled),
        "sources": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "predictions": set(protocol["operational_gates"]) == {"both_field_families_pass", "both_families_correct_movement_gt_permutation_median_cells", "agent_record_path_gt_query_cells", "agent_query_anchor_positive_cells", "agent_query_focus_positive_cells", "agent_leave_query_anchor_lowers_cells", "agent_leave_query_focus_lowers_cells", "agent_leave_focus_post_lowers_cells"},
        "boundary": all(term in protocol["claim_boundary"] for term in ("not weights", "finite permutations", "no attention/MLP")),
        "authorization": protocol["authorization"] == "execute_phase1626_c115_exact_field_capture",
    }
    save("independent_pre_model_audit", 1625, checks, protocol["authorization"])


def capture_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/capture_summary.json")
    field_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    field = np.load(field_path, mmap_mode="r")
    logits = np.load(logits_path, mmap_mode="r")
    index = core.rows(index_path)
    sample = decode(field[np.asarray([0, 19, 36])[:, None], np.asarray([0, len(index) // 2, protocol["occurrences"] - 1])[None, :]])
    checks = {
        "contract": core.load(OUT / "audit/independent_pre_model_audit.json")["all_checks_passed"],
        "shape": list(field.shape) == protocol["archive"]["shape"] and field.dtype == np.uint16,
        "hash": core.sha(field_path) == report["raw_sha256"],
        "logits": list(logits.shape) == [768, 2] and logits.dtype == np.float32 and core.sha(logits_path) == report["logits_sha256"] and bool(np.isfinite(logits).all()),
        "index": len(index) == 768 and core.sha(index_path) == report["index_sha256"] and all(row["row_index"] == i for i, row in enumerate(index)),
        "sample_finite": bool(np.isfinite(sample).all()),
        "deterministic": all(value == 0.0 for value in report["numeric"].values()),
        "numeric": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"],
        "authorization": report["authorization"] == "run_phase1627_c115_field_adjudication",
    }
    save("independent_capture_audit", 1626, checks, report["authorization"])


def field_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/field_adjudication.json")
    field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    unit_truth = np.load(OUT / "analysis/unit_truth_role_state.float32.npy", mmap_mode="r")
    mean_truth = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    role_index = {role: index for index, role in enumerate(protocol["roles"])}
    first_unit = rows[0]["unit_id"]
    recomputed = np.zeros(2560, dtype=np.float32)
    for row_index, row in enumerate(rows):
        if row["unit_id"] == first_unit:
            values = decode(field[19, lookup[(row_index, "query_anchor")]])
            recomputed += float(row["truth_factor"]) / 16.0 * np.mean(values, axis=0, dtype=np.float32)
    results = core.rows(OUT / "analysis/field_prediction_results.jsonl")
    independent_cos = {}
    for family_index, family in enumerate(protocol["families"]):
        independent_cos[family] = cos(np.asarray(mean_truth[family_index, 0, role_index["query_anchor"], 19]), np.asarray(mean_truth[family_index, 1, role_index["query_anchor"], 19]))
    checks = {
        "capture": core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"],
        "shapes": list(unit_truth.shape) == [48, 7, 37, 2560] and list(mean_truth.shape) == [2, 2, 7, 37, 2560],
        "hashes": core.sha(OUT / "analysis/unit_truth_role_state.float32.npy") == report["unit_sha256"] and core.sha(OUT / "analysis/mean_truth_role_state.float32.npy") == report["mean_sha256"],
        "sample_recompute": bool(np.array_equal(recomputed, unit_truth[0, role_index["query_anchor"], 19])),
        "cross_cosines": all(abs(independent_cos[row["family"]] - row["cross_partition_cosine"]) < 1e-7 for row in results),
        "results": len(results) == 2 and all(set(row["gates"]) == {"cross_partition", "reference", "support_overlap"} for row in results),
        "trajectory": len(core.rows(OUT / "analysis/role_state_trajectory.jsonl")) == 518,
        "authorization": report["authorization"] == "execute_phase1628_c115_coordinate_and_role_interventions_regardless_of_field_gate",
    }
    save("independent_field_adjudication_audit", 1627, checks, report["authorization"])


def intervention_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/intervention_adjudication.json")
    rows = core.rows(OUT / "analysis/intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/intervention_summary.jsonl")
    recomputed = {}
    for family in protocol["families"]:
        family_cells = [row for row in summaries if row["family"] == family]
        recomputed[f"{family}_median"] = sum(row["frozen_support_gt_permutation_median"] for row in family_cells)
    agent = [row for row in summaries if row["family"] == "agent_patient"]
    expected = {
        "attribute_median_win_cells": recomputed["attribute_binding_median"],
        "agent_median_win_cells": recomputed["agent_patient_median"],
        "attribute_strict_win_cells_descriptive": sum(row["frozen_support_gt_all_permutation_medians"] for row in summaries if row["family"] == "attribute_binding"),
        "agent_strict_win_cells_descriptive": sum(row["frozen_support_gt_all_permutation_medians"] for row in agent),
        "agent_record_path_gt_query_cells": sum(row["record_path_gt_query"] for row in agent),
        "agent_query_anchor_positive_cells": sum(row["single_role_median_gains"]["query_anchor"] > 0 for row in agent),
        "agent_query_focus_positive_cells": sum(row["single_role_median_gains"]["query_focus"] > 0 for row in agent),
        "agent_leave_query_anchor_lowers_cells": sum(row["leave_query_anchor_lowers"] for row in agent),
        "agent_leave_query_focus_lowers_cells": sum(row["leave_query_focus_lowers"] for row in agent),
        "agent_leave_focus_post_lowers_cells": sum(row["leave_focus_post_lowers"] for row in agent),
    }
    max_error = max(error for row in rows for error in row["permutation_l2_relative_errors"])
    checks = {
        "field": core.load(OUT / "audit/independent_field_adjudication_audit.json")["all_checks_passed"],
        "counts": len(rows) == 384 and len(summaries) == 8 and all(row["pairs"] == 48 and row["independent_units"] == 12 for row in summaries),
        "modes": all(set(row["modes"]) == set(protocol["modes"]) for row in rows),
        "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in rows for mode in protocol["modes"]),
        "l2": abs(max_error - report["max_permutation_l2_relative_error"]) < 1e-12 and max_error <= protocol["numeric"]["movement_permutation_actual_l2_relative_tolerance"],
        "predictions": expected == report["predictions"],
        "prediction_checks": all(report["prediction_checks"][key] == (report["predictions"][key] == value) for key, value in protocol["operational_gates"].items() if key in report["predictions"]),
        "hashes": core.sha(OUT / "analysis/intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/intervention_summary.jsonl") == report["summary_sha256"],
        "authorization": report["authorization"] == "run_phase1629_c115_synthesis_heatmap_and_closure",
    }
    save("independent_intervention_audit", 1628, checks, report["authorization"])


def closure_audit() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    internal = core.load(OUT / "audit/internal_closure_audit.json")
    payload = core.load(PUBLIC)
    effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C115"]
    raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C115"]
    checks = {
        "internal": internal["all_checks_passed"],
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == internal["asset_sha256"],
        "effect_rows": len(effects) == 280 and all(len(row["values"]) == 2560 for row in effects),
        "raw_rows": len(raw) == 28 and all(len(row["values"]) == 2560 for row in raw),
        "states": {row["state_kind"] for row in raw} == {"embedding", "hidden_state"},
        "batch": payload["campaign"] == "C109-C115" and "c115_batch" in payload and "c114_structural_atlas" in payload,
        "boundary": all(term in closure["claim_boundary"] for term in ("do not establish", "gauge symmetry", "attention/MLP")),
        "authorization": closure["next_authorization"].startswith("C116 observation-first third-relation-family campaign"),
    }
    save("independent_closure_audit", 1629, checks, "append_c115_memo_build_frontend_then_execute_c116")


STAGES = {"contract": contract_audit, "capture": capture_audit, "field": field_audit, "intervene": intervention_audit, "closure": closure_audit}


def main(stage: str) -> None:
    STAGES[stage]()

