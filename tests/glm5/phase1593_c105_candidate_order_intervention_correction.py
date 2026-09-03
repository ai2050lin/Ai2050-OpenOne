#!/usr/bin/env python3
"""Phase1593 / C105: correct the reversed candidate-margin semantics in C102 and C104."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C102 = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
OUT = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base

PHASE = 1593
CAMPAIGN = "C105"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_candidate_order() -> dict:
    rows = core.rows(C104 / "compiled/qwen3.jsonl")
    candidate_ids = rows[0]["candidate_ids"]
    if any(row["candidate_ids"] != candidate_ids for row in rows):
        raise RuntimeError("candidate order varies")
    tok = graph_base.tokenizer()
    decoded = [tok.decode(ids) for ids in candidate_ids]
    normalized = [value.strip().casefold() for value in decoded]
    if normalized != ["yes", "no"]:
        raise RuntimeError((candidate_ids, decoded))
    return {"candidate_ids": candidate_ids, "decoded": decoded, "normalized": normalized}


def correct_c102() -> dict:
    source_path = C102 / "analysis/coordinate_coalition_intervention_results.jsonl"
    source = core.rows(source_path)
    corrected = []
    for row in source:
        corrected.append({
            **{key: value for key, value in row.items() if key not in ("recipient_margin", "donor_margin", "donor_target_gap", "modes")},
            "recipient_target_margin_corrected": -row["recipient_margin"],
            "donor_target_margin_corrected": -row["donor_margin"],
            "donor_target_gap_corrected": -row["donor_target_gap"],
            "modes": {mode: {**entry, "target_gain_corrected": -entry["target_gain"]} for mode, entry in row["modes"].items()},
            "correction": "candidate order is [yes,no], while the source script computed candidate[1]-candidate[0]; negate all target-oriented margins and gains",
        })
    path = OUT / "analysis/c102_corrected_intervention_results.jsonl"
    core.write_rows(path, corrected)
    protocol = core.load(C102 / "protocol/intervention_protocol.json")
    summaries = []
    for family in sorted({row["family"] for row in corrected}):
        for partition in ("confirmation", "lockbox"):
            selected = [row for row in corrected if row["family"] == family and row["partition"] == partition]
            modes = protocol["modes"]
            medians = {mode: float(np.median([row["modes"][mode]["target_gain_corrected"] for row in selected])) for mode in modes}
            sparse = selected[0]["k"] < 2560
            controls = ["sign_reversed", "same_truth_donor"] + (["wrong_family_support"] if sparse else [])
            summaries.append({
                "family": family, "partition": partition, "state": selected[0]["state"], "k": selected[0]["k"],
                "pairs": len(selected), "median_donor_target_gap_corrected": float(np.median([row["donor_target_gap_corrected"] for row in selected])),
                "median_target_gain_corrected": medians,
                "informative_controls": controls,
                "correct_positive": medians["correct_frozen_support"] > 0.0,
                "correct_beats_informative_controls": all(medians["correct_frozen_support"] > medians[control] for control in controls),
                "typed_missing_full_support_controls": not sparse,
            })
    summary_path = OUT / "analysis/c102_corrected_intervention_summary.jsonl"
    core.write_rows(summary_path, summaries)
    rollup = []
    for family in sorted({row["family"] for row in summaries}):
        selected = [row for row in summaries if row["family"] == family]
        rollup.append({
            "family": family,
            "controlled_partitions": sum(row["correct_positive"] and row["correct_beats_informative_controls"] for row in selected),
            "total_partitions": len(selected),
            "both_partitions_controlled": all(row["correct_positive"] and row["correct_beats_informative_controls"] for row in selected),
        })
    rollup_path = OUT / "analysis/c102_corrected_family_rollup.jsonl"
    core.write_rows(rollup_path, rollup)
    return {
        "source_sha256": core.sha(source_path), "results_sha256": core.sha(path), "summary_sha256": core.sha(summary_path),
        "rollup_sha256": core.sha(rollup_path), "rows": len(corrected), "summary_rows": len(summaries),
        "fully_controlled_families": [row["family"] for row in rollup if row["both_partitions_controlled"]],
    }


def correct_c104() -> dict:
    source_path = C104 / "analysis/upstream_role_intervention_results.jsonl"
    source = core.rows(source_path)
    corrected = []
    for row in source:
        corrected.append({
            **{key: value for key, value in row.items() if key not in ("recipient_yes_margin", "donor_yes_margin", "donor_true_direction_gap", "modes")},
            "recipient_yes_minus_no_corrected": -row["recipient_yes_margin"],
            "donor_yes_minus_no_corrected": -row["donor_yes_margin"],
            "donor_true_direction_gap_corrected": -row["donor_true_direction_gap"],
            "modes": {mode: {"yes_minus_no_corrected": -entry["yes_margin"], "true_direction_gain_corrected": -entry["true_direction_gain"]} for mode, entry in row["modes"].items()},
            "correction": "candidate order is [yes,no], so Yes-minus-No is the negative of the stored candidate[1]-candidate[0] value",
        })
    path = OUT / "analysis/c104_corrected_intervention_results.jsonl"
    core.write_rows(path, corrected)
    protocol = core.load(C104 / "protocol/upstream_intervention_protocol.json")
    summaries = []
    for family in sorted({row["family"] for row in corrected}):
        for partition in ("confirmation", "lockbox"):
            for code in (1, -1):
                selected = [row for row in corrected if row["family"] == family and row["partition"] == partition and row["code"] == code]
                medians = {mode: float(np.median([row["modes"][mode]["true_direction_gain_corrected"] for row in selected])) for mode in protocol["modes"]}
                summaries.append({
                    "family": family, "partition": partition, "code": code, "codebook": selected[0]["codebook"],
                    "role": selected[0]["role"], "state": selected[0]["state"], "pairs": len(selected),
                    "median_donor_true_direction_gap_corrected": float(np.median([row["donor_true_direction_gap_corrected"] for row in selected])),
                    "median_true_direction_gain_corrected": medians,
                    "correct_positive": medians["correct_role_state"] > 0.0,
                    "correct_beats_all_controls": all(medians["correct_role_state"] > medians[mode] for mode in protocol["modes"] if mode != "correct_role_state"),
                })
    summary_path = OUT / "analysis/c104_corrected_intervention_summary.jsonl"
    core.write_rows(summary_path, summaries)
    rollup = []
    for family in sorted({row["family"] for row in summaries}):
        selected = [row for row in summaries if row["family"] == family]
        rollup.append({
            "family": family,
            "controlled_cells": sum(row["correct_positive"] and row["correct_beats_all_controls"] for row in selected),
            "total_cells": len(selected),
            "all_partition_code_cells_controlled": all(row["correct_positive"] and row["correct_beats_all_controls"] for row in selected),
        })
    rollup_path = OUT / "analysis/c104_corrected_family_rollup.jsonl"
    core.write_rows(rollup_path, rollup)
    return {
        "source_sha256": core.sha(source_path), "results_sha256": core.sha(path), "summary_sha256": core.sha(summary_path),
        "rollup_sha256": core.sha(rollup_path), "rows": len(corrected), "summary_rows": len(summaries),
        "fully_controlled_families": [row["family"] for row in rollup if row["all_partition_code_cells_controlled"]],
        "partially_controlled": {row["family"]: row["controlled_cells"] for row in rollup},
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C105 already exists: {OUT}")
    candidate_order = verify_candidate_order()
    c102 = correct_c102()
    c104 = correct_c104()
    checks = {
        "candidate_order": candidate_order["normalized"] == ["yes", "no"],
        "c102_rows": c102["rows"] == 384 and c102["summary_rows"] == 16,
        "c104_rows": c104["rows"] == 192 and c104["summary_rows"] == 16,
        "finite": all(math.isfinite(value) for path in (OUT / "analysis/c102_corrected_intervention_summary.jsonl", OUT / "analysis/c104_corrected_intervention_summary.jsonl") for row in core.rows(path) for dictionary in [row.get("median_target_gain_corrected", row.get("median_true_direction_gain_corrected"))] for value in dictionary.values()),
        "sign_identity": all(abs(new["recipient_yes_minus_no_corrected"] + old["recipient_yes_margin"]) < 1e-12 for new, old in zip(core.rows(OUT / "analysis/c104_corrected_intervention_results.jsonl"), core.rows(C104 / "analysis/upstream_role_intervention_results.jsonl"), strict=True)),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "candidate_order_readout_semantics_corrected_and_both_interventions_readjudicated",
        "candidate_order": candidate_order,
        "root_cause": "both source intervention scripts assigned candidate[0] to No and candidate[1] to Yes, while the frozen compiler emits [Yes, No]",
        "correction": "target-oriented margins and gains are exactly negated; patches and model forwards are unchanged",
        "c102": c102,
        "c104": c104,
        "checks": checks,
        "theory_update": "predictive full-coordinate barcodes and causal whole-role-state sufficiency now coincide for a subset of families; they remain conditional task-response mechanisms rather than semantic neurons",
        "claim_boundary": "deterministic readout correction only; no new model run, no threshold change, no hidden-state reselection",
        "authorization": "export_corrected_c104_heatmap_and_close_c102_c104_c105_stage",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
