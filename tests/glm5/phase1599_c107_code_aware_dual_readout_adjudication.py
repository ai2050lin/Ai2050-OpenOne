#!/usr/bin/env python3
"""Phase1599 / C107: separate truth-direction transport from code-aligned task rescue."""
from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
C105 = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
C106 = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
OUT = TESTS / "result/phase1599_c107_code_aware_dual_readout_adjudication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE = 1599
CAMPAIGN = "C107"
MODES = ["correct_role_state", "sign_reversed", "same_truth_role_state", "coordinate_permuted_correct"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def enrich_entry(*, code: int, recipient: float, donor: float, margin: float, gain: float) -> dict:
    raw_recovery = safe_ratio(margin - recipient, donor - recipient)
    return {
        "yes_minus_no": margin,
        "truth_direction_gain": gain,
        "code_aligned_task_gain": code * gain,
        "truth_target_correct": margin > 0.0,
        "task_target_correct": code * margin > 0.0,
        "truth_target_flip": recipient <= 0.0 < margin,
        "task_target_flip": code * recipient <= 0.0 < code * margin,
        "donor_truth_target_correct": donor > 0.0,
        "donor_task_target_correct": code * donor > 0.0,
        "donor_trajectory_recovery_ratio": raw_recovery,
        "task_recovery_ratio_valid": code * donor > 0.0 and raw_recovery is not None,
    }


def enrich_c104() -> tuple[list[dict], list[dict], list[dict]]:
    source = core.rows(C105 / "analysis/c104_corrected_intervention_results.jsonl")
    rows = []
    for row in source:
        recipient = row["recipient_yes_minus_no_corrected"]
        donor = row["donor_yes_minus_no_corrected"]
        modes = {
            mode: enrich_entry(
                code=row["code"], recipient=recipient, donor=donor,
                margin=row["modes"][mode]["yes_minus_no_corrected"],
                gain=row["modes"][mode]["true_direction_gain_corrected"],
            )
            for mode in MODES
        }
        rows.append({
            **{key: value for key, value in row.items() if key not in {"modes", "correction"}},
            "recipient_task_margin": row["code"] * recipient,
            "donor_task_margin": row["code"] * donor,
            "modes": modes,
        })
    summaries = summarize(rows, nested_k=None)
    rollup = rollup_families(summaries, nested=False)
    return rows, summaries, rollup


def enrich_c106() -> tuple[list[dict], list[dict], list[dict]]:
    source = core.rows(C106 / "analysis/nested_coordinate_intervention_results.jsonl")
    rows = []
    for row in source:
        recipient = row["recipient_yes_minus_no"]
        donor = row["donor_yes_minus_no"]
        nested = {}
        for k, entries in row["nested"].items():
            nested[k] = {
                mode: enrich_entry(
                    code=row["code"], recipient=recipient, donor=donor,
                    margin=entries[mode]["yes_minus_no"], gain=entries[mode]["true_direction_gain"],
                )
                for mode in MODES
            }
        rows.append({
            **{key: value for key, value in row.items() if key != "nested"},
            "recipient_task_margin": row["code"] * recipient,
            "donor_task_margin": row["code"] * donor,
            "nested": nested,
        })
    summaries = []
    for k in sorted({int(k) for row in rows for k in row["nested"]}):
        summaries.extend(summarize(rows, nested_k=k))
    rollup = rollup_families(summaries, nested=True)
    return rows, summaries, rollup


def summarize(rows: list[dict], nested_k: int | None) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["partition"], row["code"])].append(row)
    summaries = []
    for (family, partition, code), selected in sorted(grouped.items()):
        def entry(row: dict, mode: str) -> dict:
            return row["modes"][mode] if nested_k is None else row["nested"][str(nested_k)][mode]

        raw = {mode: float(np.median([entry(row, mode)["truth_direction_gain"] for row in selected])) for mode in MODES}
        aligned = {mode: float(np.median([entry(row, mode)["code_aligned_task_gain"] for row in selected])) for mode in MODES}
        correct = [entry(row, "correct_role_state") for row in selected]
        raw_controlled = raw["correct_role_state"] > 0.0 and all(raw["correct_role_state"] > raw[mode] for mode in MODES[1:])
        task_controlled = aligned["correct_role_state"] > 0.0 and all(aligned["correct_role_state"] > aligned[mode] for mode in MODES[1:])
        ratios = [item["donor_trajectory_recovery_ratio"] for item in correct if item["donor_trajectory_recovery_ratio"] is not None]
        valid_task_ratios = [item["donor_trajectory_recovery_ratio"] for item in correct if item["task_recovery_ratio_valid"]]
        summaries.append({
            "family": family,
            "partition": partition,
            "code": code,
            "codebook": selected[0]["codebook"],
            "role": selected[0]["role"],
            "state": selected[0]["state"],
            "k": nested_k,
            "pairs": len(selected),
            "independent_units": len({row["unit_id"] for row in selected}),
            "median_truth_direction_gain": raw,
            "median_code_aligned_task_gain": aligned,
            "truth_direction_controlled": raw_controlled,
            "code_aligned_task_controlled": task_controlled,
            "patched_truth_target_accuracy": float(np.mean([item["truth_target_correct"] for item in correct])),
            "patched_task_target_accuracy": float(np.mean([item["task_target_correct"] for item in correct])),
            "truth_target_flip_rate": float(np.mean([item["truth_target_flip"] for item in correct])),
            "task_target_flip_rate": float(np.mean([item["task_target_flip"] for item in correct])),
            "donor_truth_target_accuracy": float(np.mean([item["donor_truth_target_correct"] for item in correct])),
            "donor_task_target_accuracy": float(np.mean([item["donor_task_target_correct"] for item in correct])),
            "median_donor_trajectory_recovery_ratio": float(np.median(ratios)) if ratios else None,
            "task_recovery_ratio_valid_pairs": len(valid_task_ratios),
            "median_task_recovery_ratio_when_donor_valid": float(np.median(valid_task_ratios)) if valid_task_ratios else None,
        })
    return summaries


def rollup_families(summaries: list[dict], *, nested: bool) -> list[dict]:
    result = []
    for family in sorted({row["family"] for row in summaries}):
        family_rows = [row for row in summaries if row["family"] == family]
        ks = sorted({row["k"] for row in family_rows}) if nested else [None]
        curves = []
        for k in ks:
            selected = [row for row in family_rows if row["k"] == k]
            curves.append({
                "k": k,
                "truth_direction_controlled_cells": sum(row["truth_direction_controlled"] for row in selected),
                "code_aligned_task_controlled_cells": sum(row["code_aligned_task_controlled"] for row in selected),
                "total_cells": len(selected),
                "mean_patched_truth_target_accuracy": float(np.mean([row["patched_truth_target_accuracy"] for row in selected])),
                "mean_patched_task_target_accuracy": float(np.mean([row["patched_task_target_accuracy"] for row in selected])),
                "mean_truth_target_flip_rate": float(np.mean([row["truth_target_flip_rate"] for row in selected])),
                "mean_task_target_flip_rate": float(np.mean([row["task_target_flip_rate"] for row in selected])),
            })
        result.append({
            "family": family,
            "curve": curves,
            "first_tested_all_four_truth_direction_k": next((row["k"] for row in curves if row["truth_direction_controlled_cells"] == row["total_cells"] == 4), None),
            "first_tested_all_four_code_aligned_task_k": next((row["k"] for row in curves if row["code_aligned_task_controlled_cells"] == row["total_cells"] == 4), None),
        })
    return result


def export_heatmap(c104_summary: list[dict], c106_summary: list[dict], c106_rollup: list[dict]) -> dict:
    canonical = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    payload = core.load(canonical)
    payload["phase"] = PHASE
    payload["campaign"] = "C104-C107"
    payload["title"] = "C104-C107 Upstream Truth-Response and Code-Aligned Task Field"
    payload["code_aware_adjudication"] = {
        "formula": "task_gain = code * (patched_yes_minus_no - recipient_yes_minus_no)",
        "c104_whole_role_rows": c104_summary,
        "c106_nested_rows": c106_summary,
        "c106_family_rollup": c106_rollup,
    }
    payload["headline"]["legacy_minimal_k_retracted"] = True
    payload["headline"]["raw_first_tested_all_four_k"] = {
        row["family"]: row["first_tested_all_four_truth_direction_k"] for row in c106_rollup
    }
    payload["headline"]["task_aligned_all_four_k"] = {
        row["family"]: row["first_tested_all_four_code_aligned_task_k"] for row in c106_rollup
    }
    payload["claim_boundary"] = (
        "C104/C106 establish controlled code-invariant truth-direction effects for attribute binding and agent-patient, "
        "but no tested K controls all four code-aligned task cells. The 128/256 values are first tested raw-response "
        "scales selected on reused confirmation/lockbox data, not minimal, necessary, independently confirmed, or "
        "functionally sufficient coordinate coalitions. K=1024 non-monotonicity concerns a relative-to-control criterion "
        "and does not identify inhibitory coordinates."
    )
    payload["source"]["c107_final_sha256"] = "filled_after_final"
    payload["source"]["independent_audits"].append("C107 code-aware dual-readout claim audit")
    payload["created_at_utc"] = now()
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    return {"canonical": str(canonical.relative_to(ROOT)), "public": str(PUBLIC.relative_to(ROOT)), "sha256": core.sha(canonical), "bytes": canonical.stat().st_size}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C107 already exists: {OUT}")
    OUT.mkdir(parents=True)
    candidate = core.load(C105 / "analysis/final.json")["candidate_order"]
    c104_rows, c104_summary, c104_rollup = enrich_c104()
    c106_rows, c106_summary, c106_rollup = enrich_c106()
    core.write_rows(OUT / "analysis/c104_code_aware_results.jsonl", c104_rows)
    core.write_rows(OUT / "analysis/c104_code_aware_summary.jsonl", c104_summary)
    core.write_rows(OUT / "analysis/c104_code_aware_family_rollup.jsonl", c104_rollup)
    core.write_rows(OUT / "analysis/c106_code_aware_results.jsonl", c106_rows)
    core.write_rows(OUT / "analysis/c106_code_aware_summary.jsonl", c106_summary)
    core.write_rows(OUT / "analysis/c106_code_aware_family_rollup.jsonl", c106_rollup)

    c104_truth = {row["family"]: row["curve"][0]["truth_direction_controlled_cells"] for row in c104_rollup}
    c104_task = {row["family"]: row["curve"][0]["code_aligned_task_controlled_cells"] for row in c104_rollup}
    first_raw = {row["family"]: row["first_tested_all_four_truth_direction_k"] for row in c106_rollup}
    first_task = {row["family"]: row["first_tested_all_four_code_aligned_task_k"] for row in c106_rollup}
    checks = {
        "candidate_order": candidate["normalized"] == ["yes", "no"],
        "c104_rows": len(c104_rows) == 192 and len(c104_summary) == 16,
        "c106_rows": len(c106_rows) == 96 and len(c106_summary) == 80,
        "exact_code_identity": all(
            abs(entry["code_aligned_task_gain"] - row["code"] * entry["truth_direction_gain"]) < 1e-12
            for row in c106_rows for nested in row["nested"].values() for entry in nested.values()
        ),
        "finite": all(
            math.isfinite(value)
            for row in c106_summary
            for dictionary in (row["median_truth_direction_gain"], row["median_code_aligned_task_gain"])
            for value in dictionary.values()
        ),
        "raw_reproduction": first_raw == {"agent_patient": 128, "attribute_binding": 256},
        "task_boundary": first_task == {"agent_patient": None, "attribute_binding": None},
        "whole_role_reclassification": c104_truth["agent_patient"] == c104_truth["attribute_binding"] == 4 and c104_task["agent_patient"] == c104_task["attribute_binding"] == 2,
        "independence_boundary": all(row["independent_units"] == 3 for row in c106_summary),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "code_aware_dual_readout_adjudication_complete",
        "inputs": {
            "c104_corrected_results_sha256": core.sha(C105 / "analysis/c104_corrected_intervention_results.jsonl"),
            "c106_results_sha256": core.sha(C106 / "analysis/nested_coordinate_intervention_results.jsonl"),
        },
        "candidate_order": candidate,
        "formula": "aligned task gain = code * raw Yes-minus-No gain; code=+1 standard, code=-1 reversed",
        "c104_whole_role_cells": {"truth_direction": c104_truth, "code_aligned_task": c104_task},
        "c106_first_tested_k": {"truth_direction": first_raw, "code_aligned_task": first_task},
        "claim_adjudication": {
            "K276": "retained: prospective full-vector upstream barcode replication is unaffected",
            "K277-R1": "whole-role transport has a controlled code-invariant truth-direction effect in attribute binding and agent-patient; it does not close code-aligned task rescue across reversed code",
            "K279-R1": "128/256 are first tested controlled raw truth-response scales on reused confirmation/lockbox data; minimality, necessity, fresh confirmation, and functional sufficiency are retracted",
            "K280-R1": "the relative-to-control criterion is non-monotonic for agent-patient; inhibitory coordinates are not identified",
        },
        "checks": checks,
        "claim_boundary": "deterministic reanalysis of frozen logits only; no model run, threshold change, coordinate reselection, or post-hoc functional gate",
        "next_authorization": "freeze the C106 supports as discovery candidates and test raw truth transport, task-aligned rescue, target flips, necessity, and deletion-rescue on fresh lexical units without K reselection",
    }
    core.save(OUT / "analysis/final.json", final)
    heatmap = export_heatmap(c104_summary, c106_summary, c106_rollup)
    payload = core.load(C104 / "visualization/c104_upstream_role_barcode_heatmap.json")
    payload["source"]["c107_final_sha256"] = core.sha(OUT / "analysis/final.json")
    core.save(C104 / "visualization/c104_upstream_role_barcode_heatmap.json", payload)
    shutil.copyfile(C104 / "visualization/c104_upstream_role_barcode_heatmap.json", PUBLIC)
    heatmap.update({"sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size})
    final["heatmap"] = heatmap
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
