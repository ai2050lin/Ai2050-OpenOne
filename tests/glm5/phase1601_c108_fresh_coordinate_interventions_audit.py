#!/usr/bin/env python3
"""Independent audit of the Phase1601 / C108 fresh interventions."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

WRITE_CONTROLS = ("wrong_family_support", "sign_reversed", "same_truth", "coordinate_permuted")
DELETE_CONTROLS = ("wrong_family_support", "same_truth")


def main() -> None:
    producer = TESTS / "phase1601_c108_fresh_coordinate_interventions.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/independent_pre_model_audit.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/fresh_coordinate_intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/fresh_coordinate_intervention_summary.jsonl")
    family_rows = core.rows(OUT / "analysis/fresh_coordinate_family_rollup.jsonl")
    recalculated = []
    formula_ok = True
    for row in rows:
        for entry in row["write"].values():
            formula_ok &= abs(entry["code_aligned_task_gain"] - row["code"] * entry["truth_direction_gain"]) < 1e-12
        for entry in row["delete"].values():
            formula_ok &= abs(entry["code_aligned_task_loss"] - row["code"] * entry["truth_direction_loss"]) < 1e-12
    for summary in summaries:
        selected = [row for row in rows if row["family"] == summary["family"] and row["partition"] == summary["partition"] and row["code"] == summary["code"]]
        write = {mode: float(np.median([row["write"][mode]["truth_direction_gain"] for row in selected])) for mode in summary["median_write_truth_direction_gain"]}
        delete = {mode: float(np.median([row["delete"][mode]["truth_direction_loss"] for row in selected])) for mode in summary["median_delete_truth_direction_loss"]}
        recalculated.append({
            "family": summary["family"], "partition": summary["partition"], "code": summary["code"],
            "write": write["frozen_support"] > 0.0 and all(write["frozen_support"] > write[mode] for mode in WRITE_CONTROLS),
            "delete": delete["frozen_support"] > 0.0 and all(delete["frozen_support"] > delete[mode] for mode in DELETE_CONTROLS),
        })
    recomputed_family = {}
    for family in protocol["families"]:
        selected = [row for row in recalculated if row["family"] == family]
        recomputed_family[family] = {"write": sum(row["write"] for row in selected), "delete": sum(row["delete"] for row in selected)}
    reported_family = {row["family"]: {"write": row["truth_direction_write_cells"], "delete": row["truth_direction_delete_cells"]} for row in family_rows}
    behavior = {
        "recipient_truth_accuracy": float(np.mean([row["recipient_yes_minus_no"] < 0.0 for row in rows])),
        "donor_truth_accuracy": float(np.mean([row["donor_yes_minus_no"] > 0.0 for row in rows])),
        "recipient_task_accuracy": float(np.mean([-row["code"] * row["recipient_yes_minus_no"] > 0.0 for row in rows])),
        "donor_task_accuracy": float(np.mean([row["donor_task_margin"] > 0.0 for row in rows])),
        "all_case_task_accuracy": float(np.mean([value > 0.0 for row in rows for value in (-row["code"] * row["recipient_yes_minus_no"], row["code"] * row["donor_yes_minus_no"])])),
        "standard_task_accuracy": float(np.mean([value > 0.0 for row in rows if row["code"] == 1 for value in (-row["code"] * row["recipient_yes_minus_no"], row["code"] * row["donor_yes_minus_no"])])),
        "reversed_task_accuracy": float(np.mean([value > 0.0 for row in rows if row["code"] == -1 for value in (-row["code"] * row["recipient_yes_minus_no"], row["code"] * row["donor_yes_minus_no"])])),
    }
    checks = {
        "producer_compiles": py_compile.compile(str(producer), doraise=True) is not None,
        "pre_model": pre["all_checks_passed"],
        "runtime_checks": all(final["checks"].values()),
        "hashes": final["results_sha256"] == core.sha(OUT / "analysis/fresh_coordinate_intervention_results.jsonl") and final["summary_sha256"] == core.sha(OUT / "analysis/fresh_coordinate_intervention_summary.jsonl"),
        "counts": len(rows) == 192 and len(summaries) == 8 and len(family_rows) == 2,
        "units": len({row["unit_id"] for row in rows}) == 24 and all(row["independent_units"] == 6 for row in summaries),
        "formula": formula_ok,
        "recipient_task_sign": all(abs(row["recipient_task_margin"] + row["code"] * row["recipient_yes_minus_no"]) < 1e-12 for row in rows),
        "rollup": recomputed_family == reported_family == {"attribute_binding": {"write": 4, "delete": 4}, "agent_patient": {"write": 0, "delete": 3}},
        "task_boundary": {row["family"]: (row["code_aligned_task_write_cells"], row["code_aligned_task_delete_cells"]) for row in family_rows} == {"attribute_binding": (2, 2), "agent_patient": (0, 1)},
        "behavior": all(abs(final["behavior"][key] - value) < 1e-12 for key, value in behavior.items()),
        "no_reselection": protocol["frozen_k"] == {"attribute_binding": 256, "agent_patient": 128},
        "authorization": final["authorization"] == "independent_audit_synthesize_and_close_c108",
    }
    result = {"phase": 1601, "campaign": "C108", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "recomputed_family": recomputed_family, "authorization": "run_phase1602_c108_synthesis_heatmap_and_closure"}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_intervention_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
