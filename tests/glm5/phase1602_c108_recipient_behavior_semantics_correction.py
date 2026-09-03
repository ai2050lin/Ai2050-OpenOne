#!/usr/bin/env python3
"""Deterministically correct C108 false-recipient task margins from frozen raw logits."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    path = OUT / "analysis/fresh_coordinate_intervention_results.jsonl"
    rows = core.rows(path)
    old_values = [row["recipient_task_margin"] for row in rows]
    for row in rows:
        row["recipient_task_margin"] = -row["code"] * row["recipient_yes_minus_no"]
        row["recipient_task_margin_semantics"] = "false recipient correct sign = -code"
    core.write_rows(path, rows)
    final = core.load(OUT / "analysis/final.json")
    behavior = {
        "recipient_truth_accuracy": float(np.mean([row["recipient_yes_minus_no"] < 0.0 for row in rows])),
        "donor_truth_accuracy": float(np.mean([row["donor_yes_minus_no"] > 0.0 for row in rows])),
        "recipient_task_accuracy": float(np.mean([row["recipient_task_margin"] > 0.0 for row in rows])),
        "donor_task_accuracy": float(np.mean([row["donor_task_margin"] > 0.0 for row in rows])),
        "all_case_task_accuracy": float(np.mean([row[key] > 0.0 for row in rows for key in ("recipient_task_margin", "donor_task_margin")])),
        "standard_task_accuracy": float(np.mean([row[key] > 0.0 for row in rows if row["code"] == 1 for key in ("recipient_task_margin", "donor_task_margin")])),
        "reversed_task_accuracy": float(np.mean([row[key] > 0.0 for row in rows if row["code"] == -1 for key in ("recipient_task_margin", "donor_task_margin")])),
    }
    checks = {
        "rows": len(rows) == 192,
        "old_formula_identified": all(abs(old - row["code"] * row["recipient_yes_minus_no"]) < 1e-12 for old, row in zip(old_values, rows, strict=True)),
        "new_formula": all(abs(row["recipient_task_margin"] + row["code"] * row["recipient_yes_minus_no"]) < 1e-12 for row in rows),
        "interventions_unchanged": core.sha(OUT / "analysis/fresh_coordinate_intervention_summary.jsonl") == final["summary_sha256"],
        "expected_behavior": abs(behavior["standard_task_accuracy"] - 0.9739583333333334) < 1e-12 and abs(behavior["reversed_task_accuracy"] - 0.036458333333333336) < 1e-12,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final["phase"] = 1602
    final["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    final["status"] = "fresh_coordinate_write_delete_intervention_complete_with_recipient_behavior_semantics_corrected"
    final["results_sha256"] = core.sha(path)
    final["behavior"] = behavior
    final["behavior_correction"] = "recipient is truth=false, so its correct task margin is -code*(Yes-minus-No); donor and all intervention target margins remain +code aligned"
    final["behavior_correction_checks"] = checks
    core.save(OUT / "analysis/final.json", final)
    result = {"phase": 1602, "campaign": "C108", "status": "recipient_behavior_semantics_corrected", "checks": checks, "behavior": behavior, "results_sha256": final["results_sha256"], "claim_boundary": "deterministic correction of derived recipient behavior only; raw logits, interventions, summaries, coordinate supports, and thresholds unchanged"}
    core.save(OUT / "analysis/recipient_behavior_semantics_correction.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
