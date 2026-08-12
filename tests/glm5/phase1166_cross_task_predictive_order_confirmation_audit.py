#!/usr/bin/env python3
"""Independent artifact audit for Phase1166."""

from __future__ import annotations

import hashlib
import json
import py_compile
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PRIMARY = ROOT / "tests/glm5/phase1166_cross_task_predictive_order_confirmation.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1166_cross_task_predictive_order_confirmation"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1166_cross_task_predictive_order_confirmation as phase  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def main() -> None:
    py_compile.compile(str(PRIMARY), doraise=True)
    py_compile.compile(str(SCRIPT), doraise=True)
    protocol = phase.p1163.read_json(OUT_ROOT / "protocol/preregistration.json")
    model_summary = phase.p1163.read_json(OUT_ROOT / "runs/models/summary.json")
    calibration_summary = phase.p1163.read_json(OUT_ROOT / "runs/calibration/summary.json")
    selection = phase.p1163.read_json(OUT_ROOT / "predictions/selection.json")
    confirmation_summary = phase.p1163.read_json(OUT_ROOT / "runs/confirmation/summary.json")
    score = phase.p1163.read_json(OUT_ROOT / "analysis/score.json")
    final = phase.p1163.read_json(OUT_ROOT / "analysis/final.json")
    body = dict(protocol)
    protocol_digest = body.pop("protocol_digest")
    recalculated = phase.calculate_score(selection, model_summary)
    discovery, confirmation = phase.schedule_splits()
    checks = {
        "primary_compiles": True,
        "audit_compiles": True,
        "protocol_digest": digest(body) == protocol_digest,
        "primary_hash_frozen": phase.p1163.sha256_file(PRIMARY)
        == protocol["source_hashes"]["primary_script"],
        "audit_hash_frozen": phase.p1163.sha256_file(SCRIPT)
        == protocol["source_hashes"]["audit_script"],
        "phase1165_hash_frozen": phase.p1163.sha256_file(phase.P1165_SCRIPT)
        == protocol["source_hashes"]["phase1165_script"],
        "protocol_checks": all(protocol["checks"].values()),
        "task_structure_checks": all(protocol["task_structure_checks"].values()),
        "model_count": sum(1 for _ in phase.p1163.read_jsonl(OUT_ROOT / "runs/models/model_metrics.jsonl"))
        == len(phase.ALL_TASKS) * len(phase.ARCHITECTURES) * phase.REPLICATES,
        "full_task_behavior_gate": model_summary["full_task_behavior_gate_passed"],
        "confidence_not_hard_gate": protocol["behavior_gate_note"].startswith("accuracy"),
        "calibration_gate": calibration_summary["calibration_gate_passed"],
        "confirmation_gate": confirmation_summary["confirmation_gate_passed"],
        "schedule_disjoint": not bool(set(discovery).intersection(confirmation)),
        "discovery_schedule_frozen": protocol["discovery_schedules"]
        == [list(row) for row in discovery],
        "confirmation_schedule_frozen": protocol["confirmation_schedules"]
        == [list(row) for row in confirmation],
        "selection_precedes_confirmation": selection["created_at_utc"]
        < confirmation_summary["created_at_utc"],
        "selection_link": selection["selection_digest"]
        == confirmation_summary["selection_digest"],
        "confirmation_absent_at_selection": selection["confirmation_outcomes_absent_at_sealing"],
        "prediction_hashes": all(
            phase.p1163.sha256_file(OUT_ROOT / "predictions" / f"{task}.npz")
            == selection["prediction_hashes"][task]
            for task in phase.FULL_TASKS
        ),
        "confirmation_hashes": all(
            phase.p1163.sha256_file(OUT_ROOT / "runs/confirmation" / f"{task}.npz")
            == confirmation_summary["tasks"][task]["pack_sha256"]
            for task in phase.FULL_TASKS
        ),
        "recalculated_decision": recalculated["decision"] == score["results"]["decision"],
        "recalculated_results": recalculated == score["results"],
        "score_digest": digest({key: value for key, value in score.items() if key != "score_digest"})
        == score["score_digest"],
        "final_score_link": final["score_digest"] == score["score_digest"],
        "final_decision_link": final["decision"] == score["results"]["decision"],
        "branch_closed": final["branch_status"] == "closed_after_cross_task_confirmation",
        "natural_mechanism_not_claimed": final["natural_mechanism_recovered"] is False,
        "camera_unchanged": protocol["intervention_semantic"].startswith("matched residual-delta"),
        "finite_calibration": all(
            np.isfinite(np.load(OUT_ROOT / "runs/calibration" / f"{task}.npz")["response"]).all()
            for task in phase.FULL_TASKS
        ),
        "finite_confirmation": all(
            np.isfinite(np.load(OUT_ROOT / "runs/confirmation" / f"{task}.npz")["response"]).all()
            for task in phase.FULL_TASKS
        ),
        "order_library_frozen": protocol["order_library"]
        == {str(key): list(value) for key, value in phase.ORDER_LIBRARY.items()},
        "composition_separate": protocol["checks"]["composition_separate_behavior_axis"],
    }
    audit = {
        "phase": phase.PHASE,
        "created_at_utc": phase.p1163.now(),
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "decision": final["decision"],
        "protocol_digest": protocol["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "score_digest": score["score_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = digest(audit)
    phase.p1163.write_json(OUT_ROOT / "audit/independent_audit.json", audit)
    print(canonical(audit))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
