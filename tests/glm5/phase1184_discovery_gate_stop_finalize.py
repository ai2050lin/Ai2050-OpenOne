#!/usr/bin/env python3
"""Finalize the preregistered Phase1184 stop after discovery behavior failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1184_numerical_domain_qualification as phase  # noqa: E402


SCRIPT = Path(__file__).resolve()
STOP_AUDIT_SCRIPT = ROOT / "tests/glm5/phase1184_discovery_gate_stop_audit.py"
FINAL_STOP_PATH = phase.OUT_ROOT / "analysis/final_stop.json"


def main() -> None:
    if FINAL_STOP_PATH.exists() or phase.FINAL_PATH.exists():
        raise RuntimeError("Phase1184 already finalized")
    if not STOP_AUDIT_SCRIPT.exists():
        raise RuntimeError("stop audit script must exist before finalization")
    protocol = phase.validate_protocol()
    domain = phase.read_json(phase.DOMAIN_PATH)
    domain_copy = dict(domain)
    stored_domain_digest = domain_copy.pop("domain_seal_digest")
    if phase.digest(domain_copy) != stored_domain_digest:
        raise RuntimeError("domain seal digest mismatch")
    rows_path = phase.OUT_ROOT / "runs/discovery/systems.jsonl"
    if phase.file_sha256(rows_path) != domain["rows_sha256"]:
        raise RuntimeError("discovery systems changed")
    rows = phase.read_jsonl(rows_path)
    qualified = [row for row in rows if row["qualified"]]
    passing_tasks = sum(
        sum(row["qualified"] for row in rows if row["task_name"] == task.name)
        >= phase.THRESHOLDS["qualified_system_count_per_task_min"]
        for task in phase.split_tasks("discovery")
    )
    if len(qualified) != domain["qualified_system_count"] or passing_tasks != domain["passing_task_count"]:
        raise RuntimeError("discovery aggregate mismatch")
    if domain["behavior_pass"] or domain["domain_seal_pass"] or domain["bounds"]:
        raise RuntimeError("stop finalizer is only valid for an empty failed domain seal")
    downstream = {
        "confirmation_training_seal": phase.OUT_ROOT / "runs/confirmation/training_seal.json",
        "confirmation_training_metrics": phase.OUT_ROOT / "runs/confirmation/training_metrics.jsonl",
        "confirmation_systems": phase.OUT_ROOT / "runs/confirmation/systems.jsonl",
        "gauge_rows": phase.OUT_ROOT / "analysis/gauge_rows.jsonl",
        "positive_control_rows": phase.OUT_ROOT / "analysis/positive_control_rows.jsonl",
        "ordinary_final": phase.FINAL_PATH,
        "ordinary_audit": phase.OUT_ROOT / "audit/independent_audit.json",
    }
    downstream_absent = {name: not path.exists() for name, path in downstream.items()}
    if not all(downstream_absent.values()):
        raise RuntimeError("downstream artifact exists after failed discovery gate")
    final = {
        "phase": phase.PHASE,
        "created_at_utc": phase.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_status": "discovery_behavior_gate_stop_all_numerical_confirmation_panels_unread",
        "primary_pass": False,
        "component_status": {
            "fresh_discovery_training": "completed_and_sealed",
            "discovery_behavior_qualification": "failed",
            "natural_support_bounds": "not_defined_after_behavior_failure",
            "safety_domain": "not_defined_after_behavior_failure",
            "fresh_confirmation_training": "not_tested",
            "finite_precision_gauge_replay": "not_tested",
            "broken_compensation_positive_control": "not_tested",
            "restricted_backward_witness": "not_tested",
            "adversarial_stress_tier": "not_tested",
            "mechanism_camera": "not_tested_by_design",
        },
        "discovery_summary": {
            "system_count": len(rows),
            "qualified_system_count": len(qualified),
            "passing_task_count": passing_tasks,
            "required_passing_task_count": phase.THRESHOLDS["passing_task_count_per_split_min"],
            "required_qualified_system_count": phase.THRESHOLDS["qualified_system_count_per_split_min"],
            "task_summaries": domain["task_summaries"],
            "domain_seal_digest": stored_domain_digest,
            "rows_sha256": domain["rows_sha256"],
        },
        "downstream_absent": downstream_absent,
        "interpretation": (
            "The independent numerical-domain registry stopped before defining a numerical domain. "
            "All discovery systems fit training perfectly, but the diverse task panel did not provide "
            "enough holdout-qualified networks. This is a behavior/generalization gate failure; it is "
            "not a negative test of finite-precision gauge replay, K165, or a mechanism camera."
        ),
        "phase1183_status": "unchanged_frozen_failure",
        "registry": "closed_after_preregistered_discovery_behavior_gate",
        "auto_continue": {
            "authorized": False,
            "reason": (
                "Phase1184 did not seal a numerical applicability domain. Do not train confirmation, "
                "replace tasks, lower behavior thresholds, or launch mechanism confirmation automatically."
            ),
        },
        "scripts": {
            "stop_finalizer": phase.file_sha256(SCRIPT),
            "stop_audit": phase.file_sha256(STOP_AUDIT_SCRIPT),
        },
    }
    final["final_stop_digest"] = phase.digest(final)
    phase.write_json(FINAL_STOP_PATH, final)
    print(phase.canonical_json(final))


if __name__ == "__main__":
    main()
