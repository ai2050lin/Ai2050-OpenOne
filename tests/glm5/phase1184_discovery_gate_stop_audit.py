#!/usr/bin/env python3
"""Independent stop audit for Phase1184 discovery-gate termination."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1184_numerical_domain_qualification as phase  # noqa: E402
import phase1184_discovery_gate_stop_finalize as stop  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_PATH = phase.OUT_ROOT / "audit/stop_audit.json"


def check(items: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    items.append({"name": name, "pass": bool(passed), "detail": detail})


def main() -> None:
    protocol = phase.read_json(phase.PROTOCOL_PATH)
    domain = phase.read_json(phase.DOMAIN_PATH)
    final = phase.read_json(stop.FINAL_STOP_PATH)
    rows_path = phase.OUT_ROOT / "runs/discovery/systems.jsonl"
    rows = phase.read_jsonl(rows_path)
    items: list[dict[str, Any]] = []

    protocol_copy = dict(protocol)
    protocol_digest = protocol_copy.pop("protocol_digest")
    check(items, "protocol_digest", phase.digest(protocol_copy) == protocol_digest)
    check(items, "protocol_phase", protocol["phase"] == phase.PHASE)
    check(items, "frozen_runner", phase.file_sha256(phase.SCRIPT) == protocol["scripts"]["runner"])
    check(items, "frozen_ordinary_audit", phase.file_sha256(phase.AUDIT_SCRIPT) == protocol["scripts"]["audit"])
    check(items, "frozen_phase1183", phase.file_sha256(Path(phase.p1183.__file__)) == protocol["scripts"]["phase1183_source"])
    check(items, "stop_finalizer_hash", phase.file_sha256(stop.SCRIPT) == final["scripts"]["stop_finalizer"])
    check(items, "stop_audit_hash", phase.file_sha256(SCRIPT) == final["scripts"]["stop_audit"])

    domain_copy = dict(domain)
    domain_digest = domain_copy.pop("domain_seal_digest")
    check(items, "domain_digest", phase.digest(domain_copy) == domain_digest)
    check(items, "domain_rows_hash", phase.file_sha256(rows_path) == domain["rows_sha256"])
    check(items, "discovery_system_count", len(rows) == 32 == domain["system_count"])
    qualified = [row for row in rows if row["qualified"]]
    check(items, "qualified_count", len(qualified) == 9 == domain["qualified_system_count"])
    passing_tasks = sum(
        sum(row["qualified"] for row in rows if row["task_name"] == task.name)
        >= phase.THRESHOLDS["qualified_system_count_per_task_min"]
        for task in phase.split_tasks("discovery")
    )
    check(items, "passing_task_count", passing_tasks == 1 == domain["passing_task_count"])
    check(items, "behavior_failure_recompute", not (
        passing_tasks >= phase.THRESHOLDS["passing_task_count_per_split_min"]
        and len(qualified) >= phase.THRESHOLDS["qualified_system_count_per_split_min"]
    ))
    check(items, "domain_behavior_false", domain["behavior_pass"] is False)
    check(items, "domain_seal_false", domain["domain_seal_pass"] is False)
    check(items, "domain_bounds_empty", domain["bounds"] == {})
    check(items, "all_training_fit", all(row["train_accuracy"] >= 0.999 for row in rows))
    task_counts = {
        task.name: sum(row["qualified"] for row in rows if row["task_name"] == task.name)
        for task in phase.split_tasks("discovery")
    }
    check(items, "task_qualified_counts", task_counts == {
        "domain_affine_a": 0,
        "domain_product_a": 1,
        "domain_left_square_a": 0,
        "domain_xor_a": 8,
    }, task_counts)

    final_copy = dict(final)
    final_digest = final_copy.pop("final_stop_digest")
    check(items, "final_stop_digest", phase.digest(final_copy) == final_digest)
    check(items, "final_protocol_link", final["protocol_digest"] == protocol_digest)
    check(items, "final_domain_link", final["discovery_summary"]["domain_seal_digest"] == domain_digest)
    check(items, "primary_false", final["primary_pass"] is False)
    check(items, "confirmation_not_tested", final["component_status"]["fresh_confirmation_training"] == "not_tested")
    check(items, "gauge_not_tested", final["component_status"]["finite_precision_gauge_replay"] == "not_tested")
    check(items, "camera_not_tested", final["component_status"]["mechanism_camera"] == "not_tested_by_design")
    check(items, "phase1183_unchanged", final["phase1183_status"] == "unchanged_frozen_failure")
    check(items, "all_downstream_absent", all(final["downstream_absent"].values()))
    check(items, "auto_continue_false", final["auto_continue"]["authorized"] is False)
    check(items, "registry_closed", final["registry"] == "closed_after_preregistered_discovery_behavior_gate")

    passed = all(item["pass"] for item in items)
    result = {
        "phase": phase.PHASE,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol_digest,
        "final_stop_digest": final_digest,
        "check_count": len(items),
        "passed_check_count": sum(int(item["pass"]) for item in items),
        "audit_pass": passed,
        "checks": items,
    }
    result["audit_digest"] = phase.digest(result)
    phase.write_json(AUDIT_PATH, result)
    print(phase.canonical_json(result))


if __name__ == "__main__":
    main()
