#!/usr/bin/env python3
"""Deterministic stop audit after the frozen Phase1182 discovery gate failed.

This audit does not alter the preregistration, thresholds, camera, or scientific
decision.  It records that confirmation was correctly denied and separates the
three discovery-only scientific candidates from the failed gauge instrument.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1182_quotient_response_camera_and_rescue as phase  # noqa: E402


AUDIT_PATH = phase.OUT_ROOT / "audit/discovery_gate_stop_audit.json"
FINAL_STOP_PATH = phase.OUT_ROOT / "analysis/final_stop.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    protocol = phase.validate_protocol()
    summary = phase.read_json(phase.DISCOVERY_SUMMARY)
    rows = phase.read_jsonl(phase.DISCOVERY_ROWS)
    rescue = phase.read_json(phase.DISCOVERY_RESCUE)
    checks: list[dict[str, Any]] = []
    add(checks, "discovery_row_count", len(rows) == 64, len(rows))
    add(checks, "discovery_unique_checkpoints", len({row["checkpoint"] for row in rows}) == 64)
    add(checks, "discovery_rows_digest", phase.digest(rows) == summary["rows_digest"])
    summary_without_digest = dict(summary)
    stored_summary_digest = summary_without_digest.pop("summary_digest")
    add(checks, "discovery_summary_digest", phase.digest(summary_without_digest) == stored_summary_digest)
    add(checks, "endpoint_discovery_gate_pass", summary["endpoint"]["gate_pass"] is True)
    add(checks, "prefix_discovery_gate_pass", summary["prefix"]["gate_pass"] is True)
    add(checks, "rescue_discovery_gate_pass", summary["rescue"]["gate_pass"] is True)
    add(checks, "rescue_payload_matches_summary", rescue["summary"] == summary["rescue"])
    add(checks, "gauge_discovery_gate_failed", summary["feature_gauge_pass"] is False)
    add(
        checks,
        "gauge_error_exceeds_frozen_threshold",
        summary["feature_gauge_max_error"] > protocol["thresholds"]["feature_gauge_max_error_max"],
        {
            "observed": summary["feature_gauge_max_error"],
            "threshold": protocol["thresholds"]["feature_gauge_max_error_max"],
        },
    )
    add(checks, "discovery_primary_denied", summary["discovery_pass"] is False)
    add(checks, "confirmation_rows_absent", not phase.CONFIRMATION_ROWS.exists())
    add(checks, "confirmation_summary_absent", not phase.CONFIRMATION_SUMMARY.exists())
    add(checks, "confirmation_rescue_absent", not phase.CONFIRMATION_RESCUE.exists())
    add(checks, "ordinary_final_absent", not phase.FINAL_PATH.exists())
    add(checks, "camera_seal_exists_but_unused", phase.CAMERA_SEAL.exists())
    add(
        checks,
        "camera_seal_hash",
        phase.file_sha256(phase.CAMERA_SEAL) == summary["camera_seal_sha256"],
    )
    audit_pass = all(check["passed"] for check in checks)
    final_stop = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_status": "discovery_instrument_gate_stop_confirmation_untested",
        "primary_pass": False,
        "discovery_component_status": {
            "endpoint_increment": "discovery_only_candidate_pass",
            "prefix_increment": "discovery_only_candidate_pass",
            "donor_future_response_rescue": "discovery_only_candidate_pass",
            "feature_gauge": "failed",
        },
        "confirmation_status": "not_run_by_protocol",
        "mechanism_status": "untested_on_confirmation_not_refuted",
        "discovery_summary": summary,
        "auto_continue": {
            "authorized": False,
            "reason": "Frozen discovery gauge threshold failed; this camera/rescue registry is closed.",
        },
    }
    final_stop["final_stop_digest"] = phase.digest(final_stop)
    phase.write_json(FINAL_STOP_PATH, final_stop)
    audit = {
        "phase": phase.PHASE,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "integrity_and_stop_decision_pass": audit_pass,
        "scientific_primary_pass": False,
        "confirmation_untested": True,
        "check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "checks": checks,
        "final_stop_digest": final_stop["final_stop_digest"],
    }
    audit["audit_digest"] = phase.digest(audit)
    phase.write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
