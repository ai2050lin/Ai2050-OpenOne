"""Independent audit of the Phase1183 instrument-gate stop."""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1183_gauge_exact_prospective_mechanism_closure as phase  # noqa: E402
import phase1183_instrument_gate_stop_finalize as stop  # noqa: E402


AUDIT_PATH = phase.OUT_ROOT / "audit/instrument_gate_stop_audit.json"


def main() -> None:
    if AUDIT_PATH.exists():
        raise RuntimeError("stop audit already exists")
    protocol = phase.read_json(phase.PROTOCOL_PATH)
    stored_protocol_digest = protocol.pop("protocol_digest")
    preflight = phase.read_json(phase.PREFLIGHT_PATH)
    final = phase.read_json(stop.FINAL_STOP)
    stored_final_digest = final.pop("final_stop_digest")
    valid = [row for row in preflight["rows"] if row["kind"] != "leak_positive_sentinel"]
    sentinels = [row for row in preflight["rows"] if row["kind"] == "leak_positive_sentinel"]
    thresholds = protocol["thresholds"]
    feature_max = max(row["feature_error"] for row in valid)
    fp64_max = max(row["fp64_logit_error"] for row in valid)
    fp32_max = max(row["fp32_logit_error"] for row in valid)
    sentinel_min = min(row["feature_error"] for row in sentinels)
    recomputed_pass = bool(
        feature_max <= thresholds["instrument_feature_max_error_max"]
        and fp64_max <= thresholds["instrument_fp64_logit_max_error_max"]
        and fp32_max <= thresholds["instrument_fp32_logit_max_error_max"]
        and sentinel_min >= thresholds["instrument_positive_sentinel_error_min"]
    )
    discovery = phase.OUT_ROOT / "runs/discovery"
    confirmation = phase.OUT_ROOT / "runs/confirmation"
    checks = {
        "protocol_digest": phase.digest(protocol) == stored_protocol_digest,
        "frozen_runner_hash": phase.file_sha256(phase.SCRIPT) == protocol["scripts"]["runner"],
        "frozen_audit_hash": phase.file_sha256(phase.AUDIT_SCRIPT) == protocol["scripts"]["audit"],
        "preflight_summary_digest": phase.digest({key: value for key, value in preflight.items() if key != "summary_digest"}) == preflight["summary_digest"],
        "preflight_case_count": preflight["case_count"] == 30 and len(preflight["rows"]) == 30,
        "signed_permutation_count": sum(row["kind"] == "signed_permutation" for row in preflight["rows"]) == 24,
        "batch_reverse_count": sum(row["kind"] == "batch_reverse" for row in preflight["rows"]) == 3,
        "positive_sentinel_count": len(sentinels) == 3,
        "feature_aggregate": math.isclose(feature_max, preflight["feature_max_error"], rel_tol=0, abs_tol=1e-30),
        "fp64_aggregate": math.isclose(fp64_max, preflight["fp64_logit_max_error"], rel_tol=0, abs_tol=1e-30),
        "fp32_aggregate": math.isclose(fp32_max, preflight["fp32_logit_max_error"], rel_tol=0, abs_tol=1e-30),
        "sentinel_aggregate": math.isclose(sentinel_min, preflight["positive_sentinel_min_error"], rel_tol=0, abs_tol=1e-30),
        "feature_gate_passed": feature_max <= thresholds["instrument_feature_max_error_max"],
        "fp64_gate_failed": fp64_max > thresholds["instrument_fp64_logit_max_error_max"],
        "fp32_gate_failed": fp32_max > thresholds["instrument_fp32_logit_max_error_max"],
        "sentinel_gate_passed": sentinel_min >= thresholds["instrument_positive_sentinel_error_min"],
        "overall_preflight_failed": preflight["preflight_pass"] is False and recomputed_pass is False,
        "discovery_training_absent": not (discovery / "training_seal.json").exists(),
        "discovery_scan_absent": not (discovery / "systems.jsonl").exists() and not (discovery / "summary.json").exists(),
        "confirmation_training_absent": not (confirmation / "training_seal.json").exists(),
        "confirmation_scan_absent": not (confirmation / "systems.jsonl").exists() and not (confirmation / "summary.json").exists(),
        "camera_absent": not phase.CAMERA_NPZ.exists() and not phase.CAMERA_META.exists(),
        "ordinary_final_absent": not phase.FINAL_PATH.exists(),
        "final_stop_digest": phase.digest(final) == stored_final_digest,
        "primary_false": final["primary_pass"] is False,
        "registry_closed": final["registry"] == "closed_after_preregistered_instrument_gate",
        "auto_continue_false": final["auto_continue"]["authorized"] is False,
        "not_tested_preserved": all(
            final["component_status"][key] == "not_tested"
            for key in ("fresh_response_material", "endpoint_prediction", "prefix_prediction", "donor_rescue", "confirmation")
        ),
    }
    result = {
        "phase": phase.PHASE,
        "created_at_utc": phase.utc_now(),
        "protocol_digest": stored_protocol_digest,
        "check_count": len(checks),
        "pass_count": sum(checks.values()),
        "checks": checks,
        "audit_pass": all(checks.values()),
        "primary_pass": False,
    }
    result["audit_digest"] = phase.digest(result)
    phase.write_json(AUDIT_PATH, result)
    print(phase.canonical_json(result))


if __name__ == "__main__":
    main()
