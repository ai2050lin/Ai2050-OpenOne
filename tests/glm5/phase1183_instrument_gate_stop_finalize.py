"""Finalize the preregistered Phase1183 instrument-gate stop.

This helper does not alter the frozen runner, audit, thresholds, or preflight.
It records that every downstream scientific panel remained unconsumed.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1183_gauge_exact_prospective_mechanism_closure as phase  # noqa: E402


FINAL_STOP = phase.OUT_ROOT / "analysis/final_stop.json"


def main() -> None:
    if FINAL_STOP.exists() or phase.FINAL_PATH.exists():
        raise RuntimeError("Phase1183 already has a final decision")
    protocol = phase.validate_protocol(require_preflight=False)
    preflight = phase.read_json(phase.PREFLIGHT_PATH)
    discovery_root = phase.OUT_ROOT / "runs/discovery"
    confirmation_root = phase.OUT_ROOT / "runs/confirmation"
    downstream_absent = {
        "discovery_training_seal": not (discovery_root / "training_seal.json").exists(),
        "discovery_systems": not (discovery_root / "systems.jsonl").exists(),
        "discovery_summary": not (discovery_root / "summary.json").exists(),
        "confirmation_training_seal": not (confirmation_root / "training_seal.json").exists(),
        "confirmation_systems": not (confirmation_root / "systems.jsonl").exists(),
        "confirmation_summary": not (confirmation_root / "summary.json").exists(),
        "camera_seal": not phase.CAMERA_NPZ.exists(),
    }
    if preflight["preflight_pass"] or not all(downstream_absent.values()):
        raise RuntimeError("instrument stop conditions are not satisfied")
    final = {
        "phase": phase.PHASE,
        "created_at_utc": phase.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_status": "instrument_preflight_stop_all_network_panels_unread",
        "primary_pass": False,
        "component_status": {
            "algebraic_feature_gauge": "passed_frozen_threshold",
            "fp64_function_gauge": "failed_frozen_absolute_threshold",
            "fp32_function_gauge": "failed_frozen_absolute_threshold",
            "positive_leak_sentinel": "passed",
            "fresh_response_material": "not_tested",
            "endpoint_prediction": "not_tested",
            "prefix_prediction": "not_tested",
            "donor_rescue": "not_tested",
            "confirmation": "not_tested",
        },
        "preflight_summary": {
            key: preflight[key]
            for key in (
                "case_count",
                "feature_max_error",
                "fp64_logit_max_error",
                "fp32_logit_max_error",
                "positive_sentinel_min_error",
                "preflight_pass",
                "summary_digest",
            )
        },
        "downstream_absent": downstream_absent,
        "interpretation": (
            "The algebraic feature map satisfied the declared signed-permutation identity, but the complete "
            "preflight also required absolute functional replay accuracy under an extreme high-scale duplicated-"
            "channel stress case. That gate failed. This does not refute the camera candidate; it denies formal "
            "training and confirmation in this registry."
        ),
        "registry": "closed_after_preregistered_instrument_gate",
        "auto_continue": {
            "authorized": False,
            "reason": (
                "The one-shot Gauge-Exact registry failed before scientific data collection. Do not replace "
                "absolute with relative error, delete the stress case, or launch a repair phase automatically."
            ),
        },
    }
    final["final_stop_digest"] = phase.digest(final)
    phase.write_json(FINAL_STOP, final)
    print(phase.canonical_json(final))


if __name__ == "__main__":
    main()
