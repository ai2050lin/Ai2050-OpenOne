from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PHASE = 1125
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1125_pythia_controlled_bridge_calibration"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_digest(value: dict[str, Any], key: str) -> bool:
    body = dict(value)
    expected = body.pop(key)
    return canonical_digest(body) == expected


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    if not verify_digest(prereg, "protocol_digest"):
        raise RuntimeError("Phase1125 protocol digest mismatch")
    base = read_json(OUT_ROOT / "evaluation" / "base.json")["evaluation"]
    behavior = read_json(OUT_ROOT / "training" / "behavior_only" / "summary.json")
    forced = read_json(OUT_ROOT / "training" / "bridge_forced" / "summary.json")
    run = read_json(OUT_ROOT / "training" / "run_summary.json")
    if not verify_digest(behavior, "result_digest") or not verify_digest(forced, "result_digest"):
        raise RuntimeError("Phase1125 arm digest mismatch")
    if not verify_digest(run, "run_digest"):
        raise RuntimeError("Phase1125 run digest mismatch")
    if any(item["protocol_digest"] != prereg["protocol_digest"] for item in (behavior, forced, run)):
        raise RuntimeError("Phase1125 protocol linkage mismatch")

    thresholds = prereg["thresholds"]
    behavior_calibration = behavior["evaluation"]["calibration"]
    forced_calibration = forced["evaluation"]["calibration"]
    forced_transfer = forced["evaluation"]["transfer"]
    base_calibration = base["calibration"]
    base_transfer = base["transfer"]

    engineering_checks = {
        "behavior_arm_finite": behavior["nonfinite_training_steps"]
        <= thresholds["maximum_nonfinite_training_steps"],
        "forced_arm_finite": forced["nonfinite_training_steps"]
        <= thresholds["maximum_nonfinite_training_steps"],
        "behavior_base_gradients_absent": behavior["base_gradients_absent"],
        "forced_base_gradients_absent": forced["base_gradients_absent"],
        "run_base_frozen": run["all_base_parameters_frozen"],
    }
    p1 = all(engineering_checks.values())

    behavior_accuracy_gain = (
        behavior_calibration["candidate_accuracy"] - base_calibration["candidate_accuracy"]
    )
    behavior_gate = {
        "absolute_accuracy_pass": behavior_calibration["candidate_accuracy"]
        >= thresholds["minimum_behavior_only_calibration_accuracy"],
        "accuracy_gain_pass": behavior_accuracy_gain
        >= thresholds["minimum_behavior_only_calibration_accuracy_gain"],
    }
    p2 = all(behavior_gate.values())

    forced_cd_gain_base = (
        forced_calibration["median_cd_cosine_projected"] - base_calibration["median_cd_cosine_projected"]
    )
    forced_cd_gain_behavior = (
        forced_calibration["median_cd_cosine_projected"]
        - behavior_calibration["median_cd_cosine_projected"]
    )
    calibration_gate = {
        "accuracy_pass": forced_calibration["candidate_accuracy"]
        >= thresholds["minimum_forced_calibration_accuracy"],
        "projected_cd_cosine_pass": forced_calibration["median_cd_cosine_projected"]
        >= thresholds["minimum_forced_calibration_projected_cd_cosine"],
        "projected_cd_positive_rate_pass": forced_calibration["projected_cd_positive_rate"]
        >= thresholds["minimum_forced_calibration_projected_cd_positive_rate"],
        "cd_gain_over_base_pass": forced_cd_gain_base
        >= thresholds["minimum_forced_calibration_cd_gain_over_base"],
        "cd_gain_over_behavior_only_pass": forced_cd_gain_behavior
        >= thresholds["minimum_forced_calibration_cd_gain_over_behavior_only"],
        "full_projection_agreement_pass": forced_calibration["median_full_projection_gap"]
        <= thresholds["maximum_full_projection_median_gap"],
    }
    p3 = all(calibration_gate.values())

    forced_transfer_gain = (
        forced_transfer["median_cd_cosine_projected"] - base_transfer["median_cd_cosine_projected"]
    )
    transfer_gate = {
        "accuracy_pass": forced_transfer["candidate_accuracy"]
        >= thresholds["minimum_forced_transfer_accuracy"],
        "projected_cd_cosine_pass": forced_transfer["median_cd_cosine_projected"]
        >= thresholds["minimum_forced_transfer_projected_cd_cosine"],
        "projected_cd_positive_rate_pass": forced_transfer["projected_cd_positive_rate"]
        >= thresholds["minimum_forced_transfer_projected_cd_positive_rate"],
        "cd_gain_over_base_pass": forced_transfer_gain
        >= thresholds["minimum_forced_transfer_cd_gain_over_base"],
    }
    p4 = all(transfer_gate.values())

    behavior_cd_gain = (
        behavior_calibration["median_cd_cosine_projected"] - base_calibration["median_cd_cosine_projected"]
    )
    separation = {
        "behavior_only_calibration_accuracy_gain": behavior_accuracy_gain,
        "behavior_only_calibration_cd_gain": behavior_cd_gain,
        "behavior_only_calibration_gram_gain": (
            behavior_calibration["gram"]["median_same_gram_cosine"]
            - base_calibration["gram"]["median_same_gram_cosine"]
        ),
        "behavior_only_transfer_accuracy_gain": (
            behavior["evaluation"]["transfer"]["candidate_accuracy"] - base_transfer["candidate_accuracy"]
        ),
        "forced_calibration_cd_gain_over_base": forced_cd_gain_base,
        "forced_calibration_cd_gain_over_behavior_only": forced_cd_gain_behavior,
        "forced_transfer_cd_gain_over_base": forced_transfer_gain,
    }

    final: dict[str, Any] = {
        "schema_version": "phase1125_pythia_controlled_bridge_final.v1",
        "phase": PHASE,
        "protocol_revision": prereg["protocol_revision"],
        "protocol_digest": prereg["protocol_digest"],
        "run_digest": run["run_digest"],
        "arm_digests": {
            "behavior_only": behavior["result_digest"],
            "bridge_forced": forced["result_digest"],
        },
        "engineering_checks": engineering_checks,
        "behavior_only_gate": {
            **behavior_gate,
            "calibration_accuracy": behavior_calibration["candidate_accuracy"],
            "base_calibration_accuracy": base_calibration["candidate_accuracy"],
            "accuracy_gain": behavior_accuracy_gain,
            "passed": p2,
        },
        "forced_bridge_calibration_gate": {
            **calibration_gate,
            "metrics": forced_calibration,
            "cd_gain_over_base": forced_cd_gain_base,
            "cd_gain_over_behavior_only": forced_cd_gain_behavior,
            "passed": p3,
        },
        "forced_bridge_transfer_gate": {
            **transfer_gate,
            "metrics": forced_transfer,
            "cd_gain_over_base": forced_transfer_gain,
            "passed": p4,
        },
        "condition_summaries": {
            "base": base,
            "behavior_only": behavior["evaluation"],
            "bridge_forced": forced["evaluation"],
        },
        "behavior_bridge_separation": separation,
        "predictions": {
            "P1_engineering_and_frozen_base": "pass" if p1 else "fail",
            "P2_behavior_only_unseen_template_learning": "pass" if p2 else "fail",
            "P3_forced_bridge_instrument_visibility": "pass" if p3 else "fail",
            "P4_forced_bridge_unseen_concept_transfer": "pass" if p4 else "fail",
            "P5_natural_mechanism_authorized": "fail",
        },
        "instrument_calibration_passed": p1 and p3,
        "unseen_concept_generalization_passed": p4,
        "natural_mechanism_authorized": False,
        "component_or_causal_work_authorized": False,
        "theory_constraints": {
            "instrument": (
                "The full-state and independent projected C-D instruments detect an explicitly trained strong bridge "
                "on unseen templates. K59/K60 are therefore not explained by algebraic blindness to a strong shared bridge."
            ),
            "behavior": (
                "Behavior-only adapter training can strongly improve the same-concept unseen-template task while the "
                "direct projected C-D cosine remains near baseline. Behavioral learning does not require this bridge."
            ),
            "generalization": (
                "Neither controlled arm establishes unseen-concept semantic generalization. The forced bridge is an "
                "engineered calibration object, not a natural language mechanism."
            ),
        },
        "evidence_level": "E2 method calibration and E2 controlled-training boundary; no natural-mechanism evidence.",
        "auto_continue": {
            "value": 0,
            "reason": (
                "Instrument visibility is calibrated but concept transfer failed. The next valid axis requires new "
                "non-WordNet material or a separately preregistered dynamic-use protocol with independent controls."
            ),
        },
    }
    final["final_digest"] = canonical_digest(final)
    write_json(OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
        "predictions": final["predictions"],
        "behavior_only_gate": final["behavior_only_gate"],
        "forced_bridge_calibration": {
            "passed": p3,
            "projected_cd_cosine": forced_calibration["median_cd_cosine_projected"],
            "projected_cd_positive_rate": forced_calibration["projected_cd_positive_rate"],
            "full_projection_gap": forced_calibration["median_full_projection_gap"],
        },
        "forced_bridge_transfer": {
            "passed": p4,
            "accuracy": forced_transfer["candidate_accuracy"],
            "projected_cd_cosine": forced_transfer["median_cd_cosine_projected"],
            "projected_cd_positive_rate": forced_transfer["projected_cd_positive_rate"],
            "cd_gain_over_base": forced_transfer_gain,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
