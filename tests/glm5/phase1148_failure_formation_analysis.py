from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import phase1148_mandatory_mediation_calibration as p1148


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1148_mandatory_mediation_calibration"


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def analyze() -> dict[str, Any]:
    prereg = p1148.read_json(p1148.PREREG_PATH)
    p1148.verify_preregistration(prereg)
    discovery = p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    audit = p1148.read_json(
        OUT_ROOT / "audit" / "discovery_independent_result_audit.json"
    )
    if discovery["confirmation_authorized"]:
        raise RuntimeError("Failure-formation analysis is only valid after discovery failure")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Independent discovery audit did not pass")

    replicates = p1148.split_replicates(prereg, "discovery")
    threshold = float(prereg["thresholds"]["address_accuracy"])
    slots = ["0", "100", "250", "500", "1000", "2000", "final"]
    pooled: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    per_replicate: dict[str, Any] = {}

    for replicate in replicates:
        per_replicate[replicate] = {}
        for condition in p1148.CONDITIONS:
            summary = p1148.load_summary(replicate, condition, prereg)
            max_step = int(summary["training_steps"])
            trajectory_by_step = {int(point["step"]): point for point in summary["trajectory"]}
            trajectory: dict[str, Any] = {}
            for slot in slots:
                step = max_step if slot == "final" else int(slot)
                point = trajectory_by_step[step]
                trajectory[slot] = {"step": step, "evaluation": point["evaluation"]}
                for split in ("seen", "holdout", "quartet"):
                    for metric in (
                        "accuracy",
                        "row_address_accuracy",
                        "column_address_accuracy",
                        "joint_address_accuracy",
                    ):
                        pooled[condition][f"{split}.{metric}"][slot].append(
                            float(point["evaluation"][split][metric])
                        )

            peaks: dict[str, Any] = {}
            for split in ("holdout", "quartet"):
                peaks[split] = {}
                for metric in (
                    "accuracy",
                    "row_address_accuracy",
                    "column_address_accuracy",
                    "joint_address_accuracy",
                ):
                    candidates = [
                        (float(point["evaluation"][split][metric]), int(point["step"]))
                        for point in summary["trajectory"]
                    ]
                    best_value, best_step = max(candidates)
                    peaks[split][metric] = {
                        "maximum": best_value,
                        "step": best_step,
                        "ever_reached_address_gate": (
                            best_value >= threshold
                            if metric.endswith("address_accuracy")
                            else None
                        ),
                    }
            per_replicate[replicate][condition] = {
                "trajectory": trajectory,
                "peaks": peaks,
                "final_qualified": bool(summary["qualified"]),
            }

    aggregate: dict[str, Any] = {}
    for condition, metrics in pooled.items():
        aggregate[condition] = {
            metric: {slot: summarize(values) for slot, values in by_slot.items()}
            for metric, by_slot in metrics.items()
        }

    soft_diagnostics: dict[str, Any] = {}
    for condition in p1148.SOFT_PRIORITY:
        row_ever = []
        column_ever = []
        joint_ever = []
        for replicate in replicates:
            peaks = per_replicate[replicate][condition]["peaks"]
            row_ever.append(
                all(
                    peaks[split]["row_address_accuracy"]["maximum"] >= threshold
                    for split in ("holdout", "quartet")
                )
            )
            column_ever.append(
                all(
                    peaks[split]["column_address_accuracy"]["maximum"] >= threshold
                    for split in ("holdout", "quartet")
                )
            )
            joint_ever.append(
                all(
                    peaks[split]["joint_address_accuracy"]["maximum"] >= threshold
                    for split in ("holdout", "quartet")
                )
            )
        soft_diagnostics[condition] = {
            "replicates_row_ever_passed_both_panels": int(sum(row_ever)),
            "replicates_column_ever_passed_both_panels": int(sum(column_ever)),
            "replicates_joint_ever_passed_both_panels": int(sum(joint_ever)),
            "replicate_count": len(replicates),
        }

    result = {
        "phase": p1148.PHASE,
        "analysis_scope": "pre_registered_failure_formation_trajectory_descriptive_only",
        "protocol_digest": prereg["protocol_digest"],
        "discovery_selection_digest": discovery["selection_digest"],
        "independent_audit_digest": audit["audit_digest"],
        "replicates": replicates,
        "slots": slots,
        "aggregate": aggregate,
        "per_replicate": per_replicate,
        "soft_diagnostics": soft_diagnostics,
        "claim_boundary": (
            "Describes which precursor variables formed after the success claim stopped. "
            "It cannot establish a successful binding mechanism or authorize hidden-state hotspot selection."
        ),
        "next_branch": "address_acquisition_algorithm_or_task_interface_calibration",
    }
    result["analysis_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "analysis" / "failure_formation_trajectory.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
