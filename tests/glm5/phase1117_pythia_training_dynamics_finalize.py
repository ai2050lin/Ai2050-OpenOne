#!/usr/bin/env python3
"""Finalize Phase1117 checkpoint qualification and training trajectory."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import phase1117_pythia_training_dynamics_protocol as protocol


def checkpoint_step(name: str) -> int:
    return int(name.removeprefix("step"))


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [pair for pair in pairs if pair["finite"]]
    return {
        "pair_count": len(pairs),
        "finite_pair_count": len(finite),
        "finite_fraction": len(finite) / max(len(pairs), 1),
        "direction_accuracy": sum(pair["true_d"] > 0.0 for pair in finite) / max(len(finite), 1),
        "control_direction_accuracy": sum(pair["control_d"] > 0.0 for pair in finite) / max(len(finite), 1),
        "control_advantage": (
            sum(pair["true_d"] > 0.0 for pair in finite)
            - sum(pair["control_d"] > 0.0 for pair in finite)
        )
        / max(len(finite), 1),
        "bidirectional_accuracy": sum(pair["bidirectional"] for pair in finite) / max(len(finite), 1),
        "median_true_d": statistics.median(pair["true_d"] for pair in finite) if finite else None,
        "median_control_d": statistics.median(pair["control_d"] for pair in finite) if finite else None,
        "median_absolute_true_d": statistics.median(abs(pair["true_d"]) for pair in finite) if finite else None,
        "median_absolute_control_d": statistics.median(abs(pair["control_d"]) for pair in finite) if finite else None,
    }


def compute_checkpoint(checkpoint: str, detail: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[row["pair_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for pair_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["sense"])
        if len(rows) != 2 or [row["sense"] for row in rows] != [0, 1]:
            raise RuntimeError(f"malformed pair {pair_id}")
        finite = all(row["finite"] for row in rows)
        true_d = float(rows[0]["true_z"] - rows[1]["true_z"]) if finite else math.nan
        control_d = float(rows[0]["control_z"] - rows[1]["control_z"]) if finite else math.nan
        pairs.append(
            {
                "pair_id": pair_id,
                "concept_id": rows[0]["concept_id"],
                "split": rows[0]["split"],
                "template": rows[0]["template"],
                "finite": finite,
                "true_d": true_d,
                "control_d": control_d,
                "bidirectional": finite and rows[0]["true_z"] > 0.0 and rows[1]["true_z"] < 0.0,
            }
        )

    overall = summarize_pairs(pairs)
    by_split = {
        split: summarize_pairs([pair for pair in pairs if pair["split"] == split])
        for split in protocol.SPLITS
    }
    by_template = {
        str(template): summarize_pairs([pair for pair in pairs if pair["template"] == template])
        for template in range(len(protocol.TEMPLATES))
    }

    concept_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        concept_pairs[pair["concept_id"]].append(pair)
    concepts: dict[str, dict[str, Any]] = {}
    for concept_id, panel in sorted(concept_pairs.items()):
        finite = [pair for pair in panel if pair["finite"]]
        split = panel[0]["split"]
        median_d = statistics.median(pair["true_d"] for pair in finite) if finite else None
        concepts[concept_id] = {
            "split": split,
            "pair_count": len(panel),
            "finite_fraction": len(finite) / max(len(panel), 1),
            "median_true_d": median_d,
            "positive_median": median_d is not None and median_d > 0.0,
            "all_templates_positive": len(finite) == len(protocol.TEMPLATES) and all(pair["true_d"] > 0.0 for pair in finite),
        }
    positive_concepts = sum(value["positive_median"] for value in concepts.values())
    positive_by_split = {
        split: sum(value["positive_median"] for value in concepts.values() if value["split"] == split)
        for split in protocol.SPLITS
    }
    concept_summary = {
        "concept_count": len(concepts),
        "positive_median_count": positive_concepts,
        "positive_median_fraction": positive_concepts / max(len(concepts), 1),
        "positive_by_split": positive_by_split,
        "concepts": concepts,
    }

    thresholds = protocol.THRESHOLDS
    checks = {
        "finite_fraction": overall["finite_fraction"] >= thresholds["minimum_finite_fraction"],
        "overall_direction": overall["direction_accuracy"] >= thresholds["minimum_overall_direction_accuracy"],
        "split_direction": all(value["direction_accuracy"] >= thresholds["minimum_split_direction_accuracy"] for value in by_split.values()),
        "template_direction": all(value["direction_accuracy"] >= thresholds["minimum_template_direction_accuracy"] for value in by_template.values()),
        "overall_control_advantage": overall["control_advantage"] >= thresholds["minimum_overall_control_advantage"],
        "split_control_advantage": all(value["control_advantage"] >= thresholds["minimum_split_control_advantage"] for value in by_split.values()),
        "template_control_advantage": all(value["control_advantage"] >= thresholds["minimum_template_control_advantage"] for value in by_template.values()),
        "concept_positive_fraction": concept_summary["positive_median_fraction"] >= thresholds["minimum_concept_positive_fraction"],
        "concept_split_counts": all(value >= thresholds["minimum_positive_concepts_per_split"] for value in positive_by_split.values()),
    }
    onset_gate = (
        overall["direction_accuracy"] >= thresholds["trajectory_onset_direction_accuracy"]
        and overall["control_advantage"] >= thresholds["trajectory_onset_control_advantage"]
        and concept_summary["positive_median_fraction"] >= thresholds["trajectory_onset_concept_fraction"]
    )
    return {
        "schema_version": "phase1117_checkpoint_metrics.v1",
        "phase": protocol.PHASE,
        "checkpoint": checkpoint,
        "step": checkpoint_step(checkpoint),
        "overall": overall,
        "by_split": by_split,
        "by_template": by_template,
        "concept_summary": concept_summary,
        "qualification_checks": checks,
        "qualification_passed": all(checks.values()),
        "onset_gate": onset_gate,
        "pair_digest": protocol.digest(pairs),
    }


def find_onset(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(metrics, key=lambda value: value["step"])
    for left, right in zip(ordered, ordered[1:]):
        if left["onset_gate"] and right["onset_gate"]:
            return {
                "observed": True,
                "first_checkpoint": left["checkpoint"],
                "confirmation_checkpoint": right["checkpoint"],
                "interpretation": "first two consecutive sampled checkpoints passing the frozen onset gate",
            }
    return {
        "observed": False,
        "first_checkpoint": None,
        "confirmation_checkpoint": None,
        "interpretation": "no two consecutive sampled checkpoints pass the frozen onset gate",
    }


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("protocol audit failed")

    present: list[str] = []
    checkpoint_metrics: dict[str, Any] = {}
    behavior_summaries: dict[str, Any] = {}
    for checkpoint in protocol.CHECKPOINTS:
        root = protocol.OUT_ROOT / "behavior" / checkpoint
        detail_path = root / "candidate_detail.jsonl"
        summary_path = root / "summary.json"
        if not detail_path.exists() and not summary_path.exists():
            continue
        if not detail_path.exists() or not summary_path.exists():
            raise RuntimeError(f"incomplete behavior artifacts for {checkpoint}")
        detail = list(protocol.read_jsonl(detail_path))
        behavior_summary = protocol.read_json(summary_path)
        if protocol.digest(detail) != behavior_summary["detail_digest"]:
            raise RuntimeError(f"detail digest mismatch for {checkpoint}")
        present.append(checkpoint)
        behavior_summaries[checkpoint] = behavior_summary
        checkpoint_metrics[checkpoint] = compute_checkpoint(checkpoint, detail)

    if protocol.FINAL_QUALIFICATION_CHECKPOINT not in checkpoint_metrics:
        raise RuntimeError("step143000 must run before finalization")
    final_metrics = checkpoint_metrics[protocol.FINAL_QUALIFICATION_CHECKPOINT]
    trajectory_authorized = bool(final_metrics["qualification_passed"])
    authorization_core = {
        "schema_version": "phase1117_trajectory_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_qualification_checkpoint": protocol.FINAL_QUALIFICATION_CHECKPOINT,
        "trajectory_authorized": trajectory_authorized,
        "hidden_state_authorized": False,
        "reason": "final_checkpoint_passed_frozen_training_axis_gate" if trajectory_authorized else "final_checkpoint_failed_frozen_training_axis_gate",
    }
    authorization = dict(authorization_core)
    authorization["authorization_digest"] = protocol.digest(authorization_core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "trajectory_authorization.json", authorization)

    metrics_list = [checkpoint_metrics[name] for name in present]
    full_trajectory = set(present) == set(protocol.CHECKPOINTS)
    onset = find_onset(metrics_list) if full_trajectory else {
        "observed": None,
        "first_checkpoint": None,
        "confirmation_checkpoint": None,
        "interpretation": "trajectory incomplete",
    }
    step_gain = None
    p5_passed = None
    if "step0" in checkpoint_metrics:
        initial = checkpoint_metrics["step0"]["overall"]
        final = final_metrics["overall"]
        step_gain = {
            "direction_accuracy": final["direction_accuracy"] - initial["direction_accuracy"],
            "control_advantage": final["control_advantage"] - initial["control_advantage"],
            "median_true_d": final["median_true_d"] - initial["median_true_d"],
        }
        p5_passed = (
            step_gain["direction_accuracy"] >= protocol.THRESHOLDS["minimum_final_minus_step0_direction_gain"]
            and step_gain["control_advantage"] >= protocol.THRESHOLDS["minimum_final_minus_step0_advantage_gain"]
        )

    predictions = {
        "P1": "pass" if protocol_audit["all_checks_passed"] else "fail",
        "P2": "pass" if trajectory_authorized else "fail",
        "P3": "pass" if all(name == protocol.FINAL_QUALIFICATION_CHECKPOINT for name in present) or trajectory_authorized else "fail",
        "P4": ("pass" if full_trajectory and onset["observed"] else "fail") if full_trajectory else "not_tested",
        "P5": ("pass" if p5_passed else "fail") if p5_passed is not None else "not_tested",
        "P6": "pass",
    }
    final_core = {
        "schema_version": "phase1117_pythia_training_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "trajectory_authorized": trajectory_authorized,
        "hidden_state_authorized": False,
        "present_checkpoints": present,
        "missing_checkpoints": [name for name in protocol.CHECKPOINTS if name not in present],
        "full_trajectory_complete": full_trajectory,
        "checkpoint_metrics": checkpoint_metrics,
        "behavior_summary_digests": {name: behavior_summaries[name]["summary_digest"] for name in present},
        "onset": onset,
        "step0_to_final_gain": step_gain,
        "prospective_predictions": predictions,
        "automatic_continuation": {
            "decision": "run_frozen_trajectory" if trajectory_authorized and not full_trajectory else ("trajectory_complete" if full_trajectory else "stop_training_axis"),
            "run_remaining_checkpoints": trajectory_authorized and not full_trajectory,
            "run_hidden_or_causal": False,
        },
        "interpretation": {
            "positive_limit": "A positive curve establishes formation of a Pythia output-margin contextual modulation under this fixed base-LM interface.",
            "negative_limit": "A failure constrains this Pythia size, tokenizer, source material, and base-LM interface; it does not show that semantic learning dynamics are absent.",
            "not_claimed": [
                "pure semantic effect",
                "K43 exact-key formation order",
                "hidden semantic invariance",
                "causal content reading",
                "cross-family or scale conservation",
            ],
        },
    }
    final_summary = dict(final_core)
    final_summary["final_digest"] = protocol.digest(final_core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "checkpoint_metrics.json", {"phase": protocol.PHASE, "metrics": checkpoint_metrics})
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final_summary)
    return final_summary


if __name__ == "__main__":
    summary = finalize()
    print(
        json.dumps(
            {
                "phase": summary["phase"],
                "trajectory_authorized": summary["trajectory_authorized"],
                "present_checkpoints": summary["present_checkpoints"],
                "missing_checkpoints": summary["missing_checkpoints"],
                "onset": summary["onset"],
                "prospective_predictions": summary["prospective_predictions"],
                "automatic_continuation": summary["automatic_continuation"],
                "final_digest": summary["final_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
