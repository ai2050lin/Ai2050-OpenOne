#!/usr/bin/env python3
"""Finalize the Phase1120 residual readout and signed-geometry map."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import phase1120_pythia_hidden_formation_protocol as protocol


def checkpoint_step(name: str) -> int:
    return int(name.removeprefix("step"))


def load_checkpoint(checkpoint: str, prereg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = protocol.OUT_ROOT / "hidden" / checkpoint
    summary = protocol.read_json(root / "summary.json")
    artifact_path = protocol.OUT_ROOT / summary["artifact"]
    if summary["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError(f"protocol digest mismatch for {checkpoint}")
    if protocol.file_sha256(artifact_path) != summary["artifact_sha256"]:
        raise RuntimeError(f"artifact digest mismatch for {checkpoint}")
    with np.load(artifact_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    expected_shapes = {
        "case_indices": (prereg["case_count"],),
        "true_z": (prereg["case_count"], prereg["hidden_state_count"]),
        "control_z": (prereg["case_count"], prereg["hidden_state_count"]),
        "state_projection": (prereg["case_count"], prereg["hidden_state_count"], prereg["projection"]["dimension"]),
        "final_selected_logit_error": (prereg["case_count"], 4),
    }
    if set(arrays) != set(expected_shapes):
        raise RuntimeError(f"unexpected arrays for {checkpoint}: {sorted(arrays)}")
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"shape mismatch for {checkpoint}/{name}: {arrays[name].shape}")
    if not np.array_equal(arrays["case_indices"], np.arange(prereg["case_count"], dtype=np.int32)):
        raise RuntimeError(f"case order mismatch for {checkpoint}")
    return summary, arrays


def pair_index(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[row["pair_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for pair_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["sense"])
        if len(rows) != 2 or [row["sense"] for row in rows] != [0, 1]:
            raise RuntimeError(f"malformed pair: {pair_id}")
        pairs.append(
            {
                "pair_id": pair_id,
                "case0": int(rows[0]["case_index"]),
                "case1": int(rows[1]["case_index"]),
                "concept_id": rows[0]["concept_id"],
                "control_concept_id": rows[0]["control_concept_id"],
                "split": rows[0]["split"],
                "template": int(rows[0]["template"]),
            }
        )
    return pairs


def readout_summary(true_d: np.ndarray, control_d: np.ndarray, indices: list[int]) -> dict[str, Any]:
    true_values = true_d[indices]
    control_values = control_d[indices]
    finite = np.isfinite(true_values) & np.isfinite(control_values)
    count = int(finite.sum())
    total = len(indices)
    if count == 0:
        return {
            "pair_count": total,
            "finite_pair_count": 0,
            "finite_fraction": 0.0,
            "direction_accuracy": None,
            "control_direction_accuracy": None,
            "control_advantage": None,
            "median_true_d": None,
            "median_control_d": None,
        }
    true_finite = true_values[finite]
    control_finite = control_values[finite]
    direction = float(np.mean(true_finite > 0.0))
    control_direction = float(np.mean(control_finite > 0.0))
    return {
        "pair_count": total,
        "finite_pair_count": count,
        "finite_fraction": count / max(total, 1),
        "direction_accuracy": direction,
        "control_direction_accuracy": control_direction,
        "control_advantage": direction - control_direction,
        "median_true_d": float(np.median(true_finite)),
        "median_control_d": float(np.median(control_finite)),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm) or left_norm <= 1.0e-8 or right_norm <= 1.0e-8:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def geometry_summary(pair_vectors: np.ndarray, pairs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    indices = [index for index, pair in enumerate(pairs) if pair["split"] == split]
    lookup = {(pairs[index]["concept_id"], pairs[index]["template"]): index for index in indices}
    by_concept: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_concept[pairs[index]["concept_id"]].append(index)

    within: list[float] = []
    for concept_indices in by_concept.values():
        ordered = sorted(concept_indices, key=lambda index: pairs[index]["template"])
        for left_index, right_index in itertools.combinations(ordered, 2):
            value = cosine(pair_vectors[left_index], pair_vectors[right_index])
            if value is not None:
                within.append(value)

    null: list[float] = []
    nonzero_vectors = 0
    for index in indices:
        if float(np.linalg.norm(pair_vectors[index])) > 1.0e-8:
            nonzero_vectors += 1
        donor_index = lookup.get((pairs[index]["control_concept_id"], pairs[index]["template"]))
        if donor_index is None:
            raise RuntimeError("matched geometry control is missing")
        value = cosine(pair_vectors[index], pair_vectors[donor_index])
        if value is not None:
            null.append(value)

    median_within = statistics.median(within) if within else None
    median_null = statistics.median(null) if null else None
    advantage = median_within - median_null if median_within is not None and median_null is not None else None
    return {
        "pair_count": len(indices),
        "nonzero_pair_count": nonzero_vectors,
        "nonzero_pair_fraction": nonzero_vectors / max(len(indices), 1),
        "within_cosine_count": len(within),
        "null_cosine_count": len(null),
        "median_within_concept_cosine": median_within,
        "median_deranged_control_cosine": median_null,
        "geometry_advantage": advantage,
    }


def compute_checkpoint(arrays: dict[str, np.ndarray], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    case0 = np.array([pair["case0"] for pair in pairs], dtype=np.int64)
    case1 = np.array([pair["case1"] for pair in pairs], dtype=np.int64)
    true_d = arrays["true_z"][case0] - arrays["true_z"][case1]
    control_d = arrays["control_z"][case0] - arrays["control_z"][case1]
    pair_vectors = arrays["state_projection"][case0].astype(np.float32) - arrays["state_projection"][case1].astype(np.float32)
    split_indices = {
        split: [index for index, pair in enumerate(pairs) if pair["split"] == split]
        for split in protocol.SPLITS
    }
    overall_indices = list(range(len(pairs)))
    layer_metrics: dict[str, Any] = {}
    for layer_index in range(protocol.HIDDEN_STATE_COUNT):
        readout = {"overall": readout_summary(true_d[:, layer_index], control_d[:, layer_index], overall_indices)}
        geometry: dict[str, Any] = {}
        for split in protocol.SPLITS:
            readout[split] = readout_summary(true_d[:, layer_index], control_d[:, layer_index], split_indices[split])
            geometry[split] = geometry_summary(pair_vectors[:, layer_index, :], pairs, split)
        layer_metrics[str(layer_index)] = {
            "normalized_depth": layer_index / (protocol.HIDDEN_STATE_COUNT - 1),
            "readout": readout,
            "geometry": geometry,
        }
    return {
        "layer_metrics": layer_metrics,
        "maximum_final_selected_logit_error": float(np.max(arrays["final_selected_logit_error"])),
        "all_arrays_finite": bool(
            np.isfinite(arrays["true_z"]).all()
            and np.isfinite(arrays["control_z"]).all()
            and np.isfinite(arrays["state_projection"]).all()
            and np.isfinite(arrays["final_selected_logit_error"]).all()
        ),
    }


def select_layer(
    checkpoint_metrics: dict[str, Any],
    metric_path: tuple[str, ...],
) -> dict[str, Any]:
    initial = checkpoint_metrics["step0"]["layer_metrics"]
    final = checkpoint_metrics["step143000"]["layer_metrics"]
    candidates = []
    for layer_index in protocol.ELIGIBLE_LAYER_INDICES:
        before: Any = initial[str(layer_index)]
        after: Any = final[str(layer_index)]
        for key in metric_path:
            before = before[key]
            after = after[key]
        if before is None or after is None:
            gain = -math.inf
        else:
            gain = float(after - before)
        candidates.append({"layer_index": layer_index, "step0": before, "final": after, "gain": gain})
    selected = sorted(candidates, key=lambda item: (-item["gain"], item["layer_index"]))[0]
    return {"selected": selected, "candidates": candidates, "metric_path": list(metric_path)}


def split_gate(
    checkpoint_metrics: dict[str, Any],
    layer_index: int,
    split: str,
    family: str,
    final_threshold: float,
    gain_threshold: float,
) -> dict[str, Any]:
    if family == "readout":
        key_path = ("readout", split, "control_advantage")
    elif family == "geometry":
        key_path = ("geometry", split, "geometry_advantage")
    else:
        raise ValueError(family)

    def value(checkpoint: str) -> float | None:
        current: Any = checkpoint_metrics[checkpoint]["layer_metrics"][str(layer_index)]
        for key in key_path:
            current = current[key]
        return current

    initial = value("step0")
    final = value("step143000")
    gain = final - initial if initial is not None and final is not None else None
    passed = final is not None and gain is not None and final >= final_threshold and gain >= gain_threshold
    return {"split": split, "step0": initial, "final": final, "gain": gain, "passed": passed}


def event_onset(
    checkpoint_metrics: dict[str, Any],
    layer_index: int,
    family: str,
    advantage_threshold: float,
    gain_threshold: float,
) -> dict[str, Any]:
    if family == "readout":
        key = "control_advantage"
    else:
        key = "geometry_advantage"
    initial_values = {}
    for split in ("independent_confirmation", "heldout"):
        branch = "readout" if family == "readout" else "geometry"
        initial_values[split] = checkpoint_metrics["step0"]["layer_metrics"][str(layer_index)][branch][split][key]

    states = []
    for checkpoint in protocol.CHECKPOINTS:
        split_values = {}
        passed = True
        for split in ("independent_confirmation", "heldout"):
            branch = "readout" if family == "readout" else "geometry"
            current = checkpoint_metrics[checkpoint]["layer_metrics"][str(layer_index)][branch][split][key]
            initial = initial_values[split]
            gain = current - initial if current is not None and initial is not None else None
            split_values[split] = {"value": current, "gain": gain}
            passed = passed and current is not None and gain is not None and current >= advantage_threshold and gain >= gain_threshold
        states.append({"checkpoint": checkpoint, "step": checkpoint_step(checkpoint), "passed": bool(passed), "splits": split_values})
    for left, right in zip(states, states[1:]):
        if left["passed"] and right["passed"]:
            return {
                "observed": True,
                "first_checkpoint": left["checkpoint"],
                "confirmation_checkpoint": right["checkpoint"],
                "states": states,
            }
    return {"observed": False, "first_checkpoint": None, "confirmation_checkpoint": None, "states": states}


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1120 protocol audit failed")
    cases = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    if protocol.digest(cases) != prereg["case_digest"]:
        raise RuntimeError("case digest mismatch")
    pairs = pair_index(cases)

    checkpoint_metrics: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for checkpoint in protocol.CHECKPOINTS:
        summary, arrays = load_checkpoint(checkpoint, prereg)
        summaries[checkpoint] = summary
        checkpoint_metrics[checkpoint] = compute_checkpoint(arrays, pairs)

    readout_selection = select_layer(
        checkpoint_metrics,
        ("readout", "discovery", "control_advantage"),
    )
    geometry_selection = select_layer(
        checkpoint_metrics,
        ("geometry", "discovery", "geometry_advantage"),
    )
    readout_layer = int(readout_selection["selected"]["layer_index"])
    geometry_layer = int(geometry_selection["selected"]["layer_index"])
    thresholds = prereg["thresholds"]
    confirmation_splits = ("independent_confirmation", "heldout")
    readout_gates = [
        split_gate(
            checkpoint_metrics,
            readout_layer,
            split,
            "readout",
            thresholds["minimum_final_readout_advantage"],
            thresholds["minimum_step0_to_final_readout_gain"],
        )
        for split in confirmation_splits
    ]
    geometry_gates = [
        split_gate(
            checkpoint_metrics,
            geometry_layer,
            split,
            "geometry",
            thresholds["minimum_final_geometry_advantage"],
            thresholds["minimum_step0_to_final_geometry_gain"],
        )
        for split in confirmation_splits
    ]
    readout_event_passed = all(gate["passed"] for gate in readout_gates)
    geometry_event_passed = all(gate["passed"] for gate in geometry_gates)

    instrument_passed = all(
        checkpoint_metrics[name]["all_arrays_finite"]
        and summaries[name]["finite_fraction"] >= thresholds["minimum_finite_fraction"]
        and checkpoint_metrics[name]["maximum_final_selected_logit_error"]
        <= thresholds["maximum_final_logit_reproduction_error"]
        for name in protocol.CHECKPOINTS
    )

    source_final = protocol.read_json(protocol.SOURCE_ROOT / "analysis" / "final_summary.json")
    source_overall = source_final["checkpoint_metrics"]["step143000"]["overall"]
    reproduced_overall = checkpoint_metrics["step143000"]["layer_metrics"][str(protocol.HIDDEN_STATE_COUNT - 1)]["readout"]["overall"]
    output_reproduction = {
        "source_direction_accuracy": source_overall["direction_accuracy"],
        "reproduced_direction_accuracy": reproduced_overall["direction_accuracy"],
        "source_control_direction_accuracy": source_overall["control_direction_accuracy"],
        "reproduced_control_direction_accuracy": reproduced_overall["control_direction_accuracy"],
        "source_control_advantage": source_overall["control_advantage"],
        "reproduced_control_advantage": reproduced_overall["control_advantage"],
    }
    output_reproduction["aggregate_exact_match"] = all(
        abs(output_reproduction[left] - output_reproduction[right]) <= 1.0e-12
        for left, right in (
            ("source_direction_accuracy", "reproduced_direction_accuracy"),
            ("source_control_direction_accuracy", "reproduced_control_direction_accuracy"),
            ("source_control_advantage", "reproduced_control_advantage"),
        )
    )
    output_reproduction["interpretation"] = (
        "Descriptive aggregate comparison only. The frozen P2 instrument gate is the per-candidate "
        "maximum absolute error threshold; tiny near-zero margins may change sign without violating P2."
    )

    readout_onset = event_onset(
        checkpoint_metrics,
        readout_layer,
        "readout",
        thresholds["minimum_readout_onset_advantage"],
        thresholds["minimum_readout_onset_gain"],
    )
    geometry_onset = event_onset(
        checkpoint_metrics,
        geometry_layer,
        "geometry",
        thresholds["minimum_geometry_onset_advantage"],
        thresholds["minimum_geometry_onset_gain"],
    )
    joint_event = instrument_passed and readout_event_passed and geometry_event_passed
    predictions = {
        "P1": "pass",
        "P2": "pass" if instrument_passed else "fail",
        "P3": "pass" if readout_event_passed else "fail",
        "P4": "pass" if geometry_event_passed else "fail",
        "P5": "pass",
        "P6": "pass",
    }
    final_core = {
        "schema_version": "phase1120_pythia_hidden_formation_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "case_digest": prereg["case_digest"],
        "checkpoints": list(protocol.CHECKPOINTS),
        "instrument_passed": instrument_passed,
        "output_reproduction": output_reproduction,
        "readout_selection": readout_selection,
        "geometry_selection": geometry_selection,
        "readout_confirmation_gates": readout_gates,
        "geometry_confirmation_gates": geometry_gates,
        "readout_event_passed": readout_event_passed,
        "geometry_event_passed": geometry_event_passed,
        "joint_event_passed": joint_event,
        "readout_onset": readout_onset,
        "geometry_onset": geometry_onset,
        "checkpoint_metrics": checkpoint_metrics,
        "checkpoint_summary_digests": {name: summaries[name]["summary_digest"] for name in protocol.CHECKPOINTS},
        "prospective_predictions": predictions,
        "automatic_continuation": {
            "separate_component_protocol_authorized": joint_event,
            "run_component_or_causal_in_phase1120": False,
            "reason": "joint residual readout and signed geometry event confirmed" if joint_event else "one or both independently confirmed event gates failed",
        },
        "interpretation": {
            "positive_limit": "A positive readout event shows when candidate-relative information becomes linearly readable from an intermediate residual stream; a positive projected-geometry event shows a repeatable context-pair relation under a fixed random projection.",
            "negative_limit": "Failure constrains this Pythia run, WordNet material, query boundary, readout, projection, and thresholds; it does not show that hidden contextual computation is absent.",
            "not_claimed": [
                "pure semantic representation",
                "attention or MLP execution mechanism",
                "causal necessity or sufficiency",
                "cross-model physical conservation",
                "a monotonic or universal formation law",
            ],
        },
    }
    final_summary = dict(final_core)
    final_summary["final_digest"] = protocol.digest(final_core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "layer_metrics.json", {"phase": protocol.PHASE, "metrics": checkpoint_metrics})
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final_summary)
    return final_summary


if __name__ == "__main__":
    result = finalize()
    print(
        json.dumps(
            {
                "phase": result["phase"],
                "instrument_passed": result["instrument_passed"],
                "readout_selected": result["readout_selection"]["selected"],
                "geometry_selected": result["geometry_selection"]["selected"],
                "readout_confirmation_gates": result["readout_confirmation_gates"],
                "geometry_confirmation_gates": result["geometry_confirmation_gates"],
                "readout_onset": result["readout_onset"],
                "geometry_onset": result["geometry_onset"],
                "prospective_predictions": result["prospective_predictions"],
                "automatic_continuation": result["automatic_continuation"],
                "final_digest": result["final_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
