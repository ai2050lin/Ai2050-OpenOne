#!/usr/bin/env python3
"""Apply the frozen Phase603 detector at the earlier object coordinate."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase603_fruit_residual_protocol as phase603  # noqa: E402
from phase603_fruit_residual_analysis import (  # noqa: E402
    analyze_branch,
    normalized_role_surface_effects,
)
from phase603_fruit_residual_extract import output_paths as answer_output_paths  # noqa: E402
import phase604_object_coordinate_protocol as protocol  # noqa: E402
from phase604_object_coordinate_extract import output_paths  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_set(branch: dict) -> set[tuple[int, int]]:
    return {(row["layer"], row["unit"]) for row in branch["repeated_candidates"]}


def evaluate_candidates_at_coordinate(
    activations: np.ndarray,
    metadata: list[dict],
    track: str,
    candidates: list[dict],
) -> dict:
    effects = {
        split: normalized_role_surface_effects(activations, metadata, track, split)
        for split in phase603.SPLITS
    }
    same_direction = []
    either_direction = []
    for row in candidates:
        layer, unit = row["layer"], row["unit"]
        source_direction = int(row["direction"])
        thresholds = {
            "discovery": phase603.DISCOVERY_MIN_NORMALIZED_EFFECT,
            "independent_confirmation": phase603.CONFIRMATION_MIN_NORMALIZED_EFFECT,
            "heldout": phase603.HELDOUT_MIN_NORMALIZED_EFFECT,
        }
        same_pass = all(
            float(np.min(effects[split][:, :, layer, unit] * source_direction)) >= minimum
            for split, minimum in thresholds.items()
        )
        positive_pass = all(
            float(np.min(effects[split][:, :, layer, unit])) >= minimum
            for split, minimum in thresholds.items()
        )
        negative_pass = all(
            float(np.min(-effects[split][:, :, layer, unit])) >= minimum
            for split, minimum in thresholds.items()
        )
        if same_pass:
            same_direction.append([layer, unit])
        if positive_pass or negative_pass:
            either_direction.append([layer, unit])
    return {
        "source_candidate_count": len(candidates),
        "same_direction_strict_count": len(same_direction),
        "either_direction_strict_count": len(either_direction),
        "same_direction_units": same_direction,
        "either_direction_units": either_direction,
    }


def midpoint_readout_diagnostic(
    activations: np.ndarray,
    metadata: list[dict],
    track: str,
    layer: int,
    unit: int,
) -> dict:
    discovery_fruit = np.array([
        row["track"] == track and row["split"] == "discovery" and row["fruit_member"]
        for row in metadata
    ])
    discovery_nonfruit = np.array([
        row["track"] == track and row["split"] == "discovery" and not row["fruit_member"]
        for row in metadata
    ])
    fruit_mean = float(activations[discovery_fruit, layer, unit].astype(np.float32).mean())
    nonfruit_mean = float(activations[discovery_nonfruit, layer, unit].astype(np.float32).mean())
    direction = 1 if fruit_mean > nonfruit_mean else -1
    threshold = (fruit_mean + nonfruit_mean) / 2
    split_metrics = {}
    for split in phase603.SPLITS:
        indices = [
            index for index, row in enumerate(metadata)
            if row["track"] == track and row["split"] == split
        ]
        correct = {
            index: bool((float(activations[index, layer, unit]) - threshold) * direction > 0)
            == bool(metadata[index]["fruit_member"])
            for index in indices
        }
        concept_ids = {metadata[index]["concept_id"] for index in indices}
        split_metrics[split] = {
            "case_accuracy": sum(correct.values()) / max(1, len(correct)),
            "concept_all_surface_accuracy": sum(
                all(correct[index] for index in indices if metadata[index]["concept_id"] == concept_id)
                for concept_id in concept_ids
            ) / max(1, len(concept_ids)),
        }
    return {
        "discovery_fruit_mean": fruit_mean,
        "discovery_nonfruit_mean": nonfruit_mean,
        "discovery_direction": direction,
        "discovery_midpoint_threshold": threshold,
        "split_metrics": split_metrics,
    }


def analyze() -> dict:
    answer_analysis = json.loads(phase603.ANALYSIS_PATH.read_text())
    branches = {}
    coordinate_overlap = {}
    untruncated_coordinate_retention = {}
    object_sets = {}
    for model, tracks in protocol.QUALIFIED_BRANCHES.items():
        paths = output_paths(model)
        summary = json.loads(paths["summary"].read_text())
        if summary["arrays_sha256"] != sha256_file(paths["arrays"]):
            raise RuntimeError(f"Phase604 array drift: {model}")
        metadata = json.loads(paths["metadata"].read_text())
        with np.load(paths["arrays"]) as stored:
            activations = stored["activations"]
        answer_paths = answer_output_paths(model)
        answer_metadata = json.loads(answer_paths["metadata"].read_text())
        with np.load(answer_paths["arrays"]) as stored:
            answer_activations = stored["activations"]
        for track in tracks:
            key = f"{model}/{track}"
            branch = analyze_branch(activations, metadata, track)
            branches[key] = branch
            object_set = candidate_set(branch)
            answer_set = candidate_set(answer_analysis["branches"][key])
            object_sets[key] = object_set
            union = object_set | answer_set
            coordinate_overlap[key] = {
                "object_coordinate_count": len(object_set),
                "answer_boundary_count": len(answer_set),
                "intersection_count": len(object_set & answer_set),
                "jaccard": len(object_set & answer_set) / max(1, len(union)),
            }
            untruncated_coordinate_retention[key] = {
                "answer_candidates_tested_at_object_coordinate": evaluate_candidates_at_coordinate(
                    activations,
                    metadata,
                    track,
                    answer_analysis["branches"][key]["repeated_candidates"],
                ),
                "object_candidates_tested_at_answer_boundary": evaluate_candidates_at_coordinate(
                    answer_activations,
                    answer_metadata,
                    track,
                    branch["repeated_candidates"],
                ),
            }
    qwen_keys = [f"qwen3/{track}" for track in protocol.QUALIFIED_BRANCHES["qwen3"]]
    all_tracks = set.intersection(*(object_sets[key] for key in qwen_keys))
    common_records = []
    for layer, unit in sorted(all_tracks):
        directions = {}
        answer_same_direction = {}
        answer_either_direction = {}
        for track in protocol.QUALIFIED_BRANCHES["qwen3"]:
            key = f"qwen3/{track}"
            source = next(
                row for row in branches[key]["repeated_candidates"]
                if row["layer"] == layer and row["unit"] == unit
            )
            directions[track] = source["direction"]
            retention = untruncated_coordinate_retention[key]["object_candidates_tested_at_answer_boundary"]
            answer_same_direction[track] = [layer, unit] in retention["same_direction_units"]
            answer_either_direction[track] = [layer, unit] in retention["either_direction_units"]
        common_records.append({
            "layer": layer,
            "unit": unit,
            "object_direction_by_track": directions,
            "object_direction_consensus": len(set(directions.values())) == 1,
            "answer_boundary_same_direction_by_track": answer_same_direction,
            "answer_boundary_either_direction_by_track": answer_either_direction,
            "survives_answer_boundary_all_tracks_same_direction": all(answer_same_direction.values()),
            "survives_answer_boundary_all_tracks_allow_direction_change": all(answer_either_direction.values()),
        })
    priority_candidates = [
        row for row in common_records
        if row["survives_answer_boundary_all_tracks_allow_direction_change"]
    ]
    priority_diagnostics = []
    if priority_candidates:
        qwen_object_paths = output_paths("qwen3")
        qwen_answer_paths = answer_output_paths("qwen3")
        object_metadata = json.loads(qwen_object_paths["metadata"].read_text())
        answer_metadata = json.loads(qwen_answer_paths["metadata"].read_text())
        with np.load(qwen_object_paths["arrays"]) as stored:
            object_activations = stored["activations"]
        with np.load(qwen_answer_paths["arrays"]) as stored:
            answer_activations = stored["activations"]
        for candidate in priority_candidates:
            layer, unit = candidate["layer"], candidate["unit"]
            priority_diagnostics.append({
                "layer": layer,
                "unit": unit,
                "object_coordinate_by_track": {
                    track: midpoint_readout_diagnostic(
                        object_activations, object_metadata, track, layer, unit
                    )
                    for track in protocol.QUALIFIED_BRANCHES["qwen3"]
                },
                "answer_boundary_by_track": {
                    track: midpoint_readout_diagnostic(
                        answer_activations, answer_metadata, track, layer, unit
                    )
                    for track in protocol.QUALIFIED_BRANCHES["qwen3"]
                },
                "single_unit_decoder_claim_authorized": False,
            })
    payload = {
        "schema_version": "phase604_object_coordinate_analysis.v1", "phase_id": protocol.PHASE,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "branches": branches, "object_vs_answer_boundary_overlap": coordinate_overlap,
        "untruncated_coordinate_retention": untruncated_coordinate_retention,
        "qwen_all_three_track_object_coordinate_overlap_count": len(all_tracks),
        "qwen_all_three_track_object_coordinate_units": [list(value) for value in sorted(all_tracks)],
        "qwen_all_three_track_object_coordinate_records": common_records,
        "qwen_object_direction_consensus_count": sum(row["object_direction_consensus"] for row in common_records),
        "qwen_object_to_answer_all_track_same_direction_count": sum(
            row["survives_answer_boundary_all_tracks_same_direction"] for row in common_records
        ),
        "qwen_object_to_answer_all_track_allow_direction_change_count": sum(
            row["survives_answer_boundary_all_tracks_allow_direction_change"] for row in common_records
        ),
        "qwen_cross_track_coordinate_priority_diagnostics": priority_diagnostics,
        "future_option_tokens_excluded": True, "causal_intervention_authorized": False,
        "mechanism_claim_authorized": False, "theory_or_formula_update_authorized": False,
        "evidence_boundary": (
            "Object-end residual candidates are observational lexical/context states. Repetition before answer options "
            "is stronger evidence of pre-decision category-related structure, but still does not establish a code or causal mechanism."
        ),
    }
    protocol.ANALYSIS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, sort_keys=True))
