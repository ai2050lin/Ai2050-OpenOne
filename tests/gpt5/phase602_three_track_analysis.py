#!/usr/bin/env python3
"""Aggregate the three frozen Phase602 behavior runs without mechanism inference."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase602_three_track_protocol as protocol  # noqa: E402
from phase602_three_track_behavior import output_paths  # noqa: E402


OUT_PATH = protocol.OUT_DIR / "phase602_cross_model_analysis.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze() -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in protocol.MODELS:
        paths = output_paths(model)
        summary = json.loads(paths["summary"].read_text())
        if summary.get("status") != "complete" or summary["rows_sha256"] != sha256_file(paths["rows"]):
            raise RuntimeError(f"Incomplete or drifting Phase602 run: {model}")
        summaries[model] = summary
        for row in read_rows(paths["rows"]):
            rows_by_case[row["case_id"]][model] = row
    if any(set(rows) != set(protocol.MODELS) for rows in rows_by_case.values()):
        raise RuntimeError("Phase602 cross-model denominator mismatch")
    qualifications = {
        model: {track: summaries[model]["track_metrics"][track]["behavior_qualified"] for track in protocol.TRACKS}
        for model in protocol.MODELS
    }
    qualified_model_tracks = [
        f"{model}/{track}" for model in protocol.MODELS for track in protocol.TRACKS
        if qualifications[model][track]
    ]
    common_qualified_tracks = [
        track for track in protocol.TRACKS if all(qualifications[model][track] for model in protocol.MODELS)
    ]
    track_metrics: dict[str, Any] = {}
    for track in protocol.TRACKS:
        track_cases = [
            values for values in rows_by_case.values()
            if next(iter(values.values()))["track"] == track
        ]
        track_metrics[track] = {
            "case_count": len(track_cases),
            "accuracy_by_model": {
                model: summaries[model]["track_metrics"][track]["overall"]["forced_choice_accuracy"]
                for model in protocol.MODELS
            },
            "heldout_accuracy_by_model": {
                model: summaries[model]["track_metrics"][track]["split_metrics"]["heldout"]["forced_choice_accuracy"]
                for model in protocol.MODELS
            },
            "all_models_correct_rate": sum(
                all(values[model]["forced_choice_correct"] for model in protocol.MODELS)
                for values in track_cases
            ) / max(1, len(track_cases)),
            "unanimous_prediction_rate": sum(
                len({values[model]["forced_choice_prediction"] for model in protocol.MODELS}) == 1
                for values in track_cases
            ) / max(1, len(track_cases)),
        }
    matched_track_differences = {}
    for model in protocol.MODELS:
        model_rows = {
            (next(iter(values.values()))["concept_id"], next(iter(values.values()))["surface_id"], next(iter(values.values()))["track"]): values[model]
            for values in rows_by_case.values()
        }
        differences = {}
        for left, right in (("technical", "daily"), ("explicit_evidence", "technical"), ("explicit_evidence", "daily")):
            pairs = []
            for concept_id, surface_id, track in list(model_rows):
                if track != left:
                    continue
                a = model_rows[(concept_id, surface_id, left)]
                b = model_rows[(concept_id, surface_id, right)]
                pairs.append((a, b))
            differences[f"{left}_minus_{right}"] = {
                "matched_pair_count": len(pairs),
                "accuracy_difference": sum(a["forced_choice_correct"] - b["forced_choice_correct"] for a, b in pairs) / max(1, len(pairs)),
                "prediction_disagreement_rate": sum(a["forced_choice_prediction"] != b["forced_choice_prediction"] for a, b in pairs) / max(1, len(pairs)),
                "mean_margin_difference": sum(float(a["target_margin"]) - float(b["target_margin"]) for a, b in pairs) / max(1, len(pairs)),
            }
        matched_track_differences[model] = differences
    payload = {
        "schema_version": "phase602_cross_model_analysis.v1",
        "phase_id": protocol.PHASE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "case_count": len(rows_by_case),
        "concept_count": len({next(iter(values.values()))["concept_id"] for values in rows_by_case.values()}),
        "model_track_qualification": qualifications,
        "qualified_model_tracks": qualified_model_tracks,
        "common_qualified_tracks": common_qualified_tracks,
        "track_metrics": track_metrics,
        "matched_track_differences": matched_track_differences,
        "entity_role_case_count": dict(Counter(next(iter(values.values()))["entity_role"] for values in rows_by_case.values())),
        "model_specific_internal_observation_authorized": qualified_model_tracks,
        "cross_model_internal_observation_authorized_tracks": common_qualified_tracks,
        "causal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "theory_or_formula_update_authorized": False,
        "full_five_family_completion_claim_authorized": False,
        "evidence_boundary": (
            "Matched public-source behavior calibration only. Track differences are descriptive "
            "diagnostics, not additive mechanism components. No internal state or intervention was collected."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, sort_keys=True))
