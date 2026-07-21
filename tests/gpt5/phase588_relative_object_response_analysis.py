#!/usr/bin/env python3
"""Re-read Phase587 open scores as within-continuation object responses.

This is an explicitly post-hoc diagnostic.  It asks whether changing the object
orders the score of the *same* continuation correctly, so fixed continuation
length and global phrase preference cannot determine the result.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase587_counterbalanced_continuation_observer as observer  # noqa: E402
import phase587_counterbalanced_continuation_protocol as source  # noqa: E402


PHASE = "Phase588"
OUT_DIR = ROOT / "tests/gpt5/result/phase588_relative_object_response"
PROTOCOL_PATH = OUT_DIR / "phase588_relative_object_response_protocol.json"
DECISION_PATH = OUT_DIR / "phase588_relative_object_response_decision.json"

MIN_MEAN_SURFACE_AUC = 0.85
MIN_SURFACE_AUC = 0.80
MIN_QUALIFIED_SURFACES = 6


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairwise_auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        raise ValueError("Both positive and negative score sets are required")
    wins = 0.0
    for left in positive:
        for right in negative:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def score_repeat_audit(rows: list[dict[str, Any]]) -> float:
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        for candidate, score in row["candidate_mean_logprobs"].items():
            by_key[(row["case_id"], candidate)][row["execution_repeat"]] = float(score)
    deltas = []
    for repeats in by_key.values():
        if set(repeats) != set(source.NOOP_REPEATS):
            raise RuntimeError("Phase587 repeat ledger is incomplete")
        deltas.append(abs(repeats["score1"] - repeats["score2"]))
    return max(deltas, default=0.0)


def analyze_unit(
    rows: list[dict[str, Any]], split: str, relation: str
) -> dict[str, Any]:
    unit = [
        row
        for row in rows
        if row["execution_repeat"] == "score1"
        and row["split"] == split
        and row["relation"] == relation
    ]
    candidates = tuple(source.CONTINUATIONS[relation])
    surfaces = sorted({int(row["surface_id"]) for row in unit})
    by_candidate: dict[str, Any] = {}
    for candidate in candidates:
        surface_auc: dict[str, float] = {}
        positive_count = 0
        negative_count = 0
        for surface in surfaces:
            surface_rows = [row for row in unit if int(row["surface_id"]) == surface]
            positive = [
                float(row["candidate_mean_logprobs"][candidate])
                for row in surface_rows
                if row["target_continuation_class"] == candidate
            ]
            negative = [
                float(row["candidate_mean_logprobs"][candidate])
                for row in surface_rows
                if row["target_continuation_class"] != candidate
            ]
            positive_count += len(positive)
            negative_count += len(negative)
            surface_auc[str(surface)] = pairwise_auc(positive, negative)
        values = list(surface_auc.values())
        by_candidate[candidate] = {
            "positive_object_surface_count": positive_count,
            "negative_object_surface_count": negative_count,
            "surface_auc": surface_auc,
            "mean_surface_auc": mean(values),
            "minimum_surface_auc": min(values),
            "maximum_surface_auc": max(values),
            "qualified_surface_count": sum(value >= MIN_SURFACE_AUC for value in values),
            "diagnostic_pass": bool(
                mean(values) >= MIN_MEAN_SURFACE_AUC
                and sum(value >= MIN_SURFACE_AUC for value in values)
                >= MIN_QUALIFIED_SURFACES
            ),
        }
    return {
        "case_count": len(unit),
        "surface_count": len(surfaces),
        "candidate_metrics": by_candidate,
        "all_candidate_diagnostic_pass": all(
            value["diagnostic_pass"] for value in by_candidate.values()
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protocol = {
        "schema_version": "phase588_relative_object_response_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "source_phase": source.PHASE,
        "source_open_artifacts_only": True,
        "analysis_status": "post_hoc_exploratory_diagnostic",
        "score_unit": "same continuation compared across counterbalanced objects",
        "pairwise_rule": "positive object score > negative object score; ties count 0.5",
        "minimum_mean_surface_auc": MIN_MEAN_SURFACE_AUC,
        "minimum_surface_auc": MIN_SURFACE_AUC,
        "minimum_qualified_surfaces": MIN_QUALIFIED_SURFACES,
        "candidate_length_cannot_change_within_pair": True,
        "fixed_continuation_prior_cancels_within_pair": True,
        "sealed_split_read": False,
        "may_authorize_hidden_capture": False,
        "may_authorize_causal_intervention": False,
    }
    write_json(PROTOCOL_PATH, protocol)

    model_results: dict[str, Any] = {}
    for model in source.MODELS:
        paths = observer.paths(model)
        rows = read_jsonl_gz(paths["rows"])
        if any(row["sealed"] for row in rows):
            raise RuntimeError(f"Phase588 read sealed row for {model}")
        if source.sha256_file(paths["rows"]) != json.loads(
            paths["summary"].read_text(encoding="utf-8")
        )["rows_sha256"]:
            raise RuntimeError(f"Phase587 source drift for {model}")
        units = {
            f"{split}:{relation}": analyze_unit(rows, split, relation)
            for split in source.OPEN_SPLITS
            for relation in source.RELATIONS
        }
        model_results[model] = {
            "source_row_count": len(rows),
            "maximum_repeat_score_delta": score_repeat_audit(rows),
            "units": units,
            "relation_all_open_split_diagnostic_pass": {
                relation: all(
                    units[f"{split}:{relation}"]["all_candidate_diagnostic_pass"]
                    for split in source.OPEN_SPLITS
                )
                for relation in source.RELATIONS
            },
        }

    payload = {
        "schema_version": "phase588_relative_object_response_decision.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "status": "complete_post_hoc_open_artifact_diagnostic",
        "model_results": model_results,
        "cross_model_relation_diagnostic_pass": {
            relation: all(
                model_results[model]["relation_all_open_split_diagnostic_pass"][relation]
                for model in source.MODELS
            )
            for relation in source.RELATIONS
        },
        "sealed_split_read": False,
        "interpretation_boundary": {
            "tests_object_conditional_ordering_not_absolute_answer": True,
            "does_not_observe_hidden_state": True,
            "does_not_identify_encoding_mechanism": True,
            "post_hoc_cannot_authorize_hidden_capture": True,
            "post_hoc_cannot_authorize_causal_intervention": True,
            "sealed_split_read": False,
        },
    }
    write_json(DECISION_PATH, payload)
    print(
        json.dumps(
            {
                "cross_model_relation_diagnostic_pass": payload[
                    "cross_model_relation_diagnostic_pass"
                ],
                "relation_pass_by_model": {
                    model: result["relation_all_open_split_diagnostic_pass"]
                    for model, result in model_results.items()
                },
                "sealed_split_read": False,
                "hidden_capture_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
