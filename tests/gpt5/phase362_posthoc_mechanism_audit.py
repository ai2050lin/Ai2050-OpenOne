#!/usr/bin/env python3
"""Audit seven independent survivors by mechanism after blind freeze."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    audit = read_jsonl(OUT / "phase362_frozen_candidate_audit_rows.jsonl")
    survivors = {row["candidate_id"]: row for row in audit if row["b3_beats_all_alternatives_all_models"]}
    groups = read_jsonl(OUT / "phase362_candidate_group_errors.jsonl")
    execution = read_jsonl(OUT / "private" / "phase362_execution_cases.jsonl")
    group_labels = {}
    for row in execution:
        if row["phase362_split"] != "independent_calibration":
            continue
        key = (row["model"], row["phase362_group_id"])
        value = f"{row['family_id']}/{row['mechanism_id']}"
        if key in group_labels and group_labels[key] != value:
            raise RuntimeError(f"Conflicting group label: {key}")
        group_labels[key] = value

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in groups:
        if row["candidate_id"] not in survivors:
            continue
        mechanism = group_labels[(row["model"], row["anonymous_group_id"])]
        grouped[(row["candidate_id"], mechanism, row["model"])].append(
            float(row["b3_gain_over_strongest_alternative"])
        )
    rows = []
    mechanisms = sorted({key[1] for key in grouped})
    for candidate_id, candidate in survivors.items():
        support = {}
        for mechanism in mechanisms:
            model_gains = {
                model: sum(grouped[(candidate_id, mechanism, model)]) / len(grouped[(candidate_id, mechanism, model)])
                for model in MODELS
            }
            support[mechanism] = {
                "independent_group_count_per_model": 6,
                "model_mean_gains": {key: round(value, 9) for key, value in model_gains.items()},
                "all_three_models_positive": all(value > 0 for value in model_gains.values()),
            }
        positive = sorted(key for key, value in support.items() if value["all_three_models_positive"])
        rows.append({
            "candidate_id": candidate_id,
            "depth_bin": candidate["depth_bin"], "feature_id": candidate["feature_id"],
            "mechanism_support": support,
            "positive_mechanisms": positive, "positive_mechanism_count": len(positive),
            "universal_across_four_mechanisms": len(positive) == 4,
            "selective_next_layer_association": 0 < len(positive) < 4,
            "operation_specific_mechanism_proved": False,
        })
    universal = sum(row["universal_across_four_mechanisms"] for row in rows)
    selective = sum(row["selective_next_layer_association"] for row in rows)
    summary = {
        "schema_version": "39.3.0", "phase_id": "Phase362", "created_at": now(),
        "denominator": {
            "independent_next_layer_survivor_count": len(rows),
            "mechanism_count": len(mechanisms),
            "independent_groups_per_model_mechanism": 6,
        },
        "results": {
            "universal_backbone_survivor_count": universal,
            "selective_next_layer_association_count": selective,
            "unsupported_survivor_count": len(rows) - universal - selective,
            "temporal_predictive_survivor_count": 0,
            "competition_predictive_survivor_count": 0,
            "operation_specific_mechanism_count": 0,
        },
        "claim_boundary": {
            "labels_revealed_after_independent_survivor_freeze": True,
            "next_layer_selectivity_is_temporal_prediction": False,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
        },
        "next_decision": "do_not_open_physical_confirmation_missing_frozen_temporal_and_competition_predictors",
    }
    write_jsonl(OUT / "phase362_posthoc_survivor_rows.jsonl", rows)
    write_json(OUT / "phase362_posthoc_survivor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
