#!/usr/bin/env python3
"""Decompose Phase425 failures without changing any frozen gate."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase425_role_exchange_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("early", "middle", "late")
FEATURES = (
    "role_source_contrast",
    "matched_position_contrast",
    "interface_history_source_contrast",
    "lexical_source_contrast",
    "formation_functional_specificity",
    "formation_role_dominance",
    "formation_specificity",
    "role_write_contrast",
    "matched_position_write_contrast",
    "interface_history_write_contrast",
    "lexical_write_contrast",
    "transport_functional_specificity",
    "transport_role_dominance",
    "transport_specificity",
    "competition_specificity",
    "role_delta_coherence",
    "transport_delta_coherence",
    "role_interaction_ratio",
    "transport_interaction_ratio",
    "source_write_coherence",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite Phase425 posthoc scalar: {value}")
    return round(float(value), 10)


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def group_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["replica_group_id"], row["depth_bin"])].append(row)
    output = []
    for (group_id, depth), values in grouped.items():
        output.append(
            {
                "replica_group_id": group_id,
                "model": values[0]["model"],
                "block_id": values[0]["block_id"],
                "split": values[0]["split"],
                "depth_bin": depth,
                **{
                    feature: median(row[feature] for row in values)
                    for feature in FEATURES
                },
            }
        )
    return output


def dominant_count(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(max(keys, key=lambda key: float(row[key])) for row in rows)
    return {key: counts[key] for key in keys}


def main() -> None:
    audits = read_jsonl(OUT / "phase425_open_block_audits.jsonl")
    conditions = read_jsonl(OUT / "phase425_registered_conditions_open.jsonl")
    condition_geometry = {}
    for model in MODELS:
        model_rows = [row for row in conditions if row["model"] == model]
        source_gaps = [
            int(row["prediction_position"]) - mean(row["source_positions"])
            for row in model_rows
        ]
        control_gaps = [
            int(row["prediction_position"])
            - mean(row["instruction_control_positions"])
            for row in model_rows
        ]
        condition_geometry[model] = {
            "condition_count": len(model_rows),
            "source_to_query_token_gap_median": median(source_gaps),
            "control_to_query_token_gap_median": median(control_gaps),
            "control_is_closer_fraction": mean(
                control < source for source, control in zip(source_gaps, control_gaps)
            ),
            "true_same_position_control": False,
        }

    decompositions = []
    for model in MODELS:
        rows = read_jsonl(
            OUT / "models" / model / "open" / "phase425_pair_layer_rows.jsonl"
        )
        grouped = group_feature_rows(rows)
        for audit in [row for row in audits if row["model"] == model]:
            block_rows = [row for row in grouped if row["block_id"] == audit["block_id"]]
            split_rows = {}
            for split in ("discovery", "calibration", "behavior_holdout"):
                depth_payload = {}
                for depth in DEPTHS:
                    selected = [
                        row
                        for row in block_rows
                        if row["split"] == split and row["depth_bin"] == depth
                    ]
                    depth_payload[depth] = {
                        "independent_group_count": len(selected),
                        "medians": {
                            feature: median(row[feature] for row in selected)
                            for feature in FEATURES
                        },
                        "source_contrast_dominance_counts": dominant_count(
                            selected,
                            (
                                "role_source_contrast",
                                "matched_position_contrast",
                                "interface_history_source_contrast",
                                "lexical_source_contrast",
                            ),
                        ),
                        "write_contrast_dominance_counts": dominant_count(
                            selected,
                            (
                                "role_write_contrast",
                                "matched_position_write_contrast",
                                "interface_history_write_contrast",
                                "lexical_write_contrast",
                            ),
                        ),
                    }
                split_rows[split] = depth_payload
            failed_gates = [
                name
                for name, passed in (
                    ("behavior", audit["behavior_gate_pass"]),
                    ("role_coherence", audit["role_coherence_gate_pass"]),
                    (
                        "formation",
                        audit["signals"]["formation"][
                            "calibration_and_behavior_gate_pass"
                        ],
                    ),
                    (
                        "transport",
                        audit["signals"]["transport"][
                            "calibration_and_behavior_gate_pass"
                        ],
                    ),
                    (
                        "competition",
                        audit["signals"]["competition"][
                            "calibration_and_behavior_gate_pass"
                        ],
                    ),
                    ("prediction", audit["prediction"]["gate_pass"]),
                    ("partial_order", audit["partial_order_gate_pass"]),
                )
                if not passed
            ]
            decompositions.append(
                {
                    "model": model,
                    "block_id": audit["block_id"],
                    "candidate": audit["candidate"],
                    "selected_depths": audit["selected_depths"],
                    "failed_frozen_gates": failed_gates,
                    "split_depth_decomposition": split_rows,
                }
            )

    failure_counts = Counter(
        failure
        for row in decompositions
        for failure in row["failed_frozen_gates"]
    )
    summary = {
        "schema_version": "phase425_role_exchange_posthoc.v1",
        "phase_id": "Phase425-PosthocFailureDecomposition",
        "created_at": now(),
        "posthoc": True,
        "changes_frozen_gate": False,
        "sealed_data_read": False,
        "causal_claim": False,
        "model_count": 3,
        "block_count": 4,
        "model_block_count": len(decompositions),
        "failed_gate_counts_over_12_model_blocks": dict(sorted(failure_counts.items())),
        "condition_geometry": condition_geometry,
        "decompositions": decompositions,
        "hard_limits": [
            "The registered control token is later and closer to the query than the focus token; it is not a true same-position control.",
            "Each block has only 12 independent lexical-replica groups per split despite many repeated conditions.",
            "The endpoint margin compares one branch token rather than a complete generated semantic event.",
            "All conclusions concern synthetic lookup and relative-clause tasks in three small models.",
        ],
        "strict_conclusion": (
            "Phase425 closes the current strict role-dominant formation-transport-prediction path. "
            "It does not establish absence of functional role states or transport because the "
            "position control is structurally unmatched and the effective predictive sample is small."
        ),
    }
    write_json(OUT / "phase425_posthoc_failure_decomposition.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "decompositions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
