#!/usr/bin/env python3
"""Decompose Phase427 open failures without changing any registered gate."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase427_behavior_analysis import behavior_gate, summarize_route  # noqa: E402
from phase427_dual_route_protocol import (  # noqa: E402
    BLOCKS,
    MODELS,
    OUT,
    SCHEMA_VERSION,
    SCORABLE_CANDIDATE_ROUTES,
)


PHASE_ID = "Phase427-DualRouteBehaviorPosthocAudit"
GATE_SPLITS = ("behavior_calibration", "behavior_holdout")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def metric_shortfall(
    summary: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, float]:
    return {
        "group_count": float(summary["independent_group_count"])
        - float(thresholds["groups_per_block_split"]),
        "teacher_correct": float(summary["teacher_sequence_correct_fraction"])
        - float(thresholds["teacher_sequence_correct_fraction_min"]),
        "teacher_margin": float(summary["teacher_sequence_margin_median"])
        - float(thresholds["teacher_sequence_margin_median_min"]),
        "natural_target": float(summary["natural_target_first_fraction"])
        - float(thresholds["natural_target_first_fraction_min"]),
        "natural_opposite": float(thresholds["natural_opposite_first_fraction_max"])
        - float(summary["natural_opposite_first_fraction"]),
        "natural_revision": float(thresholds["natural_revision_fraction_max"])
        - float(summary["natural_revision_fraction"]),
        "natural_boundary": float(summary["natural_boundary_fraction"])
        - float(thresholds["natural_boundary_fraction_min"]),
        "natural_stop": float(summary["natural_stop_fraction"])
        - float(thresholds["natural_stop_fraction_min"]),
        "natural_censoring": float(thresholds["natural_censoring_fraction_max"])
        - float(summary["natural_censoring_fraction"]),
    }


def main() -> None:
    protocol = read_json(OUT / "phase427_protocol.json")
    gate = read_json(OUT / "phase427_open_gate_freeze.json")
    if gate["sealed_rows_read"] or gate["sealed_behavior_unlock"]:
        raise RuntimeError("Posthoc audit is defined for the closed open gate only")
    thresholds = protocol["registered_thresholds"]
    summaries = read_jsonl(OUT / "phase427_open_route_summaries.jsonl")
    lookup = {
        (row["model"], row["block_id"], row["split"], row["route_mode"]): row
        for row in summaries
    }
    candidates = [block for block in BLOCKS if block["candidate"]]
    failure_rows = []
    candidate_failure_counts: Counter[str] = Counter()
    control_failure_counts: Counter[str] = Counter()
    route_qualified_models: dict[str, list[str]] = defaultdict(list)
    for model in MODELS:
        for block in candidates:
            for route in SCORABLE_CANDIDATE_ROUTES:
                paired_splits = []
                split_rows = []
                for split in GATE_SPLITS:
                    candidate = lookup[(model, block["block_id"], split, route)]
                    control = lookup[
                        (model, block["matched_control_block_id"], split, route)
                    ]
                    candidate_gate = behavior_gate(candidate, thresholds)
                    control_gate = behavior_gate(control, thresholds)
                    candidate_failed = [
                        key for key, value in candidate_gate["checks"].items() if not value
                    ]
                    control_failed = [
                        key for key, value in control_gate["checks"].items() if not value
                    ]
                    candidate_failure_counts.update(candidate_failed)
                    control_failure_counts.update(control_failed)
                    paired = bool(
                        candidate_gate["gate_pass"] and control_gate["gate_pass"]
                    )
                    paired_splits.append(paired)
                    split_rows.append(
                        {
                            "split": split,
                            "candidate_failed_checks": candidate_failed,
                            "control_failed_checks": control_failed,
                            "candidate_shortfall": metric_shortfall(candidate, thresholds),
                            "control_shortfall": metric_shortfall(control, thresholds),
                            "paired_gate_pass": paired,
                        }
                    )
                qualified = all(paired_splits)
                key = f"{block['block_id']}::{route}"
                if qualified:
                    route_qualified_models[key].append(model)
                failure_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "model": model,
                        "block_id": block["block_id"],
                        "matched_control_block_id": block["matched_control_block_id"],
                        "route_mode": route,
                        "split_audits": split_rows,
                        "qualified_open": qualified,
                        "posthoc_only": True,
                        "registered_gate_changed": False,
                    }
                )

    raw_rows = []
    for model in MODELS:
        raw_rows.extend(
            read_jsonl(
                OUT / "models" / model / "open" / "phase427_behavior_rows.jsonl"
            )
        )
    factor_groups: dict[
        tuple[str, str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in raw_rows:
        if (
            row["candidate"]
            and row["split"] in GATE_SPLITS
            and row["route_mode"] in SCORABLE_CANDIDATE_ROUTES
        ):
            factor_groups[
                (
                    row["model"],
                    row["block_id"],
                    row["split"],
                    row["route_mode"],
                    row["interface"],
                    row["history"],
                )
            ].append(row)
    factor_rows = []
    for key, values in sorted(factor_groups.items()):
        model, block_id, split, route, interface, history = key
        factor_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "model": model,
                "block_id": block_id,
                "split": split,
                "route_mode": route,
                "interface": interface,
                "history": history,
                **summarize_route(values),
                "descriptive_only": True,
                "registered_gate_changed": False,
            }
        )

    holdout_routes = {
        model: {
            block["block_id"]: {
                route: lookup[(model, block["block_id"], "behavior_holdout", route)]
                for route in ("none", "source_only", "query_only", "consistent", "conflict")
            }
            for block in candidates
        }
        for model in MODELS
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "open_gate_remains_closed": True,
        "sealed_rows_read": False,
        "physical_hooks_run": False,
        "threshold_prompt_sample_or_window_changed": False,
        "candidate_failure_count_by_check": dict(sorted(candidate_failure_counts.items())),
        "control_failure_count_by_check": dict(sorted(control_failure_counts.items())),
        "qualified_models_by_block_route": dict(sorted(route_qualified_models.items())),
        "cross_model_qualified_block_route_count": sum(
            len(models) >= int(thresholds["cross_model_replication_min"])
            for models in route_qualified_models.values()
        ),
        "qwen_language_action_observation": {
            "consistent_calibration_teacher_correct": lookup[("qwen3", "language_action_dual_route_candidate", "behavior_calibration", "consistent")]["teacher_sequence_correct_fraction"],
            "consistent_calibration_natural_target": lookup[("qwen3", "language_action_dual_route_candidate", "behavior_calibration", "consistent")]["natural_target_first_fraction"],
            "consistent_calibration_natural_opposite": lookup[("qwen3", "language_action_dual_route_candidate", "behavior_calibration", "consistent")]["natural_opposite_first_fraction"],
            "consistent_holdout_teacher_correct": lookup[("qwen3", "language_action_dual_route_candidate", "behavior_holdout", "consistent")]["teacher_sequence_correct_fraction"],
            "consistent_holdout_natural_target": lookup[("qwen3", "language_action_dual_route_candidate", "behavior_holdout", "consistent")]["natural_target_first_fraction"],
            "consistent_holdout_natural_opposite": lookup[("qwen3", "language_action_dual_route_candidate", "behavior_holdout", "consistent")]["natural_opposite_first_fraction"],
            "interpretation": "model-specific partial positive behavior that misses the frozen opposite-answer ceiling and has no cross-model replication",
        },
        "glm4_generation_observation": {
            "candidate_gate_rows_with_zero_stop": sum(
                row["model"] == "glm4"
                and row["candidate"]
                and row["split"] in GATE_SPLITS
                and row["route_mode"] in SCORABLE_CANDIDATE_ROUTES
                and float(row["natural_stop_fraction"]) == 0.0
                for row in summaries
            ),
            "candidate_gate_rows_with_full_censoring": sum(
                row["model"] == "glm4"
                and row["candidate"]
                and row["split"] in GATE_SPLITS
                and row["route_mode"] in SCORABLE_CANDIDATE_ROUTES
                and float(row["natural_censoring_fraction"]) == 1.0
                for row in summaries
            ),
            "interpretation": "teacher preferences do not become bounded natural responses under the frozen interface",
        },
        "deepseek_generation_observation": {
            "maximum_candidate_gate_teacher_correct": max(
                float(row["teacher_sequence_correct_fraction"])
                for row in summaries
                if row["model"] == "deepseek7b"
                and row["candidate"]
                and row["split"] in GATE_SPLITS
                and row["route_mode"] in SCORABLE_CANDIDATE_ROUTES
            ),
            "maximum_candidate_gate_natural_target": max(
                float(row["natural_target_first_fraction"])
                for row in summaries
                if row["model"] == "deepseek7b"
                and row["candidate"]
                and row["split"] in GATE_SPLITS
                and row["route_mode"] in SCORABLE_CANDIDATE_ROUTES
            ),
            "interpretation": "candidate discrimination and natural readout are both below the registered behavior floor",
        },
        "holdout_route_descriptions": holdout_routes,
        "hard_limits": [
            "The prompts use artificial routing labels and do not establish natural language roles.",
            "The syntax marked-anchor control has balanced sentence position but a fixed lexical target identity, so it is not a clean lexical-bias control.",
            "Interface and history cells contain 16 independent groups each and are descriptive posthoc strata only.",
            "Small-model failure does not prove that larger models lack dual routes.",
            "No physical state, transport operator, head, channel, neuron, or causal path was tested.",
        ],
        "stop_conclusion": (
            "The frozen dual-route behavior denominator does not authorize sealed behavior or physical mapping. "
            "Do not rescue Phase427 by changing thresholds, prompts, windows, ranks, or adding samples."
        ),
    }
    write_jsonl(OUT / "phase427_posthoc_failure_rows.jsonl", failure_rows)
    write_jsonl(OUT / "phase427_posthoc_factor_rows.jsonl", factor_rows)
    write_json(OUT / "phase427_posthoc_failure_decomposition.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
