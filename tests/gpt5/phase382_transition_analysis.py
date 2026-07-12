#!/usr/bin/env python3
"""Compare frozen transition-update and static-state profiles offline."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
P381 = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
OUT = ROOT / "tests/gpt5/result/phase382_transition_event_audit"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("relation_binding", "entity_recency", "target_vs_wrong")
EFFECTS = ("content", "operation", "interaction")
ROLES = ("source", "query", "current")
SOURCES = ("transition_update", "static_layer_input")


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


def factorial(values: dict[str, torch.Tensor], effect: str) -> torch.Tensor:
    a, b, c, d = (values[key].float() for key in "ABCD")
    if effect == "content":
        return 0.5 * ((c - a) + (d - b))
    if effect == "operation":
        return 0.5 * ((a - b) + (c - d))
    if effect == "interaction":
        return 0.5 * ((a - b) - (c - d))
    raise KeyError(effect)


def signed_alignment(local: torch.Tensor, terminal: torch.Tensor) -> tuple[float, float]:
    local = local.float().flatten()
    terminal = terminal.float().flatten()
    local_norm = float(torch.linalg.vector_norm(local).item())
    terminal_norm = float(torch.linalg.vector_norm(terminal).item())
    if local_norm <= 1e-12 or terminal_norm <= 1e-12:
        return 0.0, 0.0
    cosine = float(torch.dot(local, terminal).item() / (local_norm * terminal_norm))
    ratio = min(1.0, local_norm / terminal_norm)
    return ratio * cosine, ratio


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / (left_norm * right_norm)


def main() -> None:
    protocol = read_json(OUT / "phase382_transition_protocol.json")
    split_by_group = {
        group: split
        for mechanism in MECHANISMS
        for split, groups in protocol["frozen_group_splits"][mechanism].items()
        for group in groups
    }
    cases = read_jsonl(P381 / "private/phase381_qualified_trace_cases.jsonl")
    case_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    mechanism_by_group: dict[str, str] = {}
    for case in cases:
        group = case["anonymous_parallel_group_id"]
        if group not in split_by_group:
            continue
        letter = case["contrast_condition"][0]
        case_groups[(case["private_execution_model"], group)][letter] = case
        mechanism_by_group[group] = case["mechanism_id"]
    event_rows: list[dict[str, Any]] = []
    accumulators: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (model, group), condition_cases in sorted(case_groups.items()):
        if set(condition_cases) != set("ABCD"):
            raise RuntimeError(f"Incomplete Phase382 group: {model}/{group}")
        payloads = {
            letter: torch.load(
                P381
                / "trace/private/models"
                / model
                / "cases"
                / f"{case['blind_case_id']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for letter, case in condition_cases.items()
        }
        layer_count = int(payloads["A"]["vectors"].shape[0])
        updates = {
            letter: payload["vectors"][:, 3].float()
            - payload["vectors"][:, 0].float()
            for letter, payload in payloads.items()
        }
        static_inputs = {
            letter: payload["vectors"][:, 0].float()
            for letter, payload in payloads.items()
        }
        terminals = {
            letter: payload["vectors"][-1, 3, 2].float()
            for letter, payload in payloads.items()
        }
        mechanism = mechanism_by_group[group]
        split = split_by_group[group]
        for effect in EFFECTS:
            terminal_effect = factorial(terminals, effect)
            source_effects = {
                "transition_update": factorial(updates, effect),
                "static_layer_input": factorial(static_inputs, effect),
            }
            for layer_index in range(layer_count):
                depth_bin = min(4, int(layer_index * 5 / layer_count))
                for role_index, role in enumerate(ROLES):
                    scores = {}
                    fractions = {}
                    for source in SOURCES:
                        score, fraction = signed_alignment(
                            source_effects[source][layer_index, role_index],
                            terminal_effect,
                        )
                        scores[source] = score
                        fractions[source] = fraction
                        accumulators[
                            (model, mechanism, split, effect, source, depth_bin, role)
                        ].append(score)
                    event_rows.append(
                        {
                            "schema_version": "55.1.0",
                            "phase_id": "Phase382-TransitionAnalysis",
                            "model": model,
                            "mechanism_id": mechanism,
                            "anonymous_parallel_group_id": group,
                            "split": split,
                            "effect_axis": effect,
                            "layer_index": layer_index,
                            "relative_depth": layer_index / max(layer_count - 1, 1),
                            "depth_bin": depth_bin,
                            "position_role": role,
                            "transition_signed_alignment": scores["transition_update"],
                            "static_signed_alignment": scores["static_layer_input"],
                            "transition_norm_fraction": fractions["transition_update"],
                            "static_norm_fraction": fractions["static_layer_input"],
                        }
                    )
    profiles: dict[tuple[str, str, str, str, str], list[float]] = {}
    profile_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for mechanism in MECHANISMS:
            for split in ("offline_discovery", "offline_validation"):
                for effect in EFFECTS:
                    for source in SOURCES:
                        values = [
                            mean(
                                accumulators[
                                    (
                                        model,
                                        mechanism,
                                        split,
                                        effect,
                                        source,
                                        depth_bin,
                                        role,
                                    )
                                ]
                            )
                            for depth_bin in range(5)
                            for role in ROLES
                        ]
                        key = (model, mechanism, split, effect, source)
                        profiles[key] = values
                        profile_rows.append(
                            {
                                "schema_version": "55.1.0",
                                "phase_id": "Phase382-TransitionAnalysis",
                                "model": model,
                                "mechanism_id": mechanism,
                                "split": split,
                                "effect_axis": effect,
                                "profile_source": source,
                                "profile": values,
                            }
                        )
    residuals: dict[tuple[str, str, str, str, str], list[float]] = {}
    residual_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for split in ("offline_discovery", "offline_validation"):
            for effect in EFFECTS:
                for source in SOURCES:
                    backbone = [
                        mean(
                            profiles[(model, mechanism, split, effect, source)][index]
                            for mechanism in MECHANISMS
                        )
                        for index in range(15)
                    ]
                    for mechanism in MECHANISMS:
                        profile = profiles[(model, mechanism, split, effect, source)]
                        residual = [value - base for value, base in zip(profile, backbone, strict=True)]
                        key = (model, mechanism, split, effect, source)
                        residuals[key] = residual
                        residual_rows.append(
                            {
                                "schema_version": "55.1.0",
                                "phase_id": "Phase382-TransitionAnalysis",
                                "model": model,
                                "mechanism_id": mechanism,
                                "split": split,
                                "effect_axis": effect,
                                "profile_source": source,
                                "common_backbone": backbone,
                                "function_residual": residual,
                            }
                        )
    replication_rows: list[dict[str, Any]] = []
    for source in SOURCES:
        for model in MODELS:
            for mechanism in MECHANISMS:
                for effect in EFFECTS:
                    discovery = residuals[
                        (model, mechanism, "offline_discovery", effect, source)
                    ]
                    own = cosine(
                        discovery,
                        residuals[(model, mechanism, "offline_validation", effect, source)],
                    )
                    wrong = {
                        candidate: cosine(
                            discovery,
                            residuals[(model, candidate, "offline_validation", effect, source)],
                        )
                        for candidate in MECHANISMS
                        if candidate != mechanism
                    }
                    replication_rows.append(
                        {
                            "schema_version": "55.1.0",
                            "phase_id": "Phase382-TransitionAnalysis",
                            "profile_source": source,
                            "model": model,
                            "mechanism_id": mechanism,
                            "effect_axis": effect,
                            "own_discovery_validation_cosine": own,
                            "wrong_mechanism_validation_cosines": wrong,
                            "own_profile_wins_without_threshold": own > max(wrong.values()),
                        }
                    )
    crossmodel_rows: list[dict[str, Any]] = []
    pairs = (("qwen3", "glm4"), ("qwen3", "deepseek7b"), ("glm4", "deepseek7b"))
    for source in SOURCES:
        for mechanism in MECHANISMS:
            for effect in EFFECTS:
                for left, right in pairs:
                    crossmodel_rows.append(
                        {
                            "schema_version": "55.1.0",
                            "phase_id": "Phase382-TransitionAnalysis",
                            "profile_source": source,
                            "mechanism_id": mechanism,
                            "effect_axis": effect,
                            "left_model": left,
                            "right_model": right,
                            "heterogeneous_pair": "glm4" in {left, right},
                            "validation_residual_cosine": cosine(
                                residuals[
                                    (left, mechanism, "offline_validation", effect, source)
                                ],
                                residuals[
                                    (right, mechanism, "offline_validation", effect, source)
                                ],
                            ),
                        }
                    )
    metrics = {}
    for source in SOURCES:
        source_replication = [
            row for row in replication_rows if row["profile_source"] == source
        ]
        source_crossmodel = [
            row
            for row in crossmodel_rows
            if row["profile_source"] == source and row["heterogeneous_pair"]
        ]
        metrics[source] = {
            "own_profile_win_count": sum(
                row["own_profile_wins_without_threshold"] for row in source_replication
            ),
            "comparison_count": len(source_replication),
            "within_mechanism_cosine_median": median(
                row["own_discovery_validation_cosine"] for row in source_replication
            ),
            "within_mechanism_cosine_mean": mean(
                row["own_discovery_validation_cosine"] for row in source_replication
            ),
            "heterogeneous_crossmodel_cosine_median": median(
                row["validation_residual_cosine"] for row in source_crossmodel
            ),
            "heterogeneous_crossmodel_cosine_mean": mean(
                row["validation_residual_cosine"] for row in source_crossmodel
            ),
        }
    transition = metrics["transition_update"]
    static = metrics["static_layer_input"]
    gate_vector = {
        "own_win_count_improved": transition["own_profile_win_count"]
        > static["own_profile_win_count"],
        "within_mechanism_median_improved": transition[
            "within_mechanism_cosine_median"
        ]
        > static["within_mechanism_cosine_median"],
        "heterogeneous_crossmodel_median_improved": transition[
            "heterogeneous_crossmodel_cosine_median"
        ]
        > static["heterogeneous_crossmodel_cosine_median"],
    }
    passed = all(gate_vector.values())
    write_jsonl(OUT / "phase382_transition_event_rows.jsonl", event_rows)
    write_jsonl(OUT / "phase382_profiles.jsonl", profile_rows)
    write_jsonl(OUT / "phase382_residual_profiles.jsonl", residual_rows)
    write_jsonl(OUT / "phase382_replication_rows.jsonl", replication_rows)
    write_jsonl(OUT / "phase382_crossmodel_rows.jsonl", crossmodel_rows)
    summary = {
        "schema_version": "55.1.0",
        "phase_id": "Phase382-TransitionAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": protocol["objective"],
        "denominator": {
            "replay_qualified_parallel_group_count": len(split_by_group),
            "model_group_count": len(case_groups),
            "transition_event_row_count": len(event_rows),
            "profile_count": len(profile_rows),
            "replication_comparison_count": len(replication_rows),
            "crossmodel_comparison_count": len(crossmodel_rows),
        },
        "results": {
            "metrics": metrics,
            "parameter_free_gate_vector": gate_vector,
            "transition_update_more_identifiable_than_static_state": passed,
            "causal_intervention_authorized": False,
            "complete_language_path_count": 0,
            "single_neuron_causal_count": 0,
            "language_encoding_mechanism_closed": False,
        },
        "claim_boundary": {
            "offline_profile_is_causal": False,
            "current_layer_update_is_complete_transition_operator": False,
            "phase381_data_is_independent_confirmation_of_phase382": False,
            "single_neuron_scan_opened": False,
        },
    }
    write_json(OUT / "phase382_transition_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
