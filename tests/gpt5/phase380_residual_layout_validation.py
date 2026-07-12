#!/usr/bin/env python3
"""Validate the preregistered backbone-residual layout metric on Phase380."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
P379 = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "entity_recency",
    "number_agreement",
    "relation_binding",
    "target_vs_wrong",
)
DEPTH_NAMES = ("early", "middle_early", "middle", "middle_late", "late")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / (left_norm * right_norm)


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def depth_bin(relative_depth: float) -> tuple[int, str]:
    index = min(4, max(0, int(relative_depth * 5)))
    return index, DEPTH_NAMES[index]


def contrast_axis(left: str, right: str) -> str:
    pair = frozenset((left[0], right[0]))
    if pair in {frozenset(("A", "C")), frozenset(("B", "D"))}:
        return "content_change_same_operation"
    if pair in {frozenset(("A", "B")), frozenset(("C", "D"))}:
        return "operation_change_same_content"
    if pair in {frozenset(("A", "D")), frozenset(("B", "C"))}:
        return "joint_content_operation_change"
    raise RuntimeError(f"Unknown pair {left}/{right}")


def cosine_to_terminal(vectors: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
    dot = torch.einsum("...h,h->...", vectors, terminal)
    denominator = torch.linalg.vector_norm(vectors, dim=-1) * torch.linalg.vector_norm(
        terminal
    )
    return torch.where(denominator > 1e-12, dot / denominator, torch.zeros_like(dot))


def profile_vectors(
    profiles: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], list[float]], list[tuple[str, str, int]]]:
    cells = sorted(
        {
            (row["component_type"], row["position_role"], row["depth_bin"])
            for row in profiles
        }
    )
    values = {
        (
            row["model"], row["mechanism_id"], row["contrast_axis"],
            row["component_type"], row["position_role"], row["depth_bin"],
        ): float(row["median_descriptive_layout_weight"])
        for row in profiles
    }
    keys = sorted(
        {(row["model"], row["mechanism_id"], row["contrast_axis"]) for row in profiles}
    )
    return {
        key: [
            values[(*key, component, role, depth)]
            for component, role, depth in cells
        ]
        for key in keys
    }, cells


def residualize(
    vectors: dict[tuple[str, str, str], list[float]]
) -> dict[tuple[str, str, str], list[float]]:
    result = {}
    axes = sorted({key[2] for key in vectors})
    for model in MODELS:
        for axis in axes:
            sources = [vectors[(model, mechanism, axis)] for mechanism in MECHANISMS]
            backbone = [mean(values) for values in zip(*sources, strict=True)]
            for mechanism in MECHANISMS:
                result[(model, mechanism, axis)] = [
                    value - base
                    for value, base in zip(vectors[(model, mechanism, axis)], backbone, strict=True)
                ]
    return result


def main() -> None:
    protocol = read_json(OUT / "phase380_protocol.json")
    metric = protocol["frozen_metric"]
    cases = read_jsonl(OUT / "private/phase380_qualified_trace_cases.jsonl")
    case_by_id = {row["blind_case_id"]: row for row in cases}
    metadata = {}
    for model in MODELS:
        for row in read_jsonl(OUT / "trace/models" / model / "phase380_trace_rows.jsonl"):
            metadata[row["blind_case_id"]] = row
    parallel_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        parallel_cases[case["anonymous_parallel_group_id"]].append(case)
    group_replay = {
        group: len(rows) == 12
        and all(metadata[row["blind_case_id"]]["baseline_replay_matches_observed_target_token"] for row in rows)
        for group, rows in parallel_cases.items()
    }
    replay_group_counts = Counter(
        rows[0]["mechanism_id"]
        for group, rows in parallel_cases.items()
        if group_replay[group]
    )
    event_rows = []
    buckets: dict[tuple[str, str, str, str, str, int], list[float]] = defaultdict(list)
    signed_buckets: dict[tuple[str, str, str, str, str, int], list[float]] = defaultdict(list)
    for model in MODELS:
        model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            if case["private_execution_model"] == model:
                model_groups[case["anonymous_parallel_group_id"]].append(case)
        for parallel, rows in sorted(model_groups.items()):
            rows.sort(key=lambda row: row["contrast_condition"])
            if len(rows) != 4:
                raise RuntimeError(f"Incomplete Phase380 model group {model}/{parallel}")
            payloads = {
                row["blind_case_id"]: torch.load(
                    OUT
                    / "trace/private/models"
                    / model
                    / "cases"
                    / f"{row['blind_case_id']}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for row in rows
            }
            for left, right in itertools.combinations(rows, 2):
                left_payload = payloads[left["blind_case_id"]]
                right_payload = payloads[right["blind_case_id"]]
                vectors = left_payload["vectors"].float() - right_payload["vectors"].float()
                terminal = vectors[-1, -1, -1]
                terminal_norm = float(torch.linalg.vector_norm(terminal).item())
                norms = torch.linalg.vector_norm(vectors, dim=-1)
                cosines = cosine_to_terminal(vectors, terminal)
                axis = contrast_axis(left["contrast_condition"], right["contrast_condition"])
                for layer in range(vectors.shape[0]):
                    depth_index, depth_name = depth_bin(layer / max(vectors.shape[0] - 1, 1))
                    for component_index, component in enumerate(left_payload["component_names"]):
                        for role_index, role in enumerate(left_payload["role_names"]):
                            norm_ratio = min(
                                1.0,
                                float(norms[layer, component_index, role_index].item())
                                / max(terminal_norm, 1e-12),
                            )
                            signed_cosine = float(
                                cosines[layer, component_index, role_index].item()
                            )
                            weight = norm_ratio * abs(signed_cosine)
                            key = (
                                model,
                                left["mechanism_id"],
                                axis,
                                component,
                                role,
                                depth_index,
                            )
                            if group_replay[parallel]:
                                buckets[key].append(weight)
                                signed_buckets[key].append(signed_cosine)
                            event_rows.append(
                                {
                                    "schema_version": "53.7.0",
                                    "phase_id": "Phase380-ResidualValidation",
                                    "model": model,
                                    "anonymous_parallel_group_id": parallel,
                                    "left_case_id": left["blind_case_id"],
                                    "right_case_id": right["blind_case_id"],
                                    "mechanism_id": left["mechanism_id"],
                                    "contrast_axis": axis,
                                    "layer": layer,
                                    "relative_depth": layer / max(vectors.shape[0] - 1, 1),
                                    "depth_bin": depth_index,
                                    "depth_name": depth_name,
                                    "component_type": component,
                                    "position_role": role,
                                    "descriptive_layout_weight": weight,
                                    "signed_terminal_cosine": signed_cosine,
                                    "group_replay_qualified": group_replay[parallel],
                                    "candidate_selected": False,
                                }
                            )
    event_path = OUT / "validation/private/phase380_all_event_rows.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(event_rows), event_path, compression="zstd")
    profiles = []
    for key, values in sorted(buckets.items()):
        model, mechanism, axis, component, role, depth_index = key
        signed = signed_buckets[key]
        profiles.append(
            {
                "schema_version": "53.7.0",
                "phase_id": "Phase380-ResidualValidation",
                "model": model,
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "component_type": component,
                "position_role": role,
                "depth_bin": depth_index,
                "depth_name": DEPTH_NAMES[depth_index],
                "event_row_count": len(values),
                "mean_descriptive_layout_weight": mean(values),
                "median_descriptive_layout_weight": median(values),
                "mean_signed_terminal_cosine": mean(signed),
                "median_signed_terminal_cosine": median(signed),
                "causal_reuse_claimed": False,
            }
        )
    profile_path = OUT / "validation/phase380_function_profiles.jsonl"
    write_jsonl(profile_path, profiles)
    validation_vectors, cells = profile_vectors(profiles)
    validation_residual = residualize(validation_vectors)
    discovery_profiles = read_jsonl(
        P379 / "fresh_discovery/phase379_function_profiles.jsonl"
    )
    discovery_vectors, discovery_cells = profile_vectors(discovery_profiles)
    if cells != discovery_cells:
        raise RuntimeError("Phase379/380 profile cells differ")
    discovery_residual = residualize(discovery_vectors)
    comparison_rows = []
    for key in sorted(validation_vectors):
        raw_norm = vector_norm(validation_vectors[key])
        residual_norm = vector_norm(validation_residual[key])
        comparison_rows.append(
            {
                "schema_version": "53.7.0",
                "phase_id": "Phase380-ResidualValidation",
                "model": key[0],
                "mechanism_id": key[1],
                "contrast_axis": key[2],
                "discovery_validation_residual_cosine": cosine(
                    discovery_residual[key], validation_residual[key]
                ),
                "validation_residual_norm_fraction": residual_norm
                / max(raw_norm, 1e-12),
                "passes_profile_gate": cosine(
                    discovery_residual[key], validation_residual[key]
                )
                >= float(metric["discovery_validation_profile_cosine_gate"]),
                "passes_residual_size_gate": residual_norm / max(raw_norm, 1e-12)
                >= float(metric["minimum_residual_norm_fraction"]),
            }
        )
    crossmodel_rows = []
    axes = sorted({key[2] for key in validation_residual})
    for mechanism in MECHANISMS:
        for axis in axes:
            for left_index, left in enumerate(MODELS):
                for right in MODELS[left_index + 1 :]:
                    value = cosine(
                        validation_residual[(left, mechanism, axis)],
                        validation_residual[(right, mechanism, axis)],
                    )
                    crossmodel_rows.append(
                        {
                            "schema_version": "53.7.0",
                            "phase_id": "Phase380-ResidualValidation",
                            "mechanism_id": mechanism,
                            "contrast_axis": axis,
                            "left_model": left,
                            "right_model": right,
                            "residual_profile_cosine": value,
                            "heterogeneous_pair": "glm4" in {left, right},
                            "passes_crossmodel_gate": value
                            >= float(metric["heterogeneous_crossmodel_residual_cosine_gate"]),
                        }
                    )
    stable_objects = []
    for mechanism in MECHANISMS:
        for axis in axes:
            model_passes = {
                row["model"]
                for row in comparison_rows
                if row["mechanism_id"] == mechanism
                and row["contrast_axis"] == axis
                and row["passes_profile_gate"]
                and row["passes_residual_size_gate"]
            }
            pair_passes = [
                row
                for row in crossmodel_rows
                if row["mechanism_id"] == mechanism
                and row["contrast_axis"] == axis
                and row["heterogeneous_pair"]
                and row["passes_crossmodel_gate"]
                and {row["left_model"], row["right_model"]} <= model_passes
            ]
            stable_objects.append(
                {
                    "mechanism_id": mechanism,
                    "contrast_axis": axis,
                    "individually_stable_models": sorted(model_passes),
                    "heterogeneous_passing_pairs": [
                        [row["left_model"], row["right_model"]] for row in pair_passes
                    ],
                    "heterogeneous_level2_residual_pass": bool(pair_passes),
                    "level3_individual_replication": model_passes == set(MODELS),
                    "causal_reuse_established": False,
                }
            )
    comparison_path = OUT / "validation/phase380_residual_replication.jsonl"
    crossmodel_path = OUT / "validation/phase380_crossmodel_residuals.jsonl"
    stable_path = OUT / "validation/phase380_stable_layout_objects.jsonl"
    write_jsonl(comparison_path, comparison_rows)
    write_jsonl(crossmodel_path, crossmodel_rows)
    write_jsonl(stable_path, stable_objects)
    authorized = [row for row in stable_objects if row["heterogeneous_level2_residual_pass"]]
    summary = {
        "schema_version": "53.7.0",
        "phase_id": "Phase380-ResidualValidation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "independently_validate_function_specific_layout_after_subtracting_the_common_backbone",
        "denominator": {
            "registered_parallel_group_count": len(parallel_cases),
            "replay_qualified_parallel_group_count": sum(group_replay.values()),
            "replay_qualified_groups_by_mechanism": dict(replay_group_counts),
            "registered_case_count": len(cases),
            "exact_event_vector_count": sum(
                read_json(OUT / "trace/models" / model / "complete.json")[
                    "exact_event_vector_count"
                ]
                for model in MODELS
            ),
            "all_pair_event_row_count": len(event_rows),
            "profile_cell_count": len(profiles),
            "profile_comparison_count": len(comparison_rows),
            "crossmodel_comparison_count": len(crossmodel_rows),
        },
        "quality": {
            "registered_replay_mismatch_group_count": sum(not value for value in group_replay.values()),
            "mismatch_groups_retained_in_quality_ledger": True,
            "mismatch_groups_used_for_profile_claims": False,
            "threshold_retuned": False,
            "top_k_used": False,
        },
        "results": {
            "passing_individual_profile_count": sum(
                row["passes_profile_gate"] and row["passes_residual_size_gate"]
                for row in comparison_rows
            ),
            "heterogeneous_level2_stable_object_count": len(authorized),
            "level3_individual_stable_object_count": sum(
                row["level3_individual_replication"] for row in stable_objects
            ),
            "stable_objects": authorized,
            "causal_scan_authorized": bool(authorized),
            "causal_reuse_established": False,
            "language_encoding_mechanism_closed": False,
        },
        "claim_boundary": {
            "stable_residual_profile_is_a_causal_path": False,
            "stable_residual_profile_is_a_single_neuron_region": False,
            "four_function_validation_completes_nine_family_atlas": False,
        },
    }
    write_json(OUT / "phase380_residual_validation_summary.json", summary)
    write_json(
        OUT / "phase380_causal_authorization.json",
        {
            "schema_version": "53.7.0",
            "phase_id": "Phase380-CausalAuthorization",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "authorization": {
                "run_registered_natural_boundary_causal_scan": bool(authorized),
                "open_prior_physical_holdout": False,
                "run_single_neuron_scan": False,
            },
            "stable_objects": authorized,
            "scan_contract": {
                "fixed_depth_bins": 5,
                "component_boundaries": [
                    "layer_input", "attention_output", "mlp_output", "layer_output"
                ],
                "position_roles": ["source", "query", "current"],
                "top_k_selection": False,
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
