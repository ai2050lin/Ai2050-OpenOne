#!/usr/bin/env python3
"""Map frozen Phase379 blind profiles to functional reuse/difference axes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
LABELS = OUT / "private/phase379_label_key.jsonl"
SPLITS = ("fresh_discovery", "fresh_calibration")
DEPTH_NAMES = ("early", "middle_early", "middle", "middle_late", "late")
FUNCTION_DOMAINS = {
    "relation_binding": "relational_content_operation",
    "entity_recency": "direct_content_retrieval",
    "number_agreement": "grammar_constraint",
    "target_vs_wrong": "readout_competition",
}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    raise RuntimeError(f"Unknown contrast pair {left}/{right}")


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / (left_norm * right_norm)


def weighted_jaccard(left: list[float], right: list[float]) -> float:
    numerator = sum(min(a, b) for a, b in zip(left, right, strict=True))
    denominator = sum(max(a, b) for a, b in zip(left, right, strict=True))
    return numerator / denominator if denominator > 1e-12 else 0.0


def build_profiles(split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_by_case = {
        row["blind_case_id"]: row
        for row in read_jsonl(LABELS)
        if row["phase379_split"] == split
    }
    event_path = OUT / split / "private/phase379_blind_event_rows.parquet"
    rows = pq.read_table(event_path).to_pylist()
    buckets: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    mapped_rows: list[dict[str, Any]] = []
    for row in rows:
        left = label_by_case[row["left_case_id"]]
        right = label_by_case[row["right_case_id"]]
        if left["mechanism_id"] != right["mechanism_id"]:
            raise RuntimeError("Cross-mechanism pair appeared inside one group")
        axis = contrast_axis(left["contrast_condition"], right["contrast_condition"])
        depth_index, depth_name = depth_bin(float(row["relative_depth"]))
        norm_ratio = min(1.0, float(row["norm_ratio_to_terminal"]))
        persistence = abs(float(row["cosine_to_terminal_difference"]))
        descriptive_weight = norm_ratio * persistence
        mapped = {
            **row,
            "family_id": left["family_id"],
            "mechanism_id": left["mechanism_id"],
            "functional_domain": FUNCTION_DOMAINS[left["mechanism_id"]],
            "contrast_axis": axis,
            "depth_bin": depth_index,
            "depth_name": depth_name,
            "descriptive_layout_weight": descriptive_weight,
            "causal_reuse_claimed": False,
        }
        mapped_rows.append(mapped)
        key = (
            row["model"],
            left["mechanism_id"],
            axis,
            row["component_type"],
            row["position_role"],
            depth_index,
        )
        buckets[key].append(mapped)
    profiles = []
    for key, values in sorted(buckets.items()):
        model, mechanism, axis, component, role, depth_index = key
        weights = [float(row["descriptive_layout_weight"]) for row in values]
        cosines = [float(row["cosine_to_terminal_difference"]) for row in values]
        shares = [float(row["terminal_inner_product_share"]) for row in values]
        profiles.append(
            {
                "schema_version": "52.3.0",
                "phase_id": "Phase379-SemanticLayoutMapping",
                "split": split,
                "model": model,
                "mechanism_id": mechanism,
                "functional_domain": FUNCTION_DOMAINS[mechanism],
                "contrast_axis": axis,
                "component_type": component,
                "position_role": role,
                "depth_bin": depth_index,
                "depth_name": DEPTH_NAMES[depth_index],
                "event_row_count": len(values),
                "mean_descriptive_layout_weight": mean(weights),
                "median_descriptive_layout_weight": median(weights),
                "mean_signed_terminal_cosine": mean(cosines),
                "median_signed_terminal_cosine": median(cosines),
                "mean_terminal_inner_product_share": mean(shares),
                "causal_reuse_claimed": False,
            }
        )
    return profiles, mapped_rows


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
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["component_type"],
            row["position_role"],
            row["depth_bin"],
        ): float(row["median_descriptive_layout_weight"])
        for row in profiles
    }
    keys = sorted(
        {
            (row["model"], row["mechanism_id"], row["contrast_axis"])
            for row in profiles
        }
    )
    vectors = {
        key: [
            values.get((*key, component, role, depth), 0.0)
            for component, role, depth in cells
        ]
        for key in keys
    }
    return vectors, cells


def reuse_rows(
    vectors: dict[tuple[str, str, str], list[float]]
) -> list[dict[str, Any]]:
    result = []
    models = sorted({key[0] for key in vectors})
    axes = sorted({key[2] for key in vectors})
    mechanisms = sorted({key[1] for key in vectors})
    for model in models:
        for axis in axes:
            for left_index, left in enumerate(mechanisms):
                for right in mechanisms[left_index:]:
                    left_vector = vectors[(model, left, axis)]
                    right_vector = vectors[(model, right, axis)]
                    result.append(
                        {
                            "schema_version": "52.3.0",
                            "phase_id": "Phase379-SemanticLayoutMapping",
                            "model": model,
                            "contrast_axis": axis,
                            "left_mechanism": left,
                            "right_mechanism": right,
                            "descriptive_weighted_overlap": weighted_jaccard(
                                left_vector, right_vector
                            ),
                            "profile_cosine": cosine(left_vector, right_vector),
                            "causal_reuse_claimed": False,
                        }
                    )
    return result


def crossmodel_rows(
    vectors: dict[tuple[str, str, str], list[float]]
) -> list[dict[str, Any]]:
    result = []
    models = ("qwen3", "glm4", "deepseek7b")
    mechanisms = sorted({key[1] for key in vectors})
    axes = sorted({key[2] for key in vectors})
    for mechanism in mechanisms:
        for axis in axes:
            pair_values = []
            for left_index, left in enumerate(models):
                for right in models[left_index + 1 :]:
                    value = cosine(
                        vectors[(left, mechanism, axis)],
                        vectors[(right, mechanism, axis)],
                    )
                    pair_values.append(value)
                    result.append(
                        {
                            "schema_version": "52.3.0",
                            "phase_id": "Phase379-SemanticLayoutMapping",
                            "mechanism_id": mechanism,
                            "contrast_axis": axis,
                            "left_model": left,
                            "right_model": right,
                            "profile_cosine": value,
                            "heterogeneous_pair": "glm4" in {left, right},
                            "causal_path_claimed": False,
                        }
                    )
            result.append(
                {
                    "schema_version": "52.3.0",
                    "phase_id": "Phase379-SemanticLayoutMapping",
                    "mechanism_id": mechanism,
                    "contrast_axis": axis,
                    "left_model": "all",
                    "right_model": "all",
                    "profile_cosine": mean(pair_values),
                    "minimum_pair_profile_cosine": min(pair_values),
                    "heterogeneous_pair": True,
                    "causal_path_claimed": False,
                }
            )
    return result


def discovery() -> dict[str, Any]:
    blind_summary = read_json(
        OUT / "fresh_discovery/phase379_blind_layout_summary.json"
    )
    if not blind_summary["authorization"]["open_semantic_mapping"]:
        raise RuntimeError("Discovery semantics are not authorized")
    profiles, _mapped = build_profiles("fresh_discovery")
    vectors, cells = profile_vectors(profiles)
    reuse = reuse_rows(vectors)
    crossmodel = crossmodel_rows(vectors)
    profile_path = OUT / "fresh_discovery/phase379_function_profiles.jsonl"
    reuse_path = OUT / "fresh_discovery/phase379_reuse_matrix.jsonl"
    crossmodel_path = OUT / "fresh_discovery/phase379_crossmodel_profiles.jsonl"
    write_jsonl(profile_path, profiles)
    write_jsonl(reuse_path, reuse)
    write_jsonl(crossmodel_path, crossmodel)
    gate = 0.60
    aggregate = [
        row
        for row in crossmodel
        if row["left_model"] == "all"
    ]
    summary = {
        "schema_version": "52.3.0",
        "phase_id": "Phase379-DiscoveryMapping",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "map_label_free_full_layout_profiles_to_content_operation_grammar_and_readout_axes",
        "denominator": {
            "profile_cell_count": len(profiles),
            "profile_vector_width": len(cells),
            "reuse_matrix_row_count": len(reuse),
            "crossmodel_profile_row_count": len(crossmodel),
        },
        "crossmodel_description": {
            "aggregate_profile_count": len(aggregate),
            "aggregate_mean_profile_cosine": mean(
                row["profile_cosine"] for row in aggregate
            ),
            "aggregate_minimum_profile_cosine": min(
                row["minimum_pair_profile_cosine"] for row in aggregate
            ),
            "profiles_above_frozen_calibration_gate": sum(
                row["minimum_pair_profile_cosine"] >= gate for row in aggregate
            ),
            "frozen_calibration_gate": gate,
        },
        "claim_boundary": {
            "descriptive_layout_available": True,
            "causal_reuse_established": False,
            "upstream_language_path_established": False,
            "single_neuron_mechanism_established": False,
            "language_encoding_mechanism_closed": False,
        },
        "files": {
            "profiles": profile_path.name,
            "reuse_matrix": reuse_path.name,
            "crossmodel_profiles": crossmodel_path.name,
        },
    }
    write_json(OUT / "phase379_discovery_mapping_summary.json", summary)
    freeze = {
        "schema_version": "52.3.0",
        "phase_id": "Phase379-DiscoveryMappingFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_sha256": sha256(profile_path),
        "reuse_sha256": sha256(reuse_path),
        "crossmodel_sha256": sha256(crossmodel_path),
        "calibration_gate": {
            "minimum_profile_cosine": gate,
            "glm4_required": True,
            "retuning_allowed": False,
        },
        "authorization": {
            "open_calibration_exact_trace": True,
            "open_calibration_blind_extraction": True,
            "run_causal_intervention": False,
            "open_physical_holdout": False,
        },
    }
    write_json(OUT / "phase379_discovery_mapping_freeze.json", freeze)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def calibration() -> dict[str, Any]:
    freeze = read_json(OUT / "phase379_discovery_mapping_freeze.json")
    profiles, _mapped = build_profiles("fresh_calibration")
    calibration_vectors, cells = profile_vectors(profiles)
    discovery_profiles = read_jsonl(
        OUT / "fresh_discovery/phase379_function_profiles.jsonl"
    )
    discovery_vectors, discovery_cells = profile_vectors(discovery_profiles)
    if cells != discovery_cells:
        raise RuntimeError("Discovery/calibration profile cell mismatch")
    comparisons = []
    for key, calibration_vector in sorted(calibration_vectors.items()):
        value = cosine(discovery_vectors[key], calibration_vector)
        comparisons.append(
            {
                "schema_version": "52.4.0",
                "phase_id": "Phase379-Calibration",
                "model": key[0],
                "mechanism_id": key[1],
                "contrast_axis": key[2],
                "discovery_calibration_profile_cosine": value,
                "passes_frozen_gate": value
                >= freeze["calibration_gate"]["minimum_profile_cosine"],
                "causal_reuse_claimed": False,
            }
        )
    comparison_path = OUT / "fresh_calibration/phase379_profile_replication.jsonl"
    profile_path = OUT / "fresh_calibration/phase379_function_profiles.jsonl"
    write_jsonl(profile_path, profiles)
    write_jsonl(comparison_path, comparisons)
    by_object: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in comparisons:
        if row["passes_frozen_gate"]:
            by_object[(row["mechanism_id"], row["contrast_axis"])].add(row["model"])
    replicated = [
        {
            "mechanism_id": mechanism,
            "contrast_axis": axis,
            "models": sorted(models),
            "heterogeneous_level2": "glm4" in models
            and bool(models & {"qwen3", "deepseek7b"}),
            "level3": models == {"qwen3", "glm4", "deepseek7b"},
        }
        for (mechanism, axis), models in sorted(by_object.items())
    ]
    level2 = [row for row in replicated if row["heterogeneous_level2"]]
    summary = {
        "schema_version": "52.4.0",
        "phase_id": "Phase379-Calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "replicate_discovery_layout_profiles_without_threshold_retuning",
        "denominator": {
            "profile_cell_count": len(profiles),
            "profile_comparison_count": len(comparisons),
            "mechanism_axis_object_count": 12,
        },
        "results": {
            "passing_model_profile_count": sum(
                row["passes_frozen_gate"] for row in comparisons
            ),
            "heterogeneous_level2_object_count": len(level2),
            "level3_object_count": sum(row["level3"] for row in replicated),
            "replicated_objects": replicated,
            "causal_scan_authorized": bool(level2),
            "causal_reuse_established": False,
            "language_encoding_mechanism_closed": False,
        },
        "hard_limits": [
            "profile_similarity_is_descriptive_not_causal",
            "four_representative_functions_do_not_complete_the_nine_family_atlas",
            "relative_depth_component_role_profiles_do_not_identify_single_neurons",
            "current_models_are_small_and_qwen3_deepseek7b_are_architecture_related",
        ],
    }
    write_json(OUT / "phase379_calibration_summary.json", summary)
    authorization = {
        "schema_version": "52.4.0",
        "phase_id": "Phase379-CausalAuthorization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_profile_replication_sha256": sha256(comparison_path),
        "authorization": {
            "run_registered_natural_boundary_causal_scan": bool(level2),
            "open_physical_holdout": False,
            "run_single_neuron_scan": False,
        },
        "scope": {
            "all_fixed_depth_bins": True,
            "all_component_boundaries": True,
            "all_position_roles": True,
            "top_k_selection": False,
            "mechanism_axis_objects": [
                {
                    "mechanism_id": row["mechanism_id"],
                    "contrast_axis": row["contrast_axis"],
                }
                for row in level2
            ],
        },
    }
    write_json(OUT / "phase379_causal_authorization.json", authorization)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("discovery", "calibration"), required=True)
    args = parser.parse_args()
    if args.split == "discovery":
        discovery()
    else:
        calibration()
