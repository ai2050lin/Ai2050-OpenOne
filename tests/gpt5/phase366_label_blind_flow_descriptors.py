#!/usr/bin/env python3
"""Extract all role-source directed-flow descriptors without semantic or target labels."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation"
BUNDLE_ROOT = PHASE_ROOT / "dynamic_bundle_extraction"
OUT = PHASE_ROOT / "label_blind_flow_descriptors"
MODELS = ("qwen3", "glm4", "deepseek7b")
FEATURES = (
    "route_norm_over_attention",
    "route_alignment_to_attention",
    "route_signed_projection_to_attention",
    "route_alignment_to_output_change",
    "attention_norm_over_input",
    "mlp_norm_over_post_attention",
    "attention_alignment_to_mlp",
    "attention_mlp_balance",
    "output_change_norm_over_input",
    "input_alignment_to_output",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_vector(reference: dict[str, Any], cache: dict[str, Any]) -> torch.Tensor:
    relative_path = reference["relative_path"]
    if relative_path not in cache:
        try:
            cache[relative_path] = torch.load(
                PHASE_ROOT / relative_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise RuntimeError(f"Cannot load referenced tensor: {relative_path}") from error
    value: Any = cache[relative_path]
    for part in reference["slice"]:
        value = value[part]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Vector reference did not resolve to a tensor: {relative_path}")
    return value.float().reshape(-1)


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(left, right) / denominator).item())


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(abs(denominator), 1e-12)


def depth_bin(layer_index: int, layer_count: int) -> str:
    fraction = (layer_index + 0.5) / layer_count
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def descriptor_row(
    bundle: dict[str, Any],
    route_event: dict[str, Any],
    event_by_id: dict[str, dict[str, Any]],
    layer_count: int,
    cache: dict[str, Any],
) -> dict[str, Any]:
    generation_time = int(route_event["generation_time"])
    layer_index = int(route_event["layer_index"])
    receiver = route_event["receiver_role"]
    prefix = f"t{generation_time}:l{layer_index}:r{receiver}:"
    input_event = event_by_id[prefix + "input"]
    attention_event = event_by_id[prefix + "attention_merge"]
    post_event = event_by_id[prefix + "post_attention"]
    mlp_event = event_by_id[prefix + "mlp"]
    output_event = event_by_id[prefix + "output"]

    route = resolve_vector(route_event["vector_ref"], cache)
    layer_input = resolve_vector(input_event["vector_ref"], cache)
    attention = resolve_vector(attention_event["vector_ref"], cache)
    post_attention = resolve_vector(post_event["vector_ref"], cache)
    mlp = resolve_vector(mlp_event["vector_ref"], cache)
    output = resolve_vector(output_event["vector_ref"], cache)
    output_change = output - layer_input

    route_norm = norm(route)
    input_norm = norm(layer_input)
    attention_norm = norm(attention)
    post_norm = norm(post_attention)
    mlp_norm = norm(mlp)
    output_norm = norm(output)
    output_change_norm = norm(output_change)
    projection = safe_ratio(float(torch.dot(route, attention).item()), attention_norm * attention_norm)
    balance = safe_ratio(attention_norm - mlp_norm, attention_norm + mlp_norm)
    feature_values = {
        "route_norm_over_attention": safe_ratio(route_norm, attention_norm),
        "route_alignment_to_attention": cosine(route, attention),
        "route_signed_projection_to_attention": projection,
        "route_alignment_to_output_change": cosine(route, output_change),
        "attention_norm_over_input": safe_ratio(attention_norm, input_norm),
        "mlp_norm_over_post_attention": safe_ratio(mlp_norm, post_norm),
        "attention_alignment_to_mlp": cosine(attention, mlp),
        "attention_mlp_balance": balance,
        "output_change_norm_over_input": safe_ratio(output_change_norm, input_norm),
        "input_alignment_to_output": cosine(layer_input, output),
    }
    if not all(math.isfinite(value) for value in feature_values.values()):
        raise RuntimeError(f"Non-finite descriptor in {route_event['event_id']}")
    return {
        "schema_version": "43.2.0",
        "anonymous_case_id": bundle["anonymous_case_id"],
        "anonymous_model_id": bundle["anonymous_model_id"],
        "anonymous_group_id": bundle["anonymous_group_id"],
        "anonymous_condition_slot": bundle["anonymous_condition_slot"],
        "split": bundle["split"],
        "route_event_id": route_event["event_id"],
        "typed_path": "attention_source_write>attention_merge>residual_merge|mlp_merge>residual_state",
        "generation_time": generation_time,
        "layer_index": layer_index,
        "relative_layer_numerator": layer_index,
        "relative_layer_denominator": max(layer_count - 1, 1),
        "relative_depth_bin": depth_bin(layer_index, layer_count),
        "source_role_alias": route_event["source_role"],
        "receiver_role_alias": receiver,
        "branch_degree": 1,
        "merge_degree": 2,
        "raw_route_event_retained": bool(route_event["raw_event_retained"]),
        "raw_vector_bundle_ref": str(
            (BUNDLE_ROOT / "blind_bundles" / bundle["_model_key"] / f"{bundle['anonymous_case_id']}.json")
            .relative_to(PHASE_ROOT)
        ),
        "diagnostic_norms": {
            "route": route_norm,
            "layer_input": input_norm,
            "attention_write": attention_norm,
            "post_attention": post_norm,
            "mlp_write": mlp_norm,
            "layer_output": output_norm,
            "output_change": output_change_norm,
        },
        "features": feature_values,
    }


def main() -> None:
    split_counts: Counter[str] = Counter()
    model_rows = []
    total_descriptors = 0
    all_hashes = []
    for model in MODELS:
        bundle_paths = sorted((BUNDLE_ROOT / "blind_bundles" / model).glob("*.json"))
        layer_count = {
            "qwen3": 36,
            "glm4": 40,
            "deepseek7b": 28,
        }[model]
        output_path = OUT / "private" / f"{model}_directed_flow_descriptors.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_descriptor_count = 0
        model_split_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        receiver_counts: Counter[str] = Counter()
        with output_path.open("w", encoding="utf-8") as handle:
            for case_index, bundle_path in enumerate(bundle_paths, 1):
                bundle = read_json(bundle_path)
                bundle["_model_key"] = model
                event_by_id = {event["event_id"]: event for event in bundle["events"]}
                route_events = [
                    event for event in bundle["events"] if event["event_type"] == "attention_source_write"
                ]
                cache: dict[str, Any] = {}
                for route_event in route_events:
                    row = descriptor_row(bundle, route_event, event_by_id, layer_count, cache)
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                    model_descriptor_count += 1
                    source_counts[row["source_role_alias"]] += 1
                    receiver_counts[row["receiver_role_alias"]] += 1
                model_split_counts[bundle["split"]] += 1
                split_counts[bundle["split"]] += 1
                if case_index % 8 == 0 or case_index == len(bundle_paths):
                    print(
                        f"[{model}] descriptor bundles {case_index}/{len(bundle_paths)} "
                        f"rows={model_descriptor_count}",
                        flush=True,
                    )
        output_hash = sha256_file(output_path)
        all_hashes.append(output_hash)
        total_descriptors += model_descriptor_count
        model_rows.append({
            "model": model,
            "bundle_count": len(bundle_paths),
            "descriptor_count": model_descriptor_count,
            "split_case_counts": dict(sorted(model_split_counts.items())),
            "source_role_counts": dict(sorted(source_counts.items())),
            "receiver_role_counts": dict(sorted(receiver_counts.items())),
            "relative_path": str(output_path.relative_to(OUT)),
            "byte_count": output_path.stat().st_size,
            "sha256": output_hash,
        })

    schema = {
        "schema_version": "43.2.0",
        "phase_id": "Phase366-B",
        "unit": "one_directed_source_route_into_one_receiver_local_route_transform_merge_path",
        "typed_path": "attention_source_write>attention_merge>residual_merge|mlp_merge>residual_state",
        "feature_names": list(FEATURES),
        "all_source_routes_retained": True,
        "top_k_selection_used": False,
        "raw_vectors_retained_by_reference": True,
        "semantic_or_target_labels_allowed": False,
        "unmatched_routes_must_be_retained": True,
        "scope": "four_role_aliases_all_layers_three_generation_times",
        "scope_is_all_token_positions": False,
        "scope_is_full_neuron_event_enumeration": False,
    }
    summary = {
        "schema_version": "43.2.0",
        "phase_id": "Phase366-B",
        "created_at": now(),
        "objective": "extract_direction_sensitive_typed_flow_descriptors_before_any_semantic_label_reveal",
        "denominator": {
            "model_count": len(MODELS),
            "bundle_count": sum(row["bundle_count"] for row in model_rows),
            "directed_path_descriptor_count": total_descriptors,
            "feature_count_per_descriptor": len(FEATURES),
            "blind_discovery_case_count": split_counts["blind_discovery"],
            "blind_calibration_case_count": split_counts["blind_calibration"],
        },
        "results": {
            "all_source_routes_retained": True,
            "raw_vector_references_retained": True,
            "top_k_selection_used": False,
            "condition_average_used": False,
            "direct_graph_subtraction_used": False,
            "semantic_or_target_label_used": False,
            "path_candidate_count": 0,
        },
        "models": model_rows,
        "descriptor_schema": "phase366_descriptor_schema.json",
        "combined_descriptor_digest": hashlib.sha256("".join(all_hashes).encode()).hexdigest(),
        "authorization": {
            "private_threshold_custodian_authorized": total_descriptors > 0,
            "blind_motif_scoring_authorized": False,
            "semantic_label_reveal_authorized": False,
            "physical_confirmation_authorized": False,
        },
        "claim_boundary": {
            "descriptor_extraction_complete": True,
            "descriptors_are_language_paths": False,
            "future_prediction_test_executed": False,
            "calibration_replication_executed": False,
        },
        "next_decision": "freeze_reconstruction_repeat_and_same_condition_template_floors_before_scoring",
    }
    write_json(OUT / "phase366_descriptor_schema.json", schema)
    write_json(OUT / "phase366_descriptor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
