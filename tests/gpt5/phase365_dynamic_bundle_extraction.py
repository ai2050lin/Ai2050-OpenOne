#!/usr/bin/env python3
"""Build label-blind typed dynamic bundles from the replayable engineering ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase365_dynamic_flow_instrumentation import validate_blind_bundle  # noqa: E402


COLLECTION = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection"
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/dynamic_bundle_extraction"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def alias_groups(role_names: list[str] | tuple[str, ...], positions: list[int]) -> list[dict[str, Any]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for role, position in zip(role_names, positions, strict=True):
        grouped[int(position)].append(role)
    return [
        {"position": position, "roles": sorted(roles), "alias_id": "+".join(sorted(roles))}
        for position, roles in sorted(grouped.items())
    ]


def load_weight(model: str, parameter: str) -> torch.Tensor:
    spec = get_model_spec(model)
    index = read_json(spec.local_dir / "model.safetensors.index.json")
    shard = spec.local_dir / index["weight_map"][parameter]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(parameter)


def vector_ref(path: Path, digest: str, tensor: torch.Tensor, slice_value: Any) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(OUT.parent)),
        "sha256": digest,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "slice": slice_value,
    }


def raw_vector_ref(raw_relative: str, digest: str, tensor: torch.Tensor, slice_value: Any) -> dict[str, Any]:
    return {
        "relative_path": str((COLLECTION / raw_relative).relative_to(OUT.parent)),
        "sha256": digest,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "slice": slice_value,
    }


def event(
    event_id: str,
    event_type: str,
    generation_time: int,
    layer_index: int,
    receiver_alias: str,
    reference: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "event_id": event_id, "event_type": event_type,
        "generation_time": generation_time, "layer_index": layer_index,
        "receiver_role": receiver_alias, "vector_ref": reference,
        "raw_event_retained": True, **extra,
    }


def edge(edge_id: str, edge_type: str, source: str, target: str) -> dict[str, Any]:
    return {"edge_id": edge_id, "edge_type": edge_type, "source_event_id": source, "target_event_id": target}


def existing_case_rows(
    model: str,
    case_id: str,
    layer_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    bundle_path = OUT / "blind_bundles" / model / f"{case_id}.json"
    role_edge_root = OUT / "private" / "role_edges" / model / case_id
    derived_paths = sorted(role_edge_root.glob("time_*/layer_*.pt"))
    expected_derived = layer_count * 3
    if (
        not bundle_path.is_file()
        or len(derived_paths) != expected_derived
        or any(path.stat().st_size == 0 for path in derived_paths)
    ):
        return None
    bundle = read_json(bundle_path)
    errors = validate_blind_bundle(bundle)
    if errors:
        return None
    alias_groups_by_layer_time: dict[tuple[int, int], set[str]] = defaultdict(set)
    for item in bundle["events"]:
        if item["event_type"] != "attention_source_write":
            continue
        source_role = item.get("source_role")
        if source_role and source_role != "other_sources":
            alias_groups_by_layer_time[(item["generation_time"], item["layer_index"])].add(source_role)
    derived_rows = [
        {
            "anonymous_case_id": case_id,
            "relative_path": str(path.relative_to(OUT)),
            "byte_count": path.stat().st_size,
        }
        for path in derived_paths
    ]
    bundle_row = {
        "bundle_id": bundle["bundle_id"],
        "anonymous_case_id": case_id,
        "anonymous_model_id": bundle["anonymous_model_id"],
        "anonymous_condition_slot": bundle["anonymous_condition_slot"],
        "event_count": len(bundle["events"]),
        "edge_count": len(bundle["edges"]),
        "alias_group_observation_count": sum(len(values) for values in alias_groups_by_layer_time.values()),
        "derived_byte_count": sum(row["byte_count"] for row in derived_rows),
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "relative_path": str(bundle_path.relative_to(OUT)),
    }
    return bundle_row, derived_rows


def extract_model(model: str, compute_device: torch.device, resume: bool = False) -> dict[str, Any]:
    manifest = read_json(COLLECTION / "models" / model / "manifest.json")
    raw_hashes = {row["relative_path"]: row["sha256"] for row in manifest["files"]}
    layer_count = int(manifest["layer_count"])
    files_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["files"]:
        files_by_case[row["blind_case_id"]].append(row)
    bundle_rows, derived_rows = [], []
    total_events = total_edges = total_alias_groups = 0
    pending_case_ids = []
    if resume:
        for case_id in sorted(files_by_case):
            existing = existing_case_rows(model, case_id, layer_count)
            if existing is None:
                pending_case_ids.append(case_id)
                continue
            bundle_row, case_derived_rows = existing
            bundle_rows.append(bundle_row)
            derived_rows.extend(case_derived_rows)
            total_events += bundle_row["event_count"]
            total_edges += bundle_row["edge_count"]
            total_alias_groups += bundle_row["alias_group_observation_count"]
    else:
        pending_case_ids = sorted(files_by_case)
    o_weights = {
        layer: load_weight(model, f"model.layers.{layer}.self_attn.o_proj.weight").to(compute_device, dtype=torch.float32)
        for layer in range(layer_count)
    } if pending_case_ids else {}
    for case_index, case_id in enumerate(pending_case_ids, 1):
        events, edges = [], []
        previous_layer_output: dict[tuple[int, str], str] = {}
        final_time_outputs: dict[tuple[int, str], str] = {}
        case_meta = None
        case_derived_bytes = 0
        for generation_time in range(3):
            rows = files_by_case[case_id]
            time_row = next(row for row in rows if row["generation_time"] == generation_time and row["kind"] == "time_meta")
            time_payload = torch.load(COLLECTION / time_row["relative_path"], map_location="cpu", weights_only=True)
            case_meta = time_payload
            vocab_id = f"t{generation_time}:vocab"
            events.append(event(
                vocab_id, "vocab_state", generation_time, layer_count,
                "current_generation", raw_vector_ref(
                    time_row["relative_path"], raw_hashes[time_row["relative_path"]],
                    time_payload["full_vocabulary_logits"], ["full_vocabulary_logits"],
                ),
                label_free_vocab_state=True,
            ))
            for layer_index in range(layer_count):
                raw_row = next(
                    row for row in rows
                    if row["generation_time"] == generation_time and row["kind"] == "layer" and row["layer_index"] == layer_index
                )
                raw_path = COLLECTION / raw_row["relative_path"]
                payload = torch.load(raw_path, map_location="cpu", weights_only=True)
                roles = list(payload["role_names"])
                positions = [int(value) for value in payload["role_positions"]]
                source_groups = alias_groups(roles, positions)
                receiver_groups = source_groups
                total_alias_groups += len(source_groups)
                attention = payload["attention"]
                values = attention["value_states_all_sources"].to(compute_device, dtype=torch.float32)
                probs = attention["probabilities_role_receivers_all_sources"].to(compute_device, dtype=torch.float32)
                head_count = int(attention["head_count"])
                kv_count = int(attention["key_value_head_count"])
                head_dim = int(attention["head_dim"])
                repeated = values
                if kv_count != head_count:
                    repeated = values.repeat_interleave(head_count // kv_count, dim=1)
                repeated = repeated[0]
                o_weight = o_weights[layer_index].view(o_weights[layer_index].shape[0], head_count, head_dim)
                role_edges = torch.empty(
                    (len(receiver_groups), len(source_groups), o_weight.shape[0]),
                    dtype=torch.float32, device=compute_device,
                )
                for receiver_index, receiver_group in enumerate(receiver_groups):
                    receiver_role_index = roles.index(receiver_group["roles"][0])
                    for source_index, source_group in enumerate(source_groups):
                        source_position = source_group["position"]
                        weighted = probs[0, :, receiver_role_index, source_position].unsqueeze(-1) * repeated[:, source_position, :]
                        role_edges[receiver_index, source_index] = torch.einsum("hd,ohd->o", weighted, o_weight)
                selected_attention = payload["component_vectors"]["attention_output"].to(compute_device, dtype=torch.float32)
                unique_receiver_indices = [roles.index(group["roles"][0]) for group in receiver_groups]
                selected_unique_attention = selected_attention[:, unique_receiver_indices, :][0]
                other_edges = selected_unique_attention - role_edges.sum(dim=1)
                derived_payload = {
                    "schema_version": "42.7.0", "phase_id": "Phase365-C",
                    "anonymous_case_id": case_id,
                    "anonymous_model_id": payload["anonymous_model_id"],
                    "generation_time": generation_time, "layer_index": layer_index,
                    "source_alias_groups": source_groups, "receiver_alias_groups": receiver_groups,
                    "role_attention_edges": role_edges.to(device="cpu", dtype=torch.float16),
                    "other_source_attention_edges": other_edges.to(device="cpu", dtype=torch.float16),
                }
                derived_path = OUT / "private" / "role_edges" / model / case_id / f"time_{generation_time}" / f"layer_{layer_index:03d}.pt"
                derived_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(derived_payload, derived_path)
                derived_hash = sha256_file(derived_path)
                case_derived_bytes += derived_path.stat().st_size
                derived_rows.append({
                    "anonymous_case_id": case_id, "generation_time": generation_time,
                    "layer_index": layer_index,
                    "relative_path": str(derived_path.relative_to(OUT)),
                    "byte_count": derived_path.stat().st_size, "sha256": derived_hash,
                })
                raw_digest = raw_hashes[raw_row["relative_path"]]

                for receiver_index, receiver_group in enumerate(receiver_groups):
                    receiver_alias = receiver_group["alias_id"]
                    input_index = roles.index(receiver_group["roles"][0])
                    input_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:input"
                    attention_merge_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:attention_merge"
                    post_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:post_attention"
                    mlp_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:mlp"
                    output_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:output"
                    events.extend([
                        event(input_id, "residual_state", generation_time, layer_index, receiver_alias,
                              raw_vector_ref(raw_row["relative_path"], raw_digest, payload["component_vectors"]["layer_input"], ["component_vectors", "layer_input", 0, input_index])),
                        event(attention_merge_id, "attention_merge", generation_time, layer_index, receiver_alias,
                              raw_vector_ref(raw_row["relative_path"], raw_digest, payload["component_vectors"]["attention_output"], ["component_vectors", "attention_output", 0, input_index])),
                        event(post_id, "residual_merge", generation_time, layer_index, receiver_alias,
                              raw_vector_ref(raw_row["relative_path"], raw_digest, payload["component_vectors"]["post_attention_state"], ["component_vectors", "post_attention_state", 0, input_index])),
                        event(mlp_id, "mlp_merge", generation_time, layer_index, receiver_alias,
                              raw_vector_ref(raw_row["relative_path"], raw_digest, payload["component_vectors"]["mlp_output"], ["component_vectors", "mlp_output", 0, input_index]),
                              neuron_product_ref=raw_vector_ref(raw_row["relative_path"], raw_digest, payload["mlp"]["down_projection_input_product_at_roles"], ["mlp", "down_projection_input_product_at_roles", 0, input_index])),
                        event(output_id, "residual_state", generation_time, layer_index, receiver_alias,
                              raw_vector_ref(raw_row["relative_path"], raw_digest, payload["component_vectors"]["layer_output"], ["component_vectors", "layer_output", 0, input_index])),
                    ])
                    route_ids = []
                    for source_index, source_group in enumerate(source_groups):
                        route_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:from:{source_group['alias_id']}"
                        events.append(event(
                            route_id, "attention_source_write", generation_time, layer_index, receiver_alias,
                            vector_ref(derived_path, derived_hash, derived_payload["role_attention_edges"], ["role_attention_edges", receiver_index, source_index]),
                            source_role=source_group["alias_id"], source_position=source_group["position"],
                        ))
                        route_ids.append(route_id)
                        edges.append(edge(f"{route_id}->merge", "route", route_id, attention_merge_id))
                    other_id = f"t{generation_time}:l{layer_index}:r{receiver_alias}:from:other_sources"
                    events.append(event(
                        other_id, "attention_source_write", generation_time, layer_index, receiver_alias,
                        vector_ref(derived_path, derived_hash, derived_payload["other_source_attention_edges"], ["other_source_attention_edges", receiver_index]),
                        source_role="other_sources",
                    ))
                    edges.extend([
                        edge(f"{other_id}->merge", "route", other_id, attention_merge_id),
                        edge(f"{input_id}->post", "merge", input_id, post_id),
                        edge(f"{attention_merge_id}->post", "write", attention_merge_id, post_id),
                        edge(f"{post_id}->output", "merge", post_id, output_id),
                        edge(f"{mlp_id}->output", "write", mlp_id, output_id),
                    ])
                    if layer_index > 0:
                        previous = previous_layer_output[(generation_time, receiver_alias)]
                        edges.append(edge(f"{previous}->{input_id}", "residual", previous, input_id))
                    previous_layer_output[(generation_time, receiver_alias)] = output_id
                    if layer_index == layer_count - 1:
                        final_time_outputs[(generation_time, receiver_alias)] = output_id
                        edges.append(edge(f"{output_id}->{vocab_id}", "vocab_transition", output_id, vocab_id))
                del payload, derived_payload, role_edges, other_edges, repeated, probs, values

            if generation_time > 0:
                current_alias = next(
                    group["alias_id"] for group in alias_groups(list(time_payload["role_names"]), list(time_payload["role_positions"]))
                    if "current_generation" in group["roles"]
                )
                previous_candidates = [
                    (alias, event_id) for (time, alias), event_id in final_time_outputs.items()
                    if time == generation_time - 1 and "current_generation" in alias.split("+")
                ]
                if previous_candidates:
                    previous_alias, previous_id = previous_candidates[0]
                    current_input = f"t{generation_time}:l0:r{current_alias}:input"
                    edges.append(edge(f"{previous_id}->{current_input}:generation", "time", previous_id, current_input))

        bundle = {
            "schema_version": "42.7.0", "bundle_id": "bundle_" + hashlib.sha256(case_id.encode()).hexdigest()[:20],
            "anonymous_case_id": case_id, "anonymous_model_id": case_meta["anonymous_model_id"],
            "anonymous_group_id": case_meta["anonymous_group_id"],
            "anonymous_condition_slot": case_meta["anonymous_condition_slot"],
            "split": "blind_discovery", "events": events, "edges": edges,
            "scope": "four_role_component_and_role_source_bundle",
            "derived_byte_count": case_derived_bytes,
        }
        errors = validate_blind_bundle(bundle)
        bundle_path = OUT / "blind_bundles" / model / f"{case_id}.json"
        write_json(bundle_path, bundle)
        bundle_rows.append({
            "bundle_id": bundle["bundle_id"], "anonymous_case_id": case_id,
            "anonymous_model_id": bundle["anonymous_model_id"],
            "anonymous_group_id": bundle["anonymous_group_id"],
            "anonymous_condition_slot": bundle["anonymous_condition_slot"],
            "event_count": len(events), "edge_count": len(edges),
            "derived_byte_count": case_derived_bytes,
            "validation_error_count": len(errors), "validation_errors": errors,
            "relative_path": str(bundle_path.relative_to(OUT)),
        })
        total_events += len(events)
        total_edges += len(edges)
        print(
            f"[{model}] new bundle {case_index}/{len(pending_case_ids)} "
            f"total={len(bundle_rows)}/{len(files_by_case)} events={len(events)} edges={len(edges)}",
            flush=True,
        )
    del o_weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model, "bundle_count": len(bundle_rows), "event_count": total_events,
        "edge_count": total_edges, "alias_group_observation_count": total_alias_groups,
        "derived_file_count": len(derived_rows),
        "derived_byte_count": sum(row["byte_count"] for row in derived_rows),
        "all_bundles_valid": all(row["validation_error_count"] == 0 for row in bundle_rows),
        "bundle_rows": bundle_rows, "derived_rows": derived_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--collection-root", type=Path, default=COLLECTION)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global COLLECTION, OUT
    args = parse_args()
    COLLECTION = args.collection_root
    OUT = args.output_root
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    compute_device = torch.device(device_name)
    model_rows = [extract_model(model, compute_device, resume=args.resume) for model in args.models]
    compact_models = [{key: value for key, value in row.items() if key not in {"bundle_rows", "derived_rows"}} for row in model_rows]
    summary = {
        "schema_version": "42.7.0", "phase_id": "Phase365-C", "created_at": now(),
        "denominator": {
            "model_count": len(model_rows), "bundle_count": sum(row["bundle_count"] for row in model_rows),
            "event_count": sum(row["event_count"] for row in model_rows),
            "edge_count": sum(row["edge_count"] for row in model_rows),
            "derived_role_edge_file_count": sum(row["derived_file_count"] for row in model_rows),
            "derived_role_edge_byte_count": sum(row["derived_byte_count"] for row in model_rows),
        },
        "results": {
            "valid_model_count": sum(row["all_bundles_valid"] for row in model_rows),
            "valid_bundle_count": sum(
                item["validation_error_count"] == 0 for row in model_rows for item in row["bundle_rows"]
            ),
            "receiver_source_aliasing_explicit": True,
            "other_source_bucket_retained": True,
            "raw_event_vectors_retained": True,
            "target_specific_competition_used": False,
            "direct_graph_subtraction_used": False,
            "path_candidate_count": 0,
        },
        "models": compact_models,
        "claim_boundary": {
            "dynamic_bundle_format_complete_in_four_role_scope": all(row["all_bundles_valid"] for row in model_rows),
            "all_token_position_bundle_complete": False,
            "blind_motif_discovery_executed": False,
            "language_path_discovered": False,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
        },
        "authorization": {
            "blind_motif_extraction_algorithm_design_authorized": all(row["all_bundles_valid"] for row in model_rows),
            "target_or_condition_label_reveal_authorized": False,
            "physical_confirmation_authorized": False,
        },
        "next_decision": "freeze_label_blind_motif_extraction_without_mad_only_thresholds",
    }
    suffix = "" if tuple(args.models) == MODELS else "_" + "_".join(args.models)
    write_json(OUT / f"phase365_dynamic_bundle_summary{suffix}.json", summary)
    write_json(OUT / f"phase365_model_bundle_rows{suffix}.json", compact_models)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
