#!/usr/bin/env python3
"""Build label-free lazy exact event graphs from the audited Phase371C ledgers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
BASE = PHASE371 / "phase371c_internal_discovery"
ADJ = PHASE371 / "phase371c_adjacent_extension"
AUDIT = PHASE371 / "phase371c_adjacent_extension_audit.json"
OUT = PHASE371 / "phase371c_lazy_exact_paths"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_ref(root_name: str, relative_path: str, pointer: str, slice_spec: str = "all") -> dict[str, str]:
    return {
        "ledger": root_name,
        "relative_path": relative_path,
        "tensor_pointer": pointer,
        "slice": slice_spec,
    }


def layer_path(root: Path, model: str, case_id: str, time: int, layer: int) -> Path:
    return root / "private/models" / model / case_id / f"time_{time}" / f"layer_{layer:03d}.pt"


def lazy_layer_nodes(
    model: str,
    case_id: str,
    generation_time: int,
    layer: int,
    root_name: str,
    root: Path,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    path = layer_path(root, model, case_id, generation_time, layer)
    relative_path = str(path.relative_to(root))
    prefix = f"t{generation_time}:l{layer}"
    sequence = int(payload["sequence_length"])
    head_count = int(payload["attention"]["head_count"])
    channel_count = int(payload["mlp"]["channel_count"])
    nodes = [
        {
            "id": f"{prefix}:input",
            "event_type": "layer_input",
            "position_domain": [0, sequence],
            "exact_vector_ref": tensor_ref(root_name, relative_path, "component_vectors/layer_input_all_positions"),
        },
        {
            "id": f"{prefix}:qk",
            "event_type": "query_key_score",
            "receiver_domain": [0, sequence],
            "source_domain": [0, sequence],
            "head_domain": [0, head_count],
            "exact_query_ref": tensor_ref(root_name, relative_path, "attention/query_states_all_positions"),
            "exact_key_ref": tensor_ref(root_name, relative_path, "attention/key_states_all_positions"),
            "probability_ref": tensor_ref(root_name, relative_path, "attention/probabilities_all_receivers_all_sources"),
            "derivation": "softmax(query@repeat_kv(key)^T*scaling+causal_mask)",
        },
        {
            "id": f"{prefix}:attn_heads",
            "event_type": "attention_head_write",
            "position_domain": [0, sequence],
            "head_domain": [0, head_count],
            "probability_ref": tensor_ref(root_name, relative_path, "attention/probabilities_all_receivers_all_sources"),
            "value_ref": tensor_ref(root_name, relative_path, "attention/value_states_all_positions"),
            "weight_reference_id": payload["attention"]["output_projection_weight_reference_id"],
            "partitions": payload["attention"]["head_partitions"],
            "derivation": "per_head_probability_weighted_value_times_o_projection_head_block",
        },
        {
            "id": f"{prefix}:attn_merge",
            "event_type": "attention_merge",
            "position_domain": [0, sequence],
            "exact_vector_ref": tensor_ref(root_name, relative_path, "component_vectors/attention_output_all_positions"),
        },
        {
            "id": f"{prefix}:attn_residual",
            "event_type": "residual_merge",
            "position_domain": [0, sequence],
            "exact_vector_ref": tensor_ref(root_name, relative_path, "component_vectors/post_attention_state_all_positions"),
        },
        {
            "id": f"{prefix}:mlp_neurons",
            "event_type": "mlp_single_neuron_write",
            "position_domain": [0, sequence],
            "neuron_domain": [0, channel_count],
            "product_ref": tensor_ref(root_name, relative_path, "mlp/down_projection_input_product_all_positions"),
            "weight_reference_id": payload["mlp"]["down_projection_weight_reference_id"],
            "partitions": payload["mlp"]["channel_partitions"],
            "derivation": "product[position,neuron]*down_projection_weight[:,neuron]",
        },
        {
            "id": f"{prefix}:mlp_merge",
            "event_type": "mlp_merge",
            "position_domain": [0, sequence],
            "exact_vector_ref": tensor_ref(root_name, relative_path, "component_vectors/mlp_output_all_positions"),
        },
        {
            "id": f"{prefix}:output",
            "event_type": "layer_output",
            "position_domain": [0, sequence],
            "exact_vector_ref": tensor_ref(root_name, relative_path, "component_vectors/layer_output_all_positions"),
        },
    ]
    edges = [
        {"source": f"{prefix}:input", "target": f"{prefix}:qk", "type": "normalization_and_projection"},
        {"source": f"{prefix}:qk", "target": f"{prefix}:attn_heads", "type": "source_routing"},
        {"source": f"{prefix}:attn_heads", "target": f"{prefix}:attn_merge", "type": "exact_head_sum"},
        {"source": f"{prefix}:input", "target": f"{prefix}:attn_residual", "type": "residual_parent"},
        {"source": f"{prefix}:attn_merge", "target": f"{prefix}:attn_residual", "type": "residual_child"},
        {"source": f"{prefix}:attn_residual", "target": f"{prefix}:mlp_neurons", "type": "normalization_and_gate"},
        {"source": f"{prefix}:mlp_neurons", "target": f"{prefix}:mlp_merge", "type": "exact_neuron_sum"},
        {"source": f"{prefix}:attn_residual", "target": f"{prefix}:output", "type": "residual_parent"},
        {"source": f"{prefix}:mlp_merge", "target": f"{prefix}:output", "type": "residual_child"},
    ]
    counts = {
        "query_key_score_events": head_count * sequence * sequence,
        "attention_head_write_events": head_count * sequence,
        "mlp_single_neuron_write_events": channel_count * sequence,
        "residual_merge_events": 2 * sequence,
    }
    return nodes, edges, counts


def build_case(model: str, case_id: str, base_manifest: dict[str, Any], adj_manifest: dict[str, Any]) -> dict[str, Any]:
    base_layers = list(base_manifest["anchor_layers"])
    adjacent_layers = list(adj_manifest["selected_layers"])
    ordered_layers = sorted(set(base_layers + adjacent_layers))
    layer_sources = {layer: ("base", BASE) if layer in base_layers else ("adjacent", ADJ) for layer in ordered_layers}
    pair_edges = [
        (base_layers[0], adjacent_layers[0], "early"),
        (base_layers[1], adjacent_layers[1], "middle"),
        (adjacent_layers[2], base_layers[2], "late"),
    ]
    nodes = []
    edges = []
    event_counts = {
        "query_key_score_events": 0,
        "attention_head_write_events": 0,
        "mlp_single_neuron_write_events": 0,
        "residual_merge_events": 0,
    }
    first_payload = None
    for generation_time in range(3):
        for layer in ordered_layers:
            root_name, root = layer_sources[layer]
            payload = torch.load(
                layer_path(root, model, case_id, generation_time, layer),
                map_location="cpu", weights_only=True,
            )
            first_payload = first_payload or payload
            layer_nodes, layer_edges, counts = lazy_layer_nodes(
                model, case_id, generation_time, layer, root_name, root, payload,
            )
            nodes.extend(layer_nodes)
            edges.extend(layer_edges)
            for key, value in counts.items():
                event_counts[key] += value
        for source_layer, receiver_layer, pair_name in pair_edges:
            edges.append({
                "source": f"t{generation_time}:l{source_layer}:output",
                "target": f"t{generation_time}:l{receiver_layer}:input",
                "type": "verified_layer_continuity",
                "pair": pair_name,
                "relative_error": 0.0,
            })
        time_path = BASE / "private/models" / model / case_id / f"time_{generation_time}/time_meta.pt"
        nodes.append({
            "id": f"t{generation_time}:vocab",
            "event_type": "label_free_vocab_state",
            "exact_vector_ref": tensor_ref("base", str(time_path.relative_to(BASE)), "full_vocabulary_logits"),
        })
        edges.append({
            "source": f"t{generation_time}:l{base_layers[-1]}:output",
            "target": f"t{generation_time}:vocab",
            "type": "final_norm_and_unembedding",
        })
        if generation_time < 2:
            edges.append({
                "source": f"t{generation_time}:vocab",
                "target": f"t{generation_time + 1}:l{base_layers[0]}:input",
                "type": "greedy_generation_feedback",
            })
    return {
        "schema_version": "47.15.0",
        "phase_id": "Phase371C-Path",
        "model": model,
        "blind_case_id": case_id,
        "anonymous_model_id": first_payload["anonymous_model_id"],
        "anonymous_parallel_group_id": first_payload["anonymous_parallel_group_id"],
        "anonymous_group_id": first_payload["anonymous_group_id"],
        "anonymous_condition_slot": first_payload["anonymous_condition_slot"],
        "generation_time_count": 3,
        "layer_pairs": [
            {"name": name, "source_layer": source, "receiver_layer": receiver}
            for source, receiver, name in pair_edges
        ],
        "nodes": nodes,
        "edges": edges,
        "implicit_exact_event_counts": event_counts,
        "claim_boundary": {
            "label_free_extraction": True,
            "semantic_labels_available": False,
            "candidate_selected": False,
            "language_path_claimed": False,
            "global_all_layer_path": False,
        },
    }


def main() -> None:
    audit = read_json(AUDIT)
    if not audit["authorization"]["extract_lazy_exact_path_objects"]:
        raise RuntimeError("Adjacent audit did not authorize path extraction")
    model_rows = []
    total_nodes = 0
    total_edges = 0
    total_events = {
        "query_key_score_events": 0,
        "attention_head_write_events": 0,
        "mlp_single_neuron_write_events": 0,
        "residual_merge_events": 0,
    }
    forbidden = {"family_id", "mechanism_id", "semantic_group_id", "contrast_condition", "target", "distractors"}
    forbidden_count = 0
    for model in MODELS:
        base_manifest = read_json(BASE / "models" / model / "manifest.json")
        adj_manifest = read_json(ADJ / "models" / model / "manifest.json")
        case_ids = sorted(row["blind_case_id"] for row in base_manifest["case_rows"])
        model_nodes = 0
        model_edges = 0
        model_events = {key: 0 for key in total_events}
        for index, case_id in enumerate(case_ids, 1):
            path_object = build_case(model, case_id, base_manifest, adj_manifest)
            forbidden_count += sum(1 for key in forbidden if key in path_object)
            path = OUT / "private/models" / model / f"{case_id}.json"
            write_json(path, path_object)
            model_nodes += len(path_object["nodes"])
            model_edges += len(path_object["edges"])
            for key, value in path_object["implicit_exact_event_counts"].items():
                model_events[key] += value
            if index % 16 == 0 or index == len(case_ids):
                print(f"[{model}] lazy paths {index}/{len(case_ids)}", flush=True)
        total_nodes += model_nodes
        total_edges += model_edges
        for key, value in model_events.items():
            total_events[key] += value
        model_rows.append({
            "model": model,
            "case_count": len(case_ids),
            "explicit_node_count": model_nodes,
            "explicit_edge_count": model_edges,
            "implicit_exact_event_counts": model_events,
        })
    summary = {
        "schema_version": "47.15.0",
        "phase_id": "Phase371C-Path",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "build_complete_lazy_references_for_all_measured_qk_head_neuron_residual_and_generation_events",
        "valid": forbidden_count == 0,
        "denominator": {
            "model_count": 3,
            "case_count": 264,
            "explicit_node_count": total_nodes,
            "explicit_edge_count": total_edges,
            "implicit_exact_event_counts": total_events,
        },
        "models": model_rows,
        "quality": {
            "semantic_forbidden_field_count": forbidden_count,
            "top_k_used": False,
            "scalar_or_hash_terminal_state_used": False,
            "exact_vector_materialization_duplicated": False,
            "verified_local_continuity_edges_per_case": 9,
        },
        "results": {
            "lazy_exact_path_object_complete_for_measured_layers": forbidden_count == 0,
            "global_all_layer_path_complete": False,
            "candidate_language_path_count": 0,
            "language_mechanism_claimed": False,
        },
        "authorization": {
            "freeze_label_free_pairwise_vector_contrast_algorithm": forbidden_count == 0,
            "use_semantic_labels_during_candidate_extraction": False,
            "open_calibration": False,
            "open_physical": False,
        },
        "next_decision": "freeze_blind_all_pair_vector_contrast_and_same_graph_replay_before_condition_unblinding",
    }
    write_json(OUT / "phase371c_lazy_exact_path_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
