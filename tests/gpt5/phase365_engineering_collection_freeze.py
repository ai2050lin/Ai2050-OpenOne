#!/usr/bin/env python3
"""Freeze a 96-case role-focused engineering collection and its storage budget."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P362 = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection_freeze"
MODELS = ("qwen3", "glm4", "deepseek7b")
ROLE_COUNT = 4
GENERATION_TIME_COUNT = 3
GROUPS_PER_MODEL_MECHANISM = 2
PLANNING_SEQUENCE_LENGTH = 256
NATIVE_DTYPE_BYTES = 2
LOGIT_DTYPE_BYTES = 4
METADATA_MULTIPLIER = 1.15
MINIMUM_FREE_RESERVE_BYTES = 200 * 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def rank_group(group_id: str) -> str:
    return hashlib.sha256(f"phase365-engineering-v1:{group_id}".encode()).hexdigest()


def anonymous_slot(group_id: str, condition: str) -> str:
    return "slot_" + hashlib.sha256(f"phase365-slot-v1:{group_id}:{condition}".encode()).hexdigest()[:12]


def storage_row(model: str, config: dict[str, Any], case_count: int) -> dict[str, Any]:
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim") or hidden // heads)
    vocab = int(config["vocab_size"])
    per_layer_time_case = {
        "seven_role_component_vectors": 7 * ROLE_COUNT * hidden * NATIVE_DTYPE_BYTES,
        "all_source_value_states": kv_heads * PLANNING_SEQUENCE_LENGTH * head_dim * NATIVE_DTYPE_BYTES,
        "all_head_role_source_probabilities": heads * ROLE_COUNT * PLANNING_SEQUENCE_LENGTH * NATIVE_DTYPE_BYTES,
        "mlp_gate_up_product_at_roles": 3 * ROLE_COUNT * intermediate * NATIVE_DTYPE_BYTES,
    }
    layer_payload = sum(per_layer_time_case.values()) * layers * GENERATION_TIME_COUNT * case_count
    logits = vocab * LOGIT_DTYPE_BYTES * GENERATION_TIME_COUNT * case_count
    planned = int((layer_payload + logits) * METADATA_MULTIPLIER)
    naive_neuron_writes = (
        ROLE_COUNT * intermediate * hidden * NATIVE_DTYPE_BYTES
        * layers * GENERATION_TIME_COUNT * case_count
    )
    return {
        "model": model,
        "architecture": {
            "layer_count": layers, "hidden_size": hidden, "intermediate_size": intermediate,
            "attention_head_count": heads, "key_value_head_count": kv_heads,
            "head_dim": head_dim, "vocab_size": vocab,
        },
        "case_count": case_count,
        "per_layer_time_case_bytes": per_layer_time_case,
        "planned_role_focused_bytes": planned,
        "naive_explicit_neuron_write_bytes": naive_neuron_writes,
        "explicit_neuron_write_bytes_saved": naive_neuron_writes,
    }


def main() -> None:
    cases = [
        row for row in read_jsonl(P362 / "private" / "phase362_execution_cases.jsonl")
        if row["phase362_split"] == "independent_calibration"
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[(row["model"], row["family_id"], row["mechanism_id"], row["phase362_group_id"])].append(row)
    mechanism_groups: dict[tuple[str, str, str], list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (model, family, mechanism, group_id), rows in grouped.items():
        mechanism_groups[(model, family, mechanism)].append((group_id, rows))

    selected = []
    selection_rows = []
    for (model, family, mechanism), groups in sorted(mechanism_groups.items()):
        groups.sort(key=lambda item: rank_group(item[0]))
        chosen = groups[:GROUPS_PER_MODEL_MECHANISM]
        if len(chosen) != GROUPS_PER_MODEL_MECHANISM:
            raise RuntimeError(f"Insufficient groups for {model}/{family}/{mechanism}")
        for group_id, rows in chosen:
            if len(rows) != 4:
                raise RuntimeError(f"Expected four conditions in {group_id}, got {len(rows)}")
            condition_values = sorted(row["contrast_condition"] for row in rows)
            selection_rows.append({
                "anonymous_model_id": rows[0]["anonymous_model_id"],
                "anonymous_group_id": group_id,
                "case_count": len(rows),
                "condition_slot_count": len(condition_values),
                "selection_hash": rank_group(group_id),
            })
            for row in rows:
                selected.append({
                    **row,
                    "phase365_split": "engineering_format_collection",
                    "anonymous_condition_slot": anonymous_slot(group_id, row["contrast_condition"]),
                    "semantic_labels_available_to_collection": False,
                    "target_specific_competition_available_to_collection": False,
                })

    if len(selected) != 96:
        raise RuntimeError(f"Expected 96 selected cases, got {len(selected)}")
    physical_ids = {
        row["blind_case_id"] for row in read_jsonl(P362 / "private" / "phase362_execution_cases.jsonl")
        if row["phase362_split"] == "physical_confirmation_sealed"
    }
    overlap = physical_ids & {row["blind_case_id"] for row in selected}
    if overlap:
        raise RuntimeError(f"Physical-confirmation overlap: {len(overlap)}")

    storage_rows = []
    for model in MODELS:
        config_path = ROOT / {
            "qwen3": "models/hf/qwen3-4b/config.json",
            "glm4": "models/hf/glm4-9b-chat-hf/config.json",
            "deepseek7b": "models/hf/deepseek-r1-distill-qwen-7b/config.json",
        }[model]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        storage_rows.append(storage_row(model, config, sum(row["model"] == model for row in selected)))
    disk_free = shutil.disk_usage(ROOT).free
    planned_total = sum(row["planned_role_focused_bytes"] for row in storage_rows)
    naive_total = sum(row["naive_explicit_neuron_write_bytes"] for row in storage_rows)
    blind_execution = [{
        "schema_version": "42.4.0",
        "phase_id": "Phase365-B",
        "blind_case_id": row["blind_case_id"],
        "anonymous_model_id": row["anonymous_model_id"],
        "private_execution_model": row["model"],
        "anonymous_group_id": row["phase362_group_id"],
        "anonymous_condition_slot": row["anonymous_condition_slot"],
        "prompt": row["prompt"],
        "raw_prompt": row["raw_prompt"],
        "source_fragment": row["source_fragment"],
        "query_fragment": row["query_fragment"],
        "tokenization_add_special_tokens": row["tokenization_add_special_tokens"],
        "phase365_split": row["phase365_split"],
    } for row in selected]

    collection_schema = {
        "schema_version": "42.4.0", "phase_id": "Phase365-B", "created_at": now(),
        "scope": {
            "case_count": 96, "model_count": 3, "admitted_mechanism_count": 4,
            "group_count_per_model_mechanism": 2, "condition_count_per_group": 4,
            "generation_time_count": GENERATION_TIME_COUNT,
            "role_names": ["source", "query", "answer_start", "current_generation"],
            "scope_is_all_token_positions": False,
            "scope_name": "four_role_dynamic_flow_engineering_pilot",
        },
        "saved_per_layer_time": {
            "role_component_vectors": [
                "layer_input", "input_normalized_state", "attention_output",
                "post_attention_state", "post_attention_normalized_state", "mlp_output", "layer_output",
            ],
            "attention": [
                "value_states_before_output_projection_all_sources",
                "attention_probabilities_all_heads_role_receivers_all_sources",
                "output_projection_weight_reference",
            ],
            "mlp": [
                "gate_pre_at_roles", "up_at_roles", "down_projection_input_product_at_roles",
                "down_projection_weight_reference", "channel_ids",
            ],
            "vocab": ["full_vocabulary_logits_at_current_generation"],
            "quality": [
                "native_dtype_add_order", "attention_source_conservation", "mlp_neuron_conservation",
                "block_conservation", "repeat_hash",
            ],
        },
        "derived_not_saved": {
            "attention_source_residual_writes": "replay_from_values_probabilities_and_output_weight",
            "mlp_single_neuron_residual_writes": "replay_from_product_and_down_weight_columns",
            "condition_contrasts": "only_after_typed_event_alignment_and_path_freeze",
            "public_backbone_residual": "optional_view_raw_events_must_remain",
        },
        "blindness": {
            "collection_reads_family_or_mechanism": False,
            "collection_reads_condition_semantics": False,
            "collection_reads_target_or_distractors": False,
            "private_execution_registry_contains_labels": True,
            "blind_output_strips_private_labels": True,
        },
        "threshold_policy": {
            "fixed_execution_repeat_noise_floor": 0.0,
            "template_and_condition_floor_still_missing": True,
            "mad_only_path_threshold_allowed": False,
            "multiscale_threshold_persistence_counts_as_replication": False,
        },
        "execution_order": list(MODELS),
        "causal_intervention": False,
        "physical_confirmation_opened": False,
    }
    summary = {
        "schema_version": "42.4.0", "phase_id": "Phase365-B", "created_at": now(),
        "denominator": {
            "selected_case_count": len(selected), "selected_group_count": len(selection_rows),
            "model_count": 3, "mechanism_count": 4, "generation_time_count": 3,
            "physical_confirmation_overlap_count": len(overlap),
        },
        "storage": {
            "planning_sequence_length": PLANNING_SEQUENCE_LENGTH,
            "planned_role_focused_bytes": planned_total,
            "naive_explicit_neuron_write_bytes": naive_total,
            "compression_ratio_vs_explicit_neuron_writes": naive_total / max(planned_total, 1),
            "disk_free_bytes": disk_free,
            "minimum_free_reserve_bytes": MINIMUM_FREE_RESERVE_BYTES,
            "fits_with_reserve": planned_total + MINIMUM_FREE_RESERVE_BYTES <= disk_free,
        },
        "quality": {
            "selection_is_hash_frozen": True,
            "all_groups_have_four_conditions": all(row["condition_slot_count"] == 4 for row in selection_rows),
            "registered_mechanism_strata_used_for_denominator": True,
            "model_effects_used_for_selection": False,
            "target_specific_competition_used_for_selection": False,
            "raw_vectors_retained": True,
            "explicit_all_neuron_write_tensor_saved": False,
            "single_neuron_writes_offline_recoverable": True,
        },
        "authorization": {
            "engineering_collection_authorized": planned_total + MINIMUM_FREE_RESERVE_BYTES <= disk_free,
            "language_path_discovery_authorized_after_collection": False,
            "physical_confirmation_authorized": False,
        },
        "next_decision": "implement_collection_writer_then_run_96_engineering_cases_sequentially" if planned_total + MINIMUM_FREE_RESERVE_BYTES <= disk_free else "reduce_storage_before_execution",
    }
    write_jsonl(OUT / "private" / "phase365_engineering_cases.jsonl", selected)
    write_jsonl(OUT / "private" / "phase365_collection_execution_cases.jsonl", blind_execution)
    write_jsonl(OUT / "phase365_blind_group_registry.jsonl", selection_rows)
    write_json(OUT / "phase365_dynamic_collection_schema.json", collection_schema)
    write_json(OUT / "phase365_storage_budget.json", {"models": storage_rows, "global": summary["storage"]})
    write_json(OUT / "phase365_collection_freeze_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
