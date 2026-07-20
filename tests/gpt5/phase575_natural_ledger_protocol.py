#!/usr/bin/env python3
"""Freeze the data-first Phase575 natural component ledger."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase575_source_competition_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
LEDGER_PROTOCOL = OUT_DIR / "phase575_natural_ledger_protocol.json"
MODEL_CONFIG_PATHS = {
    "qwen3": ROOT / "models/hf/qwen3-4b/config.json",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf/config.json",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b/config.json",
}
RECEIVERS = ("query_terminal", "answer_boundary")
SOURCE_GROUPS = (
    "semantic_selected",
    "semantic_other_relation",
    "anchor_base_selected",
    "anchor_base_other_relation",
)
SCALAR_COMPONENTS = (
    "post_rotary_query_norm",
    "layer_input_state_norm",
    "attention_output_norm",
    "source_post_rotary_key_norm",
    "source_value_norm",
    "source_pre_softmax_score_mean",
    "source_post_softmax_weight_mass",
    "source_projected_value_message_norm",
)
PAIR_COMPONENTS = (
    "semantic_score_margin",
    "semantic_weight_margin",
    "semantic_message_norm_margin",
    "anchor_score_margin",
    "anchor_weight_margin",
    "anchor_message_norm_margin",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> Path:
    authorized_models = []
    selected_by_model: dict[str, dict[str, list[str]]] = {}
    layer_count_by_model = {}
    behavior_artifacts = {}
    for model in protocol.MODELS:
        summary_path = OUT_DIR / f"phase575_{model}_behavior_summary.json"
        registry_path = OUT_DIR / f"phase575_{model}_behavior_registry.json"
        summary = read_json(summary_path)
        registry = read_json(registry_path)
        if summary["authorized_for_natural_ledger"] != registry[
            "authorized_for_natural_ledger"
        ]:
            raise RuntimeError(f"Phase575 behavior authorization drift: {model}")
        if summary["authorized_for_natural_ledger"]:
            selected = registry["selected_base_case_ids_by_split"]
            if set(selected) != set(protocol.STRUCTURE_SPLITS):
                raise RuntimeError(f"Phase575 selected split drift: {model}")
            if any(len(selected[split]) != 128 for split in protocol.STRUCTURE_SPLITS):
                raise RuntimeError(f"Phase575 selected count drift: {model}")
            authorized_models.append(model)
            selected_by_model[model] = selected
        config = read_json(MODEL_CONFIG_PATHS[model])
        layer_count_by_model[model] = int(config["num_hidden_layers"])
        behavior_artifacts[model] = {
            "summary_sha256": sha256_file(summary_path),
            "registry_sha256": sha256_file(registry_path),
            "authorized": summary["authorized_for_natural_ledger"],
        }
    if not authorized_models:
        raise RuntimeError("Phase575 has no behavior-qualified model")

    payload = {
        "schema_version": "phase575_natural_ledger_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "authorized_models": authorized_models,
        "selected_base_case_ids_by_model_and_split": selected_by_model,
        "layer_count_by_model": layer_count_by_model,
        "trace_every_layer": True,
        "variants": list(protocol.VARIANTS),
        "receivers": list(RECEIVERS),
        "source_groups": list(SOURCE_GROUPS),
        "scalar_components": list(SCALAR_COMPONENTS),
        "pair_components": list(PAIR_COMPONENTS),
        "post_rotary_query_and_key_required": True,
        "pre_softmax_score_reconstruction_required": True,
        "right_padding_and_explicit_position_ids_required": True,
        "full_vector_snapshot_worlds_in_discovery_per_model": 16,
        "duplicate_trace_audit_worlds_in_discovery_per_model": 8,
        "natural_event_discovery": {
            "no_component_is_predeclared_as_a_mechanism": True,
            "rank_all_frozen_component_receiver_source_layer_coordinates": True,
            "minimum_world_direction_rate_each_split": 0.70,
            "minimum_semantic_selection_rate_each_split": 0.70,
            "minimum_order_preservation_rate_each_split": 0.70,
            "effect_must_exceed_duplicate_trace_floor_multiplier": 10.0,
            "event_must_repeat_in_all_three_structure_splits": True,
            "causal_protocol_can_be_frozen_only_after_replication": True,
        },
        "attention_weight_reconstruction_max_abs_error": 0.01,
        "causal_prefix_max_relative_delta": 1e-5,
        "output_embedding_direction_used": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "behavior_artifacts": behavior_artifacts,
        "phase575_protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
    }
    if LEDGER_PROTOCOL.exists():
        existing = read_json(LEDGER_PROTOCOL)
        ignored = {"created_at"}
        if {k: v for k, v in existing.items() if k not in ignored} != {
            k: v for k, v in payload.items() if k not in ignored
        }:
            raise RuntimeError("Phase575 natural ledger protocol drift")
    else:
        write_json(LEDGER_PROTOCOL, payload)
    print(json.dumps({
        "authorized_models": authorized_models,
        "worlds_per_model": 384,
        "layer_count_by_model": {
            model: layer_count_by_model[model] for model in authorized_models
        },
        "causal_splits_read": False,
        "sealed_split_read": False,
    }, ensure_ascii=False, indent=2))
    return LEDGER_PROTOCOL


if __name__ == "__main__":
    freeze()
