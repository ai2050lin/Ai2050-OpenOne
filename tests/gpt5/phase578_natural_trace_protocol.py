#!/usr/bin/env python3
"""Freeze the Phase578 all-layer natural trace after behavior qualification."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase578_choice_world_protocol as protocol  # noqa: E402


TRACE_PROTOCOL_PATH = protocol.OUT_DIR / "phase578_natural_trace_protocol.json"
MODEL_CONFIG_PATHS = {
    "qwen3": ROOT / "models/hf/qwen3-4b/config.json",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf/config.json",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b/config.json",
}
SOURCE_GROUPS = ("object", "relation", "target_option", "foil_option")
CHANNELS = (
    "option_score_margin",
    "option_weight_margin",
    "option_message_norm_margin",
    "candidate_input_logit_margin",
    "candidate_output_logit_margin",
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
    selected_by_model = {}
    causal_holdout_hashes = {}
    layer_count_by_model = {}
    behavior_artifacts = {}
    for model in protocol.MODELS:
        summary_path = protocol.OUT_DIR / f"phase578_{model}_behavior_summary.json"
        registry_path = protocol.OUT_DIR / f"phase578_{model}_behavior_registry.json"
        summary = read_json(summary_path)
        registry = read_json(registry_path)
        if summary["natural_trace_authorized"] != registry["natural_trace_authorized"]:
            raise RuntimeError(f"Phase578 behavior authorization drift: {model}")
        if summary["natural_trace_authorized"]:
            selected = registry["natural_trace_world_ids_by_split"]
            if set(selected) != set(protocol.OPEN_SPLITS):
                raise RuntimeError(f"Phase578 natural split drift: {model}")
            if any(
                len(selected[split]) != protocol.NATURAL_TRACE_WORLDS_PER_SPLIT
                for split in protocol.OPEN_SPLITS
            ):
                raise RuntimeError(f"Phase578 natural count drift: {model}")
            authorized_models.append(model)
            selected_by_model[model] = selected
            causal_holdout_hashes[model] = hashlib.sha256(
                json.dumps(
                    registry["causal_holdout_world_ids_by_split"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        config = read_json(MODEL_CONFIG_PATHS[model])
        layer_count_by_model[model] = int(config["num_hidden_layers"])
        behavior_artifacts[model] = {
            "summary_sha256": sha256_file(summary_path),
            "registry_sha256": sha256_file(registry_path),
            "authorized": summary["natural_trace_authorized"],
        }
    if not authorized_models:
        raise RuntimeError("Phase578 has no behavior-qualified model")
    payload = {
        "schema_version": "phase578_natural_trace_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "authorized_models": authorized_models,
        "natural_trace_world_ids_by_model_and_split": selected_by_model,
        "causal_holdout_world_id_hash_by_model": causal_holdout_hashes,
        "layer_count_by_model": layer_count_by_model,
        "trace_every_layer": True,
        "variants": ["target_first", "target_second"],
        "receiver": "answer_boundary",
        "source_groups": list(SOURCE_GROUPS),
        "channels": list(CHANNELS),
        "components": [
            "layer_input", "attention_output", "mlp_output", "layer_output",
            "post_rotary_query", "source_key", "source_value", "source_message",
        ],
        "full_vector_snapshot_worlds_in_discovery_per_model": 8,
        "duplicate_trace_audit_worlds_in_discovery_per_model": 4,
        "natural_event_gate": {
            "rank_all_layers_channels_relations_and_source_groups": True,
            "minimum_target_direction_rate_each_split": 0.70,
            "minimum_option_order_preservation_rate_each_split": 0.70,
            "minimum_relation_specific_world_count_each_split": 24,
            "effect_must_exceed_duplicate_floor_multiplier": 10.0,
            "event_must_repeat_in_discovery_and_confirmation": True,
            "no_channel_is_predeclared_as_mechanism": True,
        },
        "attention_weight_reconstruction_max_abs_error": 0.01,
        "duplicate_trace_max_abs_delta": 0.0,
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "behavior_artifacts": behavior_artifacts,
        "phase578_protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "source_case_sha256": sha256_file(protocol.SOURCE_CASES_PATH),
    }
    write_json(TRACE_PROTOCOL_PATH, payload)
    print(json.dumps({
        "authorized_models": authorized_models,
        "worlds_per_model": sum(
            len(values) for values in next(iter(selected_by_model.values())).values()
        ),
        "layer_count_by_model": {
            model: layer_count_by_model[model] for model in authorized_models
        },
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
    }, ensure_ascii=False, indent=2))
    return TRACE_PROTOCOL_PATH


if __name__ == "__main__":
    freeze()
