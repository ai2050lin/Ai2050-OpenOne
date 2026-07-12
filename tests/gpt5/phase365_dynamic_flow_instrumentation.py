#!/usr/bin/env python3
"""Model-specific MLP write decomposition and label-safe dynamic bundle schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/schema_and_adapter_gate"
SCHEMA_VERSION = "42.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
EDGE_TYPES = ("route", "write", "merge", "residual", "time", "vocab_transition")
EVENT_TYPES = (
    "attention_source_write", "attention_merge", "mlp_neuron_write",
    "mlp_merge", "residual_state", "residual_merge", "generation_transition", "vocab_state",
)
FORBIDDEN_BLIND_KEYS = {
    "model", "model_name", "family_id", "family_name", "mechanism_id", "mechanism_name",
    "condition_label", "condition_semantics", "correct_answer", "target", "targets", "distractor",
    "distractors", "target_margin", "target_rank", "semantic_label", "historical_candidate",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MLPParts:
    adapter_kind: str
    gate_pre: torch.Tensor
    gate_activated: torch.Tensor
    up: torch.Tensor
    product: torch.Tensor
    down_proj: Any


def decompose_mlp_input(model_key: str, mlp: Any, hidden_states: torch.Tensor) -> MLPParts:
    """Reproduce the exact pre-down-projection product for supported MLP layouts."""
    if model_key in {"qwen3", "deepseek7b"}:
        required = ("gate_proj", "up_proj", "down_proj", "act_fn")
        if not all(hasattr(mlp, name) for name in required):
            raise TypeError(f"{model_key} MLP does not expose {required}")
        gate_pre = mlp.gate_proj(hidden_states)
        gate_activated = mlp.act_fn(gate_pre)
        up = mlp.up_proj(hidden_states)
        return MLPParts(
            adapter_kind="separate_gate_up_silu",
            gate_pre=gate_pre,
            gate_activated=gate_activated,
            up=up,
            product=gate_activated * up,
            down_proj=mlp.down_proj,
        )
    if model_key == "glm4":
        required = ("gate_up_proj", "down_proj", "activation_fn")
        if not all(hasattr(mlp, name) for name in required):
            raise TypeError(f"glm4 MLP does not expose {required}")
        combined = mlp.gate_up_proj(hidden_states)
        gate_pre, up = combined.chunk(2, dim=-1)
        gate_activated = mlp.activation_fn(gate_pre)
        return MLPParts(
            adapter_kind="fused_gate_up_silu",
            gate_pre=gate_pre,
            gate_activated=gate_activated,
            up=up,
            product=up * gate_activated,
            down_proj=mlp.down_proj,
        )
    raise ValueError(f"Unsupported model key: {model_key}")


def iter_neuron_writes(
    product: torch.Tensor,
    down_weight: torch.Tensor,
    channel_ids: Iterable[int] | None = None,
    chunk_size: int = 128,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield channel ids and per-channel residual writes without a full all-neuron tensor."""
    if product.shape[-1] != down_weight.shape[1]:
        raise ValueError("Product width and down-projection input width differ")
    ids = torch.tensor(
        list(range(product.shape[-1])) if channel_ids is None else list(channel_ids),
        device=product.device,
        dtype=torch.long,
    )
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= product.shape[-1]):
        raise IndexError("Channel id outside the MLP intermediate dimension")
    for start in range(0, int(ids.numel()), chunk_size):
        selected = ids[start:start + chunk_size]
        activations = product.index_select(-1, selected)
        columns = down_weight.index_select(1, selected).transpose(0, 1)
        yield selected, activations.unsqueeze(-1) * columns


def replay_mlp_from_neuron_writes(
    parts: MLPParts,
    chunk_size: int = 128,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    reconstructed = torch.zeros(
        (*parts.product.shape[:-1], parts.down_proj.weight.shape[0]),
        device=parts.product.device,
        dtype=accumulation_dtype,
    )
    for _ids, writes in iter_neuron_writes(parts.product, parts.down_proj.weight, chunk_size=chunk_size):
        reconstructed += writes.to(accumulation_dtype).sum(dim=-2)
    if parts.down_proj.bias is not None:
        reconstructed += parts.down_proj.bias.to(accumulation_dtype)
    return reconstructed


def direct_mlp_output(parts: MLPParts) -> torch.Tensor:
    return F.linear(parts.product, parts.down_proj.weight, parts.down_proj.bias)


def relative_error(actual: torch.Tensor, replayed: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - replayed.float())
    denominator = torch.linalg.vector_norm(actual.float()).clamp_min(1e-8)
    return float((numerator / denominator).item())


def weight_reference(
    model_key: str,
    layer_index: int,
    parameter_path: str,
    weight: torch.Tensor,
    checkpoint_index_sha256: str,
) -> dict[str, Any]:
    return {
        "model_key_private": model_key,
        "layer_index": layer_index,
        "parameter_path_private": parameter_path,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype).replace("torch.", ""),
        "checkpoint_index_sha256": checkpoint_index_sha256,
        "reference_id": hashlib.sha256(
            f"{model_key}:{layer_index}:{parameter_path}:{tuple(weight.shape)}:{checkpoint_index_sha256}".encode()
        ).hexdigest(),
        "inline_weight_saved": False,
    }


def _find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in FORBIDDEN_BLIND_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def validate_blind_bundle(bundle: dict[str, Any]) -> list[str]:
    errors = []
    required = {
        "schema_version", "bundle_id", "anonymous_case_id", "anonymous_model_id",
        "anonymous_group_id", "anonymous_condition_slot", "split", "events", "edges",
    }
    missing = sorted(required - set(bundle))
    if missing:
        errors.append(f"missing_root_fields:{','.join(missing)}")
    forbidden = _find_forbidden_keys(bundle)
    if forbidden:
        errors.append(f"forbidden_blind_keys:{','.join(sorted(forbidden))}")
    if bundle.get("split") not in {"blind_discovery", "blind_calibration"}:
        errors.append("invalid_split")
    events = bundle.get("events", [])
    event_ids = set()
    event_required = {
        "event_id", "event_type", "generation_time", "layer_index", "receiver_role",
        "vector_ref", "raw_event_retained",
    }
    for event in events:
        missing_event = sorted(event_required - set(event))
        if missing_event:
            errors.append(f"event_missing_fields:{event.get('event_id','?')}:{','.join(missing_event)}")
        if event.get("event_type") not in EVENT_TYPES:
            errors.append(f"invalid_event_type:{event.get('event_type')}")
        if event.get("event_id") in event_ids:
            errors.append(f"duplicate_event_id:{event.get('event_id')}")
        event_ids.add(event.get("event_id"))
        vector_ref = event.get("vector_ref", {})
        if not {"relative_path", "sha256", "dtype", "shape", "slice"}.issubset(vector_ref):
            errors.append(f"invalid_vector_ref:{event.get('event_id','?')}")
        if not event.get("raw_event_retained", False):
            errors.append(f"raw_event_not_retained:{event.get('event_id','?')}")
    for edge in bundle.get("edges", []):
        if edge.get("edge_type") not in EDGE_TYPES:
            errors.append(f"invalid_edge_type:{edge.get('edge_type')}")
        if edge.get("source_event_id") not in event_ids or edge.get("target_event_id") not in event_ids:
            errors.append(f"dangling_edge:{edge.get('edge_id','?')}")
    return errors


def canonical_event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["generation_time"], event["layer_index"], event["event_type"],
        event.get("source_role"), event["receiver_role"], event.get("head_index"),
        event.get("channel_id"),
    )


def align_bundle_events(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Align typed events before any condition contrast; never subtract graph ids directly."""
    left_events = {canonical_event_key(event): event for event in left["events"]}
    right_events = {canonical_event_key(event): event for event in right["events"]}
    rows = []
    for key in sorted(set(left_events) | set(right_events), key=repr):
        rows.append({
            "canonical_key": list(key),
            "left_event_id": left_events.get(key, {}).get("event_id"),
            "right_event_id": right_events.get(key, {}).get("event_id"),
            "left_present": key in left_events,
            "right_present": key in right_events,
            "vector_subtraction_authorized": key in left_events and key in right_events,
        })
    return rows


def schema_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase365",
        "created_at": now(),
        "event_types": list(EVENT_TYPES),
        "edge_types": list(EDGE_TYPES),
        "blind_discovery": {
            "forbidden_keys": sorted(FORBIDDEN_BLIND_KEYS),
            "anonymous_condition_slot_retained": True,
            "condition_semantics_private_until_path_freeze": True,
            "target_specific_competition_in_event": False,
            "label_free_vocab_state_reference_allowed": True,
        },
        "condition_contrast": {
            "direct_graph_subtraction_allowed": False,
            "typed_event_alignment_required": True,
            "unmatched_events_retained": True,
        },
        "public_backbone": {
            "raw_events_must_be_retained": True,
            "residual_view_may_be_derived": True,
            "residual_view_may_replace_raw": False,
        },
        "thresholds": {
            "mad_only_threshold_authorized": False,
            "repeat_noise_floor_required": True,
            "multiscale_persistence_is_independent_replication": False,
        },
        "weight_reference": {
            "checkpoint_index_hash_required": True,
            "private_parameter_path_required": True,
            "public_exports_strip_model_and_parameter_paths": True,
        },
    }


def main() -> None:
    payload = schema_payload()
    write_json(OUT / "phase365_dynamic_bundle_schema.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
