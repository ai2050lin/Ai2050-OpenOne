#!/usr/bin/env python3
"""Collect compact multi-position Phase399 dynamic event trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
)
from phase399_dynamic_binding_protocol import MODELS  # noqa: E402
from phase399_dynamic_trace_freeze import ATTENTION_SOURCE_ROLES  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
PRIVATE = OUT / "dynamic_trace/protocol/private"
STAGES = ("instrument", "discovery", "calibration", "physical_holdout")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
EFFECTS = {
    "RO": ("relation_level_private", "order_level_private"),
    "RQ": ("relation_level_private", "query_level_private"),
    "OQ": ("order_level_private", "query_level_private"),
    "ROQ": (
        "relation_level_private",
        "order_level_private",
        "query_level_private",
    ),
}
SOURCE_STATE_ROLES = (
    "source_entity_a",
    "source_entity_b",
    "source_value_a",
    "source_value_b",
    "clause_end_0",
    "clause_end_1",
    "query_entity",
)
RECEIVER_TIMES = ("query_end", "first_answer", "target_completion", "post_target")
MAX_BLOCK_RELATIVE_ERROR = 0.01
MAX_ATTENTION_REPLAY_RELATIVE_ERROR = 0.01
MAX_PROBABILITY_ERROR = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase399 non-finite scalar: {value}")
    return round(value, 9)


def source_path(stage: str) -> Path:
    name = (
        "phase399_instrument_dynamic_trace_cases.jsonl"
        if stage == "instrument"
        else f"phase399_{stage}_dynamic_trace_cases.jsonl"
    )
    return PRIVATE / name


def sign(row: dict[str, Any], fields: tuple[str, ...]) -> float:
    result = 1.0
    for field in fields:
        result *= 1.0 if int(row[field]) == 1 else -1.0
    return result


def repeat_kv(values: torch.Tensor, head_count: int) -> torch.Tensor:
    if values.shape[0] == head_count:
        return values
    if head_count % values.shape[0]:
        raise RuntimeError("Phase399 invalid grouped-query attention head ratio")
    return values.repeat_interleave(head_count // values.shape[0], dim=0)


def partition_for_generation(
    case: dict[str, Any], receiver: int, prompt_length: int
) -> dict[str, list[int]]:
    base = case["attention_source_partitions_private"]["answer_anchor"]
    result = {
        key: [position for position in base[key] if position < prompt_length]
        for key in ATTENTION_SOURCE_ROLES
        if key not in {"other_prior_context", "receiver_self"}
    }
    claimed = set(position for values in result.values() for position in values)
    result["generated_history"] = list(range(prompt_length, receiver))
    claimed.update(result["generated_history"])
    result["receiver_self"] = [receiver]
    claimed.add(receiver)
    result["other_prior_context"] = [
        position for position in range(receiver + 1) if position not in claimed
    ]
    flattened = [position for positions in result.values() for position in positions]
    if sorted(flattened) != list(range(receiver + 1)) or len(flattened) != len(
        set(flattened)
    ):
        raise RuntimeError("Phase399 generation role partition does not conserve prefix")
    return result


def selected_mean(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
    return tensor[0].index_select(0, index).mean(dim=0).detach().float().cpu()


def parent_events(
    captures: dict[tuple[str, int], Any],
    layers: list[Any],
    role_positions: dict[str, list[int]],
    *,
    source_roles: bool,
    receiver_name: str | None,
    local_position: int | None,
) -> tuple[dict[str, list[torch.Tensor]], float]:
    events: dict[str, list[torch.Tensor]] = defaultdict(list)
    max_error = 0.0
    for layer_index in range(len(layers)):
        layer_input = captures[("layer_input", layer_index)]
        attention = captures[("attention_output", layer_index)]
        mlp = captures[("mlp_output", layer_index)]
        output = captures[("layer_output", layer_index)]
        if source_roles:
            for role in SOURCE_STATE_ROLES:
                vector = selected_mean(output, role_positions[role])
                events[f"state:{role}:layer_output"].append(vector)
        if receiver_name is not None and local_position is not None:
            position = local_position if local_position >= 0 else layer_input.shape[1] + local_position
            vectors = {
                "layer_input": layer_input[0, position].detach().float().cpu(),
                "attention_output": attention[0, position].detach().float().cpu(),
                "mlp_output": mlp[0, position].detach().float().cpu(),
                "layer_output": output[0, position].detach().float().cpu(),
            }
            reconstructed = vectors["layer_input"] + vectors["attention_output"] + vectors["mlp_output"]
            error = float(torch.linalg.vector_norm(vectors["layer_output"] - reconstructed).item())
            scale = float(torch.linalg.vector_norm(vectors["layer_output"]).item())
            max_error = max(max_error, error / max(scale, 1e-8))
            for component, vector in vectors.items():
                events[f"state:{receiver_name}:{component}"].append(vector)
    return dict(events), max_error


def route_events(
    captures: dict[tuple[str, int], Any],
    layers: list[Any],
    receiver_name: str,
    receiver_local_position: int,
    partitions: dict[str, list[int]],
    value_history: list[torch.Tensor],
) -> tuple[
    dict[str, list[torch.Tensor]],
    dict[str, list[float]],
    float,
    float,
]:
    vectors: dict[str, list[torch.Tensor]] = defaultdict(list)
    masses: dict[str, list[float]] = defaultdict(list)
    max_replay_error = 0.0
    max_probability_error = 0.0
    for layer_index, layer in enumerate(layers):
        probabilities = captures[("attention_probabilities", layer_index)]
        local = (
            receiver_local_position
            if receiver_local_position >= 0
            else probabilities.shape[2] + receiver_local_position
        )
        selected_probs = probabilities[0, :, local].float()
        all_values = value_history[layer_index].float()
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        head_count = int(selected_probs.shape[0])
        head_dim = int(o_proj.weight.shape[1] // head_count)
        repeated = repeat_kv(all_values, head_count)
        if repeated.shape[1] != selected_probs.shape[1]:
            raise RuntimeError("Phase399 attention source/value length mismatch")
        o_blocks = o_proj.weight.float().view(o_proj.weight.shape[0], head_count, head_dim)
        reconstructed = torch.zeros(
            o_proj.weight.shape[0], device=selected_probs.device, dtype=torch.float32
        )
        for role, positions in partitions.items():
            if positions:
                index = torch.tensor(positions, dtype=torch.long, device=selected_probs.device)
                role_probs = selected_probs.index_select(1, index)
                role_values = repeated.index_select(1, index)
                weighted = torch.einsum("hs,hsd->hd", role_probs, role_values)
                write = torch.einsum("hd,ohd->o", weighted, o_blocks)
                mass = float(role_probs.sum(dim=-1).mean().item())
            else:
                write = torch.zeros_like(reconstructed)
                mass = 0.0
            reconstructed += write
            event_id = f"route:{role}->{receiver_name}:attention_write"
            vectors[event_id].append(write.detach().cpu())
            masses[event_id].append(mass)
        bias = (
            o_proj.bias.float()
            if o_proj.bias is not None
            else torch.zeros_like(reconstructed)
        )
        reconstructed += bias
        bias_id = f"route:projection_bias->{receiver_name}:attention_write"
        vectors[bias_id].append(bias.detach().cpu())
        masses[bias_id].append(0.0)
        actual_tensor = captures[("attention_output", layer_index)]
        actual_pos = (
            receiver_local_position
            if receiver_local_position >= 0
            else actual_tensor.shape[1] + receiver_local_position
        )
        actual = actual_tensor[0, actual_pos].detach().float()
        replay_error = float(torch.linalg.vector_norm(actual - reconstructed).item())
        scale = float(torch.linalg.vector_norm(actual).item())
        max_replay_error = max(max_replay_error, replay_error / max(scale, 1e-8))
        probability_error = float((selected_probs.sum(dim=-1) - 1).abs().max().item())
        max_probability_error = max(max_probability_error, probability_error)
    return dict(vectors), dict(masses), max_replay_error, max_probability_error


def merge_events(
    target: dict[str, list[torch.Tensor]], source: dict[str, list[torch.Tensor]]
) -> None:
    overlap = set(target).intersection(source)
    if overlap:
        raise RuntimeError(f"Phase399 duplicate event ids: {sorted(overlap)[:3]}")
    target.update(source)


@torch.inference_mode()
def collect_case(
    loaded: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    ids = torch.tensor(
        [case["prompt_token_ids_private"]], dtype=torch.long, device=loaded.input_device
    )
    captures.clear()
    output = loaded.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    prompt_length = int(ids.shape[1])
    first_match = int(torch.argmax(output.logits[0, -1]).item()) == int(
        case["first_answer_token_id_private"]
    )
    values: list[torch.Tensor] = []
    for layer_index, layer in enumerate(layers):
        projection = captures[("value_projection", layer_index)]
        probabilities = captures[("attention_probabilities", layer_index)]
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        head_count = int(probabilities.shape[1])
        head_dim = int(o_proj.weight.shape[1] // head_count)
        kv_heads = int(projection.shape[-1] // head_dim)
        values.append(
            projection.view(1, prompt_length, kv_heads, head_dim)[0]
            .transpose(0, 1)
            .detach()
        )
    events, block_error = parent_events(
        captures,
        layers,
        case["state_role_positions_private"],
        source_roles=True,
        receiver_name="query_end",
        local_position=case["state_role_positions_private"]["query_end"][0],
    )
    first_events, first_block = parent_events(
        captures,
        layers,
        case["state_role_positions_private"],
        source_roles=False,
        receiver_name="first_answer",
        local_position=case["state_role_positions_private"]["answer_anchor"][0],
    )
    merge_events(events, first_events)
    route_mass: dict[str, list[float]] = {}
    query_route, query_mass, query_replay, query_prob = route_events(
        captures,
        layers,
        "query_end",
        case["state_role_positions_private"]["query_end"][0],
        case["attention_source_partitions_private"]["query_end"],
        values,
    )
    first_route, first_mass, first_replay, first_prob = route_events(
        captures,
        layers,
        "first_answer",
        case["state_role_positions_private"]["answer_anchor"][0],
        case["attention_source_partitions_private"]["answer_anchor"],
        values,
    )
    merge_events(events, query_route)
    merge_events(events, first_route)
    route_mass.update(query_mass)
    route_mass.update(first_mass)

    past = output.past_key_values
    total_length = prompt_length
    completion_match = None
    target_events: dict[str, list[torch.Tensor]] | None = None
    target_mass: dict[str, list[float]] | None = None
    target_block = 0.0
    target_replay = 0.0
    target_prob = 0.0
    prefix = [int(value) for value in case["target_completion_prefix_token_ids_private"]]
    if not prefix:
        completion_match = first_match
        target_events = {
            key.replace("first_answer", "target_completion"): value
            for key, value in first_events.items()
        }
        target_route = {
            key.replace("first_answer", "target_completion"): value
            for key, value in first_route.items()
        }
        target_mass = {
            key.replace("first_answer", "target_completion"): value
            for key, value in first_mass.items()
        }
        # Keep the semantic event ledger identical when completion happens at
        # the first answer token: generated history exists as an empty role.
        zero_event = "route:generated_history->target_completion:attention_write"
        reference = next(iter(target_route.values()))
        target_route[zero_event] = [torch.zeros_like(vector) for vector in reference]
        target_mass[zero_event] = [0.0] * len(reference)
        merge_events(target_events, target_route)
        target_block, target_replay, target_prob = first_block, first_replay, first_prob
    else:
        for index, token_id in enumerate(prefix):
            total_length += 1
            captures.clear()
            token = torch.tensor([[token_id]], dtype=torch.long, device=loaded.input_device)
            output = loaded.model(
                input_ids=token,
                attention_mask=torch.ones(
                    (1, total_length), dtype=torch.long, device=loaded.input_device
                ),
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
            past = output.past_key_values
            for layer_index, layer in enumerate(layers):
                projection = captures[("value_projection", layer_index)]
                probabilities = captures[("attention_probabilities", layer_index)]
                o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
                head_count = int(probabilities.shape[1])
                head_dim = int(o_proj.weight.shape[1] // head_count)
                kv_heads = int(projection.shape[-1] // head_dim)
                current = projection.view(1, 1, kv_heads, head_dim)[0].transpose(0, 1)
                values[layer_index] = torch.cat(
                    [values[layer_index], current.detach()], dim=1
                )
            if index == len(prefix) - 1:
                completion_match = int(torch.argmax(output.logits[0, -1]).item()) == int(
                    case["target_completion_token_id_private"]
                )
                target_events, target_block = parent_events(
                    captures,
                    layers,
                    case["state_role_positions_private"],
                    source_roles=False,
                    receiver_name="target_completion",
                    local_position=-1,
                )
                partition = partition_for_generation(case, total_length - 1, prompt_length)
                target_route, target_mass, target_replay, target_prob = route_events(
                    captures,
                    layers,
                    "target_completion",
                    -1,
                    partition,
                    values,
                )
                merge_events(target_events, target_route)

    if target_events is None or target_mass is None or completion_match is None:
        raise RuntimeError("Phase399 target-completion event was not captured")
    merge_events(events, target_events)
    route_mass.update(target_mass)

    completion_token = int(case["target_completion_token_id_private"])
    total_length += 1
    captures.clear()
    token = torch.tensor([[completion_token]], dtype=torch.long, device=loaded.input_device)
    output = loaded.model(
        input_ids=token,
        attention_mask=torch.ones(
            (1, total_length), dtype=torch.long, device=loaded.input_device
        ),
        past_key_values=past,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    post_match = int(torch.argmax(output.logits[0, -1]).item()) == int(
        case["post_target_next_token_id_private"]
    )
    for layer_index, layer in enumerate(layers):
        projection = captures[("value_projection", layer_index)]
        probabilities = captures[("attention_probabilities", layer_index)]
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        head_count = int(probabilities.shape[1])
        head_dim = int(o_proj.weight.shape[1] // head_count)
        kv_heads = int(projection.shape[-1] // head_dim)
        current = projection.view(1, 1, kv_heads, head_dim)[0].transpose(0, 1)
        values[layer_index] = torch.cat([values[layer_index], current.detach()], dim=1)
    post_events, post_block = parent_events(
        captures,
        layers,
        case["state_role_positions_private"],
        source_roles=False,
        receiver_name="post_target",
        local_position=-1,
    )
    post_partition = partition_for_generation(case, total_length - 1, prompt_length)
    post_route, post_mass, post_replay, post_prob = route_events(
        captures,
        layers,
        "post_target",
        -1,
        post_partition,
        values,
    )
    merge_events(post_events, post_route)
    merge_events(events, post_events)
    route_mass.update(post_mass)
    del output, past, ids, token
    captures.clear()
    return {
        "case": case,
        "events": events,
        "route_mass": route_mass,
        "first_answer_replay_match": first_match,
        "target_completion_replay_match": completion_match,
        "post_target_replay_match": post_match,
        "max_block_relative_error": max(block_error, first_block, target_block, post_block),
        "max_attention_replay_relative_error": max(
            query_replay, first_replay, target_replay, post_replay
        ),
        "max_probability_sum_error": max(query_prob, first_prob, target_prob, post_prob),
    }


def event_metadata(event_id: str) -> dict[str, Any]:
    parts = event_id.split(":")
    if parts[0] == "state":
        return {
            "event_kind": "state_parent_component",
            "coordinate": parts[1],
            "component": parts[2],
            "source_role": None,
            "receiver_role": parts[1],
        }
    source, receiver = parts[1].split("->")
    return {
        "event_kind": "role_partitioned_attention_write",
        "coordinate": receiver,
        "component": parts[2],
        "source_role": source,
        "receiver_role": receiver,
    }


def summarize_group(
    model: str,
    stage: str,
    collected: list[dict[str, Any]],
    layer_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(collected) != 16:
        raise RuntimeError(f"Phase399 group must contain 16 cases, got {len(collected)}")
    by_axis = {
        axis: sorted(
            [item for item in collected if item["case"]["axis_private"] == axis],
            key=lambda item: item["case"]["anonymous_condition_slot"],
        )
        for axis in ("X", "Y")
    }
    if any(len(items) != 8 for items in by_axis.values()):
        raise RuntimeError("Phase399 group must contain two complete factorial axes")
    event_ids = set(collected[0]["events"])
    if any(set(item["events"]) != event_ids for item in collected):
        raise RuntimeError("Phase399 event identity changed within factorial group")
    base = collected[0]["case"]
    rows: list[dict[str, Any]] = []
    for event_id in sorted(event_ids):
        metrics = {
            effect: {"min_axis_normalized_norm": [], "cross_axis_cosine": []}
            for effect in EFFECTS
        }
        mass_metrics = {
            effect: {"min_axis_absolute_effect": [], "same_sign": []}
            for effect in EFFECTS
        }
        for layer_index in range(layer_count):
            axis_vectors: dict[str, dict[str, torch.Tensor]] = {}
            axis_scales: dict[str, float] = {}
            axis_mass: dict[str, dict[str, float]] = {}
            for axis, items in by_axis.items():
                vectors = [item["events"][event_id][layer_index] for item in items]
                axis_scales[axis] = sum(
                    float(torch.linalg.vector_norm(vector).item()) for vector in vectors
                ) / len(vectors)
                axis_vectors[axis] = {
                    effect: torch.stack(
                        [
                            vector * sign(item["case"], fields)
                            for item, vector in zip(items, vectors, strict=True)
                        ]
                    ).mean(dim=0)
                    for effect, fields in EFFECTS.items()
                }
                masses = [
                    item["route_mass"].get(event_id, [0.0] * layer_count)[layer_index]
                    for item in items
                ]
                axis_mass[axis] = {
                    effect: sum(
                        value * sign(item["case"], fields)
                        for item, value in zip(items, masses, strict=True)
                    )
                    / len(masses)
                    for effect, fields in EFFECTS.items()
                }
            for effect in EFFECTS:
                x, y = axis_vectors["X"][effect], axis_vectors["Y"][effect]
                x_norm = float(torch.linalg.vector_norm(x).item())
                y_norm = float(torch.linalg.vector_norm(y).item())
                normalized = min(
                    x_norm / max(axis_scales["X"], 1e-8),
                    y_norm / max(axis_scales["Y"], 1e-8),
                )
                cosine = float(
                    F.cosine_similarity(
                        x.unsqueeze(0), y.unsqueeze(0), dim=-1, eps=1e-8
                    ).item()
                )
                metrics[effect]["min_axis_normalized_norm"].append(clean(normalized))
                metrics[effect]["cross_axis_cosine"].append(clean(cosine))
                x_mass, y_mass = axis_mass["X"][effect], axis_mass["Y"][effect]
                mass_metrics[effect]["min_axis_absolute_effect"].append(
                    clean(min(abs(x_mass), abs(y_mass)))
                )
                mass_metrics[effect]["same_sign"].append(
                    bool(x_mass * y_mass > 0)
                )
        roq = metrics["ROQ"]["min_axis_normalized_norm"]
        roq_cos = metrics["ROQ"]["cross_axis_cosine"]
        ratios = []
        for layer_index in range(layer_count):
            competitor = max(
                metrics[effect]["min_axis_normalized_norm"][layer_index]
                for effect in ("RO", "RQ", "OQ")
            )
            ratios.append(clean(roq[layer_index] / max(competitor, 1e-8)))
        scores = [
            roq[index] * max(roq_cos[index], 0.0) for index in range(layer_count)
        ]
        peak = max(range(layer_count), key=lambda index: scores[index])
        first = next(
            (
                index
                for index in range(layer_count)
                if roq[index] >= 0.01 and roq_cos[index] >= 0.50
            ),
            None,
        )
        rows.append(
            {
                "schema_version": "73.4.0",
                "phase_id": "Phase399-DynamicTraceCollection",
                "created_at": now(),
                "model": model,
                "stage": stage,
                "public_parallel_group_id": base["phase399_public_parallel_group_id"],
                "surface_private": base["task_surface_private"],
                "event_id": event_id,
                **event_metadata(event_id),
                "layer_count": layer_count,
                "interaction_trajectories": metrics,
                "attention_mass_trajectories": mass_metrics,
                "roq_to_strongest_competing_interaction": ratios,
                "first_replicated_roq_layer": first,
                "peak_roq_layer": peak,
                "peak_roq_relative_depth": clean(peak / max(layer_count - 1, 1)),
                "peak_roq_min_axis_normalized_norm": roq[peak],
                "peak_roq_cross_axis_cosine": roq_cos[peak],
                "raw_vectors_persisted": False,
                "head_identity_persisted": False,
                "causal_intervention": False,
            }
        )
    max_block = max(item["max_block_relative_error"] for item in collected)
    max_route = max(
        item["max_attention_replay_relative_error"] for item in collected
    )
    max_prob = max(item["max_probability_sum_error"] for item in collected)
    audit = {
        "schema_version": "73.4.0",
        "phase_id": "Phase399-DynamicTraceGroupAudit",
        "created_at": now(),
        "model": model,
        "stage": stage,
        "public_parallel_group_id": base["phase399_public_parallel_group_id"],
        "surface_private": base["task_surface_private"],
        "case_count": len(collected),
        "event_count_per_case": len(event_ids),
        "first_answer_replay_match_count": sum(
            item["first_answer_replay_match"] for item in collected
        ),
        "target_completion_replay_match_count": sum(
            item["target_completion_replay_match"] for item in collected
        ),
        "post_target_replay_match_count": sum(
            item["post_target_replay_match"] for item in collected
        ),
        "max_block_relative_error": clean(max_block),
        "max_attention_replay_relative_error": clean(max_route),
        "max_probability_sum_error": clean(max_prob),
        "quality_gate_pass": (
            max_block <= MAX_BLOCK_RELATIVE_ERROR
            and max_route <= MAX_ATTENTION_REPLAY_RELATIVE_ERROR
            and max_prob <= MAX_PROBABILITY_ERROR
            and all(item["first_answer_replay_match"] for item in collected)
            and all(item["target_completion_replay_match"] for item in collected)
            and all(item["post_target_replay_match"] for item in collected)
        ),
    }
    return rows, audit


def run(
    model: str,
    stage: str,
    shard_index: int | None = None,
    shard_size: int = 3,
) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(source_path(stage))
        if row["private_execution_model"] == model
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["anonymous_parallel_group_id"]].append(case)
    all_group_ids = sorted(grouped)
    if shard_index is not None:
        if shard_index < 0 or shard_size <= 0:
            raise ValueError("Phase399 shard index and size must be positive")
        start = shard_index * shard_size
        chosen_group_ids = all_group_ids[start : start + shard_size]
        if not chosen_group_ids:
            raise RuntimeError(
                f"Phase399 empty shard {shard_index} for {model}/{stage}"
            )
    else:
        chosen_group_ids = all_group_ids
    loaded = None
    handles: list[Any] = []
    value_handles: list[Any] = []
    event_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for layer_index, layer in enumerate(layers):
            value_proj = module_attr(layer.self_attn, ("v_proj", "value"))

            def value_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                idx: int = layer_index,
            ) -> None:
                captures[("value_projection", idx)] = output.detach()

            value_handles.append(value_proj.register_forward_hook(value_post))
        for group_index, group_id in enumerate(chosen_group_ids, 1):
            group_cases = sorted(
                grouped[group_id], key=lambda row: row["anonymous_condition_slot"]
            )
            collected = [
                collect_case(loaded, layers, captures, case) for case in group_cases
            ]
            rows, audit = summarize_group(
                model, stage, collected, len(layers)
            )
            event_rows.extend(rows)
            group_rows.append(audit)
            del collected
            gc.collect()
            print(
                f"[{model}/{stage}] Phase399 group {group_index}/{len(chosen_group_ids)} "
                f"gate={audit['quality_gate_pass']}",
                flush=True,
            )
        model_root = OUT / "dynamic_trace" / stage / "private/models" / model
        if shard_index is not None:
            model_root = model_root / "shards" / f"shard_{shard_index:03d}"
        write_jsonl(model_root / "event_trajectory_rows.jsonl", event_rows)
        write_jsonl(model_root / "group_audit_rows.jsonl", group_rows)
        complete = {
            "schema_version": "73.4.0",
            "phase_id": "Phase399-DynamicTraceCollection",
            "created_at": now(),
            "model": model,
            "stage": stage,
            "shard_index": shard_index,
            "shard_size": shard_size if shard_index is not None else None,
            "stage_total_group_count": len(all_group_ids),
            "selected_group_ids": chosen_group_ids,
            "case_count": len(cases),
            "selected_case_count": len(chosen_group_ids) * 16,
            "group_count": len(chosen_group_ids),
            "layer_count": len(layers),
            "event_trajectory_row_count": len(event_rows),
            "quality_group_count": sum(row["quality_gate_pass"] for row in group_rows),
            "max_block_relative_error": max(
                row["max_block_relative_error"] for row in group_rows
            ),
            "max_attention_replay_relative_error": max(
                row["max_attention_replay_relative_error"] for row in group_rows
            ),
            "max_probability_sum_error": max(
                row["max_probability_sum_error"] for row in group_rows
            ),
            "valid": bool(chosen_group_ids)
            and all(len(grouped[group]) == 16 for group in chosen_group_ids),
        }
        write_json(model_root / "complete.json", complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    finally:
        for handle in value_handles + handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-size", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.model, args.stage, args.shard_index, args.shard_size),
            ensure_ascii=False,
            indent=2,
        )
    )
