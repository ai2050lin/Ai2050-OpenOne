#!/usr/bin/env python3
"""Phase 1001 exact source-token path decomposition for frozen Qwen3 heads.

This script does not assume a language mechanism formula. It uses the exact
attention identity at the answer boundary to split each frozen head by
mutually exclusive source-position roles, then tests every role path with
necessity and sufficiency interventions. Validation discovers and freezes a
route ordering; confirmation applies that ordering to disjoint worlds.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import COLORS, MODEL
from phase1000_scpg_confirmation import one_direction_per_pair
from phase1000_scpg_discovery import (
    batches_by_template,
    candidate_tensor,
    capture_residuals,
    case_tensors,
    prediction_colors,
    read_jsonl,
    register_source_patch,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1000_source_control_audit import valid_derangement_shifts
from phase1001_attention_head_discovery import (
    HEAD_COUNT,
    HEAD_DIM,
    PHASE1000_ROOT,
    RESULT_ROOT,
    SOURCE_DEPTH,
    TARGET_LAYERS,
    forward_with_patches,
    generate_with_patches,
    read_json,
    selected_phase1000_inputs,
    write_json,
)


PHASE_ID = 1001
NUM_KV_HEADS = 8
KV_GROUP_SIZE = HEAD_COUNT // NUM_KV_HEADS
ROLE_NAMES = (
    "record_queried_entity",
    "record_queried_value",
    "record_alternative_entity",
    "record_alternative_value",
    "chat_prefix",
    "record_relation_context",
    "query_name",
    "query_context",
    "output_instruction",
    "assistant_protocol",
    "answer_boundary",
)
TEMPLATE_SEGMENTS = {
    0: {
        "record_relation_context": (3, 16),
        "query_context": (17, 25),
        "output_instruction": (26, 32),
        "assistant_protocol": (33, 40),
    },
    1: {
        "record_relation_context": (3, 21),
        "query_context": (22, 31),
        "output_instruction": (32, 38),
        "assistant_protocol": (39, 46),
    },
    2: {
        "record_relation_context": (3, 17),
        "query_context": (18, 30),
        "output_instruction": (31, 34),
        "assistant_protocol": (35, 42),
    },
    3: {
        "record_relation_context": (3, 21),
        "query_context": (22, 30),
        "output_instruction": (31, 36),
        "assistant_protocol": (37, 44),
    },
}
JOINT_ROUTE_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 132)
PATH_THRESHOLDS = {
    "max_head_reconstruction_error": 0.05,
    "max_role_delta_reconstruction_error": 0.05,
    "max_qkv_identity_error": 1e-4,
    "single_median_mediation": 0.01,
    "single_mean_sufficiency": 0.005,
    "single_location_excess": 0.005,
    "joint_median_mediation": 0.30,
    "joint_mean_sufficiency": 0.30,
    "joint_natural_restoration": 0.50,
    "source_do_natural_flip": 0.90,
}
SOURCE_ROOT = RESULT_ROOT / "source_path_decomposition"
HEAD_DISCOVERY_ROOT = RESULT_ROOT / "head_discovery"
HEAD_CONFIRMATION_ROOT = RESULT_ROOT / "head_confirmation"


def event_from_id(event_id: str) -> dict[str, Any]:
    left, right = event_id.split(".")
    return {
        "event_id": event_id,
        "layer_number": int(left[1:]),
        "head_index": int(right[1:]),
        "role": "answer_boundary",
    }


def route_id(event_id: str, source_role: str) -> str:
    return f"{event_id}/{source_role}"


def structural_roles(row: dict[str, Any]) -> dict[str, list[int]]:
    query_slot = int(row["query_slot"])
    alternative_slot = 1 - query_slot
    positions = row["role_positions"]
    roles = {
        "record_queried_entity": [
            int(positions[f"slot{query_slot}_entity"])
        ],
        "record_queried_value": [
            int(positions[f"slot{query_slot}_color"])
        ],
        "record_alternative_entity": [
            int(positions[f"slot{alternative_slot}_entity"])
        ],
        "record_alternative_value": [
            int(positions[f"slot{alternative_slot}_color"])
        ],
        "chat_prefix": [0, 1, 2],
        "query_name": [int(positions["query_name"])],
        "answer_boundary": [int(positions["answer_boundary"])],
    }
    occupied = {
        position
        for values in roles.values()
        for position in values
    }
    for role, (start, stop) in TEMPLATE_SEGMENTS[int(row["template"])].items():
        roles[role] = [
            position
            for position in range(start, stop + 1)
            if position not in occupied
        ]
        occupied.update(roles[role])
    ordered = {role: roles[role] for role in ROLE_NAMES}
    flat = [position for values in ordered.values() for position in values]
    if len(flat) != len(set(flat)):
        raise RuntimeError("source role overlap")
    if sorted(flat) != list(range(len(row["input_ids"]))):
        missing = sorted(set(range(len(row["input_ids"]))) - set(flat))
        extra = sorted(set(flat) - set(range(len(row["input_ids"]))))
        raise RuntimeError(
            f"source role partition drift: missing={missing}, extra={extra}"
        )
    return ordered


def selected_inputs(stage: str):
    if stage == "discovery":
        protocol, _, selected_pairs, directional, _ = (
            selected_phase1000_inputs("formal")
        )
        output_root = SOURCE_ROOT / "discovery"
        return protocol, selected_pairs, directional, output_root

    protocol = read_json(PHASE1000_ROOT / "protocol" / "protocol.json")
    cases = read_jsonl(PHASE1000_ROOT / "protocol" / "cases.jsonl")
    selected_pairs = read_jsonl(
        HEAD_CONFIRMATION_ROOT / "selected_pairs.jsonl"
    )
    case_by_id = {row["record_id"]: row for row in cases}
    directional = one_direction_per_pair(selected_pairs, case_by_id)
    output_root = SOURCE_ROOT / "confirmation"
    return protocol, selected_pairs, directional, output_root


def capture_physical_attention(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any] | None = None,
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    input_ids, attention_mask = case_tensors(rows, device)
    values: dict[int, torch.Tensor] = {}
    weights: dict[int, torch.Tensor] = {}
    head_outputs: dict[int, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    handles = []
    source_handle = None
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        for layer_number in TARGET_LAYERS:
            layer = layers[layer_number - 1]
            answer_positions = torch.tensor(
                [
                    row["role_positions"]["answer_boundary"]
                    for row in rows
                ],
                dtype=torch.long,
                device=device,
            )

            def make_value_hook(number):
                def hook(module, args, output):
                    values[number] = (
                        output.detach()
                        .reshape(
                            output.shape[0],
                            output.shape[1],
                            NUM_KV_HEADS,
                            HEAD_DIM,
                        )
                    )
                    counts[f"value/{number}"] += 1

                return hook

            def make_output_hook(number, positions):
                def hook(module, args):
                    value = args[0]
                    batch_index = torch.arange(
                        value.shape[0], device=value.device
                    )
                    head_outputs[number] = (
                        value[
                            batch_index,
                            positions.to(value.device),
                            :,
                        ]
                        .reshape(value.shape[0], HEAD_COUNT, HEAD_DIM)
                        .detach()
                    )
                    counts[f"output/{number}"] += 1

                return hook

            def make_weight_hook(number, positions):
                def hook(module, args, output):
                    if (
                        not isinstance(output, tuple)
                        or len(output) < 2
                        or output[1] is None
                    ):
                        raise RuntimeError("attention weights unavailable")
                    value = output[1]
                    batch_index = torch.arange(
                        value.shape[0], device=value.device
                    )
                    weights[number] = value[
                        batch_index,
                        :,
                        positions.to(value.device),
                        :,
                    ].detach()
                    counts[f"weight/{number}"] += 1

                return hook

            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    make_value_hook(layer_number)
                )
            )
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    make_output_hook(layer_number, answer_positions)
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_weight_hook(layer_number, answer_positions)
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        candidates = candidate_tensor(
            output.logits[:, -1, :], candidate_ids
        ).detach()
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError(f"source patch count drift: {source_count[0]}")
        expected = {
            f"{kind}/{layer_number}"
            for kind in ("value", "weight", "output")
            for layer_number in TARGET_LAYERS
        }
        bad = {key: counts[key] for key in expected if counts[key] != 1}
        if bad:
            raise RuntimeError(f"physical capture count drift: {bad}")
        del output, input_ids, attention_mask
        return candidates, values, weights, head_outputs
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def component_map_for_batch(
    batch: list[dict[str, Any]],
    events: list[dict[str, Any]],
    target_values: dict[int, torch.Tensor],
    target_weights: dict[int, torch.Tensor],
    target_heads: dict[int, torch.Tensor],
    do_values: dict[int, torch.Tensor],
    do_weights: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
) -> tuple[
    dict[str, dict[str, dict[str, torch.Tensor]]],
    list[dict[str, Any]],
]:
    components: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    audit_rows = []
    for event in events:
        event_id = event["event_id"]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        kv_head = head_index // KV_GROUP_SIZE
        target_attention = target_weights[layer_number][
            :, head_index, :
        ].float()
        do_attention = do_weights[layer_number][:, head_index, :].float()
        target_value = target_values[layer_number][
            :, :, kv_head, :
        ].float()
        do_value = do_values[layer_number][:, :, kv_head, :].float()
        target_position = target_attention[:, :, None] * target_value
        do_position = do_attention[:, :, None] * do_value
        delta_attention = do_attention - target_attention
        delta_value = do_value - target_value
        qk_position = delta_attention[:, :, None] * target_value
        value_position = target_attention[:, :, None] * delta_value
        interaction_position = (
            delta_attention[:, :, None] * delta_value
        )

        target_rebuilt = target_position.sum(dim=1)
        do_rebuilt = do_position.sum(dim=1)
        target_error = (
            target_rebuilt
            - target_heads[layer_number][:, head_index, :].float()
        ).abs()
        do_error = (
            do_rebuilt
            - do_heads[layer_number][:, head_index, :].float()
        ).abs()
        by_role = {
            role: {
                "target": torch.zeros(
                    len(batch), HEAD_DIM, device=target_value.device
                ),
                "do": torch.zeros(
                    len(batch), HEAD_DIM, device=target_value.device
                ),
                "delta": torch.zeros(
                    len(batch), HEAD_DIM, device=target_value.device
                ),
                "qk": torch.zeros(
                    len(batch), HEAD_DIM, device=target_value.device
                ),
                "value": torch.zeros(
                    len(batch), HEAD_DIM, device=target_value.device
                ),
                "interaction": torch.zeros(
                    len(batch), HEAD_DIM, device=target_value.device
                ),
                "target_attention_mass": torch.zeros(
                    len(batch), device=target_value.device
                ),
                "do_attention_mass": torch.zeros(
                    len(batch), device=target_value.device
                ),
            }
            for role in ROLE_NAMES
        }
        for index, item in enumerate(batch):
            roles = structural_roles(item["target"])
            for role, positions in roles.items():
                by_role[role]["target"][index] = target_position[
                    index, positions, :
                ].sum(dim=0)
                by_role[role]["do"][index] = do_position[
                    index, positions, :
                ].sum(dim=0)
                by_role[role]["delta"][index] = (
                    by_role[role]["do"][index]
                    - by_role[role]["target"][index]
                )
                by_role[role]["qk"][index] = qk_position[
                    index, positions, :
                ].sum(dim=0)
                by_role[role]["value"][index] = value_position[
                    index, positions, :
                ].sum(dim=0)
                by_role[role]["interaction"][index] = (
                    interaction_position[index, positions, :].sum(dim=0)
                )
                by_role[role]["target_attention_mass"][index] = (
                    target_attention[index, positions].sum()
                )
                by_role[role]["do_attention_mass"][index] = (
                    do_attention[index, positions].sum()
                )
        role_delta = torch.stack(
            [by_role[role]["delta"] for role in ROLE_NAMES],
            dim=0,
        ).sum(dim=0)
        direct_delta = (
            do_heads[layer_number][:, head_index, :].float()
            - target_heads[layer_number][:, head_index, :].float()
        )
        qkv_delta = torch.stack(
            [
                by_role[role]["qk"]
                + by_role[role]["value"]
                + by_role[role]["interaction"]
                for role in ROLE_NAMES
            ],
            dim=0,
        ).sum(dim=0)
        audit_rows.append(
            {
                "event_id": event_id,
                "layer_number": layer_number,
                "head_index": head_index,
                "n": len(batch),
                "target_max_abs_head_reconstruction_error": float(
                    target_error.max()
                ),
                "do_max_abs_head_reconstruction_error": float(
                    do_error.max()
                ),
                "max_abs_role_delta_reconstruction_error": float(
                    (role_delta - direct_delta).abs().max()
                ),
                "max_abs_qkv_identity_error": float(
                    (qkv_delta - role_delta).abs().max()
                ),
            }
        )
        components[event_id] = by_role
    return components, audit_rows


def direct_logit_effect(
    model,
    event: dict[str, Any],
    vectors: torch.Tensor,
    batch: list[dict[str, Any]],
    candidate_ids: dict[str, int],
) -> torch.Tensor:
    layer_number = int(event["layer_number"])
    head_index = int(event["head_index"])
    output_weight = model.get_output_embeddings().weight.detach().float()
    color_index = {color: index for index, color in enumerate(COLORS)}
    candidate_unembed = output_weight[
        torch.tensor(
            [candidate_ids[color] for color in COLORS],
            device=output_weight.device,
        )
    ]
    source_index = torch.tensor(
        [color_index[item["source"]["gold"]] for item in batch],
        device=output_weight.device,
    )
    target_index = torch.tensor(
        [color_index[item["target"]["gold"]] for item in batch],
        device=output_weight.device,
    )
    unembed_direction = (
        candidate_unembed[source_index] - candidate_unembed[target_index]
    )
    output_projection = (
        model.model.layers[layer_number - 1]
        .self_attn.o_proj.weight.detach()
        .float()
    )
    start = head_index * HEAD_DIM
    stop = start + HEAD_DIM
    residual = vectors.float() @ output_projection[:, start:stop].T
    return torch.sum(residual * unembed_direction, dim=-1)


def observation_rows_for_batch(
    model,
    batch: list[dict[str, Any]],
    events: list[dict[str, Any]],
    components: dict[str, dict[str, dict[str, torch.Tensor]]],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
    candidate_ids: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        event_id = event["event_id"]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        total_delta = (
            do_heads[layer_number][:, head_index, :].float()
            - target_heads[layer_number][:, head_index, :].float()
        )
        for role in ROLE_NAMES:
            values = components[event_id][role]
            direct = {
                component: direct_logit_effect(
                    model,
                    event,
                    values[component],
                    batch,
                    candidate_ids,
                )
                for component in (
                    "delta",
                    "qk",
                    "value",
                    "interaction",
                )
            }
            delta_norm = torch.linalg.vector_norm(
                values["delta"], dim=-1
            )
            qk_norm = torch.linalg.vector_norm(values["qk"], dim=-1)
            value_norm = torch.linalg.vector_norm(
                values["value"], dim=-1
            )
            interaction_norm = torch.linalg.vector_norm(
                values["interaction"], dim=-1
            )
            cosine = F.cosine_similarity(
                values["delta"], total_delta, dim=-1, eps=1e-8
            )
            for index, item in enumerate(batch):
                rows.append(
                    {
                        "schema_version": (
                            "phase1001_source_path_observation.v1"
                        ),
                        "phase": PHASE_ID,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "template": item["target"]["template"],
                        "route_id": route_id(event_id, role),
                        "event_id": event_id,
                        "layer_number": layer_number,
                        "head_index": head_index,
                        "source_role": role,
                        "target_attention_mass": float(
                            values["target_attention_mass"][index]
                        ),
                        "do_attention_mass": float(
                            values["do_attention_mass"][index]
                        ),
                        "attention_mass_delta": float(
                            values["do_attention_mass"][index]
                            - values["target_attention_mass"][index]
                        ),
                        "delta_norm": float(delta_norm[index]),
                        "qk_norm": float(qk_norm[index]),
                        "value_norm": float(value_norm[index]),
                        "interaction_norm": float(
                            interaction_norm[index]
                        ),
                        "delta_to_total_cosine": float(cosine[index]),
                        "direct_logit_effect": float(
                            direct["delta"][index]
                        ),
                        "qk_direct_logit_effect": float(
                            direct["qk"][index]
                        ),
                        "value_direct_logit_effect": float(
                            direct["value"][index]
                        ),
                        "interaction_direct_logit_effect": float(
                            direct["interaction"][index]
                        ),
                    }
                )
    return rows


def causal_rows_for_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    components: dict[str, dict[str, dict[str, torch.Tensor]]],
    candidate_ids: dict[str, int],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    do_logits: torch.Tensor,
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [item["target"] for item in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    rows = []
    for route in routes:
        event = route["event"]
        event_id = event["event_id"]
        role = route["source_role"]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        delta = components[event_id][role]["delta"]
        restore_vectors = (
            do_heads[layer_number][:, head_index, :].float() - delta
        )
        sufficiency_vectors = (
            target_heads[layer_number][:, head_index, :].float() + delta
        )
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=[
                {"event": event, "vectors": restore_vectors}
            ],
        )
        sufficiency_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=[
                {"event": event, "vectors": sufficiency_vectors}
            ],
        )
        restored_margin = semantic_margin(restored_logits, batch)
        sufficiency_margin = semantic_margin(sufficiency_logits, batch)
        restored_prediction = prediction_colors(restored_logits)
        sufficiency_prediction = prediction_colors(sufficiency_logits)
        for index, item in enumerate(batch):
            source_denominator = float(
                source_margin[index] - target_margin[index]
            )
            do_effect = float(do_margin[index] - target_margin[index])
            rows.append(
                {
                    "schema_version": (
                        "phase1001_source_path_causal.v1"
                    ),
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["target"]["template"],
                    "route_id": route["route_id"],
                    "event_id": event_id,
                    "layer_number": layer_number,
                    "head_index": head_index,
                    "source_role": role,
                    "source_margin": float(source_margin[index]),
                    "target_margin": float(target_margin[index]),
                    "do_source_margin": float(do_margin[index]),
                    "restored_margin": float(restored_margin[index]),
                    "sufficiency_margin": float(
                        sufficiency_margin[index]
                    ),
                    "mediation_fraction": float(
                        (do_margin[index] - restored_margin[index])
                        / max(abs(do_effect), 1e-8)
                    ),
                    "sufficiency_transfer": float(
                        (
                            sufficiency_margin[index]
                            - target_margin[index]
                        )
                        / max(abs(source_denominator), 1e-8)
                    ),
                    "restored_prediction": restored_prediction[index],
                    "sufficiency_prediction": (
                        sufficiency_prediction[index]
                    ),
                    "restored_to_target": (
                        restored_prediction[index]
                        == item["target"]["gold"]
                    ),
                    "sufficiency_flipped": (
                        sufficiency_prediction[index]
                        == item["source"]["gold"]
                    ),
                }
            )
        del restored_logits, sufficiency_logits
    return rows


def summarize_observations(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["route_id"]].append(row)
    metrics = (
        "target_attention_mass",
        "do_attention_mass",
        "attention_mass_delta",
        "delta_norm",
        "qk_norm",
        "value_norm",
        "interaction_norm",
        "delta_to_total_cosine",
        "direct_logit_effect",
        "qk_direct_logit_effect",
        "value_direct_logit_effect",
        "interaction_direct_logit_effect",
    )
    return {
        route: {
            "route_id": route,
            "event_id": values[0]["event_id"],
            "layer_number": values[0]["layer_number"],
            "head_index": values[0]["head_index"],
            "source_role": values[0]["source_role"],
            "n": len(values),
            **{
                f"mean_{metric}": float(
                    np.mean([row[metric] for row in values])
                )
                for metric in metrics
            },
            **{
                f"median_{metric}": float(
                    np.median([row[metric] for row in values])
                )
                for metric in metrics
            },
        }
        for route, values in groups.items()
    }


def summarize_causal(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["route_id"]].append(row)
    return {
        route: {
            "route_id": route,
            "event_id": values[0]["event_id"],
            "layer_number": values[0]["layer_number"],
            "head_index": values[0]["head_index"],
            "source_role": values[0]["source_role"],
            "n": len(values),
            "median_mediation_fraction": float(
                np.median(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "mean_mediation_fraction": float(
                np.mean(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "positive_mediation_rate": float(
                np.mean(
                    [row["mediation_fraction"] > 0 for row in values]
                )
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "mean_sufficiency_transfer": float(
                np.mean(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "median_sufficiency_transfer": float(
                np.median(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "positive_sufficiency_rate": float(
                np.mean(
                    [row["sufficiency_transfer"] > 0 for row in values]
                )
            ),
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "template_median_mediation": {
                str(template): float(
                    np.median(
                        [
                            row["mediation_fraction"]
                            for row in values
                            if int(row["template"]) == template
                        ]
                    )
                )
                for template in range(4)
            },
            "template_mean_sufficiency": {
                str(template): float(
                    np.mean(
                        [
                            row["sufficiency_transfer"]
                            for row in values
                            if int(row["template"]) == template
                        ]
                    )
                )
                for template in range(4)
            },
        }
        for route, values in groups.items()
    }


def rank_routes(
    observation_summary: dict[str, dict[str, Any]],
    causal_summary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for route, causal in causal_summary.items():
        observation = observation_summary[route]
        balanced = min(
            max(0.0, causal["median_mediation_fraction"]),
            max(0.0, causal["mean_sufficiency_transfer"]),
        )
        rows.append(
            {
                **observation,
                **causal,
                "balanced_causal_score": balanced,
            }
        )
    rows.sort(
        key=lambda item: (
            -item["balanced_causal_score"],
            -item["median_mediation_fraction"],
            -item["mean_sufficiency_transfer"],
            item["layer_number"],
            item["head_index"],
            ROLE_NAMES.index(item["source_role"]),
        )
    )
    for index, item in enumerate(rows, 1):
        item["causal_rank"] = index
        item["selection_partition"] = "validation"
        item["selection_uses_holdout"] = False
    return rows


def combined_head_patches(
    ranked_routes: list[dict[str, Any]],
    size: int,
    components: dict[str, dict[str, dict[str, torch.Tensor]]],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
    mode: str,
) -> list[dict[str, Any]]:
    selected = ranked_routes[:size]
    grouped: dict[str, list[str]] = defaultdict(list)
    for route in selected:
        grouped[route["event_id"]].append(route["source_role"])
    patches = []
    for event_id, roles in grouped.items():
        event = event_from_id(event_id)
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        delta = torch.stack(
            [components[event_id][role]["delta"] for role in roles],
            dim=0,
        ).sum(dim=0)
        if mode == "restore":
            vectors = (
                do_heads[layer_number][:, head_index, :].float() - delta
            )
        elif mode == "sufficiency":
            vectors = (
                target_heads[layer_number][:, head_index, :].float()
                + delta
            )
        else:
            raise ValueError(mode)
        patches.append({"event": event, "vectors": vectors})
    return patches


def joint_rows_for_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    ranked_routes: list[dict[str, Any]],
    sizes: Iterable[int],
    components: dict[str, dict[str, dict[str, torch.Tensor]]],
    candidate_ids: dict[str, int],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    do_logits: torch.Tensor,
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [item["target"] for item in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    rows = []
    for size in sorted({min(int(size), len(ranked_routes)) for size in sizes}):
        restore_patches = combined_head_patches(
            ranked_routes,
            size,
            components,
            target_heads,
            do_heads,
            "restore",
        )
        sufficiency_patches = combined_head_patches(
            ranked_routes,
            size,
            components,
            target_heads,
            do_heads,
            "sufficiency",
        )
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=restore_patches,
        )
        sufficiency_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=sufficiency_patches,
        )
        restored_margin = semantic_margin(restored_logits, batch)
        sufficiency_margin = semantic_margin(sufficiency_logits, batch)
        restored_prediction = prediction_colors(restored_logits)
        sufficiency_prediction = prediction_colors(sufficiency_logits)
        route_ids = [
            route["route_id"] for route in ranked_routes[:size]
        ]
        for index, item in enumerate(batch):
            source_denominator = float(
                source_margin[index] - target_margin[index]
            )
            do_effect = float(do_margin[index] - target_margin[index])
            rows.append(
                {
                    "schema_version": (
                        "phase1001_source_path_joint.v1"
                    ),
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["target"]["template"],
                    "joint_size": size,
                    "route_ids": route_ids,
                    "mediation_fraction": float(
                        (do_margin[index] - restored_margin[index])
                        / max(abs(do_effect), 1e-8)
                    ),
                    "sufficiency_transfer": float(
                        (
                            sufficiency_margin[index]
                            - target_margin[index]
                        )
                        / max(abs(source_denominator), 1e-8)
                    ),
                    "restored_prediction": restored_prediction[index],
                    "sufficiency_prediction": (
                        sufficiency_prediction[index]
                    ),
                    "restored_to_target": (
                        restored_prediction[index]
                        == item["target"]["gold"]
                    ),
                    "sufficiency_flipped": (
                        sufficiency_prediction[index]
                        == item["source"]["gold"]
                    ),
                }
            )
        del restored_logits, sufficiency_logits
    return rows


def summarize_joint(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["joint_size"])].append(row)
    return {
        str(size): {
            "joint_size": size,
            "route_ids": values[0]["route_ids"],
            "n": len(values),
            "median_mediation_fraction": float(
                np.median(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "mean_mediation_fraction": float(
                np.mean(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "mean_sufficiency_transfer": float(
                np.mean(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "median_sufficiency_transfer": float(
                np.median(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "template_median_mediation": {
                str(template): float(
                    np.median(
                        [
                            row["mediation_fraction"]
                            for row in values
                            if int(row["template"]) == template
                        ]
                    )
                )
                for template in range(4)
            },
            "template_mean_sufficiency": {
                str(template): float(
                    np.mean(
                        [
                            row["sufficiency_transfer"]
                            for row in values
                            if int(row["template"]) == template
                        ]
                    )
                )
                for template in range(4)
            },
        }
        for size, values in sorted(groups.items())
    }


def choose_joint_size(summary: dict[str, dict[str, Any]]) -> int:
    best = max(
        min(
            item["median_mediation_fraction"],
            item["mean_sufficiency_transfer"],
        )
        for item in summary.values()
    )
    threshold = 0.95 * best
    eligible = [
        int(size)
        for size, item in summary.items()
        if min(
            item["median_mediation_fraction"],
            item["mean_sufficiency_transfer"],
        )
        >= threshold
    ]
    return min(eligible)


def control_rows_for_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    selected_routes: list[dict[str, Any]],
    components: dict[str, dict[str, dict[str, torch.Tensor]]],
    candidate_ids: dict[str, int],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_heads: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [item["target"] for item in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    safe_shift = valid_derangement_shifts(batch, 1)[0]
    rows = []
    for route in selected_routes:
        event = route["event"]
        event_id = event["event_id"]
        role = route["source_role"]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        delta = components[event_id][role]["delta"]
        wrong_head = (head_index + 1) % HEAD_COUNT
        wrong_event = {
            **event,
            "event_id": f"l{layer_number:02d}.h{wrong_head:02d}",
            "head_index": wrong_head,
        }
        wrong_vectors = (
            target_heads[layer_number][:, wrong_head, :].float()
            + delta
        )
        null_vectors = (
            target_heads[layer_number][:, head_index, :].float()
            + torch.roll(delta, shifts=safe_shift, dims=0)
        )
        wrong_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=[
                {"event": wrong_event, "vectors": wrong_vectors}
            ],
        )
        null_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=[{"event": event, "vectors": null_vectors}],
        )
        wrong_margin = semantic_margin(wrong_logits, batch)
        null_margin = semantic_margin(null_logits, batch)
        wrong_prediction = prediction_colors(wrong_logits)
        null_prediction = prediction_colors(null_logits)
        for index, item in enumerate(batch):
            denominator = float(
                source_margin[index] - target_margin[index]
            )
            rows.append(
                {
                    "schema_version": (
                        "phase1001_source_path_control.v1"
                    ),
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "route_id": route["route_id"],
                    "event_id": event_id,
                    "source_role": role,
                    "wrong_head_index": wrong_head,
                    "wrong_o_transfer": float(
                        (wrong_margin[index] - target_margin[index])
                        / max(abs(denominator), 1e-8)
                    ),
                    "cross_pair_null_transfer": float(
                        (null_margin[index] - target_margin[index])
                        / max(abs(denominator), 1e-8)
                    ),
                    "wrong_o_flipped": (
                        wrong_prediction[index]
                        == item["source"]["gold"]
                    ),
                    "cross_pair_null_flipped": (
                        null_prediction[index]
                        == item["source"]["gold"]
                    ),
                }
            )
        del wrong_logits, null_logits
    return rows


def summarize_controls(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["route_id"]].append(row)
    return {
        route: {
            "route_id": route,
            "event_id": values[0]["event_id"],
            "source_role": values[0]["source_role"],
            "n": len(values),
            "mean_wrong_o_transfer": float(
                np.mean([row["wrong_o_transfer"] for row in values])
            ),
            "wrong_o_flip_rate": float(
                np.mean([row["wrong_o_flipped"] for row in values])
            ),
            "mean_cross_pair_null_transfer": float(
                np.mean(
                    [row["cross_pair_null_transfer"] for row in values]
                )
            ),
            "cross_pair_null_flip_rate": float(
                np.mean(
                    [row["cross_pair_null_flipped"] for row in values]
                )
            ),
        }
        for route, values in groups.items()
    }


def natural_rows_for_batch(
    model,
    layers,
    tokenizer,
    device,
    batch: list[dict[str, Any]],
    ranked_routes: list[dict[str, Any]],
    frozen_size: int,
    components: dict[str, dict[str, dict[str, torch.Tensor]]],
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
    effective_eos: list[int],
    budget: int,
) -> list[dict[str, Any]]:
    target_cases = [item["target"] for item in batch]
    route_restore = combined_head_patches(
        ranked_routes,
        frozen_size,
        components,
        target_heads,
        do_heads,
        "restore",
    )
    full_head_restore = []
    seen = set()
    for route in ranked_routes:
        event_id = route["event_id"]
        if event_id in seen:
            continue
        seen.add(event_id)
        event = event_from_id(event_id)
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        full_head_restore.append(
            {
                "event": event,
                "vectors": target_heads[layer_number][
                    :, head_index, :
                ],
            }
        )
    conditions = {
        "source_do": [],
        "source_plus_frozen_route_restore": route_restore,
        "source_plus_full_frozen_head_restore": full_head_restore,
    }
    rows = []
    for condition, patches in conditions.items():
        generated = generate_with_patches(
            model,
            layers,
            tokenizer,
            device,
            target_cases,
            source_patch,
            patches,
            {},
            effective_eos,
            budget,
        )
        for index, item in enumerate(batch):
            result = generated[index]
            rows.append(
                {
                    "schema_version": (
                        "phase1001_source_path_natural.v1"
                    ),
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "condition": condition,
                    "frozen_joint_size": frozen_size,
                    "prediction": result["prediction"],
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "flipped_to_source": (
                        result["prediction"] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        result["prediction"] == item["target"]["gold"]
                    ),
                    "eos_seen": result["eos_seen"],
                    "exact_short": result["exact_short"],
                    "generated_text": result["text"],
                }
            )
    return rows


def summarize_natural(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "flip_rate": float(
                np.mean([row["flipped_to_source"] for row in values])
            ),
            "target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "eos_rate": float(
                np.mean([row["eos_seen"] for row in values])
            ),
            "exact_short_rate": float(
                np.mean([row["exact_short"] for row in values])
            ),
        }
        for condition, values in groups.items()
    }


def make_routes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "route_id": route_id(event["event_id"], role),
            "event_id": event["event_id"],
            "source_role": role,
            "event": event,
        }
        for event in events
        for role in ROLE_NAMES
    ]


def run(stage: str, batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 source path test requires CUDA")
    protocol, selected_pairs, directional, output_root = selected_inputs(stage)
    output_root.mkdir(parents=True, exist_ok=True)
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    frozen_heads = read_json(HEAD_DISCOVERY_ROOT / "frozen_spec.json")
    events = [
        event_from_id(event_id)
        for event_id in frozen_heads["frozen_joint_event_ids"]
    ]
    all_routes = make_routes(events)

    if stage == "confirmation":
        frozen_routes = read_json(
            SOURCE_ROOT / "discovery" / "frozen_spec.json"
        )
        ranked_ids = list(frozen_routes["ranked_route_ids"])
        route_lookup = {route["route_id"]: route for route in all_routes}
        ranked_routes = [route_lookup[item] for item in ranked_ids]
        frozen_size = int(frozen_routes["frozen_joint_size"])
        if (
            frozen_routes["frozen_joint_route_ids"]
            != ranked_ids[:frozen_size]
        ):
            raise RuntimeError("frozen source route drift")
        causal_routes = ranked_routes[:frozen_size]
    else:
        frozen_routes = None
        ranked_routes = []
        frozen_size = 0
        causal_routes = all_routes

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        effective_eos = eos_ids(model, tokenizer)

        observation_rows = []
        causal_rows = []
        audit_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [item["source"] for item in batch]
            target_cases = [item["target"] for item in batch]
            source_logits, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                target_logits,
                target_values,
                target_weights,
                target_heads,
            ) = capture_physical_attention(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
            )
            do_logits, do_values, do_weights, do_heads = (
                capture_physical_attention(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
            )
            components, batch_audit = component_map_for_batch(
                batch,
                events,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
            )
            for row in batch_audit:
                audit_rows.append(
                    {
                        **row,
                        "batch_number": batch_number,
                        "partition": batch[0]["partition"],
                    }
                )
            observation_rows.extend(
                observation_rows_for_batch(
                    model,
                    batch,
                    events,
                    components,
                    target_heads,
                    do_heads,
                    candidate_ids,
                )
            )
            causal_rows.extend(
                causal_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    causal_routes,
                    components,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            del (
                source_logits,
                source_residuals,
                target_logits,
                target_values,
                target_weights,
                target_heads,
                do_logits,
                do_values,
                do_weights,
                do_heads,
                components,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[source-{stage}-causal] "
                    f"{batch_number}/{len(batches)} batches",
                    flush=True,
                )

        observation_summary = summarize_observations(observation_rows)
        causal_summary = summarize_causal(causal_rows)
        write_rows(output_root / "observation_rows.jsonl", observation_rows)
        write_rows(output_root / "causal_rows.jsonl", causal_rows)
        write_rows(output_root / "instrument_audit_rows.jsonl", audit_rows)
        write_json(
            output_root / "observation_summary.json",
            observation_summary,
        )
        write_json(output_root / "causal_summary.json", causal_summary)

        if stage == "discovery":
            ranked_metrics = rank_routes(
                observation_summary, causal_summary
            )
            route_lookup = {
                route["route_id"]: route for route in all_routes
            }
            ranked_routes = [
                {
                    **route_lookup[item["route_id"]],
                    **item,
                }
                for item in ranked_metrics
            ]
        else:
            ranked_metrics = [
                {
                    **observation_summary[route["route_id"]],
                    **causal_summary[route["route_id"]],
                    "route_id": route["route_id"],
                    "event": route["event"],
                }
                for route in causal_routes
            ]

        joint_rows = []
        if stage == "discovery":
            joint_sizes = JOINT_ROUTE_SIZES
        else:
            joint_sizes = sorted(
                {
                    min(size, frozen_size)
                    for size in JOINT_ROUTE_SIZES
                    if size <= frozen_size
                }
                | {frozen_size}
            )
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [item["source"] for item in batch]
            target_cases = [item["target"] for item in batch]
            source_logits, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                target_logits,
                target_values,
                target_weights,
                target_heads,
            ) = capture_physical_attention(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
            )
            do_logits, do_values, do_weights, do_heads = (
                capture_physical_attention(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
            )
            components, _ = component_map_for_batch(
                batch,
                events,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
            )
            joint_rows.extend(
                joint_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    ranked_routes,
                    joint_sizes,
                    components,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            del (
                source_logits,
                source_residuals,
                target_logits,
                target_values,
                target_weights,
                target_heads,
                do_logits,
                do_values,
                do_weights,
                do_heads,
                components,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[source-{stage}-joint] "
                    f"{batch_number}/{len(batches)} batches",
                    flush=True,
                )
        joint_summary = summarize_joint(joint_rows)
        write_rows(output_root / "joint_rows.jsonl", joint_rows)
        write_json(output_root / "joint_summary.json", joint_summary)
        if stage == "discovery":
            frozen_size = choose_joint_size(joint_summary)

        selected_for_controls = ranked_routes[:frozen_size]
        control_rows = []
        natural_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [item["source"] for item in batch]
            target_cases = [item["target"] for item in batch]
            source_logits, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                _,
                target_values,
                target_weights,
                target_heads,
            ) = capture_physical_attention(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
            )
            _, do_values, do_weights, do_heads = (
                capture_physical_attention(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
            )
            components, _ = component_map_for_batch(
                batch,
                events,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
            )
            target_logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
            )
            control_rows.extend(
                control_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    selected_for_controls,
                    components,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    target_heads,
                )
            )
            natural_rows.extend(
                natural_rows_for_batch(
                    model,
                    layers,
                    tokenizer,
                    device,
                    batch,
                    ranked_routes,
                    frozen_size,
                    components,
                    source_patch,
                    target_heads,
                    do_heads,
                    effective_eos,
                    natural_budget,
                )
            )
            del (
                source_logits,
                source_residuals,
                target_logits,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
                components,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[source-{stage}-controls-natural] "
                    f"{batch_number}/{len(batches)} batches",
                    flush=True,
                )

        control_summary = summarize_controls(control_rows)
        natural_summary = summarize_natural(natural_rows)
        write_rows(output_root / "control_rows.jsonl", control_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
        write_json(output_root / "control_summary.json", control_summary)
        write_json(output_root / "natural_summary.json", natural_summary)

        audit_metrics = {
            "max_target_head_reconstruction_error": max(
                row["target_max_abs_head_reconstruction_error"]
                for row in audit_rows
            ),
            "max_do_head_reconstruction_error": max(
                row["do_max_abs_head_reconstruction_error"]
                for row in audit_rows
            ),
            "max_role_delta_reconstruction_error": max(
                row["max_abs_role_delta_reconstruction_error"]
                for row in audit_rows
            ),
            "max_qkv_identity_error": max(
                row["max_abs_qkv_identity_error"] for row in audit_rows
            ),
        }
        frozen_joint = joint_summary[str(frozen_size)]
        stable_single_routes = []
        for route in selected_for_controls:
            route_key = route["route_id"]
            causal = causal_summary[route_key]
            control = control_summary[route_key]
            if (
                causal["median_mediation_fraction"]
                >= PATH_THRESHOLDS["single_median_mediation"]
                and causal["mean_sufficiency_transfer"]
                >= PATH_THRESHOLDS["single_mean_sufficiency"]
                and (
                    causal["mean_sufficiency_transfer"]
                    - control["mean_wrong_o_transfer"]
                )
                >= PATH_THRESHOLDS["single_location_excess"]
                and (
                    causal["mean_sufficiency_transfer"]
                    - control["mean_cross_pair_null_transfer"]
                )
                >= PATH_THRESHOLDS["single_location_excess"]
            ):
                stable_single_routes.append(route_key)
        gate_checks = {
            "head_reconstruction": (
                max(
                    audit_metrics[
                        "max_target_head_reconstruction_error"
                    ],
                    audit_metrics["max_do_head_reconstruction_error"],
                )
                <= PATH_THRESHOLDS["max_head_reconstruction_error"]
            ),
            "role_partition_reconstruction": (
                audit_metrics["max_role_delta_reconstruction_error"]
                <= PATH_THRESHOLDS[
                    "max_role_delta_reconstruction_error"
                ]
            ),
            "qkv_algebra_identity": (
                audit_metrics["max_qkv_identity_error"]
                <= PATH_THRESHOLDS["max_qkv_identity_error"]
            ),
            "stable_single_route": bool(stable_single_routes),
            "frozen_joint_mediation": (
                frozen_joint["median_mediation_fraction"]
                >= PATH_THRESHOLDS["joint_median_mediation"]
            ),
            "frozen_joint_sufficiency": (
                frozen_joint["mean_sufficiency_transfer"]
                >= PATH_THRESHOLDS["joint_mean_sufficiency"]
            ),
            "source_do_natural": (
                natural_summary["source_do"]["flip_rate"]
                >= PATH_THRESHOLDS["source_do_natural_flip"]
            ),
            "frozen_joint_natural": (
                natural_summary[
                    "source_plus_frozen_route_restore"
                ]["target_rate"]
                >= PATH_THRESHOLDS["joint_natural_restoration"]
            ),
        }
        source_path_gate = all(gate_checks.values())
        frozen_spec = {
            "schema_version": "phase1001_frozen_source_path_spec.v1",
            "phase": PHASE_ID,
            "model": MODEL,
            "ranked_route_ids": [
                route["route_id"] for route in ranked_routes
            ],
            "frozen_joint_size": frozen_size,
            "frozen_joint_route_ids": [
                route["route_id"]
                for route in ranked_routes[:frozen_size]
            ],
            "selection_partition": "validation",
            "selection_uses_holdout": False,
            "frozen_before_holdout": stage == "discovery",
        }
        summary = {
            "schema_version": (
                f"phase1001_source_path_{stage}_summary.v1"
            ),
            "phase": PHASE_ID,
            "model": MODEL,
            "stage": stage,
            "partition": directional[0]["partition"],
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "source_depth": SOURCE_DEPTH,
            "frozen_head_event_ids": [
                event["event_id"] for event in events
            ],
            "source_roles": list(ROLE_NAMES),
            "route_count_observed": len(observation_summary),
            "route_count_causally_tested": len(causal_summary),
            "ranked_routes": ranked_metrics,
            "frozen_joint_size": frozen_size,
            "frozen_joint_route_ids": frozen_spec[
                "frozen_joint_route_ids"
            ],
            "audit_metrics": audit_metrics,
            "causal_summary": causal_summary,
            "control_summary": control_summary,
            "joint_summary": joint_summary,
            "natural_summary": natural_summary,
            "stable_single_routes": stable_single_routes,
            "thresholds": PATH_THRESHOLDS,
            "gate_checks": gate_checks,
            "source_path_gate_pass": source_path_gate,
            "qkv_causal_decomposition_open": source_path_gate,
            "selection_uses_current_partition": stage == "discovery",
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(output_root / "frozen_spec.json", frozen_spec)
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("discovery", "confirmation"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(
        args.stage, args.batch_size, args.natural_max_new_tokens
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "passed": summary["source_path_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "audit_metrics": summary["audit_metrics"],
                "stable_single_routes": summary[
                    "stable_single_routes"
                ],
                "frozen_joint_size": summary["frozen_joint_size"],
                "frozen_joint": summary["joint_summary"][
                    str(summary["frozen_joint_size"])
                ],
                "natural": summary["natural_summary"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
