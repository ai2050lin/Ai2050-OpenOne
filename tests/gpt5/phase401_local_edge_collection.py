#!/usr/bin/env python3
"""Collect Phase401 real-module local-edge responses and frozen controls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
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
from phase267_multifamily_continuation_physical_path_trace import get_final_norm  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
)
from phase371b_anchor_qk_collection import capture_actual_qkv  # noqa: E402
from phase398_joint_factorial_protocol import parse_condition  # noqa: E402
from phase399_dynamic_trace_collection import repeat_kv  # noqa: E402
from phase401_local_edge_protocol import MODELS, OUT  # noqa: E402


SOURCE = OUT / "trace/protocol/private/phase401_discovery_trace_cases.jsonl"
SOURCE_ROLES = (
    "source_entity_a",
    "source_entity_b",
    "source_value_a",
    "source_value_b",
    "source_structure",
)
RECEIVER_ROLES = ("query_end", "query_entity", "answer_anchor")
CONTROL_NAMES = (
    "true_relation",
    "wrong_source_order_matched_same_target",
    "wrong_receiver_role",
    "wrong_semantic_time",
    "wrong_depth_quarter_model_shift",
    "source_role_permutation",
    "same_content_wrong_structure",
    "deterministic_random_natural_donor",
    "same_absolute_mass_sign_permuted",
)


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
        raise RuntimeError(f"Phase401 non-finite local-edge scalar: {value}")
    return round(value, 9)


def median(values: list[float]) -> float | None:
    return clean(float(statistics.median(values))) if values else None


def cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def mask_row(mask: torch.Tensor | None, position: int, length: int) -> torch.Tensor | None:
    if mask is None:
        return None
    value = mask.detach()
    if value.ndim != 4:
        raise RuntimeError(f"Phase401 unexpected attention mask rank: {value.ndim}")
    query_index = position if value.shape[-2] > 1 else 0
    return cpu(value[0, 0, query_index, :length])


def component_at(
    captures: dict[tuple[str, int], Any],
    component: str,
    layer_index: int,
    position: int,
) -> torch.Tensor:
    return cpu(captures[(component, layer_index)][0, position])


@torch.inference_mode()
def capture_case(
    loaded: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    captures.clear()
    ids = torch.tensor(
        [case["prompt_token_ids_private"]],
        dtype=torch.long,
        device=loaded.input_device,
    )
    output = loaded.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    sequence_length = int(ids.shape[1])
    layer_states: list[dict[str, Any]] = []
    for layer_index in range(len(layers)):
        query = captures[("query", layer_index)]
        key = captures[("key", layer_index)]
        value = captures[("value", layer_index)]
        attention_mask = captures.get(("attention_mask", layer_index))
        receivers: dict[str, Any] = {}
        for role in RECEIVER_ROLES:
            position = int(case["state_role_positions_private"][role][0])
            layer_input = component_at(captures, "layer_input", layer_index, position)
            attention = component_at(captures, "attention_output", layer_index, position)
            mlp = component_at(captures, "mlp_output", layer_index, position)
            layer_output = component_at(captures, "layer_output", layer_index, position)
            receivers[role] = {
                "position": position,
                "query": cpu(query[0, :, position]),
                "mask": mask_row(attention_mask, position, sequence_length),
                "layer_input": layer_input,
                "attention": attention,
                "post_attention": layer_input + attention,
                "mlp": mlp,
                "layer_output": layer_output,
            }
        layer_states.append(
            {
                "key": cpu(key[0]),
                "value": cpu(value[0]),
                "scaling": float(captures[("attention_scaling", layer_index)].item()),
                "receivers": receivers,
            }
        )
    first_prediction = int(output.logits[0, -1].argmax().item())
    del output, ids
    captures.clear()
    return {
        "case": case,
        "sequence_length": sequence_length,
        "first_prediction_matches_frozen": first_prediction
        == int(case["first_generated_token_id_private"]),
        "layers": layer_states,
    }


def to_device_layer(state: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "key": state["key"].to(device),
        "value": state["value"].to(device),
        "scaling": state["scaling"],
        "receivers": {
            role: {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in receiver.items()
            }
            for role, receiver in state["receivers"].items()
        },
    }


def condition(axis: str, relation: int, order: int, query: int) -> str:
    return f"{axis}_R{relation}_O{order}_Q{query}"


def pair_specs(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for recipient_condition in sorted(cases):
        axis, relation, order, query = parse_condition(recipient_condition)
        other_axis = "Y" if axis == "X" else "X"
        true_donor = condition(axis, 1 - relation, order, query)
        if int(cases[recipient_condition]["semantic_first_token_id_private"]) == int(
            cases[true_donor]["semantic_first_token_id_private"]
        ):
            raise RuntimeError("Phase401 true relation pair did not change target token")
        specs.append(
            {
                "pair_id": f"{recipient_condition}->{true_donor}",
                "recipient_condition": recipient_condition,
                "true_donor_condition": true_donor,
                "controls": {
                    "true_relation": {
                        "donor_condition": true_donor,
                        "receiver_role": "query_end",
                    },
                    "wrong_source_order_matched_same_target": {
                        "donor_condition": condition(axis, relation, 1 - order, query),
                        "receiver_role": "query_end",
                    },
                    "wrong_receiver_role": {
                        "donor_condition": true_donor,
                        "receiver_role": "query_entity",
                    },
                    "wrong_semantic_time": {
                        "donor_condition": true_donor,
                        "receiver_role": "answer_anchor",
                    },
                    "wrong_depth_quarter_model_shift": {
                        "donor_condition": true_donor,
                        "receiver_role": "query_end",
                    },
                    "source_role_permutation": {
                        "donor_condition": true_donor,
                        "receiver_role": "query_end",
                    },
                    "same_content_wrong_structure": {
                        "donor_condition": condition(axis, 1 - relation, 1 - order, query),
                        "receiver_role": "query_end",
                    },
                    "deterministic_random_natural_donor": {
                        "donor_condition": condition(other_axis, relation, order, query),
                        "receiver_role": "query_end",
                    },
                    "same_absolute_mass_sign_permuted": {
                        "donor_condition": true_donor,
                        "receiver_role": "query_end",
                    },
                },
            }
        )
    if len(specs) != 16:
        raise RuntimeError(f"Phase401 expected 16 directed pairs, got {len(specs)}")
    return specs


def role_map(control: str) -> dict[str, str]:
    if control != "source_role_permutation":
        return {role: role for role in SOURCE_ROLES}
    return {
        "source_entity_a": "source_entity_b",
        "source_entity_b": "source_entity_a",
        "source_value_a": "source_value_b",
        "source_value_b": "source_value_a",
        "source_structure": "source_structure",
    }


def shifted_layer(layer_index: int, layer_count: int) -> int:
    shift = max(1, round(layer_count / 4))
    return layer_index + shift if layer_index + shift < layer_count else layer_index - shift


@torch.inference_mode()
def recompute_attention(
    layer: Any,
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    receiver_role: str,
    mapping: dict[str, str],
) -> torch.Tensor | None:
    receiver = recipient_state["receivers"][receiver_role]
    key = recipient_state["key"].clone()
    value = recipient_state["value"].clone()
    recipient_partitions = recipient_case["attention_source_partitions_private"][
        "query_end"
    ]
    donor_partitions = donor_case["attention_source_partitions_private"]["query_end"]
    for recipient_role, donor_role in mapping.items():
        recipient_positions = recipient_partitions[recipient_role]
        donor_positions = donor_partitions[donor_role]
        if len(recipient_positions) != len(donor_positions):
            return None
        recipient_index = torch.tensor(
            recipient_positions, dtype=torch.long, device=key.device
        )
        donor_index = torch.tensor(donor_positions, dtype=torch.long, device=key.device)
        key.index_copy_(1, recipient_index, donor_state["key"].index_select(1, donor_index))
        value.index_copy_(
            1, recipient_index, donor_state["value"].index_select(1, donor_index)
        )
    query = receiver["query"]
    head_count = int(query.shape[0])
    repeated_key = repeat_kv(key, head_count)
    repeated_value = repeat_kv(value, head_count)
    scores = torch.einsum("hd,hsd->hs", query.float(), repeated_key.float())
    scores = scores * float(recipient_state["scaling"])
    if receiver["mask"] is not None:
        scores = scores + receiver["mask"].float().unsqueeze(0)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weighted = torch.einsum("hs,hsd->hd", probabilities, repeated_value)
    o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
    projected = o_proj(weighted.reshape(1, 1, -1))
    return projected[0, 0]


def state_metrics(
    counterfactual: torch.Tensor,
    recipient: torch.Tensor,
    donor: torch.Tensor,
    minimum_relative_norm: float,
    cosine_gate: float,
    recovery_gate: float,
) -> dict[str, Any]:
    counterfactual = counterfactual.float()
    recipient = recipient.float()
    donor = donor.float()
    natural_delta = donor - recipient
    intervention_delta = counterfactual - recipient
    baseline = float(torch.linalg.vector_norm(natural_delta).item())
    scale = (
        float(torch.linalg.vector_norm(donor).item())
        + float(torch.linalg.vector_norm(recipient).item())
    ) / 2
    relative = baseline / max(scale, 1e-8)
    informative = relative >= minimum_relative_norm
    cosine = float(
        F.cosine_similarity(
            intervention_delta.unsqueeze(0), natural_delta.unsqueeze(0), dim=-1, eps=1e-8
        ).item()
    )
    residual = float(torch.linalg.vector_norm(counterfactual - donor).item())
    recovery = 1.0 - residual / max(baseline, 1e-8)
    positive = float(intervention_delta.clamp_min(0).sum().item())
    negative = float((-intervention_delta.clamp_max(0)).sum().item())
    return {
        "informative": informative,
        "baseline_relative_norm": clean(relative),
        "direction_cosine": clean(cosine),
        "state_recovery": clean(recovery),
        "pair_pass": bool(
            informative and cosine >= cosine_gate and recovery >= recovery_gate
        ),
        "positive_mass": clean(positive),
        "negative_mass": clean(negative),
        "net_mass": clean(float(intervention_delta.sum().item())),
        "absolute_mass": clean(positive + negative),
    }


@torch.inference_mode()
def normalized_states(final_norm: Any, states: torch.Tensor) -> torch.Tensor:
    return final_norm(states) if final_norm is not None else states


def aggregate_rows(
    pair_rows: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["layer_index"], row["control_name"])].append(row)
    result: list[dict[str, Any]] = []
    gate = freeze["group_layer_gate"]
    for (layer_index, control), rows in sorted(grouped.items()):
        attention = [row["stages"]["attention"] for row in rows]
        informative = [item for item in attention if item["informative"]]
        semantic = [row["semantic_prediction"] for row in rows]
        semantic_informative = [item for item in semantic if item["informative"]]
        informative_rate = len(informative) / len(rows)
        pair_pass_rate = sum(item["pair_pass"] for item in attention) / len(rows)
        semantic_info_rate = len(semantic_informative) / len(rows)
        semantic_positive_rate = (
            sum(item["positive_shift"] for item in semantic_informative)
            / len(semantic_informative)
            if semantic_informative
            else 0.0
        )
        med_recovery = median([item["state_recovery"] for item in informative])
        med_semantic = median(
            [item["competition_recovery"] for item in semantic_informative]
        )
        qualified = bool(
            control == "true_relation"
            and informative_rate >= gate["informative_pair_rate_min"]
            and pair_pass_rate >= gate["direct_attention_pair_pass_rate_min"]
            and med_recovery is not None
            and med_recovery >= gate["direct_attention_median_recovery_min"]
            and semantic_info_rate >= gate["semantic_informative_pair_rate_min"]
            and semantic_positive_rate >= gate["semantic_positive_shift_rate_min"]
            and med_semantic is not None
            and med_semantic >= gate["semantic_median_recovery_min"]
        )
        result.append(
            {
                "schema_version": "75.9.0",
                "phase_id": "Phase401-LocalEdgeGroupLayer",
                "model": rows[0]["model"],
                "surface_private": rows[0]["surface_private"],
                "public_parallel_group_id": rows[0]["public_parallel_group_id"],
                "layer_index": layer_index,
                "layer_count": rows[0]["layer_count"],
                "control_name": control,
                "pair_count": len(rows),
                "informative_pair_rate": clean(informative_rate),
                "pair_pass_rate": clean(pair_pass_rate),
                "median_attention_state_recovery": med_recovery,
                "median_attention_direction_cosine": median(
                    [item["direction_cosine"] for item in informative]
                ),
                "semantic_informative_pair_rate": clean(semantic_info_rate),
                "semantic_positive_shift_rate": clean(semantic_positive_rate),
                "median_semantic_competition_recovery": med_semantic,
                "true_group_layer_gate_pass": qualified,
            }
        )
    return result


@torch.inference_mode()
def analyze_group(
    loaded: Any,
    layers: list[Any],
    collected: dict[str, dict[str, Any]],
    freeze: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = pair_specs({key: item["case"] for key, item in collected.items()})
    device = loaded.input_device
    final_norm = get_final_norm(loaded.model)
    output_embedding = loaded.model.get_output_embeddings()
    pair_rows: list[dict[str, Any]] = []
    local_gate = freeze["pair_gate"]
    semantic_gate = freeze["semantic_gate"]
    layer_count = len(layers)
    for layer_index, layer in enumerate(layers):
        current = {
            condition_id: to_device_layer(item["layers"][layer_index], device)
            for condition_id, item in collected.items()
        }
        wrong_index = shifted_layer(layer_index, layer_count)
        wrong_depth = {
            condition_id: to_device_layer(item["layers"][wrong_index], device)
            for condition_id, item in collected.items()
        }
        tasks: list[dict[str, Any]] = []
        for pair in pairs:
            recipient_condition = pair["recipient_condition"]
            recipient_item = collected[recipient_condition]
            recipient_case = recipient_item["case"]
            true_donor_condition = pair["true_donor_condition"]
            true_donor_item = collected[true_donor_condition]
            true_attention = recompute_attention(
                layer,
                current[recipient_condition],
                current[true_donor_condition],
                recipient_case,
                true_donor_item["case"],
                "query_end",
                role_map("true_relation"),
            )
            for control in CONTROL_NAMES:
                spec = pair["controls"][control]
                donor_condition = spec["donor_condition"]
                donor_item = collected[donor_condition]
                receiver_role = spec["receiver_role"]
                if control == "same_absolute_mass_sign_permuted":
                    if true_attention is None:
                        attention_cf = None
                    else:
                        recipient_attention = current[recipient_condition]["receivers"][
                            "query_end"
                        ]["attention"]
                        delta = true_attention - recipient_attention
                        signs = torch.where(
                            torch.arange(delta.numel(), device=device) % 2 == 0,
                            torch.ones(delta.numel(), device=device),
                            -torch.ones(delta.numel(), device=device),
                        ).to(delta.dtype)
                        attention_cf = recipient_attention + delta.abs() * signs
                else:
                    donor_state = (
                        wrong_depth[donor_condition]
                        if control == "wrong_depth_quarter_model_shift"
                        else current[donor_condition]
                    )
                    attention_cf = recompute_attention(
                        layer,
                        current[recipient_condition],
                        donor_state,
                        recipient_case,
                        donor_item["case"],
                        receiver_role,
                        role_map(control),
                    )
                if attention_cf is None:
                    continue
                recipient_receiver = current[recipient_condition]["receivers"][
                    receiver_role
                ]
                donor_receiver = current[donor_condition]["receivers"][receiver_role]
                post_cf = recipient_receiver["layer_input"] + attention_cf
                tasks.append(
                    {
                        "pair": pair,
                        "control": control,
                        "recipient_condition": recipient_condition,
                        "donor_condition": donor_condition,
                        "receiver_role": receiver_role,
                        "recipient_case": recipient_case,
                        "donor_case": donor_item["case"],
                        "attention_cf": attention_cf,
                        "post_cf": post_cf,
                        "recipient": recipient_receiver,
                        "donor": donor_receiver,
                    }
                )
        if len(tasks) != len(pairs) * len(CONTROL_NAMES):
            raise RuntimeError(
                f"Phase401 role alignment lost tasks at layer {layer_index}: "
                f"{len(tasks)} != {len(pairs) * len(CONTROL_NAMES)}"
            )
        post_stack = torch.stack([task["post_cf"] for task in tasks]).unsqueeze(1)
        post_norm = module_attr(
            layer,
            ("post_attention_layernorm", "post_attention_layer_norm", "ln_2"),
        )
        mlp_cf = layer.mlp(post_norm(post_stack))[:, 0]
        out_cf = post_stack[:, 0] + mlp_cf

        recipient_out = torch.stack([task["recipient"]["layer_output"] for task in tasks])
        donor_out = torch.stack([task["donor"]["layer_output"] for task in tasks])
        normalized = normalized_states(
            final_norm,
            torch.cat([out_cf, recipient_out, donor_out], dim=0),
        )
        norm_cf, norm_recipient, norm_donor = normalized.chunk(3, dim=0)
        donor_ids = torch.tensor(
            [int(task["donor_case"]["semantic_first_token_id_private"]) for task in tasks],
            dtype=torch.long,
            device=output_embedding.weight.device,
        )
        recipient_ids = torch.tensor(
            [
                int(task["recipient_case"]["semantic_first_token_id_private"])
                for task in tasks
            ],
            dtype=torch.long,
            device=output_embedding.weight.device,
        )
        donor_weights = output_embedding.weight.index_select(0, donor_ids).to(norm_cf.dtype)
        recipient_weights = output_embedding.weight.index_select(0, recipient_ids).to(
            norm_cf.dtype
        )
        difference_weights = donor_weights - recipient_weights
        margin_cf = (norm_cf.to(difference_weights.device) * difference_weights).sum(-1)
        margin_recipient = (
            norm_recipient.to(difference_weights.device) * difference_weights
        ).sum(-1)
        margin_donor = (
            norm_donor.to(difference_weights.device) * difference_weights
        ).sum(-1)

        for task_index, task in enumerate(tasks):
            stages = {
                "attention": state_metrics(
                    task["attention_cf"],
                    task["recipient"]["attention"],
                    task["donor"]["attention"],
                    freeze["minimum_informative_baseline_relative_norm"],
                    local_gate["pair_direction_cosine_min"],
                    local_gate["pair_state_recovery_min"],
                ),
                "post_attention": state_metrics(
                    task["post_cf"],
                    task["recipient"]["post_attention"],
                    task["donor"]["post_attention"],
                    freeze["minimum_informative_baseline_relative_norm"],
                    local_gate["pair_direction_cosine_min"],
                    local_gate["pair_state_recovery_min"],
                ),
                "mlp": state_metrics(
                    mlp_cf[task_index],
                    task["recipient"]["mlp"],
                    task["donor"]["mlp"],
                    freeze["minimum_informative_baseline_relative_norm"],
                    local_gate["pair_direction_cosine_min"],
                    local_gate["pair_state_recovery_min"],
                ),
                "layer_output": state_metrics(
                    out_cf[task_index],
                    task["recipient"]["layer_output"],
                    task["donor"]["layer_output"],
                    freeze["minimum_informative_baseline_relative_norm"],
                    local_gate["pair_direction_cosine_min"],
                    local_gate["pair_state_recovery_min"],
                ),
            }
            natural_gap = float(
                (margin_donor[task_index] - margin_recipient[task_index]).item()
            )
            counterfactual_shift = float(
                (margin_cf[task_index] - margin_recipient[task_index]).item()
            )
            target_distinct = int(donor_ids[task_index].item()) != int(
                recipient_ids[task_index].item()
            )
            semantic_informative = bool(
                target_distinct
                and natural_gap >= semantic_gate["natural_competition_gap_min"]
            )
            semantic_recovery = (
                counterfactual_shift / natural_gap if semantic_informative else None
            )
            pair_rows.append(
                {
                    "schema_version": "75.9.0",
                    "phase_id": "Phase401-LocalEdgePair",
                    "created_at": now(),
                    "model": loaded.key,
                    "surface_private": task["recipient_case"]["task_surface_private"],
                    "public_parallel_group_id": task["recipient_case"][
                        "phase401_public_parallel_group_id"
                    ],
                    "pair_id_private": task["pair"]["pair_id"],
                    "recipient_condition_private": task["recipient_condition"],
                    "donor_condition_private": task["donor_condition"],
                    "control_name": task["control"],
                    "receiver_role": task["receiver_role"],
                    "layer_index": layer_index,
                    "layer_count": layer_count,
                    "stages": stages,
                    "semantic_prediction": {
                        "target_ids_distinct": target_distinct,
                        "informative": semantic_informative,
                        "natural_competition_gap": clean(natural_gap),
                        "counterfactual_competition_shift": clean(counterfactual_shift),
                        "competition_recovery": (
                            clean(semantic_recovery)
                            if semantic_recovery is not None
                            else None
                        ),
                        "positive_shift": bool(
                            semantic_informative and counterfactual_shift > 0
                        ),
                        "logit_lens_only": True,
                    },
                    "natural_generation_intervened": False,
                    "head_channel_neuron_selected": False,
                }
            )
        del current, wrong_depth, tasks, post_stack, mlp_cf, out_cf
        del recipient_out, donor_out, normalized, norm_cf, norm_recipient, norm_donor
        del donor_ids, recipient_ids, donor_weights, recipient_weights, difference_weights
        del margin_cf, margin_recipient, margin_donor
        if (layer_index + 1) % 8 == 0 or layer_index + 1 == layer_count:
            print(
                f"[{loaded.key}] Phase401 group layer {layer_index + 1}/{layer_count}",
                flush=True,
            )
    group_rows = aggregate_rows(pair_rows, freeze)
    return pair_rows, group_rows


@torch.inference_mode()
def run(model: str, smoke: bool = False) -> dict[str, Any]:
    freeze = read_json(OUT / "phase401_local_edge_execution_freeze.json")
    if not freeze["authorization"]["discovery_collection"]:
        raise RuntimeError("Phase401 discovery collection is not authorized")
    cases = [
        row for row in read_jsonl(SOURCE) if row["private_execution_model"] == model
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["phase401_public_parallel_group_id"]].append(case)
    group_ids = sorted(grouped)
    if smoke:
        group_ids = group_ids[:1]
    expected_groups = (
        1
        if smoke
        else freeze["discovery_denominator"]["groups_per_surface"]
        * len(freeze["discovery_denominator"]["eligible_surfaces"])
    )
    if len(group_ids) != expected_groups:
        raise RuntimeError(
            f"Phase401 discovery group count for {model}: {len(group_ids)} != {expected_groups}"
        )
    loaded = None
    handles: list[Any] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    group_audits: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        with capture_actual_qkv(model, tuple(range(len(layers))), captures):
            for group_index, group_id in enumerate(group_ids, 1):
                group_cases = sorted(
                    grouped[group_id], key=lambda row: row["anonymous_condition_slot"]
                )
                if len(group_cases) != 16:
                    raise RuntimeError(f"Phase401 incomplete discovery group: {group_id}")
                collected = {
                    case["anonymous_condition_slot"]: capture_case(
                        loaded, layers, captures, case
                    )
                    for case in group_cases
                }
                first_replay = all(
                    item["first_prediction_matches_frozen"] for item in collected.values()
                )
                pair_rows, group_rows = analyze_group(
                    loaded, layers, collected, freeze
                )
                all_pair_rows.extend(pair_rows)
                all_group_rows.extend(group_rows)
                group_audits.append(
                    {
                        "schema_version": "75.9.0",
                        "phase_id": "Phase401-LocalEdgeGroupAudit",
                        "model": model,
                        "surface_private": group_cases[0]["task_surface_private"],
                        "public_parallel_group_id": group_id,
                        "case_count": len(group_cases),
                        "layer_count": len(layers),
                        "pair_row_count": len(pair_rows),
                        "group_layer_control_row_count": len(group_rows),
                        "first_generated_token_replay_all_match": first_replay,
                        "valid": first_replay
                        and len(pair_rows)
                        == 16 * len(CONTROL_NAMES) * len(layers),
                    }
                )
                del collected, pair_rows, group_rows
                gc.collect()
                print(
                    f"[{model}] Phase401 discovery group {group_index}/{len(group_ids)} "
                    f"valid={group_audits[-1]['valid']}",
                    flush=True,
                )
        mode = "smoke" if smoke else "discovery"
        root = OUT / "local_edges" / mode / "private" / model
        write_jsonl(root / "pair_rows.jsonl", all_pair_rows)
        write_jsonl(root / "group_layer_control_rows.jsonl", all_group_rows)
        write_jsonl(root / "group_audit_rows.jsonl", group_audits)
        payload = {
            "schema_version": "75.9.0",
            "phase_id": "Phase401-LocalEdgeCollection",
            "created_at": now(),
            "model": model,
            "mode": mode,
            "group_count": len(group_audits),
            "case_count": len(group_audits) * 16,
            "layer_count": len(layers),
            "control_count_including_true": len(CONTROL_NAMES),
            "pair_row_count": len(all_pair_rows),
            "group_layer_control_row_count": len(all_group_rows),
            "valid_group_count": sum(row["valid"] for row in group_audits),
            "valid": bool(group_audits) and all(row["valid"] for row in group_audits),
            "claim_boundary": {
                "local_response_is_language_path": False,
                "logit_lens_shift_is_generated_behavior_change": False,
            },
        }
        write_json(OUT / "local_edges" / mode / model / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.model, smoke=args.smoke)
