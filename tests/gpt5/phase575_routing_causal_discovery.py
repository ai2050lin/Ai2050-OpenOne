#!/usr/bin/env python3
"""Run the frozen Phase575 query/score/weight causal discovery screen."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_role_position_utils import role_positions  # noqa: E402
from phase573_coarse_message_causal import edge_contribution  # noqa: E402
from phase573_natural_transition_trace import physical_anchor_positions  # noqa: E402
from phase575_natural_ledger import projected_states  # noqa: E402
import phase575_routing_causal_protocol as causal_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


MODEL = "qwen3"
SPLIT = "causal_discovery"
OUT_DIR = protocol.OUT_DIR
BEHAVIOR_REGISTRY = (
    OUT_DIR / "phase575_qwen3_routing_causal_behavior_registry.json"
)
BEHAVIOR_SUMMARY = OUT_DIR / "phase575_qwen3_routing_causal_behavior_summary.json"
ROWS_PATH = OUT_DIR / "phase575_qwen3_routing_causal_discovery_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase575_qwen3_routing_causal_discovery_summary.json"
DECISION_PATH = OUT_DIR / "phase575_routing_causal_discovery_decision.json"
CONTRACT_PATH = OUT_DIR / "phase575_qwen3_routing_causal_discovery_contract.json"

TARGET_ROLES = (
    "anchor_target_fact_object",
    "anchor_target_fact_relation",
    "anchor_target_fact_value",
)
OTHER_ROLES = (
    "anchor_other_fact_object",
    "anchor_other_fact_relation",
    "anchor_other_fact_value",
)
VARIANT_INDEX = {variant: index for index, variant in enumerate(protocol.VARIANTS)}
RESTORE_CONDITIONS = {
    "q_relation_score_restore",
    "score_equalize_restore",
    "score_relation_weight_restore",
    "weight_relation_restore",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def union_positions(groups: dict[str, list[int]], roles: tuple[str, ...]) -> list[int]:
    return sorted(position for role in roles for position in groups[role])


def replace_attention_output(output: Any, primary: torch.Tensor, weights: torch.Tensor) -> Any:
    if not isinstance(output, tuple):
        return primary
    return (primary, weights, *output[2:])


def load_worlds() -> list[list[dict[str, Any]]]:
    registry = read_json(BEHAVIOR_REGISTRY)
    summary = read_json(BEHAVIOR_SUMMARY)
    if not registry["authorized_for_routing_causal_test"]:
        raise RuntimeError("Phase575 causal behavior registry is not authorized")
    if summary["registry_sha256"] != sha256_file(BEHAVIOR_REGISTRY):
        raise RuntimeError("Phase575 causal behavior registry hash drift")
    selected = registry["selected_base_case_ids_by_split"][SPLIT]
    if len(selected) != 128:
        raise RuntimeError("Phase575 causal discovery requires 128 frozen worlds")
    selected_set = set(selected)
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(protocol.OPEN_CASES_PATH):
        if row["split"] == SPLIT and row["base_case_id"] in selected_set:
            if row["sealed"]:
                raise RuntimeError("sealed case reached Phase575 causal discovery")
            bank[row["base_case_id"]][row["variant"]] = row
    worlds = []
    for base_id in selected:
        variants = bank.get(base_id, {})
        if set(variants) != set(protocol.VARIANTS):
            raise RuntimeError(f"Phase575 incomplete causal world: {base_id}")
        worlds.append([variants[variant] for variant in protocol.VARIANTS])
    return worlds


def prepare_batch(
    tokenizer: Any,
    worlds: list[list[dict[str, Any]]],
    padding_side: str = "right",
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if padding_side not in ("left", "right"):
        raise ValueError(f"Unsupported padding side: {padding_side}")
    tokenizer.padding_side = padding_side
    rows = [row for world in worlds for row in world]
    prompts = [render_chat(tokenizer, MODEL, row["raw_prompt"]) for row in rows]
    individual = []
    for prompt, row in zip(prompts, rows):
        ids, roles = role_positions(tokenizer, prompt, row)
        anchors = physical_anchor_positions(tokenizer, prompt, row)
        individual.append((ids, roles, anchors))
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    width = int(encoded["input_ids"].shape[1])
    positions = []
    for batch_index, (ids, roles, anchors) in enumerate(individual):
        active = encoded["input_ids"][batch_index][
            encoded["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active] != ids:
            raise RuntimeError("Phase575 causal discovery tokenization drift")
        offset = width - len(ids) if padding_side == "left" else 0
        positions.append(
            {
                "query_terminal": int(roles["query_terminal"][-1]) + offset,
                "answer_boundary": int(roles["answer_boundary"][-1]) + offset,
                "anchor_selected": [
                    position + offset
                    for position in union_positions(anchors, TARGET_ROLES)
                ],
                "anchor_other": [
                    position + offset
                    for position in union_positions(anchors, OTHER_ROLES)
                ],
                "active_length": len(ids),
                "padding_offset": offset,
            }
        )
    position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    encoded["position_ids"] = position_ids

    meta = []
    for world_index, world in enumerate(worlds):
        indices = {
            variant: world_index * len(protocol.VARIANTS) + VARIANT_INDEX[variant]
            for variant in protocol.VARIANTS
        }
        meta.append(
            {
                "base_case_id": world[0]["base_case_id"],
                "indices": indices,
                "positions": {
                    variant: positions[indices[variant]]
                    for variant in protocol.VARIANTS
                },
                "base_receiver": positions[indices["base"]]["answer_boundary"],
                "base_query_terminal": positions[indices["base"]]["query_terminal"],
                "targets": {
                    variant: world[VARIANT_INDEX[variant]]["target"]
                    for variant in protocol.VARIANTS
                },
                "candidate_token_ids": world[0]["candidate_token_ids_by_model"][MODEL],
            }
        )
    return encoded, meta


def attention_mask_row(
    attention_mask: torch.Tensor | None,
    batch_index: int,
    receiver: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask[batch_index, 0, receiver, :]


def normalized_weights(
    score_row: torch.Tensor,
    mask_row: torch.Tensor | None,
) -> torch.Tensor:
    masked = score_row if mask_row is None else score_row + mask_row
    return torch.softmax(masked.float(), dim=-1).to(score_row.dtype)


def copy_group(
    destination: torch.Tensor,
    donor: torch.Tensor,
    destination_positions: list[int],
    donor_positions: list[int],
) -> None:
    if len(destination_positions) == len(donor_positions):
        destination[:, destination_positions] = donor[:, donor_positions]
        return
    donor_mean = donor[:, donor_positions].mean(dim=-1, keepdim=True)
    destination[:, destination_positions] = donor_mean


def equalize_group_means(
    score_row: torch.Tensor,
    selected: list[int],
    other: list[int],
) -> torch.Tensor:
    output = score_row.clone()
    selected_mean = output[:, selected].mean(dim=-1, keepdim=True)
    other_mean = output[:, other].mean(dim=-1, keepdim=True)
    pooled = (selected_mean + other_mean) / 2.0
    output[:, selected] += pooled - selected_mean
    output[:, other] += pooled - other_mean
    return output


def deterministic_roll(base_id: str, width: int) -> int:
    digest = hashlib.sha256(f"Phase575|q-roll|{base_id}".encode()).digest()
    return 1 + int.from_bytes(digest[:4], "big") % max(1, width - 1)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first = tensor[..., : tensor.shape[-1] // 2]
    second = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def pre_rotary_query(module: Any, hidden: torch.Tensor) -> torch.Tensor:
    hidden_shape = (*hidden.shape[:-1], -1, module.head_dim)
    query = module.q_proj(hidden).view(hidden_shape)
    if hasattr(module, "q_norm"):
        query = module.q_norm(query)
    return query.transpose(1, 2)


def route_metrics(
    module: Any,
    weights: torch.Tensor,
    values: torch.Tensor,
    batch_index: int,
    receiver: int,
    selected: list[int],
    other: list[int],
) -> dict[str, float]:
    selected_mass = finite(
        float(weights[batch_index, :, receiver, selected].float().sum(-1).mean().item())
    )
    other_mass = finite(
        float(weights[batch_index, :, receiver, other].float().sum(-1).mean().item())
    )
    selected_message = edge_contribution(
        module, weights, values, batch_index, receiver, selected
    )
    other_message = edge_contribution(
        module, weights, values, batch_index, receiver, other
    )
    return {
        "anchor_selected_weight_mass": selected_mass,
        "anchor_other_relation_weight_mass": other_mass,
        "relation_route_switch_margin": other_mass - selected_mass,
        "anchor_selected_message_norm": finite(
            float(selected_message.float().norm().item())
        ),
        "anchor_other_relation_message_norm": finite(
            float(other_message.float().norm().item())
        ),
    }


def candidate_outcomes(
    result: Any,
    metrics: list[dict[str, float]],
    meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for world_index, item in enumerate(meta):
        batch_index = int(item["indices"]["base"])
        receiver = int(item["base_receiver"])
        logits = result.logits[batch_index, receiver].float()
        candidate_scores = {
            value: finite(float(logits[int(token_ids[0])].item()))
            for value, token_ids in item["candidate_token_ids"].items()
        }
        output.append(
            {
                **metrics[world_index],
                "candidate_scores": candidate_scores,
                "candidate_winner": max(candidate_scores, key=candidate_scores.get),
            }
        )
    return output


def natural_forward(
    loaded: Any,
    layers: list[Any],
    encoded_cpu: dict[str, torch.Tensor],
    meta: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]], float]:
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    captures: dict[int, dict[str, torch.Tensor]] = {}
    metrics: list[dict[str, float]] = []
    reconstruction_max = 0.0

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            nonlocal metrics, reconstruction_max
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            mask = kwargs.get("attention_mask")
            if (
                hidden is None
                or position_embeddings is None
                or not isinstance(output, tuple)
                or output[1] is None
            ):
                raise RuntimeError("Phase575 causal discovery requires eager attention")
            query, key, value = projected_states(module, hidden, position_embeddings)
            query_pre = pre_rotary_query(module, hidden)
            raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            reconstructed = raw_scores if mask is None else raw_scores + mask
            reconstructed = torch.softmax(reconstructed.float(), dim=-1).to(query.dtype)
            reconstruction_max = max(
                reconstruction_max,
                finite(float((reconstructed - output[1]).float().abs().max().item())),
            )
            captures[layer_index] = {
                "q_pre_answer": torch.stack(
                    [
                        query_pre[
                            item["indices"][variant],
                            :,
                            item["positions"][variant]["answer_boundary"],
                            :,
                        ]
                        for item in meta
                        for variant in protocol.VARIANTS
                    ]
                ).detach(),
                "q_pre_query": torch.stack(
                    [
                        query_pre[
                            item["indices"][variant],
                            :,
                            item["positions"][variant]["query_terminal"],
                            :,
                        ]
                        for item in meta
                        for variant in protocol.VARIANTS
                    ]
                ).detach(),
            }
            if layer_index == 24:
                metrics = []
                for item in meta:
                    batch_index = int(item["indices"]["base"])
                    position = item["positions"]["base"]
                    metrics.append(
                        route_metrics(
                            module,
                            output[1],
                            value,
                            batch_index,
                            int(position["answer_boundary"]),
                            position["anchor_selected"],
                            position["anchor_other"],
                        )
                    )
        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index in (23, 24)
    ]
    try:
        with torch.inference_mode():
            result = loaded.model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != {23, 24} or len(metrics) != len(meta):
        raise RuntimeError("Phase575 causal natural capture drift")
    outcomes = candidate_outcomes(result, metrics, meta)
    del result, encoded
    return captures, outcomes, reconstruction_max


def capture_index(world_index: int, variant: str) -> int:
    return world_index * len(protocol.VARIANTS) + VARIANT_INDEX[variant]


def patched_forward(
    loaded: Any,
    layers: list[Any],
    encoded_cpu: dict[str, torch.Tensor],
    meta: list[dict[str, Any]],
    captures: dict[int, dict[str, torch.Tensor]],
    condition: str,
) -> list[dict[str, Any]]:
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    metrics: list[dict[str, float]] = []

    def hook(
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        nonlocal metrics
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get("position_embeddings")
        mask = kwargs.get("attention_mask")
        if (
            hidden is None
            or position_embeddings is None
            or not isinstance(output, tuple)
            or output[1] is None
        ):
            raise RuntimeError("Phase575 causal patch requires eager attention")
        query, key, value = projected_states(module, hidden, position_embeddings)
        cos, sin = position_embeddings
        raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        primary = output[0].clone()
        weights = output[1].clone()
        values_for_metrics = (
            value.clone()
            if condition == "value_group_swap_positive_control"
            else value
        )

        for world_index, item in enumerate(meta):
            batch_index = int(item["indices"]["base"])
            base_position = item["positions"]["base"]
            answer_receiver = int(base_position["answer_boundary"])
            query_receiver = int(base_position["query_terminal"])
            receiver = (
                query_receiver
                if condition == "wrong_receiver_q_relation_replace"
                else answer_receiver
            )
            mask_row = attention_mask_row(mask, batch_index, receiver)
            new_weight_row = weights[batch_index, :, receiver, :].clone()
            value_for_output = values_for_metrics[batch_index]

            def aligned_query(variant: str, layer_index: int = 24) -> torch.Tensor:
                donor_capture = capture_index(world_index, variant)
                capture_name = (
                    "q_pre_query"
                    if condition == "wrong_receiver_q_relation_replace"
                    else "q_pre_answer"
                )
                donor = captures[layer_index][capture_name][donor_capture]
                cos_row = cos[batch_index, receiver]
                sin_row = sin[batch_index, receiver]
                return donor * cos_row + rotate_half(donor) * sin_row

            def donor_score_row(variant: str) -> torch.Tensor:
                donor_query = aligned_query(variant)
                return torch.einsum(
                    "hd,hsd->hs", donor_query, key[batch_index]
                ) * module.scaling

            if condition in RESTORE_CONDITIONS:
                # The frozen natural score/weight tensor is the final direct overwrite.
                pass
            elif condition.startswith("q_") or condition.startswith("wrong_receiver_q"):
                if condition == "q_object_replace":
                    q_new = aligned_query("object_swap")
                elif condition == "q_order_replace":
                    q_new = aligned_query("order_swap")
                elif condition == "q_relation_wrong_depth_rescaled":
                    donor = aligned_query("relation_swap", 23).float()
                    base = query[batch_index, :, receiver, :].float()
                    q_new = donor * (
                        base.norm(dim=-1, keepdim=True)
                        / donor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    )
                    q_new = q_new.to(query.dtype)
                elif condition == "q_relation_delta_roll":
                    base = query[batch_index, :, receiver, :]
                    relation = aligned_query("relation_swap")
                    shift = deterministic_roll(item["base_case_id"], base.shape[-1])
                    q_new = base + torch.roll(relation - base, shift, dims=-1)
                    q_new = q_new * (
                        relation.float().norm(dim=-1, keepdim=True)
                        / q_new.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    ).to(q_new.dtype)
                elif condition == "wrong_receiver_q_relation_replace":
                    q_new = aligned_query("relation_swap")
                else:
                    q_new = aligned_query("relation_swap")
                score_row = torch.einsum(
                    "hd,hsd->hs", q_new, key[batch_index]
                ) * module.scaling
                new_weight_row = normalized_weights(score_row, mask_row)
            elif condition.startswith("score_"):
                score_row = raw_scores[batch_index, :, receiver, :].clone()
                if condition == "score_equalize":
                    score_row = equalize_group_means(
                        score_row,
                        base_position["anchor_selected"],
                        base_position["anchor_other"],
                    )
                else:
                    donor_variant = {
                        "score_relation_replace": "relation_swap",
                        "score_object_replace": "object_swap",
                        "score_order_replace": "order_swap",
                    }[condition]
                    donor_row = donor_score_row(donor_variant)
                    copy_group(
                        score_row,
                        donor_row,
                        base_position["anchor_selected"],
                        base_position["anchor_selected"],
                    )
                    copy_group(
                        score_row,
                        donor_row,
                        base_position["anchor_other"],
                        base_position["anchor_other"],
                    )
                new_weight_row = normalized_weights(score_row, mask_row)
            elif condition.startswith("weight_"):
                donor_variant = {
                    "weight_relation_replace": "relation_swap",
                    "weight_object_replace": "object_swap",
                    "weight_order_replace": "order_swap",
                }[condition]
                donor_score = donor_score_row(donor_variant)
                donor_row = normalized_weights(donor_score, mask_row)
                copy_group(
                    new_weight_row,
                    donor_row,
                    base_position["anchor_selected"],
                    base_position["anchor_selected"],
                )
                copy_group(
                    new_weight_row,
                    donor_row,
                    base_position["anchor_other"],
                    base_position["anchor_other"],
                )
                new_weight_row = new_weight_row / new_weight_row.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            elif condition == "value_group_swap_positive_control":
                selected = base_position["anchor_selected"]
                other = base_position["anchor_other"]
                selected_mean = value_for_output[:, selected, :].mean(
                    dim=1, keepdim=True
                )
                other_mean = value_for_output[:, other, :].mean(dim=1, keepdim=True)
                value_for_output[:, selected, :] = other_mean
                value_for_output[:, other, :] = selected_mean
            else:
                raise RuntimeError(f"Unknown Phase575 causal condition: {condition}")

            if condition not in RESTORE_CONDITIONS:
                head_output = torch.einsum(
                    "hs,hsd->hd", new_weight_row, value_for_output
                )
                projected = module.o_proj(head_output.reshape(1, -1)).squeeze(0)
                primary[batch_index, receiver, :] = projected
                weights[batch_index, :, receiver, :] = new_weight_row

        metrics = []
        for world_index, item in enumerate(meta):
            batch_index = int(item["indices"]["base"])
            position = item["positions"]["base"]
            metrics.append(
                route_metrics(
                    module,
                    weights,
                    values_for_metrics,
                    batch_index,
                    int(position["answer_boundary"]),
                    position["anchor_selected"],
                    position["anchor_other"],
                )
            )
        return replace_attention_output(output, primary, weights)

    handle = layers[24].self_attn.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            result = loaded.model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        handle.remove()
    if len(metrics) != len(meta):
        raise RuntimeError("Phase575 patched metric capture drift")
    outcomes = candidate_outcomes(result, metrics, meta)
    del result, encoded
    return outcomes


def causal_row(
    item: dict[str, Any],
    condition: str,
    baseline: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    base_target = item["targets"]["base"]
    relation_target = item["targets"]["relation_swap"]
    object_target = item["targets"]["object_swap"]
    base_scores = baseline["candidate_scores"]
    scores = outcome["candidate_scores"]
    relation_logit_effect = (
        scores[relation_target]
        - scores[base_target]
        - base_scores[relation_target]
        + base_scores[base_target]
    )
    return {
        "schema_version": "phase575_routing_causal_discovery_row.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "split": SPLIT,
        "base_case_id": item["base_case_id"],
        "condition": condition,
        "base_target": base_target,
        "relation_target": relation_target,
        "object_target": object_target,
        "baseline_candidate_scores": base_scores,
        "intervention_candidate_scores": scores,
        "baseline_candidate_winner": baseline["candidate_winner"],
        "intervention_candidate_winner": outcome["candidate_winner"],
        "baseline_relation_route_switch_margin": baseline[
            "relation_route_switch_margin"
        ],
        "intervention_relation_route_switch_margin": outcome[
            "relation_route_switch_margin"
        ],
        "relation_route_switch_effect": (
            outcome["relation_route_switch_margin"]
            - baseline["relation_route_switch_margin"]
        ),
        "relation_logit_switch_effect": relation_logit_effect,
        "relation_target_wins": outcome["candidate_winner"] == relation_target,
        "object_target_wins": outcome["candidate_winner"] == object_target,
        "maximum_candidate_logit_delta": max(
            abs(scores[value] - base_scores[value]) for value in scores
        ),
        "baseline_anchor_selected_weight_mass": baseline[
            "anchor_selected_weight_mass"
        ],
        "intervention_anchor_selected_weight_mass": outcome[
            "anchor_selected_weight_mass"
        ],
        "baseline_anchor_other_relation_weight_mass": baseline[
            "anchor_other_relation_weight_mass"
        ],
        "intervention_anchor_other_relation_weight_mass": outcome[
            "anchor_other_relation_weight_mass"
        ],
        "baseline_anchor_selected_message_norm": baseline[
            "anchor_selected_message_norm"
        ],
        "intervention_anchor_selected_message_norm": outcome[
            "anchor_selected_message_norm"
        ],
        "baseline_anchor_other_relation_message_norm": baseline[
            "anchor_other_relation_message_norm"
        ],
        "intervention_anchor_other_relation_message_norm": outcome[
            "anchor_other_relation_message_norm"
        ],
        "post_rotary_query_used": condition.startswith("q_")
        or condition.startswith("wrong_receiver_q"),
        "recipient_values_preserved": condition
        != "value_group_swap_positive_control",
        "direct_restore": condition in RESTORE_CONDITIONS,
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed": False,
    }


def condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    route = [float(row["relation_route_switch_effect"]) for row in selected]
    logits = [float(row["relation_logit_switch_effect"]) for row in selected]
    maximum_deltas = [float(row["maximum_candidate_logit_delta"]) for row in selected]
    return {
        "world_count": len(selected),
        "relation_route_effect_mean": mean(route),
        "relation_route_effect_positive_rate": rate([value > 0.0 for value in route]),
        "relation_logit_effect_mean": mean(logits),
        "relation_logit_effect_positive_rate": rate([value > 0.0 for value in logits]),
        "relation_target_win_rate": rate([row["relation_target_wins"] for row in selected]),
        "maximum_candidate_logit_delta": max(maximum_deltas) if maximum_deltas else 0.0,
        "route_effect_by_world": {
            row["base_case_id"]: float(row["relation_route_switch_effect"])
            for row in selected
        },
    }


def maximum_branch_resample(
    branch_metrics: dict[str, dict[str, Any]],
    branches: dict[str, dict[str, str]],
    count: int,
) -> dict[str, Any]:
    contrasts: dict[str, dict[str, float]] = {}
    for branch, definition in branches.items():
        relation = branch_metrics[definition["relation"]]["route_effect_by_world"]
        obj = branch_metrics[definition["object"]]["route_effect_by_world"]
        order = branch_metrics[definition["order"]]["route_effect_by_world"]
        contrasts[branch] = {
            world: relation[world] - max(obj[world], order[world]) for world in relation
        }
    observed_by_branch = {
        branch: mean(list(values.values())) for branch, values in contrasts.items()
    }
    observed_maximum = max(observed_by_branch.values())
    at_least = 0
    for permutation in range(count):
        permuted_means = []
        for branch, values in contrasts.items():
            permuted = []
            for world, value in sorted(values.items()):
                digest = hashlib.sha256(
                    f"Phase575|pipeline|{permutation}|{branch}|{world}".encode()
                ).digest()
                permuted.append(value if digest[0] & 1 else -value)
            permuted_means.append(mean(permuted))
        at_least += int(max(permuted_means) >= observed_maximum)
    return {
        "observed_contrast_mean_by_branch": observed_by_branch,
        "observed_maximum_branch_contrast_mean": observed_maximum,
        "resample_count": count,
        "count_at_least_observed": at_least,
        "smoothed_tail_fraction": (at_least + 1) / (count + 1),
    }


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    gates = frozen["discovery_gate"]
    branches = frozen["branch_definitions"]
    metrics = {
        condition: condition_metrics(rows, condition)
        for condition in frozen["discovery_conditions"]
    }
    branch_results = {}
    for branch, definition in branches.items():
        relation = metrics[definition["relation"]]
        obj = metrics[definition["object"]]
        order = metrics[definition["order"]]
        restore = metrics[definition["restore_or_mediator"]]
        relation_mean = relation["relation_route_effect_mean"]
        remaining = abs(restore["relation_route_effect_mean"]) / max(
            abs(relation_mean), 1e-12
        )
        physical_gate = (
            relation["relation_route_effect_positive_rate"]
            >= gates["relation_route_effect_positive_rate_minimum"]
            and relation_mean >= gates["relation_route_effect_mean_minimum"]
            and relation_mean - obj["relation_route_effect_mean"]
            >= gates["relation_vs_object_effect_gap_minimum"]
            and relation_mean - order["relation_route_effect_mean"]
            >= gates["relation_vs_order_effect_gap_minimum"]
            and abs(restore["relation_route_effect_mean"])
            <= gates["restore_route_maximum_absolute_delta"]
            and restore["maximum_candidate_logit_delta"]
            <= gates["restore_candidate_logit_maximum_absolute_delta"]
            and remaining <= gates["mediation_remaining_fraction_maximum"]
        )
        behavior_gate = (
            relation["relation_logit_effect_positive_rate"]
            >= gates["behavior_relation_logit_effect_positive_rate_minimum"]
            and relation["relation_logit_effect_mean"]
            >= gates["behavior_relation_logit_effect_mean_minimum"]
        )
        branch_results[branch] = {
            "relation_condition": definition["relation"],
            "object_control_condition": definition["object"],
            "order_control_condition": definition["order"],
            "restore_or_mediator_condition": definition["restore_or_mediator"],
            "relation_route_effect_mean": relation_mean,
            "relation_route_effect_positive_rate": relation[
                "relation_route_effect_positive_rate"
            ],
            "object_control_route_effect_mean": obj["relation_route_effect_mean"],
            "order_control_route_effect_mean": order["relation_route_effect_mean"],
            "relation_vs_object_gap": relation_mean
            - obj["relation_route_effect_mean"],
            "relation_vs_order_gap": relation_mean
            - order["relation_route_effect_mean"],
            "restore_route_effect_mean": restore["relation_route_effect_mean"],
            "restore_maximum_candidate_logit_delta": restore[
                "maximum_candidate_logit_delta"
            ],
            "mediation_remaining_fraction": remaining,
            "relation_logit_effect_mean": relation["relation_logit_effect_mean"],
            "relation_logit_effect_positive_rate": relation[
                "relation_logit_effect_positive_rate"
            ],
            "physical_routing_gate_pass": physical_gate,
            "behavior_gate_pass": behavior_gate,
        }
    resample = maximum_branch_resample(
        metrics,
        branches,
        int(gates["pipeline_resample_count"]),
    )
    resample_pass = resample["smoothed_tail_fraction"] <= gates[
        "maximum_branch_smoothed_tail_fraction"
    ]
    eligible = [
        branch
        for branch, values in branch_results.items()
        if values["physical_routing_gate_pass"] and resample_pass
    ]
    selected = (
        max(
            eligible,
            key=lambda branch: (
                branch_results[branch]["relation_route_effect_mean"], branch
            ),
        )
        if eligible
        else None
    )
    summary = {
        "schema_version": "phase575_routing_causal_discovery_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model": MODEL,
        "split": SPLIT,
        "world_count": len({row["base_case_id"] for row in rows}),
        "row_count": len(rows),
        "condition_metrics": {
            condition: {
                key: value
                for key, value in values.items()
                if key != "route_effect_by_world"
            }
            for condition, values in metrics.items()
        },
        "branch_results": branch_results,
        "maximum_branch_pipeline_resample": resample,
        "pipeline_resample_pass": resample_pass,
        "eligible_physical_branches": eligible,
        "selected_branch": selected,
        "selected_branch_behavior_gate_pass": bool(
            selected is not None and branch_results[selected]["behavior_gate_pass"]
        ),
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    decision = {
        "schema_version": "phase575_routing_causal_discovery_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model": MODEL,
        "selected_natural_coordinate": read_json(causal_protocol.CAUSAL_PROTOCOL)[
            "selected_coordinate"
        ],
        "selected_causal_branch": selected,
        "open_discovery_physical_routing_gate_pass": selected is not None,
        "open_discovery_behavior_gate_pass": summary[
            "selected_branch_behavior_gate_pass"
        ],
        "confirmation_internal_state_authorized": selected is not None,
        "sealed_split_authorized": False,
        "causal_candidate_not_yet_a_mechanism": True,
        "causal_confirmation_executed": False,
        "full_short_generation_executed": False,
        "sealed_split_read": False,
        "summary_sha256": None,
    }
    return summary, decision


def run(restart: bool) -> Path:
    if restart:
        for path in (ROWS_PATH, SUMMARY_PATH, DECISION_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase575 routing causal discovery requires CUDA")
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    worlds = load_worlds()
    contract = {
        "schema_version": "phase575_routing_causal_discovery_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "split": SPLIT,
        "world_count": len(worlds),
        "conditions": frozen["discovery_conditions"],
        "selected_coordinate": frozen["selected_coordinate"],
        "causal_protocol_sha256": sha256_file(causal_protocol.CAUSAL_PROTOCOL),
        "behavior_registry_sha256": sha256_file(BEHAVIOR_REGISTRY),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "torch_dtype_requested": "torch.bfloat16",
        "cuda_required": True,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    write_json(CONTRACT_PATH, contract)
    loaded = None
    rows_out: list[dict[str, Any]] = []
    reconstruction_max = 0.0
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        if loaded.input_device.type != "cuda":
            raise RuntimeError("Phase575 causal discovery model is not on CUDA")
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase575 causal discovery requires BF16, got {dtype}")
        loaded.tokenizer.padding_side = "right"
        loaded.model.config._attn_implementation = "eager"
        layers = get_layers(loaded.model)
        if len(layers) != 36:
            raise RuntimeError(f"Phase575 Qwen3 layer drift: {len(layers)}")
        batch_size = int(frozen["execution"]["world_batch_size"])
        for start in range(0, len(worlds), batch_size):
            batch_worlds = worlds[start : start + batch_size]
            encoded_cpu, meta = prepare_batch(loaded.tokenizer, batch_worlds)
            captures, baseline, error = natural_forward(
                loaded, layers, encoded_cpu, meta
            )
            reconstruction_max = max(reconstruction_max, error)
            for world_index, item in enumerate(meta):
                rows_out.append(
                    causal_row(
                        item,
                        "natural_baseline",
                        baseline[world_index],
                        baseline[world_index],
                    )
                )
            for condition in frozen["discovery_conditions"]:
                if condition == "natural_baseline":
                    continue
                outcomes = patched_forward(
                    loaded,
                    layers,
                    encoded_cpu,
                    meta,
                    captures,
                    condition,
                )
                for world_index, item in enumerate(meta):
                    rows_out.append(
                        causal_row(
                            item,
                            condition,
                            baseline[world_index],
                            outcomes[world_index],
                        )
                    )
            del encoded_cpu, captures, baseline
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 causal-discovery "
                f"{min(start + batch_size, len(worlds))}/{len(worlds)}",
                flush=True,
            )

        write_jsonl(ROWS_PATH, rows_out)
        summary, decision = analyze(rows_out)
        summary.update(
            {
                "device_type": loaded.input_device.type,
                "torch_dtype": dtype,
                "runtime_seconds": time.monotonic() - started,
                "attention_weight_reconstruction_max_abs_error": reconstruction_max,
                "attention_weight_reconstruction_pass": reconstruction_max <= 0.01,
                "rows_sha256": sha256_file(ROWS_PATH),
                "contract_sha256": sha256_file(CONTRACT_PATH),
            }
        )
        write_json(SUMMARY_PATH, summary)
        decision["summary_sha256"] = sha256_file(SUMMARY_PATH)
        write_json(DECISION_PATH, decision)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.restart)


if __name__ == "__main__":
    main()
