#!/usr/bin/env python3
"""Capture the Phase575 full-depth natural source-competition ledger."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import json
import math
import os
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
import phase575_natural_ledger_protocol as ledger_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


VARIANTS = protocol.VARIANTS
TARGET_ROLES = (
    "target_fact_object", "target_fact_relation", "target_fact_value",
)
OTHER_ROLES = (
    "other_fact_object", "other_fact_relation", "other_fact_value",
)
ANCHOR_TARGET_ROLES = (
    "anchor_target_fact_object", "anchor_target_fact_relation",
    "anchor_target_fact_value",
)
ANCHOR_OTHER_ROLES = (
    "anchor_other_fact_object", "anchor_other_fact_relation",
    "anchor_other_fact_value",
)


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


def relative_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.float()
    right_f = right.float()
    scale = 0.5 * (left_f.norm() + right_f.norm())
    return finite(float(
        (left_f - right_f).norm().div(scale.clamp_min(1e-12)).item()
    ))


def union_positions(groups: dict[str, list[int]], roles: tuple[str, ...]) -> list[int]:
    return sorted(position for role in roles for position in groups[role])


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase575_{model}_natural_ledger"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "snapshots": stem.with_name(stem.name + "_snapshots.pt"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def load_worlds(model: str) -> list[tuple[str, str, list[dict[str, Any]]]]:
    frozen = read_json(ledger_protocol.LEDGER_PROTOCOL)
    if model not in frozen["authorized_models"]:
        raise RuntimeError(f"Phase575 natural ledger is not authorized for {model}")
    selected = frozen["selected_base_case_ids_by_model_and_split"][model]
    selected_ids = set().union(*(set(ids) for ids in selected.values()))
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(protocol.OPEN_CASES_PATH):
        if row["base_case_id"] not in selected_ids:
            continue
        if row["split"] not in protocol.STRUCTURE_SPLITS or row["sealed"]:
            raise RuntimeError("Phase575 natural ledger attempted forbidden split access")
        bank[row["base_case_id"]][row["variant"]] = row
    worlds = []
    for split in protocol.STRUCTURE_SPLITS:
        for base_id in selected[split]:
            variants = bank.get(base_id, {})
            if set(variants) != set(VARIANTS):
                raise RuntimeError(f"Phase575 incomplete ledger world: {base_id}")
            rows = [variants[variant] for variant in VARIANTS]
            if any(row["split"] != split for row in rows):
                raise RuntimeError("Phase575 natural ledger split identity drift")
            worlds.append((split, base_id, rows))
    if len(worlds) != 384:
        raise RuntimeError(f"Phase575 natural ledger world count drift: {len(worlds)}")
    return worlds


def prepare_contract(
    model: str,
    worlds: list[tuple[str, str, list[dict[str, Any]]]],
    restart: bool,
) -> None:
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    frozen = read_json(ledger_protocol.LEDGER_PROTOCOL)
    payload = {
        "schema_version": "phase575_natural_ledger_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "ledger_protocol_sha256": sha256_file(ledger_protocol.LEDGER_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "world_count": len(worlds),
        "world_ids": [base_id for _, base_id, _ in worlds],
        "variants": list(VARIANTS),
        "trace_every_layer": True,
        "expected_layer_count": frozen["layer_count_by_model"][model],
        "post_rotary_query_and_key": True,
        "pre_softmax_score": True,
        "post_softmax_weight": True,
        "output_embedding_direction_used": False,
        "causal_intervention_executed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
    }
    contract_path = output["contract"]
    if contract_path.exists():
        existing = read_json(contract_path)
        for key, value in payload.items():
            if key != "created_at" and existing[key] != value:
                raise RuntimeError(f"Phase575 natural ledger contract drift: {key}")
    else:
        write_json(contract_path, payload)


def apply_model_rotary(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    model_module = importlib.import_module(module.__class__.__module__)
    apply_rotary = getattr(model_module, "apply_rotary_pos_emb")
    cos, sin = position_embeddings
    return apply_rotary(query, key, cos, sin)


def projected_states(
    module: Any,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden).view(hidden_shape)
    key = module.k_proj(hidden).view(hidden_shape)
    if hasattr(module, "q_norm"):
        query = module.q_norm(query)
    if hasattr(module, "k_norm"):
        key = module.k_norm(key)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = module.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    query, key = apply_model_rotary(module, query, key, position_embeddings)
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    return query, key, value


def source_metrics(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    raw_scores: torch.Tensor,
    weights: torch.Tensor,
    batch_index: int,
    receiver: int,
    source_positions: list[int],
    keep_tensors: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    keys = key[batch_index, :, source_positions, :].float().mean(dim=1)
    values = value[batch_index, :, source_positions, :].float().mean(dim=1)
    score_values = raw_scores[batch_index, :, receiver, source_positions].float()
    weight_values = weights[batch_index, :, receiver, source_positions].float()
    message = edge_contribution(
        module, weights, value, batch_index, receiver, source_positions
    ).detach()
    scalars = {
        "source_post_rotary_key_norm": finite(float(keys.norm().item())),
        "source_value_norm": finite(float(values.norm().item())),
        "source_pre_softmax_score_mean": finite(float(score_values.mean().item())),
        "source_pre_softmax_score_max_mean": finite(float(
            score_values.max(dim=-1).values.mean().item()
        )),
        "source_post_softmax_weight_mass": finite(float(
            weight_values.sum(dim=-1).mean().item()
        )),
        "source_projected_value_message_norm": finite(float(
            message.float().norm().item()
        )),
    }
    tensors = {}
    if keep_tensors:
        tensors = {
            "source_post_rotary_key": keys.detach().cpu(),
            "source_value": values.detach().cpu(),
            "source_score_by_head": score_values.mean(dim=-1).detach().cpu(),
            "source_weight_mass_by_head": weight_values.sum(dim=-1).detach().cpu(),
            "source_projected_value_message": message.detach().cpu(),
        }
    return scalars, tensors


def pair_margins(sources: dict[str, dict[str, float]]) -> dict[str, float]:
    selected = sources["semantic_selected"]
    other = sources["semantic_other_relation"]
    anchor_selected = sources["anchor_base_selected"]
    anchor_other = sources["anchor_base_other_relation"]
    return {
        "semantic_score_margin": (
            selected["source_pre_softmax_score_mean"]
            - other["source_pre_softmax_score_mean"]
        ),
        "semantic_weight_margin": (
            selected["source_post_softmax_weight_mass"]
            - other["source_post_softmax_weight_mass"]
        ),
        "semantic_message_norm_margin": (
            selected["source_projected_value_message_norm"]
            - other["source_projected_value_message_norm"]
        ),
        "anchor_score_margin": (
            anchor_selected["source_pre_softmax_score_mean"]
            - anchor_other["source_pre_softmax_score_mean"]
        ),
        "anchor_weight_margin": (
            anchor_selected["source_post_softmax_weight_mass"]
            - anchor_other["source_post_softmax_weight_mass"]
        ),
        "anchor_message_norm_margin": (
            anchor_selected["source_projected_value_message_norm"]
            - anchor_other["source_projected_value_message_norm"]
        ),
    }


def nested_numeric_max_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError("Phase575 duplicate trace key drift")
        return max(
            (
                nested_numeric_max_delta(left[key], right[key])
                for key in left
                if key != "created_at"
            ),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError("Phase575 duplicate trace length drift")
        return max(
            (
                nested_numeric_max_delta(left_item, right_item)
                for left_item, right_item in zip(left, right)
            ),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def trace_world(
    loaded: Any,
    layers: list[Any],
    split: str,
    base_id: str,
    rows: list[dict[str, Any]],
    keep_snapshot: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, float]]:
    prompts = [render_chat(loaded.tokenizer, loaded.key, row["raw_prompt"]) for row in rows]
    individual = []
    for prompt, row in zip(prompts, rows):
        ids, roles = role_positions(loaded.tokenizer, prompt, row)
        anchors = physical_anchor_positions(loaded.tokenizer, prompt, row)
        individual.append((ids, roles, anchors))
    encoded_cpu = loaded.tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    )
    positions = []
    for batch_index, (ids, roles, anchors) in enumerate(individual):
        active = encoded_cpu["input_ids"][batch_index][
            encoded_cpu["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active] != ids:
            raise RuntimeError("Phase575 natural ledger tokenization drift")
        positions.append({
            "query_terminal": int(roles["query_terminal"][-1]),
            "answer_boundary": int(roles["answer_boundary"][-1]),
            "source_groups": {
                "semantic_selected": union_positions(roles, TARGET_ROLES),
                "semantic_other_relation": union_positions(roles, OTHER_ROLES),
                "anchor_base_selected": union_positions(anchors, ANCHOR_TARGET_ROLES),
                "anchor_base_other_relation": union_positions(anchors, ANCHOR_OTHER_ROLES),
            },
        })
    position_ids = encoded_cpu["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded_cpu["attention_mask"] == 0, 0)
    encoded_cpu["position_ids"] = position_ids
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}

    captures: dict[int, dict[str, dict[str, Any]]] = {}
    tensor_captures: dict[int, dict[str, dict[str, Any]]] = {}
    weight_error_max = 0.0

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            nonlocal weight_error_max
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if (
                hidden is None
                or position_embeddings is None
                or not isinstance(output, tuple)
                or output[1] is None
            ):
                raise RuntimeError("Phase575 natural ledger requires eager attention weights")
            primary, weights = output[0], output[1]
            query, key, value = projected_states(
                module, hidden, position_embeddings
            )
            raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            masked_scores = raw_scores
            if attention_mask is not None:
                masked_scores = masked_scores + attention_mask
            reconstructed = torch.softmax(masked_scores.float(), dim=-1).to(query.dtype)
            weight_error_max = max(
                weight_error_max,
                finite(float((reconstructed - weights).float().abs().max().item())),
            )

            layer_capture: dict[str, dict[str, Any]] = {}
            layer_tensors: dict[str, dict[str, Any]] = {}
            for batch_index, variant in enumerate(VARIANTS):
                variant_capture: dict[str, Any] = {}
                variant_tensors: dict[str, Any] = {}
                for receiver_name in ledger_protocol.RECEIVERS:
                    receiver = positions[batch_index][receiver_name]
                    sources = {}
                    source_tensors = {}
                    for source_name, source_positions in positions[batch_index][
                        "source_groups"
                    ].items():
                        scalars, tensors = source_metrics(
                            module, query, key, value, raw_scores, weights,
                            batch_index, receiver, source_positions, keep_snapshot,
                        )
                        sources[source_name] = scalars
                        source_tensors[source_name] = tensors
                    receiver_capture = {
                        "post_rotary_query_norm": finite(float(
                            query[batch_index, :, receiver, :].float().norm().item()
                        )),
                        "attention_output_norm": finite(float(
                            primary[batch_index, receiver].float().norm().item()
                        )),
                        "sources": sources,
                        **pair_margins(sources),
                    }
                    variant_capture[receiver_name] = receiver_capture
                    variant_tensors[receiver_name] = {
                        "post_rotary_query": query[
                            batch_index, :, receiver, :
                        ].detach().cpu(),
                        "attention_output": primary[
                            batch_index, receiver
                        ].detach().cpu(),
                    }
                    if keep_snapshot:
                        variant_tensors[receiver_name]["sources"] = source_tensors
                layer_capture[variant] = variant_capture
                layer_tensors[variant] = variant_tensors
            captures[layer_index] = layer_capture
            tensor_captures[layer_index] = layer_tensors
        return hook

    handles = [
        layer.self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index, layer in enumerate(layers)
    ]
    try:
        with torch.inference_mode():
            result = loaded.model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(range(len(layers))):
        raise RuntimeError("Phase575 natural ledger layer capture drift")
    if result.hidden_states is None or len(result.hidden_states) != len(layers) + 1:
        raise RuntimeError("Phase575 natural ledger hidden-state capture drift")

    prefix_length = common_prefix_length(individual[0][0], individual[1][0])
    if prefix_length <= 0:
        raise RuntimeError("Phase575 base/relation variants lack a common prefix")
    prefix_max = 0.0
    prefix_sum = 0.0
    prefix_count = 0
    for hidden in result.hidden_states:
        left = hidden[0, :prefix_length].float()
        right = hidden[1, :prefix_length].float()
        delta = finite(float(
            (left - right).norm().div(left.norm().clamp_min(1e-12)).item()
        ))
        prefix_max = max(prefix_max, delta)
        prefix_sum += delta
        prefix_count += 1

    output_rows = []
    snapshot = {"layers": tensor_captures} if keep_snapshot else None
    for layer_index in range(len(layers)):
        layer_capture = captures[layer_index]
        variants_out = {}
        for batch_index, variant in enumerate(VARIANTS):
            receivers_out = {}
            for receiver_name in ledger_protocol.RECEIVERS:
                receiver = positions[batch_index][receiver_name]
                receivers_out[receiver_name] = {
                    **layer_capture[variant][receiver_name],
                    "layer_input_state_norm": finite(float(
                        result.hidden_states[layer_index][
                            batch_index, receiver
                        ].float().norm().item()
                    )),
                }
                if keep_snapshot:
                    snapshot["layers"][layer_index][variant][receiver_name][
                        "layer_input_state"
                    ] = result.hidden_states[layer_index][
                        batch_index, receiver
                    ].detach().cpu()
            variants_out[variant] = receivers_out

        pair_vector_deltas = {}
        for axis, variant in (
            ("relation", "relation_swap"),
            ("object", "object_swap"),
            ("relation_object", "relation_object_swap"),
            ("order", "order_swap"),
        ):
            pair_vector_deltas[axis] = {}
            for receiver_name in ledger_protocol.RECEIVERS:
                base_tensors = tensor_captures.get(layer_index, {}).get(
                    "base", {}
                ).get(receiver_name)
                variant_tensors = tensor_captures.get(layer_index, {}).get(
                    variant, {}
                ).get(receiver_name)
                if base_tensors is not None and variant_tensors is not None:
                    pair_vector_deltas[axis][receiver_name] = {
                        "post_rotary_query_relative_delta": relative_delta(
                            variant_tensors["post_rotary_query"],
                            base_tensors["post_rotary_query"],
                        ),
                        "attention_output_relative_delta": relative_delta(
                            variant_tensors["attention_output"],
                            base_tensors["attention_output"],
                        ),
                        "layer_input_state_relative_delta": relative_delta(
                            result.hidden_states[layer_index][
                                VARIANTS.index(variant),
                                positions[VARIANTS.index(variant)][receiver_name],
                            ],
                            result.hidden_states[layer_index][
                                0, positions[0][receiver_name]
                            ],
                        ),
                    }
                else:
                    pair_vector_deltas[axis][receiver_name] = {}
        output_rows.append({
            "schema_version": "phase575_natural_ledger_row.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": loaded.key,
            "split": split,
            "base_case_id": base_id,
            "layer": layer_index,
            "normalized_depth": layer_index / max(1, len(layers) - 1),
            "variants": variants_out,
            "pair_vector_deltas_if_snapshotted": pair_vector_deltas,
            "post_rotary_query_and_key": True,
            "pre_softmax_score": True,
            "output_embedding_direction_used": False,
            "causal_intervention_executed": False,
            "sealed": False,
        })

    diagnostics = {
        "weight_reconstruction_max_abs_error": weight_error_max,
        "causal_prefix_max_relative_delta": prefix_max,
        "causal_prefix_mean_relative_delta": prefix_sum / max(1, prefix_count),
    }
    del result, encoded, encoded_cpu, captures, tensor_captures
    return output_rows, snapshot, diagnostics


def run(model: str, restart: bool) -> Path:
    frozen = read_json(ledger_protocol.LEDGER_PROTOCOL)
    if model not in frozen["authorized_models"]:
        raise RuntimeError(f"Phase575 natural ledger is not authorized for {model}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase575 natural ledger requires CUDA")
    worlds = load_worlds(model)
    prepare_contract(model, worlds, restart)
    output = paths(model)
    if output["summary"].exists() and output["rows"].exists() and not restart:
        return output["summary"]

    loaded = None
    rows_out = []
    snapshots = {}
    reconstruction_max = 0.0
    prefix_max = 0.0
    prefix_sum = 0.0
    prefix_count = 0
    duplicate_max = 0.0
    started = time.monotonic()
    discovery_snapshot_count = 0
    duplicate_count = 0
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase575 requires a CUDA model, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "right"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase575 natural ledger requires BF16, got {dtype}")
        layers = get_layers(loaded.model)
        expected_layers = frozen["layer_count_by_model"][model]
        if len(layers) != expected_layers:
            raise RuntimeError(
                f"Phase575 model layer drift: {model}/{len(layers)}/{expected_layers}"
            )
        loaded.model.config._attn_implementation = "eager"

        snapshot_cap = int(frozen[
            "full_vector_snapshot_worlds_in_discovery_per_model"
        ])
        duplicate_cap = int(frozen[
            "duplicate_trace_audit_worlds_in_discovery_per_model"
        ])
        for world_index, (split, base_id, rows) in enumerate(worlds):
            keep_snapshot = split == "structure_discovery" and discovery_snapshot_count < snapshot_cap
            world_rows, snapshot, diagnostics = trace_world(
                loaded, layers, split, base_id, rows, keep_snapshot
            )
            rows_out.extend(world_rows)
            reconstruction_max = max(
                reconstruction_max,
                diagnostics["weight_reconstruction_max_abs_error"],
            )
            prefix_max = max(
                prefix_max, diagnostics["causal_prefix_max_relative_delta"]
            )
            prefix_sum += diagnostics["causal_prefix_mean_relative_delta"]
            prefix_count += 1
            if snapshot is not None:
                snapshots[base_id] = snapshot
                discovery_snapshot_count += 1

            if split == "structure_discovery" and duplicate_count < duplicate_cap:
                duplicate_rows, _, duplicate_diagnostics = trace_world(
                    loaded, layers, split, base_id, rows, False
                )
                duplicate_max = max(
                    duplicate_max,
                    nested_numeric_max_delta(world_rows, duplicate_rows),
                    abs(
                        diagnostics["weight_reconstruction_max_abs_error"]
                        - duplicate_diagnostics["weight_reconstruction_max_abs_error"]
                    ),
                )
                duplicate_count += 1
            if (world_index + 1) % 16 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase575 natural-ledger "
                    f"{world_index + 1}/{len(worlds)}",
                    flush=True,
                )

        write_jsonl(output["rows"], rows_out)
        torch.save(snapshots, output["snapshots"])
        prefix_mean = prefix_sum / max(1, prefix_count)
        gates = {
            "attention_weight_reconstruction_pass": reconstruction_max
            <= frozen["attention_weight_reconstruction_max_abs_error"],
            "causal_prefix_pass": prefix_max
            <= frozen["causal_prefix_max_relative_delta"],
            "duplicate_trace_exact_pass": duplicate_max == 0.0,
        }
        summary = {
            "schema_version": "phase575_natural_ledger_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "world_count": len(worlds),
            "world_count_by_split": {
                split: sum(world_split == split for world_split, _, _ in worlds)
                for split in protocol.STRUCTURE_SPLITS
            },
            "layer_count": len(layers),
            "ledger_row_count": len(rows_out),
            "full_vector_snapshot_world_count": len(snapshots),
            "duplicate_trace_audit_world_count": duplicate_count,
            "attention_weight_reconstruction_max_abs_error": reconstruction_max,
            "causal_prefix_max_relative_delta": prefix_max,
            "causal_prefix_mean_relative_delta": prefix_mean,
            "duplicate_trace_max_abs_delta": duplicate_max,
            "quality_gates": gates,
            "natural_structure_analysis_authorized": all(gates.values()),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "snapshots_sha256": sha256_file(output["snapshots"]),
            "output_embedding_direction_used": False,
            "causal_intervention_executed": False,
            "causal_splits_read": False,
            "sealed_split_read": False,
            "head_channel_parameter_neuron_scan_executed": False,
        }
        write_json(output["summary"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return output["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.restart)


if __name__ == "__main__":
    main()
