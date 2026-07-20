#!/usr/bin/env python3
"""Trace query-condition messages and the downstream fact-routing endpoint."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
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
from phase573_coarse_message_causal import (  # noqa: E402
    edge_contribution,
    reconstructed_receiver,
)
from phase573_natural_transition_trace import physical_anchor_positions  # noqa: E402
import phase574_query_source_protocol as protocol  # noqa: E402
import phase574_query_source_trace_protocol as trace_protocol  # noqa: E402


MODEL = trace_protocol.MODEL
OUT_DIR = protocol.OUT_DIR
VARIANTS = ("base", "relation_swap", "object_swap", "order_swap")
VARIANT_INDEX = {variant: index for index, variant in enumerate(VARIANTS)}
TARGET_ROLES = (
    "target_fact_object", "target_fact_relation", "target_fact_value",
)
OTHER_ROLES = (
    "other_fact_object", "other_fact_relation", "other_fact_value",
)
ANCHOR_ROLES = (
    "anchor_target_fact_object", "anchor_target_fact_relation",
    "anchor_target_fact_value", "anchor_other_fact_object",
    "anchor_other_fact_relation", "anchor_other_fact_value",
)
ROWS_PATH = OUT_DIR / "phase574_qwen3_query_source_trace_rows.jsonl.gz"
SNAPSHOTS_PATH = OUT_DIR / "phase574_qwen3_discovery_vector_snapshots.pt"
SUMMARY_PATH = OUT_DIR / "phase574_qwen3_query_source_trace_summary.json"
DECISION_PATH = OUT_DIR / "phase574_query_source_trace_decision.json"
CONTRACT_PATH = OUT_DIR / "phase574_qwen3_query_source_trace_contract.json"


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
    return finite(float((left_f - right_f).norm().div(scale.clamp_min(1e-12)).item()))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.float().reshape(-1)
    right_f = right.float().reshape(-1)
    denom = left_f.norm() * right_f.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return finite(float(torch.dot(left_f, right_f).div(denom).item()))


def load_worlds() -> list[tuple[str, str, list[dict[str, Any]]]]:
    frozen = read_json(trace_protocol.TRACE_PROTOCOL)
    selected = frozen["selected_base_case_ids_by_split"]
    selected_ids = set().union(*(set(ids) for ids in selected.values()))
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(protocol.OPEN_CASES_PATH):
        if row["base_case_id"] not in selected_ids:
            continue
        if row["split"] not in protocol.STRUCTURE_SPLITS or row["sealed"]:
            raise RuntimeError("Phase574 trace attempted forbidden split access")
        bank[row["base_case_id"]][row["variant"]] = row
    worlds = []
    for split in protocol.STRUCTURE_SPLITS:
        for base_id in selected[split]:
            variants = bank.get(base_id, {})
            if set(variants) != set(VARIANTS):
                raise RuntimeError(f"Phase574 incomplete trace world: {base_id}")
            rows = [variants[variant] for variant in VARIANTS]
            if any(row["split"] != split for row in rows):
                raise RuntimeError("Phase574 trace split identity drift")
            worlds.append((split, base_id, rows))
    if len(worlds) != 384:
        raise RuntimeError(f"Phase574 trace world count drift: {len(worlds)}")
    return worlds


def prepare_contract(
    worlds: list[tuple[str, str, list[dict[str, Any]]]], restart: bool
) -> None:
    if restart:
        for path in (ROWS_PATH, SNAPSHOTS_PATH, SUMMARY_PATH, DECISION_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    payload = {
        "schema_version": "phase574_query_source_trace_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "trace_protocol_sha256": sha256_file(trace_protocol.TRACE_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "world_count": len(worlds),
        "world_ids": [base_id for _, base_id, _ in worlds],
        "variants": list(VARIANTS),
        "trace_layers": list(trace_protocol.LAYERS),
        "components": list(trace_protocol.COMPONENTS),
        "qkv_projection_stage": "pre_rotary_projection",
        "output_embedding_direction_used": False,
        "causal_intervention_executed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
    }
    if CONTRACT_PATH.exists():
        existing = read_json(CONTRACT_PATH)
        for key, value in payload.items():
            if key != "created_at" and existing[key] != value:
                raise RuntimeError(f"Phase574 trace contract drift: {key}")
    else:
        write_json(CONTRACT_PATH, payload)


def role_mean(value: torch.Tensor, positions: list[int]) -> torch.Tensor:
    return value[positions].float().mean(dim=0)


def run(restart: bool) -> Path:
    worlds = load_worlds()
    prepare_contract(worlds, restart)
    if SUMMARY_PATH.exists() and DECISION_PATH.exists() and not restart:
        return SUMMARY_PATH

    loaded = None
    rows_out: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    prefix_max = 0.0
    prefix_sum = 0.0
    prefix_count = 0
    reconstruction_max = 0.0
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "right"
        layers = get_layers(loaded.model)
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError(f"Phase574 trace model drift: {dtype}/{len(layers)}")
        loaded.model.config._attn_implementation = "eager"

        for world_index, (split, base_id, rows) in enumerate(worlds):
            prompts = [render_chat(loaded.tokenizer, MODEL, row["raw_prompt"]) for row in rows]
            individual = []
            for prompt, row in zip(prompts, rows):
                ids, roles = role_positions(loaded.tokenizer, prompt, row)
                anchors = physical_anchor_positions(loaded.tokenizer, prompt, row)
                individual.append((ids, roles, anchors))
            encoded_cpu = loaded.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False
            )
            positions: list[dict[str, Any]] = []
            anchors_by_variant: list[dict[str, list[int]]] = []
            for batch_index, (ids, roles, anchors) in enumerate(individual):
                active = encoded_cpu["input_ids"][batch_index][
                    encoded_cpu["attention_mask"][batch_index].bool()
                ].tolist()
                if [int(value) for value in active] != ids:
                    raise RuntimeError("Phase574 trace tokenization drift")
                positions.append({
                    **roles,
                    "selected_source": sorted(
                        pos for role in TARGET_ROLES for pos in roles[role]
                    ),
                    "other_source": sorted(
                        pos for role in OTHER_ROLES for pos in roles[role]
                    ),
                })
                anchors_by_variant.append(anchors)
            position_ids = encoded_cpu["attention_mask"].long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(encoded_cpu["attention_mask"] == 0, 0)
            encoded_cpu["position_ids"] = position_ids
            encoded = {
                key: value.to(loaded.input_device) for key, value in encoded_cpu.items()
            }

            captures: dict[int, dict[str, dict[str, Any]]] = {}
            errors: list[float] = []

            def hook_for(layer_index: int):
                def hook(
                    module: Any,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    output: Any,
                ) -> None:
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    if hidden is None or not isinstance(output, tuple) or output[1] is None:
                        raise RuntimeError("Phase574 trace requires eager attention weights")
                    primary, weights = output[0], output[1]
                    batch, sequence, _ = hidden.shape
                    q_projected = module.q_proj(hidden)
                    k_projected = module.k_proj(hidden)
                    v_projected = module.v_proj(hidden)
                    values = v_projected.view(
                        batch, sequence, -1, module.head_dim
                    ).transpose(1, 2)
                    values = values.repeat_interleave(
                        module.num_key_value_groups, dim=1
                    )
                    layer_capture: dict[str, dict[str, Any]] = {}
                    for batch_index, variant in enumerate(VARIANTS):
                        pos = positions[batch_index]
                        receiver = int(pos["query_terminal"][-1])
                        relation_positions = pos["query_relation"]
                        object_positions = pos["query_object"]
                        relation_message = edge_contribution(
                            module, weights, values, batch_index, receiver,
                            relation_positions,
                        )
                        object_message = edge_contribution(
                            module, weights, values, batch_index, receiver,
                            object_positions,
                        )
                        reconstructed = reconstructed_receiver(
                            module, weights, values, batch_index, receiver
                        )
                        actual = primary[batch_index, receiver]
                        errors.append(float(
                            (reconstructed.float() - actual.float()).norm().item()
                            / max(actual.float().norm().item(), 1e-8)
                        ))
                        item: dict[str, Any] = {
                            "query_relation_value_message": relation_message.detach(),
                            "query_object_value_message": object_message.detach(),
                            "query_terminal_attention_output": actual.detach(),
                            "query_projection": q_projected[
                                batch_index, receiver
                            ].detach(),
                            "relation_key_projection": k_projected[
                                batch_index, relation_positions
                            ].float().mean(dim=0).detach(),
                            "relation_value_projection": v_projected[
                                batch_index, relation_positions
                            ].float().mean(dim=0).detach(),
                            "object_key_projection": k_projected[
                                batch_index, object_positions
                            ].float().mean(dim=0).detach(),
                            "object_value_projection": v_projected[
                                batch_index, object_positions
                            ].float().mean(dim=0).detach(),
                            "query_relation_attention_mass": finite(float(
                                weights[
                                    batch_index, :, receiver, relation_positions
                                ].float().sum(dim=-1).mean().item()
                            )),
                            "query_object_attention_mass": finite(float(
                                weights[
                                    batch_index, :, receiver, object_positions
                                ].float().sum(dim=-1).mean().item()
                            )),
                        }
                        if layer_index == 24:
                            selected_positions = pos["selected_source"]
                            other_positions = pos["other_source"]
                            item.update({
                                "selected_fact_attention_mass": finite(float(
                                    weights[
                                        batch_index, :, int(pos["answer_boundary"][-1]),
                                        selected_positions,
                                    ].float().sum(dim=-1).mean().item()
                                )),
                                "other_fact_attention_mass": finite(float(
                                    weights[
                                        batch_index, :, int(pos["answer_boundary"][-1]),
                                        other_positions,
                                    ].float().sum(dim=-1).mean().item()
                                )),
                                "selected_fact_value_message": edge_contribution(
                                    module, weights, values, batch_index,
                                    int(pos["answer_boundary"][-1]), selected_positions,
                                ).detach(),
                                "other_fact_value_message": edge_contribution(
                                    module, weights, values, batch_index,
                                    int(pos["answer_boundary"][-1]), other_positions,
                                ).detach(),
                            })
                        layer_capture[variant] = item
                    captures[layer_index] = layer_capture
                return hook

            handles = [
                layers[layer_index].self_attn.register_forward_hook(
                    hook_for(layer_index), with_kwargs=True
                )
                for layer_index in trace_protocol.LAYERS
            ]
            with torch.inference_mode():
                result = loaded.model(
                    **encoded,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            for handle in handles:
                handle.remove()
            if set(captures) != set(trace_protocol.LAYERS):
                raise RuntimeError("Phase574 full trace layer capture drift")
            reconstruction_max = max(reconstruction_max, max(errors))

            for layer_index in trace_protocol.LAYERS:
                capture = captures[layer_index]
                pair_metrics = {}
                for axis, variant in (
                    ("relation", "relation_swap"),
                    ("object", "object_swap"),
                    ("order", "order_swap"),
                ):
                    pair_metrics[axis] = {
                        component: {
                            "relative_delta": relative_delta(
                                capture[variant][component], capture["base"][component]
                            ),
                            "cosine": cosine(
                                capture[variant][component], capture["base"][component]
                            ),
                        }
                        for component in trace_protocol.COMPONENTS
                    }
                relation_projection_metrics = {
                    name: relative_delta(
                        capture["relation_swap"][name], capture["base"][name]
                    )
                    for name in (
                        "query_projection", "relation_key_projection",
                        "relation_value_projection", "object_key_projection",
                        "object_value_projection",
                    )
                }
                variants_payload = {}
                for variant in VARIANTS:
                    item = capture[variant]
                    variants_payload[variant] = {
                        "query_relation_attention_mass": item[
                            "query_relation_attention_mass"
                        ],
                        "query_object_attention_mass": item[
                            "query_object_attention_mass"
                        ],
                        "query_relation_message_norm": finite(float(
                            item["query_relation_value_message"].float().norm().item()
                        )),
                        "query_terminal_attention_output_norm": finite(float(
                            item["query_terminal_attention_output"].float().norm().item()
                        )),
                    }
                    if layer_index == 24:
                        variants_payload[variant].update({
                            "selected_fact_attention_mass": item[
                                "selected_fact_attention_mass"
                            ],
                            "other_fact_attention_mass": item[
                                "other_fact_attention_mass"
                            ],
                            "semantic_selection_margin": item[
                                "selected_fact_attention_mass"
                            ] - item["other_fact_attention_mass"],
                            "selected_fact_message_norm": finite(float(
                                item["selected_fact_value_message"].float().norm().item()
                            )),
                            "other_fact_message_norm": finite(float(
                                item["other_fact_value_message"].float().norm().item()
                            )),
                        })
                rows_out.append({
                    "schema_version": "phase574_query_source_trace_row.v1",
                    "phase_id": protocol.PHASE,
                    "created_at": now(),
                    "model": MODEL,
                    "split": split,
                    "base_case_id": base_id,
                    "layer": layer_index,
                    "pair_metrics": pair_metrics,
                    "relation_pre_rotary_projection_relative_delta": (
                        relation_projection_metrics
                    ),
                    "variants": variants_payload,
                    "observer_only": True,
                    "causal": False,
                    "output_embedding_direction_used": False,
                    "head_channel_parameter_neuron_scan_executed": False,
                    "sealed": False,
                })

            for layer_offset, hidden in enumerate(result.hidden_states[1:]):
                for changed_index in (
                    VARIANT_INDEX["relation_swap"], VARIANT_INDEX["object_swap"]
                ):
                    for role in ANCHOR_ROLES:
                        base_vector = role_mean(
                            hidden[VARIANT_INDEX["base"]],
                            anchors_by_variant[VARIANT_INDEX["base"]][role],
                        )
                        changed_vector = role_mean(
                            hidden[changed_index],
                            anchors_by_variant[changed_index][role],
                        )
                        value = relative_delta(changed_vector, base_vector)
                        prefix_max = max(prefix_max, value)
                        prefix_sum += value
                        prefix_count += 1

            if (
                split == "structure_discovery"
                and len(snapshots) < read_json(trace_protocol.TRACE_PROTOCOL)[
                    "open_discovery_full_vector_world_cap"
                ]
            ):
                snapshots[base_id] = {
                    "split": split,
                    "layers": {
                        layer_index: {
                            variant: {
                                key: value.detach().cpu().to(torch.float16)
                                for key, value in captures[layer_index][variant].items()
                                if torch.is_tensor(value)
                            }
                            for variant in VARIANTS
                        }
                        for layer_index in trace_protocol.LAYERS
                    },
                }

            del result, encoded, encoded_cpu, captures
            if (world_index + 1) % 16 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase574 trace "
                    f"{world_index + 1}/{len(worlds)}",
                    flush=True,
                )

        write_jsonl(ROWS_PATH, rows_out)
        torch.save({
            "schema_version": "phase574_discovery_vector_snapshots.v1",
            "phase_id": protocol.PHASE,
            "model": MODEL,
            "qkv_projection_stage": "pre_rotary_projection",
            "worlds": snapshots,
        }, SNAPSHOTS_PATH)

        gates = read_json(trace_protocol.TRACE_PROTOCOL)["natural_trace_gate"]
        metrics_by_split: dict[str, Any] = {}
        for split in protocol.STRUCTURE_SPLITS:
            split_rows = [row for row in rows_out if row["split"] == split]
            layer5 = [row for row in split_rows if row["layer"] == 5]
            layer24 = [row for row in split_rows if row["layer"] == 24]
            relation_values = [
                row["pair_metrics"]["relation"][
                    "query_terminal_attention_output"
                ]["relative_delta"]
                for row in layer5
            ]
            relation_event_rate = sum(
                value >= gates["full_attention_relation_relative_delta_minimum"]
                for value in relation_values
            ) / max(1, len(relation_values))
            relation_route = sum(
                row["variants"]["base"]["semantic_selection_margin"] > 0
                and row["variants"]["relation_swap"]["semantic_selection_margin"] > 0
                for row in layer24
            ) / max(1, len(layer24))
            object_route = sum(
                row["variants"]["base"]["semantic_selection_margin"] > 0
                and row["variants"]["object_swap"]["semantic_selection_margin"] > 0
                for row in layer24
            ) / max(1, len(layer24))
            order_route = sum(
                row["variants"]["base"]["semantic_selection_margin"] > 0
                and row["variants"]["order_swap"]["semantic_selection_margin"] > 0
                for row in layer24
            ) / max(1, len(layer24))
            metrics_by_split[split] = {
                "world_count": len(layer5),
                "layer5_relation_full_attention_mean_relative_delta": finite(
                    sum(relation_values) / max(1, len(relation_values))
                ),
                "layer5_relation_full_attention_event_rate": relation_event_rate,
                "layer24_relation_semantic_selection_pair_rate": relation_route,
                "layer24_object_semantic_selection_pair_rate": object_route,
                "layer24_order_semantic_selection_pair_rate": order_route,
            }
        prefix_pass = prefix_max <= gates["causal_prefix_maximum_relative_delta"]
        split_pass = {
            split: (
                values["layer5_relation_full_attention_event_rate"]
                >= gates["full_attention_relation_world_rate_minimum_each_split"]
                and values["layer24_relation_semantic_selection_pair_rate"]
                >= gates["layer24_relation_selection_pair_rate_minimum_each_split"]
                and values["layer24_object_semantic_selection_pair_rate"]
                >= gates["layer24_object_selection_pair_rate_minimum_each_split"]
                and values["layer24_order_semantic_selection_pair_rate"]
                >= gates["layer24_order_preservation_pair_rate_minimum_each_split"]
            )
            for split, values in metrics_by_split.items()
        }
        authorized = prefix_pass and all(split_pass.values())
        summary = {
            "schema_version": "phase574_query_source_trace_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": dtype,
            "world_count": len(worlds),
            "trace_row_count": len(rows_out),
            "full_vector_snapshot_world_count": len(snapshots),
            "metrics_by_split": metrics_by_split,
            "split_gate_pass": split_pass,
            "causal_prefix_maximum_relative_delta": prefix_max,
            "causal_prefix_mean_relative_delta": prefix_sum / max(1, prefix_count),
            "causal_prefix_audit_pass": prefix_pass,
            "maximum_attention_reconstruction_relative_error": reconstruction_max,
            "coarse_query_source_causal_authorized": authorized,
            "rows_sha256": sha256_file(ROWS_PATH),
            "snapshots_sha256": sha256_file(SNAPSHOTS_PATH),
            "runtime_seconds": time.monotonic() - started,
            "causal_intervention_executed": False,
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        decision = {
            "schema_version": "phase574_query_source_trace_decision.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "layer5_query_condition_event_replicated": all(
                values["layer5_relation_full_attention_event_rate"]
                >= gates["full_attention_relation_world_rate_minimum_each_split"]
                for values in metrics_by_split.values()
            ),
            "layer24_source_routing_replicated": all(
                split_pass.values()
            ) and prefix_pass,
            "coarse_query_source_causal_authorized": authorized,
            "reason": (
                "Fresh structure splits replicated both endpoints and the causal-prefix audit."
                if authorized else
                "At least one fresh natural endpoint or causal-prefix gate failed."
            ),
            "head_channel_parameter_neuron_scan_authorized": False,
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(DECISION_PATH, decision)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
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
