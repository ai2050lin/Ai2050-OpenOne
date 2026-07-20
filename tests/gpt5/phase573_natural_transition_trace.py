#!/usr/bin/env python3
"""Trace full-vector natural transitions and legal fact-to-receiver routing."""

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
from phase569_role_position_utils import (  # noqa: E402
    role_positions,
    span_in_parent,
    token_indices_for_span,
)
import phase573_natural_transition_protocol as protocol  # noqa: E402
import phase573_natural_transition_trace_protocol as trace_protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
MODEL = trace_protocol.MODEL
VARIANTS = ("base", "relation_swap", "object_swap", "order_swap")
VARIANT_INDEX = {name: index for index, name in enumerate(VARIANTS)}
ALL_CAPTURE_ROLES = (*trace_protocol.SEMANTIC_ROLES, *trace_protocol.PHYSICAL_PREFIX_ROLES)
SOURCE_TARGET_ROLES = (
    "target_fact_object", "target_fact_relation", "target_fact_value",
)
SOURCE_OTHER_ROLES = (
    "other_fact_object", "other_fact_relation", "other_fact_value",
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


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def raw_rows_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_natural_trace_rows.jsonl.gz"


def routing_rows_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_natural_routing_rows.jsonl.gz"


def summary_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_natural_trace_summary.json"


def decision_path() -> Path:
    return OUT_DIR / "phase573_natural_trace_decision.json"


def contract_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_natural_trace_contract.json"


def physical_anchor_positions(
    tokenizer: Any, prompt: str, row: dict[str, Any]
) -> dict[str, list[int]]:
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    fragments = row["physical_anchor_fragments"]
    target_fact = fragments["target_fact"]
    other_fact = fragments["other_fact"]
    fields = {
        "anchor_target_fact_object": (target_fact, fragments["target_fact_object"], False),
        "anchor_target_fact_relation": (target_fact, fragments["target_fact_relation"], False),
        "anchor_target_fact_value": (target_fact, fragments["target_fact_value"], True),
        "anchor_other_fact_object": (other_fact, fragments["other_fact_object"], False),
        "anchor_other_fact_relation": (other_fact, fragments["other_fact_relation"], False),
        "anchor_other_fact_value": (other_fact, fragments["other_fact_value"], True),
    }
    return {
        role: token_indices_for_span(
            offsets, *span_in_parent(prompt, parent, child, last_child=last_child)
        )
        for role, (parent, child, last_child) in fields.items()
    }


def load_worlds() -> list[tuple[str, str, list[dict[str, Any]]]]:
    frozen = read_json(trace_protocol.TRACE_PROTOCOL)
    if frozen["sealed_split_read"] or frozen["causal_splits_read"]:
        raise RuntimeError("Phase573 natural trace contract has forbidden split access")
    selected = frozen["selected_base_case_ids_by_split"]
    selected_ids = set().union(*(set(ids) for ids in selected.values()))
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(protocol.OPEN_CASES_PATH):
        if row["base_case_id"] not in selected_ids:
            continue
        if row["split"] not in trace_protocol.SPLITS or row["sealed"]:
            raise RuntimeError("Phase573 selected trace world escaped structure splits")
        bank[row["base_case_id"]][row["variant"]] = row
    worlds = []
    for split in trace_protocol.SPLITS:
        for base_id in selected[split]:
            variants = bank.get(base_id, {})
            if set(variants) != set(VARIANTS):
                raise RuntimeError(f"Phase573 incomplete trace world: {base_id}")
            ordered = [variants[name] for name in VARIANTS]
            if any(row["split"] != split for row in ordered):
                raise RuntimeError("Phase573 trace split identity drift")
            worlds.append((split, base_id, ordered))
    if len(worlds) != 384:
        raise RuntimeError(f"Phase573 natural trace world drift: {len(worlds)}")
    return worlds


def prepare_contract(worlds: list[tuple[str, str, list[dict[str, Any]]]], restart: bool) -> None:
    paths = (raw_rows_path(), routing_rows_path(), summary_path(), decision_path(), contract_path())
    if restart:
        for path in paths:
            path.unlink(missing_ok=True)
    payload = {
        "schema_version": "phase573_natural_trace_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "trace_protocol_sha256": sha256_file(trace_protocol.TRACE_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "world_count": len(worlds),
        "world_ids": [base_id for _, base_id, _ in worlds],
        "variants": list(VARIANTS),
        "components": list(trace_protocol.COMPONENTS),
        "capture_roles": list(ALL_CAPTURE_ROLES),
        "output_embedding_direction_used": False,
        "full_vectors_persisted": False,
        "causal_intervention_executed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
    }
    if contract_path().exists():
        existing = read_json(contract_path())
        for key in (
            "model", "trace_protocol_sha256", "open_cases_sha256", "world_count",
            "world_ids", "variants", "components", "capture_roles",
            "output_embedding_direction_used", "full_vectors_persisted",
            "causal_intervention_executed", "causal_splits_read", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase573 natural trace contract drift: {key}")
    else:
        write_json(contract_path(), payload)


def mean_role_vector(selected: torch.Tensor, positions: list[int]) -> torch.Tensor:
    return selected[positions].float().mean(dim=0)


def run(restart: bool) -> Path:
    worlds = load_worlds()
    prepare_contract(worlds, restart)
    if summary_path().exists() and decision_path().exists() and not restart:
        return summary_path()

    loaded = None
    handles: list[Any] = []
    current_indices: dict[str, list[list[int]]] = {}
    current_split = ""
    current_base_id = ""
    pair_arrays: dict[tuple[str, str], dict[str, list[float]]] = {}
    routing_rows: list[dict[str, Any]] = []
    component_cache: dict[tuple[int, str], torch.Tensor] = {}
    ledger_max = 0.0
    ledger_sum = 0.0
    ledger_count = 0
    prefix_max = 0.0
    prefix_sum = 0.0
    prefix_count = 0
    started = time.monotonic()
    attention_implementation_before = None
    attention_implementation_trace = None
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "right"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase573 natural trace requires BF16, got {run_dtype}")
        attention_implementation_before = getattr(
            loaded.model.config, "_attn_implementation", None
        )
        loaded.model.config._attn_implementation = "eager"
        attention_implementation_trace = getattr(
            loaded.model.config, "_attn_implementation", None
        )

        def selected_vectors(value: torch.Tensor) -> torch.Tensor:
            rows = []
            for batch_index in range(value.shape[0]):
                role_vectors = []
                for role in ALL_CAPTURE_ROLES:
                    role_vectors.append(
                        mean_role_vector(value[batch_index], current_indices[role][batch_index])
                    )
                rows.append(torch.stack(role_vectors))
            return torch.stack(rows)

        def record(layer_index: int, component: str, value: torch.Tensor) -> None:
            nonlocal ledger_max, ledger_sum, ledger_count
            nonlocal prefix_max, prefix_sum, prefix_count
            selected = selected_vectors(value)
            for axis in trace_protocol.AXES:
                variant = trace_protocol.VARIANT_BY_AXIS[axis]
                base = selected[VARIANT_INDEX["base"]]
                changed = selected[VARIANT_INDEX[variant]]
                delta = (changed - base).norm(dim=-1)
                scale = 0.5 * (changed.norm(dim=-1) + base.norm(dim=-1))
                relative = delta / scale.clamp_min(1e-12)
                for role_index, role in enumerate(ALL_CAPTURE_ROLES):
                    key = (axis, role)
                    entry = pair_arrays.setdefault(
                        key,
                        {component_name: [] for component_name in trace_protocol.COMPONENTS},
                    )
                    entry[component].append(finite(float(relative[role_index].item())))
                if axis in ("relation", "object"):
                    for role in trace_protocol.PHYSICAL_PREFIX_ROLES:
                        role_index = ALL_CAPTURE_ROLES.index(role)
                        value_relative = finite(float(relative[role_index].item()))
                        prefix_max = max(prefix_max, value_relative)
                        prefix_sum += value_relative
                        prefix_count += 1
            component_cache[(layer_index, component)] = selected
            if component == "layer_output":
                layer_input = component_cache[(layer_index, "layer_input")]
                attention = component_cache[(layer_index, "attention_output")]
                mlp = component_cache[(layer_index, "mlp_output")]
                residual = selected - layer_input - attention - mlp
                relative_error = residual.norm(dim=-1) / selected.norm(dim=-1).clamp_min(1e-12)
                ledger_max = max(ledger_max, float(relative_error.max().item()))
                ledger_sum += float(relative_error.sum().item())
                ledger_count += int(relative_error.numel())
                for name in trace_protocol.COMPONENTS:
                    component_cache.pop((layer_index, name), None)

        def make_pre_hook(layer_index: int):
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                record(layer_index, "layer_input", inputs[0])
            return hook

        def make_hook(layer_index: int, component: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                record(layer_index, component, tensor_from_output(output))
            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_index)))
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_hook(layer_index, "attention_output")
                )
            )
            handles.append(layer.mlp.register_forward_hook(make_hook(layer_index, "mlp_output")))
            handles.append(layer.register_forward_hook(make_hook(layer_index, "layer_output")))

        trace_rows: list[dict[str, Any]] = []
        for world_index, (split, base_id, rows) in enumerate(worlds):
            current_split = split
            current_base_id = base_id
            prompts = [render_chat(loaded.tokenizer, MODEL, row["raw_prompt"]) for row in rows]
            individual = []
            for prompt, row in zip(prompts, rows):
                ids, semantic = role_positions(loaded.tokenizer, prompt, row)
                anchors = physical_anchor_positions(loaded.tokenizer, prompt, row)
                individual.append((ids, {**semantic, **anchors}))
            encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
            sequence_length = int(encoded["input_ids"].shape[1])
            current_indices = {role: [] for role in ALL_CAPTURE_ROLES}
            for batch_index, (ids, groups) in enumerate(individual):
                active_ids = encoded["input_ids"][batch_index][
                    encoded["attention_mask"][batch_index].bool()
                ].tolist()
                if [int(value) for value in active_ids] != ids:
                    raise RuntimeError("Phase573 trace individual/batch tokenization drift")
                offset = 0
                for role in ALL_CAPTURE_ROLES:
                    current_indices[role].append([offset + pos for pos in groups[role]])
            pair_arrays = {}
            position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
            encoded["position_ids"] = position_ids
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(
                    **encoded, use_cache=False, output_attentions=True,
                    return_dict=True,
                )
            if component_cache:
                raise RuntimeError(f"Phase573 component ledger did not close: {component_cache}")
            attentions = result.attentions
            if attentions is None or len(attentions) != len(layers):
                raise RuntimeError("Phase573 eager trace did not return full-layer attentions")
            for axis in trace_protocol.AXES:
                for role in ALL_CAPTURE_ROLES:
                    arrays = pair_arrays[(axis, role)]
                    if any(len(arrays[name]) != len(layers) for name in trace_protocol.COMPONENTS):
                        raise RuntimeError("Phase573 full-layer trace array drift")
                    trace_rows.append({
                        "schema_version": "phase573_natural_trace_row.v1",
                        "phase_id": protocol.PHASE,
                        "created_at": now(),
                        "model": MODEL,
                        "split": split,
                        "base_case_id": base_id,
                        "axis": axis,
                        "variant": trace_protocol.VARIANT_BY_AXIS[axis],
                        "semantic_role": role,
                        "layer_count": len(layers),
                        "relative_delta_by_component": arrays,
                        "output_embedding_direction_used": False,
                        "full_vector_persisted": False,
                        "observer_only": True,
                        "causal": False,
                        "sealed": False,
                    })
            for receiver in trace_protocol.ROUTING_RECEIVERS:
                receiver_index = trace_protocol.SEMANTIC_ROLES.index(receiver)
                masses: dict[str, dict[str, list[float]]] = {
                    variant: {"selected": [], "other": []} for variant in VARIANTS
                }
                for layer_attention in attentions:
                    if layer_attention is None or layer_attention.ndim != 4:
                        raise RuntimeError("Phase573 attention tensor shape drift")
                    for variant in VARIANTS:
                        batch_index = VARIANT_INDEX[variant]
                        receiver_positions = current_indices[receiver][batch_index]
                        target_positions = sorted(
                            pos
                            for role in SOURCE_TARGET_ROLES
                            for pos in current_indices[role][batch_index]
                        )
                        other_positions = sorted(
                            pos
                            for role in SOURCE_OTHER_ROLES
                            for pos in current_indices[role][batch_index]
                        )
                        matrix = layer_attention[batch_index].float()
                        selected_mass = matrix[:, receiver_positions][:, :, target_positions].sum(-1).mean()
                        other_mass = matrix[:, receiver_positions][:, :, other_positions].sum(-1).mean()
                        masses[variant]["selected"].append(finite(float(selected_mass.item())))
                        masses[variant]["other"].append(finite(float(other_mass.item())))
                routing_rows.append({
                    "schema_version": "phase573_natural_routing_row.v1",
                    "phase_id": protocol.PHASE,
                    "created_at": now(),
                    "model": MODEL,
                    "split": split,
                    "base_case_id": base_id,
                    "receiver_role": receiver,
                    "layer_count": len(layers),
                    "attention_mass_by_variant": masses,
                    "semantic_source_alignment": True,
                    "routing_weight_only": True,
                    "source_specific_value_message_measured": False,
                    "observer_only": True,
                    "causal": False,
                    "sealed": False,
                })
            del result, attentions, encoded, prompts, individual
            current_indices = {}
            done = world_index + 1
            if world_index == 0 or done == len(worlds) or done % 16 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase573 natural trace "
                    f"{done}/{len(worlds)}",
                    flush=True,
                )

        write_jsonl(raw_rows_path(), trace_rows)
        write_jsonl(routing_rows_path(), routing_rows)

        state_gate = read_json(trace_protocol.TRACE_PROTOCOL)["state_event_gate"]
        state_candidates = []
        for component in trace_protocol.COMPONENTS:
            for role in state_gate["eligible_receivers"]:
                for layer_index in range(len(layers)):
                    split_rates = {}
                    split_means = {}
                    for split in trace_protocol.SPLITS:
                        values = [
                            row["relative_delta_by_component"][component][layer_index]
                            for row in trace_rows
                            if row["split"] == split
                            and row["axis"] == "relation"
                            and row["semantic_role"] == role
                        ]
                        if len(values) != trace_protocol.WORLDS_PER_SPLIT:
                            raise RuntimeError("Phase573 state-event denominator drift")
                        split_rates[split] = sum(
                            value >= state_gate["minimum_relative_delta"] for value in values
                        ) / len(values)
                        split_means[split] = finite(sum(values) / len(values))
                    if all(
                        rate >= state_gate["minimum_world_rate_each_split"]
                        for rate in split_rates.values()
                    ):
                        state_candidates.append({
                            "component": component,
                            "receiver_role": role,
                            "layer": layer_index,
                            "relative_depth": layer_index / max(1, len(layers) - 1),
                            "world_rate_by_split": split_rates,
                            "mean_relative_delta_by_split": split_means,
                        })
        state_candidates.sort(
            key=lambda row: (row["layer"], row["receiver_role"], row["component"])
        )

        route_gate = read_json(trace_protocol.TRACE_PROTOCOL)["routing_event_gate"]
        route_candidates = []
        for receiver in trace_protocol.ROUTING_RECEIVERS:
            for layer_index in range(len(layers)):
                metrics_by_split = {}
                passes = True
                for split in trace_protocol.SPLITS:
                    rows_for_split = [
                        row for row in routing_rows
                        if row["split"] == split and row["receiver_role"] == receiver
                    ]
                    if len(rows_for_split) != trace_protocol.WORLDS_PER_SPLIT:
                        raise RuntimeError("Phase573 routing denominator drift")
                    margins = {
                        variant: [
                            row["attention_mass_by_variant"][variant]["selected"][layer_index]
                            - row["attention_mass_by_variant"][variant]["other"][layer_index]
                            for row in rows_for_split
                        ]
                        for variant in VARIANTS
                    }
                    relation_pair_rate = sum(
                        base > 0.0 and changed > 0.0
                        for base, changed in zip(margins["base"], margins["relation_swap"])
                    ) / len(rows_for_split)
                    object_pair_rate = sum(
                        base > 0.0 and changed > 0.0
                        for base, changed in zip(margins["base"], margins["object_swap"])
                    ) / len(rows_for_split)
                    order_pair_rate = sum(
                        base > 0.0 and changed > 0.0
                        for base, changed in zip(margins["base"], margins["order_swap"])
                    ) / len(rows_for_split)
                    relation_mean = finite(
                        sum(
                            0.5 * (base + changed)
                            for base, changed in zip(
                                margins["base"], margins["relation_swap"]
                            )
                        ) / len(rows_for_split)
                    )
                    metrics_by_split[split] = {
                        "relation_semantic_selection_pair_rate": relation_pair_rate,
                        "relation_mean_semantic_selection_margin": relation_mean,
                        "object_semantic_selection_pair_rate": object_pair_rate,
                        "order_preservation_pair_rate": order_pair_rate,
                    }
                    passes = passes and (
                        relation_pair_rate
                        >= route_gate["minimum_semantic_selection_pair_rate"]
                        and relation_mean
                        > route_gate["minimum_mean_semantic_selection_margin"]
                        and object_pair_rate
                        >= route_gate["minimum_object_selection_pair_rate"]
                        and order_pair_rate
                        >= route_gate["minimum_order_preservation_pair_rate"]
                    )
                if passes:
                    route_candidates.append({
                        "receiver_role": receiver,
                        "layer": layer_index,
                        "relative_depth": layer_index / max(1, len(layers) - 1),
                        "metrics_by_split": metrics_by_split,
                    })
        route_candidates.sort(key=lambda row: (row["layer"], row["receiver_role"]))

        prefix_limit = read_json(trace_protocol.TRACE_PROTOCOL)["causal_mask_audit"][
            "maximum_allowed_relative_prefix_delta"
        ]
        prefix_audit_pass = prefix_max <= prefix_limit
        route_pass = bool(route_candidates and prefix_audit_pass)
        summary = {
            "schema_version": "phase573_natural_trace_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "attention_implementation_before": attention_implementation_before,
            "attention_implementation_trace": attention_implementation_trace,
            "world_count": len(worlds),
            "world_count_per_split": trace_protocol.WORLDS_PER_SPLIT,
            "layer_count": len(layers),
            "trace_row_count": len(trace_rows),
            "routing_row_count": len(routing_rows),
            "state_event_candidate_count": len(state_candidates),
            "earliest_state_event": state_candidates[0] if state_candidates else None,
            "routing_event_candidate_count": len(route_candidates),
            "earliest_routing_event": route_candidates[0] if route_candidates else None,
            "causal_mask_prefix_audit_pass": prefix_audit_pass,
            "maximum_fixed_prefix_relative_delta": finite(prefix_max),
            "mean_fixed_prefix_relative_delta": finite(prefix_sum / max(1, prefix_count)),
            "maximum_component_ledger_relative_error": finite(ledger_max),
            "mean_component_ledger_relative_error": finite(
                ledger_sum / max(1, ledger_count)
            ),
            "coarse_message_causal_authorized": route_pass,
            "runtime_seconds": time.monotonic() - started,
            "trace_rows_sha256": sha256_file(raw_rows_path()),
            "routing_rows_sha256": sha256_file(routing_rows_path()),
            "output_embedding_direction_used": False,
            "full_vectors_persisted": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "causal_intervention_executed": False,
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(), summary)
        decision = {
            "schema_version": "phase573_natural_trace_decision.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "prefix_causal_mask_audit_pass": prefix_audit_pass,
            "state_transition_observed": bool(state_candidates),
            "routing_transition_observed": bool(route_candidates),
            "earliest_state_event": state_candidates[0] if state_candidates else None,
            "earliest_routing_event": route_candidates[0] if route_candidates else None,
            "coarse_message_causal_authorized": route_pass,
            "reason": (
                "A discovery coordinate replicated in confirmation and heldout with "
                "object/order controls."
                if route_pass else
                "No legal source-routing coordinate passed the frozen three-split and "
                "control gates; stop before causal intervention."
            ),
            "causal_intervention_executed": False,
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(decision_path(), decision)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary_path()
    finally:
        for handle in handles:
            handle.remove()
        component_cache.clear()
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
