#!/usr/bin/env python3
"""Collect Phase1097 per-item full-depth conditional-transition invariants."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
from phase1065_multimode_response_atlas_scan import RoleCapture, event_definitions
import phase1097_conditional_transition_protocol as protocol


EPSILON = 1e-12


def pad_rows(rows: list[dict[str, Any]], pad_id: int, device):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    positions = torch.zeros((len(rows), len(protocol.CAPTURE_ROLES)), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor(
            [int(row["role_positions"][role]) for role in protocol.CAPTURE_ROLES],
            dtype=torch.long,
            device=device,
        )
    return input_ids, attention_mask, lengths, positions


def field_contrasts(values: torch.Tensor) -> torch.Tensor:
    """Return [field, role, ...] factorial contrasts for one unit."""
    indices = {protocol.state_factors(state): index for index, state in enumerate(protocol.STATES)}

    def mean(*, panel=None, task=None, orientation=None, order=None):
        selected = []
        for factors, index in indices.items():
            p_value, t_value, o_value, c_value = factors
            if panel is not None and p_value != panel:
                continue
            if task is not None and t_value != task:
                continue
            if orientation is not None and o_value != orientation:
                continue
            if order is not None and c_value != order:
                continue
            selected.append(values[index])
        if not selected:
            raise RuntimeError("empty factorial selection")
        return torch.stack(selected).mean(dim=0)

    fields: dict[str, torch.Tensor] = {}
    for panel, prefix in (("relational", "relational"), ("role_lookup", "lookup")):
        fields[f"{prefix}_representation"] = mean(panel=panel, orientation=1) - mean(panel=panel, orientation=0)
        fields[f"{prefix}_control"] = mean(panel=panel, task="max") - mean(panel=panel, task="min")
        fields[f"{prefix}_execution"] = 0.5 * (
            (mean(panel=panel, task="max", orientation=0) - mean(panel=panel, task="min", orientation=0))
            - (mean(panel=panel, task="max", orientation=1) - mean(panel=panel, task="min", orientation=1))
        )
        fields[f"{prefix}_carrier"] = 0.5 * (
            (mean(panel=panel, orientation=1, order=1) - mean(panel=panel, orientation=0, order=1))
            - (mean(panel=panel, orientation=1, order=0) - mean(panel=panel, orientation=0, order=0))
        )
    return torch.stack([fields[name] for name in protocol.FIELDS])


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    result = torch.zeros_like(numerator)
    valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (denominator > EPSILON)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def final_norm_module(model):
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "norm"):
        return base.norm
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for name in ("final_layernorm", "ln_f", "norm"):
            if hasattr(transformer, name):
                return getattr(transformer, name)
    raise RuntimeError(f"unable to locate final norm on {type(model).__name__}")


def module_device_dtype(module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise RuntimeError("final norm has no parameter")
    if getattr(parameter, "is_meta", False):
        hook = getattr(module, "_hf_hook", None)
        execution_device = getattr(hook, "execution_device", None)
        weights_map = getattr(hook, "weights_map", None)
        try:
            materialized = weights_map["weight"] if weights_map is not None else None
        except (KeyError, TypeError):
            materialized = None
        if execution_device is None or materialized is None:
            raise RuntimeError("unable to resolve offloaded final norm")
        return torch.device(execution_device), materialized.dtype
    return parameter.device, parameter.dtype


def candidate_parameters(model, units: list[dict[str, Any]]) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise RuntimeError("model has no output embedding weight")
    weight = output.weight
    if getattr(weight, "is_meta", False):
        hook = getattr(output, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        try:
            weight = weights_map["weight"] if weights_map is not None else None
        except (KeyError, TypeError):
            weight = None
        if weight is None or getattr(weight, "is_meta", False):
            raise RuntimeError("unable to resolve offloaded output embedding")
    token_ids = sorted({
        int(token_id)
        for unit in units
        for token_id in (
            unit["states"][protocol.STATES[0]]["candidate_first_token_ids"]["e0"][0],
            unit["states"][protocol.STATES[0]]["candidate_first_token_ids"]["e1"][0],
        )
    })
    index = torch.tensor(token_ids, dtype=torch.long, device=weight.device)
    selected = weight.index_select(0, index).detach().float().cpu()
    rows = {token_id: selected[position] for position, token_id in enumerate(token_ids)}
    bias_map = {token_id: 0.0 for token_id in token_ids}
    bias = getattr(output, "bias", None)
    if bias is not None:
        selected_bias = bias.index_select(0, index.to(bias.device)).detach().float().cpu()
        bias_map = {token_id: float(selected_bias[position].item()) for position, token_id in enumerate(token_ids)}
    return rows, bias_map


def local_candidate_margins(
    hidden: torch.Tensor,
    final_norm,
    norm_device: torch.device,
    norm_dtype: torch.dtype,
    e0_row: torch.Tensor,
    e1_row: torch.Tensor,
    e0_bias: float,
    e1_bias: float,
) -> torch.Tensor:
    shape = hidden.shape[:-1]
    flat = hidden.reshape(-1, hidden.shape[-1]).to(device=norm_device, dtype=norm_dtype)
    normed = final_norm(flat).float()
    e0 = normed @ e0_row.to(device=norm_device)
    e1 = normed @ e1_row.to(device=norm_device)
    return (e0 + e0_bias - e1 - e1_bias).reshape(shape)


def nearest_depth_anchors(n_layers: int) -> tuple[int, ...]:
    depths = tuple(int(round(value * n_layers)) for value in protocol.DEPTH_ANCHORS)
    if len(set(depths)) != len(depths):
        raise RuntimeError(f"depth anchors collide for {n_layers} layers")
    return depths


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1097 protocol audit failed")
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1097 hidden scan is not authorized")
    rows = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["state"])] = row
        unit_meta[str(row["unit_id"])] = {
            "relation": str(row["relation"]),
            "surface": str(row["surface"]),
            "split": str(row["split"]),
            "template": int(row["template"]),
            "item_index": int(row["item_index"]),
        }
    units = []
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit {unit_id}")
        units.append({"unit_id": unit_id, **unit_meta[unit_id], "states": grouped[unit_id]})

    started = time.time()
    model = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        events = event_definitions(len(layers))
        event_keys = [(str(row["component"]), int(row["depth"])) for row in events]
        anchor_depths = nearest_depth_anchors(len(layers))
        anchor_event_indices = [event_keys.index(("residual", depth)) for depth in anchor_depths]
        anchor_by_event = {event_index: anchor for anchor, event_index in enumerate(anchor_event_indices)}
        final_norm = final_norm_module(model)
        norm_device, norm_dtype = module_device_dtype(final_norm)
        output_rows, output_biases = candidate_parameters(model, units)

        relation_index = {value: index for index, value in enumerate(protocol.RELATIONS)}
        surface_index = {value: index for index, value in enumerate(protocol.SURFACES)}
        split_index = {value: index for index, value in enumerate(protocol.SPLITS)}
        role_index = {value: index for index, value in enumerate(protocol.CAPTURE_ROLES)}
        field_index = {value: index for index, value in enumerate(protocol.FIELDS)}
        panel_kinds = ("execution", "carrier")
        ledger_kinds = (
            "relational_execution_representation",
            "relational_execution_control",
            "lookup_execution_representation",
            "lookup_execution_control",
        )

        base_shape = (
            len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.SPLITS),
            len(protocol.FIELDS), len(protocol.CAPTURE_ROLES), len(anchor_depths),
        )
        amplitude_sum = np.zeros(base_shape, dtype=np.float64)
        amplitude_count = np.zeros(base_shape, dtype=np.int32)
        local_margin_sum = np.zeros(base_shape, dtype=np.float64)
        local_margin_count = np.zeros(base_shape, dtype=np.int32)
        gram_shape = base_shape[:-1] + (len(anchor_depths), len(anchor_depths))
        gram_sum = np.zeros(gram_shape, dtype=np.float64)
        gram_count = np.zeros(gram_shape, dtype=np.int32)
        panel_shape = (
            len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.SPLITS),
            len(panel_kinds), len(protocol.CAPTURE_ROLES), len(anchor_depths),
        )
        panel_alignment_sum = np.zeros(panel_shape, dtype=np.float64)
        panel_alignment_count = np.zeros(panel_shape, dtype=np.int32)
        ledger_shape = (
            len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.SPLITS),
            len(ledger_kinds), len(protocol.CAPTURE_ROLES), len(anchor_depths),
        )
        ledger_alignment_sum = np.zeros(ledger_shape, dtype=np.float64)
        ledger_alignment_count = np.zeros(ledger_shape, dtype=np.int32)
        physical_shape = (
            len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.SPLITS),
            len(events), len(protocol.CAPTURE_ROLES), len(protocol.FIELDS),
        )
        physical_sum = np.zeros(physical_shape, dtype=np.float64)
        physical_count = np.zeros(physical_shape, dtype=np.int32)

        unit_amplitudes = []
        unit_grams = []
        unit_local_margins = []
        unit_panel_alignments = []
        unit_ledger_alignments = []
        unit_records = []
        hidden_observations = nonfinite_hidden = 0
        local_readout_observations = nonfinite_local_readout = 0
        candidate_total = candidate_finite = candidate_hit = 0
        identity_maximum = 0.0
        pre_task_maximum = 0.0
        local_readout_maximum_error = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")
        capture = RoleCapture(model, layers)
        capture.register()
        state_order = list(protocol.STATES)
        with torch.inference_mode():
            for unit_number, unit in enumerate(units):
                state_rows = [unit["states"][state] for state in state_order]
                forward_rows = list(state_rows)
                identity_index = None
                if unit_number == 0:
                    forward_rows.append(state_rows[0])
                    identity_index = len(forward_rows) - 1
                input_ids, attention_mask, lengths, positions = pad_rows(forward_rows, int(pad_id), device)
                capture.begin(positions)
                output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                capture.validate()
                batch_axis = torch.arange(len(state_rows), dtype=torch.long, device=output.logits.device)
                last_positions = lengths[:len(state_rows)].to(output.logits.device) - 1
                logits = output.logits[batch_axis, last_positions, :].float()
                native_margins = []
                for slot, row in enumerate(state_rows):
                    e0_id = int(row["candidate_first_token_ids"]["e0"][0])
                    e1_id = int(row["candidate_first_token_ids"]["e1"][0])
                    e0_score = float(logits[slot, e0_id].item())
                    e1_score = float(logits[slot, e1_id].item())
                    native_margin = e0_score - e1_score
                    native_margins.append(native_margin)
                    expected_margin = native_margin if row["expected_class"] == "e0" else -native_margin
                    finite = math.isfinite(native_margin) and math.isfinite(expected_margin)
                    candidate_total += 1
                    candidate_finite += int(finite)
                    candidate_hit += int(finite and expected_margin > 0.0)

                ri = relation_index[unit["relation"]]
                si = surface_index[unit["surface"]]
                qi = split_index[unit["split"]]
                e0_id = int(state_rows[0]["candidate_first_token_ids"]["e0"][0])
                e1_id = int(state_rows[0]["candidate_first_token_ids"]["e1"][0])
                anchor_fields = torch.zeros(
                    (len(anchor_depths), len(protocol.FIELDS), len(protocol.CAPTURE_ROLES), output_rows[e0_id].numel()),
                    dtype=torch.float32,
                    device=device,
                )
                anchor_amplitude = torch.full(
                    (len(protocol.FIELDS), len(protocol.CAPTURE_ROLES), len(anchor_depths)),
                    float("nan"), dtype=torch.float32, device=device,
                )
                anchor_local_margin = torch.full_like(anchor_amplitude, float("nan"))
                anchor_valid = torch.zeros(
                    (len(anchor_depths), len(protocol.FIELDS), len(protocol.CAPTURE_ROLES)),
                    dtype=torch.bool, device=device,
                )

                for event_number, key in enumerate(event_keys):
                    captured = capture.values[key].float()
                    if identity_index is not None:
                        identity_maximum = max(identity_maximum, float((captured[0] - captured[identity_index]).abs().max().item()))
                    values = captured[:len(state_rows)]
                    fields = field_contrasts(values)
                    state_norm = torch.linalg.vector_norm(values, dim=-1).mean(dim=0)
                    field_norm = torch.linalg.vector_norm(fields, dim=-1)
                    relative = field_norm / torch.clamp(state_norm[None, :], min=EPSILON)
                    finite_fields = torch.isfinite(fields).all(dim=-1)
                    hidden_observations += int(finite_fields.numel())
                    nonfinite_hidden += int((~finite_fields).sum().item())
                    relative_np = relative.cpu().numpy()
                    finite_relative = torch.isfinite(relative).cpu().numpy()
                    for role_number in range(len(protocol.CAPTURE_ROLES)):
                        for field_number in range(len(protocol.FIELDS)):
                            if finite_relative[field_number, role_number]:
                                physical_sum[ri, si, qi, event_number, role_number, field_number] += float(relative_np[field_number, role_number])
                                physical_count[ri, si, qi, event_number, role_number, field_number] += 1
                    for role in protocol.PRE_TASK_ROLES:
                        role_number = role_index[role]
                        for field in ("relational_control", "lookup_control", "relational_execution", "lookup_execution"):
                            pre_task_maximum = max(pre_task_maximum, float(fields[field_index[field], role_number].abs().max().item()))

                    if event_number in anchor_by_event:
                        anchor = anchor_by_event[event_number]
                        anchor_fields[anchor] = fields
                        anchor_amplitude[:, :, anchor] = relative
                        anchor_valid[anchor] = finite_fields & (field_norm > EPSILON)
                        local_state_margins = local_candidate_margins(
                            values,
                            final_norm,
                            norm_device,
                            norm_dtype,
                            output_rows[e0_id],
                            output_rows[e1_id],
                            output_biases[e0_id],
                            output_biases[e1_id],
                        )
                        local_fields = field_contrasts(local_state_margins.unsqueeze(-1)).squeeze(-1)
                        anchor_local_margin[:, :, anchor] = local_fields.to(device)
                        local_finite = torch.isfinite(local_state_margins)
                        local_readout_observations += int(local_finite.numel())
                        nonfinite_local_readout += int((~local_finite).sum().item())
                        if anchor == len(anchor_depths) - 1:
                            local_answer = local_state_margins[:, role_index["answer_boundary"]].detach().cpu().numpy()
                            for local_value, native_value in zip(local_answer, native_margins):
                                if math.isfinite(float(local_value)) and math.isfinite(native_value):
                                    local_readout_maximum_error = max(local_readout_maximum_error, abs(float(local_value) - native_value))
                    del captured, values, fields, relative

                normalized = torch.zeros_like(anchor_fields)
                norms = torch.linalg.vector_norm(anchor_fields, dim=-1)
                valid = anchor_valid & torch.isfinite(norms) & (norms > EPSILON)
                normalized[valid] = anchor_fields[valid] / norms[valid, None]
                unit_gram = torch.full(
                    (len(protocol.FIELDS), len(protocol.CAPTURE_ROLES), len(anchor_depths), len(anchor_depths)),
                    float("nan"), dtype=torch.float32, device=device,
                )
                for field_number in range(len(protocol.FIELDS)):
                    for role_number in range(len(protocol.CAPTURE_ROLES)):
                        vectors = normalized[:, field_number, role_number, :]
                        pair_valid = valid[:, field_number, role_number, None] & valid[:, field_number, role_number][None, :]
                        gram = vectors @ vectors.T
                        gram[~pair_valid] = float("nan")
                        unit_gram[field_number, role_number] = gram

                unit_panel = torch.full(
                    (len(panel_kinds), len(protocol.CAPTURE_ROLES), len(anchor_depths)),
                    float("nan"), dtype=torch.float32, device=device,
                )
                for kind_number, kind in enumerate(panel_kinds):
                    left = anchor_fields[:, field_index[f"relational_{kind}"], :, :]
                    right = anchor_fields[:, field_index[f"lookup_{kind}"], :, :]
                    unit_panel[kind_number] = cosine(left, right).T

                unit_ledger = torch.full(
                    (len(ledger_kinds), len(protocol.CAPTURE_ROLES), len(anchor_depths)),
                    float("nan"), dtype=torch.float32, device=device,
                )
                ledger_pairs = (
                    ("relational_execution", "relational_representation"),
                    ("relational_execution", "relational_control"),
                    ("lookup_execution", "lookup_representation"),
                    ("lookup_execution", "lookup_control"),
                )
                for kind_number, (left_name, right_name) in enumerate(ledger_pairs):
                    left = anchor_fields[:, field_index[left_name], :, :]
                    right = anchor_fields[:, field_index[right_name], :, :]
                    unit_ledger[kind_number] = cosine(left, right).T

                amplitude_np = anchor_amplitude.cpu().numpy()
                gram_np = unit_gram.cpu().numpy()
                margin_np = anchor_local_margin.cpu().numpy()
                panel_np = unit_panel.cpu().numpy()
                ledger_np = unit_ledger.cpu().numpy()
                for field_number in range(len(protocol.FIELDS)):
                    for role_number in range(len(protocol.CAPTURE_ROLES)):
                        for anchor in range(len(anchor_depths)):
                            if math.isfinite(float(amplitude_np[field_number, role_number, anchor])):
                                amplitude_sum[ri, si, qi, field_number, role_number, anchor] += float(amplitude_np[field_number, role_number, anchor])
                                amplitude_count[ri, si, qi, field_number, role_number, anchor] += 1
                            if math.isfinite(float(margin_np[field_number, role_number, anchor])):
                                local_margin_sum[ri, si, qi, field_number, role_number, anchor] += float(margin_np[field_number, role_number, anchor])
                                local_margin_count[ri, si, qi, field_number, role_number, anchor] += 1
                        finite_gram = np.isfinite(gram_np[field_number, role_number])
                        gram_sum[ri, si, qi, field_number, role_number][finite_gram] += gram_np[field_number, role_number][finite_gram]
                        gram_count[ri, si, qi, field_number, role_number][finite_gram] += 1
                for kind_number in range(len(panel_kinds)):
                    finite_panel = np.isfinite(panel_np[kind_number])
                    panel_alignment_sum[ri, si, qi, kind_number][finite_panel] += panel_np[kind_number][finite_panel]
                    panel_alignment_count[ri, si, qi, kind_number][finite_panel] += 1
                for kind_number in range(len(ledger_kinds)):
                    finite_ledger = np.isfinite(ledger_np[kind_number])
                    ledger_alignment_sum[ri, si, qi, kind_number][finite_ledger] += ledger_np[kind_number][finite_ledger]
                    ledger_alignment_count[ri, si, qi, kind_number][finite_ledger] += 1

                unit_amplitudes.append(amplitude_np.astype(np.float32))
                unit_grams.append(gram_np.astype(np.float32))
                unit_local_margins.append(margin_np.astype(np.float32))
                unit_panel_alignments.append(panel_np.astype(np.float32))
                unit_ledger_alignments.append(ledger_np.astype(np.float32))
                unit_records.append({
                    "unit_index": unit_number,
                    "unit_id": unit["unit_id"],
                    "relation": unit["relation"],
                    "surface": unit["surface"],
                    "split": unit["split"],
                    "template": unit["template"],
                    "item_index": unit["item_index"],
                })
                del output, logits, input_ids, attention_mask, lengths, positions
                del anchor_fields, anchor_amplitude, anchor_local_margin, normalized, unit_gram, unit_panel, unit_ledger
                capture.values = {}
                if torch.cuda.is_available() and (unit_number + 1) % 6 == 0:
                    torch.cuda.empty_cache()
                completed = unit_number + 1
                if completed % 12 == 0 or completed == len(units):
                    print(json.dumps({"phase": protocol.PHASE, "model": model_name, "units_complete": completed, "units_total": len(units)}), flush=True)

        capture.close()
        capture = None
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            atlas_root / "transition_aggregates.npz",
            amplitude_sum=amplitude_sum,
            amplitude_count=amplitude_count,
            local_margin_sum=local_margin_sum,
            local_margin_count=local_margin_count,
            gram_sum=gram_sum,
            gram_count=gram_count,
            panel_alignment_sum=panel_alignment_sum,
            panel_alignment_count=panel_alignment_count,
            ledger_alignment_sum=ledger_alignment_sum,
            ledger_alignment_count=ledger_alignment_count,
            physical_sum=physical_sum,
            physical_count=physical_count,
        )
        np.savez_compressed(
            atlas_root / "unit_transition_invariants.npz",
            amplitude=np.stack(unit_amplitudes),
            depth_gram=np.stack(unit_grams),
            local_margin=np.stack(unit_local_margins),
            panel_alignment=np.stack(unit_panel_alignments),
            ledger_alignment=np.stack(unit_ledger_alignments),
        )
        protocol.write_jsonl(atlas_root / "unit_index.jsonl", unit_records)
        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1097_model_transition_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "behavior_formal": bool(authorization["models"][model_name]["model_behavior_passed"]),
            "precision": precision,
            "placement": placement,
            "d_model": int(model.get_input_embeddings().weight.shape[1]),
            "layer_count": len(layers),
            "event_count": len(events),
            "events": events,
            "depth_anchor_fractions": list(protocol.DEPTH_ANCHORS),
            "depth_anchor_layers": list(anchor_depths),
            "relations": list(protocol.RELATIONS),
            "surfaces": list(protocol.SURFACES),
            "splits": list(protocol.SPLITS),
            "roles": list(protocol.CAPTURE_ROLES),
            "fields": list(protocol.FIELDS),
            "panel_alignment_kinds": list(panel_kinds),
            "ledger_alignment_kinds": list(ledger_kinds),
            "candidate_count": candidate_total,
            "candidate_finite_fraction": candidate_finite / candidate_total,
            "candidate_accuracy": candidate_hit / candidate_total,
            "hidden_observation_count": hidden_observations,
            "nonfinite_hidden_count": nonfinite_hidden,
            "hidden_finite_fraction_lower_bound": 1.0 - nonfinite_hidden / hidden_observations if hidden_observations else 0.0,
            "local_readout_observation_count": local_readout_observations,
            "nonfinite_local_readout_count": nonfinite_local_readout,
            "local_readout_finite_fraction": 1.0 - nonfinite_local_readout / local_readout_observations if local_readout_observations else 0.0,
            "local_readout_maximum_native_margin_error": local_readout_maximum_error,
            "identity_maximum": identity_maximum,
            "pre_task_control_execution_maximum": pre_task_maximum,
            "unit_count": len(units),
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print({
            "phase": protocol.PHASE,
            "model": model_name,
            "behavior_formal": summary["behavior_formal"],
            "candidate_accuracy": summary["candidate_accuracy"],
            "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
            "local_readout_error": summary["local_readout_maximum_native_margin_error"],
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        })
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
