#!/usr/bin/env python3
"""Collect the Phase1109 head-wise attention-routing map."""

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
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1108_exact_key_event_protocol as source
import phase1109_attention_routing_protocol as protocol


EPSILON = 1e-12
SOURCE_NAMES = ("key0", "key1", "record0", "record1")


def pad_rows(rows: list[dict[str, Any]], pad_id: int, device):
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), maximum), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
    return input_ids, attention_mask


def source_masks(rows: list[dict[str, Any]], maximum: int) -> torch.Tensor:
    masks = torch.zeros((len(rows), len(SOURCE_NAMES), maximum), dtype=torch.float32)
    for batch, row in enumerate(rows):
        spans = {
            "key0": row["key_spans"]["key0"],
            "key1": row["key_spans"]["key1"],
            "record0": row["record_spans"]["record0"],
            "record1": row["record_spans"]["record1"],
        }
        for source_index, name in enumerate(SOURCE_NAMES):
            start, end = (int(value) for value in spans[name])
            masks[batch, source_index, start:end + 1] = 1.0
    return masks


def group_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    metadata = {}
    for row in rows:
        grouped[row["unit_id"]][row["state"]] = row
        metadata[row["unit_id"]] = {
            "unit_id": row["unit_id"],
            "relation_pair": row["relation_pair"],
            "surface": row["surface"],
            "split": row["split"],
            "template": int(row["template"]),
            "item_index": int(row["item_index"]),
        }
    units = []
    expected = set(source.STATES)
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != expected:
            raise RuntimeError(f"incomplete state cube: {unit_id}")
        units.append({**metadata[unit_id], "states": grouped[unit_id]})
    return units


def target_follow(values: np.ndarray, target: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    target_mass = values[..., offset + target]
    other_mass = values[..., offset + (1 - target)]
    total = target_mass + other_mass
    following = np.divide(
        target_mass - other_mass,
        total,
        out=np.zeros_like(total, dtype=np.float32),
        where=total > EPSILON,
    )
    return following.astype(np.float32), total.astype(np.float32)


def aggregate_unit(
    case_values: dict[str, np.ndarray],
    state_rows: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sample = next(iter(case_values.values()))
    layer_count, head_count, role_count, _ = sample.shape
    shape = (
        len(protocol.LABEL_REGIMES),
        len(protocol.ROUTE_TYPES),
        len(protocol.CONGRUENCES),
        layer_count,
        head_count,
        role_count,
    )
    key_follow = np.zeros(shape, dtype=np.float32)
    key_total = np.zeros(shape, dtype=np.float32)
    record_follow = np.zeros(shape, dtype=np.float32)
    record_total = np.zeros(shape, dtype=np.float32)
    regime_index = {value: index for index, value in enumerate(protocol.LABEL_REGIMES)}
    route_index = {value: index for index, value in enumerate(protocol.ROUTE_TYPES)}
    congruence_index = {value: index for index, value in enumerate(protocol.CONGRUENCES)}
    buckets: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for state, values in case_values.items():
        row = state_rows[state]
        target = int(row["target_relation"])
        key_f, key_t = target_follow(values, target, 0)
        record_f, record_t = target_follow(values, target, 2)
        key = (
            regime_index[row["label_regime"]],
            route_index[row["route_type"]],
            congruence_index[row["congruence"]],
        )
        buckets[key].append((key_f, key_t, record_f, record_t))
    for key, observations in buckets.items():
        if len(observations) != 8:
            raise RuntimeError(f"unbalanced Phase1109 factor cell {key}: {len(observations)}")
        key_follow[key] = np.mean([value[0] for value in observations], axis=0)
        key_total[key] = np.mean([value[1] for value in observations], axis=0)
        record_follow[key] = np.mean([value[2] for value in observations], axis=0)
        record_total[key] = np.mean([value[3] for value in observations], axis=0)

    pre_role = protocol.QUERY_ROLES.index("pre_selector")
    pre_selector_error = 0.0
    paired: dict[tuple[Any, ...], dict[int, np.ndarray]] = defaultdict(dict)
    for state, values in case_values.items():
        row = state_rows[state]
        key = (
            row["label_regime"], row["route_type"], row["congruence"],
            int(row["relation_order"]), int(row["orientation"]),
        )
        paired[key][int(row["target_relation"])] = values[:, :, pre_role, :]
    for values in paired.values():
        if set(values) != {0, 1}:
            raise RuntimeError("pre-selector target pair is incomplete")
        pre_selector_error = max(
            pre_selector_error,
            float(np.max(np.abs(values[1] - values[0]))),
        )
    return key_follow, key_total, record_follow, record_total, pre_selector_error


def denied(model_name: str) -> None:
    atlas_root = protocol.OUT_ROOT / "atlas" / model_name
    atlas_root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "phase1109_hidden_access_denial.v1",
        "phase": protocol.PHASE,
        "model": model_name,
        "hidden_access": False,
        "reason": "Phase1108 behavior authorization denied all frozen pairs for this model.",
        "source_behavior_authorization_digest": protocol.read_json(
            protocol.SOURCE_AUTHORIZATION
        )["authorization_digest"],
    }
    value["denial_digest"] = protocol.digest(value)
    protocol.write_json(atlas_root / "denial.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run(model_name: str) -> None:
    if model_name in protocol.DENIED_MODELS:
        denied(model_name)
        return
    if model_name not in protocol.AUTHORIZED_MODELS:
        raise RuntimeError(f"unknown authorization status for {model_name}")
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1109 protocol audit failed")
    rows = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    ))
    if protocol.digest(rows) != prereg["case_digests"][model_name]:
        raise RuntimeError("Phase1109 case digest mismatch")
    units = group_units(rows)
    started = time.time()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        layer_count = len(layers)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos token")

        unit_key_follow = []
        unit_key_total = []
        unit_record_follow = []
        unit_record_total = []
        unit_metadata = []
        finite_count = 0
        observed_count = 0
        identity_error = 0.0
        pre_selector_error = 0.0
        head_count = None
        with torch.inference_mode():
            for unit_number, unit in enumerate(units):
                state_rows = [unit["states"][state] for state in source.STATES]
                forward_rows = list(state_rows)
                duplicate_index = None
                if unit_number == 0:
                    forward_rows.append(state_rows[0])
                    duplicate_index = len(forward_rows) - 1
                input_ids, attention_mask = pad_rows(forward_rows, int(pad_id), device)
                masks_cpu = source_masks(forward_rows, input_ids.shape[1])
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    output_attentions=True,
                    logits_to_keep=1,
                )
                attentions = output.attentions
                if attentions is None or len(attentions) != layer_count:
                    raise RuntimeError("model did not return one attention matrix per layer")
                if head_count is None:
                    head_count = int(attentions[0].shape[1])
                case_values = {
                    row["state"]: np.zeros(
                        (layer_count, head_count, len(protocol.QUERY_ROLES), len(SOURCE_NAMES)),
                        dtype=np.float32,
                    )
                    for row in state_rows
                }
                duplicate_values = None
                if duplicate_index is not None:
                    duplicate_values = np.zeros(
                        (layer_count, head_count, len(protocol.QUERY_ROLES), len(SOURCE_NAMES)),
                        dtype=np.float32,
                    )
                for layer_index, attention in enumerate(attentions):
                    if int(attention.shape[1]) != head_count:
                        raise RuntimeError("attention head count changed across layers")
                    layer_masks = masks_cpu.to(attention.device, dtype=attention.dtype)
                    batch_index = torch.arange(len(forward_rows), device=attention.device)
                    for role_index, role in enumerate(protocol.QUERY_ROLES):
                        positions = torch.tensor(
                            [int(row["query_positions"][role]) for row in forward_rows],
                            dtype=torch.long,
                            device=attention.device,
                        )
                        selected = attention[batch_index, :, positions, :]
                        masses = torch.einsum("bhk,bsk->bhs", selected, layer_masks)
                        values = masses.float().cpu().numpy()
                        observed_count += int(values.size)
                        finite_count += int(np.isfinite(values).sum())
                        for row_index, row in enumerate(state_rows):
                            case_values[row["state"]][layer_index, :, role_index, :] = values[row_index]
                        if duplicate_values is not None:
                            duplicate_values[layer_index, :, role_index, :] = values[duplicate_index]
                if duplicate_values is not None:
                    identity_error = float(np.max(np.abs(
                        duplicate_values - case_values[state_rows[0]["state"]]
                    )))
                aggregates = aggregate_unit(
                    case_values,
                    {row["state"]: row for row in state_rows},
                )
                unit_key_follow.append(aggregates[0])
                unit_key_total.append(aggregates[1])
                unit_record_follow.append(aggregates[2])
                unit_record_total.append(aggregates[3])
                pre_selector_error = max(pre_selector_error, aggregates[4])
                unit_metadata.append({
                    key: unit[key]
                    for key in (
                        "unit_id", "relation_pair", "surface", "split",
                        "template", "item_index",
                    )
                })
                del output, attentions, input_ids, attention_mask, masks_cpu, case_values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = unit_number + 1
                if completed % 6 == 0 or completed == len(units):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "units_complete": completed,
                        "units_total": len(units),
                    }), flush=True)

        if head_count is None:
            raise RuntimeError("no attention observations were collected")
        finite_fraction = finite_count / max(observed_count, 1)
        thresholds = prereg["thresholds"]
        checks = {
            "precision_fp16_no_quantization": (
                precision["has_fp16_parameters"]
                and not precision["has_bf16_parameters"]
                and not precision["has_quantized_modules"]
            ),
            "attention_finite_fraction": (
                finite_fraction >= thresholds["minimum_attention_finite_fraction"]
            ),
            "deterministic_identity": (
                identity_error <= thresholds["maximum_deterministic_identity_error"]
            ),
            "pre_selector_identity": (
                pre_selector_error <= thresholds["maximum_pre_selector_identity_error"]
            ),
            "unit_count": len(unit_metadata) == 96,
        }
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            atlas_root / "attention_routing_fields.npz",
            key_follow=np.asarray(unit_key_follow, dtype=np.float32),
            key_total=np.asarray(unit_key_total, dtype=np.float32),
            record_follow=np.asarray(unit_record_follow, dtype=np.float32),
            record_total=np.asarray(unit_record_total, dtype=np.float32),
        )
        protocol.write_json(atlas_root / "units.json", unit_metadata)
        summary = {
            "schema_version": "phase1109_model_attention_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digests"][model_name],
            "source_behavior_authorization_digest": prereg["source"]["behavior_authorization_digest"],
            "precision": precision,
            "placement": placement,
            "layer_count": layer_count,
            "head_count": head_count,
            "query_roles": list(protocol.QUERY_ROLES),
            "source_names": list(SOURCE_NAMES),
            "unit_count": len(unit_metadata),
            "observed_attention_mass_count": observed_count,
            "observed_attention_mass_finite_fraction": finite_fraction,
            "deterministic_identity_maximum_error": identity_error,
            "pre_selector_identity_maximum_error": pre_selector_error,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "layer_count": layer_count,
            "head_count": head_count,
            "unit_count": len(unit_metadata),
            "finite_fraction": finite_fraction,
            "identity_error": identity_error,
            "pre_selector_error": pre_selector_error,
            "all_checks_passed": summary["all_checks_passed"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
