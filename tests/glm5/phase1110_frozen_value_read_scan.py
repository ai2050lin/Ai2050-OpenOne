#!/usr/bin/env python3
"""Collect frozen-head key/body V and A-times-V fields for Phase1110."""

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
import phase1110_frozen_value_read_protocol as protocol


EPSILON = 1e-12


def output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported projection output {type(output)!r}")


def attention_tensor(output: Any) -> torch.Tensor:
    if not isinstance(output, (tuple, list)):
        raise TypeError("self-attention did not return a tuple")
    candidates = [
        value for value in output
        if torch.is_tensor(value) and value.ndim == 4
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one attention tensor, found {len(candidates)}")
    return candidates[0]


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
    masks = torch.zeros(
        (len(rows), len(protocol.SOURCE_NAMES), maximum), dtype=torch.float32
    )
    for batch, row in enumerate(rows):
        for source_index, name in enumerate(protocol.SOURCE_NAMES):
            positions = [int(value) for value in row["source_positions"][name]]
            masks[batch, source_index, positions] = 1.0
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
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(source.STATES):
            raise RuntimeError(f"incomplete state cube: {unit_id}")
        units.append({**metadata[unit_id], "states": grouped[unit_id]})
    return units


class FrozenValueCapture:
    def __init__(self, layers: list[Any], events: list[dict[str, Any]]) -> None:
        self.layers = layers
        self.events = events
        self.by_layer: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for event_index, event in enumerate(events):
            self.by_layer[int(event["layer_index"])].append((event_index, event))
        self.handles = []
        self.v_cache: dict[int, torch.Tensor] = {}
        self.head_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.masks: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.contrasts: torch.Tensor | None = None
        self.mass: np.ndarray | None = None
        self.av: np.ndarray | None = None
        self.raw: np.ndarray | None = None
        self.head: np.ndarray | None = None
        self.readout: np.ndarray | None = None
        self.reconstruction: np.ndarray | None = None
        self.head_dim: int | None = None

    def register(self) -> None:
        for layer_index in sorted(self.by_layer):
            attention = self.layers[layer_index].self_attn
            self.handles.append(
                attention.v_proj.register_forward_hook(self._v_hook(layer_index))
            )
            self.handles.append(
                attention.o_proj.register_forward_pre_hook(self._o_hook(layer_index))
            )
            self.handles.append(
                attention.register_forward_hook(self._attention_hook(layer_index))
            )

    def begin(
        self,
        masks: torch.Tensor,
        positions: torch.Tensor,
        contrasts: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.masks = masks
        self.positions = positions
        self.contrasts = contrasts
        self.v_cache.clear()
        self.head_cache.clear()
        self.mass = np.full(
            (batch_size, len(self.events), len(protocol.SOURCE_NAMES)),
            np.nan, dtype=np.float32,
        )
        self.av = None
        self.raw = None
        self.head = None
        self.readout = np.full_like(self.mass, np.nan)
        self.reconstruction = np.full(
            (batch_size, len(self.events)), np.nan, dtype=np.float32
        )
        self.head_dim = None

    def end(self) -> dict[str, np.ndarray]:
        if self.v_cache or self.head_cache:
            raise RuntimeError(
                f"unconsumed capture caches: V={sorted(self.v_cache)}, O={sorted(self.head_cache)}"
            )
        arrays = {
            "attention_mass": self.mass,
            "av_vectors": self.av,
            "raw_value_means": self.raw,
            "head_vectors": self.head,
            "readout_alignment": self.readout,
            "reconstruction_relative_error": self.reconstruction,
        }
        if any(value is None for value in arrays.values()):
            raise RuntimeError("capture arrays were not fully initialized")
        if any(np.isnan(value).any() for value in arrays.values()):
            raise RuntimeError("capture arrays contain unfilled values")
        self.masks = None
        self.positions = None
        self.contrasts = None
        return arrays  # type: ignore[return-value]

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def _v_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            self.v_cache[layer_index] = output_tensor(output)
        return hook

    def _o_hook(self, layer_index: int):
        def hook(module, inputs):
            if self.positions is None:
                raise RuntimeError("o_proj hook fired outside an active batch")
            hidden = inputs[0]
            events = self.by_layer[layer_index]
            width = int(hidden.shape[-1])
            attention = self.layers[layer_index].self_attn
            n_heads = int(getattr(attention, "num_heads", 0) or getattr(attention.config, "num_attention_heads"))
            if width % n_heads:
                raise RuntimeError("pre-o_proj width is not head aligned")
            head_dim = width // n_heads
            if self.head_dim is None:
                self.head_dim = head_dim
                shape = (hidden.shape[0], len(self.events), len(protocol.SOURCE_NAMES), head_dim)
                self.av = np.full(shape, np.nan, dtype=np.float32)
                self.raw = np.full(shape, np.nan, dtype=np.float32)
                self.head = np.full(
                    (hidden.shape[0], len(self.events), head_dim), np.nan, dtype=np.float32
                )
            elif self.head_dim != head_dim:
                raise RuntimeError("head dimension changed across frozen events")
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            positions = self.positions.to(hidden.device)
            cache = {}
            for event_index, event in events:
                head = int(event["head_index"])
                value = hidden[batch, positions, head * head_dim:(head + 1) * head_dim]
                cache[event_index] = value.detach().float().cpu()
                self.head[:, event_index, :] = cache[event_index].numpy()
            self.head_cache[layer_index] = cache
        return hook

    def _attention_hook(self, layer_index: int):
        def hook(module, _inputs, output):
            if self.masks is None or self.positions is None or self.contrasts is None:
                raise RuntimeError("attention hook fired outside an active batch")
            values = self.v_cache.pop(layer_index)
            actual = self.head_cache.pop(layer_index)
            attention = attention_tensor(output).float()
            batch_size, n_heads, _, sequence = attention.shape
            head_dim = int(self.head_dim or 0)
            if head_dim <= 0:
                raise RuntimeError("head dimension was not captured")
            n_kv_heads = int(values.shape[-1] // head_dim)
            if n_heads % n_kv_heads:
                raise RuntimeError("query/KV grouping drift")
            values = values.reshape(batch_size, values.shape[1], n_kv_heads, head_dim).float()
            masks = self.masks[:, :, :sequence].to(values.device, dtype=values.dtype)
            counts = masks.sum(dim=-1).clamp_min(1.0)
            batch = torch.arange(batch_size, device=attention.device)
            positions = self.positions.to(attention.device)
            for event_index, event in self.by_layer[layer_index]:
                head = int(event["head_index"])
                kv_head = head // (n_heads // n_kv_heads)
                weights = attention[batch, head, positions, :sequence].to(values.device)
                head_values = values[:, :sequence, kv_head, :]
                mass = torch.einsum("bk,bsk->bs", weights, masks)
                av = torch.einsum("bk,bsk,bkd->bsd", weights, masks, head_values)
                raw = torch.einsum("bsk,bkd->bsd", masks, head_values) / counts[..., None]
                reconstructed = av.sum(dim=1)
                actual_head = actual[event_index].to(reconstructed.device)
                relative_error = (
                    (reconstructed - actual_head).norm(dim=-1)
                    / actual_head.norm(dim=-1).clamp_min(EPSILON)
                )
                o_weight = self.layers[layer_index].self_attn.o_proj.weight
                start = head * head_dim
                stop = start + head_dim
                projected = torch.matmul(av.to(o_weight.device), o_weight[:, start:stop].T.float())
                contrast = self.contrasts.to(projected.device)
                numerator = torch.einsum("bsd,bd->bs", projected, contrast)
                denominator = projected.norm(dim=-1) * contrast.norm(dim=-1)[:, None]
                alignment = numerator / denominator.clamp_min(EPSILON)
                self.mass[:, event_index, :] = mass.detach().float().cpu().numpy()
                self.av[:, event_index, :, :] = av.detach().float().cpu().numpy()
                self.raw[:, event_index, :, :] = raw.detach().float().cpu().numpy()
                self.readout[:, event_index, :] = alignment.detach().float().cpu().numpy()
                self.reconstruction[:, event_index] = relative_error.detach().float().cpu().numpy()
        return hook


def denied(model_name: str) -> None:
    atlas_root = protocol.OUT_ROOT / "atlas" / model_name
    atlas_root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "phase1110_hidden_access_denial.v1",
        "phase": protocol.PHASE,
        "model": model_name,
        "hidden_access": False,
        "reason": "Phase1108 behavior authorization denied hidden access for this model.",
    }
    value["denial_digest"] = protocol.digest(value)
    protocol.write_json(atlas_root / "denial.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def candidate_contrasts(model, rows: list[dict[str, Any]]) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    weight = output_embeddings.weight
    if weight.device.type == "meta":
        hook = getattr(output_embeddings, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is None:
            raise RuntimeError("offloaded output embedding has no weights map")
        weight = weights_map["weight"]
    result = []
    with torch.no_grad():
        for row in rows:
            expected = str(row["expected_class"])
            other = "e1" if expected == "e0" else "e0"
            expected_id = int(row["candidate_first_token_ids"][expected][0])
            other_id = int(row["candidate_first_token_ids"][other][0])
            value = weight[expected_id].float() - weight[other_id].float()
            result.append(value.detach().cpu())
    return torch.stack(result, dim=0)


def run(model_name: str) -> None:
    if model_name in protocol.DENIED_MODELS:
        denied(model_name)
        return
    if model_name not in protocol.AUTHORIZED_MODELS:
        raise RuntimeError(f"unknown authorization status for {model_name}")
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1110 protocol audit failed")
    rows = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    ))
    if protocol.digest(rows) != prereg["case_digests"][model_name]:
        raise RuntimeError("Phase1110 case digest mismatch")
    units = group_units(rows)
    events = prereg["selected_events"][model_name]
    started = time.time()
    model = None
    capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos token")
        capture = FrozenValueCapture(layers, events)
        capture.register()
        fields: dict[str, list[np.ndarray]] = defaultdict(list)
        unit_metadata = []
        with torch.inference_mode():
            for unit_number, unit in enumerate(units):
                state_rows = [unit["states"][state] for state in source.STATES]
                input_ids, attention_mask = pad_rows(state_rows, int(pad_id), device)
                masks = source_masks(state_rows, input_ids.shape[1])
                positions = torch.tensor(
                    [int(row["query_position"]) for row in state_rows], dtype=torch.long
                )
                contrasts = candidate_contrasts(model, state_rows)
                capture.begin(masks, positions, contrasts, len(state_rows))
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    output_attentions=True,
                    logits_to_keep=1,
                )
                arrays = capture.end()
                for key, value in arrays.items():
                    fields[key].append(value)
                unit_metadata.append({
                    key: unit[key]
                    for key in ("unit_id", "relation_pair", "surface", "split", "template", "item_index")
                })
                del output, input_ids, attention_mask, masks, positions, contrasts, arrays
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

        stacked = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
        finite_count = sum(int(np.isfinite(value).sum()) for value in stacked.values())
        observed_count = sum(int(value.size) for value in stacked.values())
        finite_fraction = finite_count / max(observed_count, 1)
        maximum_reconstruction = float(np.max(stacked["reconstruction_relative_error"]))
        thresholds = prereg["thresholds"]
        checks = {
            "precision_fp16_no_quantization": precision["has_fp16_parameters"] and not precision["has_bf16_parameters"] and not precision["has_quantized_modules"],
            "value_finite_fraction": finite_fraction >= thresholds["minimum_value_finite_fraction"],
            "head_reconstruction": maximum_reconstruction <= thresholds["maximum_head_reconstruction_relative_error"],
            "unit_count": len(unit_metadata) == 48,
            "event_count": len(events) == 4,
        }
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(atlas_root / "frozen_value_read_fields.npz", **stacked)
        protocol.write_json(atlas_root / "units.json", unit_metadata)
        summary = {
            "schema_version": "phase1110_model_value_read_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "layer_count": len(layers),
            "selected_events": events,
            "source_names": list(protocol.SOURCE_NAMES),
            "state_order": list(source.STATES),
            "unit_count": len(unit_metadata),
            "observed_value_count": observed_count,
            "observed_value_finite_fraction": finite_fraction,
            "maximum_head_reconstruction_relative_error": maximum_reconstruction,
            "array_shapes": {key: list(value.shape) for key, value in stacked.items()},
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "unit_count": len(unit_metadata),
            "finite_fraction": finite_fraction,
            "maximum_reconstruction_relative_error": maximum_reconstruction,
            "all_checks_passed": summary["all_checks_passed"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
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
