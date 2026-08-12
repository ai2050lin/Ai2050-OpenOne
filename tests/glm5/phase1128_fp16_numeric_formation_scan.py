#!/usr/bin/env python3
"""Locate FP16 non-finite formation events for one Phase1128 model."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1126_semeval_lexsub_natural_cloze_protocol as source_protocol
import phase1128_fp16_numeric_formation_protocol as protocol


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"No tensor found in hook value: {type(value).__name__}")


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


class NumericCollector:
    def __init__(self, model_name: str, layer_count: int) -> None:
        self.model_name = model_name
        self.layer_count = layer_count
        self.current_rows: list[dict[str, Any]] = []
        self.current_positions: list[list[int]] = []
        self.events: list[dict[str, Any]] = []

    def begin(self, rows: list[dict[str, Any]]) -> None:
        self.current_rows = rows
        self.current_positions = []
        for row in rows:
            positions = [
                *[int(value) - 1 for value in row["candidate_positions"]],
                *[int(value) - 1 for value in row["suffix_positions"]],
            ]
            self.current_positions.append(positions)

    def end(self) -> None:
        self.current_rows = []
        self.current_positions = []

    @staticmethod
    def summarize(tensor: torch.Tensor) -> dict[str, Any]:
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        total_count = int(detached.numel())
        nonfinite_count = total_count - finite_count
        max_abs: float | None = None
        if finite_count:
            cleaned = detached.masked_fill(~finite, 0)
            max_abs = float(cleaned.abs().amax().float().item())
        return {
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
            "max_abs_finite": max_abs,
            "dtype": str(detached.dtype).replace("torch.", ""),
            "device": str(detached.device),
            "selected_shape": list(detached.shape),
        }

    def record(
        self,
        tensor: torch.Tensor,
        *,
        order: int,
        name: str,
        event_class: str,
        layer: int | None,
    ) -> None:
        if not self.current_rows:
            raise RuntimeError(f"Hook fired without active batch: {name}")
        if tensor.ndim < 2 or int(tensor.shape[0]) != len(self.current_rows):
            raise RuntimeError(f"Unexpected tensor shape for {name}: {tuple(tensor.shape)}")
        for batch_index, row in enumerate(self.current_rows):
            positions = torch.tensor(self.current_positions[batch_index], dtype=torch.long, device=tensor.device)
            selected = tensor[batch_index].index_select(0, positions)
            summary = self.summarize(selected)
            self.events.append({
                "model": self.model_name,
                "case_index": int(row["case_index"]),
                "event_order": order,
                "event_name": name,
                "event_class": event_class,
                "layer": layer,
                "relative_depth": (layer / max(self.layer_count - 1, 1)) if layer is not None else None,
                "position_count": len(self.current_positions[batch_index]),
                **summary,
            })

    def record_case_tensor(
        self,
        row: dict[str, Any],
        tensor: torch.Tensor,
        *,
        order: int,
        name: str,
        event_class: str,
    ) -> None:
        summary = self.summarize(tensor)
        self.events.append({
            "model": self.model_name,
            "case_index": int(row["case_index"]),
            "event_order": order,
            "event_name": name,
            "event_class": event_class,
            "layer": None,
            "relative_depth": None,
            "position_count": int(tensor.shape[0]),
            **summary,
        })


def register_hooks(model: Any, collector: NumericCollector) -> list[Any]:
    layers = get_layers(model)
    if len(layers) != collector.layer_count:
        raise RuntimeError("Layer count changed after protocol freeze")
    registry = protocol.event_registry(collector.layer_count)
    by_name = {event["name"]: event for event in registry}
    handles: list[Any] = []

    def output_hook(event_name: str) -> Callable[..., None]:
        event = by_name[event_name]

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            collector.record(first_tensor(output), order=event["order"], name=event_name,
                             event_class=event["event_class"], layer=event["layer"])

        return hook

    def input_hook(event_name: str) -> Callable[..., None]:
        event = by_name[event_name]

        def hook(_module: Any, inputs: Any) -> None:
            collector.record(first_tensor(inputs), order=event["order"], name=event_name,
                             event_class=event["event_class"], layer=event["layer"])

        return hook

    handles.append(model.get_input_embeddings().register_forward_hook(output_hook("embedding")))
    for layer_index, layer in enumerate(layers):
        required = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        missing = [name for name in required if not hasattr(layer, name)]
        if missing:
            raise RuntimeError(f"Layer {layer_index} missing modules: {missing}")
        handles.append(layer.register_forward_pre_hook(input_hook(f"layer_{layer_index}.input")))
        handles.append(layer.input_layernorm.register_forward_hook(output_hook(f"layer_{layer_index}.attention_norm")))
        handles.append(layer.self_attn.register_forward_hook(output_hook(f"layer_{layer_index}.attention_output")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(output_hook(f"layer_{layer_index}.mlp_norm")))
        handles.append(layer.mlp.register_forward_hook(output_hook(f"layer_{layer_index}.mlp_output")))
        handles.append(layer.register_forward_hook(output_hook(f"layer_{layer_index}.output")))
    if not hasattr(model, "model") or not hasattr(model.model, "norm"):
        raise RuntimeError("Final norm module not found")
    handles.append(model.model.norm.register_forward_hook(output_hook("final_norm")))
    return handles


def score_batch(
    model: Any,
    input_ids: torch.Tensor,
    rows: list[dict[str, Any]],
    collector: NumericCollector,
) -> list[dict[str, Any]]:
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = output.logits
    final_event = protocol.event_registry(collector.layer_count)[-1]
    results: list[dict[str, Any]] = []
    for batch_index, row in enumerate(rows):
        candidate_positions = [int(value) for value in row["candidate_positions"]]
        suffix_positions = [int(value) for value in row["suffix_positions"]]
        target_positions = [*candidate_positions, *suffix_positions]
        prediction_positions = torch.tensor(
            [position - 1 for position in target_positions], dtype=torch.long, device=logits.device
        )
        target_ids = torch.tensor(
            [int(row["input_ids"][position]) for position in target_positions],
            dtype=torch.long,
            device=logits.device,
        )
        selected_logits = logits[batch_index].index_select(0, prediction_positions)
        collector.record_case_tensor(
            row,
            selected_logits,
            order=final_event["order"],
            name=final_event["name"],
            event_class=final_event["event_class"],
        )
        selected_float = selected_logits.float()
        log_probs = (
            selected_float.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            - torch.logsumexp(selected_float, dim=-1)
        ).detach().cpu().tolist()
        candidate_values = [float(value) for value in log_probs[:len(candidate_positions)]]
        suffix_values = [float(value) for value in log_probs[len(candidate_positions):]]
        candidate_logp = sum(candidate_values)
        suffix_mean = sum(suffix_values) / len(suffix_values) if suffix_values else 0.0
        total_score = candidate_logp + suffix_mean
        results.append({
            "candidate_logp": safe_float(candidate_logp),
            "suffix_mean_logp": safe_float(suffix_mean),
            "total_score": safe_float(total_score),
            "candidate_finite": math.isfinite(candidate_logp),
            "suffix_finite": math.isfinite(suffix_mean),
            "total_finite": math.isfinite(total_score),
        })
    del output, logits
    return results


def source_value(row: dict[str, Any], key: str) -> float | None:
    return safe_float(float(row[key]))


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["passed"] or protocol_audit["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1128 protocol is not authorized")
    rows = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    source_rows = source_protocol.read_jsonl(
        source_protocol.OUT_ROOT / "behavior" / model_name / "scores.jsonl"
    )
    source_by_case = {int(row["case_index"]): row for row in source_rows}
    if source_protocol.digest(rows) != prereg["source"]["links"][model_name]["case_digest"]:
        raise RuntimeError("Case digest mismatch")
    if source_protocol.digest(source_rows) != prereg["source"]["links"][model_name]["score_detail_digest"]:
        raise RuntimeError("Source score digest mismatch")

    started = time.time()
    model = None
    handles: list[Any] = []
    try:
        model, _tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("Phase1128 requires unquantized FP16")
        layer_count = len(get_layers(model))
        if layer_count != prereg["model_specs"][model_name]["layer_count"]:
            raise RuntimeError("Layer count mismatch")
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        if parameter_count != prereg["model_specs"][model_name]["source_parameter_count"]:
            raise RuntimeError("Parameter count mismatch")

        collector = NumericCollector(model_name, layer_count)
        handles = register_hooks(model, collector)
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        batch_size = int(prereg["case_policy"]["batch_sizes"][model_name])
        rerun_by_case: dict[int, dict[str, Any]] = {}
        completed = 0
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), batch_size):
                    batch = panel[start:start + batch_size]
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in batch], dtype=torch.long, device=device
                    )
                    collector.begin(batch)
                    scores = score_batch(model, input_ids, batch, collector)
                    collector.end()
                    for row, score in zip(batch, scores):
                        rerun_by_case[int(row["case_index"])] = score
                    completed += len(batch)
                    print(json.dumps({"phase": protocol.PHASE, "model": model_name,
                                      "completed": completed, "total": len(rows)}), flush=True)
                    del input_ids

        collector.events.sort(key=lambda row: (row["case_index"], row["event_order"]))
        events_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in collector.events:
            events_by_case[int(event["case_index"])].append(event)
        expected_event_count = int(prereg["model_specs"][model_name]["expected_event_count_per_case"])
        case_details: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: int(item["case_index"])):
            case_index = int(row["case_index"])
            source = source_by_case[case_index]
            rerun = rerun_by_case[case_index]
            case_events = events_by_case[case_index]
            first_nonfinite = next((event for event in case_events if event["nonfinite_count"] > 0), None)
            source_candidate_finite = math.isfinite(float(source["candidate_logp"]))
            source_suffix_finite = math.isfinite(float(source["suffix_mean_logp"]))
            source_total_finite = math.isfinite(float(source["total_score"]))
            case_details.append({
                "model": model_name,
                "case_index": case_index,
                "partition": row["partition"],
                "route": row["route"],
                "candidate_side": row["candidate_side"],
                "source_candidate_logp": source_value(source, "candidate_logp"),
                "source_suffix_mean_logp": source_value(source, "suffix_mean_logp"),
                "source_total_score": source_value(source, "total_score"),
                "source_candidate_finite": source_candidate_finite,
                "source_suffix_finite": source_suffix_finite,
                "source_total_finite": source_total_finite,
                "rerun_candidate_logp": rerun["candidate_logp"],
                "rerun_suffix_mean_logp": rerun["suffix_mean_logp"],
                "rerun_total_score": rerun["total_score"],
                "rerun_candidate_finite": rerun["candidate_finite"],
                "rerun_suffix_finite": rerun["suffix_finite"],
                "rerun_total_finite": rerun["total_finite"],
                "candidate_finite_parity": source_candidate_finite == rerun["candidate_finite"],
                "suffix_finite_parity": source_suffix_finite == rerun["suffix_finite"],
                "total_finite_parity": source_total_finite == rerun["total_finite"],
                "event_count": len(case_events),
                "event_count_expected": len(case_events) == expected_event_count,
                "any_tracked_nonfinite": first_nonfinite is not None,
                "first_nonfinite_order": first_nonfinite["event_order"] if first_nonfinite else None,
                "first_nonfinite_name": first_nonfinite["event_name"] if first_nonfinite else None,
                "first_nonfinite_class": first_nonfinite["event_class"] if first_nonfinite else None,
                "first_nonfinite_layer": first_nonfinite["layer"] if first_nonfinite else None,
                "first_nonfinite_relative_depth": first_nonfinite["relative_depth"] if first_nonfinite else None,
            })

        event_digest = protocol.digest(collector.events)
        case_digest = protocol.digest(case_details)
        summary_core = {
            "schema_version": "phase1128_fp16_numeric_formation_scan.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "source_case_digest": prereg["source"]["links"][model_name]["case_digest"],
            "source_score_digest": prereg["source"]["links"][model_name]["score_detail_digest"],
            "case_count": len(case_details),
            "event_count": len(collector.events),
            "expected_event_count_per_case": expected_event_count,
            "precision": precision,
            "placement": placement,
            "parameter_count": parameter_count,
            "layer_count": layer_count,
            "batch_size": batch_size,
            "elapsed_seconds": time.time() - started,
            "event_digest": event_digest,
            "case_detail_digest": case_digest,
        }
        summary = dict(summary_core)
        summary["summary_digest"] = protocol.digest(summary_core)
        output_root = protocol.OUT_ROOT / "scan" / model_name
        protocol.write_jsonl(output_root / "events.jsonl", collector.events)
        protocol.write_jsonl(output_root / "cases.jsonl", case_details)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for handle in handles:
            handle.remove()
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
