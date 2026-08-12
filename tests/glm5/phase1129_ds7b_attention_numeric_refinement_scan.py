#!/usr/bin/env python3
"""Run the one-shot DS7B layer-27 attention numerical refinement."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1128_fp16_numeric_formation_protocol as source_protocol
import phase1129_ds7b_attention_numeric_refinement_protocol as protocol


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


class Collector:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.positions: list[list[int]] = []
        self.events: list[dict[str, Any]] = []
        self.target_attention_active = False
        self.registry = {event["name"]: event for event in protocol.EVENT_REGISTRY}

    def begin(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.positions = [[
            *[int(value) - 1 for value in row["candidate_positions"]],
            *[int(value) - 1 for value in row["suffix_positions"]],
        ] for row in rows]

    def end(self) -> None:
        self.rows = []
        self.positions = []
        self.target_attention_active = False

    @staticmethod
    def summarize(tensor: torch.Tensor, score_rows: bool = False) -> dict[str, Any]:
        value = tensor.detach()
        nan_count = int(torch.isnan(value).sum().item())
        posinf_count = int(torch.isposinf(value).sum().item())
        neginf_count = int(torch.isneginf(value).sum().item())
        finite = torch.isfinite(value)
        finite_count = int(finite.sum().item())
        max_abs = None
        if finite_count:
            max_abs = float(value.masked_fill(~finite, 0).abs().amax().float().item())
        all_nonfinite_rows = 0
        if score_rows:
            matrix = value.reshape(-1, value.shape[-1])
            all_nonfinite_rows = int((~torch.isfinite(matrix).any(dim=-1)).sum().item())
        return {
            "finite_count": finite_count,
            "nan_count": nan_count,
            "posinf_count": posinf_count,
            "neginf_count": neginf_count,
            "nonfinite_count": nan_count + posinf_count + neginf_count,
            "all_nonfinite_rows": all_nonfinite_rows,
            "max_abs_finite": max_abs,
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "selected_shape": list(value.shape),
        }

    def selected(self, tensor: torch.Tensor, batch_index: int, mode: str) -> torch.Tensor:
        positions = self.positions[batch_index]
        if mode == "query":
            index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
            return tensor[batch_index].index_select(0, index)
        if mode == "prefix":
            return tensor[batch_index, :max(positions) + 1]
        if mode == "attention":
            index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
            return tensor[batch_index].index_select(1, index)
        raise ValueError(f"Unknown selection mode: {mode}")

    def record(self, name: str, tensor: torch.Tensor, mode: str, score_rows: bool = False) -> None:
        if not self.rows:
            raise RuntimeError(f"Event without active batch: {name}")
        event = self.registry[name]
        for batch_index, row in enumerate(self.rows):
            selected = self.selected(tensor, batch_index, mode)
            summary = self.summarize(selected, score_rows=score_rows)
            if name == "pre_softmax_scores":
                root_invalid = summary["nan_count"] + summary["posinf_count"] + summary["all_nonfinite_rows"]
            else:
                root_invalid = summary["nonfinite_count"]
            self.events.append({
                "model": protocol.MODEL,
                "case_index": int(row["case_index"]),
                "event_order": event["order"],
                "event_name": name,
                "event_class": event["event_class"],
                "selection_mode": mode,
                "position_count": len(self.positions[batch_index]),
                "root_invalid_count": int(root_invalid),
                **summary,
            })


def register_hooks(model: Any, collector: Collector) -> list[Any]:
    target = get_layers(model)[protocol.TARGET_LAYER]
    attention = target.self_attn
    handles: list[Any] = []

    def simple_output(name: str, mode: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            collector.record(name, tensor, mode)
        return hook

    def o_input(_module: Any, inputs: Any) -> None:
        collector.record("o_proj_input", inputs[0], "query")

    def attention_start(_module: Any, _inputs: Any) -> None:
        if collector.target_attention_active:
            raise RuntimeError("Nested target attention activation")
        collector.target_attention_active = True

    def attention_end(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, tuple) or output[1] is None:
            raise RuntimeError("Target attention weights unavailable")
        collector.record("attention_weights_fp16", output[1], "attention", score_rows=True)
        collector.target_attention_active = False

    handles.append(target.input_layernorm.register_forward_hook(simple_output("attention_norm", "query")))
    handles.append(attention.register_forward_pre_hook(attention_start))
    handles.append(attention.q_proj.register_forward_hook(simple_output("q_proj_queries", "query")))
    handles.append(attention.k_proj.register_forward_hook(simple_output("k_proj_prefix", "prefix")))
    handles.append(attention.v_proj.register_forward_hook(simple_output("v_proj_prefix", "prefix")))
    handles.append(attention.o_proj.register_forward_pre_hook(o_input))
    handles.append(attention.o_proj.register_forward_hook(simple_output("o_proj_output", "query")))
    handles.append(attention.register_forward_hook(attention_end))
    return handles


def replay_scores(model: Any, input_ids: torch.Tensor, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, return_dict=True)
    logits = output.logits
    result: list[dict[str, Any]] = []
    for batch_index, row in enumerate(rows):
        candidate_positions = [int(value) for value in row["candidate_positions"]]
        suffix_positions = [int(value) for value in row["suffix_positions"]]
        positions = [*candidate_positions, *suffix_positions]
        prediction_index = torch.tensor([value - 1 for value in positions], device=logits.device, dtype=torch.long)
        targets = torch.tensor([int(row["input_ids"][value]) for value in positions],
                               device=logits.device, dtype=torch.long)
        selected = logits[batch_index].index_select(0, prediction_index).float()
        log_probs = (selected.gather(1, targets.unsqueeze(1)).squeeze(1)
                     - torch.logsumexp(selected, dim=-1)).detach().cpu().tolist()
        candidate = sum(float(value) for value in log_probs[:len(candidate_positions)])
        suffix_values = [float(value) for value in log_probs[len(candidate_positions):]]
        suffix = sum(suffix_values) / len(suffix_values) if suffix_values else 0.0
        total = candidate + suffix
        result.append({
            "candidate_logp": safe_float(candidate),
            "suffix_mean_logp": safe_float(suffix),
            "total_score": safe_float(total),
            "candidate_finite": math.isfinite(candidate),
            "suffix_finite": math.isfinite(suffix),
            "total_finite": math.isfinite(total),
        })
    del output, logits
    return result


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["passed"]:
        raise RuntimeError("Phase1129 protocol is not authorized")
    cases = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.deepseek7b.jsonl")
    source_cases = source_protocol.read_jsonl(source_protocol.OUT_ROOT / "scan" / protocol.MODEL / "cases.jsonl")
    source_by_case = {int(row["case_index"]): row for row in source_cases}
    if source_protocol.digest(source_cases) != prereg["case_policy"]["phase1128_case_result_digest"]:
        raise RuntimeError("Phase1128 case-result digest mismatch")

    started = time.time()
    model = None
    handles: list[Any] = []
    original_softmax = functional.softmax
    collector = Collector()
    try:
        model, _tokenizer, device, placement = load_fp16(protocol.MODEL)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("Phase1129 requires unquantized FP16")
        if len(get_layers(model)) != 28 or getattr(model.config, "_attn_implementation", None) != "eager":
            raise RuntimeError("Frozen Qwen2 eager-attention implementation not active")
        handles = register_hooks(model, collector)

        def audited_softmax(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if collector.target_attention_active:
                collector.record("pre_softmax_scores", input_tensor, "attention", score_rows=True)
            result = original_softmax(input_tensor, *args, **kwargs)
            if collector.target_attention_active:
                collector.record("softmax_output_fp32", result, "attention", score_rows=True)
            return result

        functional.softmax = audited_softmax
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in cases:
            by_length[len(row["input_ids"])].append(row)
        rerun_by_case: dict[int, dict[str, Any]] = {}
        completed = 0
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), int(prereg["case_policy"]["batch_size"])):
                    batch = panel[start:start + int(prereg["case_policy"]["batch_size"])]
                    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
                    collector.begin(batch)
                    scores = replay_scores(model, input_ids, batch)
                    collector.end()
                    for row, score in zip(batch, scores):
                        rerun_by_case[int(row["case_index"])] = score
                    completed += len(batch)
                    print(json.dumps({"phase": protocol.PHASE, "completed": completed, "total": len(cases)}), flush=True)
                    del input_ids

        collector.events.sort(key=lambda row: (row["case_index"], row["event_order"]))
        by_case_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in collector.events:
            by_case_events[int(event["case_index"])].append(event)
        details: list[dict[str, Any]] = []
        for row in sorted(cases, key=lambda value: int(value["case_index"])):
            case_index = int(row["case_index"])
            source = source_by_case[case_index]
            rerun = rerun_by_case[case_index]
            events = by_case_events[case_index]
            root = next((event for event in events[1:] if int(event["root_invalid_count"]) > 0), None)
            details.append({
                "model": protocol.MODEL,
                "case_index": case_index,
                "partition": row["partition"],
                "route": row["route"],
                "source_candidate_finite": bool(source["rerun_candidate_finite"]),
                "source_suffix_finite": bool(source["rerun_suffix_finite"]),
                "source_total_finite": bool(source["rerun_total_finite"]),
                "rerun_candidate_finite": rerun["candidate_finite"],
                "rerun_suffix_finite": rerun["suffix_finite"],
                "rerun_total_finite": rerun["total_finite"],
                "candidate_finite_parity": bool(source["rerun_candidate_finite"]) == rerun["candidate_finite"],
                "suffix_finite_parity": bool(source["rerun_suffix_finite"]) == rerun["suffix_finite"],
                "total_finite_parity": bool(source["rerun_total_finite"]) == rerun["total_finite"],
                "event_count": len(events),
                "event_count_expected": len(events) == len(protocol.EVENT_REGISTRY),
                "root_event_name": root["event_name"] if root else None,
                "root_event_class": root["event_class"] if root else None,
                "root_event_order": root["event_order"] if root else None,
            })

        summary_core = {
            "schema_version": "phase1129_ds7b_attention_numeric_refinement_scan.v1",
            "phase": protocol.PHASE,
            "model": protocol.MODEL,
            "protocol_digest": prereg["protocol_digest"],
            "case_count": len(details),
            "event_count": len(collector.events),
            "precision": precision,
            "placement": placement,
            "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
            "target_layer": protocol.TARGET_LAYER,
            "elapsed_seconds": time.time() - started,
            "event_digest": protocol.digest(collector.events),
            "case_detail_digest": protocol.digest(details),
        }
        summary = dict(summary_core)
        summary["summary_digest"] = protocol.digest(summary_core)
        output_root = protocol.OUT_ROOT / "scan" / protocol.MODEL
        protocol.write_jsonl(output_root / "events.jsonl", collector.events)
        protocol.write_jsonl(output_root / "cases.jsonl", details)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        functional.softmax = original_softmax
        for handle in handles:
            handle.remove()
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
