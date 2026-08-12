#!/usr/bin/env python3
"""Run Phase1076 frozen pre-o_proj head-coalition interventions."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1076_polarity_head_causal_protocol as protocol


PAIR_BATCH_SIZE = {"qwen3": 4, "glm4": 2}
CONDITIONINGS = ("all", "behavior_conditioned")


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(
        len(rows), dtype=torch.long, device=device
    )
    positions = torch.zeros(
        len(rows), dtype=torch.long, device=device
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = int(
            row["role_positions"]["answer_boundary"]
        )
    return input_ids, attention_mask, lengths, positions


class BaselineHeadCapture:
    """Capture clean pre-o_proj heads at row-specific answer positions."""

    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        head_count: int,
    ):
        self.layers = layers
        self.depths = sorted(set(int(value) for value in depths))
        self.head_count = int(head_count)
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(_module, args):
            if self.positions is None:
                raise RuntimeError("capture positions are not set")
            value = args[0]
            batch = torch.arange(
                value.shape[0], device=value.device
            )
            positions = self.positions.to(value.device)
            selected = value[batch, positions, :]
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("pre-o_proj head width drift")
            self.values[depth] = selected.reshape(
                selected.shape[0],
                self.head_count,
                selected.shape[-1] // self.head_count,
            ).detach().clone()
            self.counts[depth] += 1

        return hook

    def register(self) -> None:
        for depth in self.depths:
            self.handles.append(
                self.layers[
                    depth - 1
                ].self_attn.o_proj.register_forward_pre_hook(
                    self._hook(depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        missing = sorted(set(self.depths) - set(self.values))
        repeated = {
            depth: count
            for depth, count in self.counts.items()
            if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"baseline capture drift missing={missing} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None


class HeadCoalitionIntervention:
    """Swap or zero a frozen multi-depth head coalition."""

    def __init__(
        self,
        layers: list[Any],
        head_count: int,
        positions: torch.Tensor,
        baseline: dict[int, torch.Tensor],
        donor_rows: torch.Tensor,
        events: list[dict[str, Any]],
        mode: str,
    ):
        self.layers = layers
        self.head_count = int(head_count)
        self.positions = positions
        self.baseline = baseline
        self.donor_rows = donor_rows
        self.mode = mode
        self.heads_by_depth: dict[int, list[int]] = defaultdict(list)
        for event in events:
            self.heads_by_depth[int(event["depth"])].append(
                int(event["head"])
            )
        self.heads_by_depth = {
            depth: sorted(set(heads))
            for depth, heads in self.heads_by_depth.items()
        }
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        heads = self.heads_by_depth[depth]

        def hook(_module, args):
            value = args[0]
            patched = value.clone()
            batch = torch.arange(
                value.shape[0], device=value.device
            )
            positions = self.positions.to(value.device)
            selected = patched[batch, positions, :]
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("pre-o_proj head width drift")
            shaped = selected.reshape(
                selected.shape[0],
                self.head_count,
                selected.shape[-1] // self.head_count,
            ).clone()
            if self.mode == "swap":
                clean = self.baseline[depth].to(
                    device=value.device, dtype=value.dtype
                )
                donors = self.donor_rows.to(value.device)
                shaped[:, heads, :] = clean[
                    donors
                ][:, heads, :]
            elif self.mode == "zero":
                shaped[:, heads, :] = 0
            else:
                raise RuntimeError(
                    f"unknown intervention mode: {self.mode}"
                )
            patched[batch, positions, :] = shaped.reshape(
                selected.shape
            )
            self.counts[depth] += 1
            return (patched,) + tuple(args[1:])

        return hook

    def register(self) -> None:
        for depth in sorted(self.heads_by_depth):
            self.handles.append(
                self.layers[
                    depth - 1
                ].self_attn.o_proj.register_forward_pre_hook(
                    self._hook(depth)
                )
            )

    def validate(self) -> None:
        missing = sorted(
            set(self.heads_by_depth) - set(self.counts)
        )
        repeated = {
            depth: count
            for depth, count in self.counts.items()
            if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"intervention hook drift missing={missing} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def candidate_margin(
    logits: torch.Tensor,
    row: dict[str, Any],
) -> tuple[float, dict[str, float], bool]:
    scores = {}
    for class_name in ("b0", "b1"):
        ids = torch.tensor(
            row["candidate_first_token_ids"][class_name],
            dtype=torch.long,
            device=logits.device,
        )
        scores[class_name] = float(logits[ids].max().item())
    expected = str(row["expected_class"])
    other = "b1" if expected == "b0" else "b0"
    margin = scores[expected] - scores[other]
    finite = (
        all(math.isfinite(value) for value in scores.values())
        and math.isfinite(margin)
    )
    return margin, scores, finite


def final_logits(
    output,
    lengths: torch.Tensor,
) -> torch.Tensor:
    axes = torch.arange(
        output.logits.shape[0], device=output.logits.device
    )
    return output.logits[
        axes, (lengths - 1).to(output.logits.device), :
    ].float()


def ordered_pairs(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    pairs = []
    for pair_id, values in sorted(grouped.items()):
        contrast = str(values[0]["contrast"])
        order = protocol.TASKS_BY_CONTRAST[contrast]
        by_task = {str(row["task"]): row for row in values}
        if len(values) != 2 or set(by_task) != set(order):
            raise RuntimeError(f"incomplete pair: {pair_id}")
        pairs.append([by_task[task] for task in order])
    return pairs


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    decision = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_decision.json"
    )
    if (
        decision["protocol_digest"] != prereg["protocol_digest"]
        or protocol.digest({
            key: value
            for key, value in decision.items()
            if key != "decision_digest"
        })
        != decision["decision_digest"]
    ):
        raise RuntimeError("Phase1076 behavior decision drift")
    if (
        not decision["should_run_causal_validation"]
        or model_name not in decision["authorized_models"]
    ):
        raise RuntimeError(
            f"Phase1076 causal scan not authorized for {model_name}"
        )
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    pairs = ordered_pairs(rows)
    if len(pairs) != prereg["pair_count_per_model"]:
        raise RuntimeError("Phase1076 pair count drift")
    behavior_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "behavior"
        / model_name
        / "candidate_behavior.jsonl"
    )
    behavior_hit = {
        str(row["record_id"]): bool(row["candidate_hit"])
        for row in behavior_rows
    }
    head_sets = prereg["head_sets"][model_name]
    selected_events = head_sets["selected"]
    control_events = head_sets["matched_controls"]
    all_depths = sorted({
        int(event["depth"])
        for event in selected_events + control_events
    })

    started = time.time()
    model = tokenizer = None
    records = []
    finite_count = 0
    margin_count = 0
    pair_conditioned_count = 0
    hook_calls = CounterLike()
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
        head_count = int(model.config.num_attention_heads)
        if max(all_depths) > len(layers):
            raise RuntimeError("frozen head depth exceeds model")
        if any(
            int(event["head"]) >= head_count
            for event in selected_events + control_events
        ):
            raise RuntimeError("frozen head index exceeds model")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos token")

        with torch.inference_mode():
            for pair_batch in chunks(
                pairs, PAIR_BATCH_SIZE[model_name]
            ):
                batch = [
                    row for pair in pair_batch for row in pair
                ]
                input_ids, attention_mask, lengths, positions = (
                    pad_rows(batch, int(pad_id), device)
                )
                donor_rows = torch.tensor(
                    [
                        index + 1 if index % 2 == 0 else index - 1
                        for index in range(len(batch))
                    ],
                    dtype=torch.long,
                    device=device,
                )
                capture = BaselineHeadCapture(
                    layers, all_depths, head_count
                )
                capture.register()
                try:
                    capture.begin(positions)
                    baseline_output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    capture.validate()
                    baseline_heads = {
                        depth: value.clone()
                        for depth, value in capture.values.items()
                    }
                finally:
                    capture.close()
                hook_calls.add(
                    "baseline",
                    sum(capture.counts.values()),
                )
                outputs = {
                    "baseline": final_logits(
                        baseline_output, lengths
                    )
                }
                del baseline_output

                definitions = {
                    "selected_swap": (
                        selected_events, "swap"
                    ),
                    "control_swap": (
                        control_events, "swap"
                    ),
                    "selected_zero": (
                        selected_events, "zero"
                    ),
                    "control_zero": (
                        control_events, "zero"
                    ),
                }
                for name in protocol.INTERVENTIONS:
                    events, mode = definitions[name]
                    intervention = HeadCoalitionIntervention(
                        layers,
                        head_count,
                        positions,
                        baseline_heads,
                        donor_rows,
                        events,
                        mode,
                    )
                    intervention.register()
                    try:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        intervention.validate()
                        outputs[name] = final_logits(
                            output, lengths
                        )
                    finally:
                        intervention.close()
                    hook_calls.add(
                        name,
                        sum(intervention.counts.values()),
                    )
                    del output

                pair_conditioned = []
                for pair in pair_batch:
                    conditioned = all(
                        behavior_hit[str(row["record_id"])]
                        for row in pair
                    )
                    pair_conditioned.extend(
                        [conditioned, conditioned]
                    )
                    pair_conditioned_count += int(conditioned)
                for index, row in enumerate(batch):
                    baseline_margin, baseline_scores, baseline_finite = (
                        candidate_margin(
                            outputs["baseline"][index], row
                        )
                    )
                    intervention_margins = {}
                    intervention_scores = {}
                    intervention_finite = {}
                    drops = {}
                    for name in protocol.INTERVENTIONS:
                        margin, scores, finite = candidate_margin(
                            outputs[name][index], row
                        )
                        intervention_margins[name] = margin
                        intervention_scores[name] = scores
                        intervention_finite[name] = finite
                        drops[name] = baseline_margin - margin
                    finite = bool(
                        baseline_finite
                        and all(intervention_finite.values())
                        and all(
                            math.isfinite(value)
                            for value in drops.values()
                        )
                    )
                    margin_count += 1 + len(protocol.INTERVENTIONS)
                    finite_count += int(baseline_finite)
                    finite_count += sum(
                        int(value)
                        for value in intervention_finite.values()
                    )
                    records.append({
                        "schema_version": (
                            "phase1076_head_causal_record.v1"
                        ),
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "record_id": row["record_id"],
                        "pair_id": row["pair_id"],
                        "factor_id": row["factor_id"],
                        "contrast": row["contrast"],
                        "task": row["task"],
                        "path": row["path"],
                        "layout": row["layout"],
                        "template_index": row["template_index"],
                        "replicate": row["replicate"],
                        "orientation": row["orientation"],
                        "lexical_branch": row["lexical_branch"],
                        "expected_answer": row["expected_answer"],
                        "expected_class": row["expected_class"],
                        "behavior_hit": behavior_hit[
                            str(row["record_id"])
                        ],
                        "pair_behavior_conditioned": bool(
                            pair_conditioned[index]
                        ),
                        "baseline_margin": (
                            baseline_margin
                            if math.isfinite(baseline_margin)
                            else None
                        ),
                        "baseline_class_scores": {
                            key: (
                                value
                                if math.isfinite(value)
                                else None
                            )
                            for key, value in baseline_scores.items()
                        },
                        "intervention_margins": {
                            key: (
                                value
                                if math.isfinite(value)
                                else None
                            )
                            for key, value in (
                                intervention_margins.items()
                            )
                        },
                        "intervention_class_scores": {
                            name: {
                                key: (
                                    value
                                    if math.isfinite(value)
                                    else None
                                )
                                for key, value in scores.items()
                            }
                            for name, scores in (
                                intervention_scores.items()
                            )
                        },
                        "margin_drops": {
                            key: (
                                value
                                if math.isfinite(value)
                                else None
                            )
                            for key, value in drops.items()
                        },
                        "all_finite": finite,
                    })
                del (
                    input_ids,
                    attention_mask,
                    lengths,
                    positions,
                    donor_rows,
                    baseline_heads,
                )
                for value in outputs.values():
                    del value

        summary = {
            "schema_version": "phase1076_causal_scan_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "behavior_decision_digest": decision[
                "decision_digest"
            ],
            "precision": precision,
            "placement": placement,
            "case_count": len(records),
            "pair_count": len(pairs),
            "behavior_conditioned_pair_count": (
                pair_conditioned_count
            ),
            "candidate_margin_finite_rate": (
                finite_count / margin_count if margin_count else 0.0
            ),
            "selected_events": selected_events,
            "control_events": control_events,
            "unique_intervention_depths": all_depths,
            "head_count": head_count,
            "hook_call_counts": hook_calls.values,
            "raw_tensor_dumps": False,
            "elapsed_seconds": float(time.time() - started),
        }
        out_dir = protocol.OUT_ROOT / "causal" / model_name
        protocol.write_jsonl(
            out_dir / "causal_records.jsonl", records
        )
        protocol.write_json(out_dir / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "case_count": len(records),
            "candidate_margin_finite_rate": summary[
                "candidate_margin_finite_rate"
            ],
            "behavior_conditioned_pair_count": (
                pair_conditioned_count
            ),
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


class CounterLike:
    """Small JSON-friendly counter without implicit key creation."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def add(self, key: str, value: int) -> None:
        self.values[key] = self.values.get(key, 0) + int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=protocol.MODELS, required=True
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
