#!/usr/bin/env python3
"""Generate and replay every Phase333 case to map tokenwise dynamic paths."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import answer_segment, target_match  # noqa: E402
import phase330_global_atlas_survey as phase330_survey  # noqa: E402
from phase333_dynamic_case_bank import ROUND_DEFAULT  # noqa: E402


PHASE = "Phase333"
SCHEMA_VERSION = "11.0.0"
OUT = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas"
COMPONENTS = (
    "residual_input", "normalized_input", "attention_output", "mlp_output", "residual_output",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


BASE_FIELDS = [
    pa.field("schema_version", pa.string()), pa.field("phase_id", pa.string()),
    pa.field("model", pa.string()), pa.field("case_id", pa.string()),
    pa.field("semantic_case_id", pa.string()), pa.field("item_id", pa.string()),
    pa.field("family_id", pa.string()), pa.field("mechanism_id", pa.string()),
    pa.field("cohort", pa.string()), pa.field("item_index", pa.int64()),
    pa.field("split", pa.string()), pa.field("template_id", pa.string()),
    pa.field("interface", pa.string()), pa.field("answer_phase", pa.string()),
    pa.field("target_class", pa.string()),
]
TOKEN_SCHEMA = pa.schema([*BASE_FIELDS,
    pa.field("created_at", pa.string()), pa.field("generated_step", pa.int64()),
    pa.field("generated_token_id", pa.int64()), pa.field("generated_token_text", pa.string()),
    pa.field("target_first_token_id", pa.int64()), pa.field("target_logprob", pa.float64()),
    pa.field("target_margin", pa.float64()), pa.field("target_rank", pa.int64()),
    pa.field("top_competitor_token_id", pa.int64()), pa.field("top_competitor_text", pa.string()),
    pa.field("functional_events", pa.string()), pa.field("evidence_level", pa.string()),
])
PATH_SCHEMA = pa.schema([*BASE_FIELDS,
    pa.field("created_at", pa.string()), pa.field("generated_step", pa.int64()),
    pa.field("functional_events", pa.string()), pa.field("component_type", pa.string()),
    pa.field("component_layer", pa.int64()), pa.field("relative_depth", pa.float64()),
    pa.field("projection", pa.float64()), pa.field("activation_norm", pa.float64()),
    pa.field("evidence_level", pa.string()),
])


class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(json_safe(rows), schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table, row_group_size=32768)
        self.row_count += len(rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def continuation_ids(loaded: Any, case: dict[str, Any], text: str) -> list[int]:
    value = (" " + text) if case["interface"] == "raw_completion" else text
    ids = loaded.tokenizer(value, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Cannot tokenize {text!r}")
    return [int(item) for item in ids]


def output_base(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "model": case["model"],
        "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "item_id": case["item_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "cohort": case["cohort"],
        "item_index": case["item_index"],
        "split": case["split"],
        "template_id": case["template_id"],
        "interface": case["interface"],
        "answer_phase": case["answer_phase"],
        "target_class": case["target_class"],
    }


def word_first_step(tokenizer: Any, ids: list[int], words: list[str]) -> int:
    lowered = [word.lower() for word in words]
    for step in range(len(ids)):
        text = tokenizer.decode(ids[: step + 1], skip_special_tokens=True).lower()
        if any(re.search(rf"\b{re.escape(word)}\b", text) for word in lowered):
            return step
    return -1


def functional_events(
    tokenizer: Any, ids: list[int], margins: list[float], case: dict[str, Any]
) -> dict[str, int]:
    formation = next((index for index, value in enumerate(margins) if value > 0), -1)
    overtake = -1
    if formation >= 0:
        overtake = next((index for index in range(formation + 1, len(margins)) if margins[index] < 0), -1)
    eos = tokenizer.eos_token_id
    stop = ids.index(eos) if eos is not None and eos in ids else -1
    return {
        "answer_start": 0 if ids else -1,
        "target_pressure_formation": formation,
        "competitor_overtake": overtake,
        "target_first_appearance": word_first_step(tokenizer, ids, case["target_aliases"]),
        "error_first_appearance": word_first_step(tokenizer, ids, case["distractors"]),
        "stop": stop,
        "final_readout": len(ids) - 1,
    }


def event_order(events: dict[str, int]) -> list[str]:
    return [
        key for key, _value in sorted(
            ((key, value) for key, value in events.items() if value >= 0),
            key=lambda item: (item[1], item[0]),
        )
    ]


def layer_norm_module(layer: Any) -> Any:
    for name in ("input_layernorm", "input_layer_norm", "ln_1"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise TypeError(f"Cannot find input normalization on {type(layer).__name__}")


@torch.inference_mode()
def replay_case(
    loaded: Any,
    case: dict[str, Any],
    prompt_ids: torch.Tensor,
    generated_ids: list[int],
    path_sink: ParquetSink,
    token_sink: ParquetSink,
    discovery_peaks: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not generated_ids:
        generated_ids = [int(loaded.tokenizer.eos_token_id or loaded.tokenizer.pad_token_id)]
    suffix = torch.tensor([generated_ids], dtype=prompt_ids.dtype, device=loaded.input_device)
    input_ids = torch.cat([prompt_ids.to(loaded.input_device), suffix], dim=1)
    attention_mask = torch.ones_like(input_ids, device=loaded.input_device)
    prompt_length = int(prompt_ids.shape[1])
    target_ids = continuation_ids(loaded, case, case["target"])
    distractor_ids = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
    weight = loaded.model.get_output_embeddings().weight.detach().float()
    direction = weight[target_ids[0]] - weight[distractor_ids].mean(dim=0)
    direction = (direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)).to(loaded.input_device)
    layers = get_layers(loaded.model)
    captures: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        norm = layer_norm_module(layer)

        def residual_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("residual_input", idx)] = inputs[0].detach()

        def normalized_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("normalized_input", idx)] = component_tensor(output).detach()

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("attention_output", idx)] = component_tensor(output).detach()

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("mlp_output", idx)] = component_tensor(output).detach()

        def residual_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("residual_output", idx)] = component_tensor(output).detach()

        handles.extend([
            layer.register_forward_pre_hook(residual_pre),
            norm.register_forward_hook(normalized_post),
            layer.self_attn.register_forward_hook(attention_post),
            layer.mlp.register_forward_hook(mlp_post),
            layer.register_forward_hook(residual_post),
        ])
    try:
        output = loaded.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    decision_positions = [prompt_length - 1 + step for step in range(len(generated_ids))]
    logits = output.logits[0, decision_positions].detach().float()
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logits = logits[:, target_ids[0]]
    distractor_logits = logits[:, distractor_ids]
    margins = (target_logits - distractor_logits.max(dim=1).values).cpu().tolist()
    events = functional_events(loaded.tokenizer, generated_ids, margins, case)
    event_at_step: dict[int, list[str]] = defaultdict(list)
    for name, step in events.items():
        if step >= 0:
            event_at_step[step].append(name)
    created_at = now()
    token_rows = []
    for step, token_id in enumerate(generated_ids):
        target_logit = float(target_logits[step].item())
        target_rank = 1 + int((logits[step] > target_logit).sum().item())
        competitor_id = int(torch.argmax(logits[step]).item())
        token_rows.append({
            **output_base(case),
            "created_at": created_at,
            "generated_step": step,
            "generated_token_id": int(token_id),
            "generated_token_text": loaded.tokenizer.decode([token_id], skip_special_tokens=False),
            "target_first_token_id": target_ids[0],
            "target_logprob": round(float(log_probs[step, target_ids[0]].item()), 7),
            "target_margin": round(float(margins[step]), 7),
            "target_rank": target_rank,
            "top_competitor_token_id": competitor_id,
            "top_competitor_text": loaded.tokenizer.decode([competitor_id], skip_special_tokens=False),
            "functional_events": ",".join(sorted(event_at_step.get(step, []))),
            "evidence_level": "L2_natural_dynamic_readout",
        })
    token_sink.write(token_rows)

    path_rows = []
    peak_by_layer = []
    selected_step = events["target_pressure_formation"]
    if selected_step < 0:
        selected_step = 0
    for layer_index in range(len(layers)):
        for component_type in COMPONENTS:
            tensor = captures[(component_type, layer_index)][0, decision_positions].detach().float()
            projections = tensor @ direction.float()
            norms = torch.linalg.vector_norm(tensor, dim=1)
            for step in range(len(generated_ids)):
                path_rows.append({
                    **output_base(case),
                    "created_at": created_at,
                    "generated_step": step,
                    "functional_events": ",".join(sorted(event_at_step.get(step, []))),
                    "component_type": component_type,
                    "component_layer": layer_index,
                    "relative_depth": round(layer_index / max(1, len(layers) - 1), 7),
                    "projection": round(float(projections[step].item()), 7),
                    "activation_norm": round(float(norms[step].item()), 7),
                    "evidence_level": "L2_natural_dynamic_path",
                })
            if component_type == "residual_output":
                peak_by_layer.append((layer_index, float(projections[selected_step].item())))
    path_sink.write(path_rows)
    if case["split"] == "discovery":
        ordered = sorted(peak_by_layer)
        writes = [
            (layer, value - (ordered[index - 1][1] if index else 0.0))
            for index, (layer, value) in enumerate(ordered)
        ]
        peak_layer, peak_value = max(writes, key=lambda item: item[1])
        discovery_peaks[(case["mechanism_id"], case["interface"])].append({
            "case_id": case["case_id"],
            "item_index": case["item_index"],
            "template_id": case["template_id"],
            "selected_event_step": selected_step,
            "target_pressure_formation_step": events["target_pressure_formation"],
            "peak_layer": peak_layer,
            "relative_peak_depth": peak_layer / max(1, len(layers) - 1),
            "peak_projection": peak_value,
            "selection_metric": "largest_positive_residual_write_increment",
            "event_order": event_order(events),
        })
    event_row = {
        **output_base(case),
        "created_at": created_at,
        **{f"{key}_step": value for key, value in events.items()},
        "event_order": json.dumps(event_order(events)),
        "generated_token_count": len(generated_ids),
        "target_pressure_formed": events["target_pressure_formation"] >= 0,
        "competitor_overtake_observed": events["competitor_overtake"] >= 0,
        "target_appeared": events["target_first_appearance"] >= 0,
        "error_appeared": events["error_first_appearance"] >= 0,
        "stop_observed": events["stop"] >= 0,
    }
    del output, logits, log_probs, captures, input_ids, attention_mask
    return event_row, {"target_ids": target_ids, "margins": margins}


def freeze_block_plans(
    model: str, discovery_peaks: dict[tuple[str, str], list[dict[str, Any]]], layer_count: int
) -> list[dict[str, Any]]:
    rows = []
    for (mechanism, interface), values in sorted(discovery_peaks.items()):
        centers = [int(row["peak_layer"]) for row in values]
        center = int(round(median(centers)))
        event_steps = [int(row["selected_event_step"]) for row in values]
        event_step = int(round(median(event_steps)))
        windows = {}
        for length in (1, 2, 4):
            start = max(0, min(layer_count - length, center - (length - 1) // 2))
            windows[str(length)] = list(range(start, start + length))
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "reasoning_constraint",
            "mechanism_id": mechanism,
            "interface": interface,
            "selection_split": "discovery_only",
            "discovery_case_count": len(values),
            "discovery_object_count": len({row["item_index"] for row in values}),
            "selected_functional_event": "target_pressure_formation_or_answer_start_fallback",
            "layer_selection_metric": "largest_positive_residual_write_increment",
            "median_selected_event_step": event_step,
            "median_peak_layer": center,
            "median_relative_peak_depth": round(center / max(1, layer_count - 1), 7),
            "block_windows": windows,
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
        })
    return rows


def refreeze_existing(model: str, round_name: str) -> dict[str, Any]:
    model_dir = OUT / round_name / "survey" / model
    events = read_jsonl(model_dir / "event_rows.jsonl")
    discovery = {row["case_id"]: row for row in events if row["split"] == "discovery"}
    selected_steps = {
        case_id: (
            int(row["target_pressure_formation_step"])
            if int(row["target_pressure_formation_step"]) >= 0 else 0
        )
        for case_id, row in discovery.items()
    }
    projections: dict[str, dict[int, float]] = defaultdict(dict)
    parquet = pq.ParquetFile(model_dir / "dynamic_path_rows.parquet")
    columns = ["case_id", "component_type", "generated_step", "component_layer", "projection"]
    max_layer = 0
    for batch in parquet.iter_batches(batch_size=65536, columns=columns):
        for row in batch.to_pylist():
            case_id = row["case_id"]
            if case_id not in discovery or row["component_type"] != "residual_output":
                continue
            if int(row["generated_step"]) != selected_steps[case_id]:
                continue
            if row["projection"] is None or not math.isfinite(float(row["projection"])):
                continue
            layer = int(row["component_layer"])
            projections[case_id][layer] = float(row["projection"])
            max_layer = max(max_layer, layer)
    discovery_peaks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case_id, by_layer in projections.items():
        ordered = sorted(by_layer.items())
        writes = [
            (layer, value - (ordered[index - 1][1] if index else 0.0))
            for index, (layer, value) in enumerate(ordered)
        ]
        peak_layer, peak_value = max(writes, key=lambda item: item[1])
        row = discovery[case_id]
        discovery_peaks[(row["mechanism_id"], row["interface"])].append({
            "case_id": case_id,
            "item_index": row["item_index"],
            "template_id": row["template_id"],
            "selected_event_step": selected_steps[case_id],
            "target_pressure_formation_step": int(row["target_pressure_formation_step"]),
            "peak_layer": peak_layer,
            "relative_peak_depth": peak_layer / max(1, max_layer),
            "peak_projection": peak_value,
            "event_order": json.loads(row["event_order"]),
            "selection_metric": "largest_positive_residual_write_increment",
        })
    plans = freeze_block_plans(model, discovery_peaks, max_layer + 1)
    write_jsonl(model_dir / "block_plans.jsonl", plans)
    return {
        "phase_id": PHASE,
        "model": model,
        "block_plan_count": len(plans),
        "selection_metric": "largest_positive_residual_write_increment",
        "valid": len(plans) == 6,
    }


@torch.inference_mode()
def run_model(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "survey" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    cases = [row for row in read_jsonl(root / "phase333_registered_cases.jsonl") if row["model"] == model]
    path_sink = ParquetSink(model_dir / "dynamic_path_rows.parquet", PATH_SCHEMA)
    token_sink = ParquetSink(model_dir / "token_rows.parquet", TOKEN_SCHEMA)
    baseline_rows = []
    event_rows = []
    discovery_peaks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    loaded = None
    try:
        loaded = load_probe_model(model)
        tokenizer = loaded.tokenizer
        for index, case in enumerate(cases, 1):
            encoded = tokenizer(
                case["prompt"], return_tensors="pt", truncation=True, max_length=256,
                add_special_tokens=bool(case["tokenization_add_special_tokens"]),
            )
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            generated = loaded.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            suffix = generated[0, encoded["input_ids"].shape[1] :]
            ids = [int(value) for value in suffix.tolist()]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            segment = answer_segment(text)
            event_row, _details = replay_case(
                loaded, case, encoded["input_ids"], ids, path_sink, token_sink, discovery_peaks,
            )
            event_rows.append(event_row)
            baseline_rows.append({
                **output_base(case),
                "created_at": now(),
                "prompt_token_count": int(encoded["input_ids"].shape[1]),
                "generated_text": text,
                "generated_token_ids": json.dumps(ids),
                "generated_token_count": len(ids),
                "answer_segment": segment,
                "target_answer_segment_match": target_match(segment, case["target_aliases"]),
                "protocol_success_answer_segment": phase330_survey.protocol_ok(case, segment),
                "behavior_success": (
                    target_match(segment, case["target_aliases"])
                    and phase330_survey.protocol_ok(case, segment)
                ),
                "eos_emitted": tokenizer.eos_token_id in ids,
            })
            if index % 8 == 0:
                print(json.dumps({
                    "quality_only": True,
                    "model": model,
                    "dynamic_cases": index,
                    "total_cases": len(cases),
                    "token_rows": token_sink.row_count,
                    "path_rows": path_sink.row_count,
                }), flush=True)
        plans = freeze_block_plans(model, discovery_peaks, len(get_layers(loaded.model)))
        write_jsonl(model_dir / "baseline_rows.jsonl", baseline_rows)
        write_jsonl(model_dir / "event_rows.jsonl", event_rows)
        write_jsonl(model_dir / "block_plans.jsonl", plans)
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "case_count": len(cases),
            "baseline_generation_count": len(baseline_rows),
            "event_row_count": len(event_rows),
            "token_row_count": token_sink.row_count,
            "dynamic_path_row_count": path_sink.row_count,
            "block_plan_count": len(plans),
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(cases) == 216 and len(event_rows) == 216 and len(plans) == 6,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        path_sink.close()
        token_sink.close()
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=("qwen3", "glm4", "deepseek7b"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--refreeze-block-plans", action="store_true")
    args = parser.parse_args()
    result = (
        refreeze_existing(args.model, args.round)
        if args.refreeze_block_plans
        else run_model(args.model, args.round, args.max_new_tokens)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
