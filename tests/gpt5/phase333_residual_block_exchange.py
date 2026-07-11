#!/usr/bin/env python3
"""Run heldout dynamic residual-block exchanges for Phase333."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import answer_segment, target_match  # noqa: E402
import phase330_global_atlas_survey as phase330_survey  # noqa: E402
from phase333_dynamic_case_bank import BLOCK_CONDITIONS, ROUND_DEFAULT  # noqa: E402
from phase333_dynamic_survey import continuation_ids, output_base  # noqa: E402


PHASE = "Phase333"
SCHEMA_VERSION = "11.0.0"
OUT = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(json_safe(rows)), path, compression="zstd", row_group_size=32768)


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def replace_output(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def event_step(row: dict[str, Any]) -> int:
    value = int(row["target_pressure_formation_step"])
    return value if value >= 0 else 0


def encoded_prompt(loaded: Any, case: dict[str, Any]) -> dict[str, torch.Tensor]:
    encoded = loaded.tokenizer(
        case["prompt"], return_tensors="pt", truncation=True, max_length=256,
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
    )
    return {key: value.to(loaded.input_device) for key, value in encoded.items()}


@torch.inference_mode()
def capture_block_values(
    loaded: Any,
    case: dict[str, Any],
    generated_ids: list[int],
    source_layers: list[int],
    target_layers: list[int],
    selected_step: int,
) -> dict[int, torch.Tensor]:
    encoded = encoded_prompt(loaded, case)
    if not generated_ids:
        generated_ids = [int(loaded.tokenizer.eos_token_id or loaded.tokenizer.pad_token_id)]
    suffix = torch.tensor([generated_ids], dtype=encoded["input_ids"].dtype, device=loaded.input_device)
    input_ids = torch.cat([encoded["input_ids"], suffix], dim=1)
    attention_mask = torch.ones_like(input_ids)
    absolute = min(input_ids.shape[1] - 1, int(encoded["input_ids"].shape[1]) - 1 + selected_step)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = get_layers(loaded.model)
    for layer_index in source_layers:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captured[idx] = component_tensor(output)[0, absolute].detach().clone()
        handles.append(layers[layer_index].register_forward_hook(hook))
    try:
        loaded.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return {
        target: captured[source]
        for source, target in zip(source_layers, target_layers, strict=True)
    }


def direction_for(loaded: Any, case: dict[str, Any]) -> torch.Tensor:
    target = continuation_ids(loaded, case, case["target"])[0]
    distractors = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
    weight = loaded.model.get_output_embeddings().weight.detach().float()
    direction = weight[target] - weight[distractors].mean(dim=0)
    return (direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)).to(loaded.input_device)


def permute_values(values: dict[int, torch.Tensor], salt: int) -> dict[int, torch.Tensor]:
    result = {}
    for layer, value in values.items():
        shift = (salt * 97 + layer * 31) % max(1, value.numel())
        if shift == 0:
            shift = 1
        result[layer] = torch.roll(value, shifts=shift, dims=0)
    return result


@torch.inference_mode()
def phrase_logprob(
    loaded: Any,
    case: dict[str, Any],
    patch_values: dict[int, torch.Tensor],
    patch_step: int,
) -> tuple[float, bool]:
    encoded = encoded_prompt(loaded, case)
    answer = continuation_ids(loaded, case, case["target"])
    append = torch.tensor([answer], dtype=encoded["input_ids"].dtype, device=loaded.input_device)
    input_ids = torch.cat([encoded["input_ids"], append], dim=1)
    attention_mask = torch.ones_like(input_ids)
    absolute = int(encoded["input_ids"].shape[1]) - 1 + patch_step
    reached = absolute < input_ids.shape[1]
    handles = []
    layers = get_layers(loaded.model)
    if reached:
        for layer_index, value in patch_values.items():
            def hook(
                _module: Any, _inputs: tuple[Any, ...], output: Any,
                vector: torch.Tensor = value, position: int = absolute,
            ) -> Any:
                tensor = component_tensor(output).clone()
                tensor[:, position, :] = vector.to(tensor.device, dtype=tensor.dtype)
                return replace_output(output, tensor)
            handles.append(layers[layer_index].register_forward_hook(hook))
    try:
        output = loaded.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
    prompt_length = int(encoded["input_ids"].shape[1])
    values = [
        float(log_probs[prompt_length + offset - 1, token].item())
        for offset, token in enumerate(answer)
    ]
    return float(sum(values)), reached


@torch.inference_mode()
def generate_condition(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    patch_values: dict[int, torch.Tensor],
    patch_step: int,
    block_layers: list[int],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoded = encoded_prompt(loaded, case)
    direction = direction_for(loaded, case)
    target_id = continuation_ids(loaded, case, case["target"])[0]
    distractor_ids = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
    layers = get_layers(loaded.model)
    call_counts = {index: 0 for index in range(len(layers))}
    current_calls = {index: -1 for index in range(len(layers))}
    trace_values: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    patch_reached = {"value": False}
    for layer_index, layer in enumerate(layers):
        def layer_pre(_module: Any, _inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            current_calls[idx] = call_counts[idx]

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            if current_calls[idx] == patch_step:
                trace_values[("attention_output", idx)] = component_tensor(output)[0, -1].detach().clone()

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            if current_calls[idx] == patch_step:
                trace_values[("mlp_output", idx)] = component_tensor(output)[0, -1].detach().clone()

        def layer_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> Any:
            tensor = component_tensor(output)
            result = output
            if current_calls[idx] == patch_step:
                before = tensor[0, -1].detach().clone()
                trace_values[("residual_before_patch", idx)] = before
                if idx in patch_values:
                    modified = tensor.clone()
                    modified[:, -1, :] = patch_values[idx].to(modified.device, dtype=modified.dtype)
                    result = replace_output(output, modified)
                    tensor = modified
                    patch_reached["value"] = True
                trace_values[("residual_output", idx)] = tensor[0, -1].detach().clone()
            call_counts[idx] += 1
            return result

        handles.extend([
            layer.register_forward_pre_hook(layer_pre),
            layer.self_attn.register_forward_hook(attention_post),
            layer.mlp.register_forward_hook(mlp_post),
            layer.register_forward_hook(layer_post),
        ])
    try:
        generated = loaded.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    finally:
        for handle in handles:
            handle.remove()
    suffix = generated.sequences[0, encoded["input_ids"].shape[1] :]
    ids = [int(value) for value in suffix.tolist()]
    text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
    segment = answer_segment(text)
    first_logits = generated.scores[0][0].detach().float()
    target_logit = float(first_logits[target_id].item())
    best_wrong = max(float(first_logits[token].item()) for token in distractor_ids)
    target_rank = 1 + int((first_logits > target_logit).sum().item())
    phrase_value, phrase_patch_reached = phrase_logprob(loaded, case, patch_values, patch_step)
    created_at = now()
    path_rows = []
    for (component_type, layer_index), vector in sorted(trace_values.items(), key=lambda item: (item[0][1], item[0][0])):
        value = vector.detach().float()
        path_rows.append({
            **output_base(case),
            "created_at": created_at,
            "condition": condition,
            "patch_step": patch_step,
            "block_layers": json.dumps(block_layers),
            "component_type": component_type,
            "component_layer": layer_index,
            "relative_depth": round(layer_index / max(1, len(layers) - 1), 7),
            "projection": round(float(value @ direction.float()), 7),
            "activation_norm": round(float(torch.linalg.vector_norm(value).item()), 7),
            "inside_patch_block": layer_index in patch_values,
            "evidence_level": "L4_registered_dynamic_block_response",
        })
    row = {
        **output_base(case),
        "created_at": created_at,
        "condition": condition,
        "patch_step": patch_step,
        "block_layers": json.dumps(block_layers),
        "block_length": len(block_layers),
        "patch_reached_generation": patch_reached["value"],
        "patch_reached_phrase": phrase_patch_reached,
        "target_margin": round(target_logit - best_wrong, 7),
        "target_rank": target_rank,
        "target_phrase_logprob": round(phrase_value, 7),
        "generated_text": text,
        "generated_token_ids": json.dumps(ids),
        "generated_token_count": len(ids),
        "target_answer_segment_match": target_match(segment, case["target_aliases"]),
        "protocol_success_answer_segment": phase330_survey.protocol_ok(case, segment),
        "behavior_success": (
            target_match(segment, case["target_aliases"])
            and phase330_survey.protocol_ok(case, segment)
        ),
        "eos_emitted": loaded.tokenizer.eos_token_id in ids,
        "evidence_level": "L4_registered_dynamic_block_exchange",
        "single_unit_causal": False,
    }
    return row, path_rows


def run_model(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "exchange" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    cases = [row for row in read_jsonl(root / "phase333_registered_cases.jsonl") if row["model"] == model]
    lookup = {
        (row["mechanism_id"], row["item_index"], row["template_id"], row["interface"]): row
        for row in cases
    }
    survey_dir = root / "survey" / model
    baseline_rows = read_jsonl(survey_dir / "baseline_rows.jsonl")
    event_rows = read_jsonl(survey_dir / "event_rows.jsonl")
    plans = read_jsonl(survey_dir / "block_plans.jsonl")
    baseline = {row["case_id"]: row for row in baseline_rows}
    events = {row["case_id"]: row for row in event_rows}
    plan_map = {(row["mechanism_id"], row["interface"]): row for row in plans}
    paired = {
        "missing_condition_control": "two_hop_blocked",
        "two_hop_blocked": "missing_condition_control",
    }
    registered = []
    for mechanism in paired:
        for item_index in (9, 10, 11):
            wrong_item = 9 + ((item_index - 9 + 1) % 3)
            for template_id in ("template_a", "template_b", "template_c"):
                for direction, donor_interface, recipient_interface in (
                    ("raw_to_answer_aligned", "raw_completion", "answer_aligned_chat"),
                    ("answer_aligned_to_raw", "answer_aligned_chat", "raw_completion"),
                ):
                    registered.append({
                        "exchange_case_id": (
                            f"phase333_exchange_{model}_{mechanism}_{item_index:02d}_"
                            f"{template_id}_{direction}"
                        ),
                        "direction": direction,
                        "recipient": lookup[(mechanism, item_index, template_id, recipient_interface)],
                        "correct_donor": lookup[(mechanism, item_index, template_id, donor_interface)],
                        "wrong_object": lookup[(mechanism, wrong_item, template_id, donor_interface)],
                        "wrong_interface": lookup[(mechanism, item_index, template_id, "native_chat")],
                        "matched_control": lookup[(paired[mechanism], item_index, template_id, donor_interface)],
                        "donor_interface": donor_interface,
                        "recipient_interface": recipient_interface,
                    })
    if len(registered) != 36:
        raise RuntimeError(f"Expected 36 exchange cases for {model}, got {len(registered)}")
    loaded = None
    rows = []
    path_rows = []
    registry_rows = []
    capture_cache: dict[tuple[Any, ...], dict[int, torch.Tensor]] = {}
    try:
        loaded = load_probe_model(model)

        def values_for(
            donor_case: dict[str, Any], source_layers: list[int], target_layers: list[int]
        ) -> dict[int, torch.Tensor]:
            step = event_step(events[donor_case["case_id"]])
            key = (donor_case["case_id"], tuple(source_layers), tuple(target_layers), step)
            if key not in capture_cache:
                capture_cache[key] = capture_block_values(
                    loaded,
                    donor_case,
                    json.loads(baseline[donor_case["case_id"]]["generated_token_ids"]),
                    source_layers,
                    target_layers,
                    step,
                )
            return capture_cache[key]

        for case_index, entry in enumerate(registered, 1):
            recipient = entry["recipient"]
            mechanism = recipient["mechanism_id"]
            recipient_plan = plan_map[(mechanism, entry["recipient_interface"])]
            donor_plan = plan_map[(mechanism, entry["donor_interface"])]
            wrong_interface_plan = plan_map[(mechanism, "native_chat")]
            control_plan = plan_map[(paired[mechanism], entry["donor_interface"])]
            recipient_step = event_step(events[recipient["case_id"]])
            recipient_count = int(baseline[recipient["case_id"]]["generated_token_count"])
            wrong_time_step = min(max(0, recipient_count - 1), recipient_step + 2)
            if wrong_time_step == recipient_step:
                wrong_time_step = max(0, recipient_step - 1)
            correct_values = {}
            correct_layers = {}
            for length in (1, 2, 4):
                source_layers = donor_plan["block_windows"][str(length)]
                target_layers = recipient_plan["block_windows"][str(length)]
                correct_values[length] = values_for(entry["correct_donor"], source_layers, target_layers)
                correct_layers[length] = target_layers
            target4 = recipient_plan["block_windows"]["4"]
            wrong_object_values = values_for(
                entry["wrong_object"], donor_plan["block_windows"]["4"], target4
            )
            wrong_interface_values = values_for(
                entry["wrong_interface"], wrong_interface_plan["block_windows"]["4"], target4
            )
            control_values = values_for(
                entry["matched_control"], control_plan["block_windows"]["4"], target4
            )
            permuted_values = permute_values(correct_values[4], recipient["item_index"] + len(rows))
            plan = {
                "baseline": ({}, recipient_step, []),
                "correct_block_1": (correct_values[1], recipient_step, correct_layers[1]),
                "correct_block_2": (correct_values[2], recipient_step, correct_layers[2]),
                "correct_block_4": (correct_values[4], recipient_step, correct_layers[4]),
                "wrong_object_block_4": (wrong_object_values, recipient_step, target4),
                "wrong_interface_block_4": (wrong_interface_values, recipient_step, target4),
                "wrong_time_block_4": (correct_values[4], wrong_time_step, target4),
                "moment_matched_permutation_block_4": (permuted_values, recipient_step, target4),
                "matched_control_block_4": (control_values, recipient_step, target4),
            }
            registry_rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "exchange_case_id": entry["exchange_case_id"],
                "model": model,
                "family_id": recipient["family_id"],
                "mechanism_id": mechanism,
                "cohort": recipient["cohort"],
                "item_index": recipient["item_index"],
                "template_id": recipient["template_id"],
                "exchange_direction": entry["direction"],
                "recipient_case_id": recipient["case_id"],
                "correct_donor_case_id": entry["correct_donor"]["case_id"],
                "wrong_object_case_id": entry["wrong_object"]["case_id"],
                "wrong_interface_case_id": entry["wrong_interface"]["case_id"],
                "matched_control_case_id": entry["matched_control"]["case_id"],
                "recipient_event_step": recipient_step,
                "wrong_time_step": wrong_time_step,
                "condition_count": len(BLOCK_CONDITIONS),
                "selection_updates_allowed": False,
            })
            case_rows = []
            for condition in BLOCK_CONDITIONS:
                values, patch_step, block_layers = plan[condition]
                row, traces = generate_condition(
                    loaded, recipient, condition, values, patch_step, block_layers, max_new_tokens
                )
                row.update({
                    "exchange_case_id": entry["exchange_case_id"],
                    "exchange_direction": entry["direction"],
                    "donor_interface": entry["donor_interface"],
                    "recipient_interface": entry["recipient_interface"],
                })
                for trace in traces:
                    trace.update({
                        "exchange_case_id": entry["exchange_case_id"],
                        "exchange_direction": entry["direction"],
                    })
                case_rows.append(row)
                path_rows.extend(traces)
            base = next(row for row in case_rows if row["condition"] == "baseline")
            for row in case_rows:
                row["delta_target_margin_vs_baseline"] = round(row["target_margin"] - base["target_margin"], 7)
                row["target_rank_improvement_vs_baseline"] = base["target_rank"] - row["target_rank"]
                row["delta_phrase_logprob_vs_baseline"] = round(
                    row["target_phrase_logprob"] - base["target_phrase_logprob"], 7
                )
                row["behavior_gained_vs_baseline"] = bool(
                    not base["behavior_success"] and row["behavior_success"]
                )
                row["behavior_lost_vs_baseline"] = bool(
                    base["behavior_success"] and not row["behavior_success"]
                )
                row["protocol_lost_vs_baseline"] = bool(
                    base["protocol_success_answer_segment"]
                    and not row["protocol_success_answer_segment"]
                )
                row["generation_changed_vs_baseline"] = (
                    row["generated_token_ids"] != base["generated_token_ids"]
                )
            rows.extend(case_rows)
            if case_index % 4 == 0:
                print(json.dumps({
                    "quality_only": True,
                    "model": model,
                    "exchange_cases": case_index,
                    "total_cases": len(registered),
                    "condition_rows": len(rows),
                    "response_rows": len(path_rows),
                }), flush=True)
        write_jsonl(model_dir / "registered_exchange_cases.jsonl", registry_rows)
        write_jsonl(model_dir / "condition_rows.jsonl", rows)
        write_parquet(model_dir / "condition_rows.parquet", rows)
        write_parquet(model_dir / "dynamic_response_rows.parquet", path_rows)
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "exchange_case_count": len(registered),
            "condition_row_count": len(rows),
            "generation_row_count": len(rows),
            "dynamic_response_row_count": len(path_rows),
            "patch_generation_reached_count": sum(row["patch_reached_generation"] for row in rows),
            "patch_phrase_reached_count": sum(row["patch_reached_phrase"] for row in rows),
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(registered) == 36 and len(rows) == 324,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def collect(round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    survey_quality = []
    exchange_quality = []
    baseline_rows = []
    event_rows = []
    plans = []
    registry = []
    conditions = []
    token_tables = []
    path_tables = []
    response_tables = []
    for model in MODELS:
        survey_dir = root / "survey" / model
        exchange_dir = root / "exchange" / model
        survey_quality.append(read_json(survey_dir / "complete.json"))
        exchange_quality.append(read_json(exchange_dir / "complete.json"))
        baseline_rows.extend(read_jsonl(survey_dir / "baseline_rows.jsonl"))
        event_rows.extend(read_jsonl(survey_dir / "event_rows.jsonl"))
        plans.extend(read_jsonl(survey_dir / "block_plans.jsonl"))
        registry.extend(read_jsonl(exchange_dir / "registered_exchange_cases.jsonl"))
        conditions.extend(read_jsonl(exchange_dir / "condition_rows.jsonl"))
        token_tables.append(pq.read_table(survey_dir / "token_rows.parquet"))
        path_tables.append(pq.read_table(survey_dir / "dynamic_path_rows.parquet"))
        response_tables.append(pq.read_table(exchange_dir / "dynamic_response_rows.parquet"))
    write_jsonl(root / "phase333_baseline_rows.jsonl", baseline_rows)
    write_jsonl(root / "phase333_event_rows.jsonl", event_rows)
    write_jsonl(root / "phase333_block_plans.jsonl", plans)
    write_jsonl(root / "phase333_registered_exchange_cases.jsonl", registry)
    write_jsonl(root / "phase333_condition_rows.jsonl", conditions)
    write_parquet(root / "phase333_condition_rows.parquet", conditions)
    pq.write_table(
        pa.concat_tables(token_tables, promote_options="permissive"),
        root / "phase333_token_rows.parquet", compression="zstd",
    )
    pq.write_table(
        pa.concat_tables(path_tables, promote_options="permissive"),
        root / "phase333_dynamic_path_rows.parquet", compression="zstd",
    )
    pq.write_table(
        pa.concat_tables(response_tables, promote_options="permissive"),
        root / "phase333_dynamic_response_rows.parquet", compression="zstd",
    )
    quality = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model_count": 3,
        "registered_case_count": len(baseline_rows),
        "baseline_generation_count": len(baseline_rows),
        "event_row_count": len(event_rows),
        "token_row_count": sum(row["token_row_count"] for row in survey_quality),
        "dynamic_path_row_count": sum(row["dynamic_path_row_count"] for row in survey_quality),
        "block_plan_count": len(plans),
        "registered_exchange_case_count": len(registry),
        "condition_row_count": len(conditions),
        "exchange_generation_count": len(conditions),
        "dynamic_response_row_count": sum(row["dynamic_response_row_count"] for row in exchange_quality),
        "all_survey_valid": all(row["valid"] for row in survey_quality),
        "all_exchange_valid": all(row["valid"] for row in exchange_quality),
        "selection_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
    }
    quality["valid"] = (
        len(baseline_rows) == 648 and len(event_rows) == 648 and len(plans) == 18
        and len(registry) == 108 and len(conditions) == 972
        and quality["all_survey_valid"] and quality["all_exchange_valid"]
    )
    write_json(root / "phase333_execution_quality.json", quality)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.model:
        result = run_model(args.model, args.round, args.max_new_tokens)
    elif args.collect:
        result = collect(args.round)
    else:
        raise SystemExit("Use --model MODEL or --collect")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
