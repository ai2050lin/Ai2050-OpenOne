#!/usr/bin/env python3
"""Calibrate and audit Phase334 receiver-path natural necessity."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
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
from phase333_dynamic_survey import continuation_ids  # noqa: E402
from phase334_natural_necessity_case_bank import ROUND_DEFAULT  # noqa: E402
from phase334_natural_contrast_survey import (  # noqa: E402
    component_tensor, encoded_prompt, role_positions, target_direction,
)


PHASE = "Phase334"
SCHEMA_VERSION = "12.0.0"
OUT = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
HELDOUT_CONDITIONS = (
    "baseline", "correct_selected_delete", "correct_attention_delete",
    "correct_mlp_delete", "correct_residual_delete", "correct_joint_delete",
    "wrong_time_delete", "wrong_object_increment", "matched_mechanism_increment",
    "moment_matched_permutation", "wrong_layer_delete",
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


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(json_safe(rows)), path, compression="zstd", row_group_size=32768
    )


def replace_output(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def case_base(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "model": case["model"],
        "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "cohort": case["cohort"],
        "paired_mechanism_id": case["paired_mechanism_id"],
        "item_index": case["item_index"],
        "split": case["split"],
        "template_id": case["template_id"],
        "interface": case["interface"],
    }


def case_positions(loaded: Any, case: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, tuple[int, bool]]]:
    encoded = encoded_prompt(loaded, case)
    ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    return encoded, role_positions(loaded, case, ids)


@torch.inference_mode()
def capture_selected_vector(
    loaded: Any, case: dict[str, Any], component: str, layer_index: int, position_role: str
) -> torch.Tensor:
    encoded, positions = case_positions(loaded, case)
    position = positions[position_role][0]
    layers = get_layers(loaded.model)
    layer = layers[layer_index]
    capture: dict[str, torch.Tensor] = {}
    before: dict[str, torch.Tensor] = {}
    handles = []

    def pre(_module: Any, inputs: tuple[Any, ...]) -> None:
        before["value"] = inputs[0].detach()

    def attention(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if component == "attention_output":
            capture["value"] = component_tensor(output)[0, position].detach().float().cpu()

    def mlp(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if component == "mlp_output":
            capture["value"] = component_tensor(output)[0, position].detach().float().cpu()

    def post(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if component == "residual_increment":
            capture["value"] = (
                component_tensor(output)[0, position] - before["value"][0, position]
            ).detach().float().cpu()

    handles.extend([
        layer.register_forward_pre_hook(pre),
        layer.self_attn.register_forward_hook(attention),
        layer.mlp.register_forward_hook(mlp),
        layer.register_forward_hook(post),
    ])
    try:
        loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if "value" not in capture:
        raise RuntimeError(f"Failed to capture {component} at layer {layer_index}")
    return capture["value"]


def permute_vector(vector: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(vector.numel(), generator=generator)
    return vector.flatten()[order].reshape(vector.shape)


def patch_specs(
    component: str, layer: int, position: int, mode: str, value: torch.Tensor | None = None
) -> list[dict[str, Any]]:
    if component == "joint_attention_mlp":
        return [
            {"component": "attention_output", "layer": layer, "position": position, "mode": mode, "value": value},
            {"component": "mlp_output", "layer": layer, "position": position, "mode": mode, "value": value},
        ]
    return [{"component": component, "layer": layer, "position": position, "mode": mode, "value": value}]


def install_hooks(
    loaded: Any,
    specs: list[dict[str, Any]],
    trace_position: int | None,
    trace_from_layer: int,
    direction: torch.Tensor,
) -> tuple[list[Any], dict[str, bool], dict[tuple[str, int], tuple[float, float]]]:
    layers = get_layers(loaded.model)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        by_layer[int(spec["layer"])].append(spec)
    reached = {"value": False}
    traces: dict[tuple[str, int], tuple[float, float]] = {}
    handles = []
    layer_inputs: dict[int, torch.Tensor] = {}
    for layer_index, layer in enumerate(layers):
        def pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            layer_inputs[idx] = inputs[0].detach()

        def attention_hook(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> Any:
            tensor = component_tensor(output)
            result = output
            for spec in by_layer.get(idx, []):
                if spec["component"] != "attention_output" or tensor.shape[1] <= spec["position"]:
                    continue
                modified = tensor.clone()
                if spec["mode"] == "delete":
                    modified[0, spec["position"]] = 0
                else:
                    modified[0, spec["position"]] = spec["value"].to(modified.device, dtype=modified.dtype)
                tensor = modified
                result = replace_output(output, modified)
                reached["value"] = True
            if trace_position is not None and idx >= trace_from_layer and tensor.shape[1] > trace_position:
                value = tensor[0, trace_position].detach().float()
                traces[("attention_output", idx)] = (
                    float(value @ direction.to(value.device)), float(torch.linalg.vector_norm(value).item())
                )
            return result

        def mlp_hook(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> Any:
            tensor = component_tensor(output)
            result = output
            for spec in by_layer.get(idx, []):
                if spec["component"] != "mlp_output" or tensor.shape[1] <= spec["position"]:
                    continue
                modified = tensor.clone()
                if spec["mode"] == "delete":
                    modified[0, spec["position"]] = 0
                else:
                    modified[0, spec["position"]] = spec["value"].to(modified.device, dtype=modified.dtype)
                tensor = modified
                result = replace_output(output, modified)
                reached["value"] = True
            if trace_position is not None and idx >= trace_from_layer and tensor.shape[1] > trace_position:
                value = tensor[0, trace_position].detach().float()
                traces[("mlp_output", idx)] = (
                    float(value @ direction.to(value.device)), float(torch.linalg.vector_norm(value).item())
                )
            return result

        def layer_hook(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> Any:
            tensor = component_tensor(output)
            result = output
            for spec in by_layer.get(idx, []):
                if spec["component"] != "residual_increment" or tensor.shape[1] <= spec["position"]:
                    continue
                modified = tensor.clone()
                before = layer_inputs[idx]
                if spec["mode"] == "delete":
                    modified[0, spec["position"]] = before[0, spec["position"]]
                else:
                    delta = spec["value"].to(modified.device, dtype=modified.dtype)
                    modified[0, spec["position"]] = before[0, spec["position"]] + delta
                tensor = modified
                result = replace_output(output, modified)
                reached["value"] = True
            if trace_position is not None and idx >= trace_from_layer and tensor.shape[1] > trace_position:
                value = tensor[0, trace_position].detach().float()
                traces[("residual_output", idx)] = (
                    float(value @ direction.to(value.device)), float(torch.linalg.vector_norm(value).item())
                )
            return result

        handles.extend([
            layer.register_forward_pre_hook(pre),
            layer.self_attn.register_forward_hook(attention_hook),
            layer.mlp.register_forward_hook(mlp_hook),
            layer.register_forward_hook(layer_hook),
        ])
    return handles, reached, traces


@torch.inference_mode()
def phrase_with_patch(
    loaded: Any, case: dict[str, Any], specs: list[dict[str, Any]],
    trace_position: int | None, trace_from_layer: int,
) -> tuple[float, bool, dict[tuple[str, int], tuple[float, float]]]:
    encoded = encoded_prompt(loaded, case)
    target_ids = continuation_ids(loaded, case, case["target"])
    suffix = torch.tensor([target_ids], dtype=encoded["input_ids"].dtype, device=loaded.input_device)
    input_ids = torch.cat([encoded["input_ids"], suffix], dim=1)
    attention_mask = torch.ones_like(input_ids)
    direction = target_direction(loaded, case)
    handles, reached, traces = install_hooks(
        loaded, specs, trace_position, trace_from_layer, direction
    )
    try:
        output = loaded.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    prompt_length = int(encoded["input_ids"].shape[1])
    positions = torch.arange(prompt_length - 1, prompt_length - 1 + len(target_ids), device=loaded.input_device)
    logits = output.logits[0, positions].float()
    token_tensor = torch.tensor(target_ids, device=loaded.input_device)
    value = float(torch.log_softmax(logits, dim=-1).gather(1, token_tensor[:, None]).sum().item())
    return value, reached["value"], traces


@torch.inference_mode()
def generate_with_patch(
    loaded: Any, case: dict[str, Any], condition: str, specs: list[dict[str, Any]],
    selected_plan: dict[str, Any], max_new_tokens: int, trace: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoded, positions = case_positions(loaded, case)
    role = selected_plan["selected_position_role"]
    trace_position = positions[role][0] if trace else None
    layer = int(selected_plan["selected_layer"])
    direction = target_direction(loaded, case)
    handles, reached, generation_traces = install_hooks(
        loaded, specs, trace_position, layer, direction
    )
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
    logits = generated.scores[0][0].detach().float()
    target_id = continuation_ids(loaded, case, case["target"])[0]
    distractors = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
    target_logit = float(logits[target_id].item())
    target_rank = 1 + int((logits > target_logit).sum().item())
    phrase_value, phrase_reached, phrase_traces = phrase_with_patch(
        loaded, case, specs, trace_position, layer
    )
    matched = target_match(segment, case["target_aliases"])
    protocol = phase330_survey.protocol_ok(case, segment)
    row = {
        **case_base(case),
        "created_at": now(),
        "condition": condition,
        "selected_depth_bin": selected_plan["depth_bin"],
        "selected_component": selected_plan["selected_component"],
        "selected_layer": layer,
        "selected_position_role": role,
        "patch_reached_generation": reached["value"] if specs else True,
        "patch_reached_phrase": phrase_reached if specs else True,
        "target_margin": round(target_logit - max(float(logits[idx].item()) for idx in distractors), 7),
        "target_rank": target_rank,
        "target_phrase_logprob": round(phrase_value, 7),
        "generated_text": text,
        "generated_token_ids": json.dumps(ids),
        "generated_token_count": len(ids),
        "target_answer_segment_match": matched,
        "protocol_success_answer_segment": protocol,
        "behavior_success": matched and protocol,
        "eos_emitted": loaded.tokenizer.eos_token_id in ids,
        "evidence_level": "L4_registered_natural_necessity_intervention",
        "single_unit_causal": False,
    }
    trace_rows = []
    if trace:
        values = phrase_traces or generation_traces
        for (component, trace_layer), (projection, norm) in sorted(values.items(), key=lambda item: (item[0][1], item[0][0])):
            trace_rows.append({
                **case_base(case),
                "created_at": now(),
                "condition": condition,
                "selected_component": selected_plan["selected_component"],
                "selected_layer": layer,
                "selected_position_role": role,
                "component_type": component,
                "component_layer": trace_layer,
                "relative_depth": round(trace_layer / max(1, len(get_layers(loaded.model)) - 1), 7),
                "target_projection": round(projection, 7),
                "activation_norm": round(norm, 7),
                "downstream_of_selected_layer": trace_layer > layer,
                "evidence_level": "L3_natural_deletion_propagation_trace",
            })
    return row, trace_rows


def with_baseline_delta(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    row["target_rank_loss_vs_baseline"] = row["target_rank"] - baseline["target_rank"]
    row["phrase_logprob_loss_vs_baseline"] = round(
        baseline["target_phrase_logprob"] - row["target_phrase_logprob"], 7
    )
    row["target_margin_loss_vs_baseline"] = round(
        baseline["target_margin"] - row["target_margin"], 7
    )
    row["behavior_lost_vs_baseline"] = bool(
        baseline["behavior_success"] and not row["behavior_success"]
    )
    row["protocol_lost_vs_baseline"] = bool(
        baseline["protocol_success_answer_segment"]
        and not row["protocol_success_answer_segment"]
    )
    row["generation_changed_vs_baseline"] = (
        row["generated_token_ids"] != baseline["generated_token_ids"]
    )
    return row


def eligible_baseline(row: dict[str, Any]) -> bool:
    value = row.get("target_phrase_logprob")
    return bool(
        row.get("behavior_success")
        and int(row.get("target_rank", 10**9)) <= 50
        and value is not None and math.isfinite(float(value))
    )


def freeze_calibration_plans(
    model: str, discovery_plans: list[dict[str, Any]], calibration_rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        grouped[(row["family_id"], row["mechanism_id"], row["interface"], row["selected_depth_bin"])].append(row)
    plan_lookup = {
        (row["family_id"], row["mechanism_id"], row["interface"], row["depth_bin"]): row
        for row in discovery_plans
    }
    summaries = []
    frozen = []
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, rows in sorted(grouped.items()):
        family, mechanism, interface, bin_name = key
        valid = [
            row for row in rows
            if eligible_baseline(baselines[row["case_id"]])
            and row["patch_reached_generation"] and row["patch_reached_phrase"]
            and row.get("target_phrase_logprob") is not None
        ]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "interface": interface,
            "depth_bin": bin_name,
            "planned_case_count": len(rows),
            "baseline_eligible_case_count": sum(eligible_baseline(baselines[row["case_id"]]) for row in rows),
            "common_valid_case_count": len(valid),
            "behavior_loss_rate": round(sum(row["behavior_lost_vs_baseline"] for row in valid) / len(valid), 7) if valid else 0.0,
            "mean_phrase_logprob_loss": round(mean(row["phrase_logprob_loss_vs_baseline"] for row in valid), 7) if valid else 0.0,
            "mean_target_rank_loss": round(mean(row["target_rank_loss_vs_baseline"] for row in valid), 7) if valid else 0.0,
            "calibration_only": True,
        }
        summaries.append(summary)
        by_cell[(family, mechanism, interface)].append(summary)
    for cell, values in sorted(by_cell.items()):
        family, mechanism, interface = cell
        chosen = max(
            values,
            key=lambda row: (
                row["common_valid_case_count"] >= 6,
                row["behavior_loss_rate"], row["mean_phrase_logprob_loss"], row["mean_target_rank_loss"],
                {"early": 2, "middle": 1, "late": 0}[row["depth_bin"]],
            ),
        )
        plan = dict(plan_lookup[(family, mechanism, interface, chosen["depth_bin"])])
        plan.update({
            "calibration_planned_case_count": chosen["planned_case_count"],
            "calibration_baseline_eligible_case_count": chosen["baseline_eligible_case_count"],
            "calibration_common_valid_case_count": chosen["common_valid_case_count"],
            "calibration_behavior_loss_rate": chosen["behavior_loss_rate"],
            "calibration_mean_phrase_logprob_loss": chosen["mean_phrase_logprob_loss"],
            "calibration_mean_target_rank_loss": chosen["mean_target_rank_loss"],
            "final_selection_split": "calibration_only_after_discovery_candidate_freeze",
            "heldout_updates_allowed": False,
        })
        frozen.append(plan)
    if len(frozen) != 18:
        raise RuntimeError(f"Expected 18 frozen calibration plans for {model}, got {len(frozen)}")
    return summaries, frozen


def run_calibration(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "calibration" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    cases = [row for row in read_jsonl(root / "phase334_registered_cases.jsonl") if row["model"] == model and row["split"] == "calibration"]
    baselines = {
        row["case_id"]: row for row in read_jsonl(root / "survey" / model / "baseline_rows.jsonl")
    }
    plans = read_jsonl(root / "survey" / model / "discovery_candidate_plans.jsonl")
    plan_map: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        plan_map[(plan["family_id"], plan["mechanism_id"], plan["interface"])].append(plan)
    loaded = None
    rows = []
    try:
        loaded = load_probe_model(model)
        for case_index, case in enumerate(cases, 1):
            _encoded, positions = case_positions(loaded, case)
            for plan in sorted(plan_map[(case["family_id"], case["mechanism_id"], case["interface"])], key=lambda row: row["depth_bin"]):
                specs = patch_specs(
                    plan["selected_component"], int(plan["selected_layer"]),
                    positions[plan["selected_position_role"]][0], "delete",
                )
                row, _traces = generate_with_patch(
                    loaded, case, f"calibration_delete_{plan['depth_bin']}", specs,
                    plan, max_new_tokens, False,
                )
                rows.append(with_baseline_delta(row, baselines[case["case_id"]]))
            if case_index % 18 == 0:
                print(json.dumps({
                    "quality_only": True, "stage": "calibration", "model": model,
                    "cases": case_index, "total_cases": len(cases), "condition_rows": len(rows),
                }), flush=True)
        summaries, frozen = freeze_calibration_plans(model, plans, rows, baselines)
        write_jsonl(model_dir / "calibration_condition_rows.jsonl", rows)
        write_jsonl(model_dir / "calibration_candidate_summary.jsonl", summaries)
        write_jsonl(model_dir / "frozen_necessity_plans.jsonl", frozen)
        quality = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "calibration_case_count": len(cases),
            "condition_row_count": len(rows), "candidate_summary_count": len(summaries),
            "frozen_plan_count": len(frozen), "heldout_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(cases) == 162 and len(rows) == 486 and len(frozen) == 18,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def run_heldout(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "heldout" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    all_cases = [row for row in read_jsonl(root / "phase334_registered_cases.jsonl") if row["model"] == model]
    cases = [row for row in all_cases if row["split"] == "heldout"]
    lookup = {
        (row["family_id"], row["mechanism_id"], row["item_index"], row["template_id"], row["interface"]): row
        for row in all_cases
    }
    frozen = read_jsonl(root / "calibration" / model / "frozen_necessity_plans.jsonl")
    plan_map = {(row["family_id"], row["mechanism_id"], row["interface"]): row for row in frozen}
    loaded = None
    rows = []
    response_rows = []
    registry = []
    capture_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)

        def vector_for(source_case: dict[str, Any], component: str, layer: int, role: str) -> torch.Tensor:
            key = (source_case["case_id"], component, layer, role)
            if key not in capture_cache:
                capture_cache[key] = capture_selected_vector(
                    loaded, source_case, component, layer, role
                )
            return capture_cache[key]

        for case_index, case in enumerate(cases, 1):
            plan = plan_map[(case["family_id"], case["mechanism_id"], case["interface"])]
            component = plan["selected_component"]
            layer = int(plan["selected_layer"])
            role = plan["selected_position_role"]
            _encoded, positions = case_positions(loaded, case)
            position = positions[role][0]
            wrong_role = {"source": "answer_start", "query": "source", "answer_start": "query"}[role]
            wrong_item = 9 + ((int(case["item_index"]) - 9 + 1) % 3)
            wrong_object = lookup[(
                case["family_id"], case["mechanism_id"], wrong_item,
                case["template_id"], case["interface"],
            )]
            paired = lookup[(
                case["family_id"], case["paired_mechanism_id"], case["item_index"],
                case["template_id"], case["interface"],
            )]
            natural = vector_for(case, component, layer, role)
            wrong_object_value = vector_for(wrong_object, component, layer, role)
            paired_value = vector_for(paired, component, layer, role)
            permuted = permute_vector(natural, int(case["item_index"]) * 101 + layer)
            wrong_layer = (layer + max(2, len(layers) // 2)) % len(layers)
            if wrong_layer == layer:
                wrong_layer = (layer + 1) % len(layers)
            condition_specs = {
                "baseline": [],
                "correct_selected_delete": patch_specs(component, layer, position, "delete"),
                "correct_attention_delete": patch_specs("attention_output", layer, position, "delete"),
                "correct_mlp_delete": patch_specs("mlp_output", layer, position, "delete"),
                "correct_residual_delete": patch_specs("residual_increment", layer, position, "delete"),
                "correct_joint_delete": patch_specs("joint_attention_mlp", layer, position, "delete"),
                "wrong_time_delete": patch_specs(component, layer, positions[wrong_role][0], "delete"),
                "wrong_object_increment": patch_specs(component, layer, position, "replace", wrong_object_value),
                "matched_mechanism_increment": patch_specs(component, layer, position, "replace", paired_value),
                "moment_matched_permutation": patch_specs(component, layer, position, "replace", permuted),
                "wrong_layer_delete": patch_specs(component, wrong_layer, position, "delete"),
            }
            registry.append({
                **case_base(case), "created_at": now(),
                "selected_depth_bin": plan["depth_bin"], "selected_component": component,
                "selected_layer": layer, "selected_position_role": role,
                "wrong_position_role": wrong_role, "wrong_layer": wrong_layer,
                "wrong_object_case_id": wrong_object["case_id"],
                "matched_mechanism_case_id": paired["case_id"],
                "condition_count": len(HELDOUT_CONDITIONS),
                "heldout_updates_allowed": False,
            })
            case_rows = []
            for condition in HELDOUT_CONDITIONS:
                trace = condition in {"baseline", "correct_selected_delete"}
                row, traces = generate_with_patch(
                    loaded, case, condition, condition_specs[condition], plan,
                    max_new_tokens, trace,
                )
                case_rows.append(row)
                response_rows.extend(traces)
            baseline = next(row for row in case_rows if row["condition"] == "baseline")
            for row in case_rows:
                rows.append(with_baseline_delta(row, baseline))
            if case_index % 9 == 0:
                print(json.dumps({
                    "quality_only": True, "stage": "heldout", "model": model,
                    "cases": case_index, "total_cases": len(cases),
                    "condition_rows": len(rows), "response_rows": len(response_rows),
                }), flush=True)
        write_jsonl(model_dir / "registered_heldout_cases.jsonl", registry)
        write_jsonl(model_dir / "heldout_condition_rows.jsonl", rows)
        write_parquet(model_dir / "heldout_condition_rows.parquet", rows)
        write_parquet(model_dir / "downstream_response_rows.parquet", response_rows)
        quality = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "heldout_case_count": len(cases),
            "registered_heldout_case_count": len(registry),
            "condition_row_count": len(rows), "generation_row_count": len(rows),
            "downstream_response_row_count": len(response_rows),
            "patch_generation_reached_count": sum(row["patch_reached_generation"] for row in rows),
            "patch_phrase_reached_count": sum(row["patch_reached_phrase"] for row in rows),
            "heldout_updates_allowed": False, "single_unit_intervention_gate_open": False,
            "valid": len(cases) == 162 and len(rows) == 1782,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    result = (
        run_calibration(args.model, args.round, args.max_new_tokens)
        if args.stage == "calibration"
        else run_heldout(args.model, args.round, args.max_new_tokens)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
