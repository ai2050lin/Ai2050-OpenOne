#!/usr/bin/env python3
"""Collect compact full-depth Phase398 joint-factorial query traces."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
PRIVATE = OUT / "query_trace/protocol/private"
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = ("instrument", "discovery", "calibration", "physical_holdout")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
COORDINATES = ("query_end", "answer_anchor")
EFFECTS = {
    "R": ("relation_level_private",),
    "O": ("order_level_private",),
    "Q": ("query_level_private",),
    "RO": ("relation_level_private", "order_level_private"),
    "RQ": ("relation_level_private", "query_level_private"),
    "OQ": ("order_level_private", "query_level_private"),
    "ROQ": ("relation_level_private", "order_level_private", "query_level_private"),
}
MAX_BLOCK_RELATIVE_ERROR = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def source_path(stage: str) -> Path:
    if stage == "instrument":
        return PRIVATE / "phase398_instrument_query_trace_cases.jsonl"
    return PRIVATE / f"phase398_{stage}_query_trace_cases.jsonl"


def sign(row: dict[str, Any], fields: tuple[str, ...]) -> float:
    result = 1.0
    for field in fields:
        result *= 1.0 if int(row[field]) == 1 else -1.0
    return result


def factorial_vector(rows: list[dict[str, Any]], vectors: list[torch.Tensor], fields: tuple[str, ...]) -> torch.Tensor:
    weighted = [vector * sign(row, fields) for row, vector in zip(rows, vectors, strict=True)]
    return torch.stack(weighted).mean(dim=0)


def factorial_scalar(rows: list[dict[str, Any]], values: list[float], fields: tuple[str, ...]) -> float:
    return sum(value * sign(row, fields) for row, value in zip(rows, values, strict=True)) / len(rows)


def clean_float(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase398 non-finite scalar: {value}")
    return round(value, 9)


def target_margin(logits: torch.Tensor, token_id: int) -> tuple[float, int, bool]:
    target = float(logits[token_id].item())
    top = torch.topk(logits, k=2)
    winner = int(top.indices[0].item())
    competitor = float(top.values[1].item()) if winner == token_id else float(top.values[0].item())
    return target - competitor, winner, winner == token_id


@torch.inference_mode()
def collect_case(loaded: Any, layers: list[Any], captures: dict[tuple[str, int], Any], case: dict[str, Any]) -> dict[str, Any]:
    captures.clear()
    ids = torch.tensor([case["prompt_token_ids_private"]], dtype=torch.long, device=loaded.input_device)
    output = loaded.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    positions = {
        "query_end": int(case["query_end_position_private"]),
        "answer_anchor": int(case["answer_anchor_position_private"]),
    }
    states: dict[tuple[str, int, str], torch.Tensor] = {}
    block_errors = []
    for layer_index in range(len(layers)):
        for coordinate, position in positions.items():
            selected = {
                component: captures[(component, layer_index)][0, position].detach().float().cpu()
                for component in COMPONENTS
            }
            reconstructed = selected["layer_input"] + selected["attention_output"] + selected["mlp_output"]
            error = float(torch.linalg.vector_norm(selected["layer_output"] - reconstructed).item())
            scale = float(torch.linalg.vector_norm(selected["layer_output"]).item())
            block_errors.append(error / max(scale, 1e-8))
            for component, vector in selected.items():
                states[(coordinate, layer_index, component)] = vector

    logits = output.logits[0, -1].detach().float()
    past = output.past_key_values
    prefix_matches: list[bool] = []
    total_length = ids.shape[1]
    for token_id in case["target_decision_prefix_token_ids_private"]:
        prefix_matches.append(int(torch.argmax(logits).item()) == int(token_id))
        total_length += 1
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=loaded.input_device)
        output = loaded.model(
            input_ids=token,
            attention_mask=torch.ones((1, total_length), dtype=torch.long, device=loaded.input_device),
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = output.logits[0, -1].detach().float()
        past = output.past_key_values
    target_token = int(case["target_encoded_completion_token_id_private"])
    margin, winner, target_match = target_margin(logits, target_token)
    del output, past, logits, ids
    captures.clear()
    return {
        "case": case,
        "states": states,
        "target_completion_margin": margin,
        "target_completion_winner": winner,
        "all_prefix_transitions_match": all(prefix_matches),
        "target_completion_argmax_match": target_match,
        "max_block_relative_error": max(block_errors),
    }


def summarize_group(model: str, stage: str, collected: list[dict[str, Any]], layer_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(collected) != 16:
        raise RuntimeError(f"Phase398 group must contain 16 cases, got {len(collected)}")
    by_axis = {
        axis: sorted(
            [item for item in collected if item["case"]["axis_private"] == axis],
            key=lambda item: item["case"]["anonymous_condition_slot"],
        )
        for axis in ("X", "Y")
    }
    if any(len(items) != 8 for items in by_axis.values()):
        raise RuntimeError("Phase398 group must contain two complete 2x2x2 lexical axes")
    base = collected[0]["case"]
    rows = []
    for layer_index in range(layer_count):
        for coordinate in COORDINATES:
            for component in COMPONENTS:
                axis_effects: dict[str, dict[str, torch.Tensor]] = {}
                axis_scales: dict[str, float] = {}
                for axis, items in by_axis.items():
                    cases = [item["case"] for item in items]
                    vectors = [item["states"][(coordinate, layer_index, component)] for item in items]
                    axis_scales[axis] = sum(float(torch.linalg.vector_norm(vector).item()) for vector in vectors) / len(vectors)
                    axis_effects[axis] = {
                        name: factorial_vector(cases, vectors, fields)
                        for name, fields in EFFECTS.items()
                    }
                metrics: dict[str, Any] = {}
                for effect_name in EFFECTS:
                    x = axis_effects["X"][effect_name]
                    y = axis_effects["Y"][effect_name]
                    x_norm = float(torch.linalg.vector_norm(x).item())
                    y_norm = float(torch.linalg.vector_norm(y).item())
                    cosine = float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=-1, eps=1e-8).item())
                    metrics[effect_name] = {
                        "x_norm": clean_float(x_norm),
                        "y_norm": clean_float(y_norm),
                        "x_normalized_norm": clean_float(x_norm / max(axis_scales["X"], 1e-8)),
                        "y_normalized_norm": clean_float(y_norm / max(axis_scales["Y"], 1e-8)),
                        "min_axis_normalized_norm": clean_float(min(x_norm / max(axis_scales["X"], 1e-8), y_norm / max(axis_scales["Y"], 1e-8))),
                        "cross_axis_cosine": clean_float(cosine),
                    }
                rows.append({
                    "schema_version": "72.4.0",
                    "phase_id": "Phase398-QueryTraceCollection",
                    "created_at": now(),
                    "model": model,
                    "stage": stage,
                    "public_parallel_group_id": base["phase398_public_parallel_group_id"],
                    "layer_index": layer_index,
                    "relative_depth": clean_float(layer_index / max(layer_count - 1, 1)),
                    "coordinate": coordinate,
                    "component": component,
                    "factorial_effect_metrics": metrics,
                    "raw_state_vectors_persisted": False,
                    "causal_intervention": False,
                })
    completion_effects = {}
    for axis, items in by_axis.items():
        cases = [item["case"] for item in items]
        values = [item["target_completion_margin"] for item in items]
        completion_effects[axis] = {
            name: clean_float(factorial_scalar(cases, values, fields))
            for name, fields in EFFECTS.items()
        }
    summary = {
        "schema_version": "72.4.0",
        "phase_id": "Phase398-QueryTraceGroupAudit",
        "created_at": now(),
        "model": model,
        "stage": stage,
        "public_parallel_group_id": base["phase398_public_parallel_group_id"],
        "case_count": len(collected),
        "surface_private": base["task_surface_private"],
        "target_completion_margin_factorial_effects": completion_effects,
        "prefix_transition_match_count": sum(item["all_prefix_transitions_match"] for item in collected),
        "target_completion_argmax_match_count": sum(item["target_completion_argmax_match"] for item in collected),
        "max_block_relative_error": clean_float(max(item["max_block_relative_error"] for item in collected)),
        "block_conservation_pass": max(item["max_block_relative_error"] for item in collected) <= MAX_BLOCK_RELATIVE_ERROR,
        "target_completion_is_first_value_divergence": False,
    }
    return rows, summary


def run(model: str, stage: str) -> dict[str, Any]:
    cases = [row for row in read_jsonl(source_path(stage)) if row["private_execution_model"] == model]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["anonymous_parallel_group_id"]].append(case)
    loaded = None
    handles: list[Any] = []
    effect_rows, group_rows = [], []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for group_index, group_id in enumerate(sorted(grouped), 1):
            group_cases = sorted(grouped[group_id], key=lambda row: row["anonymous_condition_slot"])
            collected = [collect_case(loaded, layers, captures, case) for case in group_cases]
            rows, summary = summarize_group(model, stage, collected, len(layers))
            effect_rows.extend(rows)
            group_rows.append(summary)
            del collected
            gc.collect()
            print(f"[{model}/{stage}] group {group_index}/{len(grouped)}", flush=True)
        model_root = OUT / "query_trace" / stage / "private" / "models" / model
        write_jsonl(model_root / "factorial_effect_rows.jsonl", effect_rows)
        write_jsonl(model_root / "group_audit_rows.jsonl", group_rows)
        complete = {
            "schema_version": "72.4.0",
            "phase_id": "Phase398-QueryTraceCollection",
            "created_at": now(),
            "model": model,
            "stage": stage,
            "case_count": len(cases),
            "group_count": len(grouped),
            "layer_count": len(layers),
            "factorial_effect_row_count": len(effect_rows),
            "prefix_transition_match_count": sum(row["prefix_transition_match_count"] for row in group_rows),
            "target_completion_argmax_match_count": sum(row["target_completion_argmax_match_count"] for row in group_rows),
            "max_block_relative_error": max(row["max_block_relative_error"] for row in group_rows),
            "all_block_conservation_pass": all(row["block_conservation_pass"] for row in group_rows),
            "valid": bool(cases) and all(len(items) == 16 for items in grouped.values()),
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.stage), ensure_ascii=False, indent=2))
