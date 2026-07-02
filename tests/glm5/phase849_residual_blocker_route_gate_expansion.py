#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase848_internal_route_gate_discovery as p848  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 849
RESULT_ROOT = Path("tests/result/phase849_residual_blocker_route_gate_expansion")
MODELS = p846.MODELS
PREDICTORS = (
    "global_combo",
    "internal_strength_combo",
    "residual_projection_combo",
    "blocker_field_combo",
    "route_competition_combo",
    "joint_gate_combo",
    "compact_joint_gate_combo",
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def valid_token_id(token_id: int | None, vocab_size: int) -> bool:
    return token_id is not None and 0 <= int(token_id) < int(vocab_size)


def tensor_from_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, tuple):
        if not output:
            return None
        output = output[0]
    if torch.is_tensor(output):
        return output
    return None


def lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    if hasattr(model, "embed_out"):
        return model.embed_out.weight.detach()
    raise ValueError("cannot find lm head weight")


def token_vector(weight: torch.Tensor, token_id: int | None) -> torch.Tensor | None:
    if not valid_token_id(token_id, int(weight.shape[0])):
        return None
    return weight[int(token_id)].detach().float().cpu()


def diff_vector(weight: torch.Tensor, a: int | None, b: int | None) -> torch.Tensor | None:
    va = token_vector(weight, a)
    vb = token_vector(weight, b)
    if va is None or vb is None:
        return None
    return va - vb


def projection(hidden: torch.Tensor | None, direction: torch.Tensor | None) -> float | None:
    if hidden is None or direction is None:
        return None
    if int(hidden.numel()) != int(direction.numel()):
        return None
    return float(torch.dot(hidden.float(), direction.float()).item())


def rank_of(logits: torch.Tensor, token_id: int | None) -> int | None:
    if not valid_token_id(token_id, int(logits.numel())):
        return None
    value = logits[int(token_id)]
    return int((logits > value).sum().item()) + 1


def logit_of(logits: torch.Tensor, token_id: int | None) -> float | None:
    if not valid_token_id(token_id, int(logits.numel())):
        return None
    return float(logits[int(token_id)].item())


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 1:
        return "top1"
    if rank <= 5:
        return "top5"
    if rank <= 20:
        return "top20"
    if rank <= 100:
        return "top100"
    return "tail"


def quantile(values: list[float], frac: float) -> float:
    return p848.quantile(values, frac)


def residual_class(value: float, threshold: float) -> str:
    return p848.residual_class(value, threshold)


def activation_sign(value: float, eps: float = 1e-6) -> str:
    return p848.activation_sign(value, eps)


def prompt_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return p848.prompt_key(row)


def token_ids_for_row(row: dict[str, Any]) -> dict[str, Any]:
    target_id = as_int(row.get("target_token_id"))
    polygon_id = as_int(row.get("polygon_token_id"))
    object_id = as_int(row.get("object_token_id"))
    baseline_id = as_int(row.get("baseline_token_id"))
    top_ids = [as_int(x) for x in (row.get("top_token_ids") or [])]
    target_ids = []
    for token_id in [target_id, polygon_id]:
        if token_id is not None and token_id not in target_ids:
            target_ids.append(token_id)

    blocker_id = None
    for candidate in [baseline_id, *top_ids, object_id]:
        if candidate is None:
            continue
        if candidate in target_ids:
            continue
        blocker_id = candidate
        break
    return {
        "target_id": target_id,
        "polygon_id": polygon_id,
        "object_id": object_id,
        "baseline_id": baseline_id,
        "target_ids": target_ids,
        "blocker_id": blocker_id,
    }


def selected_layer_indices(model, gears: list[dict[str, Any]], include_neighbors: bool = True) -> list[int]:
    layers = get_layers(model)
    n_layers = len(layers)
    selected: set[int] = {0, max(0, n_layers // 2), n_layers - 1}
    for gear in gears:
        layer_idx = int(gear["layer_idx"])
        for li in ([layer_idx - 1, layer_idx, layer_idx + 1] if include_neighbors else [layer_idx]):
            if 0 <= li < n_layers:
                selected.add(li)
    return sorted(selected)


def capture_prompt_state(
    model,
    tokenizer,
    device: torch.device,
    row: dict[str, Any],
    gears: list[dict[str, Any]],
    layer_indices: list[int],
    topk_entropy: int,
) -> dict[str, Any]:
    prompt = str(row.get("prompt"))
    ids = p844.encode_prompt(tokenizer, prompt)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    gear_lookup = {str(gear["gear_key"]): (int(gear["layer_idx"]), int(gear["channel_id"])) for gear in gears}
    gear_acts: dict[str, float] = {}
    residuals: dict[int, torch.Tensor] = {}
    handles = []

    def make_layer_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = tensor_from_output(output)
            if tensor is not None:
                residuals[int(layer_idx)] = tensor[0, answer_pos].detach().float().cpu()

        return hook

    def make_down_hook(layer_idx: int, keys: list[tuple[str, int]]):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            vec = inputs[0][0, answer_pos].detach().float().cpu()
            for gear_key, channel_id in keys:
                if 0 <= int(channel_id) < int(vec.numel()):
                    gear_acts[str(gear_key)] = float(vec[int(channel_id)].item())

        return hook

    by_layer: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for gear_key, (layer_idx, channel_id) in gear_lookup.items():
        by_layer[int(layer_idx)].append((gear_key, int(channel_id)))

    for layer_idx in layer_indices:
        if 0 <= int(layer_idx) < len(layers):
            handles.append(layers[int(layer_idx)].register_forward_hook(make_layer_hook(int(layer_idx))))
    for layer_idx, pairs in by_layer.items():
        if 0 <= int(layer_idx) < len(layers) and hasattr(layers[int(layer_idx)].mlp, "down_proj"):
            handles.append(layers[int(layer_idx)].mlp.down_proj.register_forward_pre_hook(make_down_hook(int(layer_idx), pairs)))

    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    for gear_key in gear_lookup:
        gear_acts.setdefault(str(gear_key), 0.0)

    token_info = token_ids_for_row(row)
    vocab_size = int(logits.numel())
    target_ids = [token_id for token_id in token_info["target_ids"] if valid_token_id(token_id, vocab_size)]
    object_id = token_info["object_id"]
    blocker_id = token_info["blocker_id"]
    target_logits = [logit_of(logits, token_id) for token_id in target_ids]
    target_logits = [x for x in target_logits if x is not None]
    target_ranks = [rank_of(logits, token_id) for token_id in target_ids]
    target_ranks = [x for x in target_ranks if x is not None]
    best_target_logit = max(target_logits) if target_logits else None
    best_target_rank = min(target_ranks) if target_ranks else None
    object_logit = logit_of(logits, object_id)
    blocker_logit = logit_of(logits, blocker_id)
    object_rank = rank_of(logits, object_id)
    blocker_rank = rank_of(logits, blocker_id)
    top_values, top_indices = torch.topk(logits, min(int(topk_entropy), int(logits.numel())))
    probs = torch.softmax(top_values - top_values.max(), dim=0)
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    top1_id = int(top_indices[0].item())
    top1_text = tokenizer.decode([top1_id], skip_special_tokens=False)
    if top1_id in target_ids:
        top1_role = "target"
    elif object_id is not None and top1_id == int(object_id):
        top1_role = "object"
    elif blocker_id is not None and top1_id == int(blocker_id):
        top1_role = "blocker"
    elif not top1_text.strip():
        top1_role = "format"
    else:
        top1_role = "other"

    weight = lm_head_weight(model)
    target_id = target_ids[0] if target_ids else token_info["target_id"]
    polygon_id = token_info["polygon_id"]
    directions = {
        "target_minus_blocker": diff_vector(weight, target_id, blocker_id),
        "polygon_minus_blocker": diff_vector(weight, polygon_id, blocker_id),
        "target_minus_object": diff_vector(weight, target_id, object_id),
        "polygon_minus_object": diff_vector(weight, polygon_id, object_id),
        "object_minus_blocker": diff_vector(weight, object_id, blocker_id),
    }

    layer_projection_rows: list[dict[str, Any]] = []
    for layer_idx in sorted(residuals):
        h = residuals[layer_idx]
        row_proj = {"layer_idx": int(layer_idx)}
        for name, direction in directions.items():
            row_proj[name] = projection(h, direction)
        layer_projection_rows.append(row_proj)

    def proj_series(name: str) -> list[float]:
        return [finite(row_proj.get(name)) for row_proj in layer_projection_rows if row_proj.get(name) is not None]

    def final_proj(name: str) -> float:
        vals = proj_series(name)
        return vals[-1] if vals else 0.0

    def max_proj(name: str) -> float:
        vals = proj_series(name)
        return max(vals) if vals else 0.0

    def min_proj(name: str) -> float:
        vals = proj_series(name)
        return min(vals) if vals else 0.0

    target_blocker_final = final_proj("target_minus_blocker")
    polygon_blocker_final = final_proj("polygon_minus_blocker")
    best_target_blocker_final = max(target_blocker_final, polygon_blocker_final)
    target_object_final = final_proj("target_minus_object")
    polygon_object_final = final_proj("polygon_minus_object")
    best_target_object_final = max(target_object_final, polygon_object_final)
    object_blocker_final = final_proj("object_minus_blocker")

    return {
        "ids": ids,
        "answer_pos": answer_pos,
        "gear_acts": gear_acts,
        "layer_indices": sorted(residuals),
        "layer_projections": layer_projection_rows,
        "target_id": token_info["target_id"],
        "polygon_id": token_info["polygon_id"],
        "object_id": object_id,
        "blocker_id": blocker_id,
        "baseline_id": token_info["baseline_id"],
        "best_target_logit": best_target_logit,
        "best_target_rank": best_target_rank,
        "object_logit": object_logit,
        "object_rank": object_rank,
        "blocker_logit": blocker_logit,
        "blocker_rank": blocker_rank,
        "top1_id": top1_id,
        "top1_text": top1_text,
        "top1_role": top1_role,
        "topk_entropy": entropy,
        "target_minus_blocker_logit": finite(best_target_logit) - finite(blocker_logit),
        "target_minus_object_logit": finite(best_target_logit) - finite(object_logit),
        "object_minus_blocker_logit": finite(object_logit) - finite(blocker_logit),
        "blocker_pressure": finite(blocker_logit) - finite(best_target_logit),
        "object_echo_pressure": finite(object_logit) - finite(best_target_logit),
        "route_gap": finite(logits[top1_id].item()) - finite(best_target_logit),
        "target_blocker_resid_final": target_blocker_final,
        "polygon_blocker_resid_final": polygon_blocker_final,
        "best_target_blocker_resid_final": best_target_blocker_final,
        "best_target_blocker_resid_max": max(max_proj("target_minus_blocker"), max_proj("polygon_minus_blocker")),
        "best_target_blocker_resid_min": min(min_proj("target_minus_blocker"), min_proj("polygon_minus_blocker")),
        "best_target_object_resid_final": best_target_object_final,
        "object_blocker_resid_final": object_blocker_final,
        "resid_target_blocker_span": max_proj("target_minus_blocker") - min_proj("target_minus_blocker"),
        "resid_polygon_blocker_span": max_proj("polygon_minus_blocker") - min_proj("polygon_minus_blocker"),
    }


def load_phase845_rows(round_name: str, model_name: str) -> list[dict[str, Any]]:
    return p848.load_phase845_rows(round_name, model_name)


def load_phase845_gears(round_name: str, model_name: str) -> list[dict[str, Any]]:
    return p848.load_phase845_gears(round_name, model_name)


def capture_internal_states(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    gears: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    layer_indices = selected_layer_indices(model, gears, include_neighbors=not args.no_neighbor_layers)
    unique_prompts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        unique_prompts[prompt_key(row)] = row
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for idx, (key, row) in enumerate(unique_prompts.items(), 1):
        out[key] = capture_prompt_state(model, tokenizer, device, row, gears, layer_indices, int(args.topk_entropy))
        if idx % max(1, int(args.log_every)) == 0 or idx == len(unique_prompts):
            log(
                f"{args.model}/{args.round_name}: captured multi-source gate state "
                f"{idx}/{len(unique_prompts)} layers={layer_indices}"
            )
    return out


def make_feature_rows(rows: list[dict[str, Any]], prompt_states: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        state = prompt_states.get(prompt_key(row), {})
        acts_by_key = state.get("gear_acts") or {}
        gkeys = [str(key) for key in row.get("gear_keys") or p846.split_combo_key(str(row.get("combo_key")))]
        acts = [float(acts_by_key.get(key, 0.0)) for key in gkeys]
        signs = [activation_sign(value) for value in acts]
        abs_values = [abs(value) for value in acts]
        pos_count = sum(1 for sign in signs if sign == "+")
        neg_count = sum(1 for sign in signs if sign == "-")
        zero_count = sum(1 for sign in signs if sign == "0")
        feature_rows.append(
            {
                "row_kind": "phase849_residual_blocker_route_gate_feature",
                "phase": PHASE,
                "source_phase": 845,
                "model": row.get("model"),
                "source_round": row.get("round"),
                "case_id": row.get("case_id"),
                "object": row.get("object"),
                "prompt_variant": row.get("prompt_variant"),
                "prompt": row.get("prompt"),
                "combo_type": row.get("combo_type"),
                "edit_mode": row.get("edit_mode"),
                "combo_key": row.get("combo_key"),
                "gear_keys": gkeys,
                "gear_count": len(gkeys),
                "activation_values": acts,
                "activation_signs": signs,
                "sign_pattern": "".join(signs),
                "pos_count": pos_count,
                "neg_count": neg_count,
                "zero_count": zero_count,
                "signed_sum": sum(acts),
                "signed_mean": finite(mean(acts)),
                "abs_sum": sum(abs_values),
                "abs_mean": finite(mean(abs_values)),
                "max_abs": max(abs_values) if abs_values else 0.0,
                "min_abs": min(abs_values) if abs_values else 0.0,
                "original_margin": finite(row.get("original_target_minus_object_logit")),
                "actual_residual": finite(row.get("interaction_residual")),
                "actual_delta": finite(row.get("margin_delta_vs_original")),
                "expected_additive_delta": finite(row.get("expected_additive_delta")),
                "actual_class": residual_class(finite(row.get("interaction_residual")), 0.5),
                "target_transition": bool(row.get("target_transition")),
                "target_gained_vs_original": bool(row.get("target_gained_vs_original")),
                "target_lost_vs_original": bool(row.get("target_lost_vs_original")),
                "target_id": state.get("target_id"),
                "polygon_id": state.get("polygon_id"),
                "object_id": state.get("object_id"),
                "blocker_id": state.get("blocker_id"),
                "baseline_id": state.get("baseline_id"),
                "best_target_rank": state.get("best_target_rank"),
                "object_rank": state.get("object_rank"),
                "blocker_rank": state.get("blocker_rank"),
                "top1_id": state.get("top1_id"),
                "top1_text": state.get("top1_text"),
                "top1_role": state.get("top1_role"),
                "topk_entropy": finite(state.get("topk_entropy")),
                "target_minus_blocker_logit": finite(state.get("target_minus_blocker_logit")),
                "target_minus_object_logit": finite(state.get("target_minus_object_logit")),
                "object_minus_blocker_logit": finite(state.get("object_minus_blocker_logit")),
                "blocker_pressure": finite(state.get("blocker_pressure")),
                "object_echo_pressure": finite(state.get("object_echo_pressure")),
                "route_gap": finite(state.get("route_gap")),
                "target_blocker_resid_final": finite(state.get("target_blocker_resid_final")),
                "polygon_blocker_resid_final": finite(state.get("polygon_blocker_resid_final")),
                "best_target_blocker_resid_final": finite(state.get("best_target_blocker_resid_final")),
                "best_target_blocker_resid_max": finite(state.get("best_target_blocker_resid_max")),
                "best_target_blocker_resid_min": finite(state.get("best_target_blocker_resid_min")),
                "best_target_object_resid_final": finite(state.get("best_target_object_resid_final")),
                "object_blocker_resid_final": finite(state.get("object_blocker_resid_final")),
                "resid_target_blocker_span": finite(state.get("resid_target_blocker_span")),
                "resid_polygon_blocker_span": finite(state.get("resid_polygon_blocker_span")),
                "layer_indices": state.get("layer_indices") or [],
            }
        )
    return feature_rows


class MultiGateResidualPredictor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        numeric_names = [
            "abs_sum",
            "best_target_blocker_resid_final",
            "best_target_object_resid_final",
            "object_blocker_resid_final",
            "target_minus_blocker_logit",
            "object_echo_pressure",
            "blocker_pressure",
            "route_gap",
            "topk_entropy",
        ]
        self.thresholds: dict[str, tuple[float, float]] = {}
        for name in numeric_names:
            values = [finite(row.get(name)) for row in rows]
            self.thresholds[name] = (quantile(values, 1 / 3), quantile(values, 2 / 3))
        self.tables: dict[str, dict[tuple[Any, ...], float]] = {}
        for name in PREDICTORS:
            self.tables[name] = self._fit(rows, lambda row, predictor=name: self.key(row, predictor))
        self.type_mode = self._fit(rows, lambda row: (row.get("combo_type"), row.get("edit_mode")))
        self.global_mean = finite(mean([finite(row.get("actual_residual")) for row in rows]))

    def bucket(self, row: dict[str, Any], name: str) -> str:
        low, high = self.thresholds.get(name, (0.0, 0.0))
        value = finite(row.get(name))
        if value <= low:
            return "low"
        if value <= high:
            return "mid"
        return "high"

    def base(self, row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("combo_type"), row.get("edit_mode"), row.get("combo_key"))

    def key(self, row: dict[str, Any], predictor: str) -> tuple[Any, ...]:
        base = self.base(row)
        if predictor == "global_combo":
            return base
        if predictor == "internal_strength_combo":
            return (*base, row.get("sign_pattern"), self.bucket(row, "abs_sum"))
        if predictor == "residual_projection_combo":
            return (
                *base,
                self.bucket(row, "best_target_blocker_resid_final"),
                self.bucket(row, "best_target_object_resid_final"),
                self.bucket(row, "object_blocker_resid_final"),
            )
        if predictor == "blocker_field_combo":
            return (
                *base,
                self.bucket(row, "target_minus_blocker_logit"),
                self.bucket(row, "object_echo_pressure"),
                rank_bucket(as_int(row.get("best_target_rank"))),
            )
        if predictor == "route_competition_combo":
            return (
                *base,
                row.get("top1_role"),
                self.bucket(row, "route_gap"),
                self.bucket(row, "topk_entropy"),
            )
        if predictor == "joint_gate_combo":
            return (
                *base,
                row.get("sign_pattern"),
                self.bucket(row, "abs_sum"),
                self.bucket(row, "best_target_blocker_resid_final"),
                self.bucket(row, "target_minus_blocker_logit"),
                self.bucket(row, "route_gap"),
            )
        if predictor == "compact_joint_gate_combo":
            return (
                *base,
                self.bucket(row, "abs_sum"),
                self.bucket(row, "best_target_blocker_resid_final"),
                self.bucket(row, "target_minus_blocker_logit"),
            )
        raise ValueError(f"unknown predictor: {predictor}")

    def _fit(self, rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[tuple[Any, ...], float]:
        groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(finite(row.get("actual_residual")))
        return {key: finite(mean(values)) for key, values in groups.items()}

    def predict(self, row: dict[str, Any], predictor: str) -> float:
        fallback_order: dict[str, list[str]] = {
            "joint_gate_combo": [
                "joint_gate_combo",
                "compact_joint_gate_combo",
                "internal_strength_combo",
                "residual_projection_combo",
                "blocker_field_combo",
                "global_combo",
            ],
            "compact_joint_gate_combo": [
                "compact_joint_gate_combo",
                "internal_strength_combo",
                "residual_projection_combo",
                "blocker_field_combo",
                "global_combo",
            ],
            "residual_projection_combo": ["residual_projection_combo", "global_combo"],
            "blocker_field_combo": ["blocker_field_combo", "global_combo"],
            "route_competition_combo": ["route_competition_combo", "global_combo"],
            "internal_strength_combo": ["internal_strength_combo", "global_combo"],
            "global_combo": ["global_combo"],
        }
        for name in fallback_order.get(predictor, [predictor, "global_combo"]):
            table = self.tables[name]
            key = self.key(row, name)
            if key in table:
                return table[key]
        type_mode_key = (row.get("combo_type"), row.get("edit_mode"))
        if type_mode_key in self.type_mode:
            return self.type_mode[type_mode_key]
        return self.global_mean


def eval_predictions(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return p848.eval_predictions(rows, threshold)


def split_specs(rows: list[dict[str, Any]], split_types: list[str]):
    return p848.split_specs(rows, split_types)


def evaluate_split(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    split_type: str,
    split_key: str,
    threshold: float,
) -> dict[str, Any]:
    predictor = MultiGateResidualPredictor(train_rows)
    predictions: list[dict[str, Any]] = []
    for predictor_name in PREDICTORS:
        for row in test_rows:
            pred = predictor.predict(row, predictor_name)
            predictions.append(
                {
                    "row_kind": "phase849_residual_blocker_route_gate_prediction",
                    "phase": PHASE,
                    "model": row.get("model"),
                    "source_round": row.get("source_round"),
                    "split_type": split_type,
                    "split_key": split_key,
                    "predictor": predictor_name,
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "combo_type": row.get("combo_type"),
                    "edit_mode": row.get("edit_mode"),
                    "combo_key": row.get("combo_key"),
                    "sign_pattern": row.get("sign_pattern"),
                    "abs_sum": row.get("abs_sum"),
                    "best_target_blocker_resid_final": row.get("best_target_blocker_resid_final"),
                    "target_minus_blocker_logit": row.get("target_minus_blocker_logit"),
                    "blocker_pressure": row.get("blocker_pressure"),
                    "route_gap": row.get("route_gap"),
                    "top1_role": row.get("top1_role"),
                    "actual_residual": row.get("actual_residual"),
                    "predicted_residual": pred,
                    "actual_class": residual_class(finite(row.get("actual_residual")), threshold),
                    "predicted_class": residual_class(pred, threshold),
                    "abs_error": abs(pred - finite(row.get("actual_residual"))),
                }
            )
    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_predictor[str(row["predictor"])].append(row)
    summary = {name: eval_predictions(rows, threshold) for name, rows in sorted(by_predictor.items())}
    global_mae = summary.get("global_combo", {}).get("mae")
    gains = {
        name: None if global_mae is None or stats.get("mae") is None else float(global_mae - stats["mae"])
        for name, stats in summary.items()
        if name != "global_combo"
    }
    return {
        "split_type": split_type,
        "split_key": split_key,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "predictor_summary": summary,
        "mae_gain_vs_global_combo": gains,
        "predictions": predictions,
    }


def aggregate(split_results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return p848.aggregate(split_results, threshold)


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "sign_patterns": dict(Counter(str(row.get("sign_pattern")) for row in rows)),
        "top1_roles": dict(Counter(str(row.get("top1_role")) for row in rows)),
        "target_rank_buckets": dict(Counter(rank_bucket(as_int(row.get("best_target_rank"))) for row in rows)),
        "residual_classes": dict(Counter(str(row.get("actual_class")) for row in rows)),
        "mean_abs_sum": mean([finite(row.get("abs_sum")) for row in rows]),
        "mean_target_minus_blocker_logit": mean([finite(row.get("target_minus_blocker_logit")) for row in rows]),
        "mean_blocker_pressure": mean([finite(row.get("blocker_pressure")) for row in rows]),
        "mean_route_gap": mean([finite(row.get("route_gap")) for row in rows]),
        "mean_best_target_blocker_resid_final": mean([finite(row.get("best_target_blocker_resid_final")) for row in rows]),
        "mean_best_target_object_resid_final": mean([finite(row.get("best_target_object_resid_final")) for row in rows]),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_phase845_rows(args.phase845_round, args.model)
    gears = load_phase845_gears(args.phase845_round, args.model)[: int(args.top_gears)]
    if args.max_rows and int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "phase845_round": args.phase845_round,
            "rows": len(rows),
            "gears": [gear["gear_key"] for gear in gears],
            "unique_prompts": len({prompt_key(row) for row in rows}),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt_states = capture_internal_states(model, tokenizer, device, rows, gears, args)
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    feature_rows = make_feature_rows(rows, prompt_states)
    split_types = [part.strip() for part in args.split_types.split(",") if part.strip()]
    split_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    specs = split_specs(feature_rows, split_types)
    for idx, (split_type, split_key, train_rows, test_rows) in enumerate(specs, 1):
        result = evaluate_split(train_rows, test_rows, split_type, split_key, float(args.interaction_threshold))
        split_results.append(result)
        all_predictions.extend(result["predictions"])
        if idx % max(1, int(args.log_every)) == 0 or idx == len(specs):
            log(f"{args.model}/{args.round_name}: evaluated split {idx}/{len(specs)} {split_type}:{split_key}")

    summary = {
        "phase": PHASE,
        "title": "Residual-stream / Blocker-field Route Gate Expansion",
        "model": args.model,
        "round": args.round_name,
        "phase845_round": args.phase845_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_feature_rows": len(feature_rows),
        "n_predictions": len(all_predictions),
        "n_unique_prompts": len(prompt_states),
        "top_gears": gears,
        "feature_summary": feature_summary(feature_rows),
        "split_summary": aggregate(split_results, float(args.interaction_threshold)),
        "boundary": (
            "This phase expands Phase 848 from MLP gear activation gates to residual-stream, blocker-field, "
            "and route-competition gate features. It is a route-gate expansion probe, not geometry closure."
        ),
    }
    p846.write_jsonl(out_dir / f"phase849_{args.model}_feature_rows.jsonl", feature_rows)
    p846.write_jsonl(out_dir / f"phase849_{args.model}_predictions.jsonl", all_predictions)
    p846.write_json(out_dir / f"phase849_{args.model}_split_results.json", [{k: v for k, v in r.items() if k != "predictions"} for r in split_results])
    p846.write_json(out_dir / f"phase849_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "feature_rows": len(feature_rows),
                "unique_prompts": len(prompt_states),
                "feature_summary": summary["feature_summary"],
                "split_summary": {
                    split: {
                        predictor: {
                            "n": stats.get("n"),
                            "mae": stats.get("mae"),
                            "sign_accuracy": stats.get("sign_accuracy"),
                            "strong_f1": (stats.get("strong") or {}).get("f1"),
                        }
                        for predictor, stats in split_row.get("predictor_summary", {}).items()
                    }
                    for split, split_row in summary["split_summary"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 849 Residual-stream / Blocker-field Route Gate Expansion ({payload['round']})",
        "",
        "- Source: Phase 845 residual rows plus fresh natural residual/logit/activation captures.",
        "- Method: compare MLP-only, residual-projection, blocker-field, route-competition, and joint gate predictors.",
        "- Boundary: internal route gate expansion probe; not geometry closure.",
        "",
        "## Model Summary",
        "",
        "| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for split, split_row in (data.get("split_summary") or {}).items():
            gains = split_row.get("mean_mae_gain_vs_global_combo") or {}
            for predictor, stats in (split_row.get("predictor_summary") or {}).items():
                lines.append(
                    f"| {model_name} | {data.get('n_feature_rows', 0)} | {data.get('n_unique_prompts', 0)} | "
                    f"`{split}` | `{predictor}` | {stats.get('n', 0)} | {fmt(stats.get('mae'))} | "
                    f"{fmt(stats.get('sign_accuracy'))} | {fmt((stats.get('strong') or {}).get('f1'))} | "
                    f"{fmt(gains.get(predictor))} |"
                )
    lines += ["", "## Feature Summary", ""]
    lines += [
        "| model | top1 roles | target rank buckets | residual classes | mean target-blocker logit | mean blocker pressure | mean residual target-blocker |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        fs = data.get("feature_summary") or {}
        lines.append(
            f"| {model_name} | `{json.dumps(fs.get('top1_roles') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(fs.get('target_rank_buckets') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(fs.get('residual_classes') or {}, ensure_ascii=False)}` | "
            f"{fmt(fs.get('mean_target_minus_blocker_logit'))} | "
            f"{fmt(fs.get('mean_blocker_pressure'))} | "
            f"{fmt(fs.get('mean_best_target_blocker_resid_final'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase849_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = p846.read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase849_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase849_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase845-round", default="smoke")
    parser.add_argument("--top-gears", type=int, default=6)
    parser.add_argument("--split-types", default="in_sample,object_holdout,prompt_holdout")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--topk-entropy", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-neighbor-layers", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)


if __name__ == "__main__":
    main()
