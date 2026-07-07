#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
import phase219_state_write_mlp_causal_validation as p219  # noqa: E402
import phase221_mlp_channel_statewrite_source as p221  # noqa: E402
import phase222_statewrite_factor_competition as p222  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 228
SOURCE_PHASE = 227
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase228_module_tree_gateup_causal_validation")


SPECS = {
    "qwen3": [
        {"spec_id": "qwen3_explain_l29_gateup_causal", "pattern_id": "answer_explain", "source_layers": [29], "observe_layers": [29, 31, 33]},
    ],
    "glm4": [
        {"spec_id": "glm4_repeat_l30_gateup_causal", "pattern_id": "answer_repeat", "source_layers": [30], "observe_layers": [28, 30, 32]},
    ],
    "deepseek7b": [
        {"spec_id": "deepseek7b_explain_l24_gateup_causal", "pattern_id": "answer_explain", "source_layers": [24], "observe_layers": [24, 26, 27]},
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def extract_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def replace_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    return output


def get_mlp(layer: Any) -> Any | None:
    return getattr(layer, "mlp", None)


def mlp_act(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    fn = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_func", None)
    return F.silu(x) if fn is None else fn(x)


def down_project(mlp: Any, z: torch.Tensor) -> torch.Tensor:
    down = getattr(mlp, "down_proj")
    weight = down.weight.detach().float().cpu()
    bias = down.bias.detach().float().cpu() if getattr(down, "bias", None) is not None else None
    return F.linear(z.float().cpu(), weight, bias)


def split_gate_up(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = tensor.shape[-1] // 2
    return tensor[..., :half], tensor[..., half:]


def module_audit_rows(model: Any, model_name: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    layers = get_layers(model)
    for layer_idx in spec["source_layers"]:
        mlp = get_mlp(layers[int(layer_idx)])
        if mlp is None:
            continue
        attrs = {name: hasattr(mlp, name) for name in ["gate_proj", "up_proj", "gate_up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h"]}
        shapes = {}
        for name in attrs:
            module = getattr(mlp, name, None)
            weight = getattr(module, "weight", None)
            if weight is not None:
                shapes[name] = list(weight.shape)
        mlp_type = "unknown"
        if attrs["gate_proj"] and attrs["up_proj"]:
            mlp_type = "split_gate_up"
        elif attrs["gate_up_proj"]:
            mlp_type = "merged_gate_up"
        elif attrs["dense_h_to_4h"]:
            mlp_type = "dense_h_to_4h"
        rows.append(
            {
                "phase": PHASE,
                "row_kind": "phase228_module_tree_audit_row",
                "model": model_name,
                "spec_id": spec["spec_id"],
                "layer_idx": int(layer_idx),
                "mlp_class": type(mlp).__name__,
                "mlp_type": mlp_type,
                "attrs": attrs,
                "weight_shapes": shapes,
            }
        )
    return rows


def capture_internal(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    text: str,
    source_layers: list[int],
    observe_layers: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor], torch.Tensor]:
    layers = get_layers(model)
    captured: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    handles = []
    for layer_idx in source_layers:
        mlp = get_mlp(layers[int(layer_idx)])
        if mlp is None:
            continue

        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        gate_up_proj = getattr(mlp, "gate_up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)

        def gate_hook(li: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                tensor = extract_tensor(output)
                if tensor is not None:
                    captured[int(li)]["gate"] = tensor[0, -1, :].detach().float().cpu()
                return None

            return hook

        def up_hook(li: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                tensor = extract_tensor(output)
                if tensor is not None:
                    captured[int(li)]["up"] = tensor[0, -1, :].detach().float().cpu()
                return None

            return hook

        def fused_hook(li: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                tensor = extract_tensor(output)
                if tensor is not None:
                    gate, up = split_gate_up(tensor[0, -1, :].detach().float().cpu())
                    captured[int(li)]["gate"] = gate
                    captured[int(li)]["up"] = up
                    captured[int(li)]["gate_up_fused"] = tensor[0, -1, :].detach().float().cpu()
                return None

            return hook

        def down_pre_hook(li: int):
            def hook(_module: Any, inputs: tuple[Any, ...]):
                if inputs and torch.is_tensor(inputs[0]):
                    captured[int(li)]["product"] = inputs[0][0, -1, :].detach().float().cpu()
                return None

            return hook

        def down_out_hook(li: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                tensor = extract_tensor(output)
                if tensor is not None:
                    captured[int(li)]["down_out"] = tensor[0, -1, :].detach().float().cpu()
                return None

            return hook

        if gate_proj is not None:
            handles.append(gate_proj.register_forward_hook(gate_hook(int(layer_idx))))
        if up_proj is not None:
            handles.append(up_proj.register_forward_hook(up_hook(int(layer_idx))))
        if gate_up_proj is not None:
            handles.append(gate_up_proj.register_forward_hook(fused_hook(int(layer_idx))))
        if down_proj is not None:
            handles.append(down_proj.register_forward_pre_hook(down_pre_hook(int(layer_idx))))
            handles.append(down_proj.register_forward_hook(down_out_hook(int(layer_idx))))

    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = {
            int(layer_idx): result.hidden_states[int(layer_idx) + 1][0, last_pos].detach().float().cpu()
            for layer_idx in observe_layers
            if int(layer_idx) + 1 < len(result.hidden_states)
        }
        logits = result.logits[0, last_pos].detach().float().cpu()
        del result
    finally:
        for handle in handles:
            handle.remove()
        del input_ids, attention_mask

    for layer_idx in source_layers:
        mlp = get_mlp(layers[int(layer_idx)])
        parts = captured.get(int(layer_idx), {})
        gate = parts.get("gate")
        up = parts.get("up")
        if mlp is not None and gate is not None and up is not None:
            parts["recomputed_product"] = mlp_act(mlp, gate) * up
            if "product" in parts:
                product = parts["product"]
                denom = torch.linalg.vector_norm(product).item() + 1e-6
                parts["product_rel_error"] = torch.tensor(float(torch.linalg.vector_norm(parts["recomputed_product"] - product).item() / denom))
    return captured, hidden, logits


def mean_internal(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    source_layers: list[int],
    observe_layers: list[int],
    max_steps: int,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    bucket: dict[int, dict[int, dict[str, list[torch.Tensor]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for step in range(1, int(max_steps) + 1):
            internal, _hidden, _logits = capture_internal(
                model,
                tokenizer,
                device,
                p219.prefix_for_step(row, int(step)),
                source_layers,
                observe_layers,
            )
            for layer_idx, part_map in internal.items():
                for component, vec in part_map.items():
                    if torch.is_tensor(vec) and vec.ndim == 1 and component != "gate_up_fused":
                        bucket[int(step)][int(layer_idx)][str(component)].append(vec)
    out: dict[int, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    for step, layer_map in bucket.items():
        for layer_idx, part_map in layer_map.items():
            for component, vecs in part_map.items():
                if vecs:
                    out[int(step)][int(layer_idx)][str(component)] = torch.stack(vecs, dim=0).mean(dim=0)
    return out


def run_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    text: str,
    patch_spec: dict[str, Any] | None = None,
) -> torch.Tensor:
    handles = []
    if patch_spec is not None:
        layers = get_layers(model)
        layer_idx = int(patch_spec["layer_idx"])
        mlp = get_mlp(layers[layer_idx])
        if mlp is not None:
            component = str(patch_spec["component"])
            alpha = float(patch_spec["alpha"])

            def add_to_output(vec: torch.Tensor):
                v = vec.to(device=device)

                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                    tensor = extract_tensor(output)
                    if tensor is None:
                        return output
                    new_tensor = tensor.clone()
                    new_tensor[0, -1, :] = new_tensor[0, -1, :] + alpha * v.to(dtype=new_tensor.dtype)
                    return replace_tensor(output, new_tensor)

                return hook

            def add_to_fused(gate_vec: torch.Tensor | None, up_vec: torch.Tensor | None):
                gate_v = gate_vec.to(device=device) if gate_vec is not None else None
                up_v = up_vec.to(device=device) if up_vec is not None else None

                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                    tensor = extract_tensor(output)
                    if tensor is None:
                        return output
                    new_tensor = tensor.clone()
                    half = new_tensor.shape[-1] // 2
                    if gate_v is not None:
                        new_tensor[0, -1, :half] = new_tensor[0, -1, :half] + alpha * gate_v.to(dtype=new_tensor.dtype)
                    if up_v is not None:
                        new_tensor[0, -1, half:] = new_tensor[0, -1, half:] + alpha * up_v.to(dtype=new_tensor.dtype)
                    return replace_tensor(output, new_tensor)

                return hook

            def add_to_down_pre(vec: torch.Tensor):
                v = vec.to(device=device)

                def hook(_module: Any, inputs: tuple[Any, ...]):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    z = inputs[0]
                    z_new = z.clone()
                    z_new[0, -1, :] = z_new[0, -1, :] + alpha * v.to(dtype=z_new.dtype)
                    return (z_new, *inputs[1:])

                return hook

            gate_proj = getattr(mlp, "gate_proj", None)
            up_proj = getattr(mlp, "up_proj", None)
            gate_up_proj = getattr(mlp, "gate_up_proj", None)
            down_proj = getattr(mlp, "down_proj", None)
            if component == "gate":
                if gate_proj is not None:
                    handles.append(gate_proj.register_forward_hook(add_to_output(patch_spec["gate_vec"])))
                elif gate_up_proj is not None:
                    handles.append(gate_up_proj.register_forward_hook(add_to_fused(patch_spec["gate_vec"], None)))
            elif component == "up":
                if up_proj is not None:
                    handles.append(up_proj.register_forward_hook(add_to_output(patch_spec["up_vec"])))
                elif gate_up_proj is not None:
                    handles.append(gate_up_proj.register_forward_hook(add_to_fused(None, patch_spec["up_vec"])))
            elif component == "gate_up_pair":
                if gate_proj is not None and up_proj is not None:
                    handles.append(gate_proj.register_forward_hook(add_to_output(patch_spec["gate_vec"])))
                    handles.append(up_proj.register_forward_hook(add_to_output(patch_spec["up_vec"])))
                elif gate_up_proj is not None:
                    handles.append(gate_up_proj.register_forward_hook(add_to_fused(patch_spec["gate_vec"], patch_spec["up_vec"])))
            elif component == "product" and down_proj is not None:
                handles.append(down_proj.register_forward_pre_hook(add_to_down_pre(patch_spec["product_vec"])))
            elif component == "down_out" and down_proj is not None:
                handles.append(down_proj.register_forward_hook(add_to_output(patch_spec["down_out_vec"])))

    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = result.logits[0, last_pos].detach().float().cpu()
        del result
    finally:
        for handle in handles:
            handle.remove()
        del input_ids, attention_mask
    return logits


def patch_rows_for_group(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    groups: dict[str, list[int]],
    model_name: str,
    spec: dict[str, Any],
    source_group: str,
    rows: list[dict[str, Any]],
    success_internal: dict[int, dict[int, dict[str, torch.Tensor]]],
    drift_internal: dict[int, dict[int, dict[str, torch.Tensor]]],
    selected: dict[str, dict[int, dict[int, list[int]]]],
    max_steps: int,
    alphas: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    direction_sign = 1.0 if source_group == "drift" else -1.0
    for row in rows:
        for step in range(1, int(max_steps) + 1):
            text = p219.prefix_for_step(row, int(step))
            base_logits = run_logits(model, tokenizer, device, text)
            base_metrics = p204.metric_for_logits(tokenizer, base_logits, row, groups)
            for layer_idx in spec["source_layers"]:
                s_map = success_internal.get(int(step), {}).get(int(layer_idx), {})
                d_map = drift_internal.get(int(step), {}).get(int(layer_idx), {})
                if not s_map or not d_map:
                    continue
                delta = {name: direction_sign * (s_map[name] - d_map[name]) for name in s_map.keys() & d_map.keys()}
                channel_sets = {
                    "all": None,
                    "top16": selected.get("pos", {}).get(int(step), {}).get(int(layer_idx), [])[:16],
                    "top64": selected.get("pos", {}).get(int(step), {}).get(int(layer_idx), [])[:64],
                }
                for channel_scope, channels in channel_sets.items():
                    masked_delta = {}
                    for component, vec in delta.items():
                        if component in {"product_rel_error"}:
                            continue
                        v = vec.clone()
                        if channels is not None and v.ndim == 1 and v.shape[0] >= max(channels, default=-1) + 1:
                            mask = torch.zeros_like(v)
                            idx = torch.tensor(channels, dtype=torch.long)
                            mask[idx] = v[idx]
                            v = mask
                        masked_delta[component] = v
                    specs = []
                    if "gate" in masked_delta:
                        specs.append({"component": "gate", "gate_vec": masked_delta["gate"]})
                    if "up" in masked_delta:
                        specs.append({"component": "up", "up_vec": masked_delta["up"]})
                    if "gate" in masked_delta and "up" in masked_delta:
                        specs.append({"component": "gate_up_pair", "gate_vec": masked_delta["gate"], "up_vec": masked_delta["up"]})
                    if "product" in masked_delta:
                        specs.append({"component": "product", "product_vec": masked_delta["product"]})
                    if "down_out" in masked_delta:
                        specs.append({"component": "down_out", "down_out_vec": masked_delta["down_out"]})
                    for patch in specs:
                        for alpha in alphas:
                            patch_spec = dict(patch)
                            patch_spec.update({"layer_idx": int(layer_idx), "alpha": float(alpha)})
                            logits = run_logits(model, tokenizer, device, text, patch_spec)
                            metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
                            out.append(
                                {
                                    "phase": PHASE,
                                    "source_phase": SOURCE_PHASE,
                                    "row_kind": "phase228_gateup_patch_validation_row",
                                    "model": model_name,
                                    "spec_id": spec["spec_id"],
                                    "pattern_id": spec["pattern_id"],
                                    "source_group": source_group,
                                    "trajectory_id": row.get("trajectory_id"),
                                    "step": int(step),
                                    "source_layer": int(layer_idx),
                                    "component": patch["component"],
                                    "channel_scope": channel_scope,
                                    "alpha": float(alpha),
                                    "target_rank_delta": finite_float(base_metrics.get("target_rank")) - finite_float(metrics.get("target_rank")),
                                    "target_logit_delta": finite_float(metrics.get("target_logit")) - finite_float(base_metrics.get("target_logit")),
                                    "prose_margin_delta": finite_float(metrics.get("prose_margin")) - finite_float(base_metrics.get("prose_margin")),
                                    "echo_margin_delta": finite_float(metrics.get("echo_margin")) - finite_float(base_metrics.get("echo_margin")),
                                    "top_token_changed": int(metrics.get("top_token_id") or -1) != int(base_metrics.get("top_token_id") or -1),
                                    "patched_top_token": str(metrics.get("top_token") or ""),
                                    "base_top_token": str(base_metrics.get("top_token") or ""),
                                }
                            )
    return out


def summarize_patch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row.get("spec_id"),
                row.get("source_group"),
                row.get("component"),
                row.get("channel_scope"),
                row.get("alpha"),
                row.get("step"),
                row.get("source_layer"),
            )
        ].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, component, channel_scope, alpha, step, layer = key
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "component": component,
                "channel_scope": channel_scope,
                "alpha": float(alpha),
                "step": int(step),
                "source_layer": int(layer),
                "rows": len(items),
                "mean_target_rank_delta": sum(finite_float(x.get("target_rank_delta")) for x in items) / len(items),
                "mean_target_logit_delta": sum(finite_float(x.get("target_logit_delta")) for x in items) / len(items),
                "mean_prose_margin_delta": sum(finite_float(x.get("prose_margin_delta")) for x in items) / len(items),
                "mean_echo_margin_delta": sum(finite_float(x.get("echo_margin_delta")) for x in items) / len(items),
                "top_token_changed": sum(1 for x in items if x.get("top_token_changed")),
            }
        )
    out.sort(key=lambda row: abs(float(row.get("mean_target_rank_delta") or 0.0)) + abs(float(row.get("mean_target_logit_delta") or 0.0)), reverse=True)
    return out


def summarize_calibration_rows(model_name: str, spec: dict[str, Any], internal: dict[int, dict[int, dict[str, torch.Tensor]]]) -> list[dict[str, Any]]:
    rows = []
    for step, layer_map in internal.items():
        for layer_idx, part_map in layer_map.items():
            product = part_map.get("product")
            recomputed = part_map.get("recomputed_product")
            rel_error = None
            cosine = None
            if product is not None and recomputed is not None:
                denom = torch.linalg.vector_norm(product).item() + 1e-6
                rel_error = float(torch.linalg.vector_norm(recomputed - product).item() / denom)
                cosine = float(F.cosine_similarity(product.float(), recomputed.float(), dim=0).item())
            rows.append(
                {
                    "phase": PHASE,
                    "row_kind": "phase228_product_recompute_calibration_row",
                    "model": model_name,
                    "spec_id": spec["spec_id"],
                    "step": int(step),
                    "source_layer": int(layer_idx),
                    "has_gate": "gate" in part_map,
                    "has_up": "up" in part_map,
                    "has_product": "product" in part_map,
                    "has_down_out": "down_out" in part_map,
                    "product_recompute_rel_error": rel_error,
                    "product_recompute_cosine": cosine,
                }
            )
    return rows


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    audit_rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        rows = load_rows(args.model, args.phase210_round)
        alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]
        for spec in SPECS[args.model]:
            audit_rows.extend(module_audit_rows(model, args.model, spec))
            success_rows, drift_rows = p219.select_rows(rows, str(spec["pattern_id"]), int(args.max_filter_rows))
            success_rows = success_rows[: int(args.max_direction_rows)]
            drift_rows = drift_rows[: int(args.max_direction_rows)]
            filter_rows.append(
                {
                    "phase": PHASE,
                    "row_kind": "phase228_source_row_count",
                    "model": args.model,
                    "spec_id": spec["spec_id"],
                    "pattern_id": spec["pattern_id"],
                    "success_rows": len(success_rows),
                    "drift_rows": len(drift_rows),
                }
            )
            if not success_rows or not drift_rows:
                continue
            source_layers = [int(x) for x in spec["source_layers"]]
            observe_layers = [int(x) for x in spec["observe_layers"]]
            all_layers = sorted(set(source_layers + observe_layers))
            residual_dirs = p219.build_direction_vectors(model, tokenizer, device, success_rows, drift_rows, all_layers, int(args.max_steps))
            success_internal = mean_internal(model, tokenizer, device, success_rows, source_layers, observe_layers, int(args.max_steps))
            drift_internal = mean_internal(model, tokenizer, device, drift_rows, source_layers, observe_layers, int(args.max_steps))
            calibration.extend(summarize_calibration_rows(args.model, spec, success_internal))
            calibration.extend(summarize_calibration_rows(args.model, spec, drift_internal))
            success_z = {step: {layer: part_map["product"] for layer, part_map in layer_map.items() if "product" in part_map} for step, layer_map in success_internal.items()}
            drift_z = {step: {layer: part_map["product"] for layer, part_map in layer_map.items() if "product" in part_map} for step, layer_map in drift_internal.items()}
            score_spec = {"spec_id": spec["spec_id"], "pattern_id": spec["pattern_id"], "layers": source_layers}
            spec_channel_rows, selected, _z_delta = p222.signed_channel_score_rows(
                model,
                args.model,
                score_spec,
                residual_dirs,
                success_z,
                drift_z,
                int(args.max_steps),
                int(args.top_channels),
            )
            channel_rows.extend(spec_channel_rows)
            for source_group, source_items in [("drift", drift_rows[: int(args.max_eval_rows)]), ("success", success_rows[: int(args.max_eval_rows)])]:
                patch_rows.extend(
                    patch_rows_for_group(
                        model,
                        tokenizer,
                        device,
                        groups,
                        args.model,
                        spec,
                        source_group,
                        source_items,
                        success_internal,
                        drift_internal,
                        selected,
                        int(args.max_steps),
                        alphas,
                    )
                )
            log(f"{args.model}|{spec['spec_id']}: patches={len(patch_rows)} audit={len(audit_rows)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    patch_summary = summarize_patch(patch_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Module tree calibrated gate/up causal validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "audit_rows": audit_rows,
        "filter_rows": filter_rows,
        "calibration_rows": len(calibration),
        "patch_rows": len(patch_rows),
        "channel_score_rows": len(channel_rows),
        "patch_summary_rows": len(patch_summary),
        "top_patch_summary": patch_summary[:80],
    }
    write_json(out_dir / f"phase228_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase228_{args.model}_audit_rows.jsonl", audit_rows)
    write_jsonl(out_dir / f"phase228_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase228_{args.model}_calibration_rows.jsonl", calibration)
    write_jsonl(out_dir / f"phase228_{args.model}_channel_score_rows.jsonl", channel_rows)
    write_jsonl(out_dir / f"phase228_{args.model}_patch_rows.jsonl", patch_rows)
    write_jsonl(out_dir / f"phase228_{args.model}_patch_summary_rows.jsonl", patch_summary)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "patch_rows": len(patch_rows)}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase228_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    audit_rows = []
    calibration_rows_all = []
    patch_summary = []
    for model in MODELS:
        audit_rows.extend(p214.iter_jsonl(out_dir / f"phase228_{model}_audit_rows.jsonl") or [])
        calibration_rows_all.extend(p214.iter_jsonl(out_dir / f"phase228_{model}_calibration_rows.jsonl") or [])
        patch_summary.extend(p214.iter_jsonl(out_dir / f"phase228_{model}_patch_summary_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model module-tree calibrated gate/up causal validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "patch_rows": sum(int(x.get("patch_rows") or 0) for x in summaries),
        "channel_score_rows": sum(int(x.get("channel_score_rows") or 0) for x in summaries),
        "audit_rows": audit_rows,
        "top_calibration_rows": sorted(calibration_rows_all, key=lambda row: finite_float(row.get("product_recompute_rel_error"), 999.0))[:50],
        "top_patch_summary": sorted(
            patch_summary,
            key=lambda row: abs(float(row.get("mean_target_rank_delta") or 0.0)) + abs(float(row.get("mean_target_logit_delta") or 0.0)),
            reverse=True,
        )[:100],
    }
    write_json(out_dir / "phase228_cross_model_summary.json", payload)
    lines = ["# Phase 228 module-tree calibrated gate/up causal validation", ""]
    lines.append(f"patch_rows: {payload['patch_rows']}")
    lines.append(f"channel_score_rows: {payload['channel_score_rows']}")
    lines.extend(["", "## Module audit", "", "| model | spec | layer | mlp type | attrs | shapes |", "| --- | --- | ---: | --- | --- | --- |"])
    for row in audit_rows:
        lines.append(f"| {row.get('model')} | {row.get('spec_id')} | {row.get('layer_idx')} | {row.get('mlp_type')} | {row.get('attrs')} | {row.get('weight_shapes')} |")
    lines.extend(["", "## Product recompute calibration", "", "| model | spec | step | layer | gate | up | product | down_out | rel error | cosine |", "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |"])
    for row in payload["top_calibration_rows"][:30]:
        lines.append(
            f"| {row.get('model')} | {row.get('spec_id')} | {row.get('step')} | {row.get('source_layer')} | {row.get('has_gate')} | {row.get('has_up')} | {row.get('has_product')} | {row.get('has_down_out')} | "
            f"{finite_float(row.get('product_recompute_rel_error')):.6f} | {finite_float(row.get('product_recompute_cosine')):.6f} |"
        )
    lines.extend(["", "## Patch summary", "", "| spec | group | component | scope | alpha | step | layer | rows | rank delta | logit delta | top changed |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in payload["top_patch_summary"][:70]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('component')} | {row.get('channel_scope')} | {row.get('alpha')} | {row.get('step')} | {row.get('source_layer')} | {row.get('rows')} | "
            f"{finite_float(row.get('mean_target_rank_delta')):.4f} | {finite_float(row.get('mean_target_logit_delta')):.4f} | {row.get('top_token_changed')} |"
        )
    (out_dir / "phase228_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"phase": PHASE, "status": "complete", "models": payload["models"], "patch_rows": payload["patch_rows"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase228 module-tree calibrated gate/up causal validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="module_tree_gateup_causal_validation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=16)
    parser.add_argument("--max-direction-rows", type=int, default=6)
    parser.add_argument("--max-eval-rows", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--top-channels", type=int, default=96)
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
