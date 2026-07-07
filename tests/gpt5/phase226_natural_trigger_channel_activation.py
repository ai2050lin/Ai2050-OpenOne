#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

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


PHASE = 226
SOURCE_PHASE = 225
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase226_natural_trigger_channel_activation")


SPECS = {
    "qwen3": [
        {
            "spec_id": "qwen3_explain_l29_natural_trigger",
            "pattern_id": "answer_explain",
            "source_layers": [29],
            "observe_layers": [11, 29, 31, 33],
        },
    ],
    "glm4": [
        {
            "spec_id": "glm4_repeat_l30_natural_trigger",
            "pattern_id": "answer_repeat",
            "source_layers": [30],
            "observe_layers": [12, 28, 30, 32],
        },
    ],
    "deepseek7b": [
        {
            "spec_id": "deepseek7b_explain_l24_natural_trigger",
            "pattern_id": "answer_explain",
            "source_layers": [24],
            "observe_layers": [20, 24, 26, 27],
        },
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


def split_prompt(prompt: str) -> tuple[str, str, str]:
    lines = str(prompt).splitlines()
    question = lines[0] if lines else str(prompt)
    answer_anchor = "Answer:"
    instruction_lines = [line for line in lines[1:] if line.strip() and line.strip() != answer_anchor]
    instruction = instruction_lines[0] if instruction_lines else ""
    return question, instruction, answer_anchor


def prompt_variants(prompt: str, pattern_id: str) -> dict[str, str]:
    question, instruction, answer_anchor = split_prompt(prompt)
    explain_instruction = "Answer with the answer first, then one short reason using because."
    repeat_instruction = "Answer with exactly the same answer word twice, separated by a comma."
    short_instruction = "Answer with a short answer."
    variants = {
        "full": str(prompt),
        "no_instruction": f"{question}\n{answer_anchor}",
        "short_answer_instruction": f"{question}\n{short_instruction}\n{answer_anchor}",
        "no_answer_anchor": f"{question}\n{instruction}".rstrip(),
    }
    if pattern_id == "answer_explain":
        variants["repeat_instruction"] = f"{question}\n{repeat_instruction}\n{answer_anchor}"
        variants["because_removed"] = f"{question}\nAnswer with the answer first, then one short reason.\n{answer_anchor}"
    elif pattern_id == "answer_repeat":
        variants["explain_instruction"] = f"{question}\n{explain_instruction}\n{answer_anchor}"
        variants["comma_removed"] = f"{question}\nAnswer with exactly the same answer word twice.\n{answer_anchor}"
    return variants


def prefix_variant(row: dict[str, Any], variant_prompt: str, step: int) -> str:
    emitted = row.get("emitted_tokens") or []
    prefix_tokens = emitted[: max(0, int(step) - 1)]
    return str(variant_prompt) + "".join(str(tok) for tok in prefix_tokens)


def capture_z_hidden_logits(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    source_layers: list[int],
    observe_layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    layers = get_layers(model)
    captured_z: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in source_layers:
        down_proj = p221.get_down_proj(layers[int(layer_idx)])
        if down_proj is None:
            continue

        def make_hook(li: int):
            def hook(_module: Any, inputs: tuple[Any, ...]):
                if inputs and torch.is_tensor(inputs[0]):
                    captured_z[int(li)] = inputs[0][0, -1, :].detach().float().cpu()
                return None

            return hook

        handles.append(down_proj.register_forward_pre_hook(make_hook(int(layer_idx))))
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
    return captured_z, hidden, logits


def activation_axis_value(z: torch.Tensor, success: torch.Tensor, drift: torch.Tensor, channels: list[int]) -> tuple[float, float, float]:
    if not channels:
        return 0.0, 0.0, 0.0
    idx = torch.tensor(channels, dtype=torch.long)
    zc = z[idx].float()
    sc = success[idx].float()
    dc = drift[idx].float()
    denom = sc - dc
    good = torch.abs(denom) > 1e-6
    if bool(good.any()):
        axis = ((zc[good] - dc[good]) / denom[good]).mean().item()
    else:
        axis = 0.0
    dist_success = torch.mean(torch.abs(zc - sc)).item()
    dist_drift = torch.mean(torch.abs(zc - dc)).item()
    return float(axis), float(dist_success), float(dist_drift)


def build_trigger_rows(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    model_name: str,
    spec: dict[str, Any],
    source_group: str,
    rows: list[dict[str, Any]],
    selected: dict[str, dict[int, dict[int, list[int]]]],
    success_z: dict[int, dict[int, torch.Tensor]],
    drift_z: dict[int, dict[int, torch.Tensor]],
    residual_dirs: dict[int, dict[int, torch.Tensor]],
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    activation_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    source_layers = [int(x) for x in spec["source_layers"]]
    observe_layers = [int(x) for x in spec["observe_layers"]]
    for row in rows:
        variants = prompt_variants(str(row.get("prompt") or ""), str(spec["pattern_id"]))
        full_metrics_by_step: dict[int, dict[str, Any]] = {}
        for step in range(1, int(max_steps) + 1):
            full_z, full_hidden, full_logits = capture_z_hidden_logits(
                model,
                tokenizer,
                device,
                prefix_variant(row, variants["full"], int(step)),
                source_layers,
                observe_layers,
            )
            full_metrics_by_step[int(step)] = p204.metric_for_logits(tokenizer, full_logits, row, groups)
            full_hidden_projection: dict[int, float] = {}
            for layer_idx, vec in full_hidden.items():
                direction = residual_dirs.get(int(step), {}).get(int(layer_idx))
                if direction is not None:
                    norm = torch.linalg.vector_norm(direction).item()
                    if norm > 0:
                        full_hidden_projection[int(layer_idx)] = float(torch.dot(vec, direction / norm).item())

            for variant_name, variant_prompt in variants.items():
                if variant_name == "full":
                    zmap, hidden_map, logits = full_z, full_hidden, full_logits
                else:
                    zmap, hidden_map, logits = capture_z_hidden_logits(
                        model,
                        tokenizer,
                        device,
                        prefix_variant(row, variant_prompt, int(step)),
                        source_layers,
                        observe_layers,
                    )
                metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
                for layer_idx in source_layers:
                    z = zmap.get(int(layer_idx))
                    s = success_z.get(int(step), {}).get(int(layer_idx))
                    d = drift_z.get(int(step), {}).get(int(layer_idx))
                    if z is None or s is None or d is None:
                        continue
                    for k in [4, 16, 64]:
                        channels = selected.get("pos", {}).get(int(step), {}).get(int(layer_idx), [])[:k]
                        axis, dist_s, dist_d = activation_axis_value(z, s, d, channels)
                        activation_rows.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase226_trigger_activation_row",
                                "model": model_name,
                                "spec_id": spec["spec_id"],
                                "pattern_id": spec["pattern_id"],
                                "source_group": source_group,
                                "variant": variant_name,
                                "trajectory_id": row.get("trajectory_id"),
                                "step": int(step),
                                "source_layer": int(layer_idx),
                                "k": int(k),
                                "activation_axis": axis,
                                "dist_to_success": dist_s,
                                "dist_to_drift": dist_d,
                                "success_closer": dist_s < dist_d,
                            }
                        )
                for layer_idx, vec in hidden_map.items():
                    direction = residual_dirs.get(int(step), {}).get(int(layer_idx))
                    if direction is None:
                        continue
                    norm = torch.linalg.vector_norm(direction).item()
                    if norm <= 0:
                        continue
                    projection = float(torch.dot(vec, direction / norm).item())
                    hidden_rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase226_trigger_hidden_projection_row",
                            "model": model_name,
                            "spec_id": spec["spec_id"],
                            "pattern_id": spec["pattern_id"],
                            "source_group": source_group,
                            "variant": variant_name,
                            "trajectory_id": row.get("trajectory_id"),
                            "step": int(step),
                            "observe_layer": int(layer_idx),
                            "projection_to_success_drift_dir": projection,
                            "projection_delta_from_full": projection - full_hidden_projection.get(int(layer_idx), projection),
                        }
                    )
                full_metrics = full_metrics_by_step[int(step)]
                readout_rows.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase226_trigger_readout_row",
                        "model": model_name,
                        "spec_id": spec["spec_id"],
                        "pattern_id": spec["pattern_id"],
                        "source_group": source_group,
                        "variant": variant_name,
                        "trajectory_id": row.get("trajectory_id"),
                        "step": int(step),
                        "top_token_id": int(metrics.get("top_token_id") or -1),
                        "top_token": str(metrics.get("top_token") or ""),
                        "target_rank": metrics.get("target_rank"),
                        "prose_margin": metrics.get("prose_margin"),
                        "echo_margin": metrics.get("echo_margin"),
                        "stop_margin": metrics.get("stop_margin"),
                        "top_token_changed_from_full": int(metrics.get("top_token_id") or -1) != int(full_metrics.get("top_token_id") or -1),
                        "target_rank_delta_from_full": finite_float(full_metrics.get("target_rank")) - finite_float(metrics.get("target_rank")),
                        "prose_margin_delta_from_full": finite_float(metrics.get("prose_margin")) - finite_float(full_metrics.get("prose_margin")),
                        "echo_margin_delta_from_full": finite_float(metrics.get("echo_margin")) - finite_float(full_metrics.get("echo_margin")),
                    }
                )
    return activation_rows, hidden_rows, readout_rows


def summarize_activation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("spec_id"), row.get("source_group"), row.get("variant"), row.get("step"), row.get("source_layer"), row.get("k"))].append(row)
    full_means: dict[tuple[Any, ...], float] = {}
    for key, items in buckets.items():
        spec_id, source_group, variant, step, layer, k = key
        if variant == "full":
            full_means[(spec_id, source_group, step, layer, k)] = sum(finite_float(x.get("activation_axis")) for x in items) / len(items)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, variant, step, layer, k = key
        axis_values = [finite_float(x.get("activation_axis")) for x in items]
        mean_axis = sum(axis_values) / len(axis_values) if axis_values else 0.0
        full_axis = full_means.get((spec_id, source_group, step, layer, k), mean_axis)
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "variant": variant,
                "step": int(step),
                "source_layer": int(layer),
                "k": int(k),
                "rows": len(items),
                "mean_activation_axis": mean_axis,
                "delta_axis_from_full": mean_axis - full_axis,
                "success_closer": sum(1 for x in items if x.get("success_closer")),
                "mean_dist_to_success": sum(finite_float(x.get("dist_to_success")) for x in items) / len(items),
                "mean_dist_to_drift": sum(finite_float(x.get("dist_to_drift")) for x in items) / len(items),
            }
        )
    out.sort(key=lambda row: abs(float(row.get("delta_axis_from_full") or 0.0)), reverse=True)
    return out


def summarize_hidden(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("spec_id"), row.get("source_group"), row.get("variant"), row.get("step"), row.get("observe_layer"))].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, variant, step, layer = key
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "variant": variant,
                "step": int(step),
                "observe_layer": int(layer),
                "rows": len(items),
                "mean_projection": sum(finite_float(x.get("projection_to_success_drift_dir")) for x in items) / len(items),
                "mean_projection_delta_from_full": sum(finite_float(x.get("projection_delta_from_full")) for x in items) / len(items),
            }
        )
    out.sort(key=lambda row: abs(float(row.get("mean_projection_delta_from_full") or 0.0)), reverse=True)
    return out


def summarize_readout(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("spec_id"), row.get("source_group"), row.get("variant"), row.get("step"))].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, variant, step = key
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "variant": variant,
                "step": int(step),
                "rows": len(items),
                "top_token_changed_from_full": sum(1 for x in items if x.get("top_token_changed_from_full")),
                "mean_target_rank_delta_from_full": sum(finite_float(x.get("target_rank_delta_from_full")) for x in items) / len(items),
                "mean_prose_margin_delta_from_full": sum(finite_float(x.get("prose_margin_delta_from_full")) for x in items) / len(items),
                "mean_echo_margin_delta_from_full": sum(finite_float(x.get("echo_margin_delta_from_full")) for x in items) / len(items),
                "top_tokens": dict(Counter(str(x.get("top_token")) for x in items).most_common(8)),
            }
        )
    out.sort(
        key=lambda row: abs(float(row.get("mean_target_rank_delta_from_full") or 0.0)) + int(row.get("top_token_changed_from_full") or 0),
        reverse=True,
    )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    activation_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        rows = load_rows(args.model, args.phase210_round)
        for spec in SPECS[args.model]:
            success_rows, drift_rows = p219.select_rows(rows, str(spec["pattern_id"]), int(args.max_filter_rows))
            success_rows = success_rows[: int(args.max_direction_rows)]
            drift_rows = drift_rows[: int(args.max_direction_rows)]
            filter_rows.append(
                {
                    "phase": PHASE,
                    "row_kind": "phase226_source_row_count",
                    "model": args.model,
                    "spec_id": spec["spec_id"],
                    "pattern_id": spec["pattern_id"],
                    "success_rows": len(success_rows),
                    "drift_rows": len(drift_rows),
                }
            )
            if not success_rows or not drift_rows:
                log(f"{args.model}|{spec['spec_id']}: insufficient phase210 success={len(success_rows)} drift={len(drift_rows)}")
                continue
            source_layers = [int(x) for x in spec["source_layers"]]
            observe_layers = [int(x) for x in spec["observe_layers"]]
            all_layers = sorted(set(source_layers + observe_layers))
            residual_dirs = p219.build_direction_vectors(
                model, tokenizer, device, success_rows, drift_rows, all_layers, int(args.max_steps)
            )
            success_z = p221.mean_mlp_z(model, tokenizer, device, success_rows, source_layers, int(args.max_steps))
            drift_z = p221.mean_mlp_z(model, tokenizer, device, drift_rows, source_layers, int(args.max_steps))
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
            for source_group, source_items in [("success", success_rows[: int(args.max_eval_rows)]), ("drift", drift_rows[: int(args.max_eval_rows)])]:
                a_rows, h_rows, r_rows = build_trigger_rows(
                    model,
                    tokenizer,
                    device,
                    groups,
                    args.model,
                    spec,
                    source_group,
                    source_items,
                    selected,
                    success_z,
                    drift_z,
                    residual_dirs,
                    int(args.max_steps),
                )
                activation_rows.extend(a_rows)
                hidden_rows.extend(h_rows)
                readout_rows.extend(r_rows)
            log(f"{args.model}|{spec['spec_id']}: activation={len(activation_rows)} hidden={len(hidden_rows)} readout={len(readout_rows)}")
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
    activation_summary = summarize_activation(activation_rows)
    hidden_summary = summarize_hidden(hidden_rows)
    readout_summary = summarize_readout(readout_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Natural trigger to channel activation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "filter_rows": filter_rows,
        "activation_rows": len(activation_rows),
        "hidden_rows": len(hidden_rows),
        "readout_rows": len(readout_rows),
        "channel_score_rows": len(channel_rows),
        "activation_summary_rows": len(activation_summary),
        "hidden_summary_rows": len(hidden_summary),
        "readout_summary_rows": len(readout_summary),
        "top_activation_summary": activation_summary[:80],
        "top_hidden_summary": hidden_summary[:80],
        "top_readout_summary": readout_summary[:80],
    }
    write_json(out_dir / f"phase226_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase226_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase226_{args.model}_activation_rows.jsonl", activation_rows)
    write_jsonl(out_dir / f"phase226_{args.model}_hidden_rows.jsonl", hidden_rows)
    write_jsonl(out_dir / f"phase226_{args.model}_readout_rows.jsonl", readout_rows)
    write_jsonl(out_dir / f"phase226_{args.model}_channel_score_rows.jsonl", channel_rows)
    write_jsonl(out_dir / f"phase226_{args.model}_activation_summary_rows.jsonl", activation_summary)
    write_jsonl(out_dir / f"phase226_{args.model}_hidden_summary_rows.jsonl", hidden_summary)
    write_jsonl(out_dir / f"phase226_{args.model}_readout_summary_rows.jsonl", readout_summary)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "activation_rows": len(activation_rows),
                "hidden_rows": len(hidden_rows),
                "readout_rows": len(readout_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase226_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    activation_summary = []
    hidden_summary = []
    readout_summary = []
    for model in MODELS:
        activation_summary.extend(p214.iter_jsonl(out_dir / f"phase226_{model}_activation_summary_rows.jsonl") or [])
        hidden_summary.extend(p214.iter_jsonl(out_dir / f"phase226_{model}_hidden_summary_rows.jsonl") or [])
        readout_summary.extend(p214.iter_jsonl(out_dir / f"phase226_{model}_readout_summary_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model natural trigger to channel activation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "activation_rows": sum(int(x.get("activation_rows") or 0) for x in summaries),
        "hidden_rows": sum(int(x.get("hidden_rows") or 0) for x in summaries),
        "readout_rows": sum(int(x.get("readout_rows") or 0) for x in summaries),
        "channel_score_rows": sum(int(x.get("channel_score_rows") or 0) for x in summaries),
        "top_activation_summary": sorted(activation_summary, key=lambda row: abs(float(row.get("delta_axis_from_full") or 0.0)), reverse=True)[:100],
        "top_hidden_summary": sorted(hidden_summary, key=lambda row: abs(float(row.get("mean_projection_delta_from_full") or 0.0)), reverse=True)[:100],
        "top_readout_summary": sorted(
            readout_summary,
            key=lambda row: abs(float(row.get("mean_target_rank_delta_from_full") or 0.0)) + int(row.get("top_token_changed_from_full") or 0),
            reverse=True,
        )[:100],
    }
    write_json(out_dir / "phase226_cross_model_summary.json", payload)
    lines = ["# Phase 226 natural trigger to channel activation", ""]
    for key in ["activation_rows", "hidden_rows", "readout_rows", "channel_score_rows"]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(["", "## Activation summary", "", "| spec | group | variant | step | layer | K | axis | delta | success closer |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in payload["top_activation_summary"][:60]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('variant')} | {row.get('step')} | {row.get('source_layer')} | {row.get('k')} | "
            f"{float(row.get('mean_activation_axis') or 0.0):.4f} | {float(row.get('delta_axis_from_full') or 0.0):.4f} | {row.get('success_closer')} |"
        )
    lines.extend(["", "## Hidden summary", "", "| spec | group | variant | step | layer | projection delta |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in payload["top_hidden_summary"][:50]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('variant')} | {row.get('step')} | {row.get('observe_layer')} | "
            f"{float(row.get('mean_projection_delta_from_full') or 0.0):.4f} |"
        )
    lines.extend(["", "## Readout summary", "", "| spec | group | variant | step | top changed | rank delta | prose delta | echo delta | top tokens |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in payload["top_readout_summary"][:50]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('variant')} | {row.get('step')} | "
            f"{row.get('top_token_changed_from_full')} | {float(row.get('mean_target_rank_delta_from_full') or 0.0):.4f} | "
            f"{float(row.get('mean_prose_margin_delta_from_full') or 0.0):.4f} | {float(row.get('mean_echo_margin_delta_from_full') or 0.0):.4f} | {row.get('top_tokens')} |"
        )
    (out_dir / "phase226_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "activation_rows": payload["activation_rows"],
                "hidden_rows": payload["hidden_rows"],
                "readout_rows": payload["readout_rows"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase226 natural trigger to channel activation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="natural_trigger_channel_activation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=16)
    parser.add_argument("--max-direction-rows", type=int, default=8)
    parser.add_argument("--max-eval-rows", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--top-channels", type=int, default=96)
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
