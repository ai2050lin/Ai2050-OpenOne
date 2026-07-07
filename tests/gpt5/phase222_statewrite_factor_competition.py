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
import phase209_pattern_running_contrast_atlas as p209  # noqa: E402
import phase212_switchpoint_causal_validation as p212  # noqa: E402
import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
import phase219_state_write_mlp_causal_validation as p219  # noqa: E402
import phase221_mlp_channel_statewrite_source as p221  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 222
SOURCE_PHASE = 221
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase222_statewrite_factor_competition")


SPECS = {
    "qwen3": [
        {"spec_id": "qwen3_explain_l29_l31_signed_channel_split", "pattern_id": "answer_explain", "layers": [29, 31]},
        {"spec_id": "qwen3_repeat_l31_signed_channel_split", "pattern_id": "answer_repeat", "layers": [31]},
    ],
    "glm4": [
        {"spec_id": "glm4_repeat_l28_l30_signed_channel_split", "pattern_id": "answer_repeat", "layers": [28, 30]},
    ],
    "deepseek7b": [
        {"spec_id": "deepseek7b_explain_l24_signed_channel_split", "pattern_id": "answer_explain", "layers": [24]},
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


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def signed_channel_score_rows(
    model,
    model_name: str,
    spec: dict[str, Any],
    residual_dirs: dict[int, dict[int, torch.Tensor]],
    success_z: dict[int, dict[int, torch.Tensor]],
    drift_z: dict[int, dict[int, torch.Tensor]],
    max_steps: int,
    top_channels: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[int, dict[int, list[int]]]],
    dict[int, dict[int, torch.Tensor]],
]:
    layers = get_layers(model)
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[int, dict[int, list[int]]]] = {
        "pos": defaultdict(dict),
        "neg": defaultdict(dict),
    }
    z_delta: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
    for step in range(1, int(max_steps) + 1):
        for layer_idx in spec["layers"]:
            s = success_z.get(int(step), {}).get(int(layer_idx))
            d = drift_z.get(int(step), {}).get(int(layer_idx))
            resid = residual_dirs.get(int(step), {}).get(int(layer_idx))
            down_proj = p221.get_down_proj(layers[int(layer_idx)])
            if s is None or d is None or resid is None or down_proj is None:
                continue
            delta = s - d
            z_delta[int(step)][int(layer_idx)] = delta
            unit = resid.float()
            unit_norm = torch.linalg.vector_norm(unit).item()
            if unit_norm == 0:
                continue
            unit = unit / unit_norm
            weight = down_proj.weight.detach().float().cpu()
            col_dot = torch.mv(weight.t(), unit)
            signed = delta.float() * col_dot
            k = min(int(top_channels), int(signed.numel()))

            pos_values, pos_indices = torch.topk(signed, k=k)
            neg_values, neg_indices = torch.topk(-signed, k=k)
            selected["pos"][int(step)][int(layer_idx)] = [int(i) for i in pos_indices.tolist()]
            selected["neg"][int(step)][int(layer_idx)] = [int(i) for i in neg_indices.tolist()]

            for kind, indices, values in [
                ("pos", pos_indices, pos_values),
                ("neg", neg_indices, -neg_values),
            ]:
                for rank, (channel, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                    c = int(channel)
                    rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase222_signed_channel_score_row",
                            "model": model_name,
                            "spec_id": spec["spec_id"],
                            "pattern_id": spec["pattern_id"],
                            "channel_kind": kind,
                            "step": int(step),
                            "layer_idx": int(layer_idx),
                            "rank": int(rank),
                            "channel_id": c,
                            "signed_channel_score": float(score),
                            "abs_channel_score": float(abs(score)),
                            "delta_z": float(delta[c].item()),
                            "down_col_dot_resid_unit": float(col_dot[c].item()),
                        }
                    )
    return rows, selected, z_delta


def install_signed_channel_hook(
    model,
    condition: str,
    step: int,
    selected: dict[str, dict[int, dict[int, list[int]]]],
    z_delta: dict[int, dict[int, torch.Tensor]],
    boost_scale: float,
):
    if condition == "none":
        return []
    parts = condition.split("_")
    if len(parts) != 5 or parts[0] != "mlpchan":
        raise ValueError(f"unknown condition: {condition}")
    channel_kind = parts[1]
    mode = parts[2]
    layer_idx = int(parts[3][1:])
    k = int(parts[4][1:])
    channels = selected.get(channel_kind, {}).get(int(step), {}).get(int(layer_idx), [])[:k]
    if not channels:
        return []
    layers = get_layers(model)
    down_proj = p221.get_down_proj(layers[int(layer_idx)])
    if down_proj is None:
        return []
    delta = z_delta.get(int(step), {}).get(int(layer_idx))

    def hook(_module: Any, inputs: tuple[Any, ...]):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        z = inputs[0].clone()
        idx = torch.tensor(channels, device=z.device, dtype=torch.long)
        if mode == "zero":
            z[:, -1, idx] = 0
        elif mode == "boost":
            if delta is not None:
                z[:, -1, idx] = z[:, -1, idx] + float(boost_scale) * delta[idx.cpu()].to(device=z.device, dtype=z.dtype)
        else:
            raise ValueError(f"unknown channel mode: {mode}")
        return (z,) + inputs[1:]

    return [down_proj.register_forward_pre_hook(hook)]


def forward_logits_condition(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    condition: str,
    step: int,
    selected: dict[str, dict[int, dict[int, list[int]]]],
    z_delta: dict[int, dict[int, torch.Tensor]],
    boost_scale: float,
) -> torch.Tensor:
    handles = install_signed_channel_hook(model, condition, int(step), selected, z_delta, float(boost_scale))
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


def generate_condition(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    row: dict[str, Any],
    condition: str,
    selected: dict[str, dict[int, dict[int, list[int]]]],
    z_delta: dict[int, dict[int, torch.Tensor]],
    max_steps: int,
    boost_scale: float,
) -> dict[str, Any]:
    generated = ""
    emitted_ids: list[int] = []
    emitted_tokens: list[str] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    token_rows = []
    for step in range(1, int(max_steps) + 1):
        logits = forward_logits_condition(
            model,
            tokenizer,
            device,
            str(row.get("prompt") or "") + generated,
            condition,
            int(step),
            selected,
            z_delta,
            float(boost_scale),
        )
        metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
        next_id = int(metrics["top_token_id"])
        next_text = str(metrics.get("top_token") or p212.token_text(tokenizer, next_id))
        emitted_ids.append(next_id)
        emitted_tokens.append(next_text)
        token_rows.append(
            {
                "step": int(step),
                "top_token": next_text,
                "target_rank": metrics.get("target_rank"),
                "prose_margin": metrics.get("prose_margin"),
                "echo_margin": metrics.get("echo_margin"),
                "stop_margin": metrics.get("stop_margin"),
            }
        )
        generated += next_text
        if next_id in eos_ids:
            break
    expected = p209.expected_output_pattern(str(row.get("pattern_id")))
    classification = p209.classify_pattern(generated, row, emitted_ids, eos_ids)
    return {
        "generated": generated,
        "emitted_ids": emitted_ids,
        "emitted_tokens": emitted_tokens,
        "steps_generated": len(emitted_ids),
        "expected_output_pattern": expected,
        "pattern_match": classification.get("output_pattern") == expected,
        "pattern_drift": classification.get("output_pattern") != expected,
        "failure_mode": "match" if classification.get("output_pattern") == expected else classification.get("output_pattern"),
        "token_rows": token_rows,
        **classification,
    }


def summarize_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["spec_id", "source_group", "condition"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        rec = {name: value for name, value in zip(keys, key)}
        rec.update(
            {
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
            }
        )
        out.append(rec)
    out.sort(key=lambda row: (str(row.get("spec_id")), str(row.get("source_group")), str(row.get("condition"))))
    return out


def transition_rows(rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rollout_rows:
        if row.get("condition") == "none":
            key = (str(row.get("spec_id")), str(row.get("source_group")), str(row.get("trajectory_id")))
            baseline[key] = row
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        if row.get("condition") == "none":
            continue
        key = (str(row.get("spec_id")), str(row.get("source_group")), str(row.get("trajectory_id")))
        base = baseline.get(key)
        if not base:
            continue
        buckets[
            (
                row.get("spec_id"),
                row.get("source_group"),
                row.get("condition"),
                base.get("output_pattern"),
                row.get("output_pattern"),
            )
        ].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, condition, from_pattern, to_pattern = key
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "condition": condition,
                "from_output_pattern": from_pattern,
                "to_output_pattern": to_pattern,
                "rows": len(items),
                "examples": [
                    {
                        "trajectory_id": item.get("trajectory_id"),
                        "generated": item.get("generated"),
                    }
                    for item in items[:3]
                ],
            }
        )
    out.sort(key=lambda row: int(row.get("rows") or 0), reverse=True)
    return out


def effect_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        buckets[str(row.get("spec_id"))][(str(row.get("source_group")), str(row.get("condition")))] = row
    out = []
    for spec_id, by in buckets.items():
        success_none = by.get(("success_repro", "none"), {})
        drift_none = by.get(("drift_repro", "none"), {})
        for condition in sorted({key[1] for key in by if key[1] != "none"}):
            success_patch = by.get(("success_repro", condition), {})
            drift_patch = by.get(("drift_repro", condition), {})
            out.append(
                {
                    "spec_id": spec_id,
                    "condition": condition,
                    "channel_kind": condition.split("_")[1] if condition.startswith("mlpchan_") else "",
                    "channel_action": condition.split("_")[2] if condition.startswith("mlpchan_") else "",
                    "success_rows": success_none.get("rows", 0),
                    "drift_rows": drift_none.get("rows", 0),
                    "success_base_match": finite_int(success_none.get("pattern_match")),
                    "success_patch_match": finite_int(success_patch.get("pattern_match")),
                    "drift_base_match": finite_int(drift_none.get("pattern_match")),
                    "drift_patch_match": finite_int(drift_patch.get("pattern_match")),
                    "damage_match_loss": finite_int(success_none.get("pattern_match")) - finite_int(success_patch.get("pattern_match")),
                    "repair_match_gain": finite_int(drift_patch.get("pattern_match")) - finite_int(drift_none.get("pattern_match")),
                    "success_patch_outputs": success_patch.get("output_patterns", {}),
                    "drift_patch_outputs": drift_patch.get("output_patterns", {}),
                }
            )
    out.sort(key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)), reverse=True)
    return out


def summarize_channel_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("spec_id"), row.get("layer_idx"), row.get("channel_kind"))].append(row)
    out = []
    for (spec_id, layer_idx, kind), items in buckets.items():
        top = sorted(items, key=lambda r: abs(float(r.get("signed_channel_score") or 0.0)), reverse=True)[:12]
        out.append(
            {
                "spec_id": spec_id,
                "layer_idx": int(layer_idx),
                "channel_kind": kind,
                "rows": len(items),
                "top_channels": [
                    {
                        "step": r.get("step"),
                        "rank": r.get("rank"),
                        "channel_id": r.get("channel_id"),
                        "signed_channel_score": r.get("signed_channel_score"),
                        "delta_z": r.get("delta_z"),
                        "down_col_dot_resid_unit": r.get("down_col_dot_resid_unit"),
                    }
                    for r in top
                ],
            }
        )
    out.sort(key=lambda row: (str(row.get("spec_id")), int(row.get("layer_idx") or 0), str(row.get("channel_kind"))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    filter_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
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
            kept_success: list[dict[str, Any]] = []
            kept_drift: list[dict[str, Any]] = []
            for source_group, source_items, target in [
                ("success", success_rows, kept_success),
                ("drift", drift_rows, kept_drift),
            ]:
                for row in source_items:
                    result = generate_condition(
                        model,
                        tokenizer,
                        device,
                        groups,
                        row,
                        "none",
                        {"pos": {}, "neg": {}},
                        {},
                        int(args.max_steps),
                        float(args.boost_scale),
                    )
                    reproducible = bool(result.get("pattern_match")) if source_group == "success" else not bool(result.get("pattern_match"))
                    filter_rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase222_baseline_filter_row",
                            "model": args.model,
                            "spec_id": spec["spec_id"],
                            "pattern_id": spec["pattern_id"],
                            "source_group": source_group,
                            "trajectory_id": row.get("trajectory_id"),
                            "reproducible": reproducible,
                            "output_pattern": result.get("output_pattern"),
                            "pattern_match": result.get("pattern_match"),
                        }
                    )
                    if reproducible:
                        target.append(row)
            kept_success = kept_success[: int(args.max_direction_rows)]
            kept_drift = kept_drift[: int(args.max_direction_rows)]
            if not kept_success or not kept_drift:
                log(f"{args.model}|{spec['spec_id']}: insufficient reproducible success={len(kept_success)} drift={len(kept_drift)}")
                continue
            layers = [int(x) for x in spec["layers"]]
            residual_dirs = p219.build_direction_vectors(
                model, tokenizer, device, kept_success, kept_drift, layers, int(args.max_channel_steps)
            )
            success_z = p221.mean_mlp_z(model, tokenizer, device, kept_success, layers, int(args.max_channel_steps))
            drift_z = p221.mean_mlp_z(model, tokenizer, device, kept_drift, layers, int(args.max_channel_steps))
            spec_channel_rows, selected, z_delta = signed_channel_score_rows(
                model,
                args.model,
                spec,
                residual_dirs,
                success_z,
                drift_z,
                int(args.max_channel_steps),
                int(args.top_channels),
            )
            channel_rows.extend(spec_channel_rows)
            conditions = ["none"]
            for layer_idx in layers:
                for channel_kind in ["pos", "neg"]:
                    for k in [4, 16, 64]:
                        conditions.append(f"mlpchan_{channel_kind}_zero_L{int(layer_idx)}_K{k}")
                        conditions.append(f"mlpchan_{channel_kind}_boost_L{int(layer_idx)}_K{k}")
            eval_success = kept_success[: int(args.max_eval_rows)]
            eval_drift = kept_drift[: int(args.max_eval_rows)]
            for source_group, eval_rows in [("success_repro", eval_success), ("drift_repro", eval_drift)]:
                for row in eval_rows:
                    for condition in conditions:
                        result = generate_condition(
                            model,
                            tokenizer,
                            device,
                            groups,
                            row,
                            condition,
                            selected,
                            z_delta,
                            int(args.max_steps),
                            float(args.boost_scale),
                        )
                        rollout_rows.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase222_signed_channel_rollout_row",
                                "model": args.model,
                                "spec_id": spec["spec_id"],
                                "pattern_id": spec["pattern_id"],
                                "source_group": source_group,
                                "condition": condition,
                                "trajectory_id": row.get("trajectory_id"),
                                "target_label": row.get("target_label"),
                                "object": row.get("object"),
                                **result,
                            }
                        )
            log(f"{args.model}|{spec['spec_id']}: kept success={len(eval_success)} drift={len(eval_drift)} conditions={len(conditions)}")
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
    summary_rows = summarize_rollouts(rollout_rows)
    effects = effect_rows(summary_rows)
    transitions = transition_rows(rollout_rows)
    channel_summary_rows = summarize_channel_scores(channel_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "StateWrite Factor Competition Signed Channel Split",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "spec_count": len(SPECS[args.model]),
        "filter_rows": len(filter_rows),
        "reproducible_success_rows": sum(1 for row in filter_rows if row.get("source_group") == "success" and row.get("reproducible")),
        "reproducible_drift_rows": sum(1 for row in filter_rows if row.get("source_group") == "drift" and row.get("reproducible")),
        "rollout_rows": len(rollout_rows),
        "channel_score_rows": len(channel_rows),
        "summary_rows": len(summary_rows),
        "transition_rows": len(transitions),
        "effect_rows": effects,
        "top_transition_rows": transitions[:80],
        "channel_summary_rows": channel_summary_rows,
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effects),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effects),
    }
    write_json(out_dir / f"phase222_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase222_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase222_{args.model}_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase222_{args.model}_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase222_{args.model}_effect_rows.jsonl", effects)
    write_jsonl(out_dir / f"phase222_{args.model}_transition_rows.jsonl", transitions)
    write_jsonl(out_dir / f"phase222_{args.model}_channel_score_rows.jsonl", channel_rows)
    write_jsonl(out_dir / f"phase222_{args.model}_channel_summary_rows.jsonl", channel_summary_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "filter_rows": len(filter_rows),
                "rollout_rows": len(rollout_rows),
                "channel_score_rows": len(channel_rows),
                "damage": payload["total_damage_match_loss"],
                "repair": payload["total_repair_match_gain"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase222_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    effects = []
    transitions = []
    channel_summaries = []
    for model in MODELS:
        effects.extend(p214.iter_jsonl(out_dir / f"phase222_{model}_effect_rows.jsonl") or [])
        transitions.extend(p214.iter_jsonl(out_dir / f"phase222_{model}_transition_rows.jsonl") or [])
        channel_summaries.extend(p214.iter_jsonl(out_dir / f"phase222_{model}_channel_summary_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model signed channel competition split",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "spec_count": sum(int(summary.get("spec_count") or 0) for summary in summaries),
        "filter_rows": sum(int(summary.get("filter_rows") or 0) for summary in summaries),
        "reproducible_success_rows": sum(int(summary.get("reproducible_success_rows") or 0) for summary in summaries),
        "reproducible_drift_rows": sum(int(summary.get("reproducible_drift_rows") or 0) for summary in summaries),
        "rollout_rows": sum(int(summary.get("rollout_rows") or 0) for summary in summaries),
        "channel_score_rows": sum(int(summary.get("channel_score_rows") or 0) for summary in summaries),
        "effect_rows": len(effects),
        "transition_rows": len(transitions),
        "channel_summary_rows": len(channel_summaries),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effects),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effects),
        "top_effect_rows": sorted(
            effects,
            key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)),
            reverse=True,
        )[:80],
        "top_transition_rows": sorted(transitions, key=lambda row: int(row.get("rows") or 0), reverse=True)[:80],
        "channel_summary_rows_detail": channel_summaries,
    }
    write_json(out_dir / "phase222_cross_model_summary.json", payload)
    lines = ["# Phase 222 StateWrite signed channel competition split", ""]
    for key in [
        "spec_count",
        "filter_rows",
        "reproducible_success_rows",
        "reproducible_drift_rows",
        "rollout_rows",
        "channel_score_rows",
        "total_damage_match_loss",
        "total_repair_match_gain",
    ]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(["", "| spec | condition | success | drift | damage | repair | success outputs | drift outputs |", "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in payload["top_effect_rows"][:45]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('condition')} | {row.get('success_rows')} | {row.get('drift_rows')} | "
            f"{row.get('damage_match_loss')} | {row.get('repair_match_gain')} | {row.get('success_patch_outputs')} | {row.get('drift_patch_outputs')} |"
        )
    lines.extend(["", "## Top transitions", "", "| spec | group | condition | from | to | rows |", "| --- | --- | --- | --- | --- | ---: |"])
    for row in payload["top_transition_rows"][:45]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('condition')} | "
            f"{row.get('from_output_pattern')} | {row.get('to_output_pattern')} | {row.get('rows')} |"
        )
    lines.extend(["", "## Channel summaries", ""])
    for row in channel_summaries:
        lines.append(f"### {row.get('spec_id')} L{row.get('layer_idx')} {row.get('channel_kind')}")
        for ch in (row.get("top_channels") or [])[:8]:
            lines.append(
                f"- step={ch.get('step')} rank={ch.get('rank')} channel={ch.get('channel_id')} "
                f"signed={ch.get('signed_channel_score')} delta_z={ch.get('delta_z')} dot={ch.get('down_col_dot_resid_unit')}"
            )
    (out_dir / "phase222_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "specs": payload["spec_count"],
                "rollout_rows": payload["rollout_rows"],
                "channel_score_rows": payload["channel_score_rows"],
                "damage": payload["total_damage_match_loss"],
                "repair": payload["total_repair_match_gain"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase222 StateWrite signed channel competition split")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="statewrite_factor_competition")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=12)
    parser.add_argument("--max-direction-rows", type=int, default=10)
    parser.add_argument("--max-eval-rows", type=int, default=5)
    parser.add_argument("--max-channel-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--top-channels", type=int, default=96)
    parser.add_argument("--boost-scale", type=float, default=1.0)
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
