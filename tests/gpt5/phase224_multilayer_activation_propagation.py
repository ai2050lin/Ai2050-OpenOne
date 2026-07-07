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
import phase223_channel_activation_gate_validation as p223  # noqa: E402


PHASE = 224
SOURCE_PHASE = 223
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase224_multilayer_activation_propagation")


SPECS = {
    "qwen3": [
        {
            "spec_id": "qwen3_explain_l29_to_l31_l33_propagation",
            "pattern_id": "answer_explain",
            "source_layers": [29],
            "observe_layers": [29, 31, 33],
        },
    ],
    "glm4": [
        {
            "spec_id": "glm4_repeat_l30_to_l31_l32_propagation",
            "pattern_id": "answer_repeat",
            "source_layers": [30],
            "observe_layers": [30, 31, 32],
        },
    ],
    "deepseek7b": [
        {
            "spec_id": "deepseek7b_explain_l24_to_l25_l26_propagation",
            "pattern_id": "answer_explain",
            "source_layers": [24],
            "observe_layers": [24, 25, 26],
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
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def capture_hidden_and_logits(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    condition: str,
    step: int,
    selected: dict[str, dict[int, dict[int, list[int]]]],
    success_z: dict[int, dict[int, torch.Tensor]],
    drift_z: dict[int, dict[int, torch.Tensor]],
    observe_layers: list[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    handles = p223.install_clamp_hook(model, condition, int(step), selected, success_z, drift_z)
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
    return hidden, logits


def propagation_rows(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    model_name: str,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    source_group: str,
    conditions: list[str],
    selected: dict[str, dict[int, dict[int, list[int]]]],
    success_z: dict[int, dict[int, torch.Tensor]],
    drift_z: dict[int, dict[int, torch.Tensor]],
    residual_dirs: dict[int, dict[int, torch.Tensor]],
    max_steps: int,
) -> list[dict[str, Any]]:
    out = []
    observe_layers = [int(x) for x in spec["observe_layers"]]
    for row in rows:
        for step in range(1, int(max_steps) + 1):
            prefix = p219.prefix_for_step(row, int(step))
            base_hidden, base_logits = capture_hidden_and_logits(
                model, tokenizer, device, prefix, "none", int(step), selected, success_z, drift_z, observe_layers
            )
            base_metrics = p204.metric_for_logits(tokenizer, base_logits, row, groups)
            for condition in conditions:
                patch_hidden, patch_logits = capture_hidden_and_logits(
                    model, tokenizer, device, prefix, condition, int(step), selected, success_z, drift_z, observe_layers
                )
                patch_metrics = p204.metric_for_logits(tokenizer, patch_logits, row, groups)
                for observe_layer in observe_layers:
                    b = base_hidden.get(int(observe_layer))
                    p = patch_hidden.get(int(observe_layer))
                    direction = residual_dirs.get(int(step), {}).get(int(observe_layer))
                    if b is None or p is None or direction is None:
                        continue
                    delta = p - b
                    dir_norm = torch.linalg.vector_norm(direction).item()
                    unit = direction / dir_norm if dir_norm > 0 else direction
                    delta_norm = torch.linalg.vector_norm(delta).item()
                    base_projection = float(torch.dot(b, unit).item())
                    patch_projection = float(torch.dot(p, unit).item())
                    out.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase224_propagation_row",
                            "model": model_name,
                            "spec_id": spec["spec_id"],
                            "pattern_id": spec["pattern_id"],
                            "source_group": source_group,
                            "condition": condition,
                            "trajectory_id": row.get("trajectory_id"),
                            "step": int(step),
                            "observe_layer": int(observe_layer),
                            "delta_norm": float(delta_norm),
                            "delta_cos_to_success_drift_dir": float(F.cosine_similarity(delta, direction, dim=0).item()) if delta_norm > 0 and dir_norm > 0 else 0.0,
                            "base_projection_to_success_drift_dir": base_projection,
                            "patch_projection_to_success_drift_dir": patch_projection,
                            "projection_shift": patch_projection - base_projection,
                            "base_top_token_id": int(base_metrics.get("top_token_id") or -1),
                            "patch_top_token_id": int(patch_metrics.get("top_token_id") or -1),
                            "base_top_token": str(base_metrics.get("top_token") or ""),
                            "patch_top_token": str(patch_metrics.get("top_token") or ""),
                            "base_target_rank": base_metrics.get("target_rank"),
                            "patch_target_rank": patch_metrics.get("target_rank"),
                            "base_prose_margin": base_metrics.get("prose_margin"),
                            "patch_prose_margin": patch_metrics.get("prose_margin"),
                            "base_echo_margin": base_metrics.get("echo_margin"),
                            "patch_echo_margin": patch_metrics.get("echo_margin"),
                            "base_stop_margin": base_metrics.get("stop_margin"),
                            "patch_stop_margin": patch_metrics.get("stop_margin"),
                        }
                    )
    return out


def summarize_propagation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("spec_id"), row.get("source_group"), row.get("condition"), row.get("observe_layer"))].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, condition, observe_layer = key
        shifts = [finite_float(x.get("projection_shift")) for x in items]
        cosines = [finite_float(x.get("delta_cos_to_success_drift_dir")) for x in items]
        delta_norms = [finite_float(x.get("delta_norm")) for x in items]
        top_changed = sum(1 for x in items if x.get("base_top_token_id") != x.get("patch_top_token_id"))
        target_rank_improved = sum(
            1
            for x in items
            if x.get("base_target_rank") is not None
            and x.get("patch_target_rank") is not None
            and finite_float(x.get("patch_target_rank")) < finite_float(x.get("base_target_rank"))
        )
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "condition": condition,
                "observe_layer": int(observe_layer),
                "rows": len(items),
                "mean_projection_shift": sum(shifts) / len(shifts) if shifts else 0.0,
                "mean_delta_cos_to_success_drift_dir": sum(cosines) / len(cosines) if cosines else 0.0,
                "mean_delta_norm": sum(delta_norms) / len(delta_norms) if delta_norms else 0.0,
                "top_token_changed": top_changed,
                "target_rank_improved": target_rank_improved,
            }
        )
    out.sort(
        key=lambda row: abs(float(row.get("mean_projection_shift") or 0.0)) + float(row.get("top_token_changed") or 0),
        reverse=True,
    )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    filter_rows: list[dict[str, Any]] = []
    prop_rows: list[dict[str, Any]] = []
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
                    result = p223.generate_condition(
                        model, tokenizer, device, groups, row, "none", {"pos": {}, "neg": {}}, {}, {}, int(args.max_steps)
                    )
                    reproducible = bool(result.get("pattern_match")) if source_group == "success" else not bool(result.get("pattern_match"))
                    filter_rows.append(
                        {
                            "phase": PHASE,
                            "row_kind": "phase224_baseline_filter_row",
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

            source_layers = [int(x) for x in spec["source_layers"]]
            observe_layers = [int(x) for x in spec["observe_layers"]]
            all_layers = sorted(set(source_layers + observe_layers))
            residual_dirs = p219.build_direction_vectors(
                model, tokenizer, device, kept_success, kept_drift, all_layers, int(args.max_channel_steps)
            )
            success_z = p221.mean_mlp_z(model, tokenizer, device, kept_success, source_layers, int(args.max_channel_steps))
            drift_z = p221.mean_mlp_z(model, tokenizer, device, kept_drift, source_layers, int(args.max_channel_steps))
            score_spec = {"spec_id": spec["spec_id"], "pattern_id": spec["pattern_id"], "layers": source_layers}
            spec_channel_rows, selected, _z_delta = p222.signed_channel_score_rows(
                model,
                args.model,
                score_spec,
                residual_dirs,
                success_z,
                drift_z,
                int(args.max_channel_steps),
                int(args.top_channels),
            )
            channel_rows.extend(spec_channel_rows)
            conditions = []
            for source_layer in source_layers:
                for k in [4, 16, 64]:
                    conditions.append(f"mlpchan_pos_zero_L{source_layer}_K{k}")
                    conditions.append(f"mlpchan_pos_success_L{source_layer}_K{k}")
                    conditions.append(f"mlpchan_pos_drift_L{source_layer}_K{k}")
            eval_success = kept_success[: int(args.max_eval_rows)]
            eval_drift = kept_drift[: int(args.max_eval_rows)]
            for source_group, eval_rows in [("success_repro", eval_success), ("drift_repro", eval_drift)]:
                prop_rows.extend(
                    propagation_rows(
                        model,
                        tokenizer,
                        device,
                        groups,
                        args.model,
                        spec,
                        eval_rows,
                        source_group,
                        conditions,
                        selected,
                        success_z,
                        drift_z,
                        residual_dirs,
                        int(args.max_channel_steps),
                    )
                )
            log(f"{args.model}|{spec['spec_id']}: success={len(eval_success)} drift={len(eval_drift)} conditions={len(conditions)}")
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
    summary_rows = summarize_propagation(prop_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Multilayer activation propagation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "spec_count": len(SPECS[args.model]),
        "filter_rows": len(filter_rows),
        "reproducible_success_rows": sum(1 for row in filter_rows if row.get("source_group") == "success" and row.get("reproducible")),
        "reproducible_drift_rows": sum(1 for row in filter_rows if row.get("source_group") == "drift" and row.get("reproducible")),
        "propagation_rows": len(prop_rows),
        "channel_score_rows": len(channel_rows),
        "summary_rows": len(summary_rows),
        "top_summary_rows": summary_rows[:80],
        "total_top_token_changed": sum(int(row.get("top_token_changed") or 0) for row in summary_rows),
        "total_target_rank_improved": sum(int(row.get("target_rank_improved") or 0) for row in summary_rows),
    }
    write_json(out_dir / f"phase224_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase224_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase224_{args.model}_propagation_rows.jsonl", prop_rows)
    write_jsonl(out_dir / f"phase224_{args.model}_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase224_{args.model}_channel_score_rows.jsonl", channel_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "filter_rows": len(filter_rows),
                "propagation_rows": len(prop_rows),
                "channel_score_rows": len(channel_rows),
                "top_token_changed": payload["total_top_token_changed"],
                "target_rank_improved": payload["total_target_rank_improved"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase224_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    rows = []
    for model in MODELS:
        rows.extend(p214.iter_jsonl(out_dir / f"phase224_{model}_summary_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model multilayer activation propagation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "spec_count": sum(int(summary.get("spec_count") or 0) for summary in summaries),
        "filter_rows": sum(int(summary.get("filter_rows") or 0) for summary in summaries),
        "reproducible_success_rows": sum(int(summary.get("reproducible_success_rows") or 0) for summary in summaries),
        "reproducible_drift_rows": sum(int(summary.get("reproducible_drift_rows") or 0) for summary in summaries),
        "propagation_rows": sum(int(summary.get("propagation_rows") or 0) for summary in summaries),
        "channel_score_rows": sum(int(summary.get("channel_score_rows") or 0) for summary in summaries),
        "total_top_token_changed": sum(int(row.get("top_token_changed") or 0) for row in rows),
        "total_target_rank_improved": sum(int(row.get("target_rank_improved") or 0) for row in rows),
        "top_summary_rows": sorted(
            rows,
            key=lambda row: abs(float(row.get("mean_projection_shift") or 0.0)) + int(row.get("top_token_changed") or 0),
            reverse=True,
        )[:80],
    }
    write_json(out_dir / "phase224_cross_model_summary.json", payload)
    lines = ["# Phase 224 multilayer activation propagation", ""]
    for key in [
        "spec_count",
        "filter_rows",
        "reproducible_success_rows",
        "reproducible_drift_rows",
        "propagation_rows",
        "channel_score_rows",
        "total_top_token_changed",
        "total_target_rank_improved",
    ]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(
        [
            "",
            "| spec | group | condition | layer | rows | mean shift | mean cos | top changed | rank improved |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["top_summary_rows"][:60]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('condition')} | {row.get('observe_layer')} | "
            f"{row.get('rows')} | {float(row.get('mean_projection_shift') or 0.0):.6f} | "
            f"{float(row.get('mean_delta_cos_to_success_drift_dir') or 0.0):.6f} | "
            f"{row.get('top_token_changed')} | {row.get('target_rank_improved')} |"
        )
    (out_dir / "phase224_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "specs": payload["spec_count"],
                "propagation_rows": payload["propagation_rows"],
                "channel_score_rows": payload["channel_score_rows"],
                "top_token_changed": payload["total_top_token_changed"],
                "target_rank_improved": payload["total_target_rank_improved"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase224 multilayer activation propagation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="multilayer_activation_propagation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=12)
    parser.add_argument("--max-direction-rows", type=int, default=10)
    parser.add_argument("--max-eval-rows", type=int, default=4)
    parser.add_argument("--max-channel-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6)
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
