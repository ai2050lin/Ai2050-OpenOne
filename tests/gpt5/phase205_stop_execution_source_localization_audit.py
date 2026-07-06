#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 205
SOURCE_PHASE = 204
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase205_stop_execution_source_localization_audit")
PHASE204_ROOT = Path("tests/result/phase204_global_trajectory_stop_execution_atlas")


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return None if not vals else float(sum(vals) / len(vals))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def scan_layers_for_model(model, text: str) -> list[int]:
    layers = get_layers(model)
    total = len(layers)
    if text:
        return sorted({idx for idx in [int(x) for x in parse_csv(text)] if 0 <= idx < total})
    fractions = [0.35, 0.5, 0.65, 0.8, 0.95]
    return sorted({min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions})


def select_phase204_trajectories(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE204_ROOT / args.phase204_round / f"phase204_{args.model}_trajectory_rows.jsonl"
    rows = read_jsonl(path)
    continued = [row for row in rows if row.get("period_seen") and row.get("continued_after_period")]
    continued.sort(
        key=lambda row: (
            str(row.get("rollout_mode")) != "post_answer",
            str(row.get("prompt_protocol")) != "stop_explicit",
            str(row.get("prompt_protocol")) != "short_answer",
            str(row.get("relation")),
        )
    )
    if int(args.max_trajectories) > 0:
        continued = continued[: int(args.max_trajectories)]
    return continued


def state_points_for_trajectory(row: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = [str(x) for x in row.get("emitted_tokens") or []]
    first_period_step = row.get("first_period_step")
    try:
        period_idx = int(first_period_step) - 1
    except (TypeError, ValueError):
        return []
    if period_idx < 0 or period_idx >= len(tokens):
        return []
    specs = [
        ("before_period", tokens[:period_idx]),
        ("after_period", tokens[: period_idx + 1]),
    ]
    if period_idx + 2 <= len(tokens):
        specs.append(("after_continue1", tokens[: period_idx + 2]))
    out = []
    for state_kind, prefix_tokens in specs:
        generated_prefix = "".join(prefix_tokens)
        out.append(
            {
                "state_key": f"{row.get('trajectory_id')}|{state_kind}",
                "trajectory_id": row.get("trajectory_id"),
                "model": row.get("model"),
                "relation": row.get("relation"),
                "language_pair": row.get("language_pair"),
                "prompt_protocol": row.get("prompt_protocol"),
                "rollout_mode": row.get("rollout_mode"),
                "object": row.get("object"),
                "target_label": row.get("target_label"),
                "state_kind": state_kind,
                "first_period_step": first_period_step,
                "prompt": row.get("prompt"),
                "generated_prefix": generated_prefix,
                "text": str(row.get("prompt") or "") + generated_prefix,
                "full_generated": row.get("generated"),
                "continued_after_period": row.get("continued_after_period"),
                "ended_with_eos": row.get("ended_with_eos"),
            }
        )
    return out


def module_output_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def capture_states(
    model,
    tokenizer,
    device: torch.device,
    state_points: list[dict[str, Any]],
    layers_to_scan: list[int],
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], torch.Tensor]]:
    groups = p201.token_groups(tokenizer)
    layers = get_layers(model)
    state_rows: dict[str, dict[str, Any]] = {}
    mlp_acts: dict[tuple[str, int], torch.Tensor] = {}
    for start in range(0, len(state_points), max(1, int(batch_size))):
        batch = state_points[start : start + max(1, int(batch_size))]
        encoded = tokenizer([row["text"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        last_pos = attention_mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        captured_mlp: dict[int, torch.Tensor] = {}
        captured_attn: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in layers_to_scan:
            layer = layers[int(layer_idx)]
            down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
            if down_proj is not None:

                def mlp_hook(_module, inputs, layer_idx=layer_idx):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    hidden = inputs[0]
                    pos = last_pos.to(device=hidden.device)
                    idx = torch.arange(hidden.shape[0], device=hidden.device)
                    captured_mlp[int(layer_idx)] = hidden[idx, pos, :].detach().float().cpu()
                    return None

                handles.append(down_proj.register_forward_pre_hook(mlp_hook))
            attn = getattr(layer, "self_attn", None)
            if attn is not None:

                def attn_hook(_module, _inputs, output, layer_idx=layer_idx):
                    hidden = module_output_tensor(output)
                    if hidden is None:
                        return None
                    pos = last_pos.to(device=hidden.device)
                    idx = torch.arange(hidden.shape[0], device=hidden.device)
                    captured_attn[int(layer_idx)] = hidden[idx, pos, :].detach().float().cpu()
                    return None

                handles.append(attn.register_forward_hook(attn_hook))
        try:
            with torch.inference_mode():
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            logits = result.logits[batch_idx, last_pos].detach().float().cpu()
            for row_idx, point in enumerate(batch):
                state_key = str(point["state_key"])
                metrics = p204.metric_for_logits(tokenizer, logits[row_idx], point, groups)
                out = {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase205_state_source_row",
                    **{key: point.get(key) for key in [
                        "state_key",
                        "trajectory_id",
                        "model",
                        "relation",
                        "language_pair",
                        "prompt_protocol",
                        "rollout_mode",
                        "object",
                        "target_label",
                        "state_kind",
                        "first_period_step",
                        "generated_prefix",
                        "continued_after_period",
                        "ended_with_eos",
                    ]},
                    "text_len": int(input_ids[row_idx].shape[0]),
                    **metrics,
                }
                for layer_idx in layers_to_scan:
                    hidden_idx = min(int(layer_idx) + 1, len(result.hidden_states) - 1)
                    h = result.hidden_states[hidden_idx][row_idx, int(last_pos[row_idx].item())].detach().float().cpu()
                    out[f"L{layer_idx}_resid_norm"] = float(torch.linalg.vector_norm(h).item())
                    out[f"L{layer_idx}_resid_mean_abs"] = float(torch.mean(torch.abs(h)).item())
                    att = captured_attn.get(int(layer_idx))
                    if att is not None:
                        v = att[row_idx]
                        out[f"L{layer_idx}_attn_norm"] = float(torch.linalg.vector_norm(v).item())
                        out[f"L{layer_idx}_attn_mean_abs"] = float(torch.mean(torch.abs(v)).item())
                    mlp = captured_mlp.get(int(layer_idx))
                    if mlp is not None:
                        v = mlp[row_idx]
                        mlp_acts[(state_key, int(layer_idx))] = v.clone()
                        out[f"L{layer_idx}_mlp_rms"] = float(torch.sqrt(torch.mean(v * v)).item())
                        out[f"L{layer_idx}_mlp_mean_abs"] = float(torch.mean(torch.abs(v)).item())
                        top_abs = torch.argmax(torch.abs(v)).item()
                        out[f"L{layer_idx}_mlp_top_abs_channel"] = int(top_abs)
                        out[f"L{layer_idx}_mlp_top_abs_value"] = float(v[int(top_abs)].item())
                state_rows[state_key] = out
            del result, logits
        finally:
            for handle in handles:
                handle.remove()
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state_rows, mlp_acts


def build_transition_rows(state_rows: dict[str, dict[str, Any]], layers_to_scan: list[int]) -> list[dict[str, Any]]:
    by_traj: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_rows.values():
        by_traj[str(row.get("trajectory_id"))][str(row.get("state_kind"))] = row
    out = []
    metric_keys = ["eos_rank", "period_rank", "prose_rank", "echo_rank", "stop_margin", "prose_margin", "echo_margin", "eos_vs_prose_margin", "period_vs_prose_margin"]
    for trajectory_id, states in by_traj.items():
        for src, dst in [("before_period", "after_period"), ("after_period", "after_continue1")]:
            a = states.get(src)
            b = states.get(dst)
            if not a or not b:
                continue
            row = {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase205_transition_delta_row",
                "trajectory_id": trajectory_id,
                "transition": f"{src}->{dst}",
                "model": a.get("model"),
                "relation": a.get("relation"),
                "language_pair": a.get("language_pair"),
                "prompt_protocol": a.get("prompt_protocol"),
                "rollout_mode": a.get("rollout_mode"),
                "continued_after_period": a.get("continued_after_period"),
            }
            for key in metric_keys:
                row[f"{key}_delta"] = None if a.get(key) is None or b.get(key) is None else finite(b.get(key)) - finite(a.get(key))
            for layer_idx in layers_to_scan:
                for name in ["resid_norm", "resid_mean_abs", "attn_norm", "attn_mean_abs", "mlp_rms", "mlp_mean_abs"]:
                    key = f"L{layer_idx}_{name}"
                    if a.get(key) is not None and b.get(key) is not None:
                        row[f"{key}_delta"] = finite(b.get(key)) - finite(a.get(key))
            out.append(row)
    return out


def top_mlp_delta_rows(
    state_rows: dict[str, dict[str, Any]],
    mlp_acts: dict[tuple[str, int], torch.Tensor],
    layers_to_scan: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    by_traj: dict[str, dict[str, str]] = defaultdict(dict)
    for state_key, row in state_rows.items():
        by_traj[str(row.get("trajectory_id"))][str(row.get("state_kind"))] = state_key
    accum: dict[tuple[str, str, int], list[torch.Tensor]] = defaultdict(list)
    meta: dict[tuple[str, str, int], dict[str, Any]] = {}
    for trajectory_id, states in by_traj.items():
        for src, dst in [("before_period", "after_period"), ("after_period", "after_continue1")]:
            a_key = states.get(src)
            b_key = states.get(dst)
            if not a_key or not b_key:
                continue
            a_row = state_rows[a_key]
            for layer_idx in layers_to_scan:
                a = mlp_acts.get((a_key, int(layer_idx)))
                b = mlp_acts.get((b_key, int(layer_idx)))
                if a is None or b is None:
                    continue
                key = (str(a_row.get("model")), f"{src}->{dst}", int(layer_idx))
                accum[key].append(torch.abs(b - a).float())
                meta[key] = {
                    "model": a_row.get("model"),
                    "transition": f"{src}->{dst}",
                    "layer_idx": int(layer_idx),
                }
    out = []
    for key, vals in accum.items():
        if not vals:
            continue
        avg = torch.stack(vals).mean(dim=0)
        values, indices = torch.topk(avg, k=min(int(top_k), int(avg.numel())))
        for rank, (value, channel) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            out.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase205_top_mlp_delta_channel_row",
                    **meta[key],
                    "rank": rank,
                    "channel_id": int(channel),
                    "mean_abs_delta": float(value),
                    "sample_count": len(vals),
                }
            )
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("transition")), int(row.get("layer_idx")), int(row.get("rank"))))
    return out


def summarize_transitions(rows: list[dict[str, Any]], layers_to_scan: list[int]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "rollout_mode", "prompt_protocol", "transition"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "eos_rank_delta_mean": mean([item.get("eos_rank_delta") for item in items]),
                "period_rank_delta_mean": mean([item.get("period_rank_delta") for item in items]),
                "prose_rank_delta_mean": mean([item.get("prose_rank_delta") for item in items]),
                "stop_margin_delta_mean": mean([item.get("stop_margin_delta") for item in items]),
                "prose_margin_delta_mean": mean([item.get("prose_margin_delta") for item in items]),
                "echo_margin_delta_mean": mean([item.get("echo_margin_delta") for item in items]),
            }
        )
        for layer_idx in layers_to_scan:
            row[f"L{layer_idx}_mlp_rms_delta_mean"] = mean([item.get(f"L{layer_idx}_mlp_rms_delta") for item in items])
            row[f"L{layer_idx}_attn_norm_delta_mean"] = mean([item.get(f"L{layer_idx}_attn_norm_delta") for item in items])
            row[f"L{layer_idx}_resid_norm_delta_mean"] = mean([item.get(f"L{layer_idx}_resid_norm_delta") for item in items])
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = select_phase204_trajectories(args)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Stop-Execution Source Localization Audit",
        "model": args.model,
        "selected_trajectory_count": len(trajectories),
        "decoder_audit": {
            "decoding": "greedy",
            "stop_sequence": None,
            "phase204_max_steps": 8,
            "phase205_uses_phase204_period_continuation_failures": True,
        },
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase205_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    state_rows: dict[str, dict[str, Any]] = {}
    transition_rows: list[dict[str, Any]] = []
    top_delta_rows: list[dict[str, Any]] = []
    layers_to_scan: list[int] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        layers_to_scan = scan_layers_for_model(model, args.scan_layers)
        state_points = []
        for row in trajectories:
            state_points.extend(state_points_for_trajectory(row))
        state_rows, mlp_acts = capture_states(model, tokenizer, device, state_points, layers_to_scan, int(args.batch_size))
        transition_rows = build_transition_rows(state_rows, layers_to_scan)
        top_delta_rows = top_mlp_delta_rows(state_rows, mlp_acts, layers_to_scan, int(args.top_channels_per_layer))
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    transition_summary_rows = summarize_transitions(transition_rows, layers_to_scan)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layers_to_scan": layers_to_scan,
        "state_rows": len(state_rows),
        "transition_rows": len(transition_rows),
        "top_mlp_delta_rows": top_delta_rows,
        "transition_summary_rows": transition_summary_rows,
        "boundary": "Source localization audit only; compares period-adjacent states from Phase204 failure trajectories without patching.",
    }
    write_json(out_dir / f"phase205_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase205_{args.model}_state_rows.jsonl", list(state_rows.values()))
    write_jsonl(out_dir / f"phase205_{args.model}_transition_rows.jsonl", transition_rows)
    write_jsonl(out_dir / f"phase205_{args.model}_top_mlp_delta_rows.jsonl", top_delta_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "state_rows": len(state_rows),
                "transition_rows": len(transition_rows),
                "layers_to_scan": layers_to_scan,
                "top_transition_summary_rows": transition_summary_rows[:12],
                "top_mlp_delta_rows": top_delta_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase205_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    transition_summary_rows = []
    top_mlp_delta_rows = []
    for summary in summaries:
        transition_summary_rows.extend(dict(row) for row in summary.get("transition_summary_rows") or [])
        top_mlp_delta_rows.extend(dict(row) for row in summary.get("top_mlp_delta_rows") or [])
    payload = {
        "schema_version": "phase205_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "transition_summary_rows": transition_summary_rows,
        "top_mlp_delta_rows": top_mlp_delta_rows,
    }
    write_json(out_dir / "phase205_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase205_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 205 stop-execution source localization audit", ""]
    lines.append("## Transition Summary")
    lines.append("| model | mode | protocol | transition | rows | eos rank delta | stop margin delta | prose margin delta |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("transition_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('rollout_mode')} | {row.get('prompt_protocol')} | {row.get('transition')} | "
            f"{row.get('rows')} | {finite(row.get('eos_rank_delta_mean')):.2f} | "
            f"{finite(row.get('stop_margin_delta_mean')):.2f} | {finite(row.get('prose_margin_delta_mean')):.2f} |"
        )
    lines.append("")
    lines.append("## Top MLP Delta Channels")
    lines.append("| model | transition | layer | rank | channel | mean abs delta | samples |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_mlp_delta_rows") or []:
        if int(row.get("rank", 99)) > 3:
            continue
        lines.append(
            f"| {row.get('model')} | {row.get('transition')} | {row.get('layer_idx')} | {row.get('rank')} | "
            f"{row.get('channel_id')} | {finite(row.get('mean_abs_delta')):.4f} | {row.get('sample_count')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="stop_execution_source_localization_audit")
    parser.add_argument("--phase204-round", default="global_trajectory_stop_execution_atlas")
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--max-trajectories", type=int, default=36)
    parser.add_argument("--top-channels-per-layer", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
