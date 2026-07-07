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
from model_utils import get_layers  # noqa: E402


PHASE = 213
SOURCE_PHASE = 212
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase213_window_direction_prompt_trigger")


WINDOWS = {
    "qwen3": [
        {"pattern_id": "answer_list", "failure_mode": "other_or_wrong", "layers": [31, 32, 33], "steps": [10, 11]},
        {"pattern_id": "answer_list", "failure_mode": "short_answer", "layers": [31, 32, 33], "steps": [8, 9]},
    ],
    "glm4": [
        {"pattern_id": "answer_list", "failure_mode": "repeat_answer", "layers": [28, 29, 30], "steps": [7, 8]},
        {"pattern_id": "answer_list", "failure_mode": "echo_then_answer", "layers": [34, 35, 36], "steps": [7, 8]},
    ],
    "deepseek7b": [
        {"pattern_id": "answer_explain", "failure_mode": "other_or_wrong", "layers": [25, 26, 27], "steps": [6, 7]},
        {"pattern_id": "answer_list", "failure_mode": "other_or_wrong", "layers": [23, 24, 25], "steps": [6, 7]},
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_trajectories(input_dir: Path, model: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(input_dir / f"phase210_{model}_trajectory_rows.jsonl") or [])


def prefix_for_step(row: dict[str, Any], step: int) -> str:
    emitted = row.get("emitted_tokens") or []
    prefix_tokens = emitted[: max(0, int(step) - 1)]
    return str(row.get("prompt") or "") + "".join(str(tok) for tok in prefix_tokens)


def hidden_at_text(model, tokenizer, device: torch.device, text: str, layer_idx: int) -> torch.Tensor:
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    vec = result.hidden_states[int(layer_idx) + 1][0, last_pos].detach().float().cpu()
    del result, input_ids, attention_mask
    return vec


def mean_hidden(model, tokenizer, device: torch.device, rows: list[dict[str, Any]], layer_idx: int, step: int) -> torch.Tensor | None:
    vectors = []
    for row in rows:
        try:
            vectors.append(hidden_at_text(model, tokenizer, device, prefix_for_step(row, step), layer_idx))
        except Exception as exc:
            log(f"skip hidden {row.get('trajectory_id')}: {exc}")
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def build_direction_vectors(
    model,
    tokenizer,
    device: torch.device,
    success_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    layers: list[int],
    steps: list[int],
) -> dict[int, dict[int, torch.Tensor]]:
    out: dict[int, dict[int, torch.Tensor]] = {}
    for step in steps:
        out[int(step)] = {}
        for layer_idx in layers:
            s = mean_hidden(model, tokenizer, device, success_rows, int(layer_idx), int(step))
            d = mean_hidden(model, tokenizer, device, drift_rows, int(layer_idx), int(step))
            if s is not None and d is not None:
                out[int(step)][int(layer_idx)] = s - d
    return out


def forward_logits_with_window_direction(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    patch_vectors: dict[int, torch.Tensor] | None,
    direction_sign: float,
    direction_scale: float,
) -> torch.Tensor:
    handles = []
    if patch_vectors:
        layers = get_layers(model)
        dtype = next(model.parameters()).dtype
        for layer_idx, vector in patch_vectors.items():
            patch = (float(direction_sign) * float(direction_scale) * vector).to(device=device, dtype=dtype)

            def make_hook(patch_vec):
                def hook(_module, _inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        hidden[:, -1, :] = hidden[:, -1, :] + patch_vec
                        return (hidden, *output[1:])
                    hidden = output.clone()
                    hidden[:, -1, :] = hidden[:, -1, :] + patch_vec
                    return hidden

                return hook

            handles.append(layers[int(layer_idx)].register_forward_hook(make_hook(patch)))
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


def generate_window(
    model,
    tokenizer,
    device: torch.device,
    row: dict[str, Any],
    groups: dict[str, list[int]],
    max_steps: int,
    direction_vectors: dict[int, dict[int, torch.Tensor]] | None = None,
    direction_sign: float = 1.0,
    direction_scale: float = 1.0,
) -> dict[str, Any]:
    generated = ""
    emitted_ids: list[int] = []
    emitted_tokens: list[str] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    token_rows = []
    for step in range(1, int(max_steps) + 1):
        patch_vectors = (direction_vectors or {}).get(int(step))
        logits = forward_logits_with_window_direction(
            model,
            tokenizer,
            device,
            str(row.get("prompt") or "") + generated,
            patch_vectors,
            direction_sign,
            direction_scale,
        )
        metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
        next_id = int(metrics["top_token_id"])
        next_text = str(metrics.get("top_token") or p212.token_text(tokenizer, next_id))
        emitted_ids.append(next_id)
        emitted_tokens.append(next_text)
        token_rows.append(
            {
                "step": step,
                "top_token": next_text,
                "target_rank": metrics.get("target_rank"),
                "prose_margin": metrics.get("prose_margin"),
                "echo_margin": metrics.get("echo_margin"),
                "stop_margin": metrics.get("stop_margin"),
                "patched_layers": sorted((patch_vectors or {}).keys()),
            }
        )
        generated += next_text
        if next_id in eos_ids:
            break
    classification = p209.classify_pattern(generated, row, emitted_ids, eos_ids)
    expected = p209.expected_output_pattern(str(row.get("pattern_id")))
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


def prompt_trigger_rows(model, tokenizer, device: torch.device, model_name: str, rows: list[dict[str, Any]], layers: list[int], max_rows: int) -> list[dict[str, Any]]:
    selected = rows[: int(max_rows)]
    out = []
    for row in selected:
        for layer_idx in layers:
            try:
                vec = hidden_at_text(model, tokenizer, device, str(row.get("prompt") or ""), int(layer_idx))
            except Exception:
                continue
            out.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase213_prompt_trigger_state_row",
                    "model": model_name,
                    "trajectory_id": row.get("trajectory_id"),
                    "pattern_id": row.get("pattern_id"),
                    "pattern_match": row.get("pattern_match"),
                    "failure_mode": row.get("failure_mode"),
                    "layer_idx": int(layer_idx),
                    "prompt_last_residual_norm": float(torch.linalg.vector_norm(vec).item()),
                    "prompt_last_residual_mean": float(vec.mean().item()),
                    "prompt_last_residual_std": float(vec.std(unbiased=False).item()),
                }
            )
    return out


def summarize_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("candidate_key"), row.get("source_group"), row.get("patch_condition"))].append(row)
    out = []
    for key, items in buckets.items():
        candidate_key, source_group, patch_condition = key
        out.append(
            {
                "candidate_key": candidate_key,
                "source_group": source_group,
                "patch_condition": patch_condition,
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
            }
        )
    out.sort(key=lambda row: (str(row.get("candidate_key")), str(row.get("source_group")), str(row.get("patch_condition"))))
    return out


def summarize_prompt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("model"), row.get("pattern_id"), row.get("layer_idx"), row.get("pattern_match"))].append(row)
    out = []
    for key, items in buckets.items():
        model, pattern_id, layer_idx, pattern_match = key
        norms = [float(item.get("prompt_last_residual_norm") or 0.0) for item in items]
        out.append(
            {
                "model": model,
                "pattern_id": pattern_id,
                "layer_idx": layer_idx,
                "pattern_match": pattern_match,
                "rows": len(items),
                "prompt_norm_mean": sum(norms) / len(norms) if norms else None,
            }
        )
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), int(row.get("layer_idx")), str(row.get("pattern_match"))))
    return out


def eval_window(model, tokenizer, device, groups, model_name: str, window: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    success_rows, drift_rows = p212.select_rows(rows, str(window["pattern_id"]), str(window["failure_mode"]), int(args.max_rows_per_group))
    donor_success = success_rows[: int(args.max_donor_rows)]
    donor_drift = drift_rows[: int(args.max_donor_rows)]
    eval_success = success_rows[: int(args.max_eval_rows)]
    eval_drift = drift_rows[: int(args.max_eval_rows)]
    direction_vectors = build_direction_vectors(model, tokenizer, device, donor_success, donor_drift, [int(x) for x in window["layers"]], [int(x) for x in window["steps"]])
    rollout_rows = []
    for source_group, eval_rows, condition, sign in [
        ("drift", eval_drift, "none", 0.0),
        ("drift", eval_drift, "success_minus_drift", 1.0),
        ("success", eval_success, "none", 0.0),
        ("success", eval_success, "drift_minus_success", -1.0),
    ]:
        for row in eval_rows:
            result = generate_window(
                model,
                tokenizer,
                device,
                row,
                groups,
                int(args.max_steps),
                direction_vectors if sign != 0.0 else None,
                direction_sign=sign,
                direction_scale=float(args.direction_scale),
            )
            rollout_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase213_window_rollout_row",
                    "model": model_name,
                    "candidate_key": f"{model_name}|{window['pattern_id']}|{window['failure_mode']}|L{','.join(map(str, window['layers']))}|S{','.join(map(str, window['steps']))}",
                    "pattern_id": window["pattern_id"],
                    "failure_mode_target": window["failure_mode"],
                    "layers": window["layers"],
                    "steps": window["steps"],
                    "source_group": source_group,
                    "patch_condition": condition,
                    "trajectory_id": row.get("trajectory_id"),
                    "target_label": row.get("target_label"),
                    "object": row.get("object"),
                    **result,
                }
            )
    summary_rows = summarize_rollouts(rollout_rows)
    by = {(r["source_group"], r["patch_condition"]): r for r in summary_rows}
    drift_base = by.get(("drift", "none"), {})
    drift_patch = by.get(("drift", "success_minus_drift"), {})
    success_base = by.get(("success", "none"), {})
    success_patch = by.get(("success", "drift_minus_success"), {})
    summary = {
        "candidate_key": f"{model_name}|{window['pattern_id']}|{window['failure_mode']}|L{','.join(map(str, window['layers']))}|S{','.join(map(str, window['steps']))}",
        "model": model_name,
        "window": window,
        "available_success_rows": len(success_rows),
        "available_drift_rows": len(drift_rows),
        "direction_sites": sum(len(v) for v in direction_vectors.values()),
        "group_rows": summary_rows,
        "repair_match_gain": int(drift_patch.get("pattern_match") or 0) - int(drift_base.get("pattern_match") or 0),
        "damage_match_loss": int(success_base.get("pattern_match") or 0) - int(success_patch.get("pattern_match") or 0),
    }
    return rollout_rows, summary


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = INPUT_ROOT / args.phase210_round
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    rollout_rows: list[dict[str, Any]] = []
    window_summaries: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        rows = load_trajectories(input_dir, args.model)
        prompt_layers = sorted({layer for window in WINDOWS[args.model] for layer in window["layers"]})
        prompt_rows = prompt_trigger_rows(model, tokenizer, device, args.model, rows, prompt_layers, int(args.max_prompt_rows))
        for window in WINDOWS[args.model]:
            wr, summary = eval_window(model, tokenizer, device, groups, args.model, window, rows, args)
            rollout_rows.extend(wr)
            window_summaries.append(summary)
            log(f"{summary['candidate_key']} repair={summary['repair_match_gain']} damage={summary['damage_match_loss']} sites={summary['direction_sites']}")
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Window Direction Patch and Prompt Trigger Atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "rollout_rows": len(rollout_rows),
        "prompt_trigger_rows": len(prompt_rows),
        "window_summaries": window_summaries,
        "prompt_summary_rows": summarize_prompt(prompt_rows),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in window_summaries),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in window_summaries),
    }
    write_json(out_dir / f"phase213_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase213_{args.model}_window_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase213_{args.model}_window_summary_rows.jsonl", window_summaries)
    write_jsonl(out_dir / f"phase213_{args.model}_prompt_trigger_rows.jsonl", prompt_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "rollout_rows": len(rollout_rows), "prompt_trigger_rows": len(prompt_rows), "repair_gain": payload["total_repair_match_gain"], "damage_loss": payload["total_damage_match_loss"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase213_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    window_summaries = []
    prompt_summary_rows = []
    for summary in summaries:
        window_summaries.extend(summary.get("window_summaries") or [])
        prompt_summary_rows.extend(summary.get("prompt_summary_rows") or [])
    payload = {
        "schema_version": "phase213_window_direction_prompt_trigger_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_name,
        "models": [summary.get("model") for summary in summaries],
        "rollout_rows": sum(int(summary.get("rollout_rows") or 0) for summary in summaries),
        "prompt_trigger_rows": sum(int(summary.get("prompt_trigger_rows") or 0) for summary in summaries),
        "window_summaries": window_summaries,
        "prompt_summary_rows": prompt_summary_rows,
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in window_summaries),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in window_summaries),
    }
    write_json(out_dir / "phase213_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase213_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 213 window direction patch and prompt trigger atlas", ""]
    lines.append(f"Rollout rows: {payload.get('rollout_rows')}")
    lines.append(f"Prompt trigger rows: {payload.get('prompt_trigger_rows')}")
    lines.append(f"Total repair match gain: {payload.get('total_repair_match_gain')}")
    lines.append(f"Total damage match loss: {payload.get('total_damage_match_loss')}")
    lines.append("")
    lines.append("| model | window | success rows | drift rows | sites | repair gain | damage loss |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("window_summaries") or []:
        win = row.get("window") or {}
        label = f"{win.get('pattern_id')}->{win.get('failure_mode')} L{win.get('layers')} S{win.get('steps')}"
        lines.append(
            f"| {row.get('model')} | {label} | {row.get('available_success_rows')} | {row.get('available_drift_rows')} | "
            f"{row.get('direction_sites')} | {row.get('repair_match_gain')} | {row.get('damage_match_loss')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="window_direction_prompt_trigger")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-rows-per-group", type=int, default=8)
    parser.add_argument("--max-donor-rows", type=int, default=6)
    parser.add_argument("--max-eval-rows", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--direction-scale", type=float, default=0.7)
    parser.add_argument("--max-prompt-rows", type=int, default=80)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models"), "rollout_rows": payload.get("rollout_rows"), "repair_gain": payload.get("total_repair_match_gain"), "damage_loss": payload.get("total_damage_match_loss")}, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
