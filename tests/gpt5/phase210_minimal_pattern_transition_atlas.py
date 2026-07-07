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
import phase209_pattern_running_contrast_atlas as p209  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 210
SOURCE_PHASE = 209
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")


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


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


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


def scan_layers_for_model(model, text: str) -> list[int]:
    layers = get_layers(model)
    total = len(layers)
    if text:
        return sorted({idx for idx in [int(x) for x in parse_csv(text)] if 0 <= idx < total})
    fractions = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.98]
    return sorted({min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions})


def build_prompt_records(args: argparse.Namespace, holdout_by_pair: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected_pairs = sorted(holdout_by_pair.keys())[: int(args.max_pairs)]
    patterns = parse_csv(args.patterns)
    for relation, language_pair in selected_pairs:
        samples = holdout_by_pair.get((relation, language_pair)) or []
        if int(args.max_samples_per_pair) > 0:
            samples = samples[: int(args.max_samples_per_pair)]
        for sample in samples:
            for pattern in patterns:
                records.append(
                    {
                        **sample,
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase210_prompt_record",
                        "model": args.model,
                        "source_sample_id": sample.get("sample_id"),
                        "trajectory_id": f"{sample.get('sample_id')}|{pattern}",
                        "language_pair": language_pair,
                        "pattern_id": pattern,
                        "expected_output_pattern": p209.expected_output_pattern(pattern),
                        "prompt": p209.pattern_prompt(sample, pattern),
                    }
                )
    return records


def forward_last_state(model, tokenizer, device: torch.device, texts: list[str], selected_layers: list[int]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1).long() - 1
    batch_idx = torch.arange(input_ids.shape[0], device=device)
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    logits = result.logits[batch_idx, last_pos].detach().float().cpu()
    layer_states: dict[int, torch.Tensor] = {}
    for layer_idx in selected_layers:
        hidden_idx = int(layer_idx) + 1
        if hidden_idx < len(result.hidden_states):
            layer_states[int(layer_idx)] = result.hidden_states[hidden_idx][batch_idx, last_pos].detach().float().cpu()
    del result, input_ids, attention_mask
    return logits, layer_states


def add_vector_stat(stats: dict[tuple[Any, ...], dict[str, Any]], key: tuple[Any, ...], vector: torch.Tensor) -> None:
    item = stats.setdefault(key, {"count": 0, "sum": None, "norm_sum": 0.0, "sq_norm_sum": 0.0})
    vec = vector.detach().float().cpu()
    if item["sum"] is None:
        item["sum"] = torch.zeros_like(vec)
    item["sum"] += vec
    norm = float(torch.linalg.vector_norm(vec).item())
    item["norm_sum"] += norm
    item["sq_norm_sum"] += norm * norm
    item["count"] += 1


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
    if denom <= 0:
        return None
    return float(torch.dot(a, b).item() / denom)


def run_transition_atlas(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, Any]],
    groups: dict[str, list[int]],
    selected_layers: list[int],
    max_steps: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    token_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    vector_stats: dict[tuple[Any, ...], dict[str, Any]] = {}
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    for start in range(0, len(prompts), max(1, int(batch_size))):
        batch = [dict(row) for row in prompts[start : start + max(1, int(batch_size))]]
        generated_parts = ["" for _ in batch]
        active = [True for _ in batch]
        emitted_ids: list[list[int]] = [[] for _ in batch]
        emitted_texts: list[list[str]] = [[] for _ in batch]
        first_period_step: list[int | None] = [None for _ in batch]
        for step in range(1, int(max_steps) + 1):
            active_indices = [idx for idx, ok in enumerate(active) if ok]
            if not active_indices:
                break
            texts = [str(batch[idx]["prompt"]) + generated_parts[idx] for idx in active_indices]
            logits_batch, layer_states = forward_last_state(model, tokenizer, device, texts, selected_layers)
            for local_idx, idx in enumerate(active_indices):
                logits = logits_batch[local_idx]
                metrics = p204.metric_for_logits(tokenizer, logits, batch[idx], groups)
                next_id = int(metrics["top_token_id"])
                next_text = str(metrics.get("top_token") or "")
                row_base = {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "model": batch[idx].get("model"),
                    "trajectory_id": batch[idx].get("trajectory_id"),
                    "source_sample_id": batch[idx].get("source_sample_id"),
                    "relation": batch[idx].get("relation"),
                    "language_pair": batch[idx].get("language_pair"),
                    "pattern_id": batch[idx].get("pattern_id"),
                    "expected_output_pattern": batch[idx].get("expected_output_pattern"),
                    "object": batch[idx].get("object"),
                    "target_label": batch[idx].get("target_label"),
                    "step": step,
                    "prefix_generated": generated_parts[idx],
                }
                token_rows.append(
                    {
                        **row_base,
                        "row_kind": "phase210_token_row",
                        "emitted_token_id": next_id,
                        "emitted_token": next_text,
                        "emitted_is_eos": next_id in eos_ids,
                        "emitted_is_period": p204.is_period_token(next_text),
                        **metrics,
                    }
                )
                for layer_idx, states in layer_states.items():
                    vec = states[local_idx]
                    norm = float(torch.linalg.vector_norm(vec).item())
                    add_vector_stat(vector_stats, (batch[idx].get("model"), batch[idx].get("pattern_id"), step, layer_idx), vec)
                    state_rows.append(
                        {
                            **row_base,
                            "row_kind": "phase210_state_row",
                            "layer_idx": int(layer_idx),
                            "residual_norm": norm,
                            "residual_mean": float(vec.mean().item()),
                            "residual_std": float(vec.std(unbiased=False).item()),
                            "target_rank": metrics.get("target_rank"),
                            "stop_margin": metrics.get("stop_margin"),
                            "prose_margin": metrics.get("prose_margin"),
                            "echo_margin": metrics.get("echo_margin"),
                            "eos_rank": metrics.get("eos_rank"),
                            "period_rank": metrics.get("period_rank"),
                        }
                    )
                emitted_ids[idx].append(next_id)
                emitted_texts[idx].append(next_text)
                if p204.is_period_token(next_text) and first_period_step[idx] is None:
                    first_period_step[idx] = step
                generated_parts[idx] += next_text
                if next_id in eos_ids:
                    active[idx] = False
            del logits_batch, layer_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for idx, item in enumerate(batch):
            generated = generated_parts[idx]
            classification = p209.classify_pattern(generated, item, emitted_ids[idx], eos_ids)
            expected = str(item.get("expected_output_pattern") or "")
            trajectory_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase210_pattern_trajectory_row",
                    "model": item.get("model"),
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "relation": item.get("relation"),
                    "language_pair": item.get("language_pair"),
                    "pattern_id": item.get("pattern_id"),
                    "expected_output_pattern": expected,
                    "object": item.get("object"),
                    "target_label": item.get("target_label"),
                    "prompt": item.get("prompt"),
                    "generated": generated,
                    "steps_generated": len(emitted_ids[idx]),
                    "emitted_ids": emitted_ids[idx],
                    "emitted_tokens": emitted_texts[idx],
                    "first_period_step": first_period_step[idx],
                    "period_seen": first_period_step[idx] is not None,
                    "pattern_match": classification.get("output_pattern") == expected,
                    "pattern_drift": classification.get("output_pattern") != expected,
                    "failure_mode": "match" if classification.get("output_pattern") == expected else classification.get("output_pattern"),
                    **classification,
                }
            )
    contrast_rows = build_contrast_rows(vector_stats)
    return token_rows, state_rows, trajectory_rows + contrast_rows


def build_contrast_rows(vector_stats: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    means: dict[tuple[Any, ...], torch.Tensor] = {}
    norm_means: dict[tuple[Any, ...], float] = {}
    for key, item in vector_stats.items():
        count = int(item.get("count") or 0)
        if count <= 0 or item.get("sum") is None:
            continue
        means[key] = item["sum"] / count
        norm_means[key] = float(item["norm_sum"] / count)
    rows: list[dict[str, Any]] = []
    index = {(model, pattern, step, layer) for (model, pattern, step, layer) in means}
    for model, pattern, step, layer_idx in sorted(index, key=lambda x: tuple(str(v) for v in x)):
        if pattern == "answer_short":
            continue
        base_key = (model, "answer_short", step, layer_idx)
        key = (model, pattern, step, layer_idx)
        if base_key not in means or key not in means:
            continue
        diff = means[key] - means[base_key]
        rows.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase210_pattern_contrast_row",
                "model": model,
                "pattern_id": pattern,
                "baseline_pattern_id": "answer_short",
                "step": step,
                "layer_idx": layer_idx,
                "pattern_count": int(vector_stats[key]["count"]),
                "baseline_count": int(vector_stats[base_key]["count"]),
                "mean_vector_l2_diff_vs_short": float(torch.linalg.vector_norm(diff).item()),
                "mean_vector_cosine_vs_short": cosine(means[key], means[base_key]),
                "residual_norm_mean": norm_means[key],
                "baseline_residual_norm_mean": norm_means[base_key],
                "residual_norm_delta_vs_short": norm_means[key] - norm_means[base_key],
            }
        )
    return rows


def summarize_trajectories(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") != "phase210_pattern_trajectory_row":
            continue
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "pattern_drift": sum(1 for item in items if item.get("pattern_drift")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "ended_with_eos": sum(1 for item in items if item.get("ended_with_eos")),
                "avg_steps": mean([item.get("steps_generated") for item in items]),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
            }
        )
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def summarize_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "pattern_id", "step", "layer_idx"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "residual_norm_mean": mean([item.get("residual_norm") for item in items]),
                "residual_mean_mean": mean([item.get("residual_mean") for item in items]),
                "residual_std_mean": mean([item.get("residual_std") for item in items]),
                "target_rank_mean": mean([item.get("target_rank") for item in items]),
                "stop_margin_mean": mean([item.get("stop_margin") for item in items]),
                "prose_margin_mean": mean([item.get("prose_margin") for item in items]),
                "echo_margin_mean": mean([item.get("echo_margin") for item in items]),
                "eos_rank_mean": mean([item.get("eos_rank") for item in items]),
            }
        )
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), int(row.get("step")), int(row.get("layer_idx"))))
    return out


def summarize_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contrast_rows = [row for row in rows if row.get("row_kind") == "phase210_pattern_contrast_row"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "pattern_id", "layer_idx"]
    for row in contrast_rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "mean_l2_diff_vs_short": mean([item.get("mean_vector_l2_diff_vs_short") for item in items]),
                "max_l2_diff_vs_short": max(finite(item.get("mean_vector_l2_diff_vs_short")) for item in items),
                "mean_cosine_vs_short": mean([item.get("mean_vector_cosine_vs_short") for item in items]),
                "mean_norm_delta_vs_short": mean([item.get("residual_norm_delta_vs_short") for item in items]),
            }
        )
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), int(row.get("layer_idx"))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Minimal Pattern Transition Atlas",
        "model": args.model,
        "patterns": parse_csv(args.patterns),
        "max_pairs": int(args.max_pairs),
        "max_samples_per_pair": int(args.max_samples_per_pair),
        "max_steps": int(args.max_steps),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase210_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    meta: dict[str, Any] = {}
    token_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    selected_layers: list[int] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        selected_layers = scan_layers_for_model(model, args.scan_layers)
        groups = p201.token_groups(tokenizer)
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        prompts = build_prompt_records(args, holdout_by_pair)
        token_rows, state_rows, mixed_rows = run_transition_atlas(
            model, tokenizer, device, prompts, groups, selected_layers, int(args.max_steps), int(args.batch_size)
        )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trajectory_rows = [row for row in mixed_rows if row.get("row_kind") == "phase210_pattern_trajectory_row"]
    contrast_rows = [row for row in mixed_rows if row.get("row_kind") == "phase210_pattern_contrast_row"]
    by_pattern = summarize_trajectories(trajectory_rows, ["model", "pattern_id"])
    state_summary_rows = summarize_states(state_rows)
    contrast_summary_rows = summarize_contrasts(mixed_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "selected_layers": selected_layers,
        "prompt_count": len(trajectory_rows),
        "token_rows": len(token_rows),
        "state_rows": len(state_rows),
        "contrast_rows": len(contrast_rows),
        "pattern_match_total": sum(1 for row in trajectory_rows if row.get("pattern_match")),
        "pattern_drift_total": sum(1 for row in trajectory_rows if row.get("pattern_drift")),
        "answer_present_total": sum(1 for row in trajectory_rows if row.get("answer_present")),
        "ended_with_eos_total": sum(1 for row in trajectory_rows if row.get("ended_with_eos")),
        "by_pattern": by_pattern,
        "state_summary_rows": state_summary_rows,
        "contrast_summary_rows": contrast_summary_rows,
        "boundary": "Internal trajectory proxy only: layer last-token hidden-state means and contrasts are not causal proof.",
    }
    write_json(out_dir / f"phase210_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase210_{args.model}_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase210_{args.model}_state_rows.jsonl", state_rows)
    write_jsonl(out_dir / f"phase210_{args.model}_trajectory_rows.jsonl", trajectory_rows)
    write_jsonl(out_dir / f"phase210_{args.model}_contrast_rows.jsonl", contrast_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "prompt_count": len(trajectory_rows),
                "state_rows": len(state_rows),
                "contrast_rows": len(contrast_rows),
                "pattern_match_total": payload["pattern_match_total"],
                "pattern_drift_total": payload["pattern_drift_total"],
                "by_pattern": by_pattern,
                "top_contrasts": sorted(
                    contrast_summary_rows,
                    key=lambda row: finite(row.get("mean_l2_diff_vs_short")),
                    reverse=True,
                )[:10],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase210_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    by_pattern = []
    contrast_summary_rows = []
    for summary in summaries:
        by_pattern.extend(dict(row) for row in summary.get("by_pattern") or [])
        contrast_summary_rows.extend(dict(row) for row in summary.get("contrast_summary_rows") or [])
    payload = {
        "schema_version": "phase210_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "prompt_count_total": sum(int(summary.get("prompt_count") or 0) for summary in summaries),
        "state_rows_total": sum(int(summary.get("state_rows") or 0) for summary in summaries),
        "contrast_rows_total": sum(int(summary.get("contrast_rows") or 0) for summary in summaries),
        "pattern_match_total": sum(int(summary.get("pattern_match_total") or 0) for summary in summaries),
        "pattern_drift_total": sum(int(summary.get("pattern_drift_total") or 0) for summary in summaries),
        "answer_present_total": sum(int(summary.get("answer_present_total") or 0) for summary in summaries),
        "ended_with_eos_total": sum(int(summary.get("ended_with_eos_total") or 0) for summary in summaries),
        "model_summaries": summaries,
        "by_pattern": by_pattern,
        "contrast_summary_rows": contrast_summary_rows,
    }
    write_json(out_dir / "phase210_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase210_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 210 minimal pattern transition atlas", ""]
    lines.append(f"Total prompts: {payload.get('prompt_count_total')}")
    lines.append(f"State rows: {payload.get('state_rows_total')}")
    lines.append(f"Contrast rows: {payload.get('contrast_rows_total')}")
    lines.append(f"Pattern match: {payload.get('pattern_match_total')}")
    lines.append(f"Pattern drift: {payload.get('pattern_drift_total')}")
    lines.append("")
    lines.append("| model | pattern | rows | match | drift | answer | eos | output patterns |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("by_pattern") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('pattern_id')} | {row.get('rows')} | {row.get('pattern_match')} | "
            f"{row.get('pattern_drift')} | {row.get('answer_present')} | {row.get('ended_with_eos')} | {row.get('output_patterns')} |"
        )
    lines.append("")
    lines.append("| model | pattern vs short | layer | rows | mean l2 diff | mean cosine | norm delta |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    top = sorted(
        payload.get("contrast_summary_rows") or [],
        key=lambda row: finite(row.get("mean_l2_diff_vs_short")),
        reverse=True,
    )[:36]
    for row in top:
        lines.append(
            f"| {row.get('model')} | {row.get('pattern_id')} | {row.get('layer_idx')} | {row.get('rows')} | "
            f"{finite(row.get('mean_l2_diff_vs_short')):.4f} | {finite(row.get('mean_cosine_vs_short')):.4f} | "
            f"{finite(row.get('mean_norm_delta_vs_short')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="minimal_pattern_transition_atlas")
    parser.add_argument("--phase944-round", default="activation_weighted_mlp_channel_causal_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--phase943-round", default="consensus_coordinate_component_mapping_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=8)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--min-phase940-bridge-gain", type=float, default=0.02)
    parser.add_argument("--max-specs-per-pair", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--min-train-specs", type=int, default=4)
    parser.add_argument("--min-holdout-specs", type=int, default=3)
    parser.add_argument("--language-pairs", default="en->en,zh->zh")
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-samples-per-pair", type=int, default=8)
    parser.add_argument("--patterns", default="answer_short,answer_explain,answer_list,answer_repeat,answer_target_seeded")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "status": payload.get("status"),
                    "models": payload.get("models"),
                    "prompt_count_total": payload.get("prompt_count_total"),
                    "state_rows_total": payload.get("state_rows_total"),
                    "pattern_match_total": payload.get("pattern_match_total"),
                    "pattern_drift_total": payload.get("pattern_drift_total"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
