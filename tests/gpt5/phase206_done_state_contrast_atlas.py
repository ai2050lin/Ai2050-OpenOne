#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
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
import phase200_protocol_gated_rollout_repair_audit as p200  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
import phase205_stop_execution_source_localization_audit as p205  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 206
SOURCE_PHASE = 205
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase206_done_state_contrast_atlas")


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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[Any]) -> float | None:
    vals: list[float] = []
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


def token_text(tokenizer, token_id: int | None) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def is_external_stop_token(text: str, stop_rule: str) -> bool:
    stripped = str(text or "").strip()
    if stop_rule == "period":
        return stripped in {".", "。"}
    if stop_rule == "newline":
        return "\n" in str(text or "")
    if stop_rule == "period_or_newline":
        return stripped in {".", "。"} or "\n" in str(text or "")
    return False


def scan_layers_for_model(model, text: str) -> list[int]:
    layers = get_layers(model)
    total = len(layers)
    if text:
        return sorted({idx for idx in [int(x) for x in parse_csv(text)] if 0 <= idx < total})
    fractions = [0.25, 0.4, 0.55, 0.7, 0.85, 0.97]
    return sorted({min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions})


def build_prompt_record(sample: dict[str, Any], protocol: str, rollout_mode: str, language_pair: str, model_name: str) -> dict[str, Any]:
    base_prompt = p200.protocol_prompt(sample, protocol)
    answer = str(sample.get("target_label") or "").strip()
    if rollout_mode == "natural":
        prompt = base_prompt
    elif rollout_mode == "post_answer":
        prompt = f"{base_prompt.rstrip()} {answer}".strip()
    else:
        raise ValueError(f"unsupported rollout_mode: {rollout_mode}")
    return {
        **sample,
        "phase": PHASE,
        "model": model_name,
        "source_sample_id": sample.get("sample_id"),
        "trajectory_id": f"{sample.get('sample_id')}|{protocol}|{rollout_mode}",
        "prompt": prompt,
        "base_prompt": base_prompt,
        "answer": answer,
        "prompt_protocol": protocol,
        "rollout_mode": rollout_mode,
        "language_pair": language_pair,
    }


def run_contrast_trajectories(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, Any]],
    groups: dict[str, list[int]],
    max_steps: int,
    batch_size: int,
    stop_rule: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    for start in range(0, len(prompts), max(1, int(batch_size))):
        batch = [dict(row) for row in prompts[start : start + max(1, int(batch_size))]]
        generated_parts = ["" for _ in batch]
        active = [True for _ in batch]
        emitted_ids: list[list[int]] = [[] for _ in batch]
        emitted_texts: list[list[str]] = [[] for _ in batch]
        first_period_step: list[int | None] = [None for _ in batch]
        first_external_stop_step: list[int | None] = [None for _ in batch]
        continued_after_period = [False for _ in batch]
        for step in range(1, int(max_steps) + 1):
            active_indices = [idx for idx, ok in enumerate(active) if ok]
            if not active_indices:
                break
            texts = [str(batch[idx]["prompt"]) + generated_parts[idx] for idx in active_indices]
            logits_batch = p204.batch_next_logits(model, tokenizer, device, texts)
            for local_idx, idx in enumerate(active_indices):
                logits = logits_batch[local_idx]
                metrics = p204.metric_for_logits(tokenizer, logits, batch[idx], groups)
                next_id = int(metrics["top_token_id"])
                next_text = str(metrics.get("top_token") or "")
                period_seen_before = first_period_step[idx] is not None
                if period_seen_before and next_id not in eos_ids:
                    continued_after_period[idx] = True
                external_stop_before_emit = is_external_stop_token(next_text, stop_rule)
                token_rows.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase206_token_row",
                        "model": batch[idx].get("model"),
                        "trajectory_id": batch[idx].get("trajectory_id"),
                        "source_sample_id": batch[idx].get("source_sample_id"),
                        "relation": batch[idx].get("relation"),
                        "language_pair": batch[idx].get("language_pair"),
                        "prompt_protocol": batch[idx].get("prompt_protocol"),
                        "rollout_mode": batch[idx].get("rollout_mode"),
                        "external_stop_rule": stop_rule,
                        "object": batch[idx].get("object"),
                        "target_label": batch[idx].get("target_label"),
                        "step": step,
                        "prefix_generated": generated_parts[idx],
                        "emitted_token_id": next_id,
                        "emitted_token": next_text,
                        "emitted_is_eos": next_id in eos_ids,
                        "emitted_is_external_stop": external_stop_before_emit,
                        "period_seen_before_step": period_seen_before,
                        "continued_after_period_before_step": period_seen_before and next_id not in eos_ids,
                        **metrics,
                    }
                )
                emitted_ids[idx].append(next_id)
                emitted_texts[idx].append(next_text)
                if p204.is_period_token(next_text) and first_period_step[idx] is None:
                    first_period_step[idx] = step
                generated_parts[idx] += next_text
                if next_id in eos_ids:
                    active[idx] = False
                elif stop_rule != "none" and external_stop_before_emit:
                    first_external_stop_step[idx] = step
                    active[idx] = False
            del logits_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for idx, item in enumerate(batch):
            generated = generated_parts[idx]
            rollout = p204.p199.classify_rollout(generated, {**item, "prompt": item.get("prompt")})
            period_seen = first_period_step[idx] is not None
            ended_with_eos = bool(emitted_ids[idx] and emitted_ids[idx][-1] in eos_ids)
            external_stop_executed = bool(first_external_stop_step[idx] is not None)
            task_stop_satisfied = bool(ended_with_eos or external_stop_executed or (period_seen and not continued_after_period[idx]))
            words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", str(generated))
            trajectory_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase206_trajectory_row",
                    "model": item.get("model"),
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "relation": item.get("relation"),
                    "language_pair": item.get("language_pair"),
                    "prompt_protocol": item.get("prompt_protocol"),
                    "rollout_mode": item.get("rollout_mode"),
                    "external_stop_rule": stop_rule,
                    "object": item.get("object"),
                    "target_label": item.get("target_label"),
                    "prompt": item.get("prompt"),
                    "generated": generated,
                    "generated_word_count": len(words),
                    "steps_generated": len(emitted_ids[idx]),
                    "emitted_ids": emitted_ids[idx],
                    "emitted_tokens": emitted_texts[idx],
                    "period_seen": period_seen,
                    "first_period_step": first_period_step[idx],
                    "continued_after_period": continued_after_period[idx],
                    "ended_with_eos": ended_with_eos,
                    "external_stop_executed": external_stop_executed,
                    "first_external_stop_step": first_external_stop_step[idx],
                    "model_stop_executed": ended_with_eos,
                    "task_stop_satisfied": task_stop_satisfied,
                    "contrast_label": "success" if task_stop_satisfied else "fail",
                    **rollout,
                }
            )
    return token_rows, trajectory_rows


def build_state_points(
    tokenizer,
    prompts: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    max_state_prompts: int,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    eos_text = tokenizer.eos_token or ""
    selected_prompts = prompts[: int(max_state_prompts)]
    for item in selected_prompts:
        answer = str(item.get("answer") or item.get("target_label") or "").strip()
        base_prompt = str(item.get("base_prompt") or item.get("prompt") or "")
        after_answer_text = f"{base_prompt.rstrip()} {answer}".strip()
        forced_period_text = f"{after_answer_text}."
        forced_eos_text = f"{after_answer_text}{eos_text}"
        for state_kind, text in [
            ("before_answer", base_prompt),
            ("after_answer", after_answer_text),
            ("forced_period", forced_period_text),
            ("forced_eos", forced_eos_text),
        ]:
            points.append(
                {
                    "phase": PHASE,
                    "model": item.get("model"),
                    "state_key": f"{item.get('trajectory_id')}|{state_kind}",
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "relation": item.get("relation"),
                    "language_pair": item.get("language_pair"),
                    "prompt_protocol": item.get("prompt_protocol"),
                    "rollout_mode": "state_replay",
                    "external_stop_rule": "forced_context",
                    "object": item.get("object"),
                    "target_label": item.get("target_label"),
                    "answer": answer,
                    "state_kind": state_kind,
                    "contrast_label": "forced_eos_success_proxy" if state_kind == "forced_eos" else None,
                    "text": text,
                }
            )
    for row in trajectory_rows:
        tokens = [str(x) for x in row.get("emitted_tokens") or []]
        prompt = str(row.get("prompt") or "")
        try:
            period_idx = int(row.get("first_period_step")) - 1
        except (TypeError, ValueError):
            period_idx = -1
        specs: list[tuple[str, list[str]]] = []
        if period_idx >= 0 and period_idx < len(tokens):
            specs.append(("after_period", tokens[: period_idx + 1]))
            if period_idx + 2 <= len(tokens):
                specs.append(("after_continue1", tokens[: period_idx + 2]))
            if period_idx + 3 <= len(tokens):
                specs.append(("after_continue2", tokens[: period_idx + 3]))
        for state_kind, prefix_tokens in specs:
            points.append(
                {
                    "phase": PHASE,
                    "model": row.get("model"),
                    "state_key": f"{row.get('trajectory_id')}|{row.get('external_stop_rule')}|{state_kind}",
                    "trajectory_id": row.get("trajectory_id"),
                    "source_sample_id": row.get("source_sample_id"),
                    "relation": row.get("relation"),
                    "language_pair": row.get("language_pair"),
                    "prompt_protocol": row.get("prompt_protocol"),
                    "rollout_mode": row.get("rollout_mode"),
                    "external_stop_rule": row.get("external_stop_rule"),
                    "object": row.get("object"),
                    "target_label": row.get("target_label"),
                    "state_kind": state_kind,
                    "contrast_label": row.get("contrast_label"),
                    "task_stop_satisfied": row.get("task_stop_satisfied"),
                    "model_stop_executed": row.get("model_stop_executed"),
                    "continued_after_period": row.get("continued_after_period"),
                    "text": prompt + "".join(prefix_tokens),
                }
            )
    return points


def capture_state_rows(
    model,
    tokenizer,
    device: torch.device,
    state_points: list[dict[str, Any]],
    layers_to_scan: list[int],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    groups = p201.token_groups(tokenizer)
    state_rows: dict[str, dict[str, Any]] = {}
    for start in range(0, len(state_points), max(1, int(batch_size))):
        batch = state_points[start : start + max(1, int(batch_size))]
        encoded = tokenizer([row["text"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        last_pos = attention_mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(input_ids.shape[0], device=device)
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
            metrics = p204.metric_for_logits(tokenizer, logits[row_idx], point, groups)
            out = {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase206_state_row",
                **{key: point.get(key) for key in [
                    "state_key",
                    "trajectory_id",
                    "source_sample_id",
                    "model",
                    "relation",
                    "language_pair",
                    "prompt_protocol",
                    "rollout_mode",
                    "external_stop_rule",
                    "object",
                    "target_label",
                    "state_kind",
                    "contrast_label",
                    "task_stop_satisfied",
                    "model_stop_executed",
                    "continued_after_period",
                ]},
                "true_text_len": int(last_pos[row_idx].item()) + 1,
                **metrics,
            }
            for layer_idx in layers_to_scan:
                hidden_idx = min(int(layer_idx) + 1, len(result.hidden_states) - 1)
                h = result.hidden_states[hidden_idx][row_idx, int(last_pos[row_idx].item())].detach().float().cpu()
                out[f"L{layer_idx}_resid_norm"] = float(torch.linalg.vector_norm(h).item())
                out[f"L{layer_idx}_resid_mean_abs"] = float(torch.mean(torch.abs(h)).item())
            state_rows[str(point["state_key"])] = out
        del result, logits, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state_rows


def summarize_by(rows: list[dict[str, Any]], keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "model_stop_executed": sum(1 for item in items if item.get("model_stop_executed")),
                "task_stop_satisfied": sum(1 for item in items if item.get("task_stop_satisfied")),
                "period_seen": sum(1 for item in items if item.get("period_seen")),
                "continued_after_period": sum(1 for item in items if item.get("continued_after_period")),
                "external_stop_executed": sum(1 for item in items if item.get("external_stop_executed")),
            }
        )
        for key in metric_keys:
            row[f"{key}_mean"] = mean([item.get(key) for item in items])
        if items and "generated" in items[0]:
            row["top_generated_prefixes"] = dict(Counter(str(item.get("generated"))[:40] for item in items).most_common(8))
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def state_contrast_rows(state_rows: list[dict[str, Any]], layers_to_scan: list[int]) -> list[dict[str, Any]]:
    keys = ["model", "state_kind", "external_stop_rule", "contrast_label"]
    metric_keys = ["eos_rank", "period_rank", "prose_rank", "echo_rank", "stop_margin", "prose_margin", "echo_margin", "eos_vs_prose_margin"]
    rows = summarize_by(state_rows, keys, metric_keys)
    for row in rows:
        matching = [
            item for item in state_rows
            if all(item.get(key) == row.get(key) for key in keys)
        ]
        for layer_idx in layers_to_scan:
            row[f"L{layer_idx}_resid_norm_mean"] = mean([item.get(f"L{layer_idx}_resid_norm") for item in matching])
    return rows


def forced_delta_rows(state_rows: list[dict[str, Any]], layers_to_scan: list[int]) -> list[dict[str, Any]]:
    by_traj: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_rows:
        if row.get("external_stop_rule") == "forced_context":
            by_traj[str(row.get("trajectory_id"))][str(row.get("state_kind"))] = row
    out = []
    metrics = ["eos_rank", "period_rank", "prose_rank", "echo_rank", "stop_margin", "prose_margin", "echo_margin", "eos_vs_prose_margin"]
    for trajectory_id, states in by_traj.items():
        for src, dst in [("after_answer", "forced_period"), ("after_answer", "forced_eos"), ("forced_period", "forced_eos")]:
            a = states.get(src)
            b = states.get(dst)
            if not a or not b:
                continue
            row = {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase206_forced_delta_row",
                "trajectory_id": trajectory_id,
                "model": a.get("model"),
                "relation": a.get("relation"),
                "language_pair": a.get("language_pair"),
                "prompt_protocol": a.get("prompt_protocol"),
                "transition": f"{src}->{dst}",
            }
            for key in metrics:
                row[f"{key}_delta"] = None if a.get(key) is None or b.get(key) is None else finite(b.get(key)) - finite(a.get(key))
            for layer_idx in layers_to_scan:
                key = f"L{layer_idx}_resid_norm"
                if a.get(key) is not None and b.get(key) is not None:
                    row[f"{key}_delta"] = finite(b.get(key)) - finite(a.get(key))
            out.append(row)
    return out


def summarize_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["model", "transition", "prompt_protocol"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
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
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_protocols = parse_csv(args.prompt_protocols)
    rollout_modes = parse_csv(args.rollout_modes)
    external_stop_rules = parse_csv(args.external_stop_rules)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done-State Success/Failure Contrast Atlas",
        "model": args.model,
        "prompt_protocols": prompt_protocols,
        "rollout_modes": rollout_modes,
        "external_stop_rules": external_stop_rules,
        "decoder_audit": {
            "decoding": "greedy",
            "internal_eos_stop_enabled": True,
            "external_stop_rules_are_client_side_simulations": True,
            "max_steps": int(args.max_steps),
        },
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase206_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    token_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    state_rows: dict[str, dict[str, Any]] = {}
    forced_deltas: list[dict[str, Any]] = []
    layers_to_scan: list[int] = []
    meta: dict[str, Any] = {}
    prompt_records: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        layers_to_scan = scan_layers_for_model(model, args.scan_layers)
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        selected_pairs = sorted(holdout_by_pair.keys())[: int(args.max_pairs)]
        for relation, language_pair in selected_pairs:
            samples = holdout_by_pair.get((relation, language_pair)) or []
            if int(args.max_samples_per_pair) > 0:
                samples = samples[: int(args.max_samples_per_pair)]
            for sample in samples:
                for protocol in prompt_protocols:
                    for rollout_mode in rollout_modes:
                        prompt_records.append(build_prompt_record(sample, protocol, rollout_mode, language_pair, args.model))
        for stop_rule in external_stop_rules:
            rows, summaries = run_contrast_trajectories(
                model,
                tokenizer,
                device,
                prompt_records,
                groups,
                int(args.max_steps),
                int(args.batch_size),
                stop_rule,
            )
            token_rows.extend(rows)
            trajectory_rows.extend(summaries)
        state_points = build_state_points(tokenizer, prompt_records, trajectory_rows, int(args.max_state_prompts))
        state_rows = capture_state_rows(model, tokenizer, device, state_points, layers_to_scan, int(args.batch_size))
        forced_deltas = forced_delta_rows(list(state_rows.values()), layers_to_scan)
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trajectory_summary_rows = summarize_by(
        trajectory_rows,
        ["model", "external_stop_rule", "rollout_mode", "prompt_protocol"],
        ["steps_generated"],
    )
    token_step_summary_rows = summarize_by(
        token_rows,
        ["model", "external_stop_rule", "rollout_mode", "prompt_protocol", "step"],
        ["eos_rank", "period_rank", "prose_rank", "stop_margin", "prose_margin", "echo_margin"],
    )
    state_summary_rows = state_contrast_rows(list(state_rows.values()), layers_to_scan)
    forced_delta_summary_rows = summarize_delta_rows(forced_deltas)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "prompt_count": len(prompt_records),
        "token_rows": len(token_rows),
        "trajectory_rows": len(trajectory_rows),
        "state_rows": len(state_rows),
        "forced_delta_rows": len(forced_deltas),
        "layers_to_scan": layers_to_scan,
        "trajectory_summary_rows": trajectory_summary_rows,
        "token_step_summary_rows": token_step_summary_rows,
        "state_summary_rows": state_summary_rows,
        "forced_delta_summary_rows": forced_delta_summary_rows,
        "boundary": "Contrast atlas only. External stop rows are client-side task-stop simulations, not internal EOS execution.",
    }
    write_json(out_dir / f"phase206_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase206_{args.model}_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase206_{args.model}_trajectory_rows.jsonl", trajectory_rows)
    write_jsonl(out_dir / f"phase206_{args.model}_state_rows.jsonl", list(state_rows.values()))
    write_jsonl(out_dir / f"phase206_{args.model}_forced_delta_rows.jsonl", forced_deltas)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "prompt_count": len(prompt_records),
                "trajectory_rows": len(trajectory_rows),
                "state_rows": len(state_rows),
                "layers_to_scan": layers_to_scan,
                "top_trajectory_summary_rows": trajectory_summary_rows[:12],
                "top_forced_delta_summary_rows": forced_delta_summary_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase206_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    trajectory_summary_rows = []
    state_summary_rows = []
    forced_delta_summary_rows = []
    for summary in summaries:
        trajectory_summary_rows.extend(dict(row) for row in summary.get("trajectory_summary_rows") or [])
        state_summary_rows.extend(dict(row) for row in summary.get("state_summary_rows") or [])
        forced_delta_summary_rows.extend(dict(row) for row in summary.get("forced_delta_summary_rows") or [])
    payload = {
        "schema_version": "phase206_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "trajectory_summary_rows": trajectory_summary_rows,
        "state_summary_rows": state_summary_rows,
        "forced_delta_summary_rows": forced_delta_summary_rows,
    }
    write_json(out_dir / "phase206_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase206_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 206 done-state contrast atlas", ""]
    lines.append("## Trajectory Contrast")
    lines.append("| model | stop rule | mode | protocol | rows | model stop | task stop | period | continued | external stop | avg steps |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("trajectory_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('external_stop_rule')} | {row.get('rollout_mode')} | {row.get('prompt_protocol')} | "
            f"{row.get('rows')} | {row.get('model_stop_executed')} | {row.get('task_stop_satisfied')} | {row.get('period_seen')} | "
            f"{row.get('continued_after_period')} | {row.get('external_stop_executed')} | {finite(row.get('steps_generated_mean')):.2f} |"
        )
    lines.append("")
    lines.append("## Forced Context Delta")
    lines.append("| model | transition | protocol | rows | eos rank delta | prose rank delta | stop margin delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("forced_delta_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('transition')} | {row.get('prompt_protocol')} | {row.get('rows')} | "
            f"{finite(row.get('eos_rank_delta_mean')):.2f} | {finite(row.get('prose_rank_delta_mean')):.2f} | "
            f"{finite(row.get('stop_margin_delta_mean')):.2f} |"
        )
    lines.append("")
    lines.append("## State Summary")
    lines.append("| model | state | stop rule | label | rows | eos rank | prose rank | stop margin |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in (payload.get("state_summary_rows") or [])[:160]:
        lines.append(
            f"| {row.get('model')} | {row.get('state_kind')} | {row.get('external_stop_rule')} | {row.get('contrast_label')} | "
            f"{row.get('rows')} | {finite(row.get('eos_rank_mean')):.2f} | {finite(row.get('prose_rank_mean')):.2f} | "
            f"{finite(row.get('stop_margin_mean')):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="done_state_contrast_atlas")
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
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--max-samples-per-pair", type=int, default=16)
    parser.add_argument("--prompt-protocols", default="plain,short_answer,stop_explicit")
    parser.add_argument("--rollout-modes", default="natural,post_answer")
    parser.add_argument("--external-stop-rules", default="none,period")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-state-prompts", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models")}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
