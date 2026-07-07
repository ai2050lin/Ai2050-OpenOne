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
from model_utils import get_layers  # noqa: E402


PHASE = 207
SOURCE_PHASE = 206
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase207_eos_positive_head_atlas")


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
    fractions = [0.25, 0.5, 0.75, 0.95]
    return sorted({min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions})


def protocol_prompt(sample: dict[str, Any], protocol: str) -> str:
    base = p200.protocol_prompt(sample, "stop_explicit" if protocol in {"eos_instruction", "final_answer", "chat_eos"} else protocol)
    obj = str(sample.get("object") or "")
    target = str(sample.get("target_label") or "")
    relation = str(sample.get("relation") or "")
    if protocol == "eos_instruction":
        return f"{base}\nAfter the answer, end the response immediately."
    if protocol == "final_answer":
        return f"Give only the final answer, no explanation, no punctuation after it.\nQuestion object: {obj}\nRelation: {relation}\nAnswer:"
    if protocol == "json_answer":
        return f'Return exactly this JSON shape and nothing after it: {{"answer":"{target}"}}\nAnswer:'
    return base


def maybe_chat_prompt(tokenizer, user_text: str, protocol: str) -> str:
    if protocol != "chat_eos" or not hasattr(tokenizer, "apply_chat_template"):
        return user_text
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_text


def build_prompt_records(args: argparse.Namespace, tokenizer, holdout_by_pair: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected_pairs = sorted(holdout_by_pair.keys())[: int(args.max_pairs)]
    for relation, language_pair in selected_pairs:
        samples = holdout_by_pair.get((relation, language_pair)) or []
        if int(args.max_samples_per_pair) > 0:
            samples = samples[: int(args.max_samples_per_pair)]
        for sample in samples:
            for protocol in parse_csv(args.prompt_protocols):
                text = maybe_chat_prompt(tokenizer, protocol_prompt(sample, protocol), protocol)
                records.append(
                    {
                        **sample,
                        "phase": PHASE,
                        "model": args.model,
                        "source_sample_id": sample.get("sample_id"),
                        "trajectory_id": f"{sample.get('sample_id')}|{protocol}",
                        "prompt": text,
                        "prompt_protocol": protocol,
                        "language_pair": language_pair,
                    }
                )
    return records


def run_search(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, Any]],
    groups: dict[str, list[int]],
    max_steps: int,
    batch_size: int,
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
        first_eos_step: list[int | None] = [None for _ in batch]
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
                token_rows.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase207_token_row",
                        "model": batch[idx].get("model"),
                        "trajectory_id": batch[idx].get("trajectory_id"),
                        "source_sample_id": batch[idx].get("source_sample_id"),
                        "relation": batch[idx].get("relation"),
                        "language_pair": batch[idx].get("language_pair"),
                        "prompt_protocol": batch[idx].get("prompt_protocol"),
                        "object": batch[idx].get("object"),
                        "target_label": batch[idx].get("target_label"),
                        "step": step,
                        "prefix_generated": generated_parts[idx],
                        "emitted_token_id": next_id,
                        "emitted_token": next_text,
                        "emitted_is_eos": next_id in eos_ids,
                        "emitted_is_period": p204.is_period_token(next_text),
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
                    first_eos_step[idx] = step
                    active[idx] = False
            del logits_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for idx, item in enumerate(batch):
            generated = generated_parts[idx]
            period_seen = first_period_step[idx] is not None
            ended_with_eos = bool(emitted_ids[idx] and emitted_ids[idx][-1] in eos_ids)
            if ended_with_eos:
                contrast_class = "model_eos_success"
            elif period_seen and continued_after_period[idx]:
                contrast_class = "period_continue_fail"
            elif period_seen:
                contrast_class = "period_client_success_proxy"
            else:
                contrast_class = "no_period_fail"
            rollout = p204.p199.classify_rollout(generated, {**item, "prompt": item.get("prompt")})
            trajectory_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase207_trajectory_row",
                    "model": item.get("model"),
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "relation": item.get("relation"),
                    "language_pair": item.get("language_pair"),
                    "prompt_protocol": item.get("prompt_protocol"),
                    "object": item.get("object"),
                    "target_label": item.get("target_label"),
                    "prompt": item.get("prompt"),
                    "generated": generated,
                    "steps_generated": len(emitted_ids[idx]),
                    "emitted_ids": emitted_ids[idx],
                    "emitted_tokens": emitted_texts[idx],
                    "period_seen": period_seen,
                    "first_period_step": first_period_step[idx],
                    "continued_after_period": continued_after_period[idx],
                    "ended_with_eos": ended_with_eos,
                    "first_eos_step": first_eos_step[idx],
                    "model_stop_executed": ended_with_eos,
                    "contrast_class": contrast_class,
                    **rollout,
                }
            )
    return token_rows, trajectory_rows


def select_state_trajectories(rows: list[dict[str, Any]], max_per_class: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("contrast_class"))].append(row)
    out: list[dict[str, Any]] = []
    priority = ["model_eos_success", "period_client_success_proxy", "period_continue_fail", "no_period_fail"]
    for cls in priority:
        items = sorted(
            buckets.get(cls) or [],
            key=lambda row: (str(row.get("prompt_protocol")), str(row.get("relation")), str(row.get("trajectory_id"))),
        )
        out.extend(items[: int(max_per_class)])
    return out


def build_state_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        tokens = [str(x) for x in row.get("emitted_tokens") or []]
        prompt = str(row.get("prompt") or "")
        specs: list[tuple[str, list[str]]] = [("after_prompt", [])]
        if row.get("first_period_step") is not None:
            pidx = int(row["first_period_step"]) - 1
            specs.append(("before_period", tokens[:pidx]))
            specs.append(("after_period", tokens[: pidx + 1]))
            if pidx + 2 <= len(tokens):
                specs.append(("after_continue1", tokens[: pidx + 2]))
        if row.get("first_eos_step") is not None:
            eidx = int(row["first_eos_step"]) - 1
            specs.append(("before_eos", tokens[:eidx]))
            specs.append(("after_eos", tokens[: eidx + 1]))
        for state_kind, prefix in specs:
            points.append(
                {
                    "phase": PHASE,
                    "model": row.get("model"),
                    "state_key": f"{row.get('trajectory_id')}|{state_kind}",
                    "trajectory_id": row.get("trajectory_id"),
                    "source_sample_id": row.get("source_sample_id"),
                    "relation": row.get("relation"),
                    "language_pair": row.get("language_pair"),
                    "prompt_protocol": row.get("prompt_protocol"),
                    "object": row.get("object"),
                    "target_label": row.get("target_label"),
                    "state_kind": state_kind,
                    "contrast_class": row.get("contrast_class"),
                    "model_stop_executed": row.get("model_stop_executed"),
                    "continued_after_period": row.get("continued_after_period"),
                    "text": prompt + "".join(prefix),
                }
            )
    return points


def module_output_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def capture_state_rows(
    model,
    tokenizer,
    device: torch.device,
    state_points: list[dict[str, Any]],
    layers_to_scan: list[int],
    batch_size: int,
    top_heads: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = p201.token_groups(tokenizer)
    layers = get_layers(model)
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or getattr(model.config, "n_head", 0) or 0)
    state_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for start in range(0, len(state_points), max(1, int(batch_size))):
        batch = state_points[start : start + max(1, int(batch_size))]
        encoded = tokenizer([row["text"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        last_pos = attention_mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        captured_heads: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in layers_to_scan:
            layer = layers[int(layer_idx)]
            o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
            if o_proj is None:
                continue

            def o_proj_pre_hook(_module, inputs, layer_idx=layer_idx):
                if not inputs or not torch.is_tensor(inputs[0]):
                    return None
                hidden = inputs[0]
                pos = last_pos.to(device=hidden.device)
                idx = torch.arange(hidden.shape[0], device=hidden.device)
                captured_heads[int(layer_idx)] = hidden[idx, pos, :].detach().float().cpu()
                return None

            handles.append(o_proj.register_forward_pre_hook(o_proj_pre_hook))
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
                metrics = p204.metric_for_logits(tokenizer, logits[row_idx], point, groups)
                out = {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase207_state_row",
                    **{key: point.get(key) for key in [
                        "state_key",
                        "trajectory_id",
                        "source_sample_id",
                        "model",
                        "relation",
                        "language_pair",
                        "prompt_protocol",
                        "object",
                        "target_label",
                        "state_kind",
                        "contrast_class",
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
                    head_input = captured_heads.get(int(layer_idx))
                    if head_input is not None and num_heads > 0 and head_input.shape[-1] % num_heads == 0:
                        per_head = head_input[row_idx].reshape(num_heads, head_input.shape[-1] // num_heads)
                        norms = torch.linalg.vector_norm(per_head, dim=1)
                        out[f"L{layer_idx}_head_norm_mean"] = float(norms.mean().item())
                        out[f"L{layer_idx}_head_norm_max"] = float(norms.max().item())
                        values, indices = torch.topk(norms, k=min(int(top_heads), int(norms.numel())))
                        for rank, (value, head_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                            head_rows.append(
                                {
                                    "phase": PHASE,
                                    "source_phase": SOURCE_PHASE,
                                    "row_kind": "phase207_head_row",
                                    "model": point.get("model"),
                                    "trajectory_id": point.get("trajectory_id"),
                                    "state_kind": point.get("state_kind"),
                                    "contrast_class": point.get("contrast_class"),
                                    "prompt_protocol": point.get("prompt_protocol"),
                                    "layer_idx": int(layer_idx),
                                    "rank": rank,
                                    "head_idx": int(head_idx),
                                    "head_o_proj_input_norm": float(value),
                                }
                            )
                state_rows.append(out)
            del result, logits
        finally:
            for handle in handles:
                handle.remove()
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state_rows, head_rows


def summarize_by(rows: list[dict[str, Any]], keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row["rows"] = len(items)
        for key in metric_keys:
            row[f"{key}_mean"] = mean([item.get(key) for item in items])
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def summarize_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "prompt_protocol", "contrast_class"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "model_stop_executed": sum(1 for item in items if item.get("model_stop_executed")),
                "period_seen": sum(1 for item in items if item.get("period_seen")),
                "continued_after_period": sum(1 for item in items if item.get("continued_after_period")),
                "avg_steps": mean([item.get("steps_generated") for item in items]),
                "top_generated_prefixes": dict(Counter(str(item.get("generated"))[:48] for item in items).most_common(8)),
            }
        )
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def summarize_heads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "contrast_class", "state_kind", "layer_idx", "head_idx"]
    for row in rows:
        if int(row.get("rank") or 99) > 2:
            continue
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row["rows"] = len(items)
        row["head_o_proj_input_norm_mean"] = mean([item.get("head_o_proj_input_norm") for item in items])
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("contrast_class")), str(row.get("state_kind")), -finite(row.get("head_o_proj_input_norm_mean"))))
    return out[:240]


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Natural EOS Positive Search and Attention-Head Done-State Atlas",
        "model": args.model,
        "prompt_protocols": parse_csv(args.prompt_protocols),
        "decoder_audit": {
            "decoding": "greedy_manual_argmax",
            "do_sample": False,
            "temperature": None,
            "external_stop_sequence": None,
            "max_steps": int(args.max_steps),
        },
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase207_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    model = None
    tokenizer = None
    token_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    layers_to_scan: list[int] = []
    meta: dict[str, Any] = {}
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        layers_to_scan = scan_layers_for_model(model, args.scan_layers)
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        prompts = build_prompt_records(args, tokenizer, holdout_by_pair)
        token_rows, trajectory_rows = run_search(model, tokenizer, device, prompts, groups, int(args.max_steps), int(args.batch_size))
        selected = select_state_trajectories(trajectory_rows, int(args.max_states_per_class))
        state_points = build_state_points(selected)
        state_rows, head_rows = capture_state_rows(model, tokenizer, device, state_points, layers_to_scan, int(args.state_batch_size), int(args.top_heads))
        dry_payload["decoder_audit"].update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "eos_token": tokenizer.eos_token,
                "pad_token_id": tokenizer.pad_token_id,
                "pad_token": tokenizer.pad_token,
                "generation_config_eos_token_id": getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                "generation_config_pad_token_id": getattr(getattr(model, "generation_config", None), "pad_token_id", None),
            }
        )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    trajectory_summary_rows = summarize_trajectories(trajectory_rows)
    state_summary_rows = summarize_by(
        state_rows,
        ["model", "contrast_class", "state_kind", "prompt_protocol"],
        ["eos_rank", "period_rank", "prose_rank", "echo_rank", "stop_margin", "prose_margin", "echo_margin"],
    )
    head_summary_rows = summarize_heads(head_rows)
    eos_positive_rows = [row for row in trajectory_rows if row.get("model_stop_executed")]
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "layers_to_scan": layers_to_scan,
        "prompt_count": len({row.get("trajectory_id") for row in trajectory_rows}),
        "token_rows": len(token_rows),
        "trajectory_rows": len(trajectory_rows),
        "state_rows": len(state_rows),
        "head_rows": len(head_rows),
        "eos_positive_count": len(eos_positive_rows),
        "trajectory_summary_rows": trajectory_summary_rows,
        "state_summary_rows": state_summary_rows,
        "head_summary_rows": head_summary_rows,
        "boundary": "EOS positive search and head atlas only. Absence of EOS positives is a negative result, not proof of impossibility.",
    }
    write_json(out_dir / f"phase207_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase207_{args.model}_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase207_{args.model}_trajectory_rows.jsonl", trajectory_rows)
    write_jsonl(out_dir / f"phase207_{args.model}_state_rows.jsonl", state_rows)
    write_jsonl(out_dir / f"phase207_{args.model}_head_rows.jsonl", head_rows)
    write_jsonl(out_dir / f"phase207_{args.model}_eos_positive_sample_bank.jsonl", eos_positive_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "prompt_count": payload["prompt_count"],
                "trajectory_rows": len(trajectory_rows),
                "eos_positive_count": len(eos_positive_rows),
                "layers_to_scan": layers_to_scan,
                "top_trajectory_summary_rows": trajectory_summary_rows[:16],
                "top_head_summary_rows": head_summary_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase207_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    trajectory_summary_rows = []
    state_summary_rows = []
    head_summary_rows = []
    for summary in summaries:
        trajectory_summary_rows.extend(dict(row) for row in summary.get("trajectory_summary_rows") or [])
        state_summary_rows.extend(dict(row) for row in summary.get("state_summary_rows") or [])
        head_summary_rows.extend(dict(row) for row in summary.get("head_summary_rows") or [])
    payload = {
        "schema_version": "phase207_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "total_eos_positive_count": sum(int(summary.get("eos_positive_count") or 0) for summary in summaries),
        "model_summaries": summaries,
        "trajectory_summary_rows": trajectory_summary_rows,
        "state_summary_rows": state_summary_rows,
        "head_summary_rows": head_summary_rows,
    }
    write_json(out_dir / "phase207_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase207_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 207 natural EOS positive and attention-head atlas", ""]
    lines.append(f"Total EOS positives: {payload.get('total_eos_positive_count')}")
    lines.append("")
    lines.append("## Trajectory Summary")
    lines.append("| model | protocol | class | rows | eos | period | continued | avg steps |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("trajectory_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('prompt_protocol')} | {row.get('contrast_class')} | {row.get('rows')} | "
            f"{row.get('model_stop_executed')} | {row.get('period_seen')} | {row.get('continued_after_period')} | {finite(row.get('avg_steps')):.2f} |"
        )
    lines.append("")
    lines.append("## Head Summary")
    lines.append("| model | class | state | layer | head | rows | norm |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in (payload.get("head_summary_rows") or [])[:160]:
        lines.append(
            f"| {row.get('model')} | {row.get('contrast_class')} | {row.get('state_kind')} | {row.get('layer_idx')} | "
            f"{row.get('head_idx')} | {row.get('rows')} | {finite(row.get('head_o_proj_input_norm_mean')):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="eos_positive_head_atlas")
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
    parser.add_argument("--max-samples-per-pair", type=int, default=12)
    parser.add_argument("--prompt-protocols", default="plain,short_answer,stop_explicit,eos_instruction,final_answer,chat_eos")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-batch-size", type=int, default=4)
    parser.add_argument("--max-states-per-class", type=int, default=24)
    parser.add_argument("--top-heads", type=int, default=3)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models"), "eos_positive": payload.get("total_eos_positive_count")}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
