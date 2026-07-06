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
import phase940_semantic_boundary_bridge_audit as p940  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402
import phase199_l4_edge_natural_gate_rollout_audit as p199  # noqa: E402
import phase200_protocol_gated_rollout_repair_audit as p200  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402


PHASE = 204
SOURCE_PHASE = 203
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase204_global_trajectory_stop_execution_atlas")


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


def max_score(logits: torch.Tensor, token_ids: list[int]) -> tuple[float | None, int | None, int | None]:
    valid = [int(x) for x in token_ids if 0 <= int(x) < int(logits.numel())]
    if not valid:
        return None, None, None
    best_id = max(valid, key=lambda token_id: float(logits[token_id].item()))
    rank = int((logits.float() > logits[best_id].float()).sum().item()) + 1
    return float(logits[best_id].item()), int(best_id), rank


def object_ids(tokenizer, sample: dict[str, Any]) -> list[int]:
    return p201.object_token_ids(tokenizer, sample)


def build_prompt_record(sample: dict[str, Any], protocol: str, rollout_mode: str, language_pair: str) -> dict[str, Any]:
    if rollout_mode == "post_answer":
        prompt = f"{p200.protocol_prompt(sample, protocol).rstrip()} {str(sample.get('target_label') or '').strip()}".strip()
    else:
        prompt = p200.protocol_prompt(sample, protocol)
    return {
        **sample,
        "trajectory_id": f"{sample.get('sample_id')}|{protocol}|{rollout_mode}",
        "source_sample_id": sample.get("sample_id"),
        "prompt": prompt,
        "prompt_protocol": protocol,
        "rollout_mode": rollout_mode,
        "language_pair": language_pair,
    }


def token_text(tokenizer, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def is_period_token(text: str | None) -> bool:
    return str(text or "").strip() in {".", "。"}


def is_newline_token(text: str | None) -> bool:
    return "\n" in str(text or "")


def metric_for_logits(tokenizer, logits: torch.Tensor, sample: dict[str, Any], groups: dict[str, list[int]]) -> dict[str, Any]:
    target_ids = p938.first_token_candidates(tokenizer, str(sample.get("target_label") or ""))
    target_score, target_id, target_rank = max_score(logits, target_ids)
    stop_score, stop_id, stop_rank = max_score(logits, groups["stop"])
    period_score, period_id, period_rank = max_score(logits, groups["period"])
    eos_score, eos_id, eos_rank = max_score(logits, groups["eos"])
    newline_score, newline_id, newline_rank = max_score(logits, groups["newline"])
    prose_score, prose_id, prose_rank = max_score(logits, groups["prose"])
    echo_score, echo_id, echo_rank = max_score(logits, object_ids(tokenizer, sample))
    top_id = int(torch.argmax(logits).item())
    top_text = token_text(tokenizer, top_id)
    return {
        "target_logit": target_score,
        "target_token_id": target_id,
        "target_token": token_text(tokenizer, target_id),
        "target_rank": target_rank,
        "stop_logit": stop_score,
        "stop_token_id": stop_id,
        "stop_token": token_text(tokenizer, stop_id),
        "stop_rank": stop_rank,
        "period_logit": period_score,
        "period_token_id": period_id,
        "period_token": token_text(tokenizer, period_id),
        "period_rank": period_rank,
        "eos_logit": eos_score,
        "eos_token_id": eos_id,
        "eos_rank": eos_rank,
        "newline_logit": newline_score,
        "newline_token_id": newline_id,
        "newline_rank": newline_rank,
        "prose_logit": prose_score,
        "prose_token_id": prose_id,
        "prose_token": token_text(tokenizer, prose_id),
        "prose_rank": prose_rank,
        "echo_logit": echo_score,
        "echo_token_id": echo_id,
        "echo_token": token_text(tokenizer, echo_id),
        "echo_rank": echo_rank,
        "stop_margin": None if stop_score is None or prose_score is None else float(stop_score - prose_score),
        "prose_margin": None if prose_score is None or stop_score is None else float(prose_score - stop_score),
        "echo_margin": None if echo_score is None or stop_score is None else float(echo_score - stop_score),
        "period_vs_prose_margin": None if period_score is None or prose_score is None else float(period_score - prose_score),
        "eos_vs_prose_margin": None if eos_score is None or prose_score is None else float(eos_score - prose_score),
        "top_token_id": top_id,
        "top_token": top_text,
    }


def batch_next_logits(model, tokenizer, device: torch.device, texts: list[str]) -> torch.Tensor:
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1).long() - 1
    batch_idx = torch.arange(input_ids.shape[0], device=device)
    with torch.inference_mode():
        result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = result.logits[batch_idx, last_pos].detach().float().cpu()
    del result, input_ids, attention_mask
    return logits


def run_trajectories(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, Any]],
    groups: dict[str, list[int]],
    max_steps: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    for start in range(0, len(prompts), max(1, int(batch_size))):
        batch = [dict(row) for row in prompts[start : start + max(1, int(batch_size))]]
        generated_parts = ["" for _ in batch]
        active = [True for _ in batch]
        emitted_ids: list[list[int]] = [[] for _ in batch]
        emitted_texts: list[list[str]] = [[] for _ in batch]
        period_seen = [False for _ in batch]
        first_period_step = [None for _ in batch]
        continued_after_period = [False for _ in batch]
        stop_positive_chain_len = [0 for _ in batch]
        for step in range(1, int(max_steps) + 1):
            active_indices = [idx for idx, ok in enumerate(active) if ok]
            if not active_indices:
                break
            texts = [str(batch[idx]["prompt"]) + generated_parts[idx] for idx in active_indices]
            logits_batch = batch_next_logits(model, tokenizer, device, texts)
            for local_idx, idx in enumerate(active_indices):
                logits = logits_batch[local_idx]
                metrics = metric_for_logits(tokenizer, logits, batch[idx], groups)
                next_id = int(metrics["top_token_id"])
                next_text = str(metrics.get("top_token") or "")
                current_period_seen = bool(period_seen[idx])
                if current_period_seen and next_id not in eos_ids:
                    continued_after_period[idx] = True
                if finite(metrics.get("stop_margin"), -999.0) > 0:
                    stop_positive_chain_len[idx] += 1
                else:
                    stop_positive_chain_len[idx] = 0
                rows.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase204_token_trajectory_row",
                        "model": batch[idx].get("model"),
                        "trajectory_id": batch[idx].get("trajectory_id"),
                        "source_sample_id": batch[idx].get("source_sample_id"),
                        "relation": batch[idx].get("relation"),
                        "language_pair": batch[idx].get("language_pair"),
                        "prompt_protocol": batch[idx].get("prompt_protocol"),
                        "rollout_mode": batch[idx].get("rollout_mode"),
                        "object": batch[idx].get("object"),
                        "target_label": batch[idx].get("target_label"),
                        "step": step,
                        "prefix_generated": generated_parts[idx],
                        "emitted_token_id": next_id,
                        "emitted_token": next_text,
                        "emitted_is_eos": next_id in eos_ids,
                        "emitted_is_period": is_period_token(next_text),
                        "emitted_is_newline": is_newline_token(next_text),
                        "period_seen_before_step": current_period_seen,
                        "continued_after_period_before_step": current_period_seen and next_id not in eos_ids,
                        "stop_positive_chain_len_before_emit": stop_positive_chain_len[idx],
                        **metrics,
                    }
                )
                emitted_ids[idx].append(next_id)
                emitted_texts[idx].append(next_text)
                if is_period_token(next_text) and not period_seen[idx]:
                    period_seen[idx] = True
                    first_period_step[idx] = step
                generated_parts[idx] += next_text
                if next_id in eos_ids:
                    active[idx] = False
            del logits_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for idx, item in enumerate(batch):
            generated = generated_parts[idx]
            rollout = p199.classify_rollout(generated, {**item, "prompt": item.get("prompt")})
            words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", str(generated))
            summary_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase204_trajectory_summary_row",
                    "model": item.get("model"),
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "relation": item.get("relation"),
                    "language_pair": item.get("language_pair"),
                    "prompt_protocol": item.get("prompt_protocol"),
                    "rollout_mode": item.get("rollout_mode"),
                    "object": item.get("object"),
                    "target_label": item.get("target_label"),
                    "prompt": item.get("prompt"),
                    "generated": generated,
                    "generated_word_count": len(words),
                    "steps_generated": len(emitted_ids[idx]),
                    "emitted_ids": emitted_ids[idx],
                    "emitted_tokens": emitted_texts[idx],
                    "period_seen": period_seen[idx],
                    "first_period_step": first_period_step[idx],
                    "continued_after_period": continued_after_period[idx],
                    "ended_with_eos": bool(emitted_ids[idx] and emitted_ids[idx][-1] in eos_ids),
                    "max_stop_positive_chain_len": max(
                        [row.get("stop_positive_chain_len_before_emit", 0) for row in rows if row.get("trajectory_id") == item.get("trajectory_id")] or [0]
                    ),
                    **rollout,
                    "strict_rollout_stable": bool(rollout.get("long_rollout_stable")),
                }
            )
    return rows, summary_rows


def summarize_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "clear": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "drift": sum(1 for row in rows if row.get("protocol_drift")),
        "stable": sum(1 for row in rows if row.get("strict_rollout_stable")),
        "period_seen": sum(1 for row in rows if row.get("period_seen")),
        "continued_after_period": sum(1 for row in rows if row.get("continued_after_period")),
        "ended_with_eos": sum(1 for row in rows if row.get("ended_with_eos")),
        "avg_steps": mean([row.get("steps_generated") for row in rows]),
        "avg_max_stop_chain": mean([row.get("max_stop_positive_chain_len") for row in rows]),
        "labels": dict(Counter(str(row.get("rollout_label")) for row in rows if row.get("rollout_label") is not None)),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(summarize_trajectory(items))
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def summarize_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "rollout_mode", "prompt_protocol", "step"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "stop_margin_mean": mean([item.get("stop_margin") for item in items]),
                "prose_margin_mean": mean([item.get("prose_margin") for item in items]),
                "echo_margin_mean": mean([item.get("echo_margin") for item in items]),
                "eos_rank_mean": mean([item.get("eos_rank") for item in items]),
                "period_rank_mean": mean([item.get("period_rank") for item in items]),
                "prose_rank_mean": mean([item.get("prose_rank") for item in items]),
                "emitted_period": sum(1 for item in items if item.get("emitted_is_period")),
                "emitted_eos": sum(1 for item in items if item.get("emitted_is_eos")),
                "continued_after_period_steps": sum(1 for item in items if item.get("continued_after_period_before_step")),
                "top_tokens": dict(Counter(str(item.get("emitted_token")) for item in items).most_common(12)),
            }
        )
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("rollout_mode")), str(row.get("prompt_protocol")), int(row.get("step"))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Global Trajectory Atlas and Stop-Execution Mechanism Mapping",
        "model": args.model,
        "prompt_protocols": parse_csv(args.prompt_protocols),
        "rollout_modes": parse_csv(args.rollout_modes),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase204_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    token_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        prompts = []
        selected_pairs = sorted(holdout_by_pair.keys())[: int(args.max_pairs)]
        for relation, language_pair in selected_pairs:
            samples = holdout_by_pair.get((relation, language_pair)) or []
            if int(args.max_samples_per_pair) > 0:
                samples = samples[: int(args.max_samples_per_pair)]
            for sample in samples:
                for protocol in parse_csv(args.prompt_protocols):
                    for rollout_mode in parse_csv(args.rollout_modes):
                        prompts.append(
                            {
                                **build_prompt_record(sample, protocol, rollout_mode, language_pair),
                                "model": args.model,
                            }
                        )
        token_rows, trajectory_rows = run_trajectories(
            model,
            tokenizer,
            device,
            prompts,
            groups,
            int(args.max_steps),
            int(args.batch_size),
        )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trajectory_summary_rows = summarize_by(trajectory_rows, ["model", "relation", "language_pair", "rollout_mode", "prompt_protocol"])
    token_step_summary_rows = summarize_token_rows(token_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "token_rows": len(token_rows),
        "trajectory_rows": len(trajectory_rows),
        "trajectory_summary_rows": trajectory_summary_rows,
        "token_step_summary_rows": token_step_summary_rows,
        "boundary": "Trajectory atlas only; no patch. Measures sequence-level stop execution failures after answer/period.",
    }
    write_json(out_dir / f"phase204_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase204_{args.model}_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase204_{args.model}_trajectory_rows.jsonl", trajectory_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "token_rows": len(token_rows),
                "trajectory_rows": len(trajectory_rows),
                "top_trajectory_summary_rows": trajectory_summary_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase204_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    trajectory_summary_rows = []
    token_step_summary_rows = []
    for summary in summaries:
        trajectory_summary_rows.extend(dict(row) for row in summary.get("trajectory_summary_rows") or [])
        token_step_summary_rows.extend(dict(row) for row in summary.get("token_step_summary_rows") or [])
    payload = {
        "schema_version": "phase204_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "trajectory_summary_rows": trajectory_summary_rows,
        "token_step_summary_rows": token_step_summary_rows,
    }
    write_json(out_dir / "phase204_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase204_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 204 global trajectory stop-execution atlas", ""]
    lines.append("## Trajectory Summary")
    lines.append("| model | relation | pair | mode | protocol | rows | stable | drift | period | continued after period | eos ended | avg stop chain |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("trajectory_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('relation')} | {row.get('language_pair')} | {row.get('rollout_mode')} | "
            f"{row.get('prompt_protocol')} | {row.get('rows')} | {row.get('stable')} | {row.get('drift')} | "
            f"{row.get('period_seen')} | {row.get('continued_after_period')} | {row.get('ended_with_eos')} | "
            f"{finite(row.get('avg_max_stop_chain')):.2f} |"
        )
    lines.append("")
    lines.append("## Token Step Summary")
    lines.append("| model | mode | protocol | step | stop margin | eos rank | period rank | period emitted | eos emitted | after-period continues | top tokens |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in (payload.get("token_step_summary_rows") or [])[:120]:
        lines.append(
            f"| {row.get('model')} | {row.get('rollout_mode')} | {row.get('prompt_protocol')} | {row.get('step')} | "
            f"{finite(row.get('stop_margin_mean')):.2f} | {finite(row.get('eos_rank_mean')):.1f} | "
            f"{finite(row.get('period_rank_mean')):.1f} | {row.get('emitted_period')} | {row.get('emitted_eos')} | "
            f"{row.get('continued_after_period_steps')} | {row.get('top_tokens')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="global_trajectory_stop_execution_atlas")
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
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-samples-per-pair", type=int, default=12)
    parser.add_argument("--prompt-protocols", default="plain,short_answer,stop_explicit")
    parser.add_argument("--rollout-modes", default="natural,post_answer")
    parser.add_argument("--max-steps", type=int, default=8)
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
