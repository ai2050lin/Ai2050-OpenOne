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


PHASE = 209
SOURCE_PHASE = 208
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase209_pattern_running_contrast_atlas")


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


def relation_question(sample: dict[str, Any]) -> str:
    relation = str(sample.get("relation") or "")
    obj = str(sample.get("object") or "")
    if relation == "color":
        return f"What color is {obj}?"
    if relation == "category":
        return f"What category does {obj} belong to?"
    if relation == "function":
        return f"What is {obj} used for?"
    return f"What is the {relation} of {obj}?"


def pattern_prompt(sample: dict[str, Any], pattern: str) -> str:
    question = relation_question(sample)
    target = str(sample.get("target_label") or "").strip()
    if pattern == "answer_short":
        return p200.protocol_prompt(sample, "short_answer")
    if pattern == "answer_stop":
        return p200.protocol_prompt(sample, "stop_explicit")
    if pattern == "answer_explain":
        return f"{question}\nAnswer with the answer first, then one short reason using because.\nAnswer:"
    if pattern == "answer_repeat":
        return f"{question}\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer:"
    if pattern == "answer_list":
        return f"{question}\nGive three plausible short answers separated by commas.\nAnswer:"
    if pattern == "answer_echo_control":
        return f"Repeat the object name once, then answer the question.\nObject: {sample.get('object')}\nQuestion: {question}\nAnswer:"
    if pattern == "answer_target_seeded":
        return f"{question}\nThe answer is likely {target}. Reply with the final answer only.\nAnswer:"
    raise ValueError(f"unsupported pattern: {pattern}")


def expected_output_pattern(pattern: str) -> str:
    mapping = {
        "answer_short": "short_answer",
        "answer_stop": "short_answer",
        "answer_explain": "explain_answer",
        "answer_repeat": "repeat_answer",
        "answer_list": "list_answer",
        "answer_echo_control": "echo_then_answer",
        "answer_target_seeded": "short_answer",
    }
    return mapping[pattern]


def count_answer_mentions(generated: str, target: str) -> int:
    if not target:
        return 0
    lower = generated.lower()
    target_lower = target.lower()
    count = lower.count(target_lower)
    if count:
        return count
    parts = [part for part in re.split(r"[\s,/;:.!?，。；：！？]+", target_lower) if part]
    return sum(1 for part in parts if part and part in lower)


def classify_pattern(generated: str, sample: dict[str, Any], emitted_ids: list[int], eos_ids: set[int]) -> dict[str, Any]:
    target = str(sample.get("target_label") or "").strip()
    obj = str(sample.get("object") or "").strip()
    text = str(generated or "")
    lower = text.lower()
    answer_mentions = count_answer_mentions(text, target)
    object_mentions = count_answer_mentions(text, obj)
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", text)
    comma_count = text.count(",") + text.count("，") + text.count(";") + text.count("；")
    because_like = bool(re.search(r"\b(because|since|reason|therefore|so)\b|因为|所以|原因", lower))
    next_task_like = bool(re.search(r"\b(question|answer|what is|next)\b|问题|答案|下一个", lower))
    ended_with_eos = bool(emitted_ids and emitted_ids[-1] in eos_ids)
    if answer_mentions >= 2:
        output_pattern = "repeat_answer"
    elif comma_count >= 2:
        output_pattern = "list_answer"
    elif because_like or len(words) >= 12:
        output_pattern = "explain_answer"
    elif object_mentions >= 1 and answer_mentions >= 1:
        output_pattern = "echo_then_answer"
    elif answer_mentions >= 1 and len(words) <= 6:
        output_pattern = "short_answer"
    elif next_task_like:
        output_pattern = "next_task_or_format"
    else:
        output_pattern = "other_or_wrong"
    return {
        "generated_word_count": len(words),
        "answer_mentions": answer_mentions,
        "object_mentions": object_mentions,
        "answer_present": answer_mentions > 0,
        "comma_count": comma_count,
        "because_like": because_like,
        "next_task_like": next_task_like,
        "ended_with_eos": ended_with_eos,
        "output_pattern": output_pattern,
    }


def build_prompt_records(
    args: argparse.Namespace,
    holdout_by_pair: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
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
                        "row_kind": "phase209_prompt_record",
                        "model": args.model,
                        "source_sample_id": sample.get("sample_id"),
                        "trajectory_id": f"{sample.get('sample_id')}|{pattern}",
                        "language_pair": language_pair,
                        "pattern_id": pattern,
                        "pattern_trigger": relation,
                        "pattern_state_variables": {
                            "object": sample.get("object"),
                            "relation": relation,
                            "target_label": sample.get("target_label"),
                            "language_pair": language_pair,
                        },
                        "expected_output_pattern": expected_output_pattern(pattern),
                        "prompt": pattern_prompt(sample, pattern),
                    }
                )
    return records


def run_pattern_trajectories(
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
                        "row_kind": "phase209_token_row",
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
                    active[idx] = False
            del logits_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for idx, item in enumerate(batch):
            generated = generated_parts[idx]
            classification = classify_pattern(generated, item, emitted_ids[idx], eos_ids)
            expected = str(item.get("expected_output_pattern") or "")
            trajectory_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase209_pattern_trajectory_row",
                    "model": item.get("model"),
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "relation": item.get("relation"),
                    "language_pair": item.get("language_pair"),
                    "pattern_id": item.get("pattern_id"),
                    "pattern_trigger": item.get("pattern_trigger"),
                    "pattern_state_variables": item.get("pattern_state_variables"),
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
                    "continued_after_period": continued_after_period[idx],
                    "pattern_match": classification.get("output_pattern") == expected,
                    "pattern_drift": classification.get("output_pattern") != expected,
                    "failure_mode": "match" if classification.get("output_pattern") == expected else classification.get("output_pattern"),
                    **classification,
                }
            )
    return token_rows, trajectory_rows


def summarize_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "pattern_drift": sum(1 for item in items if item.get("pattern_drift")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "ended_with_eos": sum(1 for item in items if item.get("ended_with_eos")),
                "period_seen": sum(1 for item in items if item.get("period_seen")),
                "continued_after_period": sum(1 for item in items if item.get("continued_after_period")),
                "avg_steps": mean([item.get("steps_generated") for item in items]),
                "avg_words": mean([item.get("generated_word_count") for item in items]),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
                "top_generated_prefixes": dict(Counter(str(item.get("generated"))[:60] for item in items).most_common(8)),
            }
        )
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def summarize_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "pattern_id", "step"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "target_rank_mean": mean([item.get("target_rank") for item in items]),
                "stop_margin_mean": mean([item.get("stop_margin") for item in items]),
                "prose_margin_mean": mean([item.get("prose_margin") for item in items]),
                "echo_margin_mean": mean([item.get("echo_margin") for item in items]),
                "eos_rank_mean": mean([item.get("eos_rank") for item in items]),
                "period_rank_mean": mean([item.get("period_rank") for item in items]),
                "top_tokens": dict(Counter(str(item.get("emitted_token")) for item in items).most_common(12)),
                "emitted_eos": sum(1 for item in items if item.get("emitted_is_eos")),
                "emitted_period": sum(1 for item in items if item.get("emitted_is_period")),
                "continued_after_period_steps": sum(1 for item in items if item.get("continued_after_period_before_step")),
            }
        )
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), int(row.get("step"))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Pattern Running Contrast Atlas",
        "model": args.model,
        "patterns": parse_csv(args.patterns),
        "relations": parse_csv(args.relations),
        "language_pairs": parse_csv(args.language_pairs),
        "max_pairs": int(args.max_pairs),
        "max_samples_per_pair": int(args.max_samples_per_pair),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase209_{args.model}_summary.json", payload)
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
        prompts = build_prompt_records(args, holdout_by_pair)
        token_rows, trajectory_rows = run_pattern_trajectories(
            model, tokenizer, device, prompts, groups, int(args.max_steps), int(args.batch_size)
        )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_pattern = summarize_rows(trajectory_rows, ["model", "pattern_id"])
    by_relation_pattern = summarize_rows(trajectory_rows, ["model", "relation", "pattern_id"])
    token_summary_rows = summarize_token_rows(token_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "prompt_count": len(trajectory_rows),
        "token_rows": len(token_rows),
        "pattern_match_total": sum(1 for row in trajectory_rows if row.get("pattern_match")),
        "pattern_drift_total": sum(1 for row in trajectory_rows if row.get("pattern_drift")),
        "answer_present_total": sum(1 for row in trajectory_rows if row.get("answer_present")),
        "ended_with_eos_total": sum(1 for row in trajectory_rows if row.get("ended_with_eos")),
        "by_pattern": by_pattern,
        "by_relation_pattern": by_relation_pattern,
        "token_summary_rows": token_summary_rows,
        "definition": {
            "Pattern": "(Trigger, StateVariables, FeatureTrajectory, PriorityProxy, OutputConstraint, FailureModes)",
            "Trigger": "relation plus prompt pattern id",
            "StateVariables": "object, relation, target label, language pair",
            "FeatureTrajectoryProxy": "greedy token trajectory plus target/stop/prose/echo/EOS ranks at each step",
            "PriorityProxy": "which output pattern wins under conflicting prompt constraints",
            "OutputConstraint": "short/explain/repeat/list/echo target pattern",
            "FailureModes": "observed output pattern when it differs from expected pattern",
        },
    }
    write_json(out_dir / f"phase209_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase209_{args.model}_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase209_{args.model}_trajectory_rows.jsonl", trajectory_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "prompt_count": len(trajectory_rows),
                "pattern_match_total": payload["pattern_match_total"],
                "pattern_drift_total": payload["pattern_drift_total"],
                "answer_present_total": payload["answer_present_total"],
                "ended_with_eos_total": payload["ended_with_eos_total"],
                "by_pattern": by_pattern,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase209_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    by_pattern = []
    by_relation_pattern = []
    token_summary_rows = []
    for summary in summaries:
        by_pattern.extend(dict(row) for row in summary.get("by_pattern") or [])
        by_relation_pattern.extend(dict(row) for row in summary.get("by_relation_pattern") or [])
        token_summary_rows.extend(dict(row) for row in summary.get("token_summary_rows") or [])
    payload = {
        "schema_version": "phase209_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "prompt_count_total": sum(int(summary.get("prompt_count") or 0) for summary in summaries),
        "pattern_match_total": sum(int(summary.get("pattern_match_total") or 0) for summary in summaries),
        "pattern_drift_total": sum(int(summary.get("pattern_drift_total") or 0) for summary in summaries),
        "answer_present_total": sum(int(summary.get("answer_present_total") or 0) for summary in summaries),
        "ended_with_eos_total": sum(int(summary.get("ended_with_eos_total") or 0) for summary in summaries),
        "model_summaries": summaries,
        "by_pattern": by_pattern,
        "by_relation_pattern": by_relation_pattern,
        "token_summary_rows": token_summary_rows,
    }
    write_json(out_dir / "phase209_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase209_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 209 pattern running contrast atlas", ""]
    lines.append(f"Total prompts: {payload.get('prompt_count_total')}")
    lines.append(f"Pattern match: {payload.get('pattern_match_total')}")
    lines.append(f"Pattern drift: {payload.get('pattern_drift_total')}")
    lines.append(f"Answer present: {payload.get('answer_present_total')}")
    lines.append(f"Ended with EOS: {payload.get('ended_with_eos_total')}")
    lines.append("")
    lines.append("| model | pattern | rows | match | drift | answer | eos | output patterns |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("by_pattern") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('pattern_id')} | {row.get('rows')} | {row.get('pattern_match')} | "
            f"{row.get('pattern_drift')} | {row.get('answer_present')} | {row.get('ended_with_eos')} | {row.get('output_patterns')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="pattern_running_contrast_atlas")
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
    parser.add_argument("--max-samples-per-pair", type=int, default=10)
    parser.add_argument("--patterns", default="answer_short,answer_stop,answer_explain,answer_repeat,answer_list,answer_echo_control,answer_target_seeded")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
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
