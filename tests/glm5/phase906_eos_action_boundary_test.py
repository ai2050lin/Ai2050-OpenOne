#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase900_protocol_stop_gate_discovery as p900  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase904_termination_control_candidate_search as p904  # noqa: E402


PHASE = 906
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase906_eos_action_boundary_test")
PHASE899_ROOT = Path("tests/result/phase899_domain_axis_rollout_protocol_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def selected_phase899_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE899_ROOT / args.phase899_round / f"phase899_{model_name}_rollout_rows.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("is_source_candidate") and row.get("rollout_answer_class") and row.get("protocol_drift")
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("eval_domain")),
            str(row.get("source_subset_key")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
        )
    )
    max_rows = int(args.max_rows_per_model)
    return rows[:max_rows] if max_rows > 0 else rows


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def ids_for_categories(groups: dict[str, list[int]], categories: list[str]) -> list[int]:
    ids: list[int] = []
    for category in categories:
        for token_id in groups.get(category) or []:
            token_id = int(token_id)
            if token_id not in ids:
                ids.append(token_id)
    return ids


def masked_metrics(tokenizer, logits: torch.Tensor, groups: dict[str, list[int]], mask_ids: list[int]) -> dict[str, Any]:
    masked = logits.clone()
    valid = [int(token_id) for token_id in mask_ids if 0 <= int(token_id) < int(masked.numel())]
    if valid:
        masked[torch.tensor(valid, dtype=torch.long, device=masked.device)] = -torch.inf
    metrics = p903.state_metrics(tokenizer, masked, groups)
    return {
        "next_top_id": metrics.get("next_top_id"),
        "next_top_token": metrics.get("next_top_token"),
        "next_top_category": metrics.get("next_top_category"),
        "eos_rank": metrics.get("eos_rank"),
        "eos_top1": bool(metrics.get("eos_rank") == 1),
        "period_rank": metrics.get("period_rank"),
        "protocol_rank": metrics.get("protocol_rank"),
    }


def category_ranks(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "eos_rank": metrics.get("eos_rank"),
        "period_rank": metrics.get("period_rank"),
        "newline_rank": metrics.get("newline_rank"),
        "comma_rank": metrics.get("comma_rank"),
        "field_word_rank": metrics.get("field_word_rank"),
        "explanation_rank": metrics.get("explanation_rank"),
        "list_word_rank": metrics.get("list_word_rank"),
        "protocol_rank": metrics.get("protocol_rank"),
        "stop_rank": metrics.get("stop_rank"),
    }


def category_logit(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(f"{key}_best_logit")
    return None if value is None else float(value)


def margin(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def forced_rollout(
    model,
    tokenizer,
    device: torch.device,
    current_ids: list[int],
    prefix_ids: list[int],
    forced_ids: list[int],
    max_after_tokens: int,
) -> tuple[str, str, list[int], list[int]]:
    current = [int(x) for x in current_ids] + [int(x) for x in forced_ids]
    generated_after: list[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(int(max_after_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        next_id = int(torch.argmax(logits).item())
        generated_after.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    suffix_ids = [int(x) for x in forced_ids] + generated_after
    combined_ids = [int(x) for x in prefix_ids] + suffix_ids
    combined_text = tokenizer.decode(combined_ids, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    return combined_text, suffix_text, suffix_ids, generated_after


def clean_flags(text: str, case: dict[str, Any]) -> dict[str, Any]:
    flags = p900.rollout_flags(text, case)
    strict_drift = p904.strict_protocol_drift(text)
    strict_clean = (
        bool(flags.get("rollout_answer_class"))
        and not bool(flags.get("rollout_object_echo"))
        and not bool(flags.get("protocol_drift"))
        and not strict_drift
    )
    return {
        **flags,
        "strict_protocol_drift": strict_drift,
        "strict_clean_answer_no_protocol": strict_clean,
    }


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    case: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    answer_seen: bool,
    groups: dict[str, list[int]],
    baseline_metrics: dict[str, Any],
    period_metrics: dict[str, Any],
    period_id: int | None,
    eos_id: int | None,
    period_combined_text: str,
    period_suffix_text: str,
    period_suffix_ids: list[int],
    period_generated_after: list[int],
    eos_forced_text: str | None,
) -> dict[str, Any]:
    eos_logit = category_logit(period_metrics, "eos")
    period_logit = category_logit(period_metrics, "period")
    protocol_logit = period_metrics.get("protocol_best_logit")
    newline_logit = category_logit(period_metrics, "newline")
    comma_logit = category_logit(period_metrics, "comma")
    period_flags = clean_flags(period_combined_text, case)
    eos_flags = clean_flags(eos_forced_text or prefix_text, case) if eos_id is not None else {}
    mask_specs = {
        "mask_newline": ids_for_categories(groups, ["newline"]),
        "mask_comma": ids_for_categories(groups, ["comma"]),
        "mask_field_explanation": ids_for_categories(groups, ["field_word", "explanation"]),
        "mask_protocol": ids_for_categories(groups, ["newline", "comma", "field_word", "explanation", "list_word"]),
        "mask_protocol_plus_period": ids_for_categories(
            groups, ["newline", "comma", "field_word", "explanation", "list_word", "period"]
        ),
    }
    # Build masked views directly from the already computed period logits.
    period_logits_for_mask = period_metrics["_raw_logits"]
    masked = {name: masked_metrics(tokenizer, period_logits_for_mask, groups, ids) for name, ids in mask_specs.items()}
    first_after = period_generated_after[0] if period_generated_after else None
    second_after = period_generated_after[1] if len(period_generated_after) > 1 else None
    return {
        "phase": PHASE,
        "row_kind": "phase906_eos_action_boundary_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "prefix_ids": prefix_ids,
        "prefix_text": prefix_text,
        "answer_prefix_seen": answer_seen,
        "eos_available": eos_id is not None,
        "eos_token_id": eos_id,
        "eos_token": p903.decode_token(tokenizer, eos_id),
        "period_force_id": period_id,
        "period_force_token": p903.decode_token(tokenizer, period_id),
        **{f"baseline_{key}": value for key, value in category_ranks(baseline_metrics).items()},
        "baseline_next_top_category": baseline_metrics.get("next_top_category"),
        "baseline_protocol_best_category": baseline_metrics.get("protocol_best_category"),
        "baseline_stop_best_category": baseline_metrics.get("stop_best_category"),
        **{f"after_period_{key}": value for key, value in category_ranks(period_metrics).items()},
        "after_period_next_top_category": period_metrics.get("next_top_category"),
        "after_period_next_top_token": period_metrics.get("next_top_token"),
        "after_period_protocol_best_category": period_metrics.get("protocol_best_category"),
        "after_period_stop_best_category": period_metrics.get("stop_best_category"),
        "after_period_eos_margin_vs_period": margin(eos_logit, period_logit),
        "after_period_eos_margin_vs_protocol": margin(eos_logit, protocol_logit),
        "after_period_eos_margin_vs_newline": margin(eos_logit, newline_logit),
        "after_period_eos_margin_vs_comma": margin(eos_logit, comma_logit),
        "period_forced_suffix_ids": period_suffix_ids,
        "period_forced_suffix_text": period_suffix_text,
        "period_forced_combined_text": period_combined_text,
        "period_after_first_id": first_after,
        "period_after_first_token": p903.decode_token(tokenizer, first_after),
        "period_after_first_category": p903.category_for_token(first_after, groups),
        "period_after_second_id": second_after,
        "period_after_second_token": p903.decode_token(tokenizer, second_after),
        "period_after_second_category": p903.category_for_token(second_after, groups),
        "period_after_generated_eos": bool(eos_id is not None and int(eos_id) in [int(x) for x in period_generated_after]),
        "period_forced_protocol_drift": bool(period_flags.get("protocol_drift")),
        "period_forced_strict_protocol_drift": bool(period_flags.get("strict_protocol_drift")),
        "period_forced_strict_clean_answer_no_protocol": bool(period_flags.get("strict_clean_answer_no_protocol")),
        "eos_forced_text": eos_forced_text,
        "eos_forced_generation_would_stop": eos_id is not None,
        "eos_forced_strict_clean_answer_no_protocol": bool(eos_flags.get("strict_clean_answer_no_protocol")),
        "masked_after_period": masked,
        "mask_newline_eos_top1": bool(masked["mask_newline"].get("eos_top1")),
        "mask_comma_eos_top1": bool(masked["mask_comma"].get("eos_top1")),
        "mask_field_explanation_eos_top1": bool(masked["mask_field_explanation"].get("eos_top1")),
        "mask_protocol_eos_top1": bool(masked["mask_protocol"].get("eos_top1")),
        "mask_protocol_plus_period_eos_top1": bool(masked["mask_protocol_plus_period"].get("eos_top1")),
    }


def strip_raw_logits(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_raw_logits"}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "answer_prefix_seen": sum(1 for row in rows if row.get("answer_prefix_seen")),
        "baseline_eos_top1": sum(1 for row in rows if row.get("baseline_eos_rank") == 1),
        "baseline_eos_top10": sum(1 for row in rows if row.get("baseline_eos_rank") is not None and int(row["baseline_eos_rank"]) <= 10),
        "baseline_eos_top50": sum(1 for row in rows if row.get("baseline_eos_rank") is not None and int(row["baseline_eos_rank"]) <= 50),
        "baseline_median_eos_rank": median([row.get("baseline_eos_rank") for row in rows]),
        "baseline_median_period_rank": median([row.get("baseline_period_rank") for row in rows]),
        "baseline_median_protocol_rank": median([row.get("baseline_protocol_rank") for row in rows]),
        "after_period_eos_top1": sum(1 for row in rows if row.get("after_period_eos_rank") == 1),
        "after_period_eos_top10": sum(1 for row in rows if row.get("after_period_eos_rank") is not None and int(row["after_period_eos_rank"]) <= 10),
        "after_period_eos_top50": sum(1 for row in rows if row.get("after_period_eos_rank") is not None and int(row["after_period_eos_rank"]) <= 50),
        "after_period_median_eos_rank": median([row.get("after_period_eos_rank") for row in rows]),
        "after_period_median_protocol_rank": median([row.get("after_period_protocol_rank") for row in rows]),
        "after_period_next_top_categories": dict(sorted(Counter(str(row.get("after_period_next_top_category")) for row in rows).items())),
        "period_after_first_categories": dict(sorted(Counter(str(row.get("period_after_first_category")) for row in rows).items())),
        "period_after_second_categories": dict(sorted(Counter(str(row.get("period_after_second_category")) for row in rows).items())),
        "period_after_generated_eos": sum(1 for row in rows if row.get("period_after_generated_eos")),
        "period_forced_protocol_drift": sum(1 for row in rows if row.get("period_forced_protocol_drift")),
        "period_forced_strict_protocol_drift": sum(1 for row in rows if row.get("period_forced_strict_protocol_drift")),
        "period_forced_strict_clean_answer_no_protocol": sum(
            1 for row in rows if row.get("period_forced_strict_clean_answer_no_protocol")
        ),
        "eos_forced_generation_would_stop": sum(1 for row in rows if row.get("eos_forced_generation_would_stop")),
        "eos_forced_strict_clean_answer_no_protocol": sum(1 for row in rows if row.get("eos_forced_strict_clean_answer_no_protocol")),
        "mask_newline_eos_top1": sum(1 for row in rows if row.get("mask_newline_eos_top1")),
        "mask_comma_eos_top1": sum(1 for row in rows if row.get("mask_comma_eos_top1")),
        "mask_field_explanation_eos_top1": sum(1 for row in rows if row.get("mask_field_explanation_eos_top1")),
        "mask_protocol_eos_top1": sum(1 for row in rows if row.get("mask_protocol_eos_top1")),
        "mask_protocol_plus_period_eos_top1": sum(1 for row in rows if row.get("mask_protocol_plus_period_eos_top1")),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    if overall["after_period_eos_top1"] > 0:
        evidence = "eos_natural_after_period_in_some_rows"
    elif overall["mask_protocol_plus_period_eos_top1"] > 0:
        evidence = "eos_available_only_after_protocol_and_period_mask"
    elif overall["eos_forced_strict_clean_answer_no_protocol"] > 0 and overall["after_period_eos_top1"] == 0:
        evidence = "eos_forced_clean_but_not_naturally_competitive"
    else:
        evidence = "eos_not_competitive_after_period"
    return {
        "phase": PHASE,
        "title": "EOS Action Boundary Test",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "overall": overall,
        "evidence_label": evidence,
        "boundary": (
            "Phase906 separates EOS action from period boundary. Forced EOS is a control, not a natural closure result."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = selected_phase899_rows(args.model, args)
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
        }
        p846.write_json(out_dir / f"phase906_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase906_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected_rows, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            prompt_ids = prompt_cache[prompt_key]
            gears = p903.parse_gears(str(source_row.get("source_subset_key")))
            _prefix_logits, prefix_ids, prefix_text, answer_seen = p901.logits_after_answer_prefix(
                model,
                tokenizer,
                device,
                prompt_ids,
                gears,
                str(source_row.get("edit_mode")),
                case,
                int(args.max_prefix_tokens),
                float(args.scale_up_factor),
            )
            current_ids = [int(x) for x in prompt_ids] + [int(x) for x in prefix_ids]
            baseline_logits = p903.logits_plain(model, device, current_ids)
            baseline_metrics = p903.state_metrics(tokenizer, baseline_logits, groups)
            period_id = baseline_metrics.get("period_best_id") or ((groups.get("period") or [None])[0])
            period_id = None if period_id is None else int(period_id)
            eos_id = tokenizer.eos_token_id
            if period_id is None:
                continue
            period_logits = p903.logits_plain(model, device, current_ids + [period_id])
            period_metrics = p903.state_metrics(tokenizer, period_logits, groups)
            period_metrics["_raw_logits"] = period_logits
            period_combined_text, period_suffix_text, period_suffix_ids, period_after_ids = forced_rollout(
                model,
                tokenizer,
                device,
                current_ids,
                prefix_ids,
                [period_id],
                int(args.max_after_period_tokens),
            )
            eos_forced_text = None
            if eos_id is not None:
                eos_forced_text = tokenizer.decode([int(x) for x in prefix_ids] + [int(eos_id)], skip_special_tokens=True)
            row = make_row(
                tokenizer,
                source_row,
                case,
                prefix_ids,
                prefix_text,
                answer_seen,
                groups,
                baseline_metrics,
                period_metrics,
                period_id,
                None if eos_id is None else int(eos_id),
                period_combined_text,
                period_suffix_text,
                period_suffix_ids,
                period_after_ids,
                eos_forced_text,
            )
            rows.append(strip_raw_logits(row))
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, attn_impl)
    p846.write_json(out_dir / f"phase906_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase906_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    for model_name in MODELS:
        path = out_dir / f"phase906_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "baseline_eos_top1",
            "baseline_eos_top10",
            "baseline_eos_top50",
            "after_period_eos_top1",
            "after_period_eos_top10",
            "after_period_eos_top50",
            "period_after_generated_eos",
            "period_forced_protocol_drift",
            "period_forced_strict_protocol_drift",
            "period_forced_strict_clean_answer_no_protocol",
            "eos_forced_generation_would_stop",
            "eos_forced_strict_clean_answer_no_protocol",
            "mask_protocol_eos_top1",
            "mask_protocol_plus_period_eos_top1",
        ]:
            scalar[key] += int(overall.get(key) or 0)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
    }
    p846.write_json(out_dir / "phase906_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase906_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 906 EOS action boundary test",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | rows | base eos top50 | after period eos top50 | after period eos top1 | period strict clean | eos forced clean | mask protocol+period eos top1 | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {base50} | {after50} | {after1} | {period_clean} | {eos_clean} | {mask} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                base50=overall.get("baseline_eos_top50"),
                after50=overall.get("after_period_eos_top50"),
                after1=overall.get("after_period_eos_top1"),
                period_clean=overall.get("period_forced_strict_clean_answer_no_protocol"),
                eos_clean=overall.get("eos_forced_strict_clean_answer_no_protocol"),
                mask=overall.get("mask_protocol_plus_period_eos_top1"),
                evidence=summary.get("evidence_label"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="eos_action_boundary_test")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--max-after-period-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
