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

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase906_eos_action_boundary_test as p906  # noqa: E402
import phase908_l0_attention_eos_proximity_fine_audit as p908  # noqa: E402


PHASE = 909
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase909_l0_attention_source_span_eos_boundary_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def span_specs() -> list[dict[str, Any]]:
    return [
        {"control_label": "L0_attn_input_prompt_all_half", "span_kind": "prompt_all", "factor": 0.5},
        {"control_label": "L0_attn_input_prompt_all_zero", "span_kind": "prompt_all", "factor": 0.0},
        {"control_label": "L0_attn_input_prompt_first8_zero", "span_kind": "prompt_first8", "factor": 0.0},
        {"control_label": "L0_attn_input_prompt_last8_zero", "span_kind": "prompt_last8", "factor": 0.0},
        {"control_label": "L0_attn_input_answer_prefix_all_zero", "span_kind": "answer_prefix_all", "factor": 0.0},
        {"control_label": "L0_attn_input_answer_prefix_last_zero", "span_kind": "answer_prefix_last", "factor": 0.0},
        {"control_label": "L0_attn_input_last8_before_period_zero", "span_kind": "last8_before_period", "factor": 0.0},
        {"control_label": "L0_attn_input_period_half", "span_kind": "period_token", "factor": 0.5},
        {"control_label": "L0_attn_input_period_zero", "span_kind": "period_token", "factor": 0.0},
    ]


def span_bounds(span_kind: str, prompt_len: int, prefix_len: int, seq_len: int) -> tuple[int, int]:
    prefix_start = int(prompt_len)
    prefix_end = int(prompt_len) + int(prefix_len)
    period_idx = int(seq_len) - 1
    if span_kind == "prompt_all":
        return 0, int(prompt_len)
    if span_kind == "prompt_first8":
        return 0, min(8, int(prompt_len))
    if span_kind == "prompt_last8":
        return max(0, int(prompt_len) - 8), int(prompt_len)
    if span_kind == "answer_prefix_all":
        return prefix_start, prefix_end
    if span_kind == "answer_prefix_last":
        return max(prefix_start, prefix_end - 1), prefix_end
    if span_kind == "last8_before_period":
        return max(0, int(seq_len) - 9), period_idx
    if span_kind == "period_token":
        return period_idx, int(seq_len)
    return 0, 0


def install_attention_input_span_scale(
    model,
    layer_idx: int,
    span_start: int,
    span_end: int,
    factor: float,
) -> list[Any]:
    module = p903.component_module(model, int(layer_idx), "attention")
    if module is None:
        return []

    def scale_hidden(hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not torch.is_tensor(hidden_states):
            return None
        patched = hidden_states.clone()
        seq_len = int(patched.shape[-2]) if patched.ndim >= 3 else int(patched.shape[0])
        start = max(0, min(int(span_start), seq_len))
        end = max(start, min(int(span_end), seq_len))
        if start == end:
            return None
        if patched.ndim >= 3:
            patched[:, start:end, :] *= float(factor)
        elif patched.ndim >= 2:
            patched[start:end, :] *= float(factor)
        return patched

    def hook_with_kwargs(_module, inputs, kwargs):
        if kwargs and torch.is_tensor(kwargs.get("hidden_states")):
            patched = scale_hidden(kwargs["hidden_states"])
            if patched is None:
                return None
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = patched
            return inputs, new_kwargs
        if inputs and torch.is_tensor(inputs[0]):
            patched = scale_hidden(inputs[0])
            if patched is None:
                return None
            return (patched, *inputs[1:]), kwargs
        return None

    def hook_positional(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            patched = scale_hidden(inputs[0])
            if patched is None:
                return None
            return (patched, *inputs[1:])
        return None

    try:
        return [module.register_forward_pre_hook(hook_with_kwargs, with_kwargs=True)]
    except TypeError:
        return [module.register_forward_pre_hook(hook_positional)]


def logits_with_span_intervention(
    model,
    device: torch.device,
    current_ids: list[int],
    spec: dict[str, Any],
    span_start: int,
    span_end: int,
) -> torch.Tensor | None:
    handles = install_attention_input_span_scale(
        model,
        0,
        int(span_start),
        int(span_end),
        float(spec["factor"]),
    )
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    spec: dict[str, Any],
    span_start: int,
    span_end: int,
    baseline_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    patched_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> dict[str, Any]:
    row = p908.make_row(
        tokenizer,
        source_row,
        {
            "intervention_kind": "l0_attention_input_span_scale",
            "control_family": "source_span",
            "component_kind": "attention",
            "layer_idx": 0,
            "last_token_only": False,
            **spec,
        },
        baseline_metrics,
        patched_metrics,
        patched_logits,
        baseline_logits,
    )
    row.update(
        {
            "phase": PHASE,
            "row_kind": "phase909_l0_attention_source_span_audit_row",
            "span_start": int(span_start),
            "span_end": int(span_end),
            "span_len": int(span_end) - int(span_start),
        }
    )
    return row


def summarize_by_control(rows: list[dict[str, Any]], top_n: int = 40) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("control_label"))].append(row)
    summaries = []
    for label, vals in buckets.items():
        summary = p908.summarize_rows(vals)
        first = vals[0]
        summary.update(
            {
                "control_label": label,
                "span_kind": first.get("span_kind"),
                "factor": first.get("factor"),
                "control_family": first.get("control_family"),
                "median_span_len": p908.median([row.get("span_len") for row in vals]),
            }
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row.get("patched_eos_top1") or 0,
            row.get("patched_eos_top10") or 0,
            row.get("patched_eos_top50") or 0,
            row.get("eos_rank_improved_1000") or 0,
            row.get("median_eos_vs_next_margin_delta") or 0,
        ),
        reverse=True,
    )
    return summaries[:top_n]


def summarize_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected_count: int,
    specs: list[dict[str, Any]],
    attn_impl: str | None,
) -> dict[str, Any]:
    overall = p908.summarize_rows(rows)
    if overall["patched_eos_top1"] > 0:
        evidence = "source_span_audit_reaches_eos_top1"
    elif overall["patched_eos_top10"] > 0:
        evidence = "source_span_audit_reaches_eos_top10"
    elif overall["patched_eos_top50"] > 0:
        evidence = "source_span_audit_reaches_eos_top50"
    elif overall["eos_rank_improved_1000"] > 0:
        evidence = "source_span_audit_improves_eos_but_not_near"
    else:
        evidence = "no_source_span_eos_boundary_signal"
    return {
        "phase": PHASE,
        "title": "L0 Attention Source-Span EOS Boundary Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "intervention_count": len(specs),
        "overall": overall,
        "control_summaries": summarize_by_control(rows),
        "evidence_label": evidence,
        "boundary": (
            "Phase909 audits source-span dependence of the L0 attention EOS proximity signal. "
            "Span ablation is a boundary locator, not a clean natural closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = p906.selected_phase899_rows(args.model, args)
    specs = span_specs()
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
            "specs": specs,
        }
        p846.write_json(out_dir / f"phase909_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase909_{args.model}_rows.jsonl", [])
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
            _prefix_logits, prefix_ids, _prefix_text, _answer_seen = p901.logits_after_answer_prefix(
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
            answer_logits = p903.logits_plain(model, device, current_ids)
            answer_metrics = p903.state_metrics(tokenizer, answer_logits, groups)
            period_id = answer_metrics.get("period_best_id") or ((groups.get("period") or [None])[0])
            if period_id is None:
                continue
            period_ids = current_ids + [int(period_id)]
            baseline_logits = p903.logits_plain(model, device, period_ids)
            baseline_metrics = p903.state_metrics(tokenizer, baseline_logits, groups)
            for spec in specs:
                start, end = span_bounds(
                    str(spec["span_kind"]),
                    len(prompt_ids),
                    len(prefix_ids),
                    len(period_ids),
                )
                if start >= end:
                    continue
                patched_logits = logits_with_span_intervention(model, device, period_ids, spec, start, end)
                if patched_logits is None:
                    continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                rows.append(
                    make_row(
                        tokenizer,
                        source_row,
                        spec,
                        start,
                        end,
                        baseline_metrics,
                        patched_metrics,
                        patched_logits,
                        baseline_logits,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, len(selected_rows), specs, attn_impl)
    p846.write_json(out_dir / f"phase909_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase909_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_controls = []
    for model_name in MODELS:
        path = out_dir / f"phase909_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "eos_rank_improved",
            "eos_rank_improved_100",
            "eos_rank_improved_1000",
            "patched_eos_top1",
            "patched_eos_top10",
            "patched_eos_top50",
            "direct_eos_lift",
            "continuation_suppressed",
            "protocol_suppressed",
            "next_top_changed",
            "next_category_changed",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        for row in summary.get("control_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_controls.append(item)
    top_controls.sort(
        key=lambda row: (
            row.get("patched_eos_top1") or 0,
            row.get("patched_eos_top10") or 0,
            row.get("patched_eos_top50") or 0,
            row.get("eos_rank_improved_1000") or 0,
            row.get("median_eos_vs_next_margin_delta") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_controls": top_controls[:50],
    }
    p846.write_json(out_dir / "phase909_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase909_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 909 L0 attention source-span EOS boundary audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append("| model | rows | eos top1 | eos top10 | eos top50 | direct eos lift | cont suppressed | evidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {top1} | {top10} | {top50} | {lift} | {supp} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                top1=overall.get("patched_eos_top1"),
                top10=overall.get("patched_eos_top10"),
                top50=overall.get("patched_eos_top50"),
                lift=overall.get("direct_eos_lift"),
                supp=overall.get("continuation_suppressed"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append("| model | control | span | factor | rows | eos top10 | eos top50 | lift | suppress | median margin delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        lines.append(
            "| {model} | {control_label} | {span_kind} | {factor} | {rows} | {patched_eos_top10} | {patched_eos_top50} | {direct_eos_lift} | {continuation_suppressed} | {median_eos_vs_next_margin_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="l0_attention_source_span_eos_boundary_audit")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
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
