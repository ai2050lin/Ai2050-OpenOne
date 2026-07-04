#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
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
from model_utils import get_layers  # noqa: E402


PHASE = 907
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase907_post_period_continuation_source_mapping")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def component_kinds() -> list[str]:
    return ["attention", "mlp"]


def logits_with_component_zero(
    model,
    device: torch.device,
    current_ids: list[int],
    layer_idx: int,
    component_kind: str,
) -> torch.Tensor | None:
    return p903.logits_with_component_zero(model, device, current_ids, layer_idx, component_kind)


def metric_delta(logits: torch.Tensor, baseline_logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    return float(logits[int(token_id)].item() - baseline_logits[int(token_id)].item())


def make_component_row(
    tokenizer,
    source_row: dict[str, Any],
    layer_idx: int,
    component_kind: str,
    baseline_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    patched_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> dict[str, Any]:
    base_eos_rank = baseline_metrics.get("eos_rank")
    patched_eos_rank = patched_metrics.get("eos_rank")
    eos_rank_delta = None
    if base_eos_rank is not None and patched_eos_rank is not None:
        eos_rank_delta = int(patched_eos_rank) - int(base_eos_rank)
    base_next_id = baseline_metrics.get("next_top_id")
    base_protocol_id = baseline_metrics.get("protocol_best_id")
    base_eos_id = baseline_metrics.get("eos_best_id")
    base_period_id = baseline_metrics.get("period_best_id")
    transition = f"{baseline_metrics.get('next_top_category')}->{patched_metrics.get('next_top_category')}"
    return {
        "phase": PHASE,
        "row_kind": "phase907_post_period_component_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "layer_idx": int(layer_idx),
        "component_kind": component_kind,
        "baseline_next_top_category": baseline_metrics.get("next_top_category"),
        "baseline_next_top_token": baseline_metrics.get("next_top_token"),
        "patched_next_top_category": patched_metrics.get("next_top_category"),
        "patched_next_top_token": patched_metrics.get("next_top_token"),
        "category_transition": transition,
        "baseline_eos_rank": base_eos_rank,
        "patched_eos_rank": patched_eos_rank,
        "eos_rank_delta": eos_rank_delta,
        "eos_rank_improved": bool(eos_rank_delta is not None and eos_rank_delta < 0),
        "eos_rank_improved_100": bool(eos_rank_delta is not None and eos_rank_delta <= -100),
        "eos_rank_improved_1000": bool(eos_rank_delta is not None and eos_rank_delta <= -1000),
        "patched_eos_top1": bool(patched_metrics.get("eos_rank") == 1),
        "patched_eos_top10": bool(patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 10),
        "patched_eos_top50": bool(patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 50),
        "baseline_protocol_rank": baseline_metrics.get("protocol_rank"),
        "patched_protocol_rank": patched_metrics.get("protocol_rank"),
        "baseline_protocol_best_category": baseline_metrics.get("protocol_best_category"),
        "patched_protocol_best_category": patched_metrics.get("protocol_best_category"),
        "protocol_rank1_removed": bool(baseline_metrics.get("protocol_rank") == 1 and patched_metrics.get("protocol_rank") != 1),
        "next_top_changed": bool(baseline_metrics.get("next_top_id") != patched_metrics.get("next_top_id")),
        "next_category_changed": bool(baseline_metrics.get("next_top_category") != patched_metrics.get("next_top_category")),
        "baseline_next_logit_delta": metric_delta(patched_logits, baseline_logits, base_next_id),
        "baseline_protocol_logit_delta": metric_delta(patched_logits, baseline_logits, base_protocol_id),
        "baseline_eos_logit_delta": metric_delta(patched_logits, baseline_logits, base_eos_id),
        "baseline_period_logit_delta": metric_delta(patched_logits, baseline_logits, base_period_id),
        "patched_eos_token": p903.decode_token(tokenizer, patched_metrics.get("eos_best_id")),
    }


def summarize_component_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "eos_rank_improved": sum(1 for row in rows if row.get("eos_rank_improved")),
        "eos_rank_improved_100": sum(1 for row in rows if row.get("eos_rank_improved_100")),
        "eos_rank_improved_1000": sum(1 for row in rows if row.get("eos_rank_improved_1000")),
        "patched_eos_top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "patched_eos_top10": sum(1 for row in rows if row.get("patched_eos_top10")),
        "patched_eos_top50": sum(1 for row in rows if row.get("patched_eos_top50")),
        "protocol_rank1_removed": sum(1 for row in rows if row.get("protocol_rank1_removed")),
        "next_top_changed": sum(1 for row in rows if row.get("next_top_changed")),
        "next_category_changed": sum(1 for row in rows if row.get("next_category_changed")),
        "median_eos_rank_delta": median([row.get("eos_rank_delta") for row in rows]),
        "category_transitions": dict(sorted(Counter(str(row.get("category_transition")) for row in rows).items())),
    }


def summarize_by_component(rows: list[dict[str, Any]], top_n: int = 20) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(int(row.get("layer_idx")), str(row.get("component_kind")), str(row.get("baseline_next_top_category")))].append(row)
    summaries = []
    for (layer_idx, component_kind, baseline_category), vals in buckets.items():
        summary = summarize_component_rows(vals)
        summary.update(
            {
                "layer_idx": layer_idx,
                "component_kind": component_kind,
                "baseline_next_top_category": baseline_category,
                "mean_eos_rank_delta": p846.mean([row.get("eos_rank_delta") for row in vals if row.get("eos_rank_delta") is not None]),
                "mean_baseline_next_logit_delta": p846.mean(
                    [row.get("baseline_next_logit_delta") for row in vals if row.get("baseline_next_logit_delta") is not None]
                ),
                "mean_eos_logit_delta": p846.mean(
                    [row.get("baseline_eos_logit_delta") for row in vals if row.get("baseline_eos_logit_delta") is not None]
                ),
            }
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row.get("patched_eos_top50") or 0,
            row.get("eos_rank_improved_1000") or 0,
            row.get("eos_rank_improved_100") or 0,
            row.get("next_category_changed") or 0,
            -(row.get("mean_eos_rank_delta") or 0),
        ),
        reverse=True,
    )
    return summaries[:top_n]


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_component_rows(rows)
    top_components = summarize_by_component(rows)
    if overall["patched_eos_top1"] > 0:
        evidence = "post_period_component_can_make_eos_top1"
    elif overall["patched_eos_top50"] > 0:
        evidence = "post_period_component_can_make_eos_near"
    elif overall["eos_rank_improved_1000"] > 0:
        evidence = "post_period_component_improves_eos_but_not_near"
    elif overall["next_category_changed"] > 0:
        evidence = "post_period_component_changes_continuation_route_without_eos"
    else:
        evidence = "no_post_period_source_signal"
    return {
        "phase": PHASE,
        "title": "Post-Period Continuation Source Mapping",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "overall": overall,
        "top_components": top_components,
        "evidence_label": evidence,
        "boundary": "Phase907 scans post-period component sources. It is source mapping, not closure.",
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = p906.selected_phase899_rows(args.model, args)
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
        }
        p846.write_json(out_dir / f"phase907_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase907_{args.model}_rows.jsonl", [])
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
        layers = get_layers(model)
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
            for layer_idx in range(0, len(layers), max(1, int(args.layer_stride))):
                for component_kind in component_kinds():
                    patched_logits = logits_with_component_zero(model, device, period_ids, layer_idx, component_kind)
                    if patched_logits is None:
                        continue
                    patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                    rows.append(
                        make_component_row(
                            tokenizer,
                            source_row,
                            layer_idx,
                            component_kind,
                            baseline_metrics,
                            patched_metrics,
                            patched_logits,
                            baseline_logits,
                        )
                    )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} component_rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, len(selected_rows), attn_impl)
    p846.write_json(out_dir / f"phase907_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase907_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_components = []
    for model_name in MODELS:
        path = out_dir / f"phase907_{model_name}_summary.json"
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
            "protocol_rank1_removed",
            "next_top_changed",
            "next_category_changed",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        for row in summary.get("top_components") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_components.append(item)
    top_components.sort(
        key=lambda row: (
            row.get("patched_eos_top50") or 0,
            row.get("eos_rank_improved_1000") or 0,
            row.get("eos_rank_improved_100") or 0,
            row.get("next_category_changed") or 0,
            -(row.get("mean_eos_rank_delta") or 0),
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
        "top_components": top_components[:30],
    }
    p846.write_json(out_dir / "phase907_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase907_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 907 post-period continuation source mapping",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append("| model | rows | eos improved | eos improved 1000 | eos top50 | next category changed | evidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {improved} | {improved1000} | {top50} | {changed} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                improved=overall.get("eos_rank_improved"),
                improved1000=overall.get("eos_rank_improved_1000"),
                top50=overall.get("patched_eos_top50"),
                changed=overall.get("next_category_changed"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Components", ""])
    lines.append("| model | layer | kind | base cat | rows | eos top50 | eos improved 1000 | next cat changed | mean eos rank delta |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_components") or []:
        lines.append(
            "| {model} | {layer_idx} | {component_kind} | {baseline_next_top_category} | {rows} | {patched_eos_top50} | {eos_rank_improved_1000} | {next_category_changed} | {mean_eos_rank_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="post_period_continuation_source_mapping")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--layer-stride", type=int, default=1)
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
