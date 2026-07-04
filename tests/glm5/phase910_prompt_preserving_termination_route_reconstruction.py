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
import phase908_l0_attention_eos_proximity_fine_audit as p908  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402


PHASE = 910
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase910_prompt_preserving_termination_route_reconstruction")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def specs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "control_label": "L0_attn_output_last_scale_0.75",
            "control_family": "prompt_intact_output_scale",
            "control_kind": "component_scale",
            "factor": 0.75,
            "prompt_input_intact": True,
            "prompt_all_zero_used": False,
        },
        {
            "control_label": "L0_attn_output_last_scale_0.50",
            "control_family": "prompt_intact_output_scale",
            "control_kind": "component_scale",
            "factor": 0.50,
            "prompt_input_intact": True,
            "prompt_all_zero_used": False,
        },
    ]
    for alpha in [0.05, 0.10, 0.25, 0.50, 1.00]:
        out.append(
            {
                "control_label": f"L0_promptzero_delta_alpha_{alpha:g}",
                "control_family": "prompt_intact_counterfactual_direction",
                "control_kind": "output_delta",
                "alpha": alpha,
                "prompt_input_intact": True,
                "prompt_all_zero_used": False,
                "direction_source": "prompt_all_zero_minus_baseline_l0_attention_output",
            }
        )
    span_specs = [
        ("L0_input_prompt_all_half", "prompt_all", 0.5),
        ("L0_input_prompt_first8_half", "prompt_first8", 0.5),
        ("L0_input_prompt_first8_zero", "prompt_first8", 0.0),
        ("L0_input_prompt_last8_half", "prompt_last8", 0.5),
        ("L0_input_prompt_last8_zero", "prompt_last8", 0.0),
        ("L0_input_answer_prefix_last_half", "answer_prefix_last", 0.5),
        ("L0_input_period_half", "period_token", 0.5),
        ("L0_input_period_zero", "period_token", 0.0),
    ]
    for label, span_kind, factor in span_specs:
        out.append(
            {
                "control_label": label,
                "control_family": "limited_span_adjustment",
                "control_kind": "input_span_scale",
                "span_kind": span_kind,
                "factor": factor,
                "prompt_input_intact": False,
                "prompt_all_zero_used": False,
            }
        )
    return out


def attn_output_tensor(output) -> torch.Tensor | None:
    if isinstance(output, tuple):
        if output and torch.is_tensor(output[0]):
            return output[0]
        return None
    return output if torch.is_tensor(output) else None


def replace_attn_output(output, patched: torch.Tensor):
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def logits_and_l0_vector(
    model,
    device: torch.device,
    current_ids: list[int],
    pre_handles: list[Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    module = p903.component_module(model, 0, "attention")
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = list(pre_handles or [])

    if module is not None:
        def capture(_module, _inputs, output):
            tensor = attn_output_tensor(output)
            if tensor is not None:
                if tensor.ndim >= 3:
                    captured["vector"] = tensor[:, -1, :].detach().float().cpu()[0]
                elif tensor.ndim >= 2:
                    captured["vector"] = tensor[-1, :].detach().float().cpu()
            return output

        handles.append(module.register_forward_hook(capture))
    try:
        logits = p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured.get("vector")


def install_l0_output_delta(model, delta: torch.Tensor, alpha: float) -> list[Any]:
    module = p903.component_module(model, 0, "attention")
    if module is None:
        return []

    def hook(_module, _inputs, output):
        tensor = attn_output_tensor(output)
        if tensor is None:
            return output
        patched = tensor.clone()
        local_delta = delta.to(device=patched.device, dtype=patched.dtype)
        if patched.ndim >= 3:
            patched[:, -1, :] += float(alpha) * local_delta
        elif patched.ndim >= 2:
            patched[-1, :] += float(alpha) * local_delta
        return replace_attn_output(output, patched)

    return [module.register_forward_hook(hook)]


def logits_with_spec(
    model,
    device: torch.device,
    current_ids: list[int],
    spec: dict[str, Any],
    prompt_len: int,
    prefix_len: int,
    delta: torch.Tensor | None,
) -> torch.Tensor | None:
    handles: list[Any] = []
    kind = spec.get("control_kind")
    if kind == "component_scale":
        handles = p908.install_component_scale(model, 0, "attention", float(spec["factor"]), True)
    elif kind == "output_delta":
        if delta is None:
            return None
        handles = install_l0_output_delta(model, delta, float(spec["alpha"]))
    elif kind == "input_span_scale":
        start, end = p909.span_bounds(str(spec["span_kind"]), int(prompt_len), int(prefix_len), len(current_ids))
        if start >= end:
            return None
        handles = p909.install_attention_input_span_scale(model, 0, start, end, float(spec["factor"]))
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def topk_tokens(tokenizer, logits: torch.Tensor, groups: dict[str, list[int]], k: int = 8) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits, k=min(int(k), int(logits.numel())))
    rows = []
    for rank, (value, token_id) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": p903.decode_token(tokenizer, int(token_id)),
                "category": p903.category_for_token(int(token_id), groups),
                "logit": float(value),
            }
        )
    return rows


def first_non_eos_top(top_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in top_rows:
        if row.get("category") != "eos":
            return row
    return None


def metric_delta(logits: torch.Tensor, baseline_logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    return float(logits[int(token_id)].item() - baseline_logits[int(token_id)].item())


def make_row(
    tokenizer,
    groups: dict[str, list[int]],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    baseline_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    patched_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    delta_norm: float | None,
    base_norm: float | None,
) -> dict[str, Any]:
    row = p908.make_row(
        tokenizer,
        source_row,
        {
            "intervention_kind": spec.get("control_kind"),
            "layer_idx": 0,
            "component_kind": "attention",
            **spec,
        },
        baseline_metrics,
        patched_metrics,
        patched_logits,
        baseline_logits,
    )
    patched_top = topk_tokens(tokenizer, patched_logits, groups, 8)
    baseline_top = topk_tokens(tokenizer, baseline_logits, groups, 8)
    blocker = first_non_eos_top(patched_top)
    eos_logit = patched_metrics.get("eos_best_logit")
    blocker_margin = None
    if blocker is not None and eos_logit is not None:
        blocker_margin = float(eos_logit) - float(blocker["logit"])
    blocker_delta = metric_delta(patched_logits, baseline_logits, blocker.get("token_id") if blocker else None)
    row.update(
        {
            "phase": PHASE,
            "row_kind": "phase910_prompt_preserving_route_reconstruction_row",
            "baseline_top8": baseline_top,
            "patched_top8": patched_top,
            "full_vocab_blocker_id": blocker.get("token_id") if blocker else None,
            "full_vocab_blocker_token": blocker.get("token") if blocker else None,
            "full_vocab_blocker_category": blocker.get("category") if blocker else None,
            "full_vocab_blocker_rank": blocker.get("rank") if blocker else None,
            "full_vocab_blocker_logit": blocker.get("logit") if blocker else None,
            "full_vocab_blocker_logit_delta": blocker_delta,
            "eos_margin_vs_full_vocab_blocker": blocker_margin,
            "prompt_input_intact": bool(spec.get("prompt_input_intact")),
            "prompt_all_zero_used": bool(spec.get("prompt_all_zero_used")),
            "prompt_destroying_control": bool(spec.get("prompt_all_zero_used")),
            "counterfactual_delta_norm": delta_norm,
            "baseline_l0_output_norm": base_norm,
            "patched_eos_top5": bool(patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 5),
            "prompt_preserving_eos_top50": bool(spec.get("prompt_input_intact") and patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 50),
            "prompt_preserving_eos_top10": bool(spec.get("prompt_input_intact") and patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 10),
            "prompt_preserving_eos_top1": bool(spec.get("prompt_input_intact") and patched_metrics.get("eos_rank") == 1),
            "strict_clean_candidate": bool(patched_metrics.get("eos_rank") == 1),
        }
    )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = p908.summarize_rows(rows)
    base.update(
        {
            "patched_eos_top5": sum(1 for row in rows if row.get("patched_eos_top5")),
            "prompt_input_intact_rows": sum(1 for row in rows if row.get("prompt_input_intact")),
            "prompt_preserving_eos_top1": sum(1 for row in rows if row.get("prompt_preserving_eos_top1")),
            "prompt_preserving_eos_top10": sum(1 for row in rows if row.get("prompt_preserving_eos_top10")),
            "prompt_preserving_eos_top50": sum(1 for row in rows if row.get("prompt_preserving_eos_top50")),
            "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
            "median_eos_margin_vs_full_vocab_blocker": median([row.get("eos_margin_vs_full_vocab_blocker") for row in rows]),
            "blocker_categories": dict(sorted(Counter(str(row.get("full_vocab_blocker_category")) for row in rows).items())),
        }
    )
    return base


def summarize_by_control(rows: list[dict[str, Any]], top_n: int = 60) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("control_label"))].append(row)
    summaries = []
    for label, vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        summary.update(
            {
                "control_label": label,
                "control_family": first.get("control_family"),
                "control_kind": first.get("control_kind"),
                "factor": first.get("factor"),
                "alpha": first.get("alpha"),
                "span_kind": first.get("span_kind"),
                "prompt_input_intact": first.get("prompt_input_intact"),
                "prompt_all_zero_used": first.get("prompt_all_zero_used"),
            }
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row.get("prompt_preserving_eos_top1") or 0,
            row.get("prompt_preserving_eos_top10") or 0,
            row.get("prompt_preserving_eos_top50") or 0,
            row.get("patched_eos_top1") or 0,
            row.get("patched_eos_top10") or 0,
            row.get("patched_eos_top50") or 0,
            row.get("eos_rank_improved_1000") or 0,
            row.get("median_eos_vs_next_margin_delta") or 0,
        ),
        reverse=True,
    )
    return summaries[:top_n]


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    control_summaries = summarize_by_control(rows)
    prompt_intact = [row for row in rows if row.get("prompt_input_intact")]
    prompt_intact_overall = summarize_rows(prompt_intact)
    if prompt_intact_overall["prompt_preserving_eos_top1"] > 0:
        evidence = "prompt_preserving_route_reaches_eos_top1"
    elif prompt_intact_overall["prompt_preserving_eos_top10"] > 0:
        evidence = "prompt_preserving_route_reaches_eos_top10"
    elif prompt_intact_overall["prompt_preserving_eos_top50"] > 0:
        evidence = "prompt_preserving_route_reaches_eos_top50"
    elif prompt_intact_overall["eos_rank_improved_1000"] > 0:
        evidence = "prompt_preserving_route_improves_eos_but_not_near"
    elif overall["patched_eos_top50"] > 0:
        evidence = "only_limited_span_route_reaches_eos_near"
    else:
        evidence = "no_prompt_preserving_eos_reconstruction"
    return {
        "phase": PHASE,
        "title": "Prompt-preserving Termination Route Reconstruction",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "intervention_count": len(specs()),
        "overall": overall,
        "prompt_intact_overall": prompt_intact_overall,
        "control_summaries": control_summaries,
        "evidence_label": evidence,
        "boundary": (
            "Phase910 excludes prompt_all_zero as a tested control. Prompt-all-zero is used only to derive a "
            "counterfactual direction; closure requires EOS top1 under prompt-preserving controls."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = p906.selected_phase899_rows(args.model, args)
    all_specs = specs()
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
            "specs": all_specs,
        }
        p846.write_json(out_dir / f"phase910_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase910_{args.model}_rows.jsonl", [])
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
            baseline_logits, base_vec = logits_and_l0_vector(model, device, period_ids)
            baseline_metrics = p903.state_metrics(tokenizer, baseline_logits, groups)
            prompt_zero_handles = p909.install_attention_input_span_scale(
                model,
                0,
                0,
                len(prompt_ids),
                0.0,
            )
            _prompt_zero_logits, prompt_zero_vec = logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
            delta = None
            delta_norm = None
            base_norm = None
            if base_vec is not None:
                base_norm = float(torch.linalg.vector_norm(base_vec).item())
            if base_vec is not None and prompt_zero_vec is not None:
                delta = prompt_zero_vec - base_vec
                delta_norm = float(torch.linalg.vector_norm(delta).item())
            for spec in all_specs:
                patched_logits = logits_with_spec(
                    model,
                    device,
                    period_ids,
                    spec,
                    len(prompt_ids),
                    len(prefix_ids),
                    delta,
                )
                if patched_logits is None:
                    continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                rows.append(
                    make_row(
                        tokenizer,
                        groups,
                        source_row,
                        spec,
                        baseline_metrics,
                        patched_metrics,
                        patched_logits,
                        baseline_logits,
                        delta_norm,
                        base_norm,
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

    payload = summarize_model(args.model, rows, len(selected_rows), attn_impl)
    p846.write_json(out_dir / f"phase910_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase910_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "prompt_intact": payload["prompt_intact_overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    prompt_intact_scalar = Counter()
    evidence = Counter()
    top_controls = []
    for model_name in MODELS:
        path = out_dir / f"phase910_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        for bucket_name, bucket, counter in [
            ("overall", summary.get("overall") or {}, scalar),
            ("prompt_intact_overall", summary.get("prompt_intact_overall") or {}, prompt_intact_scalar),
        ]:
            del bucket_name
            for key in [
                "rows",
                "eos_rank_improved",
                "eos_rank_improved_100",
                "eos_rank_improved_1000",
                "patched_eos_top1",
                "patched_eos_top10",
                "patched_eos_top50",
                "patched_eos_top5",
                "prompt_preserving_eos_top1",
                "prompt_preserving_eos_top10",
                "prompt_preserving_eos_top50",
                "strict_clean_candidate",
                "direct_eos_lift",
                "continuation_suppressed",
                "protocol_suppressed",
                "next_top_changed",
                "next_category_changed",
            ]:
                counter[key] += int(bucket.get(key) or 0)
        for row in summary.get("control_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_controls.append(item)
    top_controls.sort(
        key=lambda row: (
            row.get("prompt_preserving_eos_top1") or 0,
            row.get("prompt_preserving_eos_top10") or 0,
            row.get("prompt_preserving_eos_top50") or 0,
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
        "prompt_intact_scalar": {key: int(value) for key, value in sorted(prompt_intact_scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_controls": top_controls[:80],
    }
    p846.write_json(out_dir / "phase910_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase910_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 910 prompt-preserving termination route reconstruction",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Prompt-Intact Overall", ""])
    for key, value in (payload.get("prompt_intact_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append("| model | rows | prompt intact rows | eos top1 | eos top10 | eos top50 | prompt-intact top10 | prompt-intact top50 | evidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        intact = summary.get("prompt_intact_overall") or {}
        lines.append(
            "| {model} | {rows} | {irows} | {top1} | {top10} | {top50} | {itop10} | {itop50} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                irows=intact.get("rows"),
                top1=overall.get("patched_eos_top1"),
                top10=overall.get("patched_eos_top10"),
                top50=overall.get("patched_eos_top50"),
                itop10=intact.get("prompt_preserving_eos_top10"),
                itop50=intact.get("prompt_preserving_eos_top50"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append("| model | control | family | rows | intact | eos top10 | eos top50 | intact top10 | intact top50 | blocker median margin |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        lines.append(
            "| {model} | {control_label} | {control_family} | {rows} | {prompt_input_intact} | {patched_eos_top10} | {patched_eos_top50} | {prompt_preserving_eos_top10} | {prompt_preserving_eos_top50} | {median_eos_margin_vs_full_vocab_blocker} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="prompt_preserving_termination_route_reconstruction")
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
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"], "prompt_intact": payload["prompt_intact_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
