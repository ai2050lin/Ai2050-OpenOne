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


PHASE = 908
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase908_l0_attention_eos_proximity_fine_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def tensor_scale_output(output, factor: float, last_token_only: bool):
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            return output
        patched = output[0].clone()
        scale_tensor(patched, factor, last_token_only)
        return (patched, *output[1:])
    if torch.is_tensor(output):
        patched = output.clone()
        scale_tensor(patched, factor, last_token_only)
        return patched
    return output


def scale_tensor(tensor: torch.Tensor, factor: float, last_token_only: bool) -> None:
    if tensor.ndim >= 3:
        if last_token_only:
            tensor[:, -1, :] *= float(factor)
        else:
            tensor[:, :, :] *= float(factor)
    elif tensor.ndim >= 2:
        tensor[:, :] *= float(factor)


def install_component_scale(
    model,
    layer_idx: int,
    component_kind: str,
    factor: float,
    last_token_only: bool,
) -> list[Any]:
    module = p903.component_module(model, int(layer_idx), component_kind)
    if module is None:
        return []
    return [
        module.register_forward_hook(
            lambda _module, _inputs, output: tensor_scale_output(output, float(factor), bool(last_token_only))
        )
    ]


def find_output_projection(model, layer_idx: int):
    layers = get_layers(model)
    if not (0 <= int(layer_idx) < len(layers)):
        return None
    attn = getattr(layers[int(layer_idx)], "self_attn", None)
    if attn is None:
        return None
    for name in ["o_proj", "dense", "out_proj"]:
        module = getattr(attn, name, None)
        if module is not None:
            return module
    return None


def config_num_heads(model) -> int | None:
    for key in ["num_attention_heads", "n_head", "num_heads"]:
        value = getattr(model.config, key, None)
        if value is not None:
            return int(value)
    return None


def install_o_proj_head_scale(model, layer_idx: int, head_idx: int, factor: float) -> list[Any]:
    proj = find_output_projection(model, layer_idx)
    if proj is None:
        return []
    num_heads = config_num_heads(model)
    in_features = getattr(proj, "in_features", None)
    if num_heads is None or in_features is None:
        return []
    if int(in_features) % int(num_heads) != 0:
        return []
    head_dim = int(in_features) // int(num_heads)
    start = int(head_idx) * head_dim
    end = start + head_dim
    if start < 0 or end > int(in_features):
        return []

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        patched = inputs[0].clone()
        if patched.ndim >= 3:
            patched[:, -1, start:end] *= float(factor)
        elif patched.ndim >= 2:
            patched[:, start:end] *= float(factor)
        return (patched, *inputs[1:])

    return [proj.register_forward_pre_hook(hook)]


def logits_with_intervention(model, device: torch.device, current_ids: list[int], spec: dict[str, Any]) -> torch.Tensor | None:
    handles: list[Any] = []
    kind = spec.get("intervention_kind")
    if kind == "component_scale":
        handles = install_component_scale(
            model,
            int(spec["layer_idx"]),
            str(spec["component_kind"]),
            float(spec["factor"]),
            bool(spec.get("last_token_only", True)),
        )
    elif kind == "head_scale":
        handles = install_o_proj_head_scale(
            model,
            int(spec["layer_idx"]),
            int(spec["head_idx"]),
            float(spec["factor"]),
        )
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def intervention_specs(model, args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "intervention_kind": "component_scale",
            "control_label": "L0_attention_last_half",
            "layer_idx": 0,
            "component_kind": "attention",
            "factor": 0.5,
            "last_token_only": True,
            "control_family": "l0_attention_intensity",
        },
        {
            "intervention_kind": "component_scale",
            "control_label": "L0_attention_last_zero",
            "layer_idx": 0,
            "component_kind": "attention",
            "factor": 0.0,
            "last_token_only": True,
            "control_family": "l0_attention_intensity",
        },
        {
            "intervention_kind": "component_scale",
            "control_label": "L0_attention_last_negative_half",
            "layer_idx": 0,
            "component_kind": "attention",
            "factor": -0.5,
            "last_token_only": True,
            "control_family": "l0_attention_intensity",
        },
        {
            "intervention_kind": "component_scale",
            "control_label": "L0_attention_all_zero",
            "layer_idx": 0,
            "component_kind": "attention",
            "factor": 0.0,
            "last_token_only": False,
            "control_family": "position_scope_control",
        },
        {
            "intervention_kind": "component_scale",
            "control_label": "L0_mlp_last_zero",
            "layer_idx": 0,
            "component_kind": "mlp",
            "factor": 0.0,
            "last_token_only": True,
            "control_family": "component_control",
        },
        {
            "intervention_kind": "component_scale",
            "control_label": "L1_attention_last_zero",
            "layer_idx": 1,
            "component_kind": "attention",
            "factor": 0.0,
            "last_token_only": True,
            "control_family": "nearby_layer_control",
        },
    ]
    num_heads = config_num_heads(model)
    proj = find_output_projection(model, 0)
    if num_heads is not None and proj is not None:
        max_heads = int(args.max_heads)
        head_indices = list(range(int(num_heads)))
        if max_heads > 0:
            head_indices = head_indices[:max_heads]
        for head_idx in head_indices:
            specs.append(
                {
                    "intervention_kind": "head_scale",
                    "control_label": f"L0_attention_head{head_idx}_zero",
                    "layer_idx": 0,
                    "component_kind": "attention",
                    "head_idx": int(head_idx),
                    "factor": 0.0,
                    "last_token_only": True,
                    "control_family": "head_zero",
                }
            )
    return specs


def metric_delta(logits: torch.Tensor, baseline_logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    return float(logits[int(token_id)].item() - baseline_logits[int(token_id)].item())


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    spec: dict[str, Any],
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
    base_eos_id = baseline_metrics.get("eos_best_id")
    base_next_id = baseline_metrics.get("next_top_id")
    base_protocol_id = baseline_metrics.get("protocol_best_id")
    base_field_id = baseline_metrics.get("field_word_best_id")
    base_explanation_id = baseline_metrics.get("explanation_best_id")
    eos_delta = metric_delta(patched_logits, baseline_logits, base_eos_id)
    next_delta = metric_delta(patched_logits, baseline_logits, base_next_id)
    protocol_delta = metric_delta(patched_logits, baseline_logits, base_protocol_id)
    field_delta = metric_delta(patched_logits, baseline_logits, base_field_id)
    explanation_delta = metric_delta(patched_logits, baseline_logits, base_explanation_id)
    base_eos_logit = baseline_metrics.get("eos_best_logit")
    base_next_logit = baseline_metrics.get("next_top_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    patched_next_logit = patched_metrics.get("next_top_logit")
    margin_delta = None
    if None not in [base_eos_logit, base_next_logit, patched_eos_logit, patched_next_logit]:
        margin_delta = float(patched_eos_logit - patched_next_logit) - float(base_eos_logit - base_next_logit)
    return {
        "phase": PHASE,
        "row_kind": "phase908_l0_attention_fine_audit_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        **spec,
        "baseline_next_top_category": baseline_metrics.get("next_top_category"),
        "patched_next_top_category": patched_metrics.get("next_top_category"),
        "category_transition": f"{baseline_metrics.get('next_top_category')}->{patched_metrics.get('next_top_category')}",
        "baseline_eos_rank": base_eos_rank,
        "patched_eos_rank": patched_eos_rank,
        "eos_rank_delta": eos_rank_delta,
        "eos_rank_improved": bool(eos_rank_delta is not None and eos_rank_delta < 0),
        "eos_rank_improved_100": bool(eos_rank_delta is not None and eos_rank_delta <= -100),
        "eos_rank_improved_1000": bool(eos_rank_delta is not None and eos_rank_delta <= -1000),
        "patched_eos_top1": bool(patched_metrics.get("eos_rank") == 1),
        "patched_eos_top10": bool(patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 10),
        "patched_eos_top50": bool(patched_metrics.get("eos_rank") is not None and int(patched_metrics["eos_rank"]) <= 50),
        "baseline_eos_logit_delta": eos_delta,
        "baseline_next_logit_delta": next_delta,
        "baseline_protocol_logit_delta": protocol_delta,
        "baseline_field_logit_delta": field_delta,
        "baseline_explanation_logit_delta": explanation_delta,
        "eos_vs_next_margin_delta": margin_delta,
        "direct_eos_lift": bool(eos_delta is not None and eos_delta > 0.0),
        "continuation_suppressed": bool(next_delta is not None and next_delta < 0.0),
        "protocol_suppressed": bool(protocol_delta is not None and protocol_delta < 0.0),
        "next_top_changed": bool(baseline_metrics.get("next_top_id") != patched_metrics.get("next_top_id")),
        "next_category_changed": bool(baseline_metrics.get("next_top_category") != patched_metrics.get("next_top_category")),
        "patched_eos_token": p903.decode_token(tokenizer, patched_metrics.get("eos_best_id")),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "eos_rank_improved": sum(1 for row in rows if row.get("eos_rank_improved")),
        "eos_rank_improved_100": sum(1 for row in rows if row.get("eos_rank_improved_100")),
        "eos_rank_improved_1000": sum(1 for row in rows if row.get("eos_rank_improved_1000")),
        "patched_eos_top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "patched_eos_top10": sum(1 for row in rows if row.get("patched_eos_top10")),
        "patched_eos_top50": sum(1 for row in rows if row.get("patched_eos_top50")),
        "direct_eos_lift": sum(1 for row in rows if row.get("direct_eos_lift")),
        "continuation_suppressed": sum(1 for row in rows if row.get("continuation_suppressed")),
        "protocol_suppressed": sum(1 for row in rows if row.get("protocol_suppressed")),
        "next_top_changed": sum(1 for row in rows if row.get("next_top_changed")),
        "next_category_changed": sum(1 for row in rows if row.get("next_category_changed")),
        "median_eos_rank_delta": median([row.get("eos_rank_delta") for row in rows]),
        "median_eos_logit_delta": median([row.get("baseline_eos_logit_delta") for row in rows]),
        "median_next_logit_delta": median([row.get("baseline_next_logit_delta") for row in rows]),
        "median_eos_vs_next_margin_delta": median([row.get("eos_vs_next_margin_delta") for row in rows]),
        "category_transitions": dict(sorted(Counter(str(row.get("category_transition")) for row in rows).items())),
    }


def summarize_by_control(rows: list[dict[str, Any]], top_n: int = 40) -> list[dict[str, Any]]:
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
                "intervention_kind": first.get("intervention_kind"),
                "layer_idx": first.get("layer_idx"),
                "component_kind": first.get("component_kind"),
                "head_idx": first.get("head_idx"),
                "factor": first.get("factor"),
                "last_token_only": first.get("last_token_only"),
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


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, specs: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    control_summaries = summarize_by_control(rows)
    if overall["patched_eos_top1"] > 0:
        evidence = "l0_attention_fine_audit_reaches_eos_top1"
    elif overall["patched_eos_top10"] > 0:
        evidence = "l0_attention_fine_audit_reaches_eos_top10"
    elif overall["patched_eos_top50"] > 0:
        evidence = "l0_attention_fine_audit_reaches_eos_top50"
    elif overall["eos_rank_improved_1000"] > 0:
        evidence = "l0_attention_fine_audit_improves_eos_but_not_near"
    else:
        evidence = "no_l0_attention_eos_proximity_signal"
    return {
        "phase": PHASE,
        "title": "GLM4 L0 Attention EOS-Proximity Fine Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "intervention_count": len(specs),
        "overall": overall,
        "control_summaries": control_summaries,
        "evidence_label": evidence,
        "boundary": (
            "Phase908 is a cross-model fine audit of L0 attention EOS proximity. "
            "EOS top10/top50 is proximity, not closure."
        ),
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
        p846.write_json(out_dir / f"phase908_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase908_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    specs: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        specs = intervention_specs(model, args)
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
                patched_logits = logits_with_intervention(model, device, period_ids, spec)
                if patched_logits is None:
                    continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                rows.append(make_row(tokenizer, source_row, spec, baseline_metrics, patched_metrics, patched_logits, baseline_logits))
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
    p846.write_json(out_dir / f"phase908_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase908_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_controls = []
    for model_name in MODELS:
        path = out_dir / f"phase908_{model_name}_summary.json"
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
    p846.write_json(out_dir / "phase908_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase908_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 908 L0 attention EOS proximity fine audit",
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
    lines.append("| model | control | family | rows | eos top10 | eos top50 | lift | suppress | median margin delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        lines.append(
            "| {model} | {control_label} | {control_family} | {rows} | {patched_eos_top10} | {patched_eos_top50} | {direct_eos_lift} | {continuation_suppressed} | {median_eos_vs_next_margin_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="l0_attention_eos_proximity_fine_audit")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-heads", type=int, default=0)
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
