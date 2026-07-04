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
import phase900_protocol_stop_gate_discovery as p900  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402


PHASE = 902
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase902_protocol_continuation_suppressor_search")
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


def decode_token(tokenizer, token_id: int | None) -> str | None:
    return p901.decode_token(tokenizer, token_id)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def median(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


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


def parse_gears(subset_key: str) -> list[dict[str, Any]]:
    gears = []
    for part in str(subset_key or "").split("+"):
        if part.startswith("L") and "C" in part:
            gear = p862.parse_gear_key(part)
            if gear is not None:
                gears.append(gear)
    return gears


def unique_ids(values: list[int]) -> list[int]:
    out = []
    for value in values:
        value = int(value)
        if value not in out and value >= 0:
            out.append(value)
    return out


def token_sets(groups: dict[str, list[int]]) -> dict[str, list[int]]:
    stop_ids = unique_ids((groups.get("eos") or []) + (groups.get("period") or []))
    protocol_ids = unique_ids(
        (groups.get("field") or []) + (groups.get("explanation") or []) + (groups.get("list") or [])
    )
    return {**groups, "stop": stop_ids, "protocol": protocol_ids}


def control_specs(model, model_name: str, source_gears: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        {
            "control_type": "baseline_after_answer_prefix",
            "control_label": "baseline",
            "gear_mode": None,
            "heads": [],
        },
        {
            "control_type": "source_repeat_after_prefix",
            "control_label": "source_repeat_after_prefix",
            "gear_mode": "source",
            "heads": [],
        },
        {
            "control_type": "gear_half_after_prefix",
            "control_label": "gear_half_after_prefix",
            "gear_mode": "half",
            "heads": [],
        },
        {
            "control_type": "gear_zero_after_prefix",
            "control_label": "gear_zero_after_prefix",
            "gear_mode": "zero",
            "heads": [],
        },
        {
            "control_type": "gear_flip_after_prefix",
            "control_label": "gear_flip_after_prefix",
            "gear_mode": "flip",
            "heads": [],
        },
    ]
    head_keys = list(p900.HISTORICAL_HEAD_SETS.get(model_name, []))
    head_keys.extend(p900.same_layer_head_sets(model, source_gears, int(args.max_same_layer_heads)))
    seen = set()
    for head_key in head_keys:
        heads = p900.parse_head_set(head_key)
        if not heads:
            continue
        key = p900.head_set_key(heads)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "control_type": "head_zero_after_prefix",
                "control_label": f"head_zero_after_prefix::{key}",
                "gear_mode": None,
                "heads": heads,
            }
        )
    return specs


def install_control(
    model,
    gears: list[dict[str, Any]],
    source_mode: str,
    spec: dict[str, Any],
    scale_up_factor: float,
) -> list[Any]:
    handles: list[Any] = []
    gear_mode = spec.get("gear_mode")
    if gear_mode == "source":
        gear_mode = source_mode
    if gear_mode and gear_mode != "original" and gears:
        handles.extend(p862.install_scaled_gear_edit(model, gears, str(gear_mode), scale_up_factor))
    heads = list(spec.get("heads") or [])
    if heads:
        handles.extend(p900.install_heads(model, heads))
    return handles


def logits_with_control(
    model,
    device: torch.device,
    current_ids: list[int],
    gears: list[dict[str, Any]],
    source_mode: str,
    spec: dict[str, Any],
    scale_up_factor: float,
) -> torch.Tensor:
    input_ids = torch.tensor([current_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = install_control(model, gears, source_mode, spec, scale_up_factor)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def suffix_rollout_with_control(
    model,
    tokenizer,
    device: torch.device,
    current_ids: list[int],
    prefix_ids: list[int],
    gears: list[dict[str, Any]],
    source_mode: str,
    spec: dict[str, Any],
    max_suffix_tokens: int,
    suppress_steps: int,
    scale_up_factor: float,
) -> tuple[str, str, list[int]]:
    current = [int(x) for x in current_ids]
    suffix_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_suffix_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles: list[Any] = []
        if step < int(suppress_steps):
            handles = install_control(model, gears, source_mode, spec, scale_up_factor)
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            for handle in handles:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        suffix_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    combined_ids = [int(x) for x in prefix_ids] + suffix_ids
    combined_text = tokenizer.decode(combined_ids, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    return combined_text, suffix_text, suffix_ids


def rank_for_token(logits: torch.Tensor, token_id: int | None) -> int | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    score = float(logits[int(token_id)].item())
    return int((logits > score).sum().item()) + 1


def logit_for_token(logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    return float(logits[int(token_id)].item())


def metric_payload(
    tokenizer,
    logits: torch.Tensor,
    groups: dict[str, list[int]],
    baseline_logits: torch.Tensor | None,
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    top_id = int(torch.argmax(logits).item())
    top_logit = float(logits[top_id].item())
    payload: dict[str, Any] = {
        "next_top_id": top_id,
        "next_top_token": decode_token(tokenizer, top_id),
        "next_top_logit": top_logit,
    }
    for group_name, ids in token_sets(groups).items():
        best = p901.best_for_ids(logits, ids)
        payload[f"{group_name}_best_id"] = best.get("best_id")
        payload[f"{group_name}_best_token"] = decode_token(tokenizer, best.get("best_id"))
        payload[f"{group_name}_best_logit"] = best.get("best_logit")
        payload[f"{group_name}_rank"] = best.get("rank")
        payload[f"{group_name}_margin_vs_top"] = None if best.get("best_logit") is None else float(best["best_logit"] - top_logit)
    payload["stop_top1"] = bool(payload.get("stop_rank") == 1)
    payload["period_top1"] = bool(payload.get("period_rank") == 1)
    payload["eos_top1"] = bool(payload.get("eos_rank") == 1)
    payload["protocol_top1"] = bool(payload.get("protocol_rank") == 1)
    payload["stop_top10"] = bool(payload.get("stop_rank") is not None and int(payload["stop_rank"]) <= 10)
    if baseline_logits is not None and baseline_metrics is not None:
        base_protocol_id = baseline_metrics.get("protocol_best_id")
        base_next_id = baseline_metrics.get("next_top_id")
        protocol_delta = None
        if base_protocol_id is not None:
            protocol_delta = float(logits[int(base_protocol_id)].item() - baseline_logits[int(base_protocol_id)].item())
        next_delta = None
        if base_next_id is not None:
            next_delta = float(logits[int(base_next_id)].item() - baseline_logits[int(base_next_id)].item())
        payload["baseline_protocol_token"] = decode_token(tokenizer, base_protocol_id)
        payload["baseline_next_top_token"] = decode_token(tokenizer, base_next_id)
        payload["baseline_protocol_logit_delta"] = protocol_delta
        payload["baseline_next_top_logit_delta"] = next_delta
        payload["baseline_protocol_rank"] = baseline_metrics.get("protocol_rank")
        payload["baseline_stop_rank"] = baseline_metrics.get("stop_rank")
        payload["protocol_rank1_removed"] = bool(baseline_metrics.get("protocol_rank") == 1 and payload.get("protocol_rank") != 1)
        payload["next_top_changed"] = bool(base_next_id is not None and int(base_next_id) != int(top_id))
        if baseline_metrics.get("stop_rank") is not None and payload.get("stop_rank") is not None:
            payload["stop_rank_delta"] = int(payload["stop_rank"]) - int(baseline_metrics["stop_rank"])
            payload["stop_rank_improved"] = bool(int(payload["stop_rank"]) < int(baseline_metrics["stop_rank"]))
        else:
            payload["stop_rank_delta"] = None
            payload["stop_rank_improved"] = False
    else:
        payload["baseline_protocol_token"] = None
        payload["baseline_next_top_token"] = None
        payload["baseline_protocol_logit_delta"] = 0.0
        payload["baseline_next_top_logit_delta"] = 0.0
        payload["baseline_protocol_rank"] = payload.get("protocol_rank")
        payload["baseline_stop_rank"] = payload.get("stop_rank")
        payload["protocol_rank1_removed"] = False
        payload["next_top_changed"] = False
        payload["stop_rank_delta"] = 0
        payload["stop_rank_improved"] = False
    return payload


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    answer_seen: bool,
    metrics: dict[str, Any],
    combined_text: str,
    suffix_text: str,
    suffix_ids: list[int],
) -> dict[str, Any]:
    flags = p900.rollout_flags(combined_text, case)
    first_suffix_id = suffix_ids[0] if suffix_ids else None
    return {
        "phase": PHASE,
        "row_kind": "phase902_protocol_continuation_suppressor_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "canonical_answer": source_row.get("canonical_answer"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "prefix_ids": prefix_ids,
        "prefix_text": prefix_text,
        "answer_prefix_seen": answer_seen,
        "control_type": spec.get("control_type"),
        "control_label": spec.get("control_label"),
        "gear_mode": spec.get("gear_mode"),
        "head_set": p900.head_set_key(list(spec.get("heads") or [])),
        "first_suffix_id": first_suffix_id,
        "first_suffix_token": decode_token(tokenizer, first_suffix_id),
        "suffix_ids": suffix_ids,
        "suffix_text": suffix_text,
        "combined_text": combined_text,
        **metrics,
        **flags,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row.get("baseline_protocol_logit_delta") for row in rows if row.get("baseline_protocol_logit_delta") is not None]
    stop_deltas = [row.get("stop_rank_delta") for row in rows if row.get("stop_rank_delta") is not None]
    return {
        "rows": len(rows),
        "answer_prefix_seen": sum(1 for row in rows if row.get("answer_prefix_seen")),
        "answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "clean_answer_no_protocol": sum(1 for row in rows if row.get("rollout_clear_answer_no_protocol")),
        "class_no_echo_no_protocol": sum(1 for row in rows if row.get("rollout_class_no_echo_no_protocol")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "protocol_top1": sum(1 for row in rows if row.get("protocol_top1")),
        "protocol_rank1_removed": sum(1 for row in rows if row.get("protocol_rank1_removed")),
        "protocol_logit_delta_negative": sum(1 for row in rows if (row.get("baseline_protocol_logit_delta") or 0.0) < 0.0),
        "protocol_logit_delta_below_minus_0_5": sum(
            1 for row in rows if row.get("baseline_protocol_logit_delta") is not None and float(row["baseline_protocol_logit_delta"]) <= -0.5
        ),
        "next_top_changed": sum(1 for row in rows if row.get("next_top_changed")),
        "stop_rank_improved": sum(1 for row in rows if row.get("stop_rank_improved")),
        "stop_top1": sum(1 for row in rows if row.get("stop_top1")),
        "period_top1": sum(1 for row in rows if row.get("period_top1")),
        "eos_top1": sum(1 for row in rows if row.get("eos_top1")),
        "stop_top10": sum(1 for row in rows if row.get("stop_top10")),
        "mean_protocol_logit_delta": mean([float(value) for value in deltas]),
        "median_protocol_logit_delta": median([float(value) for value in deltas]),
        "mean_stop_rank_delta": mean([float(value) for value in stop_deltas]),
        "median_stop_rank_delta": median([float(value) for value in stop_deltas]),
        "next_top_tokens": dict(sorted(Counter(str(row.get("next_top_token")) for row in rows).items())),
        "first_suffix_tokens": dict(sorted(Counter(str(row.get("first_suffix_token")) for row in rows).items())),
        "labels": dict(sorted(Counter(str(row.get("rollout_label")) for row in rows).items())),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_control[str(row.get("control_label"))].append(row)
    control_summaries = []
    for label, vals in by_control.items():
        summary = summarize_rows(vals)
        summary.update(
            {
                "control_label": label,
                "control_type": vals[0].get("control_type"),
                "head_set": vals[0].get("head_set"),
                "gear_mode": vals[0].get("gear_mode"),
            }
        )
        control_summaries.append(summary)
    control_summaries.sort(
        key=lambda row: (
            row.get("clean_answer_no_protocol") or 0,
            row.get("protocol_rank1_removed") or 0,
            row.get("protocol_logit_delta_below_minus_0_5") or 0,
            row.get("stop_rank_improved") or 0,
        ),
        reverse=True,
    )
    baseline = summarize_rows([row for row in rows if row.get("control_type") == "baseline_after_answer_prefix"])
    non_base = [row for row in rows if row.get("control_type") != "baseline_after_answer_prefix"]
    non_base_summary = summarize_rows(non_base)
    if non_base_summary["clean_answer_no_protocol"] > 0:
        evidence_label = "limited_protocol_suppressor_rollout_closure_found"
    elif non_base_summary["protocol_rank1_removed"] > 0 and non_base_summary["stop_top1"] > 0:
        evidence_label = "logit_competition_shift_without_clean_rollout_closure"
    elif non_base_summary["protocol_logit_delta_negative"] > 0:
        evidence_label = "weak_protocol_logit_suppression_without_clean_rollout_closure"
    else:
        evidence_label = "no_simple_protocol_continuation_suppressor_found"
    return {
        "phase": PHASE,
        "title": "Protocol Continuation Suppressor Search",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": len(selected_rows),
        "overall": {
            "selected_answer_drift_rows": len(selected_rows),
            "control_rows": len(rows),
            "baseline": baseline,
            "non_baseline": non_base_summary,
            "best_control": control_summaries[0] if control_summaries else {},
        },
        "control_summaries": control_summaries,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase902 tests finite suppressors at the answer-prefix boundary and the immediate following step. "
            "It can identify local competition shifts, but it is not an exhaustive protocol-control closure test."
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
        p846.write_json(out_dir / f"phase902_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase902_{args.model}_rows.jsonl", [])
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
        groups = p901.token_groups(tokenizer)
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
            gears = parse_gears(str(source_row.get("source_subset_key")))
            source_mode = str(source_row.get("edit_mode"))
            prefix_logits, prefix_ids, prefix_text, answer_seen = p901.logits_after_answer_prefix(
                model,
                tokenizer,
                device,
                prompt_ids,
                gears,
                source_mode,
                case,
                int(args.max_prefix_tokens),
                float(args.scale_up_factor),
            )
            current_ids = [int(x) for x in prompt_ids] + [int(x) for x in prefix_ids]
            specs = control_specs(model, args.model, gears, args)
            baseline_spec = specs[0]
            baseline_logits = logits_with_control(
                model,
                device,
                current_ids,
                gears,
                source_mode,
                baseline_spec,
                float(args.scale_up_factor),
            )
            baseline_metrics = metric_payload(tokenizer, baseline_logits, groups, None, None)
            for spec in specs:
                logits = baseline_logits if spec.get("control_type") == "baseline_after_answer_prefix" else logits_with_control(
                    model,
                    device,
                    current_ids,
                    gears,
                    source_mode,
                    spec,
                    float(args.scale_up_factor),
                )
                metrics = metric_payload(tokenizer, logits, groups, baseline_logits, baseline_metrics)
                combined_text, suffix_text, suffix_ids = suffix_rollout_with_control(
                    model,
                    tokenizer,
                    device,
                    current_ids,
                    prefix_ids,
                    gears,
                    source_mode,
                    spec,
                    int(args.max_suffix_tokens),
                    int(args.suppress_steps),
                    float(args.scale_up_factor),
                )
                rows.append(
                    make_row(
                        tokenizer,
                        source_row,
                        case,
                        spec,
                        prefix_ids,
                        prefix_text,
                        answer_seen,
                        metrics,
                        combined_text,
                        suffix_text,
                        suffix_ids,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} control_rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, selected_rows, attn_impl)
    p846.write_json(out_dir / f"phase902_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase902_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 902 protocol continuation suppressor search",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Best controls", ""])
    lines.append(
        "| model | control | type | head set | rows | clean | class clean | drift | protocol removed | protocol delta < -0.5 | stop improved | stop top1 | top changed | evidence |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("best_controls") or []:
        lines.append(
            "| {model} | {control_label} | {control_type} | {head_set} | {rows} | {clean_answer_no_protocol} | "
            "{class_no_echo_no_protocol} | {protocol_drift} | {protocol_rank1_removed} | "
            "{protocol_logit_delta_below_minus_0_5} | {stop_rank_improved} | {stop_top1} | {next_top_changed} | {evidence_label} |".format(
                **row
            )
        )
    lines.extend(["", "## Model summaries", ""])
    lines.append(
        "| model | selected | control rows | non-base clean | protocol removed | protocol delta negative | stop improved | stop top1 | best control | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        non_base = overall.get("non_baseline") or {}
        best = overall.get("best_control") or {}
        lines.append(
            "| {model} | {selected} | {control_rows} | {clean} | {removed} | {negative} | {improved} | {stop_top1} | {best} | {evidence} |".format(
                model=summary.get("model"),
                selected=overall.get("selected_answer_drift_rows"),
                control_rows=overall.get("control_rows"),
                clean=non_base.get("clean_answer_no_protocol"),
                removed=non_base.get("protocol_rank1_removed"),
                negative=non_base.get("protocol_logit_delta_negative"),
                improved=non_base.get("stop_rank_improved"),
                stop_top1=non_base.get("stop_top1"),
                best=best.get("control_label"),
                evidence=summary.get("evidence_label"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    best_controls = []
    scalar = Counter()
    evidence = Counter()
    for model_name in MODELS:
        summary_path = out_dir / f"phase902_{model_name}_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        scalar["selected_answer_drift_rows"] += int(overall.get("selected_answer_drift_rows") or 0)
        scalar["control_rows"] += int(overall.get("control_rows") or 0)
        for key in [
            "clean_answer_no_protocol",
            "protocol_drift",
            "protocol_rank1_removed",
            "protocol_logit_delta_negative",
            "protocol_logit_delta_below_minus_0_5",
            "stop_rank_improved",
            "stop_top1",
            "next_top_changed",
        ]:
            scalar[f"non_base_{key}"] += int((overall.get("non_baseline") or {}).get(key) or 0)
        best = dict(overall.get("best_control") or {})
        if best:
            best["model"] = summary.get("model")
            best["evidence_label"] = summary.get("evidence_label")
            best_controls.append(best)
    best_controls.sort(
        key=lambda row: (
            row.get("clean_answer_no_protocol") or 0,
            row.get("protocol_rank1_removed") or 0,
            row.get("protocol_logit_delta_below_minus_0_5") or 0,
            row.get("stop_rank_improved") or 0,
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
        "best_controls": best_controls,
    }
    p846.write_json(out_dir / "phase902_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase902_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="protocol_continuation_suppressor_search")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--max-suffix-tokens", type=int, default=8)
    parser.add_argument("--suppress-steps", type=int, default=2)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-same-layer-heads", type=int, default=4)
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
