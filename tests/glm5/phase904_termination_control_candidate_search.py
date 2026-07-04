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
import phase902_protocol_continuation_suppressor_search as p902  # noqa: E402
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402


PHASE = 904
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase904_termination_control_candidate_search")
PHASE899_ROOT = Path("tests/result/phase899_domain_axis_rollout_protocol_audit")
PHASE903_ROOT = Path("tests/result/phase903_protocol_continuation_field_mapping")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"
PHASE903_ROUND = "protocol_continuation_field_mapping"


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
    return p903.parse_gears(subset_key)


def component_candidates(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    summary = read_json(PHASE903_ROOT / args.phase903_round / f"phase903_{model_name}_summary.json")
    candidates = []
    seen = set()
    for row in summary.get("top_components") or []:
        key = (int(row.get("layer_idx")), str(row.get("component_kind")))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "control_type": "component_zero",
                "control_label": f"{row.get('component_kind')}_zero_L{row.get('layer_idx')}",
                "layer_idx": int(row.get("layer_idx")),
                "component_kind": str(row.get("component_kind")),
                "source_category": row.get("baseline_protocol_best_category"),
                "phase903_strong": int(row.get("protocol_logit_reduced_strong") or 0),
                "phase903_removed": int(row.get("protocol_rank1_removed") or 0),
                "phase903_stop_improved": int(row.get("stop_rank_improved") or 0),
                "phase903_mean_delta": row.get("mean_protocol_logit_delta"),
            }
        )
        if len(candidates) >= int(args.max_candidates):
            break
    return candidates


def install_component_zero(model, layer_idx: int, component_kind: str) -> list[Any]:
    module = p903.component_module(model, int(layer_idx), component_kind)
    if module is None:
        return []
    return [module.register_forward_hook(lambda _module, _inputs, output: p903.zero_last_token_output(output))]


def logits_with_candidate(
    model,
    device: torch.device,
    current_ids: list[int],
    spec: dict[str, Any],
) -> torch.Tensor:
    if spec.get("control_type") == "baseline":
        return p903.logits_plain(model, device, current_ids)
    handles = install_component_zero(model, int(spec["layer_idx"]), str(spec["component_kind"]))
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def suffix_rollout_with_candidate(
    model,
    tokenizer,
    device: torch.device,
    current_ids: list[int],
    prefix_ids: list[int],
    spec: dict[str, Any],
    max_suffix_tokens: int,
    suppress_steps: int,
) -> tuple[str, str, list[int]]:
    current = [int(x) for x in current_ids]
    suffix_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_suffix_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles: list[Any] = []
        if spec.get("control_type") != "baseline" and step < int(suppress_steps):
            handles = install_component_zero(model, int(spec["layer_idx"]), str(spec["component_kind"]))
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


def metric_delta(logits: torch.Tensor, baseline_logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    return float(logits[int(token_id)].item() - baseline_logits[int(token_id)].item())


def word_count(text: str) -> int:
    import re

    return len(re.findall(r"[A-Za-z]+", str(text or "")))


def strict_protocol_drift(text: str) -> bool:
    raw = str(text or "")
    low = raw.lower()
    markers = [
        "\n",
        "</think>",
        "the item",
        " can",
        " is a",
        " are ",
        " because",
        "category:",
        "item:",
        "class:",
        "subclass:",
        "answer:",
        "the answer is",
        "okay",
        "please",
    ]
    return any(marker in low for marker in markers) or word_count(low) > 3


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    answer_seen: bool,
    baseline_metrics: dict[str, Any],
    metrics: dict[str, Any],
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    combined_text: str,
    suffix_text: str,
    suffix_ids: list[int],
) -> dict[str, Any]:
    flags = p900.rollout_flags(combined_text, case)
    strict_drift = strict_protocol_drift(combined_text)
    strict_clean = (
        bool(flags.get("rollout_answer_class"))
        and not bool(flags.get("rollout_object_echo"))
        and not bool(flags.get("protocol_drift"))
        and not strict_drift
    )
    protocol_delta = metric_delta(logits, baseline_logits, baseline_metrics.get("protocol_best_id"))
    stop_delta = metric_delta(logits, baseline_logits, baseline_metrics.get("stop_best_id"))
    stop_rank_delta = None
    if baseline_metrics.get("stop_rank") is not None and metrics.get("stop_rank") is not None:
        stop_rank_delta = int(metrics["stop_rank"]) - int(baseline_metrics["stop_rank"])
    first_suffix_id = suffix_ids[0] if suffix_ids else None
    second_suffix_id = suffix_ids[1] if len(suffix_ids) > 1 else None
    return {
        "phase": PHASE,
        "row_kind": "phase904_termination_candidate_row",
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
        "control_type": spec.get("control_type"),
        "control_label": spec.get("control_label"),
        "layer_idx": spec.get("layer_idx"),
        "component_kind": spec.get("component_kind"),
        "source_category": spec.get("source_category"),
        "phase903_strong": spec.get("phase903_strong"),
        "phase903_removed": spec.get("phase903_removed"),
        "phase903_stop_improved": spec.get("phase903_stop_improved"),
        "phase903_mean_delta": spec.get("phase903_mean_delta"),
        "first_suffix_id": first_suffix_id,
        "first_suffix_token": p903.decode_token(tokenizer, first_suffix_id),
        "first_suffix_category": p903.category_for_token(first_suffix_id, p903.protocol_category_groups(tokenizer)),
        "second_suffix_id": second_suffix_id,
        "second_suffix_token": p903.decode_token(tokenizer, second_suffix_id),
        "second_suffix_category": p903.category_for_token(second_suffix_id, p903.protocol_category_groups(tokenizer)),
        "suffix_ids": suffix_ids,
        "suffix_text": suffix_text,
        "combined_text": combined_text,
        "next_top_category": metrics.get("next_top_category"),
        "protocol_best_category": metrics.get("protocol_best_category"),
        "stop_best_category": metrics.get("stop_best_category"),
        "protocol_rank": metrics.get("protocol_rank"),
        "stop_rank": metrics.get("stop_rank"),
        "protocol_logit_delta": protocol_delta,
        "stop_logit_delta": stop_delta,
        "stop_rank_delta": stop_rank_delta,
        "protocol_logit_reduced": bool(protocol_delta is not None and protocol_delta < 0),
        "protocol_logit_reduced_strong": bool(protocol_delta is not None and protocol_delta <= -0.5),
        "protocol_rank1_removed": bool(baseline_metrics.get("protocol_rank") == 1 and metrics.get("protocol_rank") != 1),
        "stop_rank_improved": bool(stop_rank_delta is not None and stop_rank_delta < 0),
        "stop_top1": bool(metrics.get("stop_rank") == 1),
        "stop_top10": bool(metrics.get("stop_rank") is not None and int(metrics["stop_rank"]) <= 10),
        "next_top_changed": bool(metrics.get("next_top_id") != baseline_metrics.get("next_top_id")),
        "strict_protocol_drift": strict_drift,
        "strict_clean_answer_no_protocol": strict_clean,
        **flags,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row.get("protocol_logit_delta") for row in rows if row.get("protocol_logit_delta") is not None]
    return {
        "rows": len(rows),
        "answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "clean_answer_no_protocol": sum(1 for row in rows if row.get("rollout_clear_answer_no_protocol")),
        "strict_clean_answer_no_protocol": sum(
            1
            for row in rows
            if row.get("strict_clean_answer_no_protocol")
            or (
                "strict_clean_answer_no_protocol" not in row
                and row.get("rollout_answer_class")
                and not row.get("rollout_object_echo")
                and not row.get("protocol_drift")
                and not strict_protocol_drift(str(row.get("combined_text") or ""))
            )
        ),
        "class_no_echo_no_protocol": sum(1 for row in rows if row.get("rollout_class_no_echo_no_protocol")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "strict_protocol_drift": sum(
            1
            for row in rows
            if row.get("strict_protocol_drift")
            or ("strict_protocol_drift" not in row and strict_protocol_drift(str(row.get("combined_text") or "")))
        ),
        "object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "protocol_logit_reduced": sum(1 for row in rows if row.get("protocol_logit_reduced")),
        "protocol_logit_reduced_strong": sum(1 for row in rows if row.get("protocol_logit_reduced_strong")),
        "protocol_rank1_removed": sum(1 for row in rows if row.get("protocol_rank1_removed")),
        "stop_rank_improved": sum(1 for row in rows if row.get("stop_rank_improved")),
        "stop_top1": sum(1 for row in rows if row.get("stop_top1")),
        "stop_top10": sum(1 for row in rows if row.get("stop_top10")),
        "next_top_changed": sum(1 for row in rows if row.get("next_top_changed")),
        "mean_protocol_logit_delta": mean([float(value) for value in deltas]),
        "median_protocol_logit_delta": median([float(value) for value in deltas]),
        "first_suffix_categories": dict(sorted(Counter(str(row.get("first_suffix_category")) for row in rows).items())),
        "second_suffix_categories": dict(sorted(Counter(str(row.get("second_suffix_category")) for row in rows).items())),
        "labels": dict(sorted(Counter(str(row.get("rollout_label")) for row in rows).items())),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
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
                "layer_idx": vals[0].get("layer_idx"),
                "component_kind": vals[0].get("component_kind"),
                "source_category": vals[0].get("source_category"),
            }
        )
        control_summaries.append(summary)
    control_summaries.sort(
        key=lambda row: (
            row.get("strict_clean_answer_no_protocol") or 0,
            row.get("protocol_rank1_removed") or 0,
            row.get("stop_rank_improved") or 0,
            row.get("protocol_logit_reduced_strong") or 0,
        ),
        reverse=True,
    )
    baseline = summarize_rows([row for row in rows if row.get("control_type") == "baseline"])
    non_base = summarize_rows([row for row in rows if row.get("control_type") != "baseline"])
    if non_base["strict_clean_answer_no_protocol"] > 0:
        evidence_label = "termination_candidate_has_clean_rollout"
    elif non_base["stop_rank_improved"] > 0 and non_base["protocol_rank1_removed"] > 0:
        evidence_label = "termination_candidate_changes_competition_without_clean_rollout"
    elif non_base["protocol_logit_reduced_strong"] > 0:
        evidence_label = "termination_candidate_reduces_protocol_without_clean_rollout"
    else:
        evidence_label = "no_termination_candidate_found"
    return {
        "phase": PHASE,
        "title": "Termination Control Candidate Search",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": len(selected_rows),
        "candidates": candidates,
        "overall": {
            "selected_answer_drift_rows": len(selected_rows),
            "candidate_count": len(candidates),
            "control_rows": len(rows),
            "baseline": baseline,
            "non_baseline": non_base,
            "best_control": control_summaries[0] if control_summaries else {},
        },
        "control_summaries": control_summaries,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase904 applies top Phase903 layer-component candidates during the first answer-prefix suffix steps. "
            "It is a candidate rollout test, not exhaustive termination-control closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = selected_phase899_rows(args.model, args)
    candidates = component_candidates(args.model, args)
    if args.dry_run or not selected_rows or not candidates:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows and candidates else "no_rows_or_candidates",
            "selected_rows": selected_rows,
            "candidates": candidates,
        }
        p846.write_json(out_dir / f"phase904_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase904_{args.model}_rows.jsonl", [])
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
        specs = [{"control_type": "baseline", "control_label": "baseline"}] + candidates
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
            for spec in specs:
                logits = baseline_logits if spec.get("control_type") == "baseline" else logits_with_candidate(
                    model, device, current_ids, spec
                )
                metrics = p903.state_metrics(tokenizer, logits, groups)
                combined_text, suffix_text, suffix_ids = suffix_rollout_with_candidate(
                    model,
                    tokenizer,
                    device,
                    current_ids,
                    prefix_ids,
                    spec,
                    int(args.max_suffix_tokens),
                    int(args.suppress_steps),
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
                        baseline_metrics,
                        metrics,
                        logits,
                        baseline_logits,
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

    payload = summarize_model(args.model, rows, selected_rows, candidates, attn_impl)
    p846.write_json(out_dir / f"phase904_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase904_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 904 termination control candidate search",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append("| model | candidates | rows | non-base strict clean | drift | removed | stop improved | stop top1 | top changed | evidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        non_base = overall.get("non_baseline") or {}
        lines.append(
            "| {model} | {candidates} | {rows} | {clean} | {drift} | {removed} | {stop_imp} | {stop_top1} | {changed} | {evidence} |".format(
                model=summary.get("model"),
                candidates=overall.get("candidate_count"),
                rows=overall.get("control_rows"),
                clean=non_base.get("strict_clean_answer_no_protocol"),
                drift=non_base.get("protocol_drift"),
                removed=non_base.get("protocol_rank1_removed"),
                stop_imp=non_base.get("stop_rank_improved"),
                stop_top1=non_base.get("stop_top1"),
                changed=non_base.get("next_top_changed"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Best Controls", ""])
    lines.append("| model | control | layer | kind | category | rows | strict clean | nominal clean | drift | removed | stop improved | first suffix categories |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("best_controls") or []:
        row_fmt = dict(row)
        row_fmt["first_suffix_categories"] = json.dumps(row.get("first_suffix_categories") or {}, ensure_ascii=False)
        lines.append(
            "| {model} | {control_label} | {layer_idx} | {component_kind} | {source_category} | {rows} | "
            "{strict_clean_answer_no_protocol} | {clean_answer_no_protocol} | {protocol_drift} | {protocol_rank1_removed} | {stop_rank_improved} | `{first_suffix_categories}` |".format(
                **row_fmt,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    best_controls = []
    for model_name in MODELS:
        summary_path = out_dir / f"phase904_{model_name}_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        scalar["selected_answer_drift_rows"] += int(overall.get("selected_answer_drift_rows") or 0)
        scalar["control_rows"] += int(overall.get("control_rows") or 0)
        scalar["candidate_count"] += int(overall.get("candidate_count") or 0)
        non_base = overall.get("non_baseline") or {}
        for key in [
            "clean_answer_no_protocol",
            "strict_clean_answer_no_protocol",
            "protocol_drift",
            "strict_protocol_drift",
            "protocol_rank1_removed",
            "stop_rank_improved",
            "stop_top1",
            "next_top_changed",
            "protocol_logit_reduced_strong",
        ]:
            scalar[f"non_base_{key}"] += int(non_base.get(key) or 0)
        best = dict(overall.get("best_control") or {})
        if best:
            best["model"] = summary.get("model")
            best_controls.append(best)
    best_controls.sort(
        key=lambda row: (
            row.get("strict_clean_answer_no_protocol") or 0,
            row.get("protocol_rank1_removed") or 0,
            row.get("stop_rank_improved") or 0,
            row.get("protocol_logit_reduced_strong") or 0,
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
        "best_controls": best_controls[:20],
    }
    p846.write_json(out_dir / "phase904_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase904_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="termination_control_candidate_search")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--phase903-round", default=PHASE903_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--max-suffix-tokens", type=int, default=8)
    parser.add_argument("--suppress-steps", type=int, default=2)
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
