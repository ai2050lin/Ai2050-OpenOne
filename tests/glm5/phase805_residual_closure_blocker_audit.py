#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase796_global_competitor_token_identity_audit as p796  # noqa: E402
import phase798_full_vocab_blocker_identity_anchor as p798  # noqa: E402
import phase799_blocker_field_causal_suppressor_localization as p799  # noqa: E402
import phase801_target_neutral_suppressor_causal_test as p801  # noqa: E402
import phase802_new_blocker_stabilization_dose_response as p802  # noqa: E402
import phase803_semantic_new_blocker_source_localization as p803  # noqa: E402
import phase804_true_semantic_suppressor_projection_search as p804  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase805_residual_closure_blocker_audit")
FORMAT_ECHO_CLASSES = {"high_frequency_or_format", "echo_token", "punctuation", "whitespace_or_newline", "number_or_symbol"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def safe_mean(values: list[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def safe_rate(values: list[Any]) -> float | None:
    vals = [bool(v) for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def parse_float_grid(text: str, fallback: list[float]) -> list[float]:
    vals: list[float] = []
    for part in (text or "").split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals or fallback


def merge_counter_dicts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get(key) or {})
    return dict(counter)


def residual_snapshot(
    args: argparse.Namespace,
    logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    target_answer: str,
) -> dict[str, Any]:
    full = p798.full_vocab_snapshot(
        args,
        logits,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        target_answer,
    )
    counts = dict(full.get("full_above_class_counts") or {})
    classified = int(full.get("full_above_classified_count") or 0)
    semantic = int(counts.get("semantic_or_lexical_competitor", 0))
    format_echo = sum(int(counts.get(cls, 0)) for cls in FORMAT_ECHO_CLASSES)
    case_value = int(counts.get("candidate_list_or_case_value", 0))
    other = max(classified - semantic - format_echo - case_value, 0)
    dominant = None
    if counts:
        dominant = max(counts.items(), key=lambda kv: (int(kv[1]), kv[0]))[0]
    return {
        "residual_full_above_count": full.get("full_above_count"),
        "residual_full_above_classified_count": classified,
        "residual_full_above_unclassified_count": full.get("full_above_unclassified_count"),
        "residual_class_counts": counts,
        "residual_class_entropy": full.get("full_above_class_entropy"),
        "residual_class_max_gap": full.get("full_above_class_max_gap"),
        "residual_class_top_tokens": full.get("full_above_class_top_tokens"),
        "residual_rank_window": full.get("full_rank_window", [])[: int(args.residual_rank_window_saved)],
        "residual_required_bias_to_clear_all": full.get("full_required_bias_to_clear_all"),
        "residual_identity_anchor_fragmented": full.get("identity_anchor_fragmented_full_vocab"),
        "residual_surface_target_variant_count_above": full.get("surface_target_variant_count_above"),
        "residual_surface_target_variant_max_gap": full.get("surface_target_variant_max_gap"),
        "residual_dominant_class": dominant,
        "residual_semantic_count": semantic,
        "residual_format_echo_count": format_echo,
        "residual_case_value_count": case_value,
        "residual_other_count": other,
        "residual_semantic_share": semantic / max(classified, 1),
        "residual_format_echo_share": format_echo / max(classified, 1),
        "residual_case_value_share": case_value / max(classified, 1),
        "residual_other_share": other / max(classified, 1),
    }


def label_residual(row: dict[str, Any], args: argparse.Namespace) -> str:
    if row.get("token_closure_gain"):
        return "token_closure"
    full_count = safe_float(row.get("residual_full_above_count")) or 0.0
    sem_share = safe_float(row.get("residual_semantic_share")) or 0.0
    fmt_share = safe_float(row.get("residual_format_echo_share")) or 0.0
    anchor = bool(row.get("residual_identity_anchor_fragmented"))
    still = safe_float(row.get("matched_semantic_still_above_target_rate")) or 0.0
    if full_count <= float(args.max_near_closure_blockers):
        return "near_closure_small_residual"
    if sem_share >= float(args.dominant_share_threshold):
        return "residual_semantic_still_dominant"
    if fmt_share >= float(args.dominant_share_threshold):
        return "residual_format_echo_dominant"
    if anchor:
        return "residual_identity_anchor_fragmented"
    if still > float(args.max_semantic_still_rate):
        return "matched_semantic_still_not_cleared"
    return "mixed_residual_field"


def audit_case_route(
    model,
    tokenizer,
    device,
    unembed: torch.Tensor,
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    route: dict[str, Any],
    component_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    target_id, contrast_id = p796.target_ids_from_row(tokenizer, case, source_row)
    recipient_variant = args.recipient_variant
    donor_variant = route["compare_variant"]
    recipient_prompt = p796.surface_prompt_for_variant(case, recipient_variant)
    donor_prompt = p796.surface_prompt_for_variant(case, donor_variant)
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    recipient_answer_pos = len(recipient_ids) - 1
    recipient_groups = p796.source_groups_for_prompt(tokenizer, recipient_prompt, case, recipient_ids)
    donor_groups = p796.source_groups_for_prompt(tokenizer, donor_prompt, case, donor_ids)
    recipient_candidate_ids = p796.candidate_position_ids(tokenizer, recipient_ids, recipient_groups)
    donor_candidate_ids = p796.candidate_position_ids(tokenizer, donor_ids, donor_groups)
    case_vals = p796.value_strings(case)
    recipient_state = p796.capture_answer_outputs_and_sources(model, tokenizer, device, recipient_prompt, component_keys)
    donor_state = p796.capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    route_components = selected_route_components(route, set(p796.parse_csv(args.route_component_kinds) or ["attn", "mlp"]), args.max_route_components)
    target_direction = unembed[int(target_id)].float().cpu()

    alpha0_projected, _alpha0_metrics = p802.projected_component_state_alpha(
        recipient_state,
        donor_state,
        route_components,
        target_direction,
        0.0,
    )
    alpha0_logits, alpha0_error = p801.run_logits_with_projected_route(model, device, recipient_ids, alpha0_projected, recipient_answer_pos)
    if alpha0_error or not alpha0_logits.numel():
        return [
            {
                "row_kind": "phase805_error",
                "model": args.model,
                "case_id": case["case_id"],
                "route_id": route["route_id"],
                "error": alpha0_error or "empty_alpha0_logits",
            }
        ]
    alpha0_semantic = p803.semantic_new_blocker_ids(
        args,
        recipient_state["logits"],
        alpha0_logits,
        target_id,
        contrast_id,
        recipient_ids,
        recipient_prompt,
        recipient_candidate_ids,
        case_vals,
    )
    semantic_direction = p804.semantic_direction_from_blockers(unembed, target_id, alpha0_semantic, args.semantic_direction_mode)
    alpha0_target_gain = float(alpha0_logits[int(target_id)].item() - recipient_state["logits"][int(target_id)].item())
    rows: list[dict[str, Any]] = []
    for target_alpha in parse_float_grid(args.target_alpha_grid, [0.0, 0.75]):
        for semantic_beta in parse_float_grid(args.semantic_beta_grid, [0.0, 1.0]):
            projected, projection_metrics = p804.projected_state_target_semantic(
                recipient_state,
                donor_state,
                route_components,
                target_direction,
                semantic_direction,
                target_alpha,
                semantic_beta,
            )
            if not projected:
                after_logits = torch.empty(0)
                error = "no_projected_components"
            else:
                after_logits, error = p801.run_logits_with_projected_route(model, device, recipient_ids, projected, recipient_answer_pos)
            row = p801.make_row(
                args,
                case,
                route,
                route_components,
                f"target_a{p802.alpha_label(target_alpha)}_sem_b{p802.alpha_label(semantic_beta)}",
                projection_metrics,
                recipient_variant,
                donor_variant,
                recipient_state["logits"],
                donor_state["logits"],
                after_logits,
                target_id,
                contrast_id,
                recipient_prompt,
                donor_prompt,
                recipient_ids,
                donor_ids,
                recipient_candidate_ids,
                donor_candidate_ids,
                case_vals,
                error,
            )
            row.update(
                {
                    "row_kind": "phase805_residual_closure_blocker_audit",
                    "model": args.model,
                    "round": args.round_name,
                    "case_id": case["case_id"],
                    "domain": case.get("domain"),
                    "relation": case.get("relation"),
                    "object": case.get("object"),
                    "target_answer": case.get("answer"),
                    "contrast_answer": case.get("contrast_answer"),
                    "route_id": route["route_id"],
                    "compare_variant": donor_variant,
                    "route_component_signature": p801.route_signature(route_components),
                    "target_direction_alpha": float(target_alpha),
                    "semantic_suppression_beta": float(semantic_beta),
                    "semantic_direction_mode": args.semantic_direction_mode,
                    "alpha0_semantic_new_blocker_count": len(alpha0_semantic),
                    "alpha0_target_logit_gain": alpha0_target_gain,
                }
            )
            if after_logits.numel():
                row.update(
                    p802.classify_new_blockers(
                        args,
                        recipient_state["logits"],
                        after_logits,
                        target_id,
                        contrast_id,
                        recipient_ids,
                        recipient_prompt,
                        recipient_candidate_ids,
                        case_vals,
                    )
                )
                row.update(
                    p803.matched_semantic_metrics(
                        recipient_state["logits"],
                        alpha0_logits,
                        after_logits,
                        target_id,
                        alpha0_semantic,
                    )
                )
                row.update(
                    residual_snapshot(
                        args,
                        after_logits,
                        target_id,
                        contrast_id,
                        recipient_ids,
                        recipient_prompt,
                        recipient_candidate_ids,
                        case_vals,
                        str(case.get("answer", "")),
                    )
                )
                row["target_gain_delta_vs_alpha0"] = (safe_float(row.get("target_logit_gain")) or 0.0) - alpha0_target_gain
                row["phase805_label"] = label_residual(row, args)
                row["phase805_boundary"] = (
                    "This phase audits residual full-vocabulary blockers after semantic-blocker projection. "
                    "It diagnoses residual closure gaps rather than searching new causal fibers."
                )
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase805_residual_closure_blocker_audit":
            groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_target_logit_gain": safe_mean([v.get("target_logit_gain") for v in vals]),
                "mean_target_gain_delta_vs_alpha0": safe_mean([v.get("target_gain_delta_vs_alpha0") for v in vals]),
                "mean_old_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_new_blocker_rate": safe_mean([v.get("new_blocker_rate") for v in vals]),
                "mean_matched_semantic_suppression": safe_mean([v.get("matched_semantic_true_suppression_vs_alpha0") for v in vals]),
                "mean_matched_semantic_still_above": safe_mean([v.get("matched_semantic_still_above_target_rate") for v in vals]),
                "mean_residual_full_above_count": safe_mean([v.get("residual_full_above_count") for v in vals]),
                "mean_residual_required_bias_to_clear_all": safe_mean([v.get("residual_required_bias_to_clear_all") for v in vals]),
                "mean_residual_semantic_share": safe_mean([v.get("residual_semantic_share") for v in vals]),
                "mean_residual_format_echo_share": safe_mean([v.get("residual_format_echo_share") for v in vals]),
                "mean_residual_case_value_share": safe_mean([v.get("residual_case_value_share") for v in vals]),
                "identity_anchor_fragmented_rate": safe_rate([v.get("residual_identity_anchor_fragmented") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "label_counts": dict(Counter(v.get("phase805_label") for v in vals)),
                "residual_class_counts": merge_counter_dicts(vals, "residual_class_counts"),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("token_closure_gain_rate") or 0.0,
            -(r.get("mean_residual_full_above_count") or 999999.0),
            -(r.get("mean_residual_required_bias_to_clear_all") or 999999.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 805,
        "title": "Residual Closure Blocker Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows if r.get("row_kind") == "phase805_residual_closure_blocker_audit"}),
        "n_routes": len(routes),
        "target_alpha_grid": parse_float_grid(args.target_alpha_grid, [0.0, 0.75]),
        "semantic_beta_grid": parse_float_grid(args.semantic_beta_grid, [0.0, 1.0]),
        "by_alpha_beta": group_rows(rows, ["model", "target_direction_alpha", "semantic_suppression_beta"]),
        "by_route_alpha_beta": group_rows(rows, ["model", "route_component_signature", "target_direction_alpha", "semantic_suppression_beta"])[:160],
        "by_case_alpha_beta": group_rows(rows, ["model", "case_id", "target_direction_alpha", "semantic_suppression_beta"])[:160],
        "strict_boundary": (
            "This phase audits the residual full-vocabulary blocker field after semantic blocker projection. "
            "It is diagnostic and does not itself discover a neuron-level residual closure mechanism."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p796.select_surface_cases(args.model, args)
    routes = p796.select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    component_keys = p796.component_keys_for_routes(routes)
    cmap = case_map_for(args)
    log(
        f"{args.model}/{args.round_name}: cases={len(selected)} routes={len(routes)} "
        f"target_alpha={args.target_alpha_grid} semantic_beta={args.semantic_beta_grid}"
    )
    if args.dry_run:
        return {"model": args.model, "round": args.round_name, "selected_cases": len(selected), "routes": routes}
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    setattr(args, "_tokenizer", tokenizer)
    setattr(args, "_token_text_cache", {})
    try:
        p796.enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: residual closure blocker audit {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        if hasattr(args, "_tokenizer"):
            delattr(args, "_tokenizer")
        if hasattr(args, "_token_text_cache"):
            delattr(args, "_token_text_cache")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl, routes)
    write_jsonl(out_dir / f"phase805_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase805_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_alpha_beta": summary["by_alpha_beta"][:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 805 Residual Closure Blocker Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: audits residual full-vocabulary blockers after semantic-blocker projection.",
        "- It diagnoses why token closure still fails; it is not a new neuron-level closure proof.",
        "",
        "## By Target Alpha And Semantic Beta",
        "",
        "| model | target alpha | semantic beta | rows | cases | old suppress | new rate | sem suppress | sem still | residual blockers | required bias | sem share | format/echo share | anchor frag | closure | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        rows = sorted(
            data.get("by_alpha_beta", []),
            key=lambda r: (
                safe_float(r.get("target_direction_alpha")) or 0.0,
                safe_float(r.get("semantic_suppression_beta")) or 0.0,
            ),
        )
        for row in rows:
            lines.append(
                f"| {model_name} | {fmt(row.get('target_direction_alpha'))} | {fmt(row.get('semantic_suppression_beta'))} | "
                f"{row.get('n')} | {row.get('case_n')} | {fmt(row.get('mean_old_blocker_suppression'))} | "
                f"{fmt(row.get('mean_new_blocker_rate'))} | {fmt(row.get('mean_matched_semantic_suppression'))} | "
                f"{fmt(row.get('mean_matched_semantic_still_above'))} | {fmt(row.get('mean_residual_full_above_count'))} | "
                f"{fmt(row.get('mean_residual_required_bias_to_clear_all'))} | {fmt(row.get('mean_residual_semantic_share'))} | "
                f"{fmt(row.get('mean_residual_format_echo_share'))} | {fmt(row.get('identity_anchor_fragmented_rate'))} | "
                f"{fmt(row.get('token_closure_gain_rate'))} | `{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## Residual Class Counts",
        "",
        "| model | target alpha | semantic beta | class | count |",
        "|---|---:|---:|---|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_alpha_beta", []):
            for cls, count in sorted((row.get("residual_class_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"| {model_name} | {fmt(row.get('target_direction_alpha'))} | {fmt(row.get('semantic_suppression_beta'))} | `{cls}` | {count} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {"phase": 805, "round": round_name, "status": "missing", "model_summaries": {}, "models": []}
    for model_name in MODELS:
        path = out_dir / f"phase805_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase805_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase805_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p802.build_parser()
    parser.add_argument("--target-alpha-grid", default="0,0.75")
    parser.add_argument("--semantic-beta-grid", default="0,1")
    parser.add_argument("--semantic-direction-mode", choices=["semantic_mean", "semantic_minus_target"], default="semantic_minus_target")
    parser.add_argument("--max-semantic-new-blockers", type=int, default=64)
    parser.add_argument("--residual-rank-window-saved", type=int, default=24)
    parser.add_argument("--dominant-share-threshold", type=float, default=0.50)
    parser.add_argument("--max-near-closure-blockers", type=float, default=5.0)
    parser.add_argument("--max-semantic-still-rate", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    result = run_model(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
