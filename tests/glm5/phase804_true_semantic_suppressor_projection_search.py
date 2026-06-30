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
import phase799_blocker_field_causal_suppressor_localization as p799  # noqa: E402
import phase801_target_neutral_suppressor_causal_test as p801  # noqa: E402
import phase802_new_blocker_stabilization_dose_response as p802  # noqa: E402
import phase803_semantic_new_blocker_source_localization as p803  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase804_true_semantic_suppressor_projection_search")


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


def orthogonalize(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    v = vec.float()
    b = basis.float()
    denom = float(torch.dot(b, b).item())
    if denom <= 1e-9:
        return v
    return v - float(torch.dot(v, b).item() / denom) * b


def project_delta_target_semantic(
    delta: torch.Tensor,
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target = target_direction.float()
    delta_f = delta.float()
    target_denom = float(torch.dot(target, target).item())
    if target_denom <= 1e-9:
        target_parallel = torch.zeros_like(delta_f)
    else:
        target_parallel = float(torch.dot(delta_f, target).item() / target_denom) * target
    target_neutral = delta_f - target_parallel
    base = target_neutral + float(target_alpha) * target_parallel

    sem = orthogonalize(semantic_direction.float(), target)
    sem_denom = float(torch.dot(sem, sem).item())
    if sem_denom <= 1e-9:
        sem_parallel = torch.zeros_like(base)
    else:
        sem_parallel = float(torch.dot(base, sem).item() / sem_denom) * sem
    projected = base - float(semantic_beta) * sem_parallel
    return projected, {
        "target_direction_alpha": float(target_alpha),
        "semantic_suppression_beta": float(semantic_beta),
        "direct_target_component_before": float(torch.dot(delta_f, target).item()),
        "direct_target_component_after": float(torch.dot(projected.float(), target).item()),
        "semantic_component_before": float(torch.dot(base.float(), sem).item()),
        "semantic_component_after": float(torch.dot(projected.float(), sem).item()),
        "delta_norm": float(delta_f.norm().item()),
        "target_parallel_norm": float(target_parallel.norm().item()),
        "target_neutral_norm": float(target_neutral.norm().item()),
        "semantic_parallel_norm": float(sem_parallel.norm().item()),
        "projected_norm": float(projected.float().norm().item()),
    }


def projected_state_target_semantic(
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    route_components: list[dict[str, Any]],
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    projected: dict[tuple[str, int], torch.Tensor] = {}
    metrics: dict[str, list[float]] = defaultdict(list)
    for comp in route_components:
        key = (str(comp["component_kind"]), int(comp["layer"]))
        rec_vec = recipient_state.get("components", {}).get(key)
        donor_vec = donor_state.get("components", {}).get(key)
        if rec_vec is None or donor_vec is None:
            continue
        delta = donor_vec.float() - rec_vec.float()
        delta_projected, meta = project_delta_target_semantic(
            delta,
            target_direction,
            semantic_direction,
            target_alpha,
            semantic_beta,
        )
        projected[key] = (rec_vec.float() + delta_projected).detach().cpu()
        for k, v in meta.items():
            metrics[k].append(v)
    summary = {f"mean_{k}": safe_mean(vals) for k, vals in metrics.items()}
    summary["target_direction_alpha"] = float(target_alpha)
    summary["semantic_suppression_beta"] = float(semantic_beta)
    return projected, summary


def semantic_direction_from_blockers(
    unembed: torch.Tensor,
    target_id: int,
    semantic_rows: list[dict[str, Any]],
    mode: str,
) -> torch.Tensor:
    if not semantic_rows:
        return unembed[int(target_id)].float().cpu()
    ids = torch.tensor([int(x["token_id"]) for x in semantic_rows], dtype=torch.long)
    sem_mean = unembed[ids].float().mean(dim=0).cpu()
    target = unembed[int(target_id)].float().cpu()
    if mode == "semantic_mean":
        return sem_mean
    if mode == "semantic_minus_target":
        return sem_mean - target
    raise ValueError(f"unknown semantic direction mode: {mode}")


def label_row(row: dict[str, Any], args: argparse.Namespace) -> str:
    true_supp = safe_float(row.get("matched_semantic_true_suppression_vs_alpha0")) or 0.0
    still_rate = safe_float(row.get("matched_semantic_still_above_target_rate")) or 0.0
    target_gain_delta = safe_float(row.get("target_gain_delta_vs_alpha0")) or 0.0
    old_supp = safe_float(row.get("baseline_blocker_mean_suppression")) or 0.0
    if row.get("token_closure_gain"):
        return "token_closure"
    if (
        true_supp >= args.min_true_semantic_suppression
        and still_rate <= args.max_semantic_still_rate
        and abs(target_gain_delta) <= args.max_target_gain_delta
        and old_supp >= args.min_old_suppression
    ):
        return "true_semantic_suppressor_candidate_strict"
    if true_supp >= args.min_true_semantic_suppression and abs(target_gain_delta) <= args.max_target_gain_delta:
        return "semantic_logit_suppression_without_closure"
    if true_supp >= args.min_true_semantic_suppression:
        return "semantic_logit_suppression_but_target_shifted"
    if still_rate <= args.max_semantic_still_rate and true_supp < args.min_true_semantic_suppression:
        return "below_target_without_true_semantic_suppression"
    return "weak_or_mixed"


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

    alpha0_projected, alpha0_metrics = p802.projected_component_state_alpha(
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
                "row_kind": "phase804_error",
                "model": args.model,
                "case_id": case["case_id"],
                "route_id": route["route_id"],
                "error": alpha0_error or "empty_alpha0_logits",
            }
        ]
    full_alpha0_semantic = p803.semantic_new_blocker_ids(
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
    semantic_direction = semantic_direction_from_blockers(unembed, target_id, full_alpha0_semantic, args.semantic_direction_mode)
    alpha0_target_gain = float(alpha0_logits[int(target_id)].item() - recipient_state["logits"][int(target_id)].item())
    rows: list[dict[str, Any]] = []
    for target_alpha in parse_float_grid(args.target_alpha_grid, [0.0, 0.75]):
        for semantic_beta in parse_float_grid(args.semantic_beta_grid, [0.0, 1.0]):
            projected, projection_metrics = projected_state_target_semantic(
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
                {**alpha0_metrics, **projection_metrics},
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
                    "row_kind": "phase804_true_semantic_suppressor_search",
                    "model": args.model,
                    "round": args.round_name,
                    "case_id": case["case_id"],
                    "domain": case.get("domain"),
                    "relation": case.get("relation"),
                    "object": case.get("object"),
                    "target_answer": case.get("answer"),
                    "route_id": route["route_id"],
                    "route_component_signature": p801.route_signature(route_components),
                    "target_direction_alpha": float(target_alpha),
                    "semantic_suppression_beta": float(semantic_beta),
                    "semantic_direction_mode": args.semantic_direction_mode,
                    "alpha0_semantic_new_blocker_count": len(full_alpha0_semantic),
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
                        full_alpha0_semantic,
                    )
                )
                row["target_gain_delta_vs_alpha0"] = (safe_float(row.get("target_logit_gain")) or 0.0) - alpha0_target_gain
                row["phase804_true_semantic_suppression_score"] = max(
                    safe_float(row.get("matched_semantic_true_suppression_vs_alpha0")) or 0.0,
                    0.0,
                ) * max(1.0 - (safe_float(row.get("matched_semantic_still_above_target_rate")) or 0.0), 0.0)
                row["phase804_label"] = label_row(row, args)
                row["phase804_boundary"] = (
                    "This phase projects route deltas away from the matched semantic blocker readout direction. "
                    "It is a route-level direct-readout subspace test, not a neuron-level proof."
                )
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase804_true_semantic_suppressor_search":
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
                "mean_alpha0_semantic_new_count": safe_mean([v.get("alpha0_semantic_new_blocker_count") for v in vals]),
                "mean_matched_semantic_delta_vs_alpha0": safe_mean([v.get("matched_semantic_mean_logit_delta_vs_alpha0") for v in vals]),
                "mean_true_semantic_suppression_vs_alpha0": safe_mean([v.get("matched_semantic_true_suppression_vs_alpha0") for v in vals]),
                "mean_matched_semantic_gap_vs_target": safe_mean([v.get("matched_semantic_mean_gap_vs_target") for v in vals]),
                "mean_matched_semantic_still_above_target_rate": safe_mean([v.get("matched_semantic_still_above_target_rate") for v in vals]),
                "mean_true_semantic_suppression_score": safe_mean([v.get("phase804_true_semantic_suppression_score") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "label_counts": dict(Counter(v.get("phase804_label") for v in vals)),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("mean_true_semantic_suppression_score") or -999.0,
            -(r.get("mean_matched_semantic_still_above_target_rate") or 999.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 804,
        "title": "True Semantic Suppressor Projection Search",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows if r.get("row_kind") == "phase804_true_semantic_suppressor_search"}),
        "n_routes": len(routes),
        "target_alpha_grid": parse_float_grid(args.target_alpha_grid, [0.0, 0.75]),
        "semantic_beta_grid": parse_float_grid(args.semantic_beta_grid, [0.0, 1.0]),
        "semantic_direction_mode": args.semantic_direction_mode,
        "by_alpha_beta": group_rows(rows, ["model", "target_direction_alpha", "semantic_suppression_beta"]),
        "by_route_alpha_beta": group_rows(rows, ["model", "route_component_signature", "target_direction_alpha", "semantic_suppression_beta"])[:160],
        "strict_boundary": (
            "This phase tests whether removing matched semantic blocker readout direction from route deltas lowers "
            "the same semantic blocker logits. It does not establish neuron-level causal fiber closure."
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
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "target_alpha_grid": args.target_alpha_grid,
            "semantic_beta_grid": args.semantic_beta_grid,
        }
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
                log(f"{args.model}: true semantic suppressor projection search {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase804_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase804_{args.model}_summary.json", summary)
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
        f"# Phase 804 True Semantic Suppressor Projection Search ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: removes matched semantic blocker readout direction from route deltas.",
        "- A candidate must lower matched semantic blocker logits; lower new-blocker rate alone is insufficient.",
        "",
        "## By Target Alpha And Semantic Beta",
        "",
        "| model | target alpha | semantic beta | rows | cases | target gain | target gain vs a0 | old suppress | new rate | true semantic suppress | still above | closure | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, data in payload.get("model_summaries", {}).items():
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
                f"{row.get('n')} | {row.get('case_n')} | {fmt(row.get('mean_target_logit_gain'))} | "
                f"{fmt(row.get('mean_target_gain_delta_vs_alpha0'))} | {fmt(row.get('mean_old_blocker_suppression'))} | "
                f"{fmt(row.get('mean_new_blocker_rate'))} | {fmt(row.get('mean_true_semantic_suppression_vs_alpha0'))} | "
                f"{fmt(row.get('mean_matched_semantic_still_above_target_rate'))} | {fmt(row.get('token_closure_gain_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": 804,
        "round": round_name,
        "status": "missing",
        "model_summaries": {},
        "models": [],
    }
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        path = out_dir / f"phase804_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == 3 else "partial"
    write_json(out_dir / "phase804_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase804_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p802.build_parser()
    parser.add_argument("--target-alpha-grid", default="0,0.75")
    parser.add_argument("--semantic-beta-grid", default="0,1")
    parser.add_argument("--semantic-direction-mode", choices=["semantic_mean", "semantic_minus_target"], default="semantic_minus_target")
    parser.add_argument("--max-semantic-new-blockers", type=int, default=64)
    parser.add_argument("--min-true-semantic-suppression", type=float, default=0.20)
    parser.add_argument("--max-semantic-still-rate", type=float, default=0.20)
    parser.add_argument("--max-target-gain-delta", type=float, default=0.50)
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
