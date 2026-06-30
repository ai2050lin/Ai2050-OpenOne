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
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import normalize_token_text  # noqa: E402


OUT_ROOT = Path("results/glm5_phase798_full_vocab_blocker_identity_anchor")
RESULT_ROOT = Path("tests/result/phase798_full_vocab_blocker_identity_anchor")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def safe_rate(values: list[Any]) -> float | None:
    vals = [bool(v) for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def norm(value: Any) -> str:
    return normalize_token_text("" if value is None else str(value)).strip().lower()


def cached_token_text(args: argparse.Namespace, token_id: int) -> str:
    cache = getattr(args, "_token_text_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_token_text_cache", cache)
    tid = int(token_id)
    if tid not in cache:
        cache[tid] = p796.token_text(args._tokenizer, tid)
    return cache[tid]


def entropy(counts: dict[str, int]) -> float | None:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return None
    out = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0:
            out -= p * math.log(p)
    return out


def full_vocab_snapshot(
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
    vals = logits.detach().float().cpu()
    target_logit = float(vals[int(target_id)].item())
    above_mask = vals > target_logit
    above_mask[int(target_id)] = False
    above_ids = torch.nonzero(above_mask, as_tuple=False).flatten()
    above_count = int(above_ids.numel())
    if above_count:
        above_vals = vals[above_ids]
        sorted_vals, sorted_idx = torch.sort(above_vals, descending=True)
        sorted_ids = above_ids[sorted_idx]
    else:
        sorted_vals = torch.empty(0)
        sorted_ids = torch.empty(0, dtype=torch.long)

    classify_limit = max(0, int(args.max_full_above_classify))
    if classify_limit == 0:
        classify_n = above_count
    else:
        classify_n = min(above_count, classify_limit)

    class_counts: Counter[str] = Counter()
    class_max_gap: dict[str, float] = {}
    class_top_token: dict[str, dict[str, Any]] = {}
    surface_variants: list[dict[str, Any]] = []
    target_norm = norm(target_answer)
    rank_window: list[dict[str, Any]] = []

    for offset in range(classify_n):
        tid = int(sorted_ids[offset].item())
        logit = float(sorted_vals[offset].item())
        text = cached_token_text(args, tid)
        cls = p796.classify_competitor(
            args._tokenizer,
            tid,
            text,
            int(target_id),
            int(contrast_id),
            prompt_ids,
            prompt_text,
            candidate_ids,
            case_values,
        )
        gap = float(logit - target_logit)
        class_counts[cls] += 1
        class_max_gap[cls] = max(float(class_max_gap.get(cls, -float("inf"))), gap)
        if cls not in class_top_token:
            class_top_token[cls] = {
                "token_id": tid,
                "token_text": text,
                "gap_above_target": gap,
                "global_rank": offset + 1,
            }
        if target_norm and norm(text) == target_norm and tid != int(target_id):
            surface_variants.append(
                {
                    "token_id": tid,
                    "token_text": text,
                    "logit": logit,
                    "gap_above_target": gap,
                    "global_rank": offset + 1,
                }
            )
        if offset < int(args.full_rank_window):
            rank_window.append(
                {
                    "global_rank": offset + 1,
                    "token_id": tid,
                    "token_text": text,
                    "token_text_norm": normalize_token_text(text),
                    "class": cls,
                    "logit": logit,
                    "gap_above_target": gap,
                }
            )

    single_class_closers = [cls for cls, count in class_counts.items() if int(count) == above_count and above_count > 0]
    return {
        "target_logit": target_logit,
        "full_above_count": above_count,
        "full_above_classification_complete": classify_n == above_count,
        "full_above_classified_count": classify_n,
        "full_above_unclassified_count": max(above_count - classify_n, 0),
        "full_above_class_counts": dict(class_counts),
        "full_above_class_entropy": entropy(dict(class_counts)),
        "full_above_class_max_gap": class_max_gap,
        "full_above_class_top_tokens": class_top_token,
        "full_single_class_closure_possible": bool(single_class_closers),
        "full_single_class_closure_classes": single_class_closers,
        "full_required_bias_to_clear_all": float(sorted_vals[0].item() - target_logit) if above_count else 0.0,
        "full_rank_window": rank_window,
        "surface_target_variant_count_above": len(surface_variants),
        "surface_target_variant_max_gap": max((float(v["gap_above_target"]) for v in surface_variants), default=None),
        "surface_target_variant_token_texts": [v["token_text"] for v in surface_variants[: int(args.max_surface_variants_saved)]],
        "surface_target_variant_token_ids": [int(v["token_id"]) for v in surface_variants[: int(args.max_surface_variants_saved)]],
        "surface_target_identity_rank": len(surface_variants) + 1,
        "identity_anchor_fragmented_full_vocab": bool(surface_variants),
    }


def enhanced_make_audit_row(*call_args: Any, **call_kwargs: Any) -> dict[str, Any]:
    row = p796._phase798_original_make_audit_row(*call_args, **call_kwargs)
    (
        args,
        case,
        _route,
        _route_components,
        _meta,
        _ladder_id,
        _source_group,
        _paired_count,
        _recipient_variant,
        _donor_variant,
        recipient_logits,
        _donor_logits,
        after_logits,
        target_id,
        contrast_id,
        recipient_prompt,
        _donor_prompt,
        recipient_ids,
        _donor_ids,
        recipient_candidate_ids,
        _donor_candidate_ids,
        case_values,
        _error,
    ) = call_args
    target_answer = str(case.get("answer", ""))
    recipient_full = full_vocab_snapshot(
        args,
        recipient_logits,
        target_id,
        contrast_id,
        recipient_ids,
        recipient_prompt,
        recipient_candidate_ids,
        case_values,
        target_answer,
    )
    if after_logits.numel():
        after_full = full_vocab_snapshot(
            args,
            after_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_values,
            target_answer,
        )
    else:
        after_full = {}
    after_topk_above = [
        r
        for r in row.get("after_topk", [])
        if (not r.get("is_target")) and r.get("gap_above_target") is not None and float(r["gap_above_target"]) > 0
    ]
    row.update(
        {
            "row_kind": "phase798_full_vocab_blocker_identity_anchor",
            "phase798_boundary": (
                "This row reruns the Phase 796 intervention path, but extracts full-vocabulary blockers above the target "
                "token when possible. It is still a logit-space audit, not direct localization of a biological mechanism."
            ),
            "recipient_full_above_count": recipient_full.get("full_above_count"),
            "recipient_full_above_class_counts": recipient_full.get("full_above_class_counts"),
            "recipient_identity_anchor_fragmented_full_vocab": recipient_full.get("identity_anchor_fragmented_full_vocab"),
            "recipient_surface_target_variant_count_above": recipient_full.get("surface_target_variant_count_above"),
            "after_full_above_count": after_full.get("full_above_count"),
            "after_full_above_classification_complete": after_full.get("full_above_classification_complete"),
            "after_full_above_classified_count": after_full.get("full_above_classified_count"),
            "after_full_above_unclassified_count": after_full.get("full_above_unclassified_count"),
            "after_full_above_class_counts": after_full.get("full_above_class_counts"),
            "after_full_above_class_entropy": after_full.get("full_above_class_entropy"),
            "after_full_above_class_max_gap": after_full.get("full_above_class_max_gap"),
            "after_full_above_class_top_tokens": after_full.get("full_above_class_top_tokens"),
            "after_full_required_bias_to_clear_all": after_full.get("full_required_bias_to_clear_all"),
            "after_full_single_class_closure_possible": after_full.get("full_single_class_closure_possible"),
            "after_full_single_class_closure_classes": after_full.get("full_single_class_closure_classes"),
            "after_full_rank_window": after_full.get("full_rank_window", []),
            "after_full_surface_target_variant_count_above": after_full.get("surface_target_variant_count_above"),
            "after_full_surface_target_variant_max_gap": after_full.get("surface_target_variant_max_gap"),
            "after_full_surface_target_variant_token_texts": after_full.get("surface_target_variant_token_texts"),
            "after_full_surface_target_variant_token_ids": after_full.get("surface_target_variant_token_ids"),
            "after_full_surface_target_identity_rank": after_full.get("surface_target_identity_rank"),
            "after_identity_anchor_fragmented_full_vocab": after_full.get("identity_anchor_fragmented_full_vocab"),
            "topk_above_count_saved": len(after_topk_above),
            "full_minus_topk_above_count": (
                int(after_full.get("full_above_count", 0)) - len(after_topk_above) if after_full else None
            ),
            "topk_blind_spot_resolved": after_full.get("full_above_count") is not None,
        }
    )
    return row


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        class_counts = Counter()
        for row in vals:
            class_counts.update(row.get("after_full_above_class_counts") or {})
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "mean_after_target_rank": safe_mean([v.get("after_target_rank") for v in vals]),
                "mean_after_full_above_count": safe_mean([v.get("after_full_above_count") for v in vals]),
                "mean_full_minus_topk_above_count": safe_mean([v.get("full_minus_topk_above_count") for v in vals]),
                "mean_after_full_required_bias_to_clear_all": safe_mean([v.get("after_full_required_bias_to_clear_all") for v in vals]),
                "mean_delta_global_margin_vs_recipient": safe_mean([v.get("delta_global_margin_vs_recipient") for v in vals]),
                "mean_target_logit_gain_vs_recipient": safe_mean([v.get("target_logit_gain_vs_recipient") for v in vals]),
                "mean_full_class_entropy": safe_mean([v.get("after_full_above_class_entropy") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "full_single_class_closure_possible_rate": safe_rate([v.get("after_full_single_class_closure_possible") for v in vals]),
                "identity_anchor_fragmented_full_vocab_rate": safe_rate([v.get("after_identity_anchor_fragmented_full_vocab") for v in vals]),
                "mean_surface_target_identity_rank": safe_mean([v.get("after_full_surface_target_identity_rank") for v in vals]),
                "full_classification_complete_rate": safe_rate([v.get("after_full_above_classification_complete") for v in vals]),
                "full_above_class_counts": dict(class_counts),
            }
        )
        payload["full_vocab_closure_pressure_score"] = max(payload["mean_delta_global_margin_vs_recipient"] or 0.0, 0.0) / (
            1.0 + max(payload["mean_after_full_above_count"] or 0.0, 0.0)
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("token_closure_gain_rate") or 0.0,
            r.get("full_vocab_closure_pressure_score") or 0.0,
            -(r.get("mean_after_full_above_count") or 0.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], ladders: list[str], source_groups: list[str]) -> dict[str, Any]:
    by_model = group_rows(rows, ["model"])
    by_ladder = group_rows(rows, ["model", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "ladder_id", "source_group"])
    by_case = group_rows(rows, ["model", "case_id", "ladder_id", "source_group"])
    class_counts = Counter()
    for row in rows:
        class_counts.update(row.get("after_full_above_class_counts") or {})
    return {
        "phase": 798,
        "title": "Full-Vocabulary Blocker Extraction and Identity-Anchor Candidate Localization",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "top_k": args.top_k,
        "full_rank_window": args.full_rank_window,
        "max_full_above_classify": args.max_full_above_classify,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "ladders": ladders,
        "source_groups": source_groups,
        "full_above_class_counts": dict(class_counts),
        "by_model": by_model,
        "by_ladder": by_ladder,
        "by_case": by_case,
        "top_full_vocab_effects": by_ladder[:80],
        "strict_boundary": (
            "This phase extracts all observed vocabulary tokens above the target from logits, or marks the extraction incomplete "
            "if a configured safety cap is hit. It does not prove that any internal unit is the suppressor."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = p796.select_surface_cases(args.model, args)
    routes = p796.select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    specs = p796.subspace_specs(p796.parse_csv(args.subspace_modes), p796.parse_budgets(args.budgets))
    source_groups = p796.source_groups_for(args)
    ladders = p796.parse_csv(args.ladders) or p796.DEFAULT_LADDERS
    route_allowed_kinds = set(p796.parse_csv(args.route_component_kinds) or ["attn", "mlp"])
    log(
        f"{args.model}/{args.round_name}: cases={len(selected)} routes={len(routes)} specs={len(specs)} "
        f"ladders={ladders} groups={source_groups} top_k={args.top_k} rank_window={args.full_rank_window}"
    )
    cmap = p796.case_map_for(args)
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "source_groups": source_groups,
            "ladders": ladders,
        }
    component_keys = p796.component_keys_for_routes(routes)
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    setattr(args, "_tokenizer", tokenizer)
    setattr(args, "_token_text_cache", {})
    if not hasattr(p796, "_phase798_original_make_audit_row"):
        p796._phase798_original_make_audit_row = p796.make_audit_row
    p796.make_audit_row = enhanced_make_audit_row
    try:
        p796.enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = p796.lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(
                    p796.audit_case_route(
                        model,
                        tokenizer,
                        device,
                        unembed,
                        args,
                        case,
                        source_row,
                        route,
                        component_keys,
                        specs,
                        ladders,
                        source_groups,
                        route_allowed_kinds,
                    )
                )
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: full-vocab blocker audit {ci}/{len(selected)} cases; rows={len(rows)}")
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
    summary = summarize(rows, args, attn_impl, routes, ladders, source_groups)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase798_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase798_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_model": summary["by_model"][:3],
                "top_full_vocab_effects": summary["top_full_vocab_effects"][:5],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def build_atlas(payload: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(node_id: str, node_type: str, **attrs: Any) -> None:
        nodes[node_id] = {**nodes.get(node_id, {}), "id": node_id, "type": node_type, **attrs}

    task = "phase798:full_vocab_blocker_identity_anchor"
    node(task, "task", label="Phase 798 full-vocabulary blocker and identity-anchor audit")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for row in summary.get("by_model", []):
            metrics_node = f"{model_name}:full_vocab_competition"
            node(metrics_node, "competition_field", label=f"{model_name} full-vocab blockers", metrics=row)
            edges.append({"id": f"{model_name}:competition", "source": model_node, "target": metrics_node, "type": "has_competition_field", "metrics": row})
        for cls, count in (summary.get("full_above_class_counts") or {}).items():
            cls_node = f"blocker_class:{cls}"
            node(cls_node, "blocker_class", label=cls)
            edges.append({"id": f"{model_name}:blocker:{cls}", "source": model_node, "target": cls_node, "type": "blocked_by_full_vocab_class", "weight": count})
    return {"schema_version": "atlas_graph_v1", "phase": 798, "graph": {"nodes": list(nodes.values()), "edges": edges}}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 798 Full-Vocabulary Blocker and Identity-Anchor Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: reruns Phase 796 intervention paths and extracts full-vocabulary blockers above the target token.",
        "- This is still a logit-space audit; internal suppressor localization remains a later phase.",
        "",
        "## By Model",
        "",
        "| model | rows | cases | target rank | full blockers | hidden outside saved top-k | single-class close | identity fragmented | token gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        row = (data.get("by_model") or [{}])[0]
        lines.append(
            f"| {model_name} | {row.get('n')} | {row.get('case_n')} | {fmt(row.get('mean_after_target_rank'))} | "
            f"{fmt(row.get('mean_after_full_above_count'))} | {fmt(row.get('mean_full_minus_topk_above_count'))} | "
            f"{fmt(row.get('full_single_class_closure_possible_rate'))} | "
            f"{fmt(row.get('identity_anchor_fragmented_full_vocab_rate'))} | {fmt(row.get('token_closure_gain_rate'))} |"
        )
    lines += [
        "",
        "## Full-Vocabulary Blocker Class Counts",
        "",
        "| model | class | count |",
        "|---|---|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for cls, count in sorted((data.get("full_above_class_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {model_name} | `{cls}` | {count} |")
    lines += [
        "",
        "## Top Full-Vocab Effects",
        "",
        "| model | selection | ladder | subspace | source group | rows | full blockers | delta global | target gain | identity fragmented | single-class close |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_full_vocab_effects", [])[:24]:
            lines.append(
                f"| {model_name} | `{row.get('source_selection_kind')}` | `{row.get('ladder_id')}` | `{row.get('subspace_mode')}` | `{row.get('source_group')}` | "
                f"{row.get('n')} | {fmt(row.get('mean_after_full_above_count'))} | {fmt(row.get('mean_delta_global_margin_vs_recipient'))} | "
                f"{fmt(row.get('mean_target_logit_gain_vs_recipient'))} | {fmt(row.get('identity_anchor_fragmented_full_vocab_rate'))} | "
                f"{fmt(row.get('full_single_class_closure_possible_rate'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase798_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 798,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    for root in (OUT_ROOT / round_name, RESULT_ROOT / round_name):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "phase798_cross_model_summary.json", payload)
        write_json(root / "phase798_atlas_graph.json", build_atlas(payload))
        write_markdown(root / "phase798_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p796.build_parser()
    parser.add_argument("--full-rank-window", type=int, default=128)
    parser.add_argument("--max-full-above-classify", type=int, default=20000)
    parser.add_argument("--max-surface-variants-saved", type=int, default=32)
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
