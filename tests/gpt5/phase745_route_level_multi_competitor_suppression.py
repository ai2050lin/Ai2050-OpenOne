#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, get_unembed, prepare_joint_install  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import first_token_id, load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, combine_installers  # noqa: E402
from phase742_combined_threshold_component_closure import install_combo_add, load_phase741_ranked_candidates  # noqa: E402
from phase743_competitor_format_suppression_audit import (  # noqa: E402
    competitor_direction,
    forward_with_final_delta,
    margin,
    suppression_needed,
    taxonomy_context,
    top_vocab_with_classes,
)


OUT_ROOT = Path("results/glm5_phase745_route_level_multi_competitor_suppression")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_combo_installer(
    model,
    install_joint: Callable[[], list[Any]],
    combo: list[dict[str, Any]],
    deltas: dict[str, torch.Tensor],
) -> Callable[[], list[Any]]:
    return combine_installers(install_joint, install_combo_add(model, combo, deltas))


def unique_competitors(vocab: list[dict[str, Any]], donor_id: int, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in vocab:
        tid = int(item["token_id"])
        if tid == int(donor_id) or tid in seen:
            continue
        out.append(item)
        seen.add(tid)
        if len(out) >= limit:
            break
    return out


def group_by_class(competitors: list[dict[str, Any]], max_classes: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for item in competitors:
        cls = item["class"]
        if cls not in groups:
            order.append(cls)
        groups[cls].append(item)
    out: list[dict[str, Any]] = []
    for cls in order[:max_classes]:
        out.append({"class": cls, "tokens": groups[cls], "representative": groups[cls][0]})
    return out


def route_margins(logits: torch.Tensor, donor_id: int, groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    donor_logit = float(logits[int(donor_id)].item())
    out: dict[str, dict[str, Any]] = {}
    for group in groups:
        vals = []
        for item in group["tokens"]:
            tid = int(item["token_id"])
            vals.append((float(logits[tid].item()), tid, item["token_text"]))
        if not vals:
            continue
        max_logit, max_id, max_text = max(vals, key=lambda x: x[0])
        out[group["class"]] = {
            "route_class": group["class"],
            "max_token_id": max_id,
            "max_token_text": max_text,
            "max_logit": max_logit,
            "margin_donor_vs_route_max": donor_logit - max_logit,
            "token_count": len(vals),
        }
    return out


def individual_suppression_delta(
    logits: torch.Tensor,
    unembed: torch.Tensor,
    donor_id: int,
    token_ids: list[int],
    scale: float,
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    delta: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for token_id in token_ids:
        tid = int(token_id)
        if tid == int(donor_id) or tid in seen:
            continue
        seen.add(tid)
        direction = competitor_direction(unembed, tid, donor_id)
        alpha = suppression_needed(logits, unembed, tid, donor_id, direction)
        if direction is None or alpha is None or alpha <= 0:
            records.append({"token_id": tid, "alpha_needed": alpha, "used": False})
            continue
        part = -float(scale) * float(alpha) * direction
        delta = part if delta is None else delta + part
        records.append({"token_id": tid, "alpha_needed": float(alpha), "used": True})
    return delta, records


def centroid_suppression_delta(
    logits: torch.Tensor,
    unembed: torch.Tensor,
    donor_id: int,
    groups: list[dict[str, Any]],
    scale: float,
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    delta: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    donor_vec = unembed[int(donor_id)]
    donor_logit = float(logits[int(donor_id)].item())
    for group in groups:
        token_ids = [int(t["token_id"]) for t in group["tokens"] if int(t["token_id"]) != int(donor_id)]
        if not token_ids:
            continue
        mean_vec = torch.stack([unembed[tid] for tid in token_ids], dim=0).mean(dim=0)
        direction = mean_vec - donor_vec
        norm = float(torch.linalg.vector_norm(direction).item())
        if norm <= 1e-8:
            records.append({"route_class": group["class"], "used": False, "alpha_needed": None, "reason": "zero_direction"})
            continue
        direction = direction / norm
        max_id = max(token_ids, key=lambda tid: float(logits[tid].item()))
        gap = float(logits[max_id].item()) - donor_logit
        denom = float(torch.dot(unembed[max_id] - donor_vec, direction).item())
        if gap <= 0 or denom <= 1e-8:
            records.append({"route_class": group["class"], "used": False, "alpha_needed": 0.0 if gap <= 0 else None, "max_token_id": max_id})
            continue
        alpha = gap / denom
        part = -float(scale) * float(alpha) * direction
        delta = part if delta is None else delta + part
        records.append(
            {
                "route_class": group["class"],
                "used": True,
                "alpha_needed": float(alpha),
                "max_token_id": max_id,
                "token_count": len(token_ids),
            }
        )
    return delta, records


def make_condition_specs(
    competitors: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    max_topk_tokens: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"condition": "joint_add_topK", "strategy": "none", "tokens": [], "groups": []}]
    if competitors:
        specs.append({"condition": "suppress_current_top", "strategy": "individual", "tokens": [competitors[0]], "groups": []})
        top_class = competitors[0]["class"]
        same_class = [c for c in competitors if c["class"] == top_class]
        specs.append({"condition": "suppress_current_top_class", "strategy": "individual", "tokens": same_class, "groups": []})
    reps = [g["representative"] for g in groups]
    specs.append({"condition": "suppress_route_representatives", "strategy": "individual", "tokens": reps, "groups": []})
    specs.append({"condition": "suppress_route_centroids", "strategy": "centroid", "tokens": [], "groups": groups})
    specs.append({"condition": "suppress_all_topk_competitors", "strategy": "individual", "tokens": competitors[:max_topk_tokens], "groups": []})
    for group in groups:
        specs.append(
            {
                "condition": f"suppress_class:{group['class']}",
                "strategy": "individual",
                "tokens": group["tokens"],
                "groups": [],
            }
        )
    return specs


def evaluate_condition(
    model,
    tokenizer,
    device,
    recipient_ids: list[int],
    install_combo: Callable[[], list[Any]],
    logits_base: torch.Tensor,
    unembed: torch.Tensor,
    donor_id: int,
    recipient_id: int,
    groups: list[dict[str, Any]],
    ctx: dict[str, Any],
    top_k_vocab: int,
    spec: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    if spec["strategy"] == "none" or scale == 0.0:
        logits = logits_base
        final_vec = None
        delta_norm = 0.0
        suppression_records: list[dict[str, Any]] = []
    elif spec["strategy"] == "individual":
        token_ids = [int(t["token_id"]) for t in spec["tokens"]]
        delta, suppression_records = individual_suppression_delta(logits_base, unembed, donor_id, token_ids, scale)
        delta_norm = float(torch.linalg.vector_norm(delta.float()).item()) if delta is not None else 0.0
        logits, final_vec = forward_with_final_delta(model, device, recipient_ids, install_combo, delta)
    elif spec["strategy"] == "centroid":
        delta, suppression_records = centroid_suppression_delta(logits_base, unembed, donor_id, spec["groups"], scale)
        delta_norm = float(torch.linalg.vector_norm(delta.float()).item()) if delta is not None else 0.0
        logits, final_vec = forward_with_final_delta(model, device, recipient_ids, install_combo, delta)
    else:
        raise ValueError(f"unknown strategy: {spec['strategy']}")

    vocab = top_vocab_with_classes(logits, tokenizer, ctx, top_k_vocab)
    top = vocab[0]
    donor_diag = logit_diag(logits, donor_id)
    recipient_diag = logit_diag(logits, recipient_id)
    route_after = route_margins(logits, donor_id, groups)
    return {
        "condition": spec["condition"],
        "strategy": spec["strategy"],
        "suppression_scale": float(scale),
        "suppressed_token_ids": [int(t["token_id"]) for t in spec.get("tokens", [])],
        "suppressed_token_texts": [t["token_text"] for t in spec.get("tokens", [])],
        "suppressed_token_classes": [t["class"] for t in spec.get("tokens", [])],
        "suppressed_route_classes": [g["class"] for g in spec.get("groups", [])],
        "suppression_records": suppression_records,
        "suppression_delta_norm": delta_norm,
        "final_norm_output_norm": float(torch.linalg.vector_norm(final_vec).item()) if final_vec is not None else None,
        "top_token_id": int(top["token_id"]),
        "top_token_text": top["token_text"],
        "top_token_class": top["class"],
        "donor_target_rank": donor_diag["target_rank"],
        "donor_top1": donor_diag["target_top1"],
        "recipient_target_rank": recipient_diag["target_rank"],
        "margin_donor_vs_top": margin(logits, donor_id, int(top["token_id"])),
        "margin_donor_vs_recipient": margin(logits, donor_id, recipient_id),
        "route_margins": route_after,
        "top_vocab": vocab,
    }


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    combo: list[dict[str, Any]],
    top_k_vocab: int,
    suppress_scales: list[float],
    max_route_classes: int,
    max_topk_tokens: int,
) -> list[dict[str, Any]]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, donor, recipient)
    donor_id = int(ctx["donor_id"])
    recipient_id = int(ctx["recipient_id"])
    combo_sites = [c["component_id"] for c in combo]

    _meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)
    recipient_state = capture_state(model, device, recipient_ids, combo_sites)
    donor_state = capture_state(model, device, donor_ids, combo_sites)
    deltas = {site: donor_state["components"][site] - recipient_state["components"][site] for site in combo_sites}
    install_combo = build_combo_installer(model, install_joint, combo, deltas)
    logits_base, _final_vec = forward_with_final_delta(model, device, recipient_ids, install_combo, None)
    base_vocab = top_vocab_with_classes(logits_base, tokenizer, ctx, top_k_vocab)
    competitors = unique_competitors(base_vocab, donor_id, max_topk_tokens)
    groups = group_by_class(competitors, max_route_classes)
    base_top_class = base_vocab[0]["class"]
    base_route_margins = route_margins(logits_base, donor_id, groups)
    unembed = get_unembed(model)
    condition_specs = make_condition_specs(competitors, groups, max_topk_tokens)

    rows: list[dict[str, Any]] = []
    for spec in condition_specs:
        scales = [0.0] if spec["strategy"] == "none" else suppress_scales
        for scale in scales:
            metrics = evaluate_condition(
                model,
                tokenizer,
                device,
                recipient_ids,
                install_combo,
                logits_base,
                unembed,
                donor_id,
                recipient_id,
                groups,
                ctx,
                top_k_vocab,
                spec,
                scale,
            )
            route_margin_delta = {}
            for cls, after in metrics["route_margins"].items():
                before = base_route_margins.get(cls)
                if before:
                    route_margin_delta[cls] = after["margin_donor_vs_route_max"] - before["margin_donor_vs_route_max"]
            rows.append(
                {
                    "model": model_name,
                    "target_site": target_site,
                    "pair_id": pair["pair_id"],
                    "direction": direction_name,
                    "object": donor["object"],
                    "relation": donor["relation"],
                    "donor_answer": donor["answer"],
                    "recipient_answer": recipient["answer"],
                    "donor_token_id": donor_id,
                    "recipient_token_id": recipient_id,
                    "intervention_label": intervention_label(intervention),
                    "intervention_mode": intervention["mode"],
                    "combo_components": combo_sites,
                    "base_top_token_id": int(base_vocab[0]["token_id"]),
                    "base_top_token_text": base_vocab[0]["token_text"],
                    "base_top_token_class": base_top_class,
                    "base_donor_rank": logit_diag(logits_base, donor_id)["target_rank"],
                    "base_donor_top1": logit_diag(logits_base, donor_id)["target_top1"],
                    "base_margin_donor_vs_top": margin(logits_base, donor_id, int(base_vocab[0]["token_id"])),
                    "base_route_classes": [g["class"] for g in groups],
                    "base_route_margins": base_route_margins,
                    "route_margin_delta": route_margin_delta,
                    "route_shifted_from_base_top_class": metrics["top_token_class"] != base_top_class,
                    **metrics,
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], float(row["suppression_scale"]))].append(row)
    out = []
    for (condition, scale), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "condition": condition,
                "suppression_scale": scale,
                "n": n,
                "donor_top1_rate": sum(1 for v in vals if v["donor_top1"]) / n,
                "mean_donor_rank": safe_mean([v["donor_target_rank"] for v in vals]),
                "mean_margin_donor_vs_top": safe_mean([v["margin_donor_vs_top"] for v in vals]),
                "mean_base_margin_donor_vs_top": safe_mean([v["base_margin_donor_vs_top"] for v in vals]),
                "mean_margin_gain_vs_base_top": safe_mean([v["margin_donor_vs_top"] - v["base_margin_donor_vs_top"] for v in vals]),
                "mean_margin_donor_vs_recipient": safe_mean([v["margin_donor_vs_recipient"] for v in vals]),
                "mean_suppression_delta_norm": safe_mean([v["suppression_delta_norm"] for v in vals]),
                "mean_suppressed_token_count": safe_mean([len(v["suppressed_token_ids"]) for v in vals]),
                "mean_suppressed_route_count": safe_mean([len(v["suppressed_route_classes"]) for v in vals]),
                "top_token_class_counts": dict(Counter(v["top_token_class"] for v in vals)),
                "base_top_token_class_counts": dict(Counter(v["base_top_token_class"] for v in vals)),
                "route_shift_rate": sum(1 for v in vals if v["route_shifted_from_base_top_class"]) / n,
            }
        )
    return sorted(out, key=lambda r: (r["donor_top1_rate"], r["mean_margin_gain_vs_base_top"] or -999), reverse=True)


def summarize_by_base_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["base_top_token_class"], row["condition"], float(row["suppression_scale"]))].append(row)
    out = []
    for (base_class, condition, scale), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "base_top_token_class": base_class,
                "condition": condition,
                "suppression_scale": scale,
                "n": n,
                "donor_top1_rate": sum(1 for v in vals if v["donor_top1"]) / n,
                "mean_margin_gain_vs_base_top": safe_mean([v["margin_donor_vs_top"] - v["base_margin_donor_vs_top"] for v in vals]),
                "top_token_class_counts": dict(Counter(v["top_token_class"] for v in vals)),
            }
        )
    return sorted(out, key=lambda r: (r["base_top_token_class"], r["donor_top1_rate"], r["mean_margin_gain_vs_base_top"] or -999), reverse=True)


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    combo = load_phase741_ranked_candidates(args.model, args.phase741_round, args.top_candidates)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} "
        f"audits={len(audit_payload['audits'])} combo_candidates={len(combo)}"
    )
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                rows.extend(
                    audit_pair(
                        model,
                        tokenizer,
                        device,
                        args.model,
                        audit_payload["target_site"],
                        pair,
                        audit,
                        combo,
                        args.top_k_vocab,
                        args.suppress_scales,
                        args.max_route_classes,
                        args.max_topk_tokens,
                    )
                )
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: route-level suppression audit {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    condition_summary = summarize_rows(rows)
    base_class_summary = summarize_by_base_class(rows)
    summary = {
        "phase": 745,
        "title": "Route-Level Multi-Competitor Suppression Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "quantization": "off",
        "dtype": "bfloat16",
        "phase739_round": args.phase739_round,
        "phase741_round": args.phase741_round,
        "target_site": audit_payload["target_site"],
        "max_pairs": args.max_pairs,
        "top_audits": args.top_audits,
        "top_candidates": args.top_candidates,
        "top_k_vocab": args.top_k_vocab,
        "suppress_scales": args.suppress_scales,
        "max_route_classes": args.max_route_classes,
        "max_topk_tokens": args.max_topk_tokens,
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "threshold_combo_components": combo,
        "n_rows": len(rows),
        "condition_summary": condition_summary,
        "base_class_summary": base_class_summary,
        "top_condition_summary": condition_summary[:12],
        "strict_interpretation": "This phase uses final-norm route-level geometric suppression to test whether closure failure is single-token, single-route, or multi-route competition. It is not yet proof of a natural suppression circuit.",
    }
    write_jsonl(out_dir / f"phase745_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase745_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "top_condition_summary": summary["top_condition_summary"][:10],
                "top_base_class_summary": summary["base_class_summary"][:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def build_atlas_graph(payload: dict[str, Any], round_name: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(node: dict[str, Any]) -> None:
        if node["id"] in seen:
            return
        seen.add(node["id"])
        nodes.append(node)

    for model_index, model in enumerate(payload.get("models", [])):
        lane_z = (model_index - (len(payload.get("models", [])) - 1) / 2) * 12
        summary = payload["by_model"][model]
        base_node = f"{model}:combo_state"
        field_node = f"{model}:route_competition_field"
        add_node({"id": f"{model}:model", "type": "model", "label": model, "model": model, "position": [-28, 0, lane_z], "role": "tested_model"})
        add_node({"id": base_node, "type": "state", "label": "joint+topK", "model": model, "position": [-10, 0, lane_z], "role": "near_closure_state"})
        add_node({"id": field_node, "type": "competition", "label": "route competition field", "model": model, "position": [8, 0, lane_z], "role": "multi_route_competition"})
        edges.append({"source": f"{model}:model", "target": base_node, "relation": "produces_state", "phase": 745})
        edges.append({"source": base_node, "target": field_node, "relation": "feeds_competition_field", "phase": 745})
        for row in summary.get("top_condition_summary", [])[:8]:
            cond = f"{model}:condition:{row['condition']}:{row['suppression_scale']}"
            add_node(
                {
                    "id": cond,
                    "type": "intervention_condition",
                    "label": f"{row['condition']}@{row['suppression_scale']}",
                    "model": model,
                    "position": [26, row.get("donor_top1_rate") or 0, lane_z],
                    "role": "route_level_suppression_test",
                    "donor_top1_rate": row.get("donor_top1_rate"),
                    "mean_margin_gain": row.get("mean_margin_gain_vs_base_top"),
                    "route_shift_rate": row.get("route_shift_rate"),
                }
            )
            edges.append({"source": field_node, "target": cond, "relation": "tested_by", "weight": row.get("donor_top1_rate"), "phase": 745})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 745 Route-Level Multi-Competitor Suppression ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 745, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> combo state -> route field -> tested suppression condition", "y": "donor top1 rate", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 745},
        "source_files": [str(OUT_ROOT / round_name / "phase745_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase745_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 745,
        "title": "Route-Level Multi-Competitor Suppression Validation",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "final-norm route-level geometric suppression measured against joint+threshold-component state",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase745_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase745_atlas_graph.json", graph)
    lines = [
        f"# Phase 745 Route-Level Multi-Competitor Suppression Validation ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: final-norm route-level suppression on the Phase743/744 joint+topK near-closure state.",
        "",
        "| model | condition | scale | n | donor top1 | mean donor rank | margin gain | top classes | route shift |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for model, summary in payload["by_model"].items():
        for row in summary.get("top_condition_summary", [])[:12]:
            lines.append(
                f"| {model} | {row['condition']} | {row['suppression_scale']:.2f} | {row['n']} | "
                f"{(row.get('donor_top1_rate') or 0):.3f} | "
                f"{(row.get('mean_donor_rank') or 0):.2f} | "
                f"{(row.get('mean_margin_gain_vs_base_top') or 0):.3f} | "
                f"`{json.dumps(row.get('top_token_class_counts') or {}, ensure_ascii=False)}` | "
                f"{(row.get('route_shift_rate') or 0):.3f} |"
            )
    lines.extend(
        [
            "",
            "## By Base Top Class",
            "",
            "| model | base class | condition | scale | n | donor top1 | margin gain | new top classes |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for model, summary in payload["by_model"].items():
        for row in summary.get("base_class_summary", [])[:24]:
            lines.append(
                f"| {model} | {row['base_top_token_class']} | {row['condition']} | {row['suppression_scale']:.2f} | {row['n']} | "
                f"{(row.get('donor_top1_rate') or 0):.3f} | "
                f"{(row.get('mean_margin_gain_vs_base_top') or 0):.3f} | "
                f"`{json.dumps(row.get('top_token_class_counts') or {}, ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- If route-level suppression beats current-top suppression, the failure is multi-token or multi-route competition.",
            "- If all-topK suppression beats route representatives, route classes are internally multi-token rather than represented by one token.",
            "- If donor still fails after multi-route suppression, the remaining bottleneck is donor boost, continuation policy, or a route outside the measured top-k window.",
            "- This phase is a readout geometry validation, not yet a natural circuit proof.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase745_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase739_audits(model, args.phase739_round, args.top_audits)
        combo = load_phase741_ranked_candidates(model, args.phase741_round, args.top_candidates)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
            "threshold_combo_components": combo,
            "top_k_vocab": args.top_k_vocab,
            "max_route_classes": args.max_route_classes,
            "max_topk_tokens": args.max_topk_tokens,
            "suppress_scales": args.suppress_scales,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase739-round", default="confirm")
    parser.add_argument("--phase741-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--top-audits", type=int, default=2)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--top-k-vocab", type=int, default=12)
    parser.add_argument("--suppress-scales", type=float, nargs="+", default=[1.0, 1.25])
    parser.add_argument("--max-route-classes", type=int, default=5)
    parser.add_argument("--max-topk-tokens", type=int, default=8)
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only or --dry-run is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
