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
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, prepare_joint_install  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, combine_installers  # noqa: E402
from phase742_combined_threshold_component_closure import install_combo_add, load_phase741_ranked_candidates  # noqa: E402
from phase743_competitor_format_suppression_audit import top_vocab_with_classes, taxonomy_context  # noqa: E402
from phase744_competitor_suppression_source_localization import build_scan_candidates, module_for_component, parse_component_site  # noqa: E402


OUT_ROOT = Path("results/glm5_phase748_natural_route_suppressor_matrix")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def margin(logits: torch.Tensor, left_id: int, right_id: int) -> float:
    return float((logits[int(left_id)] - logits[int(right_id)]).item())


def build_combo_installer(
    model,
    install_joint: Callable[[], list[Any]],
    combo: list[dict[str, Any]],
    deltas: dict[str, torch.Tensor],
) -> Callable[[], list[Any]]:
    return combine_installers(install_joint, install_combo_add(model, combo, deltas))


def group_competitors_by_route(
    vocab: list[dict[str, Any]],
    donor_id: int,
    max_topk_tokens: int,
    max_route_classes: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    seen: set[int] = set()
    used = 0
    for item in vocab:
        tid = int(item["token_id"])
        if tid == int(donor_id) or tid in seen:
            continue
        seen.add(tid)
        cls = item["class"]
        if cls not in groups:
            order.append(cls)
        groups[cls].append(item)
        used += 1
        if used >= max_topk_tokens:
            break
    out = []
    for cls in order[:max_route_classes]:
        out.append({"class": cls, "tokens": groups[cls]})
    return out


def route_max_logits(logits: torch.Tensor, groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
            "token_count": len(vals),
        }
    return out


def selected_distribution(logits: torch.Tensor, token_ids: list[int]) -> dict[str, float]:
    ids = []
    seen: set[int] = set()
    for tid in token_ids:
        tid = int(tid)
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    if not ids:
        return {}
    vals = torch.stack([logits[tid].float() for tid in ids], dim=0)
    probs = torch.softmax(vals, dim=0)
    return {str(tid): float(probs[i].item()) for i, tid in enumerate(ids)}


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    eps = 1e-12
    total = 0.0
    for key in keys:
        pv = max(float(p.get(key, 0.0)), eps)
        qv = max(float(q.get(key, 0.0)), eps)
        mv = 0.5 * (pv + qv)
        total += 0.5 * pv * torch.log(torch.tensor(pv / mv)).item()
        total += 0.5 * qv * torch.log(torch.tensor(qv / mv)).item()
    return float(total)


def classify_effect(
    base_donor_top1: bool,
    boost_target: float,
    total_route_suppression: float,
    route_coverage: int,
    donor_top1: bool,
    mean_margin_gain: float,
) -> str:
    if base_donor_top1 and route_coverage >= 2 and total_route_suppression > 0.20 and boost_target > 0.10:
        return "mixed_boost_global_suppressor_maintenance_candidate"
    if base_donor_top1 and route_coverage >= 2 and total_route_suppression > 0.20:
        return "global_suppressor_maintenance_candidate"
    if base_donor_top1 and boost_target > 0.20 and mean_margin_gain > 0.20:
        return "booster_maintenance_candidate"
    if (not base_donor_top1) and donor_top1 and route_coverage >= 2 and total_route_suppression > 0.20 and boost_target > 0.10:
        return "mixed_boost_global_suppressor_closure_candidate"
    if (not base_donor_top1) and donor_top1 and route_coverage >= 2 and total_route_suppression > 0.20:
        return "global_suppressor_closure_candidate"
    if route_coverage >= 2 and total_route_suppression > 0.20 and mean_margin_gain > 0.30:
        return "global_suppressor_margin_candidate"
    if route_coverage == 1 and total_route_suppression > 0.10 and mean_margin_gain > 0.20:
        return "route_specific_suppressor_candidate"
    if boost_target > 0.20 and mean_margin_gain > 0.20:
        return "booster_candidate"
    if mean_margin_gain < -0.20:
        return "harmful_or_competitor_support"
    return "small_or_no_effect"


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    combo: list[dict[str, Any]],
    scan_candidates: list[dict[str, Any]],
    top_k_vocab: int,
    max_topk_tokens: int,
    max_route_classes: int,
) -> list[dict[str, Any]]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, donor, recipient)
    donor_id = int(ctx["donor_id"])
    recipient_id = int(ctx["recipient_id"])
    combo_ids = [c["component_id"] for c in combo]
    scan_ids = [c["component_id"] for c in scan_candidates]
    all_sites = sorted(set(combo_ids + scan_ids))

    _meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)
    recipient_state = capture_state(model, device, recipient_ids, all_sites)
    donor_state = capture_state(model, device, donor_ids, all_sites)
    deltas = {site: donor_state["components"][site] - recipient_state["components"][site] for site in all_sites}
    install_combo = build_combo_installer(model, install_joint, combo, deltas)
    base_state = capture_state(model, device, recipient_ids, [], install_combo)
    base_logits = base_state["logits"]
    base_vocab = top_vocab_with_classes(base_logits, tokenizer, ctx, top_k_vocab)
    route_groups = group_competitors_by_route(base_vocab, donor_id, max_topk_tokens, max_route_classes)
    base_route_max = route_max_logits(base_logits, route_groups)
    if not base_route_max:
        return []

    base_donor_logit = float(base_logits[donor_id].item())
    selected_ids = [donor_id, recipient_id]
    for group in route_groups:
        selected_ids.extend(int(t["token_id"]) for t in group["tokens"])
    base_dist = selected_distribution(base_logits, selected_ids)
    base_top = base_vocab[0]
    base_donor_diag = logit_diag(base_logits, donor_id)

    rows: list[dict[str, Any]] = []
    for cand in scan_candidates:
        site = cand["component_id"]
        candidate_delta = deltas[site]

        def install_candidate(site=site, candidate_delta=candidate_delta) -> list[Any]:
            from phase741_threshold_candidate_causal_validation import install_component_edit

            return install_component_edit(model, site, add_delta=candidate_delta)

        installer = combine_installers(install_combo, install_candidate)
        state = capture_state(model, device, recipient_ids, [], installer)
        logits = state["logits"]
        vocab = top_vocab_with_classes(logits, tokenizer, ctx, top_k_vocab)
        new_top = vocab[0]
        new_donor_diag = logit_diag(logits, donor_id)
        new_recipient_diag = logit_diag(logits, recipient_id)
        new_donor_logit = float(logits[donor_id].item())
        after_route_max = route_max_logits(logits, route_groups)
        after_dist = selected_distribution(logits, selected_ids)
        route_matrix: dict[str, dict[str, Any]] = {}
        positive_suppression = []
        margin_gains = []
        for cls, before in base_route_max.items():
            after = after_route_max.get(cls)
            if not after:
                continue
            before_margin = base_donor_logit - float(before["max_logit"])
            after_margin = new_donor_logit - float(after["max_logit"])
            suppress = float(before["max_logit"]) - float(after["max_logit"])
            gain = after_margin - before_margin
            positive_suppression.append(max(0.0, suppress))
            margin_gains.append(gain)
            route_matrix[cls] = {
                "base_route_max_token_id": before["max_token_id"],
                "base_route_max_token_text": before["max_token_text"],
                "base_route_max_logit": before["max_logit"],
                "after_route_max_token_id": after["max_token_id"],
                "after_route_max_token_text": after["max_token_text"],
                "after_route_max_logit": after["max_logit"],
                "route_suppression": suppress,
                "margin_gain_donor_vs_route": gain,
                "after_margin_donor_vs_route": after_margin,
                "base_margin_donor_vs_route": before_margin,
                "route_token_count": before["token_count"],
            }
        boost_target = new_donor_logit - base_donor_logit
        total_route_suppression = float(sum(positive_suppression))
        route_coverage = sum(1 for v in route_matrix.values() if float(v["route_suppression"]) > 0.05)
        mean_margin_gain = safe_mean(margin_gains) or 0.0
        donor_prob_base = float(base_dist.get(str(donor_id), 0.0))
        donor_prob_after = float(after_dist.get(str(donor_id), 0.0))
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
                "combo_components": combo_ids,
                "candidate_component_id": site,
                "candidate_layer": cand["layer"],
                "candidate_component": cand["component"],
                "candidate_already_in_threshold_combo": cand["already_in_threshold_combo"],
                "base_top_token_id": int(base_top["token_id"]),
                "base_top_token_text": base_top["token_text"],
                "base_top_token_class": base_top["class"],
                "base_donor_rank": base_donor_diag["target_rank"],
                "base_donor_top1": base_donor_diag["target_top1"],
                "base_donor_logit": base_donor_logit,
                "base_margin_donor_vs_top": margin(base_logits, donor_id, int(base_top["token_id"])),
                "base_route_classes": [g["class"] for g in route_groups],
                "new_top_token_id": int(new_top["token_id"]),
                "new_top_token_text": new_top["token_text"],
                "new_top_token_class": new_top["class"],
                "new_donor_rank": new_donor_diag["target_rank"],
                "new_donor_top1": new_donor_diag["target_top1"],
                "new_recipient_rank": new_recipient_diag["target_rank"],
                "new_margin_donor_vs_top": margin(logits, donor_id, int(new_top["token_id"])),
                "new_margin_donor_vs_recipient": margin(logits, donor_id, recipient_id),
                "boost_target_logit": boost_target,
                "total_positive_route_suppression": total_route_suppression,
                "route_suppression_coverage": route_coverage,
                "mean_margin_gain_donor_vs_routes": mean_margin_gain,
                "min_margin_gain_donor_vs_routes": min(margin_gains) if margin_gains else None,
                "route_suppressor_matrix": route_matrix,
                "candidate_delta_norm": float(torch.linalg.vector_norm(candidate_delta.float()).item()),
                "readout_jsd_on_selected_vocab": js_divergence(base_dist, after_dist),
                "donor_selected_prob_base": donor_prob_base,
                "donor_selected_prob_after": donor_prob_after,
                "donor_selected_prob_gain": donor_prob_after - donor_prob_base,
                "selected_vocab_token_ids": selected_ids,
                "effect_guess": classify_effect(
                    bool(base_donor_diag["target_top1"]),
                    boost_target,
                    total_route_suppression,
                    route_coverage,
                    bool(new_donor_diag["target_top1"]),
                    mean_margin_gain,
                ),
                "base_top_vocab": base_vocab,
                "new_top_vocab": vocab,
            }
        )
    return rows


def summarize_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["candidate_component_id"]].append(row)
    out = []
    for cid, vals in groups.items():
        n = len(vals)
        route_sums: dict[str, list[float]] = defaultdict(list)
        route_margin_gains: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            for cls, item in row["route_suppressor_matrix"].items():
                route_sums[cls].append(float(item["route_suppression"]))
                route_margin_gains[cls].append(float(item["margin_gain_donor_vs_route"]))
        route_summary = {
            cls: {
                "mean_route_suppression": safe_mean(route_sums[cls]),
                "mean_margin_gain": safe_mean(route_margin_gains[cls]),
                "positive_suppression_rate": sum(1 for v in route_sums[cls] if v > 0.05) / len(route_sums[cls]),
                "n": len(route_sums[cls]),
            }
            for cls in sorted(route_sums)
        }
        out.append(
            {
                "candidate_component_id": cid,
                "candidate_layer": vals[0]["candidate_layer"],
                "candidate_component": vals[0]["candidate_component"],
                "candidate_already_in_threshold_combo": vals[0]["candidate_already_in_threshold_combo"],
                "n": n,
                "mean_boost_target_logit": safe_mean([v["boost_target_logit"] for v in vals]),
                "mean_total_positive_route_suppression": safe_mean([v["total_positive_route_suppression"] for v in vals]),
                "mean_route_suppression_coverage": safe_mean([v["route_suppression_coverage"] for v in vals]),
                "mean_margin_gain_donor_vs_routes": safe_mean([v["mean_margin_gain_donor_vs_routes"] for v in vals]),
                "mean_min_margin_gain_donor_vs_routes": safe_mean([v["min_margin_gain_donor_vs_routes"] for v in vals]),
                "new_donor_top1_rate": sum(1 for v in vals if v["new_donor_top1"]) / n,
                "mean_new_donor_rank": safe_mean([v["new_donor_rank"] for v in vals]),
                "mean_readout_jsd_on_selected_vocab": safe_mean([v["readout_jsd_on_selected_vocab"] for v in vals]),
                "mean_donor_selected_prob_gain": safe_mean([v["donor_selected_prob_gain"] for v in vals]),
                "base_top_class_counts": dict(Counter(v["base_top_token_class"] for v in vals)),
                "new_top_class_counts": dict(Counter(v["new_top_token_class"] for v in vals)),
                "effect_guess_counts": dict(Counter(v["effect_guess"] for v in vals)),
                "dominant_effect_guess": Counter(v["effect_guess"] for v in vals).most_common(1)[0][0],
                "route_summary": route_summary,
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["new_donor_top1_rate"] or 0,
            r["mean_total_positive_route_suppression"] or 0,
            r["mean_route_suppression_coverage"] or 0,
            r["mean_margin_gain_donor_vs_routes"] or -999,
        ),
        reverse=True,
    )


def summarize_routes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for cls, item in row["route_suppressor_matrix"].items():
            groups[(cls, row["candidate_component_id"])].append({**row, **item})
    out = []
    for (cls, cid), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "route_class": cls,
                "candidate_component_id": cid,
                "n": n,
                "mean_route_suppression": safe_mean([v["route_suppression"] for v in vals]),
                "positive_suppression_rate": sum(1 for v in vals if v["route_suppression"] > 0.05) / n,
                "mean_margin_gain_donor_vs_route": safe_mean([v["margin_gain_donor_vs_route"] for v in vals]),
                "mean_boost_target_logit": safe_mean([v["boost_target_logit"] for v in vals]),
                "new_donor_top1_rate": sum(1 for v in vals if v["new_donor_top1"]) / n,
                "effect_guess_counts": dict(Counter(v["effect_guess"] for v in vals)),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["route_class"],
            r["positive_suppression_rate"] or 0,
            r["mean_route_suppression"] or -999,
        ),
        reverse=True,
    )


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
        scan_candidates = build_scan_candidates(model, args.model, combo, args.include_combo_candidates)
        if args.max_scan_candidates and args.max_scan_candidates > 0:
            scan_candidates = scan_candidates[: args.max_scan_candidates]
        log(f"{args.model}: scan_candidates={len(scan_candidates)} {[c['component_id'] for c in scan_candidates]}")
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
                        scan_candidates,
                        args.top_k_vocab,
                        args.max_topk_tokens,
                        args.max_route_classes,
                    )
                )
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: route suppressor matrix scan {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    component_summary = summarize_components(rows)
    route_summary = summarize_routes(rows)
    summary = {
        "phase": 748,
        "title": "Natural Route Suppressor Matrix",
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
        "max_topk_tokens": args.max_topk_tokens,
        "max_route_classes": args.max_route_classes,
        "include_combo_candidates": args.include_combo_candidates,
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "threshold_combo_components": combo,
        "n_rows": len(rows),
        "component_summary": component_summary,
        "route_summary": route_summary,
        "top_component_summary": component_summary[:12],
        "top_route_summary": route_summary[:24],
        "strict_interpretation": "Whole-component donor-recipient deltas are tested for route-specific max-logit suppression. Positive route_suppression means the candidate lowers that route's measured top-k route max. This localizes natural-ish component effects but is not yet neuron-level or training-origin proof.",
    }
    write_jsonl(out_dir / f"phase748_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase748_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "top_component_summary": summary["top_component_summary"][:8],
                "top_route_summary": summary["top_route_summary"][:12],
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
        model_node = f"{model}:model"
        matrix_node = f"{model}:suppressor_matrix"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-26, 0, lane_z], "role": "tested_model"})
        add_node({"id": matrix_node, "type": "matrix", "label": "route suppressor matrix", "model": model, "position": [0, 0, lane_z], "role": "suppressor_field"})
        edges.append({"source": model_node, "target": matrix_node, "relation": "has_route_suppressor_matrix", "phase": 748})
        for row in summary.get("top_component_summary", [])[:8]:
            cid = row["candidate_component_id"]
            node = f"{model}:component:{cid}"
            add_node(
                {
                    "id": node,
                    "type": "component",
                    "label": cid,
                    "model": model,
                    "position": [22, row.get("mean_total_positive_route_suppression") or 0, lane_z],
                    "role": row.get("dominant_effect_guess"),
                    "mean_total_positive_route_suppression": row.get("mean_total_positive_route_suppression"),
                    "mean_boost_target_logit": row.get("mean_boost_target_logit"),
                    "new_donor_top1_rate": row.get("new_donor_top1_rate"),
                    "mean_route_suppression_coverage": row.get("mean_route_suppression_coverage"),
                }
            )
            edges.append({"source": node, "target": matrix_node, "relation": "contributes_to", "weight": row.get("mean_total_positive_route_suppression"), "phase": 748})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 748 Natural Route Suppressor Matrix ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 748, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> route suppressor matrix -> component", "y": "total positive route suppression", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 748},
        "source_files": [str(OUT_ROOT / round_name / "phase748_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase748_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 748,
        "title": "Natural Route Suppressor Matrix",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "whole-component donor-recipient delta add measured as route-specific max-logit suppression and target boost",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase748_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase748_atlas_graph.json", graph)
    lines = [
        f"# Phase 748 Natural Route Suppressor Matrix ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: whole-component donor-recipient deltas measured against route-level max logits.",
        "",
        "| model | component | n | donor top1 | target boost | route suppression | route coverage | margin gain | selected prob gain | effect |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        for row in summary.get("top_component_summary", [])[:12]:
            lines.append(
                f"| {model} | {row['candidate_component_id']} | {row['n']} | "
                f"{(row.get('new_donor_top1_rate') or 0):.3f} | "
                f"{(row.get('mean_boost_target_logit') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_suppression') or 0):.3f} | "
                f"{(row.get('mean_route_suppression_coverage') or 0):.2f} | "
                f"{(row.get('mean_margin_gain_donor_vs_routes') or 0):.3f} | "
                f"{(row.get('mean_donor_selected_prob_gain') or 0):.3f} | "
                f"`{row.get('dominant_effect_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Route-Specific Matrix Slices",
            "",
            "| model | route | component | n | suppression | positive rate | margin gain | donor top1 | effect counts |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model, summary in payload["by_model"].items():
        for row in summary.get("top_route_summary", [])[:28]:
            lines.append(
                f"| {model} | {row['route_class']} | {row['candidate_component_id']} | {row['n']} | "
                f"{(row.get('mean_route_suppression') or 0):.3f} | "
                f"{(row.get('positive_suppression_rate') or 0):.3f} | "
                f"{(row.get('mean_margin_gain_donor_vs_route') or 0):.3f} | "
                f"{(row.get('new_donor_top1_rate') or 0):.3f} | "
                f"`{json.dumps(row.get('effect_guess_counts') or {}, ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- `target boost` measures constructive force toward the donor answer.",
            "- `route suppression` measures selective force against measured route maxima; positive values are suppressor evidence.",
            "- `route coverage` estimates whether the component is route-specific or broad/global over measured top-k routes.",
            "- This is still whole-component donor-recipient delta evidence; it does not yet prove a natural neuron-level suppressor or training-origin suppressor.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase748_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
            "max_topk_tokens": args.max_topk_tokens,
            "max_route_classes": args.max_route_classes,
            "include_combo_candidates": args.include_combo_candidates,
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
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
    parser.add_argument("--max-scan-candidates", type=int, default=0)
    parser.add_argument("--include-combo-candidates", action="store_true")
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
