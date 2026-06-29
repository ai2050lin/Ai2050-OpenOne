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

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads, get_v_proj  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, build_source_groups, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, get_unembed  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, parse_component_site  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context, top_vocab_with_classes  # noqa: E402
from phase748_natural_route_suppressor_matrix import group_competitors_by_route, margin, route_max_logits, selected_distribution, js_divergence  # noqa: E402
from phase749_suppressor_component_decomposition import capture_oproj_inputs, direct_delta_score, projected_head_deltas, route_token_ids  # noqa: E402


OUT_ROOT = Path("results/glm5_phase751_natural_attention_head_mechanism_backtrace")

DEFAULT_ATTENTION_COMPONENTS = {
    "qwen3": ["L32:attn_out", "L33:attn_out"],
    "glm4": ["L34:attn_out", "L35:attn_out"],
    "deepseek7b": ["L22:attn_out", "L23:attn_out"],
}

FOCUS_HEADS = {
    "qwen3": {},
    "glm4": {"L34:attn_out": [4], "L35:attn_out": [23]},
    "deepseek7b": {"L22:attn_out": [1, 7, 25], "L23:attn_out": [0, 11, 14]},
}

SOURCE_GROUPS = [
    "target_record_line",
    "target_value_tokens",
    "records_all",
    "records_other",
    "object_tokens",
    "relation_tokens",
    "question",
    "instruction",
    "answer_prefix",
    "all_pre_answer",
    "self_last",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def component_list_for(args) -> list[str]:
    if args.components:
        return [x.strip() for x in args.components.split(",") if x.strip()]
    comps = list(DEFAULT_ATTENTION_COMPONENTS[args.model])
    return comps[: args.max_components] if args.max_components else comps


def build_route_context(
    logits: torch.Tensor,
    tokenizer,
    ctx: dict[str, Any],
    target_id: int,
    top_k_vocab: int,
    max_topk_tokens: int,
    max_route_classes: int,
) -> dict[str, Any] | None:
    vocab = top_vocab_with_classes(logits, tokenizer, ctx, top_k_vocab)
    route_groups = group_competitors_by_route(vocab, target_id, max_topk_tokens, max_route_classes)
    route_max = route_max_logits(logits, route_groups)
    if not route_max:
        return None
    selected_ids = [int(target_id), int(ctx["recipient_id"])]
    for group in route_groups:
        selected_ids.extend(int(t["token_id"]) for t in group["tokens"])
    return {
        "vocab": vocab,
        "route_groups": route_groups,
        "route_max": route_max,
        "selected_ids": selected_ids,
        "selected_dist": selected_distribution(logits, selected_ids),
    }


def capture_attention_value_state(model, tokenizer, device, case: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    ids = tokenizer.encode(prompt_for(case), add_special_tokens=False)
    handles = []
    value_store: dict[int, torch.Tensor] = {}
    for li in sorted(set(layers)):
        attn = get_attention_module(get_layers(model)[li])
        v_proj = get_v_proj(attn)

        def v_hook(_module, _inputs, output, li=li):
            value_store[li] = output.detach().float().cpu()

        handles.append(v_proj.register_forward_hook(v_hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in sorted(set(layers))}
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return {
        "ids": ids,
        "prompt": prompt_for(case),
        "answer_pos": len(ids) - 1,
        "logits": logits,
        "attentions": attentions,
        "values": value_store,
        "source_groups": build_source_groups(tokenizer, prompt_for(case), case, ids),
    }


def attention_group_masses(attn_row: torch.Tensor, source_groups: dict[str, list[int]]) -> dict[str, float]:
    n = attn_row.numel()
    out = {}
    for group in SOURCE_GROUPS:
        idxs = [i for i in source_groups.get(group, []) if 0 <= i < n]
        out[f"mass_{group}"] = float(attn_row[idxs].sum().item()) if idxs else 0.0
    return out


def project_source_contribution(
    model,
    layer_idx: int,
    head_ids: list[int],
    contribution: torch.Tensor,
) -> torch.Tensor:
    attn = get_attention_module(get_layers(model)[layer_idx])
    o_proj = get_o_proj(attn)
    n_heads = get_num_heads(model, attn)
    in_features = int(o_proj.in_features)
    head_dim = in_features // n_heads
    full = torch.zeros(in_features, dtype=torch.float32)
    contrib = contribution.detach().float().cpu()
    if contrib.ndim == 3:
        contrib = contrib[0]
    for h in sorted({int(x) for x in head_ids if 0 <= int(x) < n_heads}):
        start = h * head_dim
        full[start : start + head_dim] = contrib[h, :head_dim]
    return torch.mv(o_proj.weight.detach().float().cpu(), full)


def install_source_contribution_removal(
    model,
    site: str,
    head_ids: list[int],
    contribution: torch.Tensor,
) -> Callable[[], list[Any]]:
    layer_idx, component = parse_component_site(site)
    if component != "attn_out":
        raise ValueError(site)
    attn = get_attention_module(get_layers(model)[layer_idx])
    o_proj = get_o_proj(attn)
    n_heads = get_num_heads(model, attn)
    in_features = int(o_proj.in_features)
    head_dim = in_features // n_heads
    contrib = contribution.detach().float().cpu()
    if contrib.ndim == 3:
        contrib = contrib[0]
    head_set = sorted({int(h) for h in head_ids if 0 <= int(h) < n_heads})

    def install() -> list[Any]:
        def pre_hook(_module, inputs):
            x = inputs[0]
            y = x.clone()
            yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
            for h in head_set:
                yv[0, -1, h, :] = yv[0, -1, h, :] - contrib[h, :head_dim].to(yv.device, yv.dtype)
            return (y,) + tuple(inputs[1:])

        return [o_proj.register_forward_pre_hook(pre_hook)]

    return install


def run_with_hooks(model, device, ids: list[int], install: Callable[[], list[Any]]) -> torch.Tensor:
    handles = install()
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def eval_after_logits(
    base_logits: torch.Tensor,
    after_logits: torch.Tensor,
    route_ctx: dict[str, Any],
    target_id: int,
    contrast_id: int,
) -> dict[str, Any]:
    base_target_diag = logit_diag(base_logits, target_id)
    after_target_diag = logit_diag(after_logits, target_id)
    after_route_max = route_max_logits(after_logits, route_ctx["route_groups"])
    after_dist = selected_distribution(after_logits, route_ctx["selected_ids"])
    target_drop = float(base_logits[target_id].item() - after_logits[target_id].item())
    contrast_gain = float(after_logits[contrast_id].item() - base_logits[contrast_id].item())
    route_matrix = {}
    releases = []
    margin_drops = []
    for cls, before in route_ctx["route_max"].items():
        after = after_route_max.get(cls)
        if not after:
            continue
        release = float(after["max_logit"]) - float(before["max_logit"])
        before_margin = float(base_logits[target_id].item()) - float(before["max_logit"])
        after_margin = float(after_logits[target_id].item()) - float(after["max_logit"])
        margin_drop = before_margin - after_margin
        releases.append(max(0.0, release))
        margin_drops.append(margin_drop)
        route_matrix[cls] = {
            "route_release": release,
            "margin_drop": margin_drop,
            "base_route_token": before["max_token_text"],
            "after_route_token": after["max_token_text"],
        }
    return {
        "base_target_top1": base_target_diag["target_top1"],
        "after_target_top1": after_target_diag["target_top1"],
        "top1_loss": bool(base_target_diag["target_top1"]) and not bool(after_target_diag["target_top1"]),
        "target_logit_drop": target_drop,
        "contrast_logit_gain": contrast_gain,
        "margin_drop_target_vs_contrast": margin(base_logits, target_id, contrast_id) - margin(after_logits, target_id, contrast_id),
        "total_positive_route_release": float(sum(releases)),
        "route_release_coverage": sum(1 for v in route_matrix.values() if float(v["route_release"]) > 0.05),
        "mean_margin_drop_target_vs_routes": safe_mean(margin_drops) or 0.0,
        "readout_jsd_on_selected_vocab": js_divergence(route_ctx["selected_dist"], after_dist),
        "route_release_matrix": route_matrix,
    }


def classify_role(target_drop: float, route_release: float, margin_drop: float, attn_mass: float, source_target_contrib: float) -> str:
    if target_drop > 0.20 and route_release > 0.20 and margin_drop > 0.20:
        return "mixed_target_support_and_route_guard"
    if target_drop > 0.20 and margin_drop > 0.10:
        return "target_support_content"
    if route_release > 0.20 and margin_drop > 0.20:
        return "route_suppressor_content"
    if attn_mass > 0.20 and abs(target_drop) < 0.10 and route_release < 0.20:
        return "qk_pattern_visible_content_weak"
    if margin_drop < -0.20:
        return "inverse_or_compensatory_content"
    if source_target_contrib > 0.20:
        return "readout_target_aligned_observational"
    return "small_or_unclear"


def select_head_candidates_for_pair(
    model,
    tokenizer,
    device,
    args,
    donor: dict[str, Any],
    recipient: dict[str, Any],
    components: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, donor, recipient)
    donor_id = int(ctx["donor_id"])
    donor_state = capture_state(model, device, donor_ids, components)
    route_ctx = build_route_context(donor_state["logits"], tokenizer, ctx, donor_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return []
    route_ids = route_token_ids(route_ctx["route_max"])
    layers = [parse_component_site(c)[0] for c in components]
    donor_oproj = capture_oproj_inputs(model, device, donor_ids, layers)
    recipient_oproj = capture_oproj_inputs(model, device, recipient_ids, layers)
    candidates: list[dict[str, Any]] = []
    for site in components:
        layer, _component = parse_component_site(site)
        head_rows = projected_head_deltas(model, layer, donor_oproj[layer], recipient_oproj[layer])
        for row in head_rows:
            row["direct"] = direct_delta_score(row["delta"], unembed, donor_id, route_ids)
        head_rows.sort(key=lambda r: (r["direct"]["direct_total_route_suppression"], r["direct"]["direct_mean_margin_gain"]), reverse=True)
        for k in args.headset_sizes:
            heads = [int(h["head"]) for h in head_rows[: min(k, len(head_rows))]]
            if heads:
                candidates.append({"site": site, "subunit_id": f"{site}:topH{k}", "subunit_kind": "attn_headset", "heads": heads, "selection": "dynamic_top_direct_suppression"})
        for row in head_rows[: args.individual_heads]:
            h = int(row["head"])
            candidates.append({"site": site, "subunit_id": f"{site}:H{h}", "subunit_kind": "attn_head", "heads": [h], "selection": "dynamic_top_direct_suppression"})
        for h in FOCUS_HEADS.get(args.model, {}).get(site, [])[: args.max_focus_heads]:
            if 0 <= int(h) < len(head_rows):
                candidates.append({"site": site, "subunit_id": f"{site}:H{h}", "subunit_kind": "attn_head_focus", "heads": [int(h)], "selection": "phase750_focus_head"})
    dedup = {}
    for cand in candidates:
        key = (cand["site"], cand["subunit_id"], tuple(cand["heads"]))
        dedup[key] = cand
    return list(dedup.values())


def audit_context(
    model,
    tokenizer,
    device,
    args,
    context_name: str,
    target_item: dict[str, Any],
    contrast_item: dict[str, Any],
    candidates: list[dict[str, Any]],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_attention_value_state(model, tokenizer, device, target_item, layers)
    ctx = taxonomy_context(tokenizer, target_item, contrast_item)
    target_id = int(ctx["donor_id"])
    contrast_id = int(ctx["recipient_id"])
    route_ctx = build_route_context(state["logits"], tokenizer, ctx, target_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return []
    route_ids = route_token_ids(route_ctx["route_max"])
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        answer_pos = state["answer_pos"]
        head_ids = [h for h in cand["heads"] if 0 <= int(h) < n_heads]
        if not head_ids:
            continue
        attn_rows = []
        for h in head_ids:
            attn_rows.append(torch.tensor(state["attentions"][layer][0, h, answer_pos, :], dtype=torch.float32))
        mean_attn_row = torch.stack(attn_rows, dim=0).mean(dim=0)
        mass = attention_group_masses(mean_attn_row, state["source_groups"])
        for source_group in SOURCE_GROUPS[: args.max_source_groups]:
            src_positions = state["source_groups"].get(source_group, [])
            if not src_positions:
                continue
            contribution = compute_source_contribution(
                state["attentions"][layer],
                state["values"][layer],
                [answer_pos],
                [src_positions],
                n_heads,
                num_kv_heads,
            )
            projected = project_source_contribution(model, layer, head_ids, contribution)
            direct = direct_delta_score(projected, unembed, target_id, route_ids)
            install = install_source_contribution_removal(model, site, head_ids, contribution)
            after_logits = run_with_hooks(model, device, state["ids"], install)
            metrics = eval_after_logits(state["logits"], after_logits, route_ctx, target_id, contrast_id)
            attn_mass = mass.get(f"mass_{source_group}", 0.0)
            role = classify_role(
                metrics["target_logit_drop"],
                metrics["total_positive_route_release"],
                metrics["mean_margin_drop_target_vs_routes"],
                attn_mass,
                direct["direct_target_boost"],
            )
            rows.append(
                {
                    "context_name": context_name,
                    "target_object": target_item["object"],
                    "target_relation": target_item["relation"],
                    "target_answer": target_item["answer"],
                    "contrast_answer": contrast_item["answer"],
                    "site": site,
                    "layer": layer,
                    "subunit_id": cand["subunit_id"],
                    "subunit_kind": cand["subunit_kind"],
                    "heads": head_ids,
                    "selection": cand["selection"],
                    "source_group": source_group,
                    "source_positions_n": len(src_positions),
                    "mean_attention_mass_to_source": attn_mass,
                    "attention_masses": mass,
                    "source_projected_delta_norm": float(torch.linalg.vector_norm(projected.float()).item()),
                    "source_direct_score": direct,
                    "role_guess": role,
                    **metrics,
                }
            )
    return rows


def audit_pair(model, tokenizer, device, args, pair: dict[str, Any], audit: dict[str, Any], components: list[str], unembed: torch.Tensor) -> list[dict[str, Any]]:
    donor, recipient = choose_donor_recipient(pair, audit["direction"])
    candidates = select_head_candidates_for_pair(model, tokenizer, device, args, donor, recipient, components, unembed)
    rows: list[dict[str, Any]] = []
    contexts = [("natural_donor", donor, recipient)]
    if not args.donor_context_only:
        contexts.append(("natural_recipient", recipient, donor))
    for context_name, target_item, contrast_item in contexts:
        for row in audit_context(model, tokenizer, device, args, context_name, target_item, contrast_item, candidates, unembed):
            row.update({"pair_id": pair["pair_id"], "direction": audit["direction"], "intervention_label": intervention_label(audit["intervention"])})
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["context_name"], row["site"], row["subunit_id"], row["subunit_kind"], row["source_group"])].append(row)
    summary = []
    for (context_name, site, subunit_id, kind, source_group), vals in grouped.items():
        n = len(vals)
        summary.append(
            {
                "context_name": context_name,
                "site": site,
                "subunit_id": subunit_id,
                "subunit_kind": kind,
                "source_group": source_group,
                "n": n,
                "mean_attention_mass_to_source": safe_mean([v["mean_attention_mass_to_source"] for v in vals]),
                "mean_source_target_logit_contribution": safe_mean([v["source_direct_score"]["direct_target_boost"] for v in vals]),
                "mean_source_total_route_suppression_contribution": safe_mean([v["source_direct_score"]["direct_total_route_suppression"] for v in vals]),
                "mean_source_margin_contribution": safe_mean([v["source_direct_score"]["direct_mean_margin_gain"] for v in vals]),
                "mean_target_logit_drop_after_source_removal": safe_mean([v["target_logit_drop"] for v in vals]),
                "mean_total_positive_route_release_after_source_removal": safe_mean([v["total_positive_route_release"] for v in vals]),
                "mean_route_release_coverage": safe_mean([v["route_release_coverage"] for v in vals]),
                "mean_margin_drop_target_vs_routes": safe_mean([v["mean_margin_drop_target_vs_routes"] for v in vals]),
                "top1_loss_rate": sum(1 for v in vals if v["top1_loss"]) / n,
                "role_guess_counts": dict(Counter(v["role_guess"] for v in vals)),
                "dominant_role_guess": Counter(v["role_guess"] for v in vals).most_common(1)[0][0],
                "heads_seen": sorted({tuple(v["heads"]) for v in vals})[:12],
            }
        )
    summary.sort(
        key=lambda r: (
            r["mean_target_logit_drop_after_source_removal"] or 0,
            r["mean_total_positive_route_release_after_source_removal"] or 0,
            r["mean_margin_drop_target_vs_routes"] or 0,
            r["mean_attention_mass_to_source"] or 0,
        ),
        reverse=True,
    )
    return {
        "phase": 751,
        "title": "Natural Attention Head Mechanism Backtrace",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "components": component_list_for(args),
        "summary": summary,
        "top_source_mechanism_candidates": summary[:32],
        "strict_interpretation": "QK is observational attention mass only. V/O is causal source-contribution removal before o_proj. This still does not prove neuron-level coding.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audits = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    components = component_list_for(args)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} components={components} audits={len(audits['audits'])}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    unembed = get_unembed(model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audits["audits"]:
                rows.extend(audit_pair(model, tokenizer, device, args, pair, audit, components, unembed))
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: mechanism backtrace {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args)
    summary["attn_implementation"] = attn_impl
    summary["dtype"] = "bfloat16"
    summary["quantization"] = "off"
    write_jsonl(out_dir / f"phase751_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase751_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "top": summary["top_source_mechanism_candidates"][:10]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase751_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 751,
        "title": "Natural Attention Head Mechanism Backtrace",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "QK pattern is observational; V/O content is tested by source-contribution removal.",
    }
    write_json(out_dir / "phase751_cross_model_summary.json", payload)
    lines = [
        f"# Phase 751 Natural Attention Head Mechanism Backtrace ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence: source attention mass plus causal source V/O contribution removal.",
        "",
        "| model | context | site | subunit | source | n | attn mass | source target contrib | source route supp contrib | remove target drop | route release | coverage | margin drop | top1 loss | role |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_source_mechanism_candidates", [])[:18]:
            lines.append(
                f"| {model_name} | {row['context_name']} | {row['site']} | {row['subunit_id']} | {row['source_group']} | {row['n']} | "
                f"{(row.get('mean_attention_mass_to_source') or 0):.3f} | "
                f"{(row.get('mean_source_target_logit_contribution') or 0):.3f} | "
                f"{(row.get('mean_source_total_route_suppression_contribution') or 0):.3f} | "
                f"{(row.get('mean_target_logit_drop_after_source_removal') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_release_after_source_removal') or 0):.3f} | "
                f"{(row.get('mean_route_release_coverage') or 0):.2f} | "
                f"{(row.get('mean_margin_drop_target_vs_routes') or 0):.3f} | "
                f"{(row.get('top1_loss_rate') or 0):.3f} | "
                f"`{row.get('dominant_role_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- QK/pattern evidence is attention mass only.",
            "- V/O/content evidence is causal source contribution removal before o_proj.",
            "- A source group with high attention but weak removal effect is not a causal content source.",
            "- This is head/source-path evidence, not neuron-level evidence.",
            "",
        ]
    )
    (out_dir / "phase751_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model_name in MODELS:
        args.model = model_name
        audits = load_phase739_audits(model_name, args.phase739_round, args.top_audits)
        payload["models"][model_name] = {
            "components": component_list_for(args),
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
            "headset_sizes": args.headset_sizes,
            "individual_heads": args.individual_heads,
            "source_groups": SOURCE_GROUPS[: args.max_source_groups],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase739-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--components", default="")
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--top-audits", type=int, default=2)
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
    parser.add_argument("--headset-sizes", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument("--individual-heads", type=int, default=1)
    parser.add_argument("--max-focus-heads", type=int, default=2)
    parser.add_argument("--max-source-groups", type=int, default=8)
    parser.add_argument("--donor-context-only", action="store_true")
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
