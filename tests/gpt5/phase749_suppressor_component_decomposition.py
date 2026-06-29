#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
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

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, get_unembed, prepare_joint_install  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, combine_installers, install_component_edit, parse_component_site  # noqa: E402
from phase742_combined_threshold_component_closure import install_combo_add, load_phase741_ranked_candidates  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context, top_vocab_with_classes  # noqa: E402
from phase748_natural_route_suppressor_matrix import (  # noqa: E402
    classify_effect,
    group_competitors_by_route,
    js_divergence,
    margin,
    route_max_logits,
    selected_distribution,
)


OUT_ROOT = Path("results/glm5_phase749_suppressor_component_decomposition")

DEFAULT_COMPONENTS = {
    "qwen3": ["L32:mlp_out", "L33:attn_out", "L32:attn_out"],
    "glm4": ["L34:attn_out", "L35:attn_out", "L36:mlp_out"],
    "deepseek7b": ["L22:attn_out", "L23:attn_out", "L25:mlp_out", "L26:mlp_out"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_combo_installer(
    model,
    install_joint: Callable[[], list[Any]],
    combo: list[dict[str, Any]],
    deltas: dict[str, torch.Tensor],
) -> Callable[[], list[Any]]:
    return combine_installers(install_joint, install_combo_add(model, combo, deltas))


def capture_oproj_inputs(model, device, ids: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    handles = []
    captured: dict[int, torch.Tensor] = {}
    model_layers = get_layers(model)
    for li in sorted(set(layers)):
        attn = get_attention_module(model_layers[li])
        o_proj = get_o_proj(attn)

        def pre_hook(_module, inputs, li=li):
            captured[li] = inputs[0][0, -1].detach().float().cpu()

        handles.append(o_proj.register_forward_pre_hook(pre_hook))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return captured
    finally:
        for h in handles:
            h.remove()


def projected_head_deltas(model, layer_idx: int, donor_in: torch.Tensor, recipient_in: torch.Tensor) -> list[dict[str, Any]]:
    layer = get_layers(model)[layer_idx]
    attn = get_attention_module(layer)
    o_proj = get_o_proj(attn)
    n_heads = get_num_heads(model, attn)
    in_features = int(o_proj.in_features)
    if in_features % n_heads != 0:
        raise RuntimeError(f"L{layer_idx}: o_proj input dim {in_features} not divisible by heads {n_heads}")
    head_dim = in_features // n_heads
    diff = donor_in.float() - recipient_in.float()
    weight = o_proj.weight.detach().float().cpu()
    out = []
    for h in range(n_heads):
        masked = torch.zeros_like(diff)
        start = h * head_dim
        masked[start : start + head_dim] = diff[start : start + head_dim]
        delta = torch.mv(weight, masked)
        out.append({"head": h, "delta": delta, "input_delta_norm": float(torch.linalg.vector_norm(masked).item()), "projected_delta_norm": float(torch.linalg.vector_norm(delta).item())})
    return out


def route_token_ids(base_route_max: dict[str, dict[str, Any]]) -> list[int]:
    return [int(v["max_token_id"]) for v in base_route_max.values()]


def direct_delta_score(delta: torch.Tensor, unembed: torch.Tensor, donor_id: int, route_ids: list[int]) -> dict[str, float]:
    d = delta.float()
    donor_vec = unembed[int(donor_id)].float()
    boost = float(torch.dot(donor_vec, d).item())
    supp = 0.0
    mg = 0.0
    pos = 0
    for tid in route_ids:
        route_vec = unembed[int(tid)].float()
        route_change = float(torch.dot(route_vec, d).item())
        margin_gain = float(torch.dot(donor_vec - route_vec, d).item())
        supp += max(0.0, -route_change)
        mg += margin_gain
        if -route_change > 0.0:
            pos += 1
    return {
        "direct_target_boost": boost,
        "direct_total_route_suppression": supp,
        "direct_mean_margin_gain": mg / max(len(route_ids), 1),
        "direct_positive_route_count": float(pos),
    }


def channel_scores(delta: torch.Tensor, unembed: torch.Tensor, donor_id: int, route_ids: list[int]) -> list[dict[str, Any]]:
    donor_vec = unembed[int(donor_id)].float()
    rows = []
    for idx, value in enumerate(delta.float().tolist()):
        v = float(value)
        if abs(v) <= 1e-12:
            continue
        boost = float(donor_vec[idx].item()) * v
        supp = 0.0
        mg = 0.0
        for tid in route_ids:
            rv = float(unembed[int(tid), idx].item())
            route_change = rv * v
            supp += max(0.0, -route_change)
            mg += (float(donor_vec[idx].item()) - rv) * v
        rows.append(
            {
                "index": idx,
                "delta_value": v,
                "direct_target_boost": boost,
                "direct_total_route_suppression": supp,
                "direct_mean_margin_gain": mg / max(len(route_ids), 1),
                "score": supp + max(0.0, mg / max(len(route_ids), 1)),
            }
        )
    return sorted(rows, key=lambda r: (r["score"], abs(r["delta_value"])), reverse=True)


def masked_channel_delta(delta: torch.Tensor, indices: list[int]) -> torch.Tensor:
    out = torch.zeros_like(delta.float())
    for idx in sorted(set(int(i) for i in indices)):
        if 0 <= idx < out.numel():
            out[idx] = delta.float()[idx]
    return out


def eval_subdelta(
    model,
    device,
    recipient_ids: list[int],
    install_combo: Callable[[], list[Any]],
    site: str,
    sub_delta: torch.Tensor,
    base_logits: torch.Tensor,
    base_vocab: list[dict[str, Any]],
    base_route_max: dict[str, dict[str, Any]],
    route_groups: list[dict[str, Any]],
    donor_id: int,
    recipient_id: int,
    selected_ids: list[int],
    ctx: dict[str, Any],
    tokenizer,
    top_k_vocab: int,
) -> dict[str, Any]:
    base_donor_logit = float(base_logits[donor_id].item())
    base_dist = selected_distribution(base_logits, selected_ids)

    def install_delta() -> list[Any]:
        return install_component_edit(model, site, add_delta=sub_delta)

    state = capture_state(model, device, recipient_ids, [], combine_installers(install_combo, install_delta))
    logits = state["logits"]
    vocab = top_vocab_with_classes(logits, tokenizer, ctx, top_k_vocab)
    new_top = vocab[0]
    new_donor_diag = logit_diag(logits, donor_id)
    new_recipient_diag = logit_diag(logits, recipient_id)
    after_route_max = route_max_logits(logits, route_groups)
    after_dist = selected_distribution(logits, selected_ids)
    new_donor_logit = float(logits[donor_id].item())
    route_matrix = {}
    positive = []
    margin_gains = []
    for cls, before in base_route_max.items():
        after = after_route_max.get(cls)
        if not after:
            continue
        before_margin = base_donor_logit - float(before["max_logit"])
        after_margin = new_donor_logit - float(after["max_logit"])
        suppress = float(before["max_logit"]) - float(after["max_logit"])
        gain = after_margin - before_margin
        positive.append(max(0.0, suppress))
        margin_gains.append(gain)
        route_matrix[cls] = {
            "base_route_max_token_id": before["max_token_id"],
            "base_route_max_token_text": before["max_token_text"],
            "after_route_max_token_id": after["max_token_id"],
            "after_route_max_token_text": after["max_token_text"],
            "route_suppression": suppress,
            "margin_gain_donor_vs_route": gain,
        }
    boost = new_donor_logit - base_donor_logit
    total_supp = float(sum(positive))
    coverage = sum(1 for v in route_matrix.values() if float(v["route_suppression"]) > 0.05)
    mean_gain = safe_mean(margin_gains) or 0.0
    return {
        "new_top_token_id": int(new_top["token_id"]),
        "new_top_token_text": new_top["token_text"],
        "new_top_token_class": new_top["class"],
        "new_donor_rank": new_donor_diag["target_rank"],
        "new_donor_top1": new_donor_diag["target_top1"],
        "new_recipient_rank": new_recipient_diag["target_rank"],
        "new_margin_donor_vs_top": margin(logits, donor_id, int(new_top["token_id"])),
        "new_margin_donor_vs_recipient": margin(logits, donor_id, recipient_id),
        "boost_target_logit": boost,
        "total_positive_route_suppression": total_supp,
        "route_suppression_coverage": coverage,
        "mean_margin_gain_donor_vs_routes": mean_gain,
        "route_suppressor_matrix": route_matrix,
        "readout_jsd_on_selected_vocab": js_divergence(base_dist, after_dist),
        "donor_selected_prob_gain": float(after_dist.get(str(donor_id), 0.0)) - float(base_dist.get(str(donor_id), 0.0)),
        "effect_guess": classify_effect(
            bool(logit_diag(base_logits, donor_id)["target_top1"]),
            boost,
            total_supp,
            coverage,
            bool(new_donor_diag["target_top1"]),
            mean_gain,
        ),
    }


def component_list_for(args) -> list[str]:
    if args.components:
        return [x.strip() for x in args.components.split(",") if x.strip()]
    comps = list(DEFAULT_COMPONENTS[args.model])
    return comps[: args.max_components] if args.max_components else comps


def audit_pair(
    model,
    tokenizer,
    device,
    args,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    combo: list[dict[str, Any]],
    components: list[str],
    rng: random.Random,
    unembed: torch.Tensor,
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
    all_sites = sorted(set(combo_ids + components))

    _meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)
    recipient_state = capture_state(model, device, recipient_ids, all_sites)
    donor_state = capture_state(model, device, donor_ids, all_sites)
    deltas = {site: donor_state["components"][site] - recipient_state["components"][site] for site in all_sites}
    install_combo = build_combo_installer(model, install_joint, combo, deltas)
    base_state = capture_state(model, device, recipient_ids, [], install_combo)
    base_logits = base_state["logits"]
    base_vocab = top_vocab_with_classes(base_logits, tokenizer, ctx, args.top_k_vocab)
    route_groups = group_competitors_by_route(base_vocab, donor_id, args.max_topk_tokens, args.max_route_classes)
    base_route_max = route_max_logits(base_logits, route_groups)
    if not base_route_max:
        return []
    selected_ids = [donor_id, recipient_id]
    for group in route_groups:
        selected_ids.extend(int(t["token_id"]) for t in group["tokens"])

    attn_layers = [parse_component_site(c)[0] for c in components if parse_component_site(c)[1] == "attn_out"]
    donor_oproj = capture_oproj_inputs(model, device, donor_ids, attn_layers) if attn_layers else {}
    recipient_oproj = capture_oproj_inputs(model, device, recipient_ids, attn_layers) if attn_layers else {}

    rows: list[dict[str, Any]] = []
    route_ids = route_token_ids(base_route_max)
    base_donor_diag = logit_diag(base_logits, donor_id)
    for site in components:
        layer, component = parse_component_site(site)
        whole_delta = deltas[site]
        candidates: list[dict[str, Any]] = []
        candidates.append({"kind": "whole_component", "unit_id": site, "delta": whole_delta, "direct": direct_delta_score(whole_delta, unembed, donor_id, route_ids), "unit_meta": {}})

        if component == "attn_out" and layer in donor_oproj and layer in recipient_oproj:
            head_rows = projected_head_deltas(model, layer, donor_oproj[layer], recipient_oproj[layer])
            for hr in head_rows:
                hr["direct"] = direct_delta_score(hr["delta"], unembed, donor_id, route_ids)
            head_rows.sort(key=lambda r: (r["direct"]["direct_total_route_suppression"], r["direct"]["direct_mean_margin_gain"]), reverse=True)
            selected = head_rows[: args.top_heads_per_component]
            avoid = {int(r["head"]) for r in selected}
            controls = [r for r in head_rows if int(r["head"]) not in avoid]
            rng.shuffle(controls)
            selected.extend(controls[: args.random_heads_per_component])
            for hr in selected:
                candidates.append({"kind": "attn_head", "unit_id": f"{site}:H{hr['head']}", "delta": hr["delta"], "direct": hr["direct"], "unit_meta": {"head": hr["head"], "projected_delta_norm": hr["projected_delta_norm"], "control": int(hr["head"]) not in avoid}})
            for k in args.headset_sizes:
                heads = head_rows[: min(k, len(head_rows))]
                if not heads:
                    continue
                delta = torch.stack([h["delta"] for h in heads], dim=0).sum(dim=0)
                candidates.append({"kind": "attn_headset", "unit_id": f"{site}:topH{k}", "delta": delta, "direct": direct_delta_score(delta, unembed, donor_id, route_ids), "unit_meta": {"heads": [h["head"] for h in heads]}})

        if component == "mlp_out":
            scored = channel_scores(whole_delta, unembed, donor_id, route_ids)
            for ch in scored[: args.individual_channels]:
                delta = masked_channel_delta(whole_delta, [ch["index"]])
                candidates.append({"kind": "mlp_channel", "unit_id": f"{site}:C{ch['index']}", "delta": delta, "direct": direct_delta_score(delta, unembed, donor_id, route_ids), "unit_meta": {"channel": ch["index"], "channel_score": ch["score"]}})
            universe = list(range(whole_delta.numel()))
            top_indices = [int(r["index"]) for r in scored]
            for k in args.channelset_sizes:
                idxs = top_indices[: min(k, len(top_indices))]
                if not idxs:
                    continue
                delta = masked_channel_delta(whole_delta, idxs)
                candidates.append({"kind": "mlp_channelset", "unit_id": f"{site}:topC{k}", "delta": delta, "direct": direct_delta_score(delta, unembed, donor_id, route_ids), "unit_meta": {"channels": idxs[:32], "n_channels": len(idxs), "selection": "top_direct_score"}})
                ctrl = rng.sample(universe, min(k, len(universe)))
                cdelta = masked_channel_delta(whole_delta, ctrl)
                candidates.append({"kind": "mlp_channelset_control", "unit_id": f"{site}:randC{k}", "delta": cdelta, "direct": direct_delta_score(cdelta, unembed, donor_id, route_ids), "unit_meta": {"n_channels": len(ctrl), "selection": "random"}})

        for cand in candidates:
            metrics = eval_subdelta(
                model,
                device,
                recipient_ids,
                install_combo,
                site,
                cand["delta"],
                base_logits,
                base_vocab,
                base_route_max,
                route_groups,
                donor_id,
                recipient_id,
                selected_ids,
                ctx,
                tokenizer,
                args.top_k_vocab,
            )
            rows.append(
                {
                    "model": args.model,
                    "target_site": target_site,
                    "pair_id": pair["pair_id"],
                    "direction": direction_name,
                    "object": donor["object"],
                    "relation": donor["relation"],
                    "donor_answer": donor["answer"],
                    "recipient_answer": recipient["answer"],
                    "intervention_label": intervention_label(intervention),
                    "component_id": site,
                    "component_layer": layer,
                    "component_type": component,
                    "subunit_kind": cand["kind"],
                    "subunit_id": cand["unit_id"],
                    "subunit_meta": cand["unit_meta"],
                    "direct_score": cand["direct"],
                    "sub_delta_norm": float(torch.linalg.vector_norm(cand["delta"].float()).item()),
                    "whole_delta_norm": float(torch.linalg.vector_norm(whole_delta.float()).item()),
                    "base_top_token_text": base_vocab[0]["token_text"],
                    "base_top_token_class": base_vocab[0]["class"],
                    "base_donor_rank": base_donor_diag["target_rank"],
                    "base_donor_top1": base_donor_diag["target_top1"],
                    "base_route_classes": [g["class"] for g in route_groups],
                    **metrics,
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["component_id"], row["subunit_kind"], row["subunit_id"])].append(row)
    out = []
    for (component_id, kind, subunit_id), vals in groups.items():
        n = len(vals)
        route_sums: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            for cls, item in row["route_suppressor_matrix"].items():
                route_sums[cls].append(float(item["route_suppression"]))
        out.append(
            {
                "component_id": component_id,
                "subunit_kind": kind,
                "subunit_id": subunit_id,
                "n": n,
                "mean_boost_target_logit": safe_mean([v["boost_target_logit"] for v in vals]),
                "mean_total_positive_route_suppression": safe_mean([v["total_positive_route_suppression"] for v in vals]),
                "mean_route_suppression_coverage": safe_mean([v["route_suppression_coverage"] for v in vals]),
                "mean_margin_gain_donor_vs_routes": safe_mean([v["mean_margin_gain_donor_vs_routes"] for v in vals]),
                "new_donor_top1_rate": sum(1 for v in vals if v["new_donor_top1"]) / n,
                "mean_new_donor_rank": safe_mean([v["new_donor_rank"] for v in vals]),
                "mean_sub_delta_fraction": safe_mean([(v["sub_delta_norm"] / max(v["whole_delta_norm"], 1e-8)) for v in vals]),
                "effect_guess_counts": dict(Counter(v["effect_guess"] for v in vals)),
                "dominant_effect_guess": Counter(v["effect_guess"] for v in vals).most_common(1)[0][0],
                "route_summary": {
                    cls: {
                        "mean_route_suppression": safe_mean(vals2),
                        "positive_suppression_rate": sum(1 for x in vals2 if x > 0.05) / len(vals2),
                        "n": len(vals2),
                    }
                    for cls, vals2 in sorted(route_sums.items())
                },
            }
        )
    out.sort(
        key=lambda r: (
            r["component_id"],
            r["subunit_kind"] != "whole_component",
            r["mean_total_positive_route_suppression"] or 0,
            r["mean_margin_gain_donor_vs_routes"] or -999,
        ),
        reverse=True,
    )
    return {
        "phase": 749,
        "title": "Suppressor Component Decomposition",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "components": component_list_for(args),
        "summary": out,
        "top_subunits": sorted(out, key=lambda r: (r["mean_total_positive_route_suppression"] or 0, r["mean_route_suppression_coverage"] or 0), reverse=True)[:24],
        "strict_interpretation": "Attention components are decomposed into projected o_proj head deltas. MLP components are decomposed into residual output channels, not true MLP neurons. Evidence remains donor-recipient delta evidence.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    combo = load_phase741_ranked_candidates(args.model, args.phase741_round, args.top_candidates)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    components = component_list_for(args)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} components={components} audits={len(audit_payload['audits'])}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    unembed = get_unembed(model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                rows.extend(audit_pair(model, tokenizer, device, args, audit_payload["target_site"], pair, audit, combo, components, rng, unembed))
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: decomposition scan {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase749_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase749_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "top_subunits": summary["top_subunits"][:12]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase749_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 749,
        "title": "Suppressor Component Decomposition",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "First-pass decomposition only. Attention head deltas are projected o_proj slices; MLP channels are residual output dimensions, not neurons.",
    }
    write_json(out_dir / "phase749_cross_model_summary.json", payload)
    lines = [
        f"# Phase 749 Suppressor Component Decomposition ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence: subunit donor-recipient deltas measured against route-level max logits.",
        "",
        "| model | component | subunit | kind | n | donor top1 | target boost | route suppression | coverage | margin gain | delta fraction | effect |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        for row in summary.get("top_subunits", [])[:18]:
            lines.append(
                f"| {model} | {row['component_id']} | {row['subunit_id']} | {row['subunit_kind']} | {row['n']} | "
                f"{(row.get('new_donor_top1_rate') or 0):.3f} | "
                f"{(row.get('mean_boost_target_logit') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_suppression') or 0):.3f} | "
                f"{(row.get('mean_route_suppression_coverage') or 0):.2f} | "
                f"{(row.get('mean_margin_gain_donor_vs_routes') or 0):.3f} | "
                f"{(row.get('mean_sub_delta_fraction') or 0):.3f} | "
                f"`{row.get('dominant_effect_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Attention decomposition is head-level o_proj projected delta evidence.",
            "- MLP decomposition is residual output channel evidence, not true neuron evidence.",
            "- A small subunit matching the whole component's route suppression is a localization hint, not proof of natural coding origin.",
            "",
        ]
    )
    (out_dir / "phase749_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        args.model = model
        audits = load_phase739_audits(model, args.phase739_round, args.top_audits)
        combo = load_phase741_ranked_candidates(model, args.phase741_round, args.top_candidates)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "components": component_list_for(args),
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
            "combo_components": [c["component_id"] for c in combo],
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
    parser.add_argument("--components", default="")
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--top-audits", type=int, default=2)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
    parser.add_argument("--top-heads-per-component", type=int, default=4)
    parser.add_argument("--random-heads-per-component", type=int, default=2)
    parser.add_argument("--headset-sizes", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument("--channelset-sizes", type=int, nargs="*", default=[1, 4, 16, 64])
    parser.add_argument("--individual-channels", type=int, default=4)
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--seed", type=int, default=749)
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
