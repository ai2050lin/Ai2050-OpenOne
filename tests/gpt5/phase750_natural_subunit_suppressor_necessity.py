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

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, get_unembed  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, module_for_component, parse_component_site  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context, top_vocab_with_classes  # noqa: E402
from phase748_natural_route_suppressor_matrix import group_competitors_by_route, margin, route_max_logits, selected_distribution, js_divergence  # noqa: E402
from phase749_suppressor_component_decomposition import (  # noqa: E402
    DEFAULT_COMPONENTS,
    capture_oproj_inputs,
    channel_scores,
    direct_delta_score,
    projected_head_deltas,
    route_token_ids,
)


OUT_ROOT = Path("results/glm5_phase750_natural_subunit_suppressor_necessity")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def component_list_for(args) -> list[str]:
    if args.components:
        return [x.strip() for x in args.components.split(",") if x.strip()]
    comps = list(DEFAULT_COMPONENTS[args.model])
    return comps[: args.max_components] if args.max_components else comps


def install_attn_head_erase(model, site: str, heads: list[int]) -> Callable[[], list[Any]]:
    layer_idx, component = parse_component_site(site)
    if component != "attn_out":
        raise ValueError(f"not attention site: {site}")
    layer = get_layers(model)[layer_idx]
    attn = get_attention_module(layer)
    o_proj = get_o_proj(attn)
    n_heads = get_num_heads(model, attn)
    in_features = int(o_proj.in_features)
    if in_features % n_heads != 0:
        raise RuntimeError(f"{site}: o_proj input dim {in_features} not divisible by heads {n_heads}")
    head_dim = in_features // n_heads
    head_ids = sorted({int(h) for h in heads if 0 <= int(h) < n_heads})

    def install() -> list[Any]:
        def pre_hook(_module, inputs):
            x = inputs[0]
            y = x.clone()
            for h in head_ids:
                start = h * head_dim
                y[:, -1, start : start + head_dim] = 0
            return (y,) + tuple(inputs[1:])

        return [o_proj.register_forward_pre_hook(pre_hook)]

    return install


def install_mlp_channel_erase(model, site: str, channels: list[int]) -> Callable[[], list[Any]]:
    _layer_idx, component = parse_component_site(site)
    if component != "mlp_out":
        raise ValueError(f"not mlp site: {site}")
    module = module_for_component(model, site)
    channel_ids = sorted({int(c) for c in channels})

    def install() -> list[Any]:
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            width = y_new.shape[-1]
            idxs = [c for c in channel_ids if 0 <= c < width]
            if idxs:
                y_new[:, -1, idxs] = 0
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        return [module.register_forward_hook(hook)]

    return install


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
    base_route_max = route_max_logits(logits, route_groups)
    if not base_route_max:
        return None
    selected_ids = [int(target_id), int(ctx["recipient_id"])]
    for group in route_groups:
        selected_ids.extend(int(t["token_id"]) for t in group["tokens"])
    return {
        "vocab": vocab,
        "route_groups": route_groups,
        "route_max": base_route_max,
        "selected_ids": selected_ids,
        "selected_dist": selected_distribution(logits, selected_ids),
    }


def classify_natural_effect(
    base_top1: bool,
    after_top1: bool,
    target_drop: float,
    total_route_release: float,
    mean_margin_drop: float,
    coverage: int,
) -> str:
    if base_top1 and not after_top1 and target_drop > 0.10 and total_route_release > 0.20 and mean_margin_drop > 0.20:
        return "natural_closure_necessity_candidate"
    if target_drop > 0.20 and total_route_release > 0.20 and mean_margin_drop > 0.20 and coverage >= 2:
        return "natural_suppressor_necessity_candidate"
    if total_route_release > 0.20 and mean_margin_drop > 0.20 and coverage >= 2:
        return "route_guard_necessity_candidate"
    if target_drop > 0.20 and mean_margin_drop > 0.10:
        return "target_support_necessity_candidate"
    if mean_margin_drop < -0.20:
        return "erase_improves_or_inverse_effect"
    return "small_or_no_effect"


def eval_natural_erase(
    model,
    tokenizer,
    device,
    target_item: dict[str, Any],
    contrast_item: dict[str, Any],
    context_name: str,
    site: str,
    subunit_kind: str,
    subunit_id: str,
    subunit_meta: dict[str, Any],
    install_erase: Callable[[], list[Any]],
    args,
) -> dict[str, Any] | None:
    ids = tokenizer.encode(prompt_for(target_item), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, target_item, contrast_item)
    target_id = int(ctx["donor_id"])
    contrast_id = int(ctx["recipient_id"])
    base_state = capture_state(model, device, ids, [])
    base_logits = base_state["logits"]
    route_ctx = build_route_context(base_logits, tokenizer, ctx, target_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return None
    handles = install_erase()
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        after_logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()

    base_target_diag = logit_diag(base_logits, target_id)
    after_target_diag = logit_diag(after_logits, target_id)
    base_contrast_diag = logit_diag(base_logits, contrast_id)
    after_contrast_diag = logit_diag(after_logits, contrast_id)
    after_vocab = top_vocab_with_classes(after_logits, tokenizer, ctx, args.top_k_vocab)
    after_route_max = route_max_logits(after_logits, route_ctx["route_groups"])
    after_dist = selected_distribution(after_logits, route_ctx["selected_ids"])
    target_drop = float(base_logits[target_id].item() - after_logits[target_id].item())
    contrast_gain = float(after_logits[contrast_id].item() - base_logits[contrast_id].item())
    route_matrix = {}
    route_releases = []
    margin_drops = []
    for cls, before in route_ctx["route_max"].items():
        after = after_route_max.get(cls)
        if not after:
            continue
        release = float(after["max_logit"]) - float(before["max_logit"])
        before_margin = float(base_logits[target_id].item()) - float(before["max_logit"])
        after_margin = float(after_logits[target_id].item()) - float(after["max_logit"])
        margin_drop = before_margin - after_margin
        route_releases.append(max(0.0, release))
        margin_drops.append(margin_drop)
        route_matrix[cls] = {
            "base_route_max_token_id": before["max_token_id"],
            "base_route_max_token_text": before["max_token_text"],
            "after_route_max_token_id": after["max_token_id"],
            "after_route_max_token_text": after["max_token_text"],
            "route_release_after_erase": release,
            "margin_drop_target_vs_route": margin_drop,
        }
    total_release = float(sum(route_releases))
    coverage = sum(1 for v in route_matrix.values() if float(v["route_release_after_erase"]) > 0.05)
    mean_margin_drop = safe_mean(margin_drops) or 0.0
    return {
        "context_name": context_name,
        "target_object": target_item["object"],
        "target_relation": target_item["relation"],
        "target_answer": target_item["answer"],
        "contrast_answer": contrast_item["answer"],
        "site": site,
        "subunit_kind": subunit_kind,
        "subunit_id": subunit_id,
        "subunit_meta": subunit_meta,
        "erase_operation": "zero_last_token_subunit_output",
        "base_top_token_id": int(route_ctx["vocab"][0]["token_id"]),
        "base_top_token_text": route_ctx["vocab"][0]["token_text"],
        "base_top_token_class": route_ctx["vocab"][0]["class"],
        "after_top_token_id": int(after_vocab[0]["token_id"]),
        "after_top_token_text": after_vocab[0]["token_text"],
        "after_top_token_class": after_vocab[0]["class"],
        "base_target_rank": base_target_diag["target_rank"],
        "after_target_rank": after_target_diag["target_rank"],
        "base_target_top1": base_target_diag["target_top1"],
        "after_target_top1": after_target_diag["target_top1"],
        "base_contrast_rank": base_contrast_diag["target_rank"],
        "after_contrast_rank": after_contrast_diag["target_rank"],
        "target_logit_drop_after_erase": target_drop,
        "contrast_logit_gain_after_erase": contrast_gain,
        "base_margin_target_vs_contrast": margin(base_logits, target_id, contrast_id),
        "after_margin_target_vs_contrast": margin(after_logits, target_id, contrast_id),
        "margin_drop_target_vs_contrast": margin(base_logits, target_id, contrast_id) - margin(after_logits, target_id, contrast_id),
        "total_positive_route_release_after_erase": total_release,
        "route_release_coverage": coverage,
        "mean_margin_drop_target_vs_routes": mean_margin_drop,
        "route_release_matrix": route_matrix,
        "readout_jsd_on_selected_vocab": js_divergence(route_ctx["selected_dist"], after_dist),
        "effect_guess": classify_natural_effect(
            bool(base_target_diag["target_top1"]),
            bool(after_target_diag["target_top1"]),
            target_drop,
            total_release,
            mean_margin_drop,
            coverage,
        ),
    }


def build_candidates_for_pair(
    model,
    tokenizer,
    device,
    donor: dict[str, Any],
    recipient: dict[str, Any],
    components: list[str],
    args,
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, donor, recipient)
    donor_id = int(ctx["donor_id"])
    donor_state = capture_state(model, device, donor_ids, components)
    recipient_state = capture_state(model, device, recipient_ids, components)
    route_ctx = build_route_context(donor_state["logits"], tokenizer, ctx, donor_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return []
    route_ids = route_token_ids(route_ctx["route_max"])
    attn_layers = [parse_component_site(c)[0] for c in components if parse_component_site(c)[1] == "attn_out"]
    donor_oproj = capture_oproj_inputs(model, device, donor_ids, attn_layers) if attn_layers else {}
    recipient_oproj = capture_oproj_inputs(model, device, recipient_ids, attn_layers) if attn_layers else {}
    candidates: list[dict[str, Any]] = []

    for site in components:
        layer, component = parse_component_site(site)
        whole_delta = donor_state["components"][site] - recipient_state["components"][site]
        if component == "attn_out" and layer in donor_oproj and layer in recipient_oproj:
            head_rows = projected_head_deltas(model, layer, donor_oproj[layer], recipient_oproj[layer])
            for row in head_rows:
                row["direct"] = direct_delta_score(row["delta"], unembed, donor_id, route_ids)
            head_rows.sort(key=lambda r: (r["direct"]["direct_total_route_suppression"], r["direct"]["direct_mean_margin_gain"]), reverse=True)
            for k in args.headset_sizes:
                heads = head_rows[: min(k, len(head_rows))]
                if not heads:
                    continue
                head_ids = [int(h["head"]) for h in heads]
                candidates.append(
                    {
                        "site": site,
                        "subunit_kind": "attn_headset",
                        "subunit_id": f"{site}:topH{k}",
                        "subunit_meta": {"heads": head_ids},
                        "install_erase": install_attn_head_erase(model, site, head_ids),
                    }
                )
            for row in head_rows[: args.individual_heads]:
                h = int(row["head"])
                candidates.append(
                    {
                        "site": site,
                        "subunit_kind": "attn_head",
                        "subunit_id": f"{site}:H{h}",
                        "subunit_meta": {"head": h},
                        "install_erase": install_attn_head_erase(model, site, [h]),
                    }
                )

        if component == "mlp_out":
            scored = channel_scores(whole_delta, unembed, donor_id, route_ids)
            top_indices = [int(r["index"]) for r in scored]
            for k in args.channelset_sizes:
                idxs = top_indices[: min(k, len(top_indices))]
                if not idxs:
                    continue
                candidates.append(
                    {
                        "site": site,
                        "subunit_kind": "mlp_channelset",
                        "subunit_id": f"{site}:topC{k}",
                        "subunit_meta": {"channels": idxs[:128], "n_channels": len(idxs), "selection": "top_direct_score"},
                        "install_erase": install_mlp_channel_erase(model, site, idxs),
                    }
                )
            for ch in scored[: args.individual_channels]:
                idx = int(ch["index"])
                candidates.append(
                    {
                        "site": site,
                        "subunit_kind": "mlp_channel",
                        "subunit_id": f"{site}:C{idx}",
                        "subunit_meta": {"channel": idx},
                        "install_erase": install_mlp_channel_erase(model, site, [idx]),
                    }
                )
    return candidates


def audit_pair(
    model,
    tokenizer,
    device,
    args,
    pair: dict[str, Any],
    audit: dict[str, Any],
    components: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    candidates = build_candidates_for_pair(model, tokenizer, device, donor, recipient, components, args, unembed)
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        for context_name, target_item, contrast_item in [
            ("natural_donor", donor, recipient),
            ("natural_recipient", recipient, donor),
        ][: 1 if args.donor_context_only else 2]:
            row = eval_natural_erase(
                model,
                tokenizer,
                device,
                target_item,
                contrast_item,
                context_name,
                cand["site"],
                cand["subunit_kind"],
                cand["subunit_id"],
                cand["subunit_meta"],
                cand["install_erase"],
                args,
            )
            if row is None:
                continue
            row.update(
                {
                    "model": args.model,
                    "pair_id": pair["pair_id"],
                    "direction": direction_name,
                    "intervention_label": intervention_label(audit["intervention"]),
                }
            )
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["context_name"], row["site"], row["subunit_kind"], row["subunit_id"])].append(row)
    summary = []
    for (context_name, site, kind, subunit_id), vals in groups.items():
        n = len(vals)
        route_sums: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            for cls, item in row["route_release_matrix"].items():
                route_sums[cls].append(float(item["route_release_after_erase"]))
        summary.append(
            {
                "context_name": context_name,
                "site": site,
                "subunit_kind": kind,
                "subunit_id": subunit_id,
                "n": n,
                "base_target_top1_rate": sum(1 for v in vals if v["base_target_top1"]) / n,
                "after_target_top1_rate": sum(1 for v in vals if v["after_target_top1"]) / n,
                "top1_loss_rate": sum(1 for v in vals if v["base_target_top1"] and not v["after_target_top1"]) / n,
                "mean_target_logit_drop_after_erase": safe_mean([v["target_logit_drop_after_erase"] for v in vals]),
                "mean_contrast_logit_gain_after_erase": safe_mean([v["contrast_logit_gain_after_erase"] for v in vals]),
                "mean_margin_drop_target_vs_contrast": safe_mean([v["margin_drop_target_vs_contrast"] for v in vals]),
                "mean_total_positive_route_release_after_erase": safe_mean([v["total_positive_route_release_after_erase"] for v in vals]),
                "mean_route_release_coverage": safe_mean([v["route_release_coverage"] for v in vals]),
                "mean_margin_drop_target_vs_routes": safe_mean([v["mean_margin_drop_target_vs_routes"] for v in vals]),
                "mean_readout_jsd_on_selected_vocab": safe_mean([v["readout_jsd_on_selected_vocab"] for v in vals]),
                "effect_guess_counts": dict(Counter(v["effect_guess"] for v in vals)),
                "dominant_effect_guess": Counter(v["effect_guess"] for v in vals).most_common(1)[0][0],
                "route_summary": {
                    cls: {
                        "mean_route_release": safe_mean(vals2),
                        "positive_release_rate": sum(1 for x in vals2 if x > 0.05) / len(vals2),
                        "n": len(vals2),
                    }
                    for cls, vals2 in sorted(route_sums.items())
                },
            }
        )
    summary.sort(
        key=lambda r: (
            r["mean_total_positive_route_release_after_erase"] or 0,
            r["mean_margin_drop_target_vs_routes"] or 0,
            r["mean_target_logit_drop_after_erase"] or 0,
        ),
        reverse=True,
    )
    return {
        "phase": 750,
        "title": "Natural Subunit Suppressor Necessity Test",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "components": component_list_for(args),
        "summary": summary,
        "top_natural_necessity_candidates": summary[:24],
        "strict_interpretation": "Natural erase evidence. It tests whether a subunit is needed in ordinary forward passes, but erase can create off-manifold perturbations and still does not prove neuron-level origin.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audits_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    components = component_list_for(args)
    log(f"{args.model}/{args.round_name}: natural necessity pairs={len(pairs)} components={components} audits={len(audits_payload['audits'])}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    unembed = get_unembed(model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audits_payload["audits"]:
                rows.extend(audit_pair(model, tokenizer, device, args, pair, audit, components, unembed))
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: natural necessity scan {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase750_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase750_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "top": summary["top_natural_necessity_candidates"][:12]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase750_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 750,
        "title": "Natural Subunit Suppressor Necessity Test",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "Natural erase evidence. It can show necessity-like behavior, but erase is an intervention and does not by itself prove complete natural coding origin.",
    }
    write_json(out_dir / "phase750_cross_model_summary.json", payload)
    lines = [
        f"# Phase 750 Natural Subunit Suppressor Necessity Test ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence: natural forward erase of selected headsets/channelsets, no donor delta installed.",
        "",
        "| model | context | site | subunit | kind | n | base top1 | after top1 | top1 loss | target drop | route release | coverage | margin drop | effect |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_natural_necessity_candidates", [])[:18]:
            lines.append(
                f"| {model_name} | {row['context_name']} | {row['site']} | {row['subunit_id']} | {row['subunit_kind']} | {row['n']} | "
                f"{(row.get('base_target_top1_rate') or 0):.3f} | "
                f"{(row.get('after_target_top1_rate') or 0):.3f} | "
                f"{(row.get('top1_loss_rate') or 0):.3f} | "
                f"{(row.get('mean_target_logit_drop_after_erase') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_release_after_erase') or 0):.3f} | "
                f"{(row.get('mean_route_release_coverage') or 0):.2f} | "
                f"{(row.get('mean_margin_drop_target_vs_routes') or 0):.3f} | "
                f"`{row.get('dominant_effect_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- This phase tests natural necessity-like behavior, not donor-recipient patch success.",
            "- Attention erase zeroes selected o_proj input head slices at the final token.",
            "- MLP erase zeroes selected residual output channels at the final token; this is still not neuron-level evidence.",
            "- If erase releases competitor routes or drops the target, the subunit is a natural-route candidate, not yet a complete mechanism proof.",
            "",
        ]
    )
    (out_dir / "phase750_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
            "channelset_sizes": args.channelset_sizes,
            "contexts": ["natural_donor"] if args.donor_context_only else ["natural_donor", "natural_recipient"],
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
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--top-audits", type=int, default=2)
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
    parser.add_argument("--headset-sizes", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument("--individual-heads", type=int, default=1)
    parser.add_argument("--channelset-sizes", type=int, nargs="*", default=[16, 64])
    parser.add_argument("--individual-channels", type=int, default=1)
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
