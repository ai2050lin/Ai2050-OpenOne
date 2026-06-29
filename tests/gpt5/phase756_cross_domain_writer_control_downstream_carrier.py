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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean, select_evenly  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, combine_installers, install_component_edit, parse_component_site  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context, top_vocab_with_classes  # noqa: E402
from phase748_natural_route_suppressor_matrix import js_divergence, route_max_logits, selected_distribution  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score, route_token_ids  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    build_route_context,
    capture_attention_value_state,
    eval_after_logits,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group, norm  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import (  # noqa: E402
    FIXED_HEADS,
    get_first_token_id,
    select_global_pairs,
)


OUT_ROOT = Path("results/glm5_phase756_cross_domain_writer_control_downstream_carrier")

DEFAULT_SOURCE_GROUPS = ["records_all", "target_record_line", "target_value_tokens"]

MODEL_DOWNSTREAM_WINDOWS = {
    "qwen3": [34, 35],
    "glm4": [36, 37, 38, 39],
    "deepseek7b": [23, 24, 25, 26, 27],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidate_key(site: str, head: int) -> str:
    return f"{site}:H{int(head)}"


def source_groups_for(args) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def base_candidates_for_model(model_name: str, max_candidates: int) -> list[dict[str, Any]]:
    rows = []
    for idx, cand in enumerate(FIXED_HEADS[model_name][:max_candidates]):
        kind = "phase755_top_candidate" if idx < 2 else "phase755_secondary_candidate"
        rows.append(
            {
                "site": cand["site"],
                "head": int(cand["head"]),
                "subunit_id": candidate_key(cand["site"], int(cand["head"])),
                "candidate_kind": kind,
                "selection": cand.get("source", "phase755"),
                "control_of": None,
            }
        )
    return rows


def expanded_candidates(model, model_name: str, args) -> list[dict[str, Any]]:
    rows = base_candidates_for_model(model_name, args.max_candidates)
    seen = {(r["site"], int(r["head"])) for r in rows}
    if not args.include_controls:
        return rows
    for cand in list(rows):
        layer, _component = parse_component_site(cand["site"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if n_heads <= 1:
            continue
        added = 0
        for offset in [args.control_offset, args.control_offset + 5, args.control_offset + 11, 1]:
            h = (int(cand["head"]) + offset) % n_heads
            key = (cand["site"], h)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "site": cand["site"],
                    "head": h,
                    "subunit_id": candidate_key(cand["site"], h),
                    "candidate_kind": "same_layer_control_head",
                    "selection": f"deterministic_offset_{offset}",
                    "control_of": cand["subunit_id"],
                }
            )
            added += 1
            if added >= args.controls_per_candidate:
                break
    return rows


def downstream_sites_for(model_name: str, writer_site: str, n_layers: int, max_sites: int) -> list[str]:
    writer_layer, _component = parse_component_site(writer_site)
    layers = [li for li in MODEL_DOWNSTREAM_WINDOWS.get(model_name, []) if writer_layer < li < n_layers]
    if not layers:
        layers = [li for li in range(writer_layer + 1, min(n_layers, writer_layer + 1 + max(1, max_sites // 2)))]
    sites: list[str] = []
    for li in layers:
        sites.append(f"L{li}:attn_out")
        sites.append(f"L{li}:mlp_out")
    return sites[:max_sites]


def unique_downstream_sites(model_name: str, candidates: list[dict[str, Any]], n_layers: int, max_sites: int) -> list[str]:
    seen: list[str] = []
    for cand in candidates:
        for site in downstream_sites_for(model_name, cand["site"], n_layers, max_sites):
            if site not in seen:
                seen.append(site)
    return seen


def recovery_fraction(erase_drop: float, recovered: float) -> float | None:
    if erase_drop <= 1e-6:
        return None
    return recovered / erase_drop


def run_logits(model, device, ids: list[int], install: Callable[[], list[Any]] | None = None) -> torch.Tensor:
    handles = install() if install else []
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def audit_pair(
    model,
    tokenizer,
    device,
    args,
    pair: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    all_downstream_sites: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    target = pair["explicit_profile"]
    contrast = pair["conflict_profile"]
    candidate_layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_attention_value_state(model, tokenizer, device, target, candidate_layers)
    ctx = taxonomy_context(tokenizer, target, contrast)
    target_id = get_first_token_id(tokenizer, target["answer"])
    contrast_id = get_first_token_id(tokenizer, contrast["answer"])
    route_ctx = build_route_context(state["logits"], tokenizer, ctx, target_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return []
    route_ids = route_token_ids(route_ctx["route_max"])
    base_component_state = capture_state(model, device, state["ids"], all_downstream_sites)
    top_vocab = top_vocab_with_classes(state["logits"], tokenizer, ctx, args.top_k_vocab)
    top = top_vocab[0] if top_vocab else {}
    target_diag = logit_diag(state["logits"], target_id)
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "base_observation",
            "pair_id": pair["pair_id"],
            "domain": target["domain"],
            "object": target["object"],
            "relation": target["relation"],
            "target_answer": target["answer"],
            "contrast_answer": contrast["answer"],
            "target_token_id": target_id,
            "contrast_token_id": contrast_id,
            "target_rank": target_diag["target_rank"],
            "target_top1": target_diag["target_top1"],
            "top_token_id": int(top.get("token_id", -1)),
            "top_token_text": top.get("token_text", ""),
            "top_token_class": top.get("class", ""),
        }
    ]

    answer_pos = state["answer_pos"]
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if not (0 <= head < n_heads):
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        cand_downstream_sites = downstream_sites_for(args.model, site, len(get_layers(model)), args.max_downstream_sites)
        for source_group in source_groups:
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
            projected = project_source_contribution(model, layer, [head], contribution)
            direct = direct_delta_score(projected, unembed, target_id, route_ids)
            removal_install = install_source_contribution_removal(model, site, [head], contribution)
            erased_state = capture_state(model, device, state["ids"], cand_downstream_sites, removal_install)
            erase_metrics = eval_after_logits(state["logits"], erased_state["logits"], route_ctx, target_id, contrast_id)
            rows.append(
                {
                    "row_kind": "source_removal_control",
                    "pair_id": pair["pair_id"],
                    "domain": target["domain"],
                    "object": target["object"],
                    "relation": target["relation"],
                    "target_answer": target["answer"],
                    "contrast_answer": contrast["answer"],
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "subunit_id": cand["subunit_id"],
                    "candidate_kind": cand["candidate_kind"],
                    "selection": cand["selection"],
                    "control_of": cand.get("control_of"),
                    "source_group": source_group,
                    "source_positions_n": len(src_positions),
                    "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                    "source_projected_delta_norm": norm(projected),
                    "source_direct_score": direct,
                    **erase_metrics,
                }
            )
            for downstream_site in cand_downstream_sites:
                if downstream_site not in base_component_state["components"] or downstream_site not in erased_state["components"]:
                    continue
                restore_install = combine_installers(
                    removal_install,
                    lambda downstream_site=downstream_site: install_component_edit(
                        model,
                        downstream_site,
                        replace_vec=base_component_state["components"][downstream_site],
                    ),
                )
                restored_logits = run_logits(model, device, state["ids"], restore_install)
                restore_metrics = eval_after_logits(state["logits"], restored_logits, route_ctx, target_id, contrast_id)
                recovered = float(erase_metrics["target_logit_drop"] - restore_metrics["target_logit_drop"])
                release_reduced = float(erase_metrics["total_positive_route_release"] - restore_metrics["total_positive_route_release"])
                frac = recovery_fraction(float(erase_metrics["target_logit_drop"]), recovered)
                rows.append(
                    {
                        "row_kind": "downstream_component_restore",
                        "pair_id": pair["pair_id"],
                        "domain": target["domain"],
                        "object": target["object"],
                        "relation": target["relation"],
                        "target_answer": target["answer"],
                        "contrast_answer": contrast["answer"],
                        "site": site,
                        "layer": layer,
                        "head": head,
                        "subunit_id": cand["subunit_id"],
                        "candidate_kind": cand["candidate_kind"],
                        "selection": cand["selection"],
                        "control_of": cand.get("control_of"),
                        "source_group": source_group,
                        "downstream_site": downstream_site,
                        "downstream_component_delta_norm_after_removal": norm(erased_state["components"][downstream_site] - base_component_state["components"][downstream_site]),
                        "erase_target_logit_drop": erase_metrics["target_logit_drop"],
                        "erase_total_positive_route_release": erase_metrics["total_positive_route_release"],
                        "restored_target_logit_drop": restore_metrics["target_logit_drop"],
                        "restored_total_positive_route_release": restore_metrics["total_positive_route_release"],
                        "target_logit_recovered_by_restore": recovered,
                        "route_release_reduced_by_restore": release_reduced,
                        "target_recovery_fraction": frac,
                        "effective_restore": bool(
                            erase_metrics["target_logit_drop"] > args.min_erase_drop
                            and recovered > args.min_restore_recovery
                            and (frac or 0.0) > args.min_recovery_fraction
                        ),
                        **{f"restored_{k}": v for k, v in restore_metrics.items()},
                    }
                )
    return rows


def classify_removal(vals: list[dict[str, Any]]) -> str:
    n = len(vals)
    if not n:
        return "empty"
    support = sum(1 for v in vals if v["target_logit_drop"] > 0.20) / n
    release = sum(1 for v in vals if v["total_positive_route_release"] > 0.20) / n
    mean_drop = safe_mean([v["target_logit_drop"] for v in vals]) or 0.0
    mean_rel = safe_mean([v["total_positive_route_release"] for v in vals]) or 0.0
    domains = {v["domain"] for v in vals}
    domain_support = []
    for d in domains:
        xs = [v for v in vals if v["domain"] == d]
        domain_support.append(sum(1 for v in xs if v["target_logit_drop"] > 0.20) / len(xs))
    active_domains = sum(1 for x in domain_support if x >= 0.50)
    if active_domains >= 4 and support >= 0.50 and mean_drop >= 0.30:
        if release >= 0.30 or mean_rel >= 0.20:
            return "cross_domain_writer_guard_candidate"
        return "cross_domain_writer_candidate"
    if release >= 0.35 and mean_rel >= 0.20:
        return "route_guard_candidate"
    if support >= 0.30 and mean_drop >= 0.20:
        return "partial_writer_candidate"
    return "control_or_weak"


def classify_carrier(vals: list[dict[str, Any]]) -> str:
    n = len(vals)
    if not n:
        return "empty"
    success = sum(1 for v in vals if v["effective_restore"]) / n
    mean_frac = safe_mean([v["target_recovery_fraction"] for v in vals]) or 0.0
    mean_rec = safe_mean([v["target_logit_recovered_by_restore"] for v in vals]) or 0.0
    mean_rel = safe_mean([v["route_release_reduced_by_restore"] for v in vals]) or 0.0
    if success >= 0.40 and mean_frac >= 0.35 and mean_rec >= 0.10:
        if mean_rel >= 0.05:
            return "downstream_writer_guard_carrier_candidate"
        return "downstream_target_carrier_candidate"
    if success >= 0.25 and mean_frac >= 0.20 and mean_rec >= 0.06:
        return "partial_downstream_carrier_candidate"
    if mean_rec < -0.05:
        return "anti_restore_or_off_path"
    return "weak_or_unclear"


def summarize_removals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rem = [r for r in rows if r["row_kind"] == "source_removal_control"]
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rem:
        groups[(row["site"], int(row["head"]), row["candidate_kind"], row["source_group"])].append(row)
    out: list[dict[str, Any]] = []
    for (site, head, kind, source), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "site": site,
                "head": head,
                "subunit_id": candidate_key(site, head),
                "candidate_kind": kind,
                "source_group": source,
                "n": n,
                "domains": sorted({v["domain"] for v in vals}),
                "relations": sorted({v["relation"] for v in vals}),
                "mean_attention_mass_to_source": safe_mean([v["attention_mass_to_source"] for v in vals]),
                "mean_source_target_logit_contribution": safe_mean([v["source_direct_score"]["direct_target_boost"] for v in vals]),
                "mean_source_total_route_suppression_contribution": safe_mean([v["source_direct_score"]["direct_total_route_suppression"] for v in vals]),
                "mean_target_logit_drop_after_source_removal": safe_mean([v["target_logit_drop"] for v in vals]),
                "support_rate_drop_gt_0_20": sum(1 for v in vals if v["target_logit_drop"] > 0.20) / n,
                "mean_total_positive_route_release_after_source_removal": safe_mean([v["total_positive_route_release"] for v in vals]),
                "route_guard_rate_release_gt_0_20": sum(1 for v in vals if v["total_positive_route_release"] > 0.20) / n,
                "top1_loss_rate": sum(1 for v in vals if v["top1_loss"]) / n,
                "removal_role_guess": classify_removal(vals),
            }
        )
    out.sort(
        key=lambda r: (
            r["support_rate_drop_gt_0_20"],
            r["mean_target_logit_drop_after_source_removal"] or 0.0,
            r["route_guard_rate_release_gt_0_20"],
            r["mean_total_positive_route_release_after_source_removal"] or 0.0,
        ),
        reverse=True,
    )
    return out


def summarize_downstream(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rest = [r for r in rows if r["row_kind"] == "downstream_component_restore"]
    groups: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rest:
        groups[(row["site"], int(row["head"]), row["candidate_kind"], row["source_group"], row["downstream_site"])].append(row)
    out: list[dict[str, Any]] = []
    for (site, head, kind, source, downstream_site), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "site": site,
                "head": head,
                "subunit_id": candidate_key(site, head),
                "candidate_kind": kind,
                "source_group": source,
                "downstream_site": downstream_site,
                "n": n,
                "domains": sorted({v["domain"] for v in vals}),
                "relations": sorted({v["relation"] for v in vals}),
                "mean_downstream_component_delta_norm_after_removal": safe_mean([v["downstream_component_delta_norm_after_removal"] for v in vals]),
                "mean_erase_target_logit_drop": safe_mean([v["erase_target_logit_drop"] for v in vals]),
                "mean_restored_target_logit_drop": safe_mean([v["restored_target_logit_drop"] for v in vals]),
                "mean_target_logit_recovered_by_restore": safe_mean([v["target_logit_recovered_by_restore"] for v in vals]),
                "mean_target_recovery_fraction": safe_mean([v["target_recovery_fraction"] for v in vals]),
                "mean_erase_route_release": safe_mean([v["erase_total_positive_route_release"] for v in vals]),
                "mean_restored_route_release": safe_mean([v["restored_total_positive_route_release"] for v in vals]),
                "mean_route_release_reduced_by_restore": safe_mean([v["route_release_reduced_by_restore"] for v in vals]),
                "effective_restore_rate": sum(1 for v in vals if v["effective_restore"]) / n,
                "carrier_role_guess": classify_carrier(vals),
            }
        )
    out.sort(
        key=lambda r: (
            r["effective_restore_rate"],
            r["mean_target_recovery_fraction"] or 0.0,
            r["mean_target_logit_recovered_by_restore"] or 0.0,
            r["mean_route_release_reduced_by_restore"] or 0.0,
        ),
        reverse=True,
    )
    return out


def control_baseline(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["candidate_kind"]].append(row)
    return {
        kind: {
            "n_groups": len(vals),
            "mean_support_rate": safe_mean([v["support_rate_drop_gt_0_20"] for v in vals]),
            "mean_drop": safe_mean([v["mean_target_logit_drop_after_source_removal"] for v in vals]),
            "mean_guard_rate": safe_mean([v["route_guard_rate_release_gt_0_20"] for v in vals]),
            "mean_release": safe_mean([v["mean_total_positive_route_release_after_source_removal"] for v in vals]),
            "role_counts": dict(Counter(v["removal_role_guess"] for v in vals)),
        }
        for kind, vals in sorted(grouped.items())
    }


def build_summary(args, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], source_groups: list[str], all_downstream_sites: list[str], attn_impl: str) -> dict[str, Any]:
    removal_summary = summarize_removals(rows)
    downstream_summary = summarize_downstream(rows)
    return {
        "phase": 756,
        "title": "Cross-Domain Writer Control and Downstream Carrier Test",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_base_observations": sum(1 for r in rows if r["row_kind"] == "base_observation"),
        "n_source_removals": sum(1 for r in rows if r["row_kind"] == "source_removal_control"),
        "n_downstream_restores": sum(1 for r in rows if r["row_kind"] == "downstream_component_restore"),
        "candidates": candidates,
        "source_groups": source_groups,
        "downstream_sites": all_downstream_sites,
        "control_baseline": control_baseline(removal_summary),
        "top_controlled_writer_candidates": removal_summary[:32],
        "top_downstream_carrier_candidates": downstream_summary[:48],
        "strict_interpretation": "Control heads test specificity. Downstream component restore is coarse carrier evidence; it restores whole component output at answer position and is not neuron-level sufficiency.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_global_pairs(args.max_pairs)
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} seed_candidates={args.max_candidates} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        candidates = expanded_candidates(model, args.model, args)
        all_downstream_sites = unique_downstream_sites(args.model, candidates, len(get_layers(model)), args.max_downstream_sites)
        unembed = get_unembed(model)
        log(f"{args.model}: expanded_candidates={len(candidates)} downstream_sites={all_downstream_sites}")
        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(pairs, 1):
            rows.extend(audit_pair(model, tokenizer, device, args, pair, candidates, source_groups, all_downstream_sites, unembed))
            if idx % args.log_every == 0 or idx == len(pairs):
                log(f"{args.model}: control+carrier {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, candidates, source_groups, all_downstream_sites, attn_impl)
    write_jsonl(out_dir / f"phase756_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase756_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "control_baseline": summary["control_baseline"],
                "top_writers": summary["top_controlled_writer_candidates"][:8],
                "top_carriers": summary["top_downstream_carrier_candidates"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase756_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 756,
        "title": "Cross-Domain Writer Control and Downstream Carrier Test",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "This phase adds same-layer controls and coarse downstream restore. It can promote a path to controlled carrier evidence, not to neuron-level language invariance.",
    }
    write_json(out_dir / "phase756_cross_model_summary.json", payload)
    lines = [
        f"# Phase 756 Cross-Domain Writer Control and Downstream Carrier Test ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence: fixed source removal vs same-layer controls, then downstream component restoration under the same removal.",
        "",
        "## Candidate vs Control Baseline",
        "",
        "| model | candidate kind | groups | mean support | mean drop | mean guard | mean release | roles |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        for kind, row in summary.get("control_baseline", {}).items():
            lines.append(
                f"| {model_name} | `{kind}` | {row['n_groups']} | "
                f"{(row.get('mean_support_rate') or 0):.3f} | {(row.get('mean_drop') or 0):.3f} | "
                f"{(row.get('mean_guard_rate') or 0):.3f} | {(row.get('mean_release') or 0):.3f} | "
                f"`{row.get('role_counts')}` |"
            )
    lines.extend(
        [
            "",
            "## Top Controlled Writer / Guard Candidates",
            "",
            "| model | kind | site | head | source | n | domains | support | drop | guard | release | top1 loss | guess |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_controlled_writer_candidates", [])[:12]:
            lines.append(
                f"| {model_name} | `{row['candidate_kind']}` | {row['site']} | {row['head']} | {row['source_group']} | {row['n']} | "
                f"{len(row.get('domains') or [])} | {(row.get('support_rate_drop_gt_0_20') or 0):.3f} | "
                f"{(row.get('mean_target_logit_drop_after_source_removal') or 0):.3f} | "
                f"{(row.get('route_guard_rate_release_gt_0_20') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_release_after_source_removal') or 0):.3f} | "
                f"{(row.get('top1_loss_rate') or 0):.3f} | `{row.get('removal_role_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Top Downstream Carrier Restores",
            "",
            "| model | kind | writer | source | downstream | n | restore rate | erase drop | restored drop | recovered | recovery frac | release reduced | guess |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_downstream_carrier_candidates", [])[:16]:
            lines.append(
                f"| {model_name} | `{row['candidate_kind']}` | {row['subunit_id']} | {row['source_group']} | {row['downstream_site']} | {row['n']} | "
                f"{(row.get('effective_restore_rate') or 0):.3f} | "
                f"{(row.get('mean_erase_target_logit_drop') or 0):.3f} | "
                f"{(row.get('mean_restored_target_logit_drop') or 0):.3f} | "
                f"{(row.get('mean_target_logit_recovered_by_restore') or 0):.3f} | "
                f"{(row.get('mean_target_recovery_fraction') or 0):.3f} | "
                f"{(row.get('mean_route_release_reduced_by_restore') or 0):.3f} | `{row.get('carrier_role_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- A candidate stronger than same-layer controls supports specificity, not universality.",
            "- Downstream restore replaces the whole downstream component output at the answer position; it localizes a coarse carrier, not a neuron-level code.",
            "- If qwen3 / GLM4 remain weak, DS7B results must stay model-local.",
            "",
        ]
    )
    (out_dir / "phase756_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {
        "phase": 756,
        "round": args.round_name,
        "pairs": len(select_global_pairs(args.max_pairs)),
        "source_groups": source_groups_for(args),
        "models": {},
    }
    for model_name in MODELS:
        payload["models"][model_name] = {
            "seed_candidates": base_candidates_for_model(model_name, args.max_candidates),
            "downstream_window": MODEL_DOWNSTREAM_WINDOWS.get(model_name, []),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--include-controls", action="store_true", default=True)
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-offset", type=int, default=13)
    parser.add_argument("--max-source-groups", type=int, default=2)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--max-downstream-sites", type=int, default=6)
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
    parser.add_argument("--min-erase-drop", type=float, default=0.20)
    parser.add_argument("--min-restore-recovery", type=float, default=0.10)
    parser.add_argument("--min-recovery-fraction", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=4)
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
