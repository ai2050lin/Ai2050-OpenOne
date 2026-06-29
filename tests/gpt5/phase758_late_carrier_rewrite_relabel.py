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
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, install_component_edit, parse_component_site  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score, route_token_ids  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    build_route_context,
    capture_attention_value_state,
    eval_after_logits,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group, norm  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id, select_global_pairs  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import (  # noqa: E402
    candidate_key,
    expanded_candidates,
    recovery_fraction,
    source_groups_for,
    summarize_removals,
)


OUT_ROOT = Path("results/glm5_phase758_late_carrier_rewrite_relabel")

PRIMARY_PATH_LAYERS = {
    "qwen3": [34],
    "glm4": [36, 37],
    "deepseek7b": [23, 24],
}

# Phase 757 showed that these layers are not valid "off-path" controls.
# They are relabeled here as late carrier / rewrite candidates.
LATE_CANDIDATE_LAYERS = {
    "qwen3": [35],
    "glm4": [38, 39],
    "deepseek7b": [25, 26],
}

# True later controls are only available when the model still has layers after
# the late candidate window. If empty, the summary explicitly records that lack.
LATE_CONTROL_LAYERS = {
    "qwen3": [],
    "glm4": [],
    "deepseek7b": [27],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sites_for_layers(layers: list[int], n_layers: int, writer_layer: int) -> list[str]:
    out: list[str] = []
    for li in layers:
        if writer_layer < li < n_layers:
            out.append(f"L{li}:attn_out")
            out.append(f"L{li}:mlp_out")
    return out


def combo_defs_for(model_name: str, writer_site: str, n_layers: int, max_combos: int) -> list[dict[str, Any]]:
    writer_layer, _component = parse_component_site(writer_site)
    primary_sites = sites_for_layers(PRIMARY_PATH_LAYERS.get(model_name, []), n_layers, writer_layer)
    late_sites = sites_for_layers(LATE_CANDIDATE_LAYERS.get(model_name, []), n_layers, writer_layer)
    control_sites = sites_for_layers(LATE_CONTROL_LAYERS.get(model_name, []), n_layers, writer_layer)
    combos: list[dict[str, Any]] = []

    # Put the hypothesis-bearing combinations first so --max-combos cannot
    # accidentally truncate away the late-candidate comparison.
    if primary_sites:
        combos.append({"combo_name": "primary_all", "combo_kind": "primary_multisite_all", "sites": primary_sites})
    if late_sites:
        combos.append({"combo_name": "late_candidate_all", "combo_kind": "late_candidate_all", "sites": late_sites[: max(1, len(primary_sites))]})
    if primary_sites and late_sites:
        combos.append(
            {
                "combo_name": "primary_plus_late_all",
                "combo_kind": "primary_plus_late_all",
                "sites": (primary_sites + late_sites)[: max(1, len(primary_sites) + len(late_sites))],
            }
        )
    if control_sites:
        combos.append(
            {
                "combo_name": "late_control_same_count",
                "combo_kind": "true_late_control",
                "sites": control_sites[: max(1, min(len(late_sites) or len(primary_sites), len(control_sites)))],
            }
        )

    first_layer_sites: dict[int, list[str]] = defaultdict(list)
    for site in primary_sites:
        li, _ = parse_component_site(site)
        first_layer_sites[li].append(site)
    for li, sites in sorted(first_layer_sites.items()):
        if len(sites) >= 2:
            combos.append({"combo_name": f"L{li}:attn+mlp", "combo_kind": "same_layer_primary_pair", "sites": sites})
    late_layer_sites: dict[int, list[str]] = defaultdict(list)
    for site in late_sites:
        li, _ = parse_component_site(site)
        late_layer_sites[li].append(site)
    for li, sites in sorted(late_layer_sites.items()):
        if len(sites) >= 2:
            combos.append({"combo_name": f"L{li}:late_attn+mlp", "combo_kind": "same_layer_late_candidate_pair", "sites": sites})
    for site in primary_sites:
        combos.append({"combo_name": site, "combo_kind": "single_primary_site", "sites": [site]})
    for site in late_sites:
        combos.append({"combo_name": site, "combo_kind": "single_late_candidate_site", "sites": [site]})
    attn_chain = [s for s in primary_sites if s.endswith(":attn_out")]
    mlp_chain = [s for s in primary_sites if s.endswith(":mlp_out")]
    late_attn_chain = [s for s in late_sites if s.endswith(":attn_out")]
    late_mlp_chain = [s for s in late_sites if s.endswith(":mlp_out")]
    if len(attn_chain) >= 2:
        combos.append({"combo_name": "primary_attn_chain", "combo_kind": "cross_layer_primary_chain", "sites": attn_chain})
    if len(mlp_chain) >= 2:
        combos.append({"combo_name": "primary_mlp_chain", "combo_kind": "cross_layer_primary_chain", "sites": mlp_chain})
    if len(late_attn_chain) >= 2:
        combos.append({"combo_name": "late_attn_chain", "combo_kind": "cross_layer_late_candidate_chain", "sites": late_attn_chain})
    if len(late_mlp_chain) >= 2:
        combos.append({"combo_name": "late_mlp_chain", "combo_kind": "cross_layer_late_candidate_chain", "sites": late_mlp_chain})
    seen = set()
    dedup = []
    for combo in combos:
        key = tuple(combo["sites"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(combo)
    return dedup[:max_combos]


def unique_combo_sites(model_name: str, candidates: list[dict[str, Any]], n_layers: int, max_combos: int) -> list[str]:
    out: list[str] = []
    for cand in candidates:
        for combo in combo_defs_for(model_name, cand["site"], n_layers, max_combos):
            for site in combo["sites"]:
                if site not in out:
                    out.append(site)
    return out


def install_multisite_restore(
    model,
    removal_install: Callable[[], list[Any]],
    sites: list[str],
    base_components: dict[str, torch.Tensor],
) -> Callable[[], list[Any]]:
    def install() -> list[Any]:
        handles = removal_install()
        for site in sites:
            handles.extend(install_component_edit(model, site, replace_vec=base_components[site]))
        return handles

    return install


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
    all_combo_sites: list[str],
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
    base_state = capture_state(model, device, state["ids"], all_combo_sites)
    answer_pos = state["answer_pos"]
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if not (0 <= head < n_heads):
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        combo_defs = combo_defs_for(args.model, site, len(get_layers(model)), args.max_combos)
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
            erased_state = capture_state(model, device, state["ids"], all_combo_sites, removal_install)
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
            for combo in combo_defs:
                sites = [s for s in combo["sites"] if s in base_state["components"] and s in erased_state["components"]]
                if not sites:
                    continue
                install = install_multisite_restore(model, removal_install, sites, base_state["components"])
                restored_logits = run_logits(model, device, state["ids"], install)
                restore_metrics = eval_after_logits(state["logits"], restored_logits, route_ctx, target_id, contrast_id)
                recovered = float(erase_metrics["target_logit_drop"] - restore_metrics["target_logit_drop"])
                release_reduced = float(erase_metrics["total_positive_route_release"] - restore_metrics["total_positive_route_release"])
                frac = recovery_fraction(float(erase_metrics["target_logit_drop"]), recovered)
                rows.append(
                    {
                        "row_kind": "multisite_restore",
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
                        "combo_name": combo["combo_name"],
                        "combo_kind": combo["combo_kind"],
                        "combo_sites": sites,
                        "combo_size": len(sites),
                        "mean_combo_delta_norm_after_removal": safe_mean([norm(erased_state["components"][s] - base_state["components"][s]) for s in sites]),
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


def classify_combo(vals: list[dict[str, Any]]) -> str:
    n = len(vals)
    if not n:
        return "empty"
    success = sum(1 for v in vals if v["effective_restore"]) / n
    mean_rec = safe_mean([v["target_logit_recovered_by_restore"] for v in vals]) or 0.0
    mean_frac = safe_mean([v["target_recovery_fraction"] for v in vals]) or 0.0
    mean_rel = safe_mean([v["route_release_reduced_by_restore"] for v in vals]) or 0.0
    kinds = {v["combo_kind"] for v in vals}
    if "true_late_control" in kinds and (success >= 0.30 or mean_rec >= 0.08):
        return "true_late_control_suspicious"
    if any(k in kinds for k in {"late_candidate_all", "same_layer_late_candidate_pair", "single_late_candidate_site", "cross_layer_late_candidate_chain"}):
        if success >= 0.45 and mean_rec >= 0.10 and mean_frac >= 0.25:
            if mean_rel >= 0.03:
                return "late_writer_guard_closure_candidate"
            return "late_target_rewrite_candidate"
        if success >= 0.25 and mean_rec >= 0.05 and mean_frac >= 0.15:
            return "partial_late_rewrite_candidate"
    if "primary_plus_late_all" in kinds and success >= 0.35 and mean_rec >= 0.08 and mean_frac >= 0.20:
        if mean_rel >= 0.03:
            return "primary_late_joint_closure_candidate"
        return "primary_late_joint_target_candidate"
    if success >= 0.45 and mean_rec >= 0.10 and mean_frac >= 0.25:
        if mean_rel >= 0.03:
            return "primary_writer_guard_carrier_candidate"
        return "primary_target_carrier_candidate"
    if success >= 0.25 and mean_rec >= 0.05 and mean_frac >= 0.15:
        return "partial_primary_carrier_candidate"
    if mean_rec < -0.05:
        return "anti_restore_or_off_path"
    return "weak_or_unclear"


def summarize_combos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rest = [r for r in rows if r["row_kind"] == "multisite_restore"]
    groups: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rest:
        groups[(row["site"], int(row["head"]), row["candidate_kind"], row["source_group"], row["combo_name"])].append(row)
    out: list[dict[str, Any]] = []
    for (site, head, kind, source, combo_name), vals in groups.items():
        n = len(vals)
        combo_kind = vals[0]["combo_kind"]
        out.append(
            {
                "site": site,
                "head": head,
                "subunit_id": candidate_key(site, head),
                "candidate_kind": kind,
                "source_group": source,
                "combo_name": combo_name,
                "combo_kind": combo_kind,
                "combo_sites": vals[0]["combo_sites"],
                "combo_size": vals[0]["combo_size"],
                "n": n,
                "domains": sorted({v["domain"] for v in vals}),
                "relations": sorted({v["relation"] for v in vals}),
                "mean_combo_delta_norm_after_removal": safe_mean([v["mean_combo_delta_norm_after_removal"] for v in vals]),
                "mean_erase_target_logit_drop": safe_mean([v["erase_target_logit_drop"] for v in vals]),
                "mean_restored_target_logit_drop": safe_mean([v["restored_target_logit_drop"] for v in vals]),
                "mean_target_logit_recovered_by_restore": safe_mean([v["target_logit_recovered_by_restore"] for v in vals]),
                "mean_target_recovery_fraction": safe_mean([v["target_recovery_fraction"] for v in vals]),
                "mean_erase_route_release": safe_mean([v["erase_total_positive_route_release"] for v in vals]),
                "mean_restored_route_release": safe_mean([v["restored_total_positive_route_release"] for v in vals]),
                "mean_route_release_reduced_by_restore": safe_mean([v["route_release_reduced_by_restore"] for v in vals]),
                "effective_restore_rate": sum(1 for v in vals if v["effective_restore"]) / n,
                "combo_role_guess": classify_combo(vals),
            }
        )
    out.sort(
        key=lambda r: (
            r["effective_restore_rate"],
            r["mean_target_logit_recovered_by_restore"] or 0.0,
            r["mean_target_recovery_fraction"] or 0.0,
            r["mean_route_release_reduced_by_restore"] or 0.0,
        ),
        reverse=True,
    )
    return out


def combo_kind_baseline(combo_summary: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combo_summary:
        groups[row["combo_kind"]].append(row)
    return {
        kind: {
            "n_groups": len(vals),
            "mean_effective_restore_rate": safe_mean([v["effective_restore_rate"] for v in vals]),
            "mean_recovered": safe_mean([v["mean_target_logit_recovered_by_restore"] for v in vals]),
            "mean_recovery_fraction": safe_mean([v["mean_target_recovery_fraction"] for v in vals]),
            "mean_release_reduced": safe_mean([v["mean_route_release_reduced_by_restore"] for v in vals]),
            "role_counts": dict(Counter(v["combo_role_guess"] for v in vals)),
        }
        for kind, vals in sorted(groups.items())
    }


def build_summary(args, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], source_groups: list[str], all_combo_sites: list[str], attn_impl: str) -> dict[str, Any]:
    removal_summary = summarize_removals(rows)
    combo_summary = summarize_combos(rows)
    return {
        "phase": 758,
        "title": "Late Carrier Rewrite Relabel Test",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_source_removals": sum(1 for r in rows if r["row_kind"] == "source_removal_control"),
        "n_multisite_restores": sum(1 for r in rows if r["row_kind"] == "multisite_restore"),
        "candidates": candidates,
        "source_groups": source_groups,
        "all_combo_sites": all_combo_sites,
        "top_controlled_writer_candidates": removal_summary[:24],
        "combo_kind_baseline": combo_kind_baseline(combo_summary),
        "top_multisite_carrier_candidates": combo_summary[:64],
        "strict_interpretation": "Relabels Phase 757 off-path recovery as a late candidate. Strong only if late candidate beats primary path and true late controls.",
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
        all_combo_sites = unique_combo_sites(args.model, candidates, len(get_layers(model)), args.max_combos)
        unembed = get_unembed(model)
        log(f"{args.model}: expanded_candidates={len(candidates)} combo_sites={all_combo_sites}")
        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(pairs, 1):
            rows.extend(audit_pair(model, tokenizer, device, args, pair, candidates, source_groups, all_combo_sites, unembed))
            if idx % args.log_every == 0 or idx == len(pairs):
                log(f"{args.model}: multisite carrier {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, candidates, source_groups, all_combo_sites, attn_impl)
    write_jsonl = __import__("phase722_functional_head_atlas_causal_ablation").write_jsonl
    write_json = __import__("phase722_functional_head_atlas_causal_ablation").write_json
    write_jsonl(out_dir / f"phase758_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase758_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "combo_baseline": summary["combo_kind_baseline"],
                "top_writers": summary["top_controlled_writer_candidates"][:8],
                "top_multisite": summary["top_multisite_carrier_candidates"][:10],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    write_json = __import__("phase722_functional_head_atlas_causal_ablation").write_json
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase758_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 758,
        "title": "Late Carrier Rewrite Relabel Test",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "Late rewrite relabel test. Strong late evidence requires late_candidate groups to beat primary path and true_late_control groups.",
    }
    write_json(out_dir / "phase758_cross_model_summary.json", payload)
    lines = [
        f"# Phase 758 Late Carrier Rewrite Relabel Test ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence: source removal followed by primary, late-candidate, joint, and true-late-control component restores.",
        "",
        "## Combo Kind Baseline",
        "",
        "| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        for kind, row in summary.get("combo_kind_baseline", {}).items():
            lines.append(
                f"| {model_name} | `{kind}` | {row['n_groups']} | "
                f"{(row.get('mean_effective_restore_rate') or 0):.3f} | "
                f"{(row.get('mean_recovered') or 0):.3f} | "
                f"{(row.get('mean_recovery_fraction') or 0):.3f} | "
                f"{(row.get('mean_release_reduced') or 0):.3f} | `{row.get('role_counts')}` |"
            )
    lines.extend(
        [
            "",
            "## Top Multi-Site Restores",
            "",
            "| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_multisite_carrier_candidates", [])[:20]:
            lines.append(
                f"| {model_name} | `{row['candidate_kind']}` | {row['subunit_id']} | {row['source_group']} | `{row['combo_name']}` | "
                f"`{row['combo_sites']}` | {row['n']} | "
                f"{(row.get('effective_restore_rate') or 0):.3f} | "
                f"{(row.get('mean_erase_target_logit_drop') or 0):.3f} | "
                f"{(row.get('mean_target_logit_recovered_by_restore') or 0):.3f} | "
                f"{(row.get('mean_target_recovery_fraction') or 0):.3f} | "
                f"{(row.get('mean_route_release_reduced_by_restore') or 0):.3f} | `{row.get('combo_role_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Phase 758 relabels Phase 757 off-path recovery as a late carrier / rewrite candidate.",
            "- Strong evidence requires late_candidate groups to beat primary path and true_late_control groups.",
            "- If target recovery rises but route release is not reduced, the mechanism is target rewrite rather than route closure.",
            "",
        ]
    )
    (out_dir / "phase758_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"phase": 758, "round": args.round_name, "pairs": len(select_global_pairs(args.max_pairs)), "source_groups": source_groups_for(args), "models": {}}
    for model_name in MODELS:
        payload["models"][model_name] = {
            "primary_layers": PRIMARY_PATH_LAYERS.get(model_name, []),
            "late_candidate_layers": LATE_CANDIDATE_LAYERS.get(model_name, []),
            "true_late_control_layers": LATE_CONTROL_LAYERS.get(model_name, []),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--include-controls", action="store_true", default=True)
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-offset", type=int, default=13)
    parser.add_argument("--max-source-groups", type=int, default=2)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--max-combos", type=int, default=8)
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
