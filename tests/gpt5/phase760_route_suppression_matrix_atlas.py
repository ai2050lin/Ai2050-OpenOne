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
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, parse_component_site  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context, top_vocab_with_classes  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    capture_attention_value_state,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group, norm  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id, select_global_pairs  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import expanded_candidates, recovery_fraction, source_groups_for  # noqa: E402
from phase758_late_carrier_rewrite_relabel import combo_defs_for, install_multisite_restore, unique_combo_sites, run_logits  # noqa: E402


OUT_ROOT = Path("results/glm5_phase760_route_suppression_matrix_atlas")

FORMAT_TEXTS = [
    "Answer",
    "answer",
    ":",
    ".",
    ",",
    "\n",
    " is",
    " are",
    " category",
    " color",
    " shape",
    " taste",
    " edible",
]

GENERIC_TEXTS = [
    "unknown",
    "none",
    "object",
    "thing",
    "entity",
    "item",
    "yes",
    "no",
    "true",
    "false",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def first_token_or_none(tokenizer, text: str) -> int | None:
    try:
        ids = target_token_ids(tokenizer, text)
        if not ids:
            return None
        return int(ids[0])
    except Exception:
        return None


def add_tokens(group_map: dict[str, dict[int, str]], group: str, tokens: list[tuple[int | None, str]]) -> None:
    bucket = group_map.setdefault(group, {})
    for tid, label in tokens:
        if tid is None:
            continue
        bucket[int(tid)] = label


def build_explicit_route_groups(
    tokenizer,
    logits: torch.Tensor,
    target: dict[str, Any],
    contrast: dict[str, Any],
    target_id: int,
    contrast_id: int,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    ctx = taxonomy_context(tokenizer, target, contrast)
    group_map: dict[str, dict[int, str]] = {}
    add_tokens(group_map, "contrast_answer", [(contrast_id, str(contrast.get("answer")))])
    add_tokens(
        group_map,
        "object_relation_echo",
        [
            (first_token_or_none(tokenizer, str(target.get("object", ""))), f"object:{target.get('object')}"),
            (first_token_or_none(tokenizer, str(target.get("relation", ""))), f"relation:{target.get('relation')}"),
        ],
    )
    value_tokens = []
    for text in sorted(set(str(x) for x in ctx.get("all_values", []) if x)):
        tid = first_token_or_none(tokenizer, text)
        if tid is None or tid in {int(target_id), int(contrast_id)}:
            continue
        value_tokens.append((tid, f"value:{text}"))
    add_tokens(group_map, "other_record_value", value_tokens)
    add_tokens(group_map, "format_schema", [(first_token_or_none(tokenizer, x), x) for x in FORMAT_TEXTS])
    add_tokens(group_map, "generic_answer", [(first_token_or_none(tokenizer, x), x) for x in GENERIC_TEXTS])

    top_vocab = top_vocab_with_classes(logits, tokenizer, ctx, args.top_k_vocab)
    top_added = 0
    by_class: dict[str, list[tuple[int | None, str]]] = defaultdict(list)
    for item in top_vocab:
        tid = int(item["token_id"])
        if tid == int(target_id):
            continue
        label = f"{item.get('class')}:{item.get('token_text')}"
        add_tokens(group_map, "top_non_target", [(tid, label)])
        by_class[f"top_class:{item.get('class')}"].append((tid, label))
        top_added += 1
        if top_added >= args.max_topk_tokens:
            break
    for group, toks in list(by_class.items())[: args.max_dynamic_route_classes]:
        add_tokens(group_map, group, toks[: args.max_topk_tokens])

    out: dict[str, dict[str, Any]] = {}
    for group, members in sorted(group_map.items()):
        filtered = {tid: label for tid, label in members.items() if 0 <= tid < logits.numel() and tid != int(target_id)}
        if not filtered:
            continue
        out[group] = {
            "route_group": group,
            "tokens": [
                {"token_id": tid, "token_text": decode_token(tokenizer, tid), "label": label}
                for tid, label in sorted(filtered.items())
            ],
        }
    return out


def group_max(logits: torch.Tensor, group: dict[str, Any]) -> dict[str, Any] | None:
    vals = []
    for tok in group["tokens"]:
        tid = int(tok["token_id"])
        if 0 <= tid < logits.numel():
            vals.append((float(logits[tid].item()), tid, tok["token_text"], tok["label"]))
    if not vals:
        return None
    max_logit, token_id, text, label = max(vals, key=lambda x: x[0])
    return {
        "max_logit": max_logit,
        "max_token_id": token_id,
        "max_token_text": text,
        "max_token_label": label,
        "token_count": len(vals),
    }


def route_matrix(
    base_logits: torch.Tensor,
    after_logits: torch.Tensor,
    groups: dict[str, dict[str, Any]],
    target_id: int,
) -> dict[str, dict[str, Any]]:
    out = {}
    base_target = float(base_logits[int(target_id)].item())
    after_target = float(after_logits[int(target_id)].item())
    for group_name, group in groups.items():
        before = group_max(base_logits, group)
        after = group_max(after_logits, group)
        if before is None or after is None:
            continue
        release = float(after["max_logit"] - before["max_logit"])
        before_margin = base_target - float(before["max_logit"])
        after_margin = after_target - float(after["max_logit"])
        out[group_name] = {
            "route_group": group_name,
            "base_route_max_logit": before["max_logit"],
            "base_route_max_token_id": before["max_token_id"],
            "base_route_max_token_text": before["max_token_text"],
            "base_route_max_token_label": before["max_token_label"],
            "after_route_max_logit": after["max_logit"],
            "after_route_max_token_id": after["max_token_id"],
            "after_route_max_token_text": after["max_token_text"],
            "after_route_max_token_label": after["max_token_label"],
            "route_release": release,
            "margin_drop_target_vs_route": before_margin - after_margin,
            "token_count": before["token_count"],
        }
    return out


def total_positive_release(matrix: dict[str, dict[str, Any]]) -> float:
    return float(sum(max(0.0, float(v["route_release"])) for v in matrix.values()))


def route_ids_from_groups(groups: dict[str, dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for group in groups.values():
        for tok in group["tokens"]:
            tid = int(tok["token_id"])
            if tid not in ids:
                ids.append(tid)
    return ids


def target_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    erase = float(row.get("erase_target_logit_drop") or 0.0)
    recovered = float(row.get("target_logit_recovered_by_restore") or 0.0)
    frac = float(row.get("target_recovery_fraction") or 0.0)
    return erase >= args.min_erase_drop and recovered >= args.min_target_recovery and frac >= args.min_target_fraction


def route_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    erase_release = float(row.get("erase_route_release") or 0.0)
    reduced = float(row.get("route_release_reduced_by_restore") or 0.0)
    return erase_release >= args.min_route_release and reduced >= args.min_route_reduced


def classify_cell(vals: list[dict[str, Any]], args: argparse.Namespace) -> str:
    if not vals:
        return "empty"
    n = len(vals)
    target_rate = sum(target_success(v, args) for v in vals) / n
    route_rate = sum(route_success(v, args) for v in vals) / n
    route_only_rate = sum(route_success(v, args) and not target_success(v, args) for v in vals) / n
    mean_reduced = safe_mean([v.get("route_release_reduced_by_restore") for v in vals]) or 0.0
    mean_recovered = safe_mean([v.get("target_logit_recovered_by_restore") for v in vals]) or 0.0
    if route_only_rate >= args.route_only_rate and mean_reduced >= args.min_route_reduced:
        return "route_only_suppressor_candidate"
    if route_rate >= args.route_rate and target_rate >= args.target_rate:
        return "joint_rewrite_suppressor_candidate"
    if target_rate >= args.target_rate and mean_recovered >= args.min_target_recovery:
        return "target_rewrite_candidate"
    if route_rate >= args.route_rate and mean_reduced >= args.min_route_reduced:
        return "route_suppression_candidate"
    if mean_reduced < -args.min_route_reduced:
        return "route_release_amplifier_or_nonclosure"
    return "weak_or_unclear"


def audit_pair(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
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
    target_id = get_first_token_id(tokenizer, target["answer"])
    contrast_id = get_first_token_id(tokenizer, contrast["answer"])
    route_groups = build_explicit_route_groups(tokenizer, state["logits"], target, contrast, target_id, contrast_id, args)
    if not route_groups:
        return []
    route_ids = route_ids_from_groups(route_groups)
    base_components = capture_state(model, device, state["ids"], all_combo_sites)
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "base_route_matrix",
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
            "contrast_rank": contrast_diag["target_rank"],
            "route_groups": route_groups,
            "route_group_names": sorted(route_groups),
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
            erased_logits = run_logits(model, device, state["ids"], removal_install)
            erased_components = capture_state(model, device, state["ids"], all_combo_sites, removal_install)
            erase_matrix = route_matrix(state["logits"], erased_logits, route_groups, target_id)
            erase_target_drop = float(state["logits"][target_id].item() - erased_logits[target_id].item())
            rows.append(
                {
                    "row_kind": "source_removal_overview",
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
                    "erase_target_logit_drop": erase_target_drop,
                    "erase_total_positive_route_release": total_positive_release(erase_matrix),
                    "erase_route_matrix": erase_matrix,
                }
            )
            for combo in combo_defs:
                sites = [s for s in combo["sites"] if s in base_components["components"] and s in erased_components["components"]]
                if not sites:
                    continue
                install = install_multisite_restore(model, removal_install, sites, base_components["components"])
                restored_logits = run_logits(model, device, state["ids"], install)
                restored_matrix = route_matrix(state["logits"], restored_logits, route_groups, target_id)
                restored_target_drop = float(state["logits"][target_id].item() - restored_logits[target_id].item())
                target_recovered = erase_target_drop - restored_target_drop
                frac = recovery_fraction(erase_target_drop, target_recovered)
                for group_name, erase_cell in erase_matrix.items():
                    restored_cell = restored_matrix.get(group_name)
                    if not restored_cell:
                        continue
                    route_reduced = float(erase_cell["route_release"] - restored_cell["route_release"])
                    rows.append(
                        {
                            "row_kind": "route_restore_matrix",
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
                            "route_group": group_name,
                            "erase_target_logit_drop": erase_target_drop,
                            "restored_target_logit_drop": restored_target_drop,
                            "target_logit_recovered_by_restore": target_recovered,
                            "target_recovery_fraction": frac,
                            "erase_route_release": erase_cell["route_release"],
                            "restored_route_release": restored_cell["route_release"],
                            "route_release_reduced_by_restore": route_reduced,
                            "erase_margin_drop_target_vs_route": erase_cell["margin_drop_target_vs_route"],
                            "restored_margin_drop_target_vs_route": restored_cell["margin_drop_target_vs_route"],
                            "margin_drop_reduced_by_restore": float(
                                erase_cell["margin_drop_target_vs_route"] - restored_cell["margin_drop_target_vs_route"]
                            ),
                            "erase_route_max_token_text": erase_cell["after_route_max_token_text"],
                            "restored_route_max_token_text": restored_cell["after_route_max_token_text"],
                            "route_token_count": erase_cell["token_count"],
                            "mean_combo_delta_norm_after_removal": safe_mean(
                                [norm(erased_components["components"][s] - base_components["components"][s]) for s in sites]
                            ),
                        }
                    )
    return rows


def summarize_cells(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    mat = [r for r in rows if r["row_kind"] == "route_restore_matrix"]
    groups: dict[tuple[str, int, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mat:
        groups[(row["site"], int(row["head"]), row["source_group"], row["combo_kind"], row["combo_name"], row["route_group"])].append(row)
    out: list[dict[str, Any]] = []
    for (site, head, source, combo_kind, combo_name, route_group), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "site": site,
                "head": head,
                "subunit_id": f"{site}:H{head}",
                "source_group": source,
                "combo_kind": combo_kind,
                "combo_name": combo_name,
                "combo_sites": vals[0]["combo_sites"],
                "route_group": route_group,
                "n": n,
                "domains": sorted({v["domain"] for v in vals}),
                "relations": sorted({v["relation"] for v in vals}),
                "target_success_rate": sum(target_success(v, args) for v in vals) / n,
                "route_success_rate": sum(route_success(v, args) for v in vals) / n,
                "route_only_success_rate": sum(route_success(v, args) and not target_success(v, args) for v in vals) / n,
                "joint_success_rate": sum(route_success(v, args) and target_success(v, args) for v in vals) / n,
                "mean_erase_target_logit_drop": safe_mean([v["erase_target_logit_drop"] for v in vals]),
                "mean_target_recovered": safe_mean([v["target_logit_recovered_by_restore"] for v in vals]),
                "mean_target_recovery_fraction": safe_mean([v["target_recovery_fraction"] for v in vals]),
                "mean_erase_route_release": safe_mean([v["erase_route_release"] for v in vals]),
                "mean_restored_route_release": safe_mean([v["restored_route_release"] for v in vals]),
                "mean_route_reduced": safe_mean([v["route_release_reduced_by_restore"] for v in vals]),
                "mean_margin_drop_reduced": safe_mean([v["margin_drop_reduced_by_restore"] for v in vals]),
                "role_guess": classify_cell(vals, args),
                "erase_route_token_counts": dict(Counter(v["erase_route_max_token_text"] for v in vals).most_common(8)),
                "restored_route_token_counts": dict(Counter(v["restored_route_max_token_text"] for v in vals).most_common(8)),
            }
        )
    out.sort(
        key=lambda r: (
            r["route_only_success_rate"],
            r["route_success_rate"],
            r["mean_route_reduced"] or 0.0,
            -r["target_success_rate"],
        ),
        reverse=True,
    )
    return out


def summarize_by_combo_kind(cell_summary: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_summary:
        groups[(row["combo_kind"], row["route_group"])].append(row)
    out = {}
    for (combo_kind, route_group), vals in sorted(groups.items()):
        key = f"{combo_kind}::{route_group}"
        out[key] = {
            "combo_kind": combo_kind,
            "route_group": route_group,
            "n_groups": len(vals),
            "mean_route_only_success_rate": safe_mean([v["route_only_success_rate"] for v in vals]),
            "mean_route_success_rate": safe_mean([v["route_success_rate"] for v in vals]),
            "mean_target_success_rate": safe_mean([v["target_success_rate"] for v in vals]),
            "mean_route_reduced": safe_mean([v["mean_route_reduced"] for v in vals]),
            "mean_target_recovered": safe_mean([v["mean_target_recovered"] for v in vals]),
            "role_counts": dict(Counter(v["role_guess"] for v in vals)),
        }
    return out


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    all_combo_sites: list[str],
    attn_impl: str,
) -> dict[str, Any]:
    cell_summary = summarize_cells(rows, args)
    return {
        "phase": 760,
        "title": "Route Suppression Matrix Atlas",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_base_route_matrices": sum(1 for r in rows if r["row_kind"] == "base_route_matrix"),
        "n_source_removal_overviews": sum(1 for r in rows if r["row_kind"] == "source_removal_overview"),
        "n_route_restore_matrix_rows": sum(1 for r in rows if r["row_kind"] == "route_restore_matrix"),
        "candidates": candidates,
        "source_groups": source_groups,
        "all_combo_sites": all_combo_sites,
        "combo_kind_route_baseline": summarize_by_combo_kind(cell_summary),
        "top_route_suppression_cells": cell_summary[:96],
        "strict_interpretation": "Explicit route matrix after source removal and multisite restore. Route-only cells are candidates, not confirmed global suppressors.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_global_pairs(args.max_pairs)
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        candidates = expanded_candidates(model, args.model, args)
        all_combo_sites = unique_combo_sites(args.model, candidates, len(get_layers(model)), args.max_combos)
        unembed = get_unembed(model)
        log(f"{args.model}: candidates={len(candidates)} combo_sites={all_combo_sites}")
        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(pairs, 1):
            rows.extend(audit_pair(model, tokenizer, device, args, pair, candidates, source_groups, all_combo_sites, unembed))
            if idx % args.log_every == 0 or idx == len(pairs):
                log(f"{args.model}: route matrix {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, candidates, source_groups, all_combo_sites, attn_impl)
    write_jsonl(out_dir / f"phase760_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase760_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "n_rows": summary["n_rows"],
                "top_route_cells": summary["top_route_suppression_cells"][:10],
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
    for model_name in MODELS:
        path = out_dir / f"phase760_{model_name}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 760,
        "title": "Route Suppression Matrix Atlas",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "A route matrix split of target recovery vs explicit route suppression. It locates candidate route cells, not neuron-level invariants.",
    }
    write_json(out_dir / "phase760_cross_model_summary.json", payload)
    lines = [
        f"# Phase 760 Route Suppression Matrix Atlas ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Route groups: contrast_answer / object_relation_echo / other_record_value / format_schema / generic_answer / top_non_target / dynamic top classes.",
        "",
        "## Combo Kind x Route Group",
        "",
        "| model | combo kind | route group | groups | route-only | route rate | target rate | route reduced | target recovered | roles |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        rows = list(summary.get("combo_kind_route_baseline", {}).values())
        rows.sort(
            key=lambda r: (
                r.get("mean_route_only_success_rate") or 0.0,
                r.get("mean_route_success_rate") or 0.0,
                r.get("mean_route_reduced") or 0.0,
            ),
            reverse=True,
        )
        for row in rows[:32]:
            lines.append(
                f"| {model_name} | `{row['combo_kind']}` | `{row['route_group']}` | {row['n_groups']} | "
                f"{(row.get('mean_route_only_success_rate') or 0):.3f} | "
                f"{(row.get('mean_route_success_rate') or 0):.3f} | "
                f"{(row.get('mean_target_success_rate') or 0):.3f} | "
                f"{(row.get('mean_route_reduced') or 0):.3f} | "
                f"{(row.get('mean_target_recovered') or 0):.3f} | `{row.get('role_counts')}` |"
            )
    lines.extend(
        [
            "",
            "## Top Route Suppression Cells",
            "",
            "| model | writer | source | combo | route | n | route-only | route rate | target rate | route reduced | recovered | role |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_route_suppression_cells", [])[:24]:
            lines.append(
                f"| {model_name} | {row['subunit_id']} | {row['source_group']} | `{row['combo_name']}` | `{row['route_group']}` | "
                f"{row['n']} | {row['route_only_success_rate']:.3f} | {row['route_success_rate']:.3f} | {row['target_success_rate']:.3f} | "
                f"{(row.get('mean_route_reduced') or 0):.3f} | {(row.get('mean_target_recovered') or 0):.3f} | `{row['role_guess']}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- A route-only cell means restore reduces a route group after source removal while target recovery does not satisfy the target threshold.",
            "- Negative route reduced means the restored component amplifies or fails to close that route release.",
            "- This phase separates route classes explicitly, but still works at component/head level and is not a neuron atlas.",
            "",
        ]
    )
    (out_dir / "phase760_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {
        "phase": 760,
        "round": args.round_name,
        "pairs": len(select_global_pairs(args.max_pairs)),
        "max_candidates": args.max_candidates,
        "source_groups": source_groups_for(args),
        "route_groups": [
            "contrast_answer",
            "object_relation_echo",
            "other_record_value",
            "format_schema",
            "generic_answer",
            "top_non_target",
            "top_class:*",
        ],
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
    parser.add_argument("--max-combos", type=int, default=10)
    parser.add_argument("--top-k-vocab", type=int, default=18)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-dynamic-route-classes", type=int, default=5)
    parser.add_argument("--min-erase-drop", type=float, default=0.20)
    parser.add_argument("--min-target-recovery", type=float, default=0.10)
    parser.add_argument("--min-target-fraction", type=float, default=0.25)
    parser.add_argument("--min-route-release", type=float, default=0.10)
    parser.add_argument("--min-route-reduced", type=float, default=0.05)
    parser.add_argument("--route-only-rate", type=float, default=0.30)
    parser.add_argument("--route-rate", type=float, default=0.30)
    parser.add_argument("--target-rate", type=float, default=0.30)
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
