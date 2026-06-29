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

from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for, records_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean, select_evenly  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context, top_vocab_with_classes  # noqa: E402
from phase748_natural_route_suppressor_matrix import group_competitors_by_route, route_max_logits, js_divergence  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score, route_token_ids  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    build_route_context,
    capture_attention_value_state,
    eval_after_logits,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group, norm  # noqa: E402
from phase741_threshold_candidate_causal_validation import parse_component_site  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from model_utils import get_layers  # noqa: E402


OUT_ROOT = Path("results/glm5_phase755_cross_domain_route_invariance_atlas")

RELATION_KEYS = [
    ("category", "category"),
    ("color", "color"),
    ("taste", "taste"),
    ("shape", "shape"),
    ("edible", "edible"),
    ("grows_on_tree", "tree"),
]

DOMAIN_OBJECTS = [
    {"object": "apple", "domain": "fruit", "category": "fruit", "color": "red", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    {"object": "banana", "domain": "fruit", "category": "fruit", "color": "yellow", "taste": "sweet", "shape": "long", "edible": "yes", "tree": "no"},
    {"object": "cat", "domain": "animal", "category": "animal", "color": "black", "taste": "none", "shape": "small", "edible": "no", "tree": "no"},
    {"object": "bird", "domain": "animal", "category": "animal", "color": "blue", "taste": "none", "shape": "small", "edible": "no", "tree": "no"},
    {"object": "oak", "domain": "plant", "category": "plant", "color": "green", "taste": "none", "shape": "tall", "edible": "no", "tree": "yes"},
    {"object": "rose", "domain": "plant", "category": "plant", "color": "red", "taste": "none", "shape": "round", "edible": "no", "tree": "no"},
    {"object": "chair", "domain": "object", "category": "furniture", "color": "brown", "taste": "none", "shape": "rectangular", "edible": "no", "tree": "no"},
    {"object": "stone", "domain": "object", "category": "object", "color": "gray", "taste": "none", "shape": "irregular", "edible": "no", "tree": "no"},
    {"object": "hammer", "domain": "tool", "category": "tool", "color": "silver", "taste": "none", "shape": "long", "edible": "no", "tree": "no"},
    {"object": "knife", "domain": "tool", "category": "tool", "color": "silver", "taste": "none", "shape": "long", "edible": "no", "tree": "no"},
    {"object": "freedom", "domain": "abstract", "category": "abstract", "color": "none", "taste": "none", "shape": "none", "edible": "no", "tree": "no"},
    {"object": "time", "domain": "abstract", "category": "abstract", "color": "none", "taste": "none", "shape": "none", "edible": "no", "tree": "no"},
]

CONFLICT_BY_DOMAIN = {
    "fruit": {"category": "tool", "color": "blue", "taste": "bitter", "shape": "square", "edible": "no", "tree": "no"},
    "animal": {"category": "fruit", "color": "yellow", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    "plant": {"category": "animal", "color": "black", "taste": "bitter", "shape": "small", "edible": "yes", "tree": "no"},
    "object": {"category": "plant", "color": "green", "taste": "sweet", "shape": "tall", "edible": "yes", "tree": "yes"},
    "tool": {"category": "abstract", "color": "none", "taste": "none", "shape": "none", "edible": "no", "tree": "no"},
    "abstract": {"category": "object", "color": "gray", "taste": "none", "shape": "irregular", "edible": "no", "tree": "no"},
}

FIXED_HEADS = {
    "qwen3": [
        {"site": "L33:attn_out", "head": 15, "source": "phase751_752"},
        {"site": "L33:attn_out", "head": 23, "source": "phase751_752"},
        {"site": "L32:attn_out", "head": 11, "source": "phase752"},
    ],
    "glm4": [
        {"site": "L35:attn_out", "head": 29, "source": "phase751_752"},
        {"site": "L34:attn_out", "head": 4, "source": "phase751_752"},
        {"site": "L34:attn_out", "head": 9, "source": "phase752"},
    ],
    "deepseek7b": [
        {"site": "L22:attn_out", "head": 24, "source": "phase751_752"},
        {"site": "L22:attn_out", "head": 1, "source": "phase750_752"},
        {"site": "L22:attn_out", "head": 7, "source": "phase750_752"},
        {"site": "L23:attn_out", "head": 6, "source": "phase751_752"},
    ],
}

DEFAULT_SOURCE_GROUPS = ["target_record_line", "target_value_tokens", "records_all", "relation_tokens", "object_tokens"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_global_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cid = 0
    for obj in DOMAIN_OBJECTS:
        explicit = {
            "object": obj["object"],
            "object_group": obj["domain"],
            "domain": obj["domain"],
            "category": obj["category"],
            "color": obj["color"],
            "taste": obj["taste"],
            "shape": obj["shape"],
            "edible": obj["edible"],
            "tree": obj["tree"],
        }
        conflict_vals = {**obj, **CONFLICT_BY_DOMAIN[obj["domain"]]}
        conflict = {
            "object": obj["object"],
            "object_group": obj["domain"],
            "domain": obj["domain"],
            "category": conflict_vals["category"],
            "color": conflict_vals["color"],
            "taste": conflict_vals["taste"],
            "shape": conflict_vals["shape"],
            "edible": conflict_vals["edible"],
            "tree": conflict_vals["tree"],
        }
        for relation, key in RELATION_KEYS:
            if explicit[key] == conflict[key]:
                continue
            cid += 1
            cases.append(
                {
                    "pair_id": f"p755_{cid:04d}_{obj['domain']}:{obj['object']}:{relation}",
                    "explicit_profile": {
                        "case_id": f"p755_explicit_{cid:04d}",
                        "prompt_type": "explicit_profile",
                        "object": obj["object"],
                        "object_group": obj["domain"],
                        "domain": obj["domain"],
                        "relation": relation,
                        "answer": explicit[key],
                        "records": records_for(explicit),
                    },
                    "conflict_profile": {
                        "case_id": f"p755_conflict_{cid:04d}",
                        "prompt_type": "conflict_profile",
                        "object": obj["object"],
                        "object_group": obj["domain"],
                        "domain": obj["domain"],
                        "relation": relation,
                        "answer": conflict[key],
                        "records": records_for(conflict),
                    },
                }
            )
    return cases


def select_global_pairs(max_pairs: int | None) -> list[dict[str, Any]]:
    pairs = build_global_cases()
    if not max_pairs or max_pairs >= len(pairs):
        return pairs
    return [pairs[i] for i in select_evenly(len(pairs), max_pairs)]


def candidate_heads_for(args) -> list[dict[str, Any]]:
    return FIXED_HEADS[args.model][: args.max_candidates]


def source_groups_for(args) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def route_class_profile(route_max: dict[str, dict[str, Any]]) -> dict[str, float]:
    if not route_max:
        return {}
    keys = sorted(route_max)
    vals = torch.tensor([float(route_max[k]["max_logit"]) for k in keys], dtype=torch.float32)
    probs = torch.softmax(vals, dim=0)
    return {k: float(probs[i].item()) for i, k in enumerate(keys)}


def mean_profile(profiles: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for p in profiles for k in p})
    if not keys:
        return {}
    return {k: sum(float(p.get(k, 0.0)) for p in profiles) / len(profiles) for k in keys}


def get_first_token_id(tokenizer, text: str) -> int:
    from phase740_natural_readout_boost_source_backtrace import first_token_id

    return int(first_token_id(tokenizer, text))


def run_logits(model, device, ids: list[int], install=None) -> torch.Tensor:
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
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    target = pair["explicit_profile"]
    contrast = pair["conflict_profile"]
    layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_attention_value_state(model, tokenizer, device, target, layers)
    ctx = taxonomy_context(tokenizer, target, contrast)
    target_id = get_first_token_id(tokenizer, target["answer"])
    contrast_id = get_first_token_id(tokenizer, contrast["answer"])
    route_ctx = build_route_context(state["logits"], tokenizer, ctx, target_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return []
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    route_profile = route_class_profile(route_ctx["route_max"])
    top_vocab = top_vocab_with_classes(state["logits"], tokenizer, ctx, args.top_k_vocab)
    top = top_vocab[0] if top_vocab else {}
    rows: list[dict[str, Any]] = []
    base_row = {
        "row_kind": "route_observation",
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
        "target_logit": target_diag["target_logit"],
        "target_logprob": target_diag["target_logprob"],
        "contrast_rank": contrast_diag["target_rank"],
        "top_token_id": int(top.get("token_id", -1)),
        "top_token_text": top.get("token_text", ""),
        "top_token_class": top.get("class", ""),
        "route_profile": route_profile,
        "route_classes": sorted(route_profile),
        "route_max": route_ctx["route_max"],
    }
    rows.append(base_row)

    route_ids = route_token_ids(route_ctx["route_max"])
    answer_pos = state["answer_pos"]
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if head < 0 or head >= n_heads:
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
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
            install = install_source_contribution_removal(model, site, [head], contribution)
            after_logits = run_logits(model, device, state["ids"], install)
            metrics = eval_after_logits(state["logits"], after_logits, route_ctx, target_id, contrast_id)
            rows.append(
                {
                    "row_kind": "source_removal",
                    "pair_id": pair["pair_id"],
                    "domain": target["domain"],
                    "object": target["object"],
                    "relation": target["relation"],
                    "target_answer": target["answer"],
                    "contrast_answer": contrast["answer"],
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "subunit_id": f"{site}:H{head}",
                    "selection": cand["source"],
                    "source_group": source_group,
                    "source_positions_n": len(src_positions),
                    "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                    "source_projected_delta_norm": norm(projected),
                    "source_direct_score": direct,
                    **metrics,
                }
            )
    return rows


def summarize_route_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    obs = [r for r in rows if r["row_kind"] == "route_observation"]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obs:
        by_domain[row["domain"]].append(row)
        by_relation[row["relation"]].append(row)
    domain_profiles = {d: mean_profile([r["route_profile"] for r in vals]) for d, vals in sorted(by_domain.items())}
    pairwise_js = []
    domains = sorted(domain_profiles)
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            pairwise_js.append({"domain_a": a, "domain_b": b, "js": js_divergence(domain_profiles[a], domain_profiles[b])})
    pairwise_js.sort(key=lambda x: x["js"])

    def pack(vals: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(vals)
        return {
            "n": n,
            "target_top1_rate": sum(1 for v in vals if v["target_top1"]) / n if n else None,
            "mean_target_rank": safe_mean([v["target_rank"] for v in vals]),
            "top_token_class_counts": dict(Counter(v["top_token_class"] for v in vals)),
            "route_class_counts": dict(Counter(c for v in vals for c in v["route_classes"])),
            "mean_route_profile": mean_profile([v["route_profile"] for v in vals]),
        }

    return {
        "n_route_observations": len(obs),
        "by_domain": {k: pack(v) for k, v in sorted(by_domain.items())},
        "by_relation": {k: pack(v) for k, v in sorted(by_relation.items())},
        "domain_pairwise_route_js": pairwise_js,
        "mean_pairwise_route_js": safe_mean([x["js"] for x in pairwise_js]),
        "lowest_js_pairs": pairwise_js[:8],
        "highest_js_pairs": sorted(pairwise_js, key=lambda x: x["js"], reverse=True)[:8],
    }


def classify_cross_domain(vals: list[dict[str, Any]]) -> str:
    n = len(vals)
    if not n:
        return "empty"
    domains = {v["domain"] for v in vals}
    support_rate = sum(1 for v in vals if v["target_logit_drop"] > 0.20) / n
    guard_rate = sum(1 for v in vals if v["total_positive_route_release"] > 0.20) / n
    mean_drop = safe_mean([v["target_logit_drop"] for v in vals]) or 0.0
    mean_release = safe_mean([v["total_positive_route_release"] for v in vals]) or 0.0
    domain_support = {}
    for d in domains:
        xs = [v for v in vals if v["domain"] == d]
        domain_support[d] = sum(1 for v in xs if v["target_logit_drop"] > 0.20) / len(xs)
    active_domains = sum(1 for v in domain_support.values() if v >= 0.50)
    if active_domains >= 4 and support_rate >= 0.50 and mean_drop >= 0.30:
        if guard_rate >= 0.35 or mean_release >= 0.25:
            return "cross_domain_mixed_writer_guard"
        return "cross_domain_writer_candidate"
    if active_domains >= 3 and guard_rate >= 0.35 and mean_release >= 0.20:
        return "cross_domain_route_guard_candidate"
    if active_domains >= 2 and mean_drop >= 0.25:
        return "multi_domain_but_not_global"
    return "domain_specific_or_weak"


def summarize_removal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rem = [r for r in rows if r["row_kind"] == "source_removal"]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rem:
        groups[(row["site"], int(row["head"]), row["source_group"])].append(row)
    out = []
    for (site, head, source), vals in groups.items():
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for v in vals:
            by_domain[v["domain"]].append(v)
        domain_breakdown = {}
        for d, xs in sorted(by_domain.items()):
            domain_breakdown[d] = {
                "n": len(xs),
                "support_rate": sum(1 for v in xs if v["target_logit_drop"] > 0.20) / len(xs),
                "route_guard_rate": sum(1 for v in xs if v["total_positive_route_release"] > 0.20) / len(xs),
                "mean_target_drop": safe_mean([v["target_logit_drop"] for v in xs]),
                "mean_route_release": safe_mean([v["total_positive_route_release"] for v in xs]),
            }
        n = len(vals)
        out.append(
            {
                "site": site,
                "head": head,
                "subunit_id": f"{site}:H{head}",
                "source_group": source,
                "n": n,
                "domains": sorted(by_domain),
                "relations": sorted({v["relation"] for v in vals}),
                "mean_attention_mass_to_source": safe_mean([v["attention_mass_to_source"] for v in vals]),
                "mean_source_target_logit_contribution": safe_mean([v["source_direct_score"]["direct_target_boost"] for v in vals]),
                "mean_source_total_route_suppression_contribution": safe_mean([v["source_direct_score"]["direct_total_route_suppression"] for v in vals]),
                "mean_target_logit_drop_after_source_removal": safe_mean([v["target_logit_drop"] for v in vals]),
                "support_rate_drop_gt_0_20": sum(1 for v in vals if v["target_logit_drop"] > 0.20) / n,
                "mean_total_positive_route_release_after_source_removal": safe_mean([v["total_positive_route_release"] for v in vals]),
                "route_guard_rate_release_gt_0_20": sum(1 for v in vals if v["total_positive_route_release"] > 0.20) / n,
                "top1_loss_rate": sum(1 for v in vals if v["top1_loss"]) / n,
                "domain_breakdown": domain_breakdown,
                "cross_domain_guess": classify_cross_domain(vals),
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


def build_summary(args, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], source_groups: list[str], attn_impl: str) -> dict[str, Any]:
    return {
        "phase": 755,
        "title": "Cross-Domain Route Invariance Atlas",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_route_observations": sum(1 for r in rows if r["row_kind"] == "route_observation"),
        "n_source_removals": sum(1 for r in rows if r["row_kind"] == "source_removal"),
        "candidates": candidates,
        "source_groups": source_groups,
        "route_invariance": summarize_route_rows(rows),
        "top_cross_domain_subunits": summarize_removal_rows(rows)[:32],
        "strict_interpretation": "This is a first cross-domain atlas. Route profile similarity is observational; source removal is path-level causal evidence, not neuron-level proof.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_global_pairs(args.max_pairs)
    candidates = candidate_heads_for(args)
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} candidates={len(candidates)} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    unembed = get_unembed(model)
    try:
        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(pairs, 1):
            rows.extend(audit_pair(model, tokenizer, device, args, pair, candidates, source_groups, unembed))
            if idx % args.log_every == 0 or idx == len(pairs):
                log(f"{args.model}: cross-domain atlas {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, candidates, source_groups, attn_impl)
    write_jsonl(out_dir / f"phase755_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase755_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "route_js": summary["route_invariance"]["mean_pairwise_route_js"],
                "top": summary["top_cross_domain_subunits"][:8],
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
        p = out_dir / f"phase755_{model}_summary.json"
        if p.exists():
            summaries.append(json.loads(p.read_text(encoding="utf-8")))
    payload = {
        "phase": 755,
        "title": "Cross-Domain Route Invariance Atlas",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "First cross-domain route atlas. It can show route-profile similarity and path-level source removal effects, but not a complete language graph.",
    }
    write_json(out_dir / "phase755_cross_model_summary.json", payload)
    lines = [
        f"# Phase 755 Cross-Domain Route Invariance Atlas ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Scope: fruit / animal / plant / object / tool / abstract.",
        "- Evidence: natural route class profile + fixed head/source contribution removal.",
        "",
        "## Route Profile",
        "",
        "| model | route observations | mean pairwise domain JS | strongest shared top classes |",
        "|---|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        route = summary["route_invariance"]
        class_counter = Counter()
        for drow in route["by_domain"].values():
            class_counter.update(drow["top_token_class_counts"])
        lines.append(
            f"| {model_name} | {route['n_route_observations']} | {(route.get('mean_pairwise_route_js') or 0):.4f} | "
            f"`{dict(class_counter.most_common(5))}` |"
        )
    lines.extend(
        [
            "",
            "## Top Cross-Domain Writer / Guard Candidates",
            "",
            "| model | site | head | source | n | domains | support rate | mean drop | guard rate | mean release | guess |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_cross_domain_subunits", [])[:12]:
            lines.append(
                f"| {model_name} | {row['site']} | {row['head']} | {row['source_group']} | {row['n']} | {len(row.get('domains') or [])} | "
                f"{(row.get('support_rate_drop_gt_0_20') or 0):.3f} | "
                f"{(row.get('mean_target_logit_drop_after_source_removal') or 0):.3f} | "
                f"{(row.get('route_guard_rate_release_gt_0_20') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_release_after_source_removal') or 0):.3f} | "
                f"`{row.get('cross_domain_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Low JS across domains supports route-profile similarity, not a discovered invariant by itself.",
            "- Fixed source removal effects support path-level necessity, not a full neuron graph.",
            "- If a candidate is strong only in DS7B, it is a model-local atlas result until replicated.",
            "",
        ]
    )
    (out_dir / "phase755_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_global_pairs(args.max_pairs)), "models": {}}
    for model_name in MODELS:
        args.model = model_name
        payload["models"][model_name] = {"candidates": candidate_heads_for(args), "source_groups": source_groups_for(args)}
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-source-groups", type=int, default=3)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
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
