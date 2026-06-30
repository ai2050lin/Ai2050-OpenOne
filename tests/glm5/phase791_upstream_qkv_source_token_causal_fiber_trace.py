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

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads, get_v_proj  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import (  # noqa: E402
    MODELS,
    bare_phrase_positions,
    line_span_positions,
    load_model_bf16_eager,
    phrase_positions,
)
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import install_source_contribution_removal, project_source_contribution  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402
from phase785_positive_negative_subspace_split import parse_budgets, parse_csv  # noqa: E402
from phase786_head_mlp_source_audit import (  # noqa: E402
    capture_answer_outputs_and_sources,
    component_keys_for_routes,
    enrich_selected_rows_with_target_id,
)
from phase788_matched_source_unit_causal_fiber_validation import route_source_candidates, subspace_specs  # noqa: E402


OUT_ROOT = Path("results/glm5_phase791_upstream_qkv_source_token_causal_fiber_trace")
RESULT_ROOT = Path("tests/result/phase791_upstream_qkv_source_token_causal_fiber_trace")

DEFAULT_SOURCE_GROUPS = [
    "object_tokens",
    "relation_tokens",
    "target_value_tokens",
    "candidate_tokens",
    "answer_prefix",
    "instruction",
    "question",
    "all_pre_answer",
]


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
    vals = [bool(v) for v in values]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def relation_label(relation: str) -> str:
    if relation == "grows_on_tree":
        return "whether it grows on a tree"
    if relation == "edible":
        return "whether it is edible"
    return str(relation).replace("_", " ")


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def source_groups_for_prompt(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    answer_pos = len(ids) - 1
    obj = str(case.get("object") or "")
    relation = str(case.get("relation") or "")
    answer = str(case.get("answer") or case.get("target") or "")
    question = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Question:") or s.startswith("Task:"))
    answer_prefix = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Answer:"))
    candidate_tokens = line_span_positions(
        tokenizer,
        prompt,
        lambda s: s.startswith("Allowed values:") or s.startswith("Candidate") or s.startswith("Options:"),
    )
    instruction = line_span_positions(
        tokenizer,
        prompt,
        lambda s: (
            s.startswith("Answer using")
            or s.startswith("Use exactly")
            or s.startswith("Use common")
            or s.startswith("Answer with")
            or s.startswith("Write exactly")
            or s.startswith("Do not")
            or s.startswith("Output exactly")
            or s.startswith("No capital")
            or s.startswith("Return only")
        ),
    )
    relation_phrases = [relation, relation.replace("_", " "), relation_label(relation)]
    if relation == "grows_on_tree":
        relation_phrases += ["tree", "grow", "grows"]
    if relation == "edible":
        relation_phrases += ["edible", "eat"]
    answer_hits = [p for p in bare_phrase_positions(tokenizer, ids, [answer, answer.lower(), answer.capitalize()]) if p < answer_pos]
    groups = {
        "instruction": instruction,
        "question": question,
        "object_tokens": phrase_positions(tokenizer, ids, [obj, obj.lower(), obj.capitalize()]),
        "relation_tokens": phrase_positions(tokenizer, ids, relation_phrases),
        "target_value_tokens": answer_hits,
        "candidate_tokens": candidate_tokens,
        "answer_prefix": [p for p in answer_prefix if p < answer_pos],
        "all_pre_answer": list(range(0, max(0, answer_pos))),
        "self_last": [answer_pos],
    }
    return {k: [p for p in sorted(set(v)) if 0 <= p < len(ids)] for k, v in groups.items()}


def capture_path_state(model, tokenizer, device, prompt: str, case: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
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
        if out.attentions is None:
            raise RuntimeError("model returned no attentions; Phase 791 requires eager attention outputs")
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in sorted(set(layers))}
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return {
        "ids": ids,
        "prompt": prompt,
        "answer_pos": len(ids) - 1,
        "logits": logits,
        "attentions": attentions,
        "values": value_store,
        "source_groups": source_groups_for_prompt(tokenizer, prompt, case, ids),
    }


def run_with_hooks(model, device, ids: list[int], install) -> torch.Tensor:
    handles = install()
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def attention_mass_to_positions(attentions: Any, head_ids: list[int], answer_pos: int, src_positions: list[int]) -> float | None:
    if not src_positions or not head_ids:
        return None
    vals = []
    for head_id in head_ids:
        try:
            vals.append(float(torch.tensor(attentions[0, int(head_id), answer_pos, src_positions], dtype=torch.float32).sum().item()))
        except Exception:
            continue
    return sum(vals) / len(vals) if vals else None


def selected_contribution_norm(contribution: torch.Tensor, head_ids: list[int]) -> float | None:
    if contribution.numel() == 0 or not head_ids:
        return None
    contrib = contribution.detach().float().cpu()
    if contrib.ndim == 3:
        contrib = contrib[0]
    valid = [int(h) for h in head_ids if 0 <= int(h) < contrib.shape[0]]
    if not valid:
        return None
    return float(torch.linalg.vector_norm(contrib[valid].reshape(-1)).item())


def make_path_row(
    args: argparse.Namespace,
    case: dict[str, Any],
    route: dict[str, Any],
    meta: dict[str, Any],
    source_group: str,
    src_positions: list[int],
    donor_state: dict[str, Any],
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    projected: torch.Tensor,
    contribution: torch.Tensor,
    direct: dict[str, float],
) -> dict[str, Any]:
    base_logits = donor_state["logits"]
    target_before = logit_diag(base_logits, target_id)
    target_after = logit_diag(after_logits, target_id)
    contrast_before = logit_diag(base_logits, contrast_id)
    contrast_after = logit_diag(after_logits, contrast_id)
    before_margin = margin(base_logits, target_id, contrast_id)
    after_margin = margin(after_logits, target_id, contrast_id)
    head_ids = [int(x) for x in meta["source_unit_ids"]]
    layer = int(meta["layer"])
    attn_mass = attention_mass_to_positions(donor_state["attentions"][layer], head_ids, donor_state["answer_pos"], src_positions)
    return {
        "row_kind": "phase791_upstream_qkv_source_token_causal_fiber_trace",
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "domain": case.get("domain"),
        "relation": case.get("relation"),
        "object": case.get("object"),
        "target_answer": case.get("answer"),
        "route_id": route["route_id"],
        "compare_variant": route["compare_variant"],
        "component_kind": meta["component_kind"],
        "layer": layer,
        "source_component_label": meta["source_component_label"],
        "subspace_mode": meta["subspace_mode"],
        "budget_label": meta["budget_label"],
        "source_selection_kind": meta["source_selection_kind"],
        "source_unit_kind": meta["source_unit_kind"],
        "source_set_size": meta["source_set_size"],
        "source_unit_ids": head_ids,
        "source_group": source_group,
        "source_positions_n": len(src_positions),
        "qk_attention_mass_to_source": attn_mass,
        "v_source_contribution_norm": selected_contribution_norm(contribution, head_ids),
        "o_projected_delta_norm": float(torch.linalg.vector_norm(projected.float()).item()),
        "direct_target_boost": direct.get("direct_target_boost"),
        "direct_total_route_suppression": direct.get("direct_total_route_suppression"),
        "direct_mean_margin_gain": direct.get("direct_mean_margin_gain"),
        "direct_positive_route_count": direct.get("direct_positive_route_count"),
        "donor_target_top1": bool(target_before["target_top1"]),
        "after_target_top1": bool(target_after["target_top1"]),
        "top1_loss": bool(target_before["target_top1"]) and not bool(target_after["target_top1"]),
        "donor_target_rank": target_before["target_rank"],
        "after_target_rank": target_after["target_rank"],
        "donor_contrast_rank": contrast_before["target_rank"],
        "after_contrast_rank": contrast_after["target_rank"],
        "target_logit_drop": float(base_logits[target_id].item() - after_logits[target_id].item()),
        "contrast_logit_gain": float(after_logits[contrast_id].item() - base_logits[contrast_id].item()),
        "margin_drop_target_vs_contrast": float(before_margin - after_margin),
        "source_set_signed_score": meta.get("source_set_signed_score"),
        "source_set_abs_score": meta.get("source_set_abs_score"),
        "source_set_mean_delta_norm": meta.get("source_set_mean_delta_norm"),
        "source_set_mean_base_norm": meta.get("source_set_mean_base_norm"),
        "source_set_mean_matched_distance": meta.get("source_set_mean_matched_distance"),
        "path_stage": "source_token_attention_value_o_projection",
        "interpretation_boundary": "attention mass is a Q/K proxy; V contribution and O projection are tested, but Q/K are not patched separately.",
    }


def attention_path_candidates(
    model,
    route: dict[str, Any],
    base_answer_state: dict[str, Any],
    donor_answer_state: dict[str, Any],
    readout_direction: torch.Tensor,
    specs: list[tuple[str, str, int | None]],
    args: argparse.Namespace,
    seed_parts: tuple[Any, ...],
) -> list[dict[str, Any]]:
    out = []
    for mode, budget_label, budget in specs:
        rows = route_source_candidates(model, route, base_answer_state, donor_answer_state, readout_direction, mode, budget_label, budget, args, seed_parts)
        for row in rows:
            if row.get("source_unit_kind") != "attention_head_set":
                continue
            if row.get("source_selection_kind") not in {"top", "matched"}:
                continue
            out.append(row)
    return out


def audit_case_route(
    model,
    tokenizer,
    device,
    unembed: torch.Tensor,
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    route: dict[str, Any],
    component_keys: set[tuple[str, int]],
    specs: list[tuple[str, str, int | None]],
    source_groups: list[str],
) -> list[dict[str, Any]]:
    target_id = int(source_row.get("target_token_id") or source_row.get("target_id") or 0)
    contrast_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id") or 0)
    if not target_id or not contrast_id:
        raise ValueError(f"source row lacks target/contrast token ids: {source_row.keys()}")
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_variant = route["compare_variant"]
    donor_prompt = surface_prompt_for_variant(case, donor_variant)
    base_answer_state = capture_answer_outputs_and_sources(model, tokenizer, device, baseline_prompt, component_keys)
    donor_answer_state = capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    readout_direction = unembed[target_id].float() - unembed[contrast_id].float()
    seed_parts = (args.model, args.round_name, case["case_id"], route["route_id"])
    candidates = attention_path_candidates(model, route, base_answer_state, donor_answer_state, readout_direction, specs, args, seed_parts)
    layers = sorted({int(c["layer"]) for c in candidates})
    if not layers:
        return []
    donor_path_state = capture_path_state(model, tokenizer, device, donor_prompt, case, layers)
    rows = []
    for meta in candidates:
        layer = int(meta["layer"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        head_ids = [int(x) for x in meta["source_unit_ids"] if 0 <= int(x) < n_heads]
        if not head_ids:
            continue
        for source_group in source_groups:
            src_positions = [int(p) for p in donor_path_state["source_groups"].get(source_group, [])]
            if not src_positions:
                continue
            contribution = compute_source_contribution(
                donor_path_state["attentions"][layer],
                donor_path_state["values"][layer],
                [donor_path_state["answer_pos"]],
                [src_positions],
                n_heads,
                num_kv_heads,
            )
            projected = project_source_contribution(model, layer, head_ids, contribution)
            direct = direct_delta_score(projected, unembed, target_id, [contrast_id])
            install = install_source_contribution_removal(model, f"L{layer}:attn_out", head_ids, contribution)
            after_logits = run_with_hooks(model, device, donor_path_state["ids"], install)
            rows.append(
                make_path_row(
                    args,
                    case,
                    route,
                    meta,
                    source_group,
                    src_positions,
                    donor_path_state,
                    after_logits,
                    target_id,
                    contrast_id,
                    projected,
                    contribution,
                    direct,
                )
            )
    return rows


def group_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "component_counts": dict(Counter(v.get("source_component_label") for v in vals)),
                "mean_qk_attention_mass_to_source": safe_mean([v.get("qk_attention_mass_to_source") for v in vals]),
                "mean_v_source_contribution_norm": safe_mean([v.get("v_source_contribution_norm") for v in vals]),
                "mean_o_projected_delta_norm": safe_mean([v.get("o_projected_delta_norm") for v in vals]),
                "mean_direct_target_boost": safe_mean([v.get("direct_target_boost") for v in vals]),
                "mean_direct_total_route_suppression": safe_mean([v.get("direct_total_route_suppression") for v in vals]),
                "mean_direct_mean_margin_gain": safe_mean([v.get("direct_mean_margin_gain") for v in vals]),
                "mean_target_logit_drop": safe_mean([v.get("target_logit_drop") for v in vals]),
                "mean_contrast_logit_gain": safe_mean([v.get("contrast_logit_gain") for v in vals]),
                "mean_margin_drop_target_vs_contrast": safe_mean([v.get("margin_drop_target_vs_contrast") for v in vals]),
                "top1_loss_rate": safe_rate([v.get("top1_loss") for v in vals]),
                "donor_target_top1_rate": safe_rate([v.get("donor_target_top1") for v in vals]),
                "mean_source_set_abs_score": safe_mean([v.get("source_set_abs_score") for v in vals]),
                "mean_source_set_mean_delta_norm": safe_mean([v.get("source_set_mean_delta_norm") for v in vals]),
                "mean_source_set_mean_matched_distance": safe_mean([v.get("source_set_mean_matched_distance") for v in vals]),
            }
        )
        payload["path_effect_score"] = max(payload["mean_margin_drop_target_vs_contrast"] or 0.0, 0.0) * (
            1.0 + max(payload["mean_qk_attention_mass_to_source"] or 0.0, 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("path_effect_score") or 0.0, r.get("mean_margin_drop_target_vs_contrast") or 0.0), reverse=True)
    return out


def matched_path_comparisons(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ["model", "subspace_mode", "budget_label", "source_set_size", "source_group"]
    index: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    for row in summary_rows:
        key = tuple(row.get(k) for k in key_fields)
        index[(key, str(row.get("source_selection_kind")))] = row
    out = []
    for key in sorted({k for k, _sel in index}):
        top = index.get((key, "top"))
        matched = index.get((key, "matched"))
        if not top or not matched:
            continue
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "top_n": top.get("n"),
                "matched_n": matched.get("n"),
                "top_attention_mass": top.get("mean_qk_attention_mass_to_source"),
                "matched_attention_mass": matched.get("mean_qk_attention_mass_to_source"),
                "top_v_norm": top.get("mean_v_source_contribution_norm"),
                "matched_v_norm": matched.get("mean_v_source_contribution_norm"),
                "top_direct_margin_gain": top.get("mean_direct_mean_margin_gain"),
                "matched_direct_margin_gain": matched.get("mean_direct_mean_margin_gain"),
                "top_margin_drop": top.get("mean_margin_drop_target_vs_contrast"),
                "matched_margin_drop": matched.get("mean_margin_drop_target_vs_contrast"),
                "top_target_logit_drop": top.get("mean_target_logit_drop"),
                "matched_target_logit_drop": matched.get("mean_target_logit_drop"),
                "top1_loss_gap": (top.get("top1_loss_rate") or 0.0) - (matched.get("top1_loss_rate") or 0.0),
                "top_minus_matched_attention_mass": (top.get("mean_qk_attention_mass_to_source") or 0.0)
                - (matched.get("mean_qk_attention_mass_to_source") or 0.0),
                "top_minus_matched_margin_drop": (top.get("mean_margin_drop_target_vs_contrast") or 0.0)
                - (matched.get("mean_margin_drop_target_vs_contrast") or 0.0),
                "top_minus_matched_direct_margin_gain": (top.get("mean_direct_mean_margin_gain") or 0.0)
                - (matched.get("mean_direct_mean_margin_gain") or 0.0),
            }
        )
        payload["matched_path_specificity_score"] = max(payload["top_minus_matched_margin_drop"], 0.0) * (
            1.0 + max(payload["top_minus_matched_attention_mass"], 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("matched_path_specificity_score") or 0.0, r.get("top_minus_matched_margin_drop") or 0.0), reverse=True)
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], specs: list[tuple[str, str, int | None]], source_groups: list[str]) -> dict[str, Any]:
    by_path = group_rows(rows, ["model", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "source_group"])
    by_component = group_rows(rows, ["model", "source_component_label", "source_selection_kind", "subspace_mode", "source_group"])
    comparisons = matched_path_comparisons(by_path)
    return {
        "phase": 791,
        "title": "Upstream Q/K/V and Source-Token Causal Fiber Trace",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "source_groups": source_groups,
        "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in specs],
        "by_path": by_path,
        "by_component": by_component,
        "matched_path_comparisons": comparisons,
        "top_path_effects": by_path[:30],
        "top_matched_path_specificity": comparisons[:30],
        "strict_interpretation": (
            "This phase traces donor source-token groups through attention mass, V contribution, O projection, and source-contribution removal. "
            "Attention mass is a Q/K proxy; Q and K are not independently patched yet. It is a path-level audit, not full generation closure."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = select_surface_cases(args.model, args)
    routes = select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    component_keys = component_keys_for_routes(routes)
    specs = subspace_specs(parse_csv(args.subspace_modes), parse_budgets(args.budgets))
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)} specs={len(specs)} source_groups={source_groups}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys, specs, source_groups))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: upstream qkv/source-token path trace {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl, routes, specs, source_groups)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase791_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase791_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "top_matched_path_specificity": summary["top_matched_path_specificity"][:8],
                "top_path_effects": summary["top_path_effects"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 791 Upstream Q/K/V and Source-Token Causal Fiber Trace ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: donor source-token group contribution removal for Phase 788 matched-control attention source units.",
        "- Q/K path is represented by attention mass; V path by source value contribution; O path by projected contribution.",
        "- This is path-level audit, not full Q/K causal patch or generation closure.",
        "",
        "## Cross-Model Path Summary",
        "",
        "| model | selection | subspace | source group | cases | attn mass | v norm | direct margin | margin drop | top1 loss |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_path_effects", [])[:20]:
            lines.append(
                f"| {model_name} | `{row['source_selection_kind']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{row['case_n']} | {fmt(row['mean_qk_attention_mass_to_source'])} | {fmt(row['mean_v_source_contribution_norm'])} | "
                f"{fmt(row['mean_direct_mean_margin_gain'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | {fmt(row['top1_loss_rate'])} |"
            )
    lines += [
        "",
        "## Top Minus Matched Path Specificity",
        "",
        "| model | subspace | source group | top mass | matched mass | mass gap | top drop | matched drop | drop gap | direct gap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_matched_path_specificity", [])[:20]:
            lines.append(
                f"| {model_name} | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{fmt(row['top_attention_mass'])} | {fmt(row['matched_attention_mass'])} | {fmt(row['top_minus_matched_attention_mass'])} | "
                f"{fmt(row['top_margin_drop'])} | {fmt(row['matched_margin_drop'])} | {fmt(row['top_minus_matched_margin_drop'])} | "
                f"{fmt(row['top_minus_matched_direct_margin_gain'])} |"
            )
    lines += [
        "",
        "## Boundary",
        "",
        "- Attention mass is a Q/K proxy, not an independent Q/K patch.",
        "- The intervention removes source-group value contribution at donor prompt answer site.",
        "- Positive margin drop means this source path supported the target-vs-contrast margin in donor context.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_atlas_graph(payload: dict[str, Any]) -> dict[str, Any]:
    model_lanes = {name: idx for idx, name in enumerate(MODELS)}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node: dict[str, Any]) -> None:
        nodes[node["id"]] = {**nodes.get(node["id"], {}), **node}

    task_id = "phase791:task:upstream_qkv_source_token_trace"
    add_node(
        {
            "id": task_id,
            "type": "task",
            "model": "cross_model",
            "label": "Phase 791 upstream Q/K/V source-token trace",
            "role": "path_level_trace",
            "evidence_level": "path_contribution_removal",
            "phase": 791,
            "position": [0, 0, -1],
        }
    )
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        lane = model_lanes.get(model_name, 0)
        model_id = f"{model_name}:model"
        add_node({"id": model_id, "type": "model", "model": model_name, "label": model_name, "phase": 791, "position": [-4, 0, lane]})
        edges.append({"source": model_id, "target": task_id, "relation": "tested_by", "phase": 791, "weight": 1.0, "evidence": "path_trace"})
        for row in (data.get("top_matched_path_specificity") or [])[:24]:
            group = row.get("source_group")
            mode = row.get("subspace_mode")
            node_id = f"{model_name}:source_path:{mode}:{group}"
            add_node(
                {
                    "id": node_id,
                    "type": "cluster",
                    "model": model_name,
                    "label": f"{mode}:{group}",
                    "role": "source_token_path",
                    "evidence_level": "path_contribution_removal",
                    "phase": 791,
                    "source_group": group,
                    "subspace_mode": mode,
                    "score": row.get("matched_path_specificity_score"),
                    "top_minus_matched_margin_drop": row.get("top_minus_matched_margin_drop"),
                    "top_minus_matched_attention_mass": row.get("top_minus_matched_attention_mass"),
                    "position": [row.get("top_attention_mass") or 0.0, 30 + (row.get("top_margin_drop") or 0.0), lane],
                }
            )
            readout_id = f"{model_name}:readout:donor_margin"
            add_node({"id": readout_id, "type": "intervention", "model": model_name, "label": "donor margin removal", "role": "readout_competition", "phase": 791, "position": [6, 40, lane]})
            edges.append(
                {
                    "source": node_id,
                    "target": readout_id,
                    "relation": "upstream_of" if (row.get("top_minus_matched_margin_drop") or 0.0) >= 0 else "weak_or_inverse",
                    "weight": abs(row.get("top_minus_matched_margin_drop") or 0.0),
                    "phase": 791,
                    "evidence": "top_vs_matched_source_path_removal",
                    "top_minus_matched_margin_drop": row.get("top_minus_matched_margin_drop"),
                    "top_minus_matched_attention_mass": row.get("top_minus_matched_attention_mass"),
                }
            )
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 791 Upstream Q/K/V and Source-Token Causal Fiber Trace ({payload['round']})",
        "model_info": {
            "model": "cross_model",
            "models": payload.get("models", []),
            "phase": 791,
            "timestamp": payload.get("timestamp"),
            "evidence_type": "source-token attention value contribution removal",
        },
        "layout": {"x": "attention mass", "y": "path margin effect band", "z": "model lane"},
        "graph": {"nodes": list(nodes.values()), "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 791},
        "source_files": [f"results/glm5_phase791_upstream_qkv_source_token_causal_fiber_trace/{payload['round']}/phase791_cross_model_summary.json"],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase791_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 791,
        "title": "Upstream Q/K/V and Source-Token Causal Fiber Trace",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase791_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase791_cross_model_summary.md", payload)
        write_json(out_dir / "phase791_atlas_graph.json", build_atlas_graph(payload))
    print(json.dumps({"round": round_name, "status": payload["status"], "models": list(by_model)}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "models": {}}
    for model_name in MODELS:
        sub = argparse.Namespace(**{**vars(args), "model": model_name})
        selected = select_surface_cases(model_name, sub)
        routes = select_routes(model_name, sub)
        if sub.max_routes and len(routes) > sub.max_routes:
            routes = routes[: sub.max_routes]
        payload["models"][model_name] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "routes": routes,
            "source_groups": source_groups_for(sub),
            "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in subspace_specs(parse_csv(sub.subspace_modes), parse_budgets(sub.budgets))],
            "attn_source_set_size": sub.attn_source_set_size,
            "max_components_per_kind": sub.max_components_per_kind,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--max-routes", type=int, default=1)
    parser.add_argument("--budgets", default="1024")
    parser.add_argument("--subspace-modes", default="positive")
    parser.add_argument("--attn-source-set-size", type=int, default=4)
    parser.add_argument("--mlp-source-set-size", type=int, default=8)
    parser.add_argument("--max-components-per-kind", type=int, default=1)
    parser.add_argument("--max-source-groups", type=int, default=6)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --dry-run or --summarize-only")
    run_model(args)


if __name__ == "__main__":
    main()
