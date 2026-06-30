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
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash, normalize_token_text  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight, tensor_from_output  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402
from phase785_positive_negative_subspace_split import parse_budgets, parse_csv  # noqa: E402
from phase786_head_mlp_source_audit import capture_answer_outputs_and_sources, component_keys_for_routes, enrich_selected_rows_with_target_id  # noqa: E402
from phase788_matched_source_unit_causal_fiber_validation import subspace_specs  # noqa: E402
from phase791_upstream_qkv_source_token_causal_fiber_trace import source_groups_for_prompt  # noqa: E402
from phase793_qkvo_independent_causal_decomposition import target_ids_from_row  # noqa: E402
from phase794_qkvo_replacement_closure_validation import (  # noqa: E402
    attention_candidates,
    capture_projection_tensor,
    generate_text_with_install,
    install_qkvo_replacement,
    pair_positions,
    run_logits_with_install,
    source_groups_for,
    top1_id,
)


OUT_ROOT = Path("results/glm5_phase795_multi_component_causal_fiber_closure")
RESULT_ROOT = Path("tests/result/phase795_multi_component_causal_fiber_closure")

DEFAULT_LADDERS = [
    "o_only",
    "kv_source",
    "kv_o",
    "route_answer",
    "kv_o_route",
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


def norm(value: Any) -> str:
    return normalize_token_text("" if value is None else str(value)).strip().lower()


def phrase_hit(text: str | None, answer: str) -> bool:
    if text is None:
        return False
    answer_norm = norm(answer)
    text_norm = norm(text)
    if not answer_norm:
        return False
    return answer_norm == text_norm or answer_norm in text_norm.split() or answer_norm in text_norm


def selected_route_components(route: dict[str, Any], allowed_kinds: set[str], max_components: int) -> list[dict[str, Any]]:
    comps = []
    for comp in route.get("components") or []:
        if comp.get("component_kind") in allowed_kinds:
            comps.append(comp)
    if max_components > 0:
        comps = comps[:max_components]
    return comps


def install_route_component_replacements(
    model,
    donor_state: dict[str, Any],
    route_components: list[dict[str, Any]],
    recipient_answer_pos: int,
) -> list[Any]:
    handles = []
    layers = get_layers(model)
    for comp in route_components:
        key = (comp["component_kind"], int(comp["layer"]))
        vec = donor_state.get("components", {}).get(key)
        if vec is None or not torch.is_tensor(vec) or vec.numel() == 0:
            continue
        layer = layers[key[1]]
        module = getattr(layer, "self_attn" if key[0] == "attn" else "mlp", None)
        if module is None:
            continue

        def hook(_module, _inputs, output, vec=vec, answer_pos=recipient_answer_pos):
            tensor = tensor_from_output(output)
            if tensor is None:
                return output
            patched = tensor.clone()
            patched[0, answer_pos] = vec.to(device=patched.device, dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handles.append(module.register_forward_hook(hook))
    return handles


def normal_generate(model, tokenizer, device, ids: list[int], max_new_tokens: int) -> tuple[str | None, str | None]:
    return generate_text_with_install(model, tokenizer, device, ids, lambda: [], max_new_tokens)


def build_ladder_install(
    model,
    donor_tensors: dict[str, torch.Tensor],
    donor_state: dict[str, Any],
    ladder_id: str,
    layer: int,
    head_ids: list[int],
    recipient_answer_pos: int,
    donor_answer_pos: int,
    source_group: str,
    recipient_source_positions: list[int],
    donor_source_positions: list[int],
    route_components: list[dict[str, Any]],
) -> Callable[[], list[Any]]:
    def install() -> list[Any]:
        handles: list[Any] = []
        if ladder_id in {"o_only", "kv_o", "kv_o_route", "q_o", "q_kv_o"}:
            tensor = donor_tensors.get("o_answer_replace")
            if tensor is not None and tensor.numel() > 0:
                handles.extend(
                    install_qkvo_replacement(
                        model,
                        layer,
                        head_ids,
                        "o_answer_replace",
                        tensor,
                        [recipient_answer_pos],
                        [donor_answer_pos],
                        "o_answer_replace:answer_position",
                    )
                )
        if ladder_id in {"q_o", "q_kv_o"}:
            tensor = donor_tensors.get("q_answer_replace")
            if tensor is not None and tensor.numel() > 0:
                handles.extend(
                    install_qkvo_replacement(
                        model,
                        layer,
                        head_ids,
                        "q_answer_replace",
                        tensor,
                        [recipient_answer_pos],
                        [donor_answer_pos],
                        "q_answer_replace:answer_position",
                    )
                )
        if ladder_id in {"kv_source", "kv_o", "kv_o_route", "q_kv_o"}:
            for op in ("k_source_replace", "v_source_replace"):
                tensor = donor_tensors.get(op)
                if tensor is None or tensor.numel() == 0:
                    continue
                handles.extend(
                    install_qkvo_replacement(
                        model,
                        layer,
                        head_ids,
                        op,
                        tensor,
                        recipient_source_positions,
                        donor_source_positions,
                        source_group,
                    )
                )
        if ladder_id in {"route_answer", "kv_o_route"}:
            handles.extend(install_route_component_replacements(model, donor_state, route_components, recipient_answer_pos))
        return handles

    return install


def make_row(
    args: argparse.Namespace,
    case: dict[str, Any],
    route: dict[str, Any],
    route_components: list[dict[str, Any]],
    meta: dict[str, Any],
    ladder_id: str,
    source_group: str,
    paired_count: int,
    recipient_variant: str,
    donor_variant: str,
    recipient_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    error: str | None,
    generated_text: str | None,
    generate_error: str | None,
    recipient_generated_text: str | None,
    donor_generated_text: str | None,
    recipient_generate_error: str | None,
    donor_generate_error: str | None,
) -> dict[str, Any]:
    recipient_target = logit_diag(recipient_logits, target_id)
    donor_target = logit_diag(donor_logits, target_id)
    after_target = logit_diag(after_logits, target_id) if after_logits.numel() else {}
    recipient_margin = margin(recipient_logits, target_id, contrast_id)
    donor_margin = margin(donor_logits, target_id, contrast_id)
    after_margin = margin(after_logits, target_id, contrast_id) if after_logits.numel() else float("nan")
    recipient_rank = int(recipient_target["target_rank"])
    donor_rank = int(donor_target["target_rank"])
    after_rank = int(after_target.get("target_rank", recipient_rank)) if after_target else None
    recipient_top1 = bool(recipient_target["target_top1"])
    donor_top1 = bool(donor_target["target_top1"])
    after_top1 = bool(after_target.get("target_top1")) if after_target else None
    generated_hit = phrase_hit(generated_text, str(case.get("answer") or "")) if generated_text is not None else None
    recipient_generated_hit = phrase_hit(recipient_generated_text, str(case.get("answer") or "")) if recipient_generated_text is not None else None
    donor_generated_hit = phrase_hit(donor_generated_text, str(case.get("answer") or "")) if donor_generated_text is not None else None
    route_labels = [f"{c['component_kind']}:L{int(c['layer'])}" for c in route_components]
    return {
        "row_kind": "phase795_multi_component_causal_fiber_closure",
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "domain": case.get("domain"),
        "relation": case.get("relation"),
        "object": case.get("object"),
        "target_answer": case.get("answer"),
        "route_id": route["route_id"],
        "compare_variant": route["compare_variant"],
        "recipient_variant": recipient_variant,
        "donor_variant": donor_variant,
        "ladder_id": ladder_id,
        "source_group": source_group,
        "paired_position_count": int(paired_count),
        "route_component_labels": route_labels,
        "route_component_count": len(route_labels),
        "source_component_label": meta["source_component_label"],
        "source_selection_kind": meta["source_selection_kind"],
        "source_unit_kind": meta["source_unit_kind"],
        "source_set_size": meta["source_set_size"],
        "source_unit_ids": [int(x) for x in meta["source_unit_ids"]],
        "subspace_mode": meta["subspace_mode"],
        "budget_label": meta["budget_label"],
        "source_qkvo_layer": int(meta["layer"]),
        "target_token_id": int(target_id),
        "contrast_token_id": int(contrast_id),
        "recipient_target_top1": recipient_top1,
        "donor_target_top1": donor_top1,
        "after_target_top1": after_top1,
        "recipient_top1_token_id": top1_id(recipient_logits),
        "donor_top1_token_id": top1_id(donor_logits),
        "after_top1_token_id": top1_id(after_logits),
        "recipient_target_rank": recipient_rank,
        "donor_target_rank": donor_rank,
        "after_target_rank": after_rank,
        "target_rank_delta_vs_recipient": (after_rank - recipient_rank) if after_rank is not None else None,
        "target_rank_improved": (after_rank < recipient_rank) if after_rank is not None else None,
        "recipient_margin_target_vs_contrast": float(recipient_margin),
        "donor_margin_target_vs_contrast": float(donor_margin),
        "after_margin_target_vs_contrast": float(after_margin) if after_logits.numel() else None,
        "delta_margin_vs_recipient": float(after_margin - recipient_margin) if after_logits.numel() else None,
        "donor_delta_margin_vs_recipient": float(donor_margin - recipient_margin),
        "target_logit_gain_vs_recipient": float(after_logits[target_id].item() - recipient_logits[target_id].item()) if after_logits.numel() else None,
        "token_closure_gain": bool(after_top1 and not recipient_top1) if after_top1 is not None else None,
        "token_closure_loss_vs_donor": bool(donor_top1 and not after_top1) if after_top1 is not None else None,
        "generated_text": generated_text,
        "recipient_generated_text": recipient_generated_text,
        "donor_generated_text": donor_generated_text,
        "generated_phrase_hit": generated_hit,
        "recipient_generated_phrase_hit": recipient_generated_hit,
        "donor_generated_phrase_hit": donor_generated_hit,
        "phrase_closure_gain": bool(generated_hit and not recipient_generated_hit) if generated_hit is not None and recipient_generated_hit is not None else None,
        "intervention_error": error,
        "generate_error": generate_error,
        "recipient_generate_error": recipient_generate_error,
        "donor_generate_error": donor_generate_error,
        "interpretation_boundary": (
            "This phase tests a multi-component replacement ladder. "
            "A ladder can raise margin/rank without achieving token or phrase closure. "
            "Route component replacement at the same attention layer can override lower-level O-path replacement, so ladder effects are causal assemblies, not simple sums."
        ),
    }


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
    ladders: list[str],
    source_groups: list[str],
    route_allowed_kinds: set[str],
) -> list[dict[str, Any]]:
    target_id, contrast_id = target_ids_from_row(tokenizer, case, source_row)
    recipient_variant = args.recipient_variant
    donor_variant = route["compare_variant"]
    recipient_prompt = surface_prompt_for_variant(case, recipient_variant)
    donor_prompt = surface_prompt_for_variant(case, donor_variant)
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    recipient_answer_pos = len(recipient_ids) - 1
    donor_answer_pos = len(donor_ids) - 1
    recipient_groups = source_groups_for_prompt(tokenizer, recipient_prompt, case, recipient_ids)
    donor_groups = source_groups_for_prompt(tokenizer, donor_prompt, case, donor_ids)
    recipient_state = capture_answer_outputs_and_sources(model, tokenizer, device, recipient_prompt, component_keys)
    donor_state = capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    recipient_generated_text: str | None = None
    donor_generated_text: str | None = None
    recipient_generate_error: str | None = None
    donor_generate_error: str | None = None
    if args.max_new_tokens > 0:
        recipient_generated_text, recipient_generate_error = normal_generate(model, tokenizer, device, recipient_ids, args.max_new_tokens)
        donor_generated_text, donor_generate_error = normal_generate(model, tokenizer, device, donor_ids, args.max_new_tokens)
    readout_direction = unembed[target_id].float() - unembed[contrast_id].float()
    route_components = selected_route_components(route, route_allowed_kinds, args.max_route_components)
    candidates = attention_candidates(
        model,
        route,
        recipient_state,
        donor_state,
        readout_direction,
        specs,
        args,
        (args.model, args.round_name, case["case_id"], route["route_id"]),
    )
    if not candidates and route_components:
        pseudo_layer = int(route_components[0]["layer"])
        candidates = []
        for selection_kind in ("top", "matched"):
            candidates.append(
                {
                    "component_kind": route_components[0]["component_kind"],
                    "layer": pseudo_layer,
                    "source_component_label": f"route_only:L{pseudo_layer}",
                    "source_selection_kind": selection_kind,
                    "source_unit_kind": "route_component_only",
                    "source_set_size": 0,
                    "source_unit_ids": [],
                    "subspace_mode": "route",
                    "budget_label": "route_only",
                    "budget_requested": None,
                }
            )
    rows: list[dict[str, Any]] = []
    for meta in candidates:
        layer = int(meta["layer"])
        head_ids = [int(x) for x in meta["source_unit_ids"]]
        tensor_cache: dict[str, torch.Tensor] = {}
        if head_ids:
            for op in ("q_answer_replace", "k_source_replace", "v_source_replace", "o_answer_replace"):
                try:
                    tensor_cache[op] = capture_projection_tensor(model, device, donor_ids, layer, op)
                except Exception:
                    tensor_cache[op] = torch.empty(0)
        for source_group in source_groups:
            recipient_positions = recipient_groups.get(source_group, [])
            donor_positions = donor_groups.get(source_group, [])
            pairs = pair_positions(recipient_positions, donor_positions, source_group)
            if not pairs:
                continue
            for ladder_id in ladders:
                if not head_ids and ladder_id != "route_answer":
                    continue
                if ladder_id in {"route_answer"} and not route_components:
                    continue
                if ladder_id in {"kv_o_route"} and not route_components:
                    continue
                install = build_ladder_install(
                    model,
                    tensor_cache,
                    donor_state,
                    ladder_id,
                    layer,
                    head_ids,
                    recipient_answer_pos,
                    donor_answer_pos,
                    source_group,
                    recipient_positions,
                    donor_positions,
                    route_components,
                )
                after_logits, error = run_logits_with_install(model, device, recipient_ids, install)
                gen_text: str | None = None
                gen_error: str | None = None
                if args.max_new_tokens > 0:
                    gen_text, gen_error = generate_text_with_install(model, tokenizer, device, recipient_ids, install, args.max_new_tokens)
                rows.append(
                    make_row(
                        args,
                        case,
                        route,
                        route_components,
                        meta,
                        ladder_id,
                        source_group,
                        len(pairs),
                        recipient_variant,
                        donor_variant,
                        recipient_state["logits"],
                        donor_state["logits"],
                        after_logits,
                        target_id,
                        contrast_id,
                        error,
                        gen_text,
                        gen_error,
                        recipient_generated_text,
                        donor_generated_text,
                        recipient_generate_error,
                        donor_generate_error,
                    )
                )
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "route_component_counts": dict(Counter(",".join(v.get("route_component_labels") or []) for v in vals)),
                "source_component_counts": dict(Counter(v.get("source_component_label") for v in vals)),
                "error_rate": safe_rate([v.get("intervention_error") for v in vals]),
                "generate_error_rate": safe_rate([v.get("generate_error") for v in vals if v.get("generated_text") is not None or v.get("generate_error") is not None]),
                "mean_paired_position_count": safe_mean([v.get("paired_position_count") for v in vals]),
                "mean_delta_margin_vs_recipient": safe_mean([v.get("delta_margin_vs_recipient") for v in vals]),
                "mean_donor_delta_margin_vs_recipient": safe_mean([v.get("donor_delta_margin_vs_recipient") for v in vals]),
                "mean_target_logit_gain_vs_recipient": safe_mean([v.get("target_logit_gain_vs_recipient") for v in vals]),
                "target_rank_improve_rate": safe_rate([v.get("target_rank_improved") for v in vals if v.get("target_rank_improved") is not None]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals if v.get("token_closure_gain") is not None]),
                "after_target_top1_rate": safe_rate([v.get("after_target_top1") for v in vals if v.get("after_target_top1") is not None]),
                "recipient_target_top1_rate": safe_rate([v.get("recipient_target_top1") for v in vals]),
                "donor_target_top1_rate": safe_rate([v.get("donor_target_top1") for v in vals]),
                "generated_phrase_hit_rate": safe_rate([v.get("generated_phrase_hit") for v in vals if v.get("generated_phrase_hit") is not None]),
                "recipient_generated_phrase_hit_rate": safe_rate([v.get("recipient_generated_phrase_hit") for v in vals if v.get("recipient_generated_phrase_hit") is not None]),
                "phrase_closure_gain_rate": safe_rate([v.get("phrase_closure_gain") for v in vals if v.get("phrase_closure_gain") is not None]),
            }
        )
        payload["closure_score"] = max(payload["mean_delta_margin_vs_recipient"] or 0.0, 0.0) * (
            1.0
            + max(payload["target_rank_improve_rate"] or 0.0, 0.0)
            + 3.0 * max(payload["token_closure_gain_rate"] or 0.0, 0.0)
            + 1.5 * max(payload["phrase_closure_gain_rate"] or 0.0, 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("closure_score") or 0.0, r.get("mean_delta_margin_vs_recipient") or 0.0), reverse=True)
    return out


def matched_comparisons(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ["model", "ladder_id", "subspace_mode", "budget_label", "source_set_size", "source_group"]
    idx: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    for row in grouped:
        key = tuple(row.get(k) for k in key_fields)
        idx[(key, str(row.get("source_selection_kind")))] = row
    out = []
    for key in sorted({k for k, _sel in idx}):
        top = idx.get((key, "top"))
        matched = idx.get((key, "matched"))
        if not top or not matched:
            continue
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "top_n": top.get("n"),
                "matched_n": matched.get("n"),
                "top_delta_margin": top.get("mean_delta_margin_vs_recipient"),
                "matched_delta_margin": matched.get("mean_delta_margin_vs_recipient"),
                "top_minus_matched_delta_margin": (top.get("mean_delta_margin_vs_recipient") or 0.0)
                - (matched.get("mean_delta_margin_vs_recipient") or 0.0),
                "top_token_gain": top.get("token_closure_gain_rate"),
                "matched_token_gain": matched.get("token_closure_gain_rate"),
                "top_minus_matched_token_gain": (top.get("token_closure_gain_rate") or 0.0)
                - (matched.get("token_closure_gain_rate") or 0.0),
                "top_phrase_gain": top.get("phrase_closure_gain_rate"),
                "matched_phrase_gain": matched.get("phrase_closure_gain_rate"),
                "top_minus_matched_phrase_gain": (top.get("phrase_closure_gain_rate") or 0.0)
                - (matched.get("phrase_closure_gain_rate") or 0.0),
                "top_rank_improve": top.get("target_rank_improve_rate"),
                "matched_rank_improve": matched.get("target_rank_improve_rate"),
                "top_minus_matched_rank_improve": (top.get("target_rank_improve_rate") or 0.0)
                - (matched.get("target_rank_improve_rate") or 0.0),
            }
        )
        payload["ladder_specificity_score"] = max(payload["top_minus_matched_delta_margin"], 0.0) * (
            1.0
            + max(payload["top_minus_matched_rank_improve"], 0.0)
            + 3.0 * max(payload["top_minus_matched_token_gain"], 0.0)
            + 1.5 * max(payload["top_minus_matched_phrase_gain"], 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("ladder_specificity_score") or 0.0, r.get("top_minus_matched_delta_margin") or 0.0), reverse=True)
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], ladders: list[str], source_groups: list[str]) -> dict[str, Any]:
    by_ladder = group_rows(rows, ["model", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "ladder_id", "source_group"])
    by_case_ladder = group_rows(rows, ["model", "case_id", "source_selection_kind", "ladder_id", "source_group"])
    comparisons = matched_comparisons(by_ladder)
    return {
        "phase": 795,
        "title": "Multi-Component Causal Fiber Closure Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "recipient_variant": args.recipient_variant,
        "ladders": ladders,
        "source_groups": source_groups,
        "max_new_tokens": args.max_new_tokens,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "by_ladder": by_ladder,
        "by_case_ladder": by_case_ladder,
        "matched_comparisons": comparisons,
        "top_ladder_effects": by_ladder[:60],
        "top_matched_specificity": comparisons[:60],
        "strict_boundary": (
            "This phase tests whether combining source K/V, answer O, and answer-position route outputs crosses token or phrase closure. "
            "It still uses approximate source groups and donor/recipient prompt variants; margin-only gains are not final language closure."
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
    specs = subspace_specs(parse_csv(args.subspace_modes), parse_budgets(args.budgets))
    source_groups = source_groups_for(args)
    ladders = parse_csv(args.ladders) or DEFAULT_LADDERS
    route_allowed_kinds = set(parse_csv(args.route_component_kinds) or ["attn", "mlp"])
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)} specs={len(specs)} ladders={ladders} source_groups={source_groups}")
    cmap = case_map_for(args)
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "domains": dict(Counter(cmap[row["case_id"]]["domain"] for row in selected if row["case_id"] in cmap)),
            "routes": routes,
            "source_groups": source_groups,
            "ladders": ladders,
            "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in specs],
        }
    component_keys = component_keys_for_routes(routes)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys, specs, ladders, source_groups, route_allowed_kinds))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: multi-component fiber closure {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl, routes, ladders, source_groups)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase795_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase795_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "top_matched_specificity": summary["top_matched_specificity"][:8],
                "top_ladder_effects": summary["top_ladder_effects"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def build_atlas(payload: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(node_id: str, node_type: str, **attrs: Any) -> None:
        nodes[node_id] = {**nodes.get(node_id, {}), "id": node_id, "type": node_type, **attrs}

    task = "phase795:multi_component_causal_fiber_closure"
    node(task, "task", label="Phase 795 multi-component causal fiber closure")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for row in summary.get("top_matched_specificity", [])[:30]:
            ladder_node = f"{model_name}:ladder:{row['ladder_id']}"
            group_node = f"{model_name}:source:{row['source_group']}"
            node(ladder_node, "causal_fiber_ladder", label=row["ladder_id"])
            node(group_node, "source_group", label=row["source_group"])
            edges.append(
                {
                    "id": f"{model_name}:{row['ladder_id']}:{row['source_group']}:{row['subspace_mode']}",
                    "source": model_node,
                    "target": ladder_node,
                    "type": "has_closure_pressure",
                    "weight": row.get("ladder_specificity_score"),
                    "metrics": row,
                }
            )
            edges.append(
                {
                    "id": f"{model_name}:{row['ladder_id']}->{row['source_group']}:{row['subspace_mode']}",
                    "source": ladder_node,
                    "target": group_node,
                    "type": "conditioned_on_source_group",
                    "weight": row.get("top_minus_matched_delta_margin"),
                }
            )
    return {
        "schema_version": "atlas_graph_v1",
        "phase": 795,
        "graph": {"nodes": list(nodes.values()), "edges": edges},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 795 Multi-Component Causal Fiber Closure ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Intervention ladder: O only, K/V source, K/V + O, route answer, K/V + O + route.",
        "- Goal: test whether multi-component assembly crosses token or phrase closure.",
        "- Strict interpretation: margin/rank gains alone are not full generation closure.",
        "",
        "## Top Minus Matched Ladder Specificity",
        "",
        "| model | ladder | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_matched_specificity", [])[:24]:
            lines.append(
                f"| {model_name} | `{row['ladder_id']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{fmt(row['top_delta_margin'])} | {fmt(row['matched_delta_margin'])} | {fmt(row['top_minus_matched_delta_margin'])} | "
                f"{fmt(row['top_minus_matched_token_gain'])} | {fmt(row['top_minus_matched_phrase_gain'])} | {fmt(row['top_minus_matched_rank_improve'])} |"
            )
    lines += [
        "",
        "## Top Ladder Effects",
        "",
        "| model | selection | ladder | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_ladder_effects", [])[:30]:
            lines.append(
                f"| {model_name} | `{row['source_selection_kind']}` | `{row['ladder_id']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{row['case_n']} | {fmt(row['mean_delta_margin_vs_recipient'])} | {fmt(row['target_rank_improve_rate'])} | "
                f"{fmt(row['token_closure_gain_rate'])} | {fmt(row['phrase_closure_gain_rate'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase795_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 795,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    for root in (OUT_ROOT / round_name, RESULT_ROOT / round_name):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "phase795_cross_model_summary.json", payload)
        write_json(root / "phase795_atlas_graph.json", build_atlas(payload))
        write_markdown(root / "phase795_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--recipient-variant", default=COMPARE_BASELINE)
    parser.add_argument("--relations", default="")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--budgets", default="1024")
    parser.add_argument("--subspace-modes", default="positive")
    parser.add_argument("--attn-source-set-size", type=int, default=4)
    parser.add_argument("--mlp-source-set-size", type=int, default=8)
    parser.add_argument("--max-components-per-kind", type=int, default=1)
    parser.add_argument("--max-source-groups", type=int, default=3)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--ladders", default=",".join(DEFAULT_LADDERS))
    parser.add_argument("--route-component-kinds", default="attn,mlp")
    parser.add_argument("--max-route-components", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    result = run_model(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
