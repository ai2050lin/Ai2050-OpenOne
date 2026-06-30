#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import string
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
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash, normalize_token_text  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402
from phase785_positive_negative_subspace_split import parse_budgets, parse_csv  # noqa: E402
from phase786_head_mlp_source_audit import capture_answer_outputs_and_sources, component_keys_for_routes, enrich_selected_rows_with_target_id  # noqa: E402
from phase788_matched_source_unit_causal_fiber_validation import subspace_specs  # noqa: E402
from phase791_upstream_qkv_source_token_causal_fiber_trace import source_groups_for_prompt  # noqa: E402
from phase793_qkvo_independent_causal_decomposition import target_ids_from_row  # noqa: E402
from phase794_qkvo_replacement_closure_validation import (  # noqa: E402
    attention_candidates,
    capture_projection_tensor,
    pair_positions,
    run_logits_with_install,
    source_groups_for,
    top1_id,
)
from phase795_multi_component_causal_fiber_closure import (  # noqa: E402
    DEFAULT_LADDERS,
    build_ladder_install,
    selected_route_components,
)


OUT_ROOT = Path("results/glm5_phase796_global_competitor_token_identity_audit")
RESULT_ROOT = Path("tests/result/phase796_global_competitor_token_identity_audit")

HIGH_FREQUENCY = {
    "the",
    "a",
    "an",
    "it",
    "its",
    "this",
    "that",
    "they",
    "there",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "as",
    "and",
    "or",
    "because",
    "usually",
    "typically",
    "generally",
    "answer",
    "yes",
    "no",
}


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
    vals = [bool(v) for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def norm_text(value: Any) -> str:
    return normalize_token_text("" if value is None else str(value)).strip().lower()


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def value_strings(case: dict[str, Any]) -> set[str]:
    vals = {
        case.get("answer"),
        case.get("contrast_answer"),
        case.get("object"),
        case.get("domain"),
        case.get("relation"),
    }
    return {norm_text(v) for v in vals if norm_text(v)}


def candidate_position_ids(tokenizer, ids: list[int], groups: dict[str, list[int]]) -> set[int]:
    out = set()
    for group_name in ("candidate_tokens", "target_value_tokens"):
        for pos in groups.get(group_name, []) or []:
            if 0 <= int(pos) < len(ids):
                out.add(int(ids[int(pos)]))
    return out


def classify_competitor(
    tokenizer,
    token_id: int,
    text: str,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
) -> str:
    tid = int(token_id)
    normalized = norm_text(text)
    lower_prompt = prompt_text.lower()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if tid == int(target_id):
        return "target"
    if tid == int(contrast_id):
        return "designated_contrast"
    if tid in special_ids:
        return "special_token"
    if not normalized or any(ch in text for ch in ["\n", "\r", "\t"]):
        return "whitespace_or_newline"
    if all(ch in string.punctuation for ch in normalized):
        return "punctuation"
    if tid in candidate_ids or normalized in case_values:
        return "candidate_list_or_case_value"
    if tid in set(prompt_ids) or (normalized and normalized in lower_prompt):
        return "echo_token"
    if normalized in HIGH_FREQUENCY:
        return "high_frequency_or_format"
    if any(ch.isdigit() for ch in normalized):
        return "number_or_symbol"
    if any(ch.isalpha() for ch in normalized):
        return "semantic_or_lexical_competitor"
    return "other_token"


def topk_snapshot(
    tokenizer,
    logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    top_k: int,
) -> dict[str, Any]:
    target_logit = float(logits[target_id].item())
    k = min(int(top_k), int(logits.numel()))
    vals, ids = torch.topk(logits, k)
    top_rows = []
    for rank, (val, tid_tensor) in enumerate(zip(vals.tolist(), ids.tolist()), 1):
        tid = int(tid_tensor)
        text = token_text(tokenizer, tid)
        cls = classify_competitor(tokenizer, tid, text, target_id, contrast_id, prompt_ids, prompt_text, candidate_ids, case_values)
        top_rows.append(
            {
                "rank": rank,
                "token_id": tid,
                "token_text": text,
                "token_text_norm": normalize_token_text(text),
                "class": cls,
                "is_target": tid == int(target_id),
                "logit": float(val),
                "gap_above_target": float(val - target_logit),
            }
        )
    non_target = next((row for row in top_rows if not row["is_target"]), None)
    target_diag = logit_diag(logits, target_id)
    global_margin = float(target_logit - non_target["logit"]) if non_target else float("inf")
    return {
        "target_logit": target_logit,
        "target_rank": int(target_diag["target_rank"]),
        "target_top1": bool(target_diag["target_top1"]),
        "top1_token_id": top1_id(logits),
        "top_non_target": non_target,
        "global_margin_target_vs_top_non_target": global_margin,
        "topk": top_rows,
    }


def logit_at(logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not logits.numel():
        return None
    return float(logits[int(token_id)].item())


def make_audit_row(
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
    recipient_prompt: str,
    donor_prompt: str,
    recipient_ids: list[int],
    donor_ids: list[int],
    recipient_candidate_ids: set[int],
    donor_candidate_ids: set[int],
    case_values: set[str],
    error: str | None,
) -> dict[str, Any]:
    rec = topk_snapshot(tokenizer=args._tokenizer, logits=recipient_logits, target_id=target_id, contrast_id=contrast_id, prompt_ids=recipient_ids, prompt_text=recipient_prompt, candidate_ids=recipient_candidate_ids, case_values=case_values, top_k=args.top_k)
    donor = topk_snapshot(tokenizer=args._tokenizer, logits=donor_logits, target_id=target_id, contrast_id=contrast_id, prompt_ids=donor_ids, prompt_text=donor_prompt, candidate_ids=donor_candidate_ids, case_values=case_values, top_k=args.top_k)
    after = topk_snapshot(tokenizer=args._tokenizer, logits=after_logits, target_id=target_id, contrast_id=contrast_id, prompt_ids=recipient_ids, prompt_text=recipient_prompt, candidate_ids=recipient_candidate_ids, case_values=case_values, top_k=args.top_k) if after_logits.numel() else None
    rec_comp = rec.get("top_non_target") or {}
    after_comp = after.get("top_non_target") if after else {}
    rec_comp_id = rec_comp.get("token_id")
    after_comp_id = after_comp.get("token_id") if after_comp else None
    rec_comp_after_logit = logit_at(after_logits, rec_comp_id)
    after_comp_rec_logit = logit_at(recipient_logits, after_comp_id)
    target_gain = float(after["target_logit"] - rec["target_logit"]) if after else None
    recipient_comp_delta = (rec_comp_after_logit - rec_comp["logit"]) if rec_comp and rec_comp_after_logit is not None else None
    after_comp_delta_vs_recipient = (after_comp["logit"] - after_comp_rec_logit) if after_comp and after_comp_rec_logit is not None else None
    global_margin_delta = (after["global_margin_target_vs_top_non_target"] - rec["global_margin_target_vs_top_non_target"]) if after else None
    route_labels = [f"{c['component_kind']}:L{int(c['layer'])}" for c in route_components]
    return {
        "row_kind": "phase796_global_competitor_token_identity_audit",
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "domain": case.get("domain"),
        "relation": case.get("relation"),
        "object": case.get("object"),
        "target_answer": case.get("answer"),
        "contrast_answer": case.get("contrast_answer"),
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
        "recipient_margin_target_vs_contrast": float(margin(recipient_logits, target_id, contrast_id)),
        "after_margin_target_vs_contrast": float(margin(after_logits, target_id, contrast_id)) if after_logits.numel() else None,
        "delta_margin_vs_recipient": float(margin(after_logits, target_id, contrast_id) - margin(recipient_logits, target_id, contrast_id)) if after_logits.numel() else None,
        "recipient_target_rank": rec["target_rank"],
        "after_target_rank": after["target_rank"] if after else None,
        "target_rank_delta_vs_recipient": (after["target_rank"] - rec["target_rank"]) if after else None,
        "target_rank_improved": (after["target_rank"] < rec["target_rank"]) if after else None,
        "recipient_target_top1": rec["target_top1"],
        "donor_target_top1": donor["target_top1"],
        "after_target_top1": after["target_top1"] if after else None,
        "token_closure_gain": bool(after and after["target_top1"] and not rec["target_top1"]) if after else None,
        "recipient_global_margin": rec["global_margin_target_vs_top_non_target"],
        "donor_global_margin": donor["global_margin_target_vs_top_non_target"],
        "after_global_margin": after["global_margin_target_vs_top_non_target"] if after else None,
        "delta_global_margin_vs_recipient": global_margin_delta,
        "global_margin_improved": (global_margin_delta is not None and global_margin_delta > 0),
        "global_margin_crossed": bool(after and after["global_margin_target_vs_top_non_target"] > 0 and rec["global_margin_target_vs_top_non_target"] <= 0),
        "target_logit_gain_vs_recipient": target_gain,
        "recipient_top_competitor_token_id": rec_comp.get("token_id"),
        "recipient_top_competitor_text": rec_comp.get("token_text"),
        "recipient_top_competitor_class": rec_comp.get("class"),
        "recipient_top_competitor_gap_above_target": rec_comp.get("gap_above_target"),
        "recipient_top_competitor_logit_delta_after": recipient_comp_delta,
        "recipient_top_competitor_suppressed": (recipient_comp_delta is not None and recipient_comp_delta < 0),
        "after_top_competitor_token_id": after_comp_id,
        "after_top_competitor_text": after_comp.get("token_text") if after_comp else None,
        "after_top_competitor_class": after_comp.get("class") if after_comp else None,
        "after_top_competitor_gap_above_target": after_comp.get("gap_above_target") if after_comp else None,
        "after_top_competitor_logit_delta_vs_recipient": after_comp_delta_vs_recipient,
        "after_top_competitor_new": bool(after_comp_id is not None and after_comp_id != rec_comp_id),
        "recipient_topk": rec["topk"],
        "after_topk": after["topk"] if after else [],
        "intervention_error": error,
        "interpretation_boundary": (
            "This phase audits full-vocabulary top-k competitors after the Phase 795 ladder. "
            "It distinguishes target boosting from suppressing the actual non-target token that blocks top1 closure."
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
    recipient_candidate_ids = candidate_position_ids(tokenizer, recipient_ids, recipient_groups)
    donor_candidate_ids = candidate_position_ids(tokenizer, donor_ids, donor_groups)
    case_vals = value_strings(case)
    recipient_state = capture_answer_outputs_and_sources(model, tokenizer, device, recipient_prompt, component_keys)
    donor_state = capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
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
                if ladder_id in {"route_answer", "kv_o_route"} and not route_components:
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
                rows.append(
                    make_audit_row(
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
                        recipient_prompt,
                        donor_prompt,
                        recipient_ids,
                        donor_ids,
                        recipient_candidate_ids,
                        donor_candidate_ids,
                        case_vals,
                        error,
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
        class_counts = Counter(v.get("after_top_competitor_class") for v in vals)
        recipient_class_counts = Counter(v.get("recipient_top_competitor_class") for v in vals)
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "mean_delta_margin_vs_recipient": safe_mean([v.get("delta_margin_vs_recipient") for v in vals]),
                "mean_delta_global_margin_vs_recipient": safe_mean([v.get("delta_global_margin_vs_recipient") for v in vals]),
                "mean_target_logit_gain_vs_recipient": safe_mean([v.get("target_logit_gain_vs_recipient") for v in vals]),
                "mean_recipient_top_competitor_delta_after": safe_mean([v.get("recipient_top_competitor_logit_delta_after") for v in vals]),
                "mean_after_top_competitor_delta_vs_recipient": safe_mean([v.get("after_top_competitor_logit_delta_vs_recipient") for v in vals]),
                "target_rank_improve_rate": safe_rate([v.get("target_rank_improved") for v in vals]),
                "global_margin_improve_rate": safe_rate([v.get("global_margin_improved") for v in vals]),
                "global_margin_cross_rate": safe_rate([v.get("global_margin_crossed") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "recipient_top_competitor_suppression_rate": safe_rate([v.get("recipient_top_competitor_suppressed") for v in vals]),
                "after_top_competitor_new_rate": safe_rate([v.get("after_top_competitor_new") for v in vals]),
                "after_top_competitor_class_counts": dict(class_counts),
                "recipient_top_competitor_class_counts": dict(recipient_class_counts),
            }
        )
        payload["global_competition_score"] = max(payload["mean_delta_global_margin_vs_recipient"] or 0.0, 0.0) * (
            1.0
            + max(payload["target_rank_improve_rate"] or 0.0, 0.0)
            + max(payload["recipient_top_competitor_suppression_rate"] or 0.0, 0.0)
            + 3.0 * max(payload["token_closure_gain_rate"] or 0.0, 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("global_competition_score") or 0.0, r.get("mean_delta_global_margin_vs_recipient") or 0.0), reverse=True)
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
                "top_delta_global_margin": top.get("mean_delta_global_margin_vs_recipient"),
                "matched_delta_global_margin": matched.get("mean_delta_global_margin_vs_recipient"),
                "top_minus_matched_delta_global_margin": (top.get("mean_delta_global_margin_vs_recipient") or 0.0)
                - (matched.get("mean_delta_global_margin_vs_recipient") or 0.0),
                "top_target_gain": top.get("mean_target_logit_gain_vs_recipient"),
                "matched_target_gain": matched.get("mean_target_logit_gain_vs_recipient"),
                "top_minus_matched_target_gain": (top.get("mean_target_logit_gain_vs_recipient") or 0.0)
                - (matched.get("mean_target_logit_gain_vs_recipient") or 0.0),
                "top_competitor_suppression_rate": top.get("recipient_top_competitor_suppression_rate"),
                "matched_competitor_suppression_rate": matched.get("recipient_top_competitor_suppression_rate"),
                "top_minus_matched_competitor_suppression_rate": (top.get("recipient_top_competitor_suppression_rate") or 0.0)
                - (matched.get("recipient_top_competitor_suppression_rate") or 0.0),
                "top_global_margin_cross_rate": top.get("global_margin_cross_rate"),
                "matched_global_margin_cross_rate": matched.get("global_margin_cross_rate"),
                "top_token_gain": top.get("token_closure_gain_rate"),
                "matched_token_gain": matched.get("token_closure_gain_rate"),
            }
        )
        payload["specific_global_score"] = max(payload["top_minus_matched_delta_global_margin"], 0.0) * (
            1.0 + max(payload["top_minus_matched_competitor_suppression_rate"], 0.0) + 3.0 * max(payload["top_token_gain"] or 0.0, 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("specific_global_score") or 0.0, r.get("top_minus_matched_delta_global_margin") or 0.0), reverse=True)
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], ladders: list[str], source_groups: list[str]) -> dict[str, Any]:
    by_ladder = group_rows(rows, ["model", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "ladder_id", "source_group"])
    by_case_ladder = group_rows(rows, ["model", "case_id", "source_selection_kind", "ladder_id", "source_group"])
    comparisons = matched_comparisons(by_ladder)
    top_comp_classes = Counter(r.get("after_top_competitor_class") for r in rows)
    recipient_comp_classes = Counter(r.get("recipient_top_competitor_class") for r in rows)
    return {
        "phase": 796,
        "title": "Global Competitor Suppression and Token Identity Closure Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "top_k": args.top_k,
        "recipient_variant": args.recipient_variant,
        "ladders": ladders,
        "source_groups": source_groups,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "recipient_top_competitor_class_counts": dict(recipient_comp_classes),
        "after_top_competitor_class_counts": dict(top_comp_classes),
        "by_ladder": by_ladder,
        "by_case_ladder": by_case_ladder,
        "matched_comparisons": comparisons,
        "top_global_effects": by_ladder[:80],
        "top_matched_specificity": comparisons[:80],
        "strict_boundary": (
            "This phase audits open-vocabulary competitors. "
            "Positive target-vs-contrast margin is not token closure unless target beats every non-target token."
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
    log(f"{args.model}/{args.round_name}: cases={len(selected)} routes={len(routes)} specs={len(specs)} ladders={ladders} groups={source_groups} top_k={args.top_k}")
    cmap = case_map_for(args)
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "source_groups": source_groups,
            "ladders": ladders,
            "top_k": args.top_k,
        }
    component_keys = component_keys_for_routes(routes)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    setattr(args, "_tokenizer", tokenizer)
    try:
        enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys, specs, ladders, source_groups, route_allowed_kinds))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: global competitor audit {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        if hasattr(args, "_tokenizer"):
            delattr(args, "_tokenizer")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl, routes, ladders, source_groups)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase796_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase796_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "after_top_competitor_class_counts": summary["after_top_competitor_class_counts"],
                "top_matched_specificity": summary["top_matched_specificity"][:8],
                "top_global_effects": summary["top_global_effects"][:8],
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

    task = "phase796:global_competitor_token_identity_audit"
    node(task, "task", label="Phase 796 global competitor and token identity audit")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for cls, count in (summary.get("after_top_competitor_class_counts") or {}).items():
            cls_node = f"competitor_class:{cls}"
            node(cls_node, "competitor_class", label=cls)
            edges.append(
                {
                    "id": f"{model_name}:after_competitor:{cls}",
                    "source": model_node,
                    "target": cls_node,
                    "type": "blocked_by_competitor_class",
                    "weight": count,
                }
            )
        for row in summary.get("top_matched_specificity", [])[:30]:
            ladder_node = f"{model_name}:ladder:{row['ladder_id']}"
            node(ladder_node, "causal_fiber_ladder", label=row["ladder_id"])
            edges.append(
                {
                    "id": f"{model_name}:{row['ladder_id']}:{row['source_group']}:{row['subspace_mode']}",
                    "source": model_node,
                    "target": ladder_node,
                    "type": "changes_global_margin",
                    "weight": row.get("specific_global_score"),
                    "metrics": row,
                }
            )
    return {"schema_version": "atlas_graph_v1", "phase": 796, "graph": {"nodes": list(nodes.values()), "edges": edges}}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 796 Global Competitor Suppression and Token Identity Closure Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Goal: audit the real top-k vocabulary competitors that block token top1 closure.",
        "- Boundary: target-vs-contrast improvement is not sufficient; target must beat every non-target token.",
        "",
        "## Competitor Class Counts After Intervention",
        "",
        "| model | class | count |",
        "|---|---|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for cls, count in sorted((data.get("after_top_competitor_class_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {model_name} | `{cls}` | {count} |")
    lines += [
        "",
        "## Top Minus Matched Global Specificity",
        "",
        "| model | ladder | subspace | source group | top global delta | matched global delta | gap | top target gain | top suppress rate | top token gain |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_matched_specificity", [])[:24]:
            lines.append(
                f"| {model_name} | `{row['ladder_id']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{fmt(row['top_delta_global_margin'])} | {fmt(row['matched_delta_global_margin'])} | "
                f"{fmt(row['top_minus_matched_delta_global_margin'])} | {fmt(row['top_target_gain'])} | "
                f"{fmt(row['top_competitor_suppression_rate'])} | {fmt(row['top_token_gain'])} |"
            )
    lines += [
        "",
        "## Top Global Effects",
        "",
        "| model | selection | ladder | subspace | source group | cases | global delta | target gain | suppress rate | cross rate | token gain |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_global_effects", [])[:30]:
            lines.append(
                f"| {model_name} | `{row['source_selection_kind']}` | `{row['ladder_id']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{row['case_n']} | {fmt(row['mean_delta_global_margin_vs_recipient'])} | {fmt(row['mean_target_logit_gain_vs_recipient'])} | "
                f"{fmt(row['recipient_top_competitor_suppression_rate'])} | {fmt(row['global_margin_cross_rate'])} | {fmt(row['token_closure_gain_rate'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase796_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 796,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    for root in (OUT_ROOT / round_name, RESULT_ROOT / round_name):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "phase796_cross_model_summary.json", payload)
        write_json(root / "phase796_atlas_graph.json", build_atlas(payload))
        write_markdown(root / "phase796_cross_model_summary.md", payload)
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
    parser.add_argument("--ladders", default="o_only,kv_source,kv_o,route_answer,kv_o_route")
    parser.add_argument("--route-component-kinds", default="attn,mlp")
    parser.add_argument("--max-route-components", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
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
