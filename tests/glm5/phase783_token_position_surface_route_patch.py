#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, line_span_positions  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import build_disentangled_source_groups, fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, observation_for, tensor_from_output  # noqa: E402
from phase781_surface_form_candidate_causal_patch import make_row, replacement_output  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase783_token_position_surface_route_patch")
RESULT_ROOT = Path("tests/result/phase783_token_position_surface_route_patch")


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


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def sorted_positions(values: list[int], n_tokens: int) -> list[int]:
    return sorted({int(v) for v in values if 0 <= int(v) < n_tokens})


def position_groups(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    groups = build_disentangled_source_groups(tokenizer, prompt, case, ids)
    answer_pos = len(ids) - 1
    surface_format = line_span_positions(
        tokenizer,
        prompt,
        lambda s: (
            s.startswith("Write exactly one short lowercase value.")
            or s.startswith("Do not use capital letters")
            or s.startswith("Output exactly one lowercase value.")
            or s.startswith("No capital letters")
            or s.startswith("Return only the canonical lowercase answer token")
            or s.startswith("Do not add spaces")
        ),
    )
    if surface_format:
        groups["format_cue"] = sorted_positions(groups.get("format_cue", []) + surface_format, len(ids))
        groups["instruction_no_candidate"] = sorted_positions(groups.get("instruction_no_candidate", []) + surface_format, len(ids))
        groups["protocol_all"] = sorted_positions(groups.get("protocol_all", []) + surface_format, len(ids))
    groups["answer_site"] = [answer_pos]
    groups["semantic_pair_plus_answer"] = sorted_positions(groups.get("semantic_pair", []) + [answer_pos], len(ids))
    groups["protocol_all_plus_answer"] = sorted_positions(groups.get("protocol_all", []) + [answer_pos], len(ids))
    groups["all_pre_answer_plus_answer"] = list(range(0, len(ids)))
    return {k: sorted_positions(v, len(ids)) for k, v in groups.items()}


def component_keys_for_routes(routes: list[dict[str, Any]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for route in routes:
        for comp in route.get("components") or []:
            out.add((str(comp["component_kind"]), int(comp["layer"])))
    return out


def capture_selected_components_for_prompt(
    model,
    tokenizer,
    device,
    prompt: str,
    component_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    layers = get_layers(model)
    captured: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    for kind, layer_idx in sorted(component_keys):
        layer = layers[layer_idx]
        module = getattr(layer, "self_attn" if kind == "attn" else "mlp")

        def hook(_module, _inputs, output, key=(kind, layer_idx)):
            tensor = tensor_from_output(output)
            if tensor is not None:
                captured[key] = tensor[0].detach().float().cpu()

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
            )
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return {"ids": ids, "logits": logits, "components": captured}


def values_for_scope(state: dict[str, Any], key: tuple[str, int], positions: list[int]) -> torch.Tensor | None:
    tensor = state["components"].get(key)
    if tensor is None or not positions:
        return None
    valid = [p for p in positions if 0 <= p < tensor.shape[0]]
    if not valid:
        return None
    return tensor[valid].clone()


def align_values(target_positions: list[int], donor_values: torch.Tensor | None) -> torch.Tensor | None:
    if donor_values is None or not target_positions:
        return None
    if donor_values.shape[0] == len(target_positions):
        return donor_values.clone()
    mean_vec = donor_values.mean(dim=0, keepdim=True)
    return mean_vec.repeat(len(target_positions), 1)


def fiber_replacements(
    source_state: dict[str, Any],
    route: dict[str, Any],
    source_positions: list[int],
    target_positions: list[int],
) -> dict[tuple[str, int], tuple[list[int], torch.Tensor]] | None:
    out: dict[tuple[str, int], tuple[list[int], torch.Tensor]] = {}
    if not source_positions or not target_positions:
        return None
    for comp in route["components"]:
        key = (comp["component_kind"], int(comp["layer"]))
        src_values = values_for_scope(source_state, key, source_positions)
        aligned = align_values(target_positions, src_values)
        if aligned is None:
            return None
        out[key] = (list(target_positions), aligned)
    return out


def run_with_fiber_replacement(
    model,
    tokenizer,
    device,
    prompt: str,
    replacements: dict[tuple[str, int], tuple[list[int], torch.Tensor]],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    layers = get_layers(model)
    handles = []
    for (kind, layer_idx), (positions, values) in replacements.items():
        layer = layers[layer_idx]
        module = getattr(layer, "self_attn" if kind == "attn" else "mlp")

        def hook(_module, _inputs, output, positions=positions, values=values):
            tensor = tensor_from_output(output)
            if tensor is None:
                return output
            patched = tensor.clone()
            max_pos = patched.shape[1]
            for idx, pos in enumerate(positions):
                if 0 <= int(pos) < max_pos and idx < values.shape[0]:
                    patched[0, int(pos)] = values[idx].to(device=patched.device, dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
            )
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return {"ids": ids, "logits": logits}


def observe_logits(tokenizer, logits: torch.Tensor, case: dict[str, Any], prompt_variant: str, source_row: dict[str, Any], case_variant_token_id: int, top_k: int) -> dict[str, Any]:
    obs, _top = observation_for(tokenizer, logits, case, prompt_variant, source_row, case_variant_token_id, top_k)
    return obs


def make_fiber_row(
    model_name: str,
    route: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    position_scope: str,
    intervention_kind: str,
    obs: dict[str, Any],
    reference_obs: dict[str, Any] | None,
    base_positions: list[int],
    donor_positions: list[int],
) -> dict[str, Any]:
    row = make_row(model_name, route, case, prompt_variant, intervention_kind, obs, reference_obs)
    row["row_kind"] = "phase783_token_position_fiber_observation"
    row["route_id"] = route["route_id"]
    row["route_size"] = route["route_size"]
    row["component_labels"] = route["component_labels"]
    row["components"] = route["components"]
    row["position_scope"] = position_scope
    row["base_position_n"] = len(base_positions)
    row["donor_position_n"] = len(donor_positions)
    row["position_alignment"] = "same_count" if len(base_positions) == len(donor_positions) else "mean_broadcast"
    return row


def audit_case_route_scope(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    route: dict[str, Any],
    component_keys: set[tuple[str, int]],
    position_scope: str,
) -> list[dict[str, Any]]:
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_variant = route["compare_variant"]
    donor_prompt = surface_prompt_for_variant(case, donor_variant)

    base_state = capture_selected_components_for_prompt(model, tokenizer, device, baseline_prompt, component_keys)
    donor_state = capture_selected_components_for_prompt(model, tokenizer, device, donor_prompt, component_keys)
    base_groups = position_groups(tokenizer, baseline_prompt, case, base_state["ids"])
    donor_groups = position_groups(tokenizer, donor_prompt, case, donor_state["ids"])
    base_positions = base_groups.get(position_scope, [])
    donor_positions = donor_groups.get(position_scope, [])
    if not base_positions or not donor_positions:
        return []

    base_obs = observe_logits(tokenizer, base_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    donor_obs = observe_logits(tokenizer, donor_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)

    rows = [
        make_fiber_row(args.model, route, case, COMPARE_BASELINE, position_scope, "normal_baseline", base_obs, None, base_positions, donor_positions),
        make_fiber_row(args.model, route, case, donor_variant, position_scope, "normal_donor", donor_obs, None, base_positions, donor_positions),
    ]

    donor_to_base = fiber_replacements(donor_state, route, donor_positions, base_positions)
    base_to_donor = fiber_replacements(base_state, route, base_positions, donor_positions)
    if donor_to_base is None or base_to_donor is None:
        return rows

    patch_state = run_with_fiber_replacement(model, tokenizer, device, baseline_prompt, donor_to_base)
    patch_obs = observe_logits(tokenizer, patch_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    rows.append(
        make_fiber_row(
            args.model,
            route,
            case,
            COMPARE_BASELINE,
            position_scope,
            "patch_baseline_from_donor_fiber",
            patch_obs,
            base_obs,
            base_positions,
            donor_positions,
        )
    )

    replace_state = run_with_fiber_replacement(model, tokenizer, device, donor_prompt, base_to_donor)
    replace_obs = observe_logits(tokenizer, replace_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    rows.append(
        make_fiber_row(
            args.model,
            route,
            case,
            donor_variant,
            position_scope,
            "replace_donor_fiber_with_baseline",
            replace_obs,
            donor_obs,
            base_positions,
            donor_positions,
        )
    )

    if args.include_zero:
        zero_repl = {key: (positions, torch.zeros_like(values)) for key, (positions, values) in donor_to_base.items()}
        zero_state = run_with_fiber_replacement(model, tokenizer, device, donor_prompt, zero_repl)
        zero_obs = observe_logits(tokenizer, zero_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
        rows.append(
            make_fiber_row(
                args.model,
                route,
                case,
                donor_variant,
                position_scope,
                "zero_donor_fiber",
                zero_obs,
                donor_obs,
                base_positions,
                donor_positions,
            )
        )
    return rows


def group_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(items),
                "case_n": len({r["case_id"] for r in items}),
                "component_labels": items[0].get("component_labels"),
                "mean_base_position_n": safe_mean([r.get("base_position_n") for r in items]),
                "mean_donor_position_n": safe_mean([r.get("donor_position_n") for r in items]),
                "alignment_counts": dict(Counter(str(r.get("position_alignment")) for r in items)),
                "strict_open_rate": safe_rate([r.get("target_top1") for r in items]),
                "semantic_equiv_open_rate": safe_rate([r.get("semantic_equiv_open") for r in items]),
                "pool_top1_rate": safe_rate([r.get("pool_target_top1") for r in items]),
                "mean_margin_target_vs_case_variant": safe_mean([r.get("margin_target_vs_case_variant") for r in items]),
                "mean_target_rank": safe_mean([r.get("target_rank") for r in items]),
                "mean_delta_margin_vs_reference": safe_mean([r.get("delta_margin_vs_reference") for r in items]),
                "strict_gain_rate_vs_reference": safe_rate([r.get("strict_gain_vs_reference") for r in items if "strict_gain_vs_reference" in r]),
                "strict_loss_rate_vs_reference": safe_rate([r.get("strict_loss_vs_reference") for r in items if "strict_loss_vs_reference" in r]),
                "semantic_loss_rate_vs_reference": safe_rate([r.get("semantic_loss_vs_reference") for r in items if "semantic_loss_vs_reference" in r]),
                "pool_loss_rate_vs_reference": safe_rate([r.get("pool_loss_vs_reference") for r in items if "pool_loss_vs_reference" in r]),
                "top1_classes": dict(Counter(str(r.get("top1_competitor_class")) for r in items)),
            }
        )
        payload["sufficiency_score"] = (
            (payload["mean_delta_margin_vs_reference"] or 0.0) * (payload["strict_gain_rate_vs_reference"] or 0.0)
            if payload.get("intervention_kind") == "patch_baseline_from_donor_fiber"
            else 0.0
        )
        payload["necessity_score"] = (
            (-(payload["mean_delta_margin_vs_reference"] or 0.0))
            * ((payload["strict_loss_rate_vs_reference"] or 0.0) + 0.5 * (payload["semantic_loss_rate_vs_reference"] or 0.0))
            if payload.get("intervention_kind") in {"replace_donor_fiber_with_baseline", "zero_donor_fiber"}
            else 0.0
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("model") or "",
            r.get("compare_variant") or "",
            r.get("route_id") or "",
            r.get("position_scope") or "",
            r.get("intervention_kind") or "",
        )
    )
    return out


def add_answer_site_advantage(by_group: list[dict[str, Any]]) -> None:
    answer_patch: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in by_group:
        if row.get("position_scope") == "answer_site" and row.get("intervention_kind") == "patch_baseline_from_donor_fiber":
            key = (row.get("model"), row.get("route_id"), row.get("compare_variant"), row.get("route_size"))
            answer_patch[key] = row
    for row in by_group:
        if row.get("intervention_kind") != "patch_baseline_from_donor_fiber":
            continue
        key = (row.get("model"), row.get("route_id"), row.get("compare_variant"), row.get("route_size"))
        ref = answer_patch.get(key)
        if not ref:
            continue
        row["delta_strict_gain_vs_answer_site"] = (row.get("strict_gain_rate_vs_reference") or 0.0) - (ref.get("strict_gain_rate_vs_reference") or 0.0)
        row["delta_margin_gain_vs_answer_site"] = (row.get("mean_delta_margin_vs_reference") or 0.0) - (ref.get("mean_delta_margin_vs_reference") or 0.0)


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], position_scopes: list[str]) -> dict[str, Any]:
    by_scope = group_rows(rows, ["model", "route_id", "compare_variant", "route_size", "position_scope", "intervention_kind"])
    add_answer_site_advantage(by_scope)
    top_suff = sorted(
        [r for r in by_scope if r.get("intervention_kind") == "patch_baseline_from_donor_fiber"],
        key=lambda r: (
            r.get("sufficiency_score") or 0.0,
            r.get("strict_gain_rate_vs_reference") or 0.0,
            r.get("mean_delta_margin_vs_reference") or 0.0,
        ),
        reverse=True,
    )
    top_adv = sorted(
        [r for r in by_scope if r.get("intervention_kind") == "patch_baseline_from_donor_fiber" and r.get("position_scope") != "answer_site"],
        key=lambda r: (
            r.get("delta_strict_gain_vs_answer_site") or 0.0,
            r.get("delta_margin_gain_vs_answer_site") or 0.0,
        ),
        reverse=True,
    )
    top_nec = sorted(
        [r for r in by_scope if r.get("intervention_kind") in {"replace_donor_fiber_with_baseline", "zero_donor_fiber"}],
        key=lambda r: (r.get("necessity_score") or 0.0, -(r.get("mean_delta_margin_vs_reference") or 0.0)),
        reverse=True,
    )
    return {
        "phase": 783,
        "title": "Token-Position Surface Route Patch",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase780_round": args.source_phase780_round,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "position_scopes": position_scopes,
        "routes": routes,
        "method_note": "Patch/replace Phase 782 routes over token-position scopes. Non-equal source/target scope lengths are mean-broadcast.",
        "by_scope_intervention": by_scope,
        "top_sufficiency_fibers": top_suff,
        "top_answer_site_advantages": top_adv,
        "top_necessity_fibers": top_nec,
        "strict_interpretation": (
            "This is a position-scope fiber test at block-level component granularity. "
            "It does not yet prove head/channel/neuron-level mechanisms."
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
    position_scopes = parse_csv(args.position_scopes)
    component_keys = component_keys_for_routes(routes)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)} scopes={position_scopes}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                for scope in position_scopes:
                    rows.extend(audit_case_route_scope(model, tokenizer, device, args, case, source_row, route, component_keys, scope))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: token-position route patch {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, args, attn_impl, routes, position_scopes)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase783_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase783_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_routes": summary["n_routes"],
                "position_scopes": position_scopes,
                "top_sufficiency_fibers": summary["top_sufficiency_fibers"][:8],
                "top_answer_site_advantages": summary["top_answer_site_advantages"][:8],
                "top_necessity_fibers": summary["top_necessity_fibers"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 783 Token-Position Surface Route Patch ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: patch/replace Phase 782 route components over token-position scopes.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.",
        "- Strict interpretation: block-level position fiber test, not head/channel/neuron-level proof.",
        "",
        "## Routes And Scopes",
        "",
        "| model | route | compare | size | scopes | components |",
        "|---|---|---|---:|---|---|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        scopes = ", ".join(data.get("position_scopes") or [])
        for route in data.get("routes") or []:
            labels = ", ".join(route.get("component_labels") or [])
            lines.append(f"| {model} | `{route['route_id']}` | `{route['compare_variant']}` | {route['route_size']} | `{scopes}` | `{labels}` |")

    lines += [
        "",
        "## Top Sufficiency Fibers",
        "",
        "| model | route | scope | size | strict gain | delta margin | gain vs answer | margin vs answer | score | alignment |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_sufficiency_fibers") or [])[:12]:
            lines.append(
                f"| {model} | `{row['route_id']}` | `{row['position_scope']}` | {row['route_size']} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['mean_delta_margin_vs_reference'])} | "
                f"{fmt(row.get('delta_strict_gain_vs_answer_site'))} | {fmt(row.get('delta_margin_gain_vs_answer_site'))} | "
                f"{fmt(row['sufficiency_score'])} | `{json.dumps(row['alignment_counts'], ensure_ascii=False)}` |"
            )

    lines += [
        "",
        "## Top Answer-Site Advantages",
        "",
        "| model | route | scope | strict gain vs answer | margin gain vs answer | strict gain | delta margin |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_answer_site_advantages") or [])[:12]:
            lines.append(
                f"| {model} | `{row['route_id']}` | `{row['position_scope']}` | "
                f"{fmt(row.get('delta_strict_gain_vs_answer_site'))} | {fmt(row.get('delta_margin_gain_vs_answer_site'))} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['mean_delta_margin_vs_reference'])} |"
            )

    lines += [
        "",
        "## Top Necessity Fibers",
        "",
        "| model | route | scope | intervention | size | strict loss | semantic loss | delta margin | score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_necessity_fibers") or [])[:12]:
            lines.append(
                f"| {model} | `{row['route_id']}` | `{row['position_scope']}` | `{row['intervention_kind']}` | {row['route_size']} | "
                f"{fmt(row['strict_loss_rate_vs_reference'])} | {fmt(row['semantic_loss_rate_vs_reference'])} | "
                f"{fmt(row['mean_delta_margin_vs_reference'])} | {fmt(row['necessity_score'])} |"
            )

    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- If non-answer scopes beat answer_site, the route should be treated as a position-component fiber.",
        "- If answer_site remains best, Phase 782 likely captured a readout-side route.",
        "- Mean-broadcast rows are useful boundary probes, but same-count rows are cleaner causal evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase783_{model}_summary.json"
        if path.exists():
            by_model[model] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 783,
        "title": "Token-Position Surface Route Patch",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase783_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase783_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase780_round": args.source_phase780_round, "models": {}}
    for model in MODELS:
        args.model = model
        selected = select_surface_cases(model, args)
        routes = select_routes(model, args)
        if args.max_routes and len(routes) > args.max_routes:
            routes = routes[: args.max_routes]
        payload["models"][model] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "position_scopes": parse_csv(args.position_scopes),
            "routes": routes,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--position-scopes", default="answer_site,answer_prefix,format_cue,object_tokens,relation_tokens,semantic_pair,protocol_all")
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
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
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
