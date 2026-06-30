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
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import (  # noqa: E402
    COMPARE_BASELINE,
    capture_components_for_prompt,
    observation_for,
    tensor_from_output,
)
from phase781_surface_form_candidate_causal_patch import (  # noqa: E402
    load_phase780_summary,
    make_row,
    replacement_output,
)


OUT_ROOT = Path("results/glm5_phase782_multi_component_surface_route_patch")
RESULT_ROOT = Path("tests/result/phase782_multi_component_surface_route_patch")


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


def parse_sizes(text: str) -> list[int]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return sorted(set(v for v in vals if v > 0))


def select_routes(model: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    summary = load_phase780_summary(model, args.source_phase780_round)
    allowed_variants = {x.strip() for x in args.route_compare_variants.split(",") if x.strip()}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary.get("top_delta_by_compare_kind_layer") or []:
        variant = row.get("compare_variant")
        if variant == COMPARE_BASELINE:
            continue
        if allowed_variants and variant not in allowed_variants:
            continue
        if float(row.get("candidate_score") or 0.0) < args.min_candidate_score:
            continue
        key = (row.get("component_kind"), int(row.get("layer")))
        if any((x["component_kind"], x["layer"]) == key for x in by_variant[variant]):
            continue
        by_variant[variant].append(
            {
                "component_kind": row["component_kind"],
                "layer": int(row["layer"]),
                "phase780_candidate_score": row.get("candidate_score"),
                "phase780_direct_delta": row.get("mean_delta_direct_target_vs_case_variant"),
                "phase780_actual_margin_delta": row.get("mean_delta_actual_margin"),
            }
        )

    routes = []
    sizes = parse_sizes(args.route_sizes)
    for variant, comps in sorted(by_variant.items()):
        comps = comps[: args.max_route_candidates]
        for size in sizes:
            if size > len(comps):
                continue
            selected = comps[:size]
            route_id = f"{variant}:route_k{size}"
            routes.append(
                {
                    "candidate_id": route_id,
                    "route_id": route_id,
                    "compare_variant": variant,
                    "component_kind": "route",
                    "layer": -1,
                    "route_size": size,
                    "components": selected,
                    "component_labels": [f"{c['component_kind']}:L{c['layer']}" for c in selected],
                    "phase780_candidate_score": safe_mean([c.get("phase780_candidate_score") for c in selected]),
                    "phase780_direct_delta": safe_mean([c.get("phase780_direct_delta") for c in selected]),
                    "phase780_actual_margin_delta": safe_mean([c.get("phase780_actual_margin_delta") for c in selected]),
                    "phase780_strict_repair_rate": None,
                }
            )
    return routes


def run_with_route_replacement(
    model,
    tokenizer,
    device,
    prompt: str,
    replacements: dict[tuple[str, int], torch.Tensor],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    handles = []
    for (kind, layer_idx), vec in replacements.items():
        layer = layers[layer_idx]
        module = getattr(layer, "self_attn" if kind == "attn" else "mlp")

        def hook(_module, _inputs, output, answer_pos=answer_pos, vec=vec):
            return replacement_output(output, answer_pos, vec)

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
    return {"ids": ids, "answer_pos": answer_pos, "logits": logits}


def observe_logits(tokenizer, logits: torch.Tensor, case: dict[str, Any], prompt_variant: str, source_row: dict[str, Any], case_variant_token_id: int, top_k: int) -> dict[str, Any]:
    obs, _top = observation_for(tokenizer, logits, case, prompt_variant, source_row, case_variant_token_id, top_k)
    return obs


def make_route_row(
    model_name: str,
    route: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    intervention_kind: str,
    obs: dict[str, Any],
    reference_obs: dict[str, Any] | None,
) -> dict[str, Any]:
    row = make_row(model_name, route, case, prompt_variant, intervention_kind, obs, reference_obs)
    row["row_kind"] = "phase782_multi_component_route_observation"
    row["route_id"] = route["route_id"]
    row["route_size"] = route["route_size"]
    row["component_labels"] = route["component_labels"]
    row["components"] = route["components"]
    return row


def route_vectors(state: dict[str, Any], route: dict[str, Any]) -> dict[tuple[str, int], torch.Tensor] | None:
    out = {}
    for comp in route["components"]:
        key = (comp["component_kind"], int(comp["layer"]))
        vec = state["components"].get(key)
        if vec is None:
            return None
        out[key] = vec
    return out


def audit_case_route(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    route: dict[str, Any],
) -> list[dict[str, Any]]:
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_variant = route["compare_variant"]
    donor_prompt = surface_prompt_for_variant(case, donor_variant)

    base_state = capture_components_for_prompt(model, tokenizer, device, baseline_prompt)
    donor_state = capture_components_for_prompt(model, tokenizer, device, donor_prompt)
    base_obs = observe_logits(tokenizer, base_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    donor_obs = observe_logits(tokenizer, donor_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    base_repl = route_vectors(base_state, route)
    donor_repl = route_vectors(donor_state, route)
    if base_repl is None or donor_repl is None:
        return []

    rows = [
        make_route_row(args.model, route, case, COMPARE_BASELINE, "normal_baseline", base_obs, None),
        make_route_row(args.model, route, case, donor_variant, "normal_donor", donor_obs, None),
    ]

    patch_state = run_with_route_replacement(model, tokenizer, device, baseline_prompt, donor_repl)
    patch_obs = observe_logits(tokenizer, patch_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    rows.append(make_route_row(args.model, route, case, COMPARE_BASELINE, "patch_baseline_from_donor_route", patch_obs, base_obs))

    replace_state = run_with_route_replacement(model, tokenizer, device, donor_prompt, base_repl)
    replace_obs = observe_logits(tokenizer, replace_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    rows.append(make_route_row(args.model, route, case, donor_variant, "replace_donor_route_with_baseline", replace_obs, donor_obs))

    zero_repl = {key: torch.zeros_like(vec) for key, vec in donor_repl.items()}
    zero_state = run_with_route_replacement(model, tokenizer, device, donor_prompt, zero_repl)
    zero_obs = observe_logits(tokenizer, zero_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    rows.append(make_route_row(args.model, route, case, donor_variant, "zero_donor_route", zero_obs, donor_obs))
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
            if payload.get("intervention_kind") == "patch_baseline_from_donor_route"
            else 0.0
        )
        payload["necessity_score"] = (
            (-(payload["mean_delta_margin_vs_reference"] or 0.0))
            * ((payload["strict_loss_rate_vs_reference"] or 0.0) + 0.5 * (payload["semantic_loss_rate_vs_reference"] or 0.0))
            if payload.get("intervention_kind") in {"replace_donor_route_with_baseline", "zero_donor_route"}
            else 0.0
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("model") or "", r.get("compare_variant") or "", int(r.get("route_size") or 0), r.get("intervention_kind") or ""))
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    by_route = group_rows(rows, ["model", "route_id", "compare_variant", "route_size", "intervention_kind"])
    top_suff = sorted(
        [r for r in by_route if r.get("intervention_kind") == "patch_baseline_from_donor_route"],
        key=lambda r: (r.get("sufficiency_score") or 0.0, r.get("strict_open_rate") or 0.0, r.get("mean_delta_margin_vs_reference") or 0.0),
        reverse=True,
    )
    top_nec = sorted(
        [r for r in by_route if r.get("intervention_kind") in {"replace_donor_route_with_baseline", "zero_donor_route"}],
        key=lambda r: (r.get("necessity_score") or 0.0, -(r.get("mean_delta_margin_vs_reference") or 0.0)),
        reverse=True,
    )
    return {
        "phase": 782,
        "title": "Multi-Component Surface Route Patch",
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
        "routes": routes,
        "method_note": "Patch/replace/zero multi-component routes at answer position using Phase 780 ranked candidates.",
        "by_route_intervention": by_route,
        "top_sufficiency_routes": top_suff,
        "top_necessity_routes": top_nec,
        "strict_interpretation": (
            "Route-level patch tests whether combined candidates are stronger than single blocks. "
            "Failure may indicate missing token positions or downstream generation closure, not absence of causal structure."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = select_surface_cases(args.model, args)
    routes = select_routes(args.model, args)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, args, case, source_row, route))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: multi-route patch {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, args, attn_impl, routes)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase782_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase782_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_routes": summary["n_routes"],
                "top_sufficiency_routes": summary["top_sufficiency_routes"][:8],
                "top_necessity_routes": summary["top_necessity_routes"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 782 Multi-Component Surface Route Patch ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: patch/replace/zero multiple Phase 780 candidate components as a route.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.",
        "- Strict interpretation: route-level answer-position patch, not full-token-sequence mechanism.",
        "",
        "## Routes",
        "",
        "| model | route | compare | size | components |",
        "|---|---|---|---:|---|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for route in data.get("routes") or []:
            labels = ", ".join(route.get("component_labels") or [])
            lines.append(f"| {model} | `{route['route_id']}` | `{route['compare_variant']}` | {route['route_size']} | `{labels}` |")

    lines += [
        "",
        "## Route Intervention Summary",
        "",
        "| model | route | intervention | cases | strict | semantic-equiv | pool top1 | margin | delta margin | strict gain | strict loss | semantic loss | pool loss | top1 classes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data.get("by_route_intervention") or []:
            lines.append(
                f"| {model} | `{row['route_id']}` | `{row['intervention_kind']}` | {row['case_n']} | "
                f"{fmt(row['strict_open_rate'])} | {fmt(row['semantic_equiv_open_rate'])} | {fmt(row['pool_top1_rate'])} | "
                f"{fmt(row['mean_margin_target_vs_case_variant'])} | {fmt(row['mean_delta_margin_vs_reference'])} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['strict_loss_rate_vs_reference'])} | "
                f"{fmt(row['semantic_loss_rate_vs_reference'])} | {fmt(row['pool_loss_rate_vs_reference'])} | "
                f"`{json.dumps(row['top1_classes'], ensure_ascii=False)}` |"
            )

    lines += [
        "",
        "## Top Sufficiency Routes",
        "",
        "| model | route | size | delta margin | strict gain | score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_sufficiency_routes") or [])[:10]:
            lines.append(
                f"| {model} | `{row['route_id']}` | {row['route_size']} | {fmt(row['mean_delta_margin_vs_reference'])} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['sufficiency_score'])} |"
            )

    lines += [
        "",
        "## Top Necessity Routes",
        "",
        "| model | route | intervention | size | delta margin | strict loss | semantic loss | score |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_necessity_routes") or [])[:10]:
            lines.append(
                f"| {model} | `{row['route_id']}` | `{row['intervention_kind']}` | {row['route_size']} | "
                f"{fmt(row['mean_delta_margin_vs_reference'])} | {fmt(row['strict_loss_rate_vs_reference'])} | "
                f"{fmt(row['semantic_loss_rate_vs_reference'])} | {fmt(row['necessity_score'])} |"
            )

    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- Route-level patch tests whether candidate combinations beat single-component patch.",
        "- It still patches only the answer position.",
        "- If route patch remains weak, next tests must include token-position ranges and downstream readout closure.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase782_{model}_summary.json"
        if path.exists():
            by_model[model] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 782,
        "title": "Multi-Component Surface Route Patch",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase782_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase782_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase780_round": args.source_phase780_round, "models": {}}
    for model in MODELS:
        args.model = model
        selected = select_surface_cases(model, args)
        routes = select_routes(model, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
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
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--route-sizes", default="1,2,4")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
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
